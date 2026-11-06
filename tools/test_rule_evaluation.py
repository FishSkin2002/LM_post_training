"""Local two-stage evaluation through the real checkpoint evaluator, with no API."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import test_instruct_sft as fixtures
from utils import answer_judge, evaluation, math_data, rescoring, rule_scoring


class RuleEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InstructSFTTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.raw = [fixtures.raw_row(i) for i in range(40)]
        for row in self.fixture.raw:
            row["answer"] = "1/12"
            row["generations"][1] = r"\boxed{B}"
        self.dataset, self.info = self.fixture.prepared()
        self.model = self.fixture.model_loader(self.fixture.model_name)
        self.output = self.fixture.root / "evaluation.json"
        self.answers = [(r"\boxed{\frac{1}{12}}", 3, False),
                        (r"\boxed{B}", 3, False),
                        (r"\boxed{\frac{1}{12}}", 4, True),
                        (r"\boxed{1}\boxed{2}", 3, False)]
        self.no_api = patch.object(answer_judge, "urlopen", side_effect=AssertionError("Local evaluation cannot call API"))
        self.no_api.start()
        self.addCleanup(self.no_api.stop)

    def evaluate(self, *, batch_size=4):
        with redirect_stdout(io.StringIO()):
            return evaluation.evaluate_checkpoint(
                model_name=self.fixture.model_name, model=self.model, tokenizer=self.fixture.tokenizer,
                dataset=self.dataset, data_info=self.info, config=self.fixture.config,
                output=self.output, device="cpu", generation_backend="transformers", disable_tqdm=True,
                batch_size=batch_size)

    def test_all_generation_precedes_second_level_and_decisions_stay_separate(self):
        original = rule_scoring.grade_rules

        def grade(row):
            saved = self.output.with_suffix(".progress.jsonl").read_text().splitlines()
            self.assertEqual(len(saved), 4, "Second-level rules must wait for every generated answer")
            return original(row)

        with patch.object(evaluation, "generate_answers", side_effect=[self.answers[:2], self.answers[2:]]), \
                patch.object(rule_scoring, "grade_rules", side_effect=grade):
            report = self.evaluate(batch_size=2)
        rows = report["predictions"]
        self.assertEqual([row["correct"] for row in rows], [True, False, False, False])
        results = [row["rule_scoring"] for row in rows]
        self.assertEqual([result["correct"] for result in results], [True, True, False, None])
        self.assertEqual([result["decision_stage"] for result in results], [1, 2, 1, 2])
        self.assertEqual(results[0]["stages"]["stage2"]["status"], "not_run")
        self.assertEqual(results[2]["stages"]["stage2"]["status"], "not_run")
        self.assertEqual(results[3]["stages"]["stage2"]["reason"], "model_multiple_boxes")
        self.assertTrue(all(result["stages"]["stage3"]["status"] == "not_run" for result in results))
        self.assertTrue(report["rule_scoring"]["complete"])
        self.assertEqual(report["metrics"]["answer_accuracy"], .25)
        first = [json.loads(line) for line in self.output.with_suffix(".progress.jsonl").read_text().splitlines()]
        self.assertTrue(all("rule_scoring" not in row for row in first))
        self.assertNotEqual(report["data"]["indices"], list(range(4)))

    def test_generation_interruption_never_starts_second_level(self):
        with patch.object(evaluation, "generate_answers", side_effect=[self.answers[:2], KeyboardInterrupt]), \
                patch.object(rule_scoring, "grade_rules", side_effect=AssertionError("Generation is not complete")), \
                self.assertRaises(KeyboardInterrupt):
            self.evaluate(batch_size=2)
        self.assertEqual(len(self.output.with_suffix(".progress.jsonl").read_text().splitlines()), 2)
        self.assertFalse(self.output.with_suffix(".stage2.progress.jsonl").exists())
        self.assertFalse(self.output.with_suffix(".stage2.md").exists())

    def test_second_level_resume_preserves_both_histories_and_completed_rerun(self):
        original = rule_scoring.grade_rules
        processed = []

        def interrupt(row):
            processed.append(row["index"])
            if len(processed) == 2:
                raise KeyboardInterrupt
            return original(row)

        with patch.object(evaluation, "generate_answers", return_value=self.answers), \
                patch.object(rule_scoring, "grade_rules", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        partial = json.loads(self.output.read_text())
        self.assertEqual(partial["rule_scoring"]["completed_samples"], 1)
        self.assertFalse(partial["rule_scoring"]["complete"])
        with patch.object(rescoring, "APIJudge", side_effect=AssertionError("Local rules are incomplete")), \
                self.assertRaisesRegex(ValueError, "Complete both local rule stages"):
            rescoring.rescore_evaluation(self.output)
        progress_path = self.output.with_suffix(".stage2.progress.jsonl")
        old = json.loads(progress_path.read_text())
        old["result"]["stages"]["stage1"]["reason"] = "historical decision must be reused"
        progress_path.write_text(json.dumps(old) + "\n")
        history = {suffix: self.output.with_suffix(suffix).read_bytes()
                   for suffix in (".progress.jsonl", ".md", ".stage2.progress.jsonl", ".stage2.md")}

        def only_new(row):
            self.assertNotEqual(row["index"], old["index"])
            return original(row)

        with patch.object(evaluation, "create_generator", side_effect=AssertionError("No generation on second-level resume")), \
                patch.object(evaluation, "score_answer", side_effect=AssertionError("Saved first-level decisions are final")), \
                patch.object(rule_scoring, "grade_rules", side_effect=only_new) as grade:
            report = self.evaluate()
        self.assertEqual(grade.call_count, 3)
        self.assertTrue(report["rule_scoring"]["complete"])
        self.assertEqual(report["predictions"][0]["rule_scoring"], old["result"])
        for suffix, prefix in history.items():
            self.assertTrue(self.output.with_suffix(suffix).read_bytes().startswith(prefix))
        for suffix in (".progress.jsonl", ".md"):
            self.assertEqual(self.output.with_suffix(suffix).read_bytes(), history[suffix])
        finished = {suffix: self.output.with_suffix(suffix).read_bytes() for suffix in history}
        with patch.object(evaluation, "create_generator", side_effect=AssertionError("Already generated")), \
                patch.object(evaluation, "score_answer", side_effect=AssertionError("No first-level rescoring")), \
                patch.object(rule_scoring, "grade_rules", side_effect=AssertionError("No second-level rescoring")), \
                patch.object(rule_scoring, "readable_rule_result", side_effect=AssertionError("No old Markdown rendering")):
            repeated = self.evaluate()
        self.assertEqual(repeated["predictions"], report["predictions"])
        self.assertEqual({suffix: self.output.with_suffix(suffix).read_bytes() for suffix in history}, finished)

    def test_flushed_second_level_decision_is_reused_after_markdown_failure(self):
        with patch.object(evaluation, "generate_answers", return_value=self.answers), \
                patch.object(rule_scoring, "readable_rule_result", side_effect=OSError("disk error")), \
                self.assertRaises(OSError):
            self.evaluate()
        old = json.loads(self.output.with_suffix(".stage2.progress.jsonl").read_text())
        prefix = self.output.with_suffix(".stage2.md").read_bytes()
        original = rule_scoring.grade_rules

        def only_new(row):
            self.assertNotEqual(row["index"], old["index"])
            return original(row)

        with patch.object(evaluation, "create_generator", side_effect=AssertionError("Already generated")), \
                patch.object(rule_scoring, "grade_rules", side_effect=only_new):
            self.evaluate()
        markdown = self.output.with_suffix(".stage2.md").read_text()
        self.assertTrue(self.output.with_suffix(".stage2.md").read_bytes().startswith(prefix))
        self.assertNotIn(f"## Dataset index {old['index']}\n", markdown)

    def test_existing_rescoring_uses_saved_local_decisions_without_repeating_rules(self):
        with patch.object(evaluation, "generate_answers", return_value=self.answers):
            report = self.evaluate()
        before = deepcopy(report)
        judge = Mock(return_value={"status": "passed", "reason": "API confirmed", "attempts": []})
        with patch.object(rescoring, "grade_rules", side_effect=AssertionError("Use the saved rules")):
            results = [rescoring.grade_prediction(row, judge) for row in report["predictions"]]
        self.assertEqual(judge.call_count, 1)
        self.assertEqual([result["decision_stage"] for result in results], [1, 2, 1, 3])
        self.assertEqual(report, before, "API results must not mutate saved local decisions")

    def test_gsm8k_remains_on_its_existing_basic_evaluation_path(self):
        self.fixture.config["data"] = dict(math_data.GSM8K)
        self.fixture.raw = [{"question": f"question {i}", "answer": f"Compute carefully. #### {i}"}
                            for i in range(20)]
        self.dataset, self.info = self.fixture.prepared()
        with patch.object(evaluation, "generate_answers", return_value=self.answers[:2]), \
                patch.object(evaluation, "apply_rule_scoring", side_effect=AssertionError("No R1 grading for GSM8K")):
            report = self.evaluate()
        self.assertNotIn("rule_scoring", report)
        self.assertFalse(self.output.with_suffix(".stage2.progress.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
