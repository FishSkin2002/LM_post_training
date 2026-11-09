"""Local two-stage evaluation through the real checkpoint evaluator, with no API."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import test_instruct_sft as fixtures
import evaluate as evaluation
import summarize_evaluation as summarizing
from utils import answer_judge, math_data, rule_scoring


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

    def store(self):
        return evaluation.EvaluationStore(self.output, root=self.fixture.root)

    def test_all_generation_precedes_second_level_and_decisions_stay_separate(self):
        original = rule_scoring.grade_second_stage

        def grade(row):
            self.assertEqual(len(self.store().answers), 4)
            return original(row)

        with patch.object(evaluation, "generate_answers", side_effect=[self.answers[:2], self.answers[2:]]), \
                patch.object(rule_scoring, "grade_second_stage", side_effect=grade):
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
        store = self.store()
        self.assertTrue(all(set(row) == {"index", "answer", "generated_tokens", "truncated",
                                         "generation_batch_size", "generation_runtime"} for row in store.answers))
        judgments = [json.loads(line) for line in store.judgments_path.read_text().splitlines()]
        self.assertEqual([row["stage"] for row in judgments], [1, 1, 1, 1, 2, 2])
        self.assertTrue(all("stages" not in row and "answer" not in row for row in judgments))
        self.assertNotEqual([row["source_index"] for row in rows], list(range(4)))
        self.assertEqual({path.name for path in store.directory.iterdir()},
                         {"metadata.json", "answers.jsonl", "judgments.jsonl"})

    def test_generation_interruption_never_starts_second_level(self):
        with patch.object(evaluation, "generate_answers", side_effect=[self.answers[:2], KeyboardInterrupt]), \
                patch.object(rule_scoring, "grade_second_stage", side_effect=AssertionError("Generation is not complete")), \
                self.assertRaises(KeyboardInterrupt):
            self.evaluate(batch_size=2)
        store = self.store()
        self.assertEqual(len(store.answers), 2)
        self.assertTrue(all(set(levels) == {1} for levels in store.judgments.values()))
        self.assertFalse(list(store.directory.glob("*.md")))

    def test_second_level_resume_preserves_histories_without_reopening_the_dataset(self):
        original = rule_scoring.grade_second_stage
        processed = []

        def interrupt(row):
            processed.append(row["index"])
            if len(processed) == 2:
                raise KeyboardInterrupt
            return original(row)

        with patch.object(evaluation, "generate_answers", return_value=self.answers), \
                patch.object(rule_scoring, "grade_second_stage", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        store = self.store()
        partial = evaluation.saved_evaluation(store)
        self.assertEqual(partial["rule_scoring"]["completed_samples"], 3)
        self.assertFalse(partial["rule_scoring"]["complete"])
        records = [json.loads(line) for line in store.judgments_path.read_text().splitlines()]
        old = records[-1]
        old["reason"] = "historical decision must be reused"
        store.judgments_path.write_text("".join(json.dumps(row) + "\n" for row in records))
        history = {path: path.read_bytes() for path in (store.answers_path, store.metadata_path, store.judgments_path)}

        def only_new(row):
            self.assertNotEqual(row["index"], old["index"])
            return original(row)

        self.dataset = None
        with patch.object(evaluation, "create_generator", side_effect=AssertionError("Already generated")), \
                patch.object(evaluation, "score_answer", side_effect=AssertionError("Reuse stage one")), \
                patch.object(evaluation, "prepare_dataset", side_effect=AssertionError("Use the snapshot")), \
                patch.object(evaluation, "ensure_prepared_sft_data", side_effect=AssertionError("Use the snapshot")), \
                patch.object(rule_scoring, "grade_second_stage", side_effect=only_new) as grade:
            report = self.evaluate()
        self.assertEqual(grade.call_count, 1)
        self.assertTrue(report["rule_scoring"]["complete"])
        self.assertEqual(report["predictions"][old["index"]]["rule_scoring"]["stages"]["stage2"]["reason"], old["reason"])
        for path, prefix in history.items():
            self.assertTrue(path.read_bytes().startswith(prefix))
        for path in (store.answers_path, store.metadata_path):
            self.assertEqual(path.read_bytes(), history[path])
        finished = {path: path.read_bytes() for path in history}
        with patch.object(evaluation, "create_generator", side_effect=AssertionError("Already generated")), \
                patch.object(evaluation, "score_answer", side_effect=AssertionError("No first-level rescoring")), \
                patch.object(rule_scoring, "grade_second_stage", side_effect=AssertionError("No second-level rescoring")):
            repeated = self.evaluate()
        self.assertEqual(repeated["predictions"], report["predictions"])
        self.assertEqual({path: path.read_bytes() for path in history}, finished)

    def test_flushed_second_level_decision_survives_failure_after_the_write(self):
        original = evaluation.EvaluationStore.save_judgment

        def interrupt(store, index, stage, result):
            original(store, index, stage, result)
            if stage == 2:
                raise KeyboardInterrupt

        with patch.object(evaluation, "generate_answers", return_value=self.answers), \
                patch.object(evaluation.EvaluationStore, "save_judgment", new=interrupt), self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        store = self.store()
        saved = store.judgments_path.read_bytes()
        original_grade = rule_scoring.grade_second_stage

        def only_new(row):
            self.assertNotIn(2, store.judgments.get(row["index"], {}))
            return original_grade(row)

        with patch.object(evaluation, "create_generator", side_effect=AssertionError("Already generated")), \
                patch.object(rule_scoring, "grade_second_stage", side_effect=only_new) as grade:
            self.evaluate()
        self.assertEqual(grade.call_count, 1)
        self.assertTrue(store.judgments_path.read_bytes().startswith(saved))

    def test_answer_saved_before_missing_first_decision_is_not_generated_again(self):
        original = evaluation.score_answer
        calls = []

        def interrupt(*args):
            calls.append(args)
            if len(calls) == 4:
                raise KeyboardInterrupt
            return original(*args)

        with patch.object(evaluation, "generate_answers", return_value=self.answers), \
                patch.object(evaluation, "score_answer", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        store = self.store()
        self.assertEqual(len(store.answers), 4)
        self.assertEqual(len(store.judgments), 3)
        saved_answers = store.answers_path.read_bytes()
        with patch.object(evaluation, "create_generator", side_effect=AssertionError("Saved answers cannot be regenerated")), \
                patch.object(evaluation, "score_answer", wraps=original) as score:
            report = self.evaluate()
        score.assert_called_once()
        self.assertTrue(report["rule_scoring"]["complete"])
        self.assertEqual(store.answers_path.read_bytes(), saved_answers)

    def test_summary_uses_saved_local_decisions_without_repeating_rules(self):
        with patch.object(evaluation, "generate_answers", return_value=self.answers):
            report = self.evaluate()
        store = self.store()
        before = {path: path.read_bytes() for path in (store.metadata_path, store.answers_path, store.judgments_path)}
        with patch.object(summarizing, "ROOT", self.fixture.root), \
                patch.object(summarizing, "APIJudge", side_effect=AssertionError("No API in local-only summary")), \
                patch.object(rule_scoring, "grade_rules", side_effect=AssertionError("Use the saved rules")), \
                redirect_stdout(io.StringIO()):
            summary = summarizing.summarize_evaluation(self.output, no_api=True)
        self.assertEqual(summary["metrics"]["confirmed_correct"], 2)
        self.assertEqual(summary["metrics"]["incorrect"], 1)
        self.assertEqual(summary["metrics"]["unprocessed_samples"], 1)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_default_full_evaluation_resumes_without_the_original_dataset(self):
        options = dict(model_name=self.fixture.model_name, tokenizer=self.fixture.tokenizer,
                       config=self.fixture.config, device="cpu", generation_backend="transformers", disable_tqdm=True)
        with redirect_stdout(io.StringIO()), patch.object(evaluation, "generate_answers", return_value=self.answers):
            original = evaluation.evaluate_checkpoint(model=self.model, dataset=self.dataset, data_info=self.info, **options)
        with redirect_stdout(io.StringIO()), \
                patch.object(evaluation, "create_generator", side_effect=AssertionError("Reuse all generated answers")), \
                patch.object(evaluation, "prepare_dataset", side_effect=AssertionError("Original data unavailable")), \
                patch.object(evaluation, "ensure_prepared_sft_data", side_effect=AssertionError("No cache preparation")), \
                patch.object(evaluation, "score_answer", side_effect=AssertionError("Reuse all first decisions")):
            repeated = evaluation.evaluate_checkpoint(**options)
        self.assertEqual(original, repeated)

    def test_gsm8k_remains_on_its_existing_basic_evaluation_path(self):
        self.fixture.config["data"] = dict(math_data.GSM8K)
        self.fixture.raw = [{"question": f"question {i}", "answer": f"Compute carefully. #### {i}"}
                            for i in range(20)]
        self.dataset, self.info = self.fixture.prepared()
        with patch.object(evaluation, "generate_answers", return_value=self.answers[:2]), \
                patch.object(evaluation, "apply_rule_scoring", side_effect=AssertionError("No R1 grading for GSM8K")):
            report = self.evaluate()
        self.assertNotIn("rule_scoring", report)
        self.assertTrue(all(set(levels) == {1} for levels in self.store().judgments.values()))


if __name__ == "__main__":
    unittest.main()
