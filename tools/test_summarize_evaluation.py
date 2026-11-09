"""Offline summary integration: frozen samples, real local rules and mocked HTTP."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_evaluation as summary
from tools.test_answer_judge import completion, response, row
from utils import answer_judge as api, rule_scoring as rules
from utils.evaluation_artifacts import EvaluationStore


class SummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "evaluation"
        self.run = None
        self.config_path = self.root / "judge.yaml"
        self.config = api.load_judge_config()
        self.config.update(max_attempts=3, retry_delay_seconds=0)
        self.config_path.write_text(yaml.safe_dump(self.config))
        for change in (patch.object(summary, "ROOT", self.root),
                       patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-test-secret"}),
                       patch.object(api, "urlopen", side_effect=AssertionError("Unexpected live API request"))):
            change.start()
            self.addCleanup(change.stop)

    def write_input(self, values, *, source="open-r1/OpenR1-Math-220k"):
        store = EvaluationStore(self.input, root=self.root)
        store.create([{"question": value["question"], "answer": value["reference_answer"],
                       "final_answer": value["reference_final_answer"], "source_index": value["source_index"],
                       "generation_index": value["generation_index"]} for value in values],
                     {"model_name": "DO_NOT_SHOW_MODEL", "data": {"source": {"name": source}},
                      "generation": {"engine": "DO_NOT_SHOW_ENGINE", "max_new_tokens": 8192}})
        for position, value in enumerate(values):
            store.save_answer({key: value[key] for key in ("answer", "generated_tokens", "generation_runtime",
                                                           "generation_batch_size", "truncated")} | {"index": position})
            result = rules.grade_rules(value)
            first = result["stages"]["stage1"]
            store.save_judgment(position, 1, {**first, "format_valid": value["format_valid"], "runtime": 0.001})
            if source == summary.OPENR1 and result["stages"]["stage2"]["status"] != "not_run":
                store.save_judgment(position, 2, {**result["stages"]["stage2"], "runtime": 0.002})
        return store

    def summarize(self, **kwargs):
        source = {"evaluation": self.input} if self.run is None else {"resume_from": self.run}
        with redirect_stdout(StringIO()):
            report = summary.summarize_evaluation(**source, config_path=self.config_path, disable_tqdm=True, **kwargs)
        self.run = Path(report["scoring_run"])
        return report

    def last_run(self):
        return sorted((self.input / "scorings").iterdir())[-1]

    def reading(self):
        return {path.name: path.read_bytes() for path in self.input.glob("*.md")}

    def test_saved_local_passes_and_truncation_never_load_a_judge_or_request_credentials(self):
        self.write_input([row(0, correct=True), row(1, truncated=True), row(2, r1=r"\boxed{4}")])
        self.config_path = self.root / "missing-config.yaml"
        with patch.object(summary, "APIJudge", side_effect=AssertionError("No API work")), \
                patch.object(summary, "api_key", side_effect=AssertionError("No credentials needed")), \
                patch.object(rules, "grade_rules", side_effect=AssertionError("Use saved local decisions")):
            report = self.summarize()
        self.assertEqual([value["correct"] for value in report["predictions"]], [True, False, True])
        self.assertEqual(report["metrics"]["api_samples"], 0)
        self.assertEqual(report["metrics"]["answer_accuracy"], 2 / 3)
        self.assertIsNone(json.loads((self.run / "metadata.json").read_text())["judge"])

    def test_default_api_accuracy_payload_and_two_linked_review_lists(self):
        values = [row(0, correct=True), row(1, r1=r"\boxed{4}"), row(2), row(3),
                  row(4, truncated=True), row(5), row(6)]
        values[2]["question"] = "A two-part question in its original order."
        store = self.write_input(values)
        original = {path: path.read_bytes() for path in (store.metadata_path, store.answers_path, store.snapshot_path)}
        replies = [completion(), completion("incorrect"), completion("uncertain"), {"choices": []}]
        with patch.object(api, "urlopen", side_effect=[response(value) for value in replies]) as request:
            report = self.summarize()
        metrics = report["metrics"]
        self.assertEqual((metrics["confirmed_correct"], metrics["incorrect"], metrics["pending_review"]), (3, 2, 2))
        self.assertEqual(metrics["answer_accuracy"], 3 / 7)
        self.assertEqual([metrics[f"stage{i}_correct"] for i in (1, 2, 3)], [1, 1, 1])
        self.assertEqual((metrics["api_samples"], metrics["api_calls"], metrics["unprocessed_samples"]), (4, 4, 0))
        self.assertEqual(metrics["generation_runtime"], 14)
        sent = json.loads(request.call_args_list[0].args[0].data)
        submitted = json.loads(sent["messages"][1]["content"])
        self.assertEqual(sent["messages"][0]["content"], api.JUDGE_PROMPT)
        self.assertEqual(submitted["problem"], values[2]["question"])
        self.assertEqual(submitted["candidate_solution"], values[2]["answer"])
        self.assertEqual(submitted["r1_reference_solution"], values[2]["reference_answer"])
        self.assertNotIn("correct", submitted)
        decided = (self.input / "api-decided.md").read_text()
        pending = (self.input / "api-pending.md").read_text()
        self.assertIn("## Sample 3\n", decided)
        self.assertIn("## Sample 4\n", decided)
        self.assertIn("## Sample 6\n", pending)
        self.assertIn("## Sample 7\n", pending)
        for text in (decided, pending):
            self.assertNotIn(values[2]["question"], text)
            self.assertNotIn("### Model answer", text)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)
        before = self.reading()
        with patch.object(api, "urlopen", side_effect=AssertionError("Completed and pending are reused")):
            self.assertEqual(self.summarize()["metrics"], metrics)
        self.assertEqual(self.reading(), before)

    def test_no_api_skips_config_key_and_pending_retry(self):
        store = self.write_input([row(0), row(1)])
        before = store.judgments_path.read_bytes()
        self.config_path = self.root / "missing.yaml"
        with patch.object(summary, "load_judge_config", side_effect=AssertionError("No config needed")), \
                patch.object(summary, "api_key", side_effect=AssertionError("No key needed")), \
                patch.object(api, "urlopen", side_effect=AssertionError("No API")):
            report = self.summarize(no_api=True, retry_pending=True)
        self.assertEqual(report["metrics"]["unprocessed_samples"], 2)
        self.assertEqual(report["metrics"]["pending_review"], 0)
        self.assertTrue(all(value["stages"]["stage3"]["status"] == "not_run" for value in report["predictions"]))
        self.assertEqual(store.judgments_path.read_bytes(), before)
        self.assertIsNone(json.loads((self.run / "metadata.json").read_text())["judge"])

    def test_api_interruption_saves_only_machine_progress_and_resume_processes_remaining_samples(self):
        self.write_input([row(0), row(1), row(2)])
        with patch.object(api, "urlopen", side_effect=[response(completion()), KeyboardInterrupt()]), \
                self.assertRaises(KeyboardInterrupt):
            self.summarize()
        self.assertEqual(self.reading(), {})
        self.run = self.last_run()
        self.config_path.unlink()
        with patch.object(api, "urlopen", side_effect=lambda *args, **kwargs: response(completion())) as request:
            report = self.summarize()
        self.assertEqual(request.call_count, 2)
        self.assertEqual(report["metrics"]["confirmed_correct"], 3)
        self.assertEqual((self.input / "answers-0001.md").read_text().count('id="sample-0"'), 1)

    def test_partial_report_after_interruption_keeps_fixed_denominator_and_statuses_distinct(self):
        self.write_input([row(i, correct=True) for i in range(60)] + [row(i) for i in range(60, 100)])
        replies = [completion()] * 10 + [completion("incorrect")] * 8 + [completion("uncertain")] * 2
        with patch.object(api, "urlopen", side_effect=[response(value) for value in replies] + [KeyboardInterrupt()]), \
                self.assertRaises(KeyboardInterrupt):
            self.summarize()
        self.assertFalse(self.reading())
        self.run = self.last_run()
        with patch.object(api, "urlopen", side_effect=AssertionError("Partial summary must not call API")), \
                patch.object(summary, "api_key", side_effect=AssertionError("No key for partial summary")):
            report = self.summarize(no_api=True)
        metrics = report["metrics"]
        self.assertEqual((metrics["confirmed_correct"], metrics["incorrect"], metrics["pending_review"],
                          metrics["unprocessed_samples"]), (70, 8, 2, 20))
        self.assertEqual(metrics["answer_accuracy"], .7)
        self.assertIn("70.00% (70/100)", (self.input / "answers-0001.md").read_text())

    def test_interrupting_api_preserves_the_last_successful_reading_view(self):
        self.write_input([row(0), row(1)])
        self.summarize(no_api=True)
        previous = self.reading()
        with patch.object(api, "urlopen", side_effect=[response(completion()), KeyboardInterrupt()]), \
                self.assertRaises(KeyboardInterrupt):
            self.summarize()
        self.assertEqual(self.reading(), previous)
        report = self.summarize(no_api=True)
        self.assertEqual(report["metrics"]["confirmed_correct"], 1)
        self.assertEqual(report["metrics"]["unprocessed_samples"], 1)

    def test_retry_appends_only_new_api_attempts_and_does_not_double_count(self):
        self.write_input([row(0), row(1)])
        with patch.object(api, "urlopen", side_effect=[response(completion()), response(completion("uncertain"))]):
            self.summarize()
        with patch.object(api, "urlopen", return_value=response(completion())) as request:
            report = self.summarize(retry_pending=True)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(report["metrics"]["api_calls"], 3)
        self.assertEqual(report["metrics"]["confirmed_correct"], 2)
        self.assertEqual(report["metrics"]["pending_review"], 0)
        records = EvaluationStore(self.input, root=self.root).judgment_records
        self.assertEqual([len(value["attempts"]) for value in records if value["stage"] == 3], [1, 1, 1])
        self.assertIn("None.", (self.input / "api-pending.md").read_text())

    def test_independent_runs_share_the_journal_and_one_reading_view(self):
        store = self.write_input([row(0)])
        local_history = store.judgments_path.read_bytes()
        with patch.object(api, "urlopen", return_value=response(completion())):
            first = self.summarize()
        self.run = None
        with patch.object(api, "urlopen", return_value=response(completion("incorrect"))):
            second = self.summarize()
        self.assertNotEqual(first["scoring_run"], second["scoring_run"])
        self.assertEqual(second["metrics"]["confirmed_correct"], 0)
        self.run = Path(first["scoring_run"])
        self.assertEqual(self.summarize(no_api=True)["metrics"]["confirmed_correct"], 1)
        self.assertTrue(store.judgments_path.read_bytes().startswith(local_history))
        self.assertEqual(len(list(self.input.rglob("answers-*.md"))), 1)
        self.assertTrue(all({path.name for path in directory.iterdir()} == {"metadata.json"}
                            for directory in (self.input / "scorings").iterdir()))
        records = EvaluationStore(self.input, root=self.root).judgment_records
        self.assertEqual([value["stage"] for value in records], [1, 2, 3, 3])

    def test_101_questions_split_and_links_follow_regenerated_volumes_without_duplicate_text(self):
        self.write_input([row(i, correct=True) for i in range(99)] + [row(99), row(100)])
        with patch.object(api, "urlopen", side_effect=[response(completion()), response(completion("uncertain"))]):
            self.summarize()
        first = (self.input / "answers-0001.md").read_text()
        second = (self.input / "answers-0002.md").read_text()
        self.assertEqual(len(re.findall(r"^## Question ", first, re.M)), 100)
        self.assertEqual(len(re.findall(r"^## Question ", second, re.M)), 1)
        self.assertEqual((first + second).count("## Summary"), 1)
        for token in ("DO_NOT_SHOW_MODEL", "DO_NOT_SHOW_ENGINE", "max_new_tokens", "Final conclusions match.", "API audit"):
            self.assertNotIn(token, first + second)
        self.assertIn("answers-0001.md#sample-99", (self.input / "api-decided.md").read_text())
        self.assertIn("answers-0002.md#sample-100", (self.input / "api-pending.md").read_text())
        (self.input / "answers-notes.md").write_text("Preserve unrelated notes.")
        with patch.object(api, "urlopen", side_effect=AssertionError("Do not rejudge while changing pagination")):
            self.summarize(questions_per_file=7)
            self.assertTrue((self.input / "answers-0015.md").exists())
            self.summarize(questions_per_file=200)
        self.assertFalse((self.input / "answers-0002.md").exists())
        self.assertEqual((self.input / "answers-notes.md").read_text(), "Preserve unrelated notes.")
        for name in ("api-decided.md", "api-pending.md"):
            for filename, anchor in re.findall(r"\]\((answers-\d+\.md)#([^)]*)\)", (self.input / name).read_text()):
                self.assertIn(f'id="{anchor}"', (self.input / filename).read_text())

    def test_saved_judge_and_prompt_are_reused_even_when_current_config_changes(self):
        self.write_input([row(0)])
        with patch.object(api, "urlopen", return_value=response(completion("uncertain"))):
            self.summarize()
        saved_metadata = (self.run / "metadata.json").read_bytes()
        self.config_path.write_text(yaml.safe_dump({**self.config, "max_tokens": 17}))
        with patch.object(summary, "JUDGE_PROMPT", "New prompt must not replace the saved prompt"), \
                patch.object(api, "urlopen", return_value=response(completion())) as request:
            self.summarize(retry_pending=True)
        sent = json.loads(request.call_args.args[0].data)
        self.assertEqual(sent["max_tokens"], self.config["max_tokens"])
        self.assertEqual(sent["messages"][0]["content"], api.JUDGE_PROMPT)
        self.assertEqual((self.run / "metadata.json").read_bytes(), saved_metadata)

    def test_missing_noninteractive_cancelled_and_empty_credentials_leave_samples_unprocessed(self):
        store = self.write_input([row(0)])
        original = store.judgments_path.read_bytes()
        for interactive, value in ((False, None), (True, KeyboardInterrupt()), (True, "")):
            with self.subTest(interactive=interactive, value=value), \
                    patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
                    patch.object(summary.sys.stdin, "isatty", return_value=interactive), \
                    patch.object(summary.getpass, "getpass", side_effect=value if isinstance(value, BaseException) else None,
                                 return_value=value), \
                    patch.object(api, "urlopen", side_effect=AssertionError("No request without a key")), \
                    self.assertRaises(ValueError):
                self.summarize()
            self.assertEqual(store.judgments_path.read_bytes(), original)
            self.assertFalse(self.reading())
        self.run = self.last_run()
        self.assertEqual(self.summarize(no_api=True)["metrics"]["unprocessed_samples"], 1)

    def test_hidden_key_is_used_only_for_this_run_and_redacted_from_all_artifacts(self):
        self.write_input([row(0)])
        key = "typed-private-key"
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
                patch.object(summary.sys.stdin, "isatty", return_value=True), \
                patch.object(summary.getpass, "getpass", return_value=key) as prompt, \
                patch.object(api, "urlopen", return_value=response(completion(reason="Final answer matches " + key))) as request:
            self.summarize()
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "")
        prompt.assert_called_once()
        self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer " + key)
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(key.encode(), path.read_bytes())

    def test_completed_pending_work_does_not_prompt_or_retry_without_the_flag(self):
        self.write_input([row(0)])
        with patch.object(api, "urlopen", return_value=response(completion("uncertain"))):
            self.summarize()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
                patch.object(summary, "api_key", side_effect=AssertionError("No new work")), \
                patch.object(summary, "APIJudge", side_effect=AssertionError("No new requests")):
            self.assertEqual(self.summarize()["metrics"]["pending_review"], 1)

    def test_reading_failure_preserves_ledger_and_previous_view(self):
        self.write_input([row(0)])
        self.summarize(no_api=True)
        previous = self.reading()
        with patch.object(api, "urlopen", return_value=response(completion())), \
                patch.object(summary, "readable_row", side_effect=OSError("disk failure")), self.assertRaises(OSError):
            self.summarize()
        self.assertEqual(self.reading(), previous)
        with patch.object(api, "urlopen", side_effect=AssertionError("Saved verdict must be reused")):
            self.assertEqual(self.summarize()["metrics"]["confirmed_correct"], 1)

    def test_publication_failure_rolls_back_the_complete_previous_view(self):
        self.write_input([row(0), row(1)])
        self.summarize(no_api=True, questions_per_file=1)
        previous = self.reading()
        original = Path.replace

        def fail(path, target):
            if path.parent.name == "prepared" and path.name == "api-pending.md":
                raise OSError("publication failed")
            return original(path, target)

        with patch.object(Path, "replace", new=fail), self.assertRaises(OSError):
            self.summarize(no_api=True, questions_per_file=2)
        self.assertEqual(self.reading(), previous)

    def test_incomplete_local_stages_and_invalid_arguments_do_not_start_api(self):
        store = self.write_input([row(0)])
        local = store.judgments_path.read_text().splitlines()
        store.judgments_path.write_text(local[0] + "\n")
        with patch.object(api, "urlopen", side_effect=AssertionError("Local rules are not complete")), \
                self.assertRaisesRegex(ValueError, "Complete both local rule stages"):
            self.summarize()
        self.assertFalse((self.input / "scorings").exists())
        for cap in (0, -1, True):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "positive integer"):
                self.summarize(questions_per_file=cap)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            summary.summarize_evaluation(self.input, resume_from=self.root)

    def test_gsm8k_uses_basic_saved_results_and_never_calls_r1_api(self):
        self.write_input([row(0, correct=True), row(1)], source="openai/gsm8k")
        with patch.object(summary, "APIJudge", side_effect=AssertionError("No OpenR1 API for GSM8K")):
            report = self.summarize()
        self.assertEqual(report["metrics"]["answer_accuracy"], .5)
        self.assertEqual(report["metrics"]["incorrect"], 1)
        self.assertEqual(report["metrics"]["unprocessed_samples"], 0)

    def test_cli_runs_outside_project_directory_and_does_not_require_keys_for_no_api(self):
        store = self.write_input([row(0, correct=True), row(1, truncated=True)])
        metadata = {**store.metadata, "snapshot": str(store.snapshot_path)}
        store.metadata_path.write_text(json.dumps(metadata))
        script = str(Path(summary.__file__))
        completed = subprocess.run([sys.executable, script, "--evaluation", str(self.input), "--no-api"],
                                   cwd=self.root, env={**os.environ, "DEEPSEEK_API_KEY": ""},
                                   capture_output=True, text=True, check=True)
        self.assertIn("50.00% (1/2)", completed.stdout)
        run = self.last_run()
        repeated = subprocess.run([sys.executable, script, "--resume", str(run)],
                                  cwd=self.root, env={**os.environ, "DEEPSEEK_API_KEY": ""},
                                  capture_output=True, text=True, check=True)
        self.assertIn("50.00% (1/2)", repeated.stdout)


if __name__ == "__main__":
    unittest.main()
