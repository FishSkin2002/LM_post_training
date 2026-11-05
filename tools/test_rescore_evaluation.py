"""Offline rescoring integration: actual Math-Verify, mocked Chat Completions."""

from copy import deepcopy
from contextlib import redirect_stdout
from http.client import IncompleteRead
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import answer_judge as api
import rescore_evaluation as rescoring


def row(index, *, reference="2", r1=r"\boxed{3}", answer=r"\boxed{4}", truncated=False):
    return {"index": index, "source_index": index + 100, "generation_index": 1,
            "question": f"Problem {index}", "reference_final_answer": reference,
            "reference_answer": r1, "answer": answer, "correct": False,
            "format_valid": True, "truncated": truncated, "generated_tokens": 25,
            "generation_runtime": 2, "generation_batch_size": 4}


def completion(verdict="correct", reason="Final conclusions match.", *, model="glm-5.3", finish="stop"):
    return {"model": model, "choices": [{"finish_reason": finish, "message": {
        "role": "assistant", "content": json.dumps({"verdict": verdict, "reason": reason}),
        "reasoning_content": "Provider-specific reasoning is not the verdict.",
    }}], "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}}


def response(value):
    return BytesIO(json.dumps(value).encode())


class RescoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "evaluation.json"
        self.output = self.root / "rescored.json"
        self.config_path = self.root / "judge.yaml"
        self.config = api.load_judge_config()
        self.config.update(max_attempts=3, retry_delay_seconds=0)
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.env = patch.dict(os.environ, {"ZAI_API_KEY": "sk-test-secret", "DEEPSEEK_API_KEY": "ds-test-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_input(self, rows):
        report = {
            "model_name": "test-policy", "stage": "sft", "scoring": {"original": True},
            "data": {"source": {"name": "open-r1/OpenR1-Math-220k"},
                     "indices": [r["index"] for r in rows], "fingerprint": "saved-split"},
            "generation": {"max_new_tokens": 8192, "batch_size": 4},
            "metrics": {"answer_accuracy": 0, "generation_samples": len(rows),
                        "generation_runtime": 99, "generation_samples_per_second": 0.25},
            "predictions": rows,
        }
        self.input.write_text(json.dumps(report), encoding="utf-8")
        return report

    def run_rescoring(self, **kwargs):
        with redirect_stdout(StringIO()):
            return rescoring.rescore_evaluation(
                self.input, output=self.output, config_path=self.config_path, disable_tqdm=True, **kwargs)

    def test_first_stage_and_truncation_stop_before_r1_and_api(self):
        rows = [row(0, answer=r"\boxed{\frac{4}{2}}"),
                row(1, reference="proof", r1=r"\boxed{2}", answer=r"\boxed{2}", truncated=True)]
        self.write_input(rows)
        with patch.object(rescoring, "single_formal_box", side_effect=AssertionError("R1 must not run")), \
                patch.object(api, "urlopen", side_effect=AssertionError("API must not run")):
            report = self.run_rescoring()
        self.assertEqual([r["correct"] for r in report["predictions"]], [True, False])
        self.assertEqual(report["metrics"]["answer_accuracy"], 0.5)
        for item in report["predictions"]:
            self.assertEqual(item["decision_stage"], 1)
            self.assertEqual(item["stages"]["stage2"]["status"], "not_run")
            self.assertEqual(item["stages"]["stage3"]["status"], "not_run")
        self.assertEqual(report["metrics"]["api_samples"], 0)
        self.assertEqual(report["pending_review_indices"], [])

    def test_second_stage_formal_box_and_numeric_equivalence(self):
        rows = [
            row(0, reference=r"\frac{1}{12}", r1=r"\boxed{A}</think>Final \boxed{B}",
                answer=r"\boxed{D}</think>Answer: \boxed{B}"),
            row(1, reference="17", r1=r"\boxed{\frac{1}{4}}", answer=r"\boxed{0.25}"),
        ]
        self.write_input(rows)
        with patch.object(api, "urlopen", side_effect=AssertionError("API must not run")):
            report = self.run_rescoring()
        self.assertEqual(report["metrics"]["stage2_correct"], 2)
        self.assertEqual(report["metrics"]["api_samples"], 0)
        self.assertTrue(all(r["correct"] for r in report["predictions"]))

    def test_single_formal_box_edge_cases(self):
        cases = [
            (r"\boxed{1}\boxed{2}", "multiple_boxes"),
            ("No final box", "missing_box"),
            (r"\boxed{}", "empty_or_incomplete_box"),
            (r"\boxed{\frac{1}{2}", "empty_or_incomplete_box"),
            (r"\boxed{proof}", "unparseable_box"),
            (r"\boxed{\frac{1}{2}}", None),
            (r"\boxed{99}</think>\boxed {2}", None),
            (r"before </think> more \boxed{99} </think> \boxed{2}", None),
        ]
        for text, error in cases:
            with self.subTest(text=text):
                self.assertEqual(rescoring.single_formal_box(text)[1], error)

    def test_missing_multiple_and_unparseable_boxes_reach_api(self):
        rows = [
            row(0, r1="no box"),
            row(1, answer="no box"),
            row(2, r1=r"\boxed{1}\boxed{2}"),
            row(3, answer=r"\boxed{1}\boxed{2}", reference="17"),
            row(4, r1=r"\boxed{proof}"),
            row(5, r1=r"\boxed{", answer=r"\boxed{}"),
        ]
        self.write_input(rows)
        with patch.object(api, "urlopen", side_effect=lambda *a, **k: response(completion())) as request:
            report = self.run_rescoring()
        self.assertEqual(request.call_count, 6)
        self.assertEqual(report["metrics"]["stage3_correct"], 6)

    def test_accuracy_audit_and_pending_include_all_api_cases(self):
        rows = [
            row(0, answer=r"\boxed{2}"),
            row(1, reference=r"\frac{1}{12}", r1=r"\boxed{B}", answer=r"\boxed{B}"),
            row(2, reference="1008",
                r1=r"100 pieces: \boxed{No}; 2017 pieces: \boxed{Yes}",
                answer=r"\boxed{Yes} for 2017 pieces and \boxed{No} for 100 pieces."),
            row(3), row(4, truncated=True), row(5), row(6),
        ]
        rows[2]["question"] = ("A sheet of paper is torn into 3 pieces. One piece is torn again into 3. "
                               "Can we obtain 100 pieces? 2017 pieces?")
        source = self.write_input(rows)
        original = self.input.read_bytes()
        replies = [completion(), completion("incorrect"), completion("uncertain"), {"choices": []}]
        with patch.object(api, "urlopen", side_effect=[response(r) for r in replies]) as request:
            report = self.run_rescoring()
        metrics = report["metrics"]
        self.assertEqual((metrics["confirmed_correct"], metrics["incorrect"], metrics["pending_review"]), (3, 2, 2))
        self.assertEqual(metrics["answer_accuracy"], 3 / 7)
        self.assertEqual([metrics[f"stage{i}_correct"] for i in (1, 2, 3)], [1, 1, 1])
        self.assertEqual((metrics["api_samples"], metrics["api_calls"]), (4, 4))
        self.assertEqual(report["pending_review_indices"], [5, 6])
        self.assertEqual(report["original_metrics"], source["metrics"])
        self.assertEqual(metrics["generation_runtime"], 99)
        self.assertEqual(self.input.read_bytes(), original)
        sent = json.loads(request.call_args_list[0].args[0].data)
        submitted = json.loads(sent["messages"][1]["content"])
        self.assertEqual(submitted["candidate_solution"], rows[2]["answer"])
        self.assertEqual(submitted["r1_reference_solution"], rows[2]["reference_answer"])
        self.assertEqual(submitted["dataset_reference_answer"], "1008")
        self.assertNotIn("correct", submitted)
        audit = self.output.with_suffix(".api-review.md").read_text()
        for index in (2, 3, 5, 6):
            self.assertIn(f"## Dataset index {index}\n", audit)
        for index in (0, 1, 4):
            self.assertNotIn(f"## Dataset index {index}\n", audit)
        self.assertIn("API incorrect (1)", audit)
        self.assertIn("Pending review (2)", audit)
        with patch.object(api, "urlopen", side_effect=AssertionError("Completed/pending results are reused")):
            repeated = self.run_rescoring()
        self.assertEqual(repeated["metrics"], report["metrics"])

    def test_pending_retries_do_not_repeat_completed_calls_or_double_count(self):
        self.write_input([row(0), row(1)])
        with patch.object(api, "urlopen", side_effect=[response(completion()), response(completion("uncertain"))]):
            first = self.run_rescoring()
        self.assertEqual(first["metrics"]["answer_accuracy"], 0.5)
        with patch.object(api, "urlopen", return_value=response(completion())) as request:
            retried = self.run_rescoring(retry_pending=True)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(retried["metrics"]["answer_accuracy"], 1)
        self.assertEqual(retried["metrics"]["api_calls"], 3)
        self.assertEqual(retried["pending_review_indices"], [])
        with patch.object(api, "urlopen", side_effect=AssertionError("No extra calls")):
            self.assertEqual(self.run_rescoring()["metrics"], retried["metrics"])

    def test_interrupt_keeps_decisions_and_rebuilds_reports(self):
        self.write_input([row(0), row(1), row(2)])
        with patch.object(api, "urlopen", side_effect=[response(completion()), KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                self.run_rescoring()
        partial = json.loads(self.output.read_text())
        self.assertEqual(partial["metrics"]["confirmed_correct"], 1)
        self.assertEqual(partial["metrics"]["unprocessed_samples"], 2)
        self.output.with_suffix(".md").unlink()
        with patch.object(api, "urlopen", side_effect=[response(completion()), response(completion())]) as request:
            finished = self.run_rescoring()
        self.assertEqual(request.call_count, 2)
        self.assertEqual(finished["metrics"]["answer_accuracy"], 1)
        self.assertTrue(self.output.with_suffix(".md").is_file())

    def test_ledger_survives_report_write_failure(self):
        self.write_input([row(0)])
        with patch.object(api, "urlopen", return_value=response(completion())), \
                patch.object(rescoring, "atomic_write", side_effect=OSError("simulated disk error")):
            with self.assertRaises(OSError):
                self.run_rescoring()
        with patch.object(api, "urlopen", side_effect=AssertionError("Saved verdict must be reused")):
            self.assertEqual(self.run_rescoring()["metrics"]["confirmed_correct"], 1)

    def test_changed_input_config_or_prompt_cannot_reuse_progress(self):
        original = self.write_input([row(0)])
        with patch.object(api, "urlopen", return_value=response(completion())):
            self.run_rescoring()
        for kind in ("input", "config", "prompt"):
            with self.subTest(kind=kind), patch.object(api, "urlopen", side_effect=AssertionError("Must fail before request")):
                changed = deepcopy(original)
                if kind == "input":
                    changed["predictions"][0]["question"] += " changed"
                self.input.write_text(json.dumps(changed))
                config = {**self.config, "max_tokens": 4097 if kind == "config" else self.config["max_tokens"]}
                self.config_path.write_text(yaml.safe_dump(config))
                with patch.object(rescoring, "JUDGE_PROMPT", api.JUDGE_PROMPT + ("changed" if kind == "prompt" else "")):
                    with self.assertRaisesRegex(ValueError, "differs"):
                        self.run_rescoring()

    def test_bad_reports_and_output_collisions_are_rejected(self):
        valid = self.write_input([row(0), row(1)])
        invalid = []
        for key in ("question", "reference_answer", "answer", "truncated", "source_index"):
            bad = deepcopy(valid)
            del bad["predictions"][0][key]
            invalid.append(bad)
        for indices in ([0, 0], [0], [1, 0]):
            bad = deepcopy(valid)
            bad["data"]["indices"] = indices
            invalid.append(bad)
        bad = deepcopy(valid)
        bad["metrics"]["generation_samples"] = 3
        invalid.append(bad)
        bad = deepcopy(valid)
        bad["rescoring"] = {}
        invalid.append(bad)
        for report in invalid:
            self.input.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                self.run_rescoring()
        self.input.write_text(json.dumps(valid) + "\n" + json.dumps(valid))
        with self.assertRaisesRegex(ValueError, "JSONL"):
            self.run_rescoring()
        self.input.write_text(json.dumps(valid))
        original = self.input.read_bytes()
        with self.assertRaises(ValueError):
            rescoring.rescore_evaluation(self.input, output=self.input)
        self.assertEqual(self.input.read_bytes(), original)
        self.output.with_suffix(".md").write_text("unrelated report")
        with self.assertRaisesRegex(ValueError, "artifacts exist"):
            self.run_rescoring()
        self.assertEqual(self.output.with_suffix(".md").read_text(), "unrelated report")

    def test_missing_or_invalid_credentials_are_pending_without_request(self):
        self.write_input([row(0)])
        for key in ("", "bad\nkey"):
            with self.subTest(key=key), patch.dict(os.environ, {"ZAI_API_KEY": key}), \
                    patch.object(api, "urlopen", side_effect=AssertionError("No request allowed")):
                report = self.run_rescoring(retry_pending=True)
                self.assertIsNone(report["predictions"][0]["correct"])
                self.assertEqual(report["metrics"]["api_calls"], 0)

    def test_api_contract_for_glm_and_deepseek_and_secret_redaction(self):
        for provider, base, env in (
                ("glm-5.3", "https://open.bigmodel.cn/api/paas/v4/", "ZAI_API_KEY"),
                ("configured-deepseek-model", "https://api.deepseek.com", "DEEPSEEK_API_KEY")):
            config = {**self.config, "model": provider, "base_url": base, "api_key_env": env}
            key = os.environ[env]
            reply = completion(reason="Compared final answers; " + key, model=provider)
            with self.subTest(provider=provider), patch.object(api, "urlopen", return_value=response(reply)) as request:
                decision = api.APIJudge(config)(row(0))
                sent = request.call_args.args[0]
                body = json.loads(sent.data)
                self.assertEqual(sent.full_url, base.rstrip("/") + "/chat/completions")
                self.assertEqual(sent.get_header("Authorization"), "Bearer " + key)
                self.assertEqual(body["model"], provider)
                self.assertFalse(body["stream"])
                self.assertEqual(body["max_tokens"], 4096)
                self.assertEqual(decision["status"], "passed")
                self.assertNotIn(key, json.dumps(decision))
                self.assertEqual(decision["attempts"][0]["returned_model"], provider)
                self.assertEqual(decision["attempts"][0]["usage"]["total_tokens"], 125)
        self.write_input([row(0)])
        with patch.object(api, "urlopen", return_value=response(completion(reason="Contains sk-test-secret"))):
            self.run_rescoring()
        for artifact in self.root.glob("rescored*"):
            self.assertNotIn("sk-test-secret", artifact.read_text())

    def test_http_retry_exhaustion_and_nonretryable_errors(self):
        judge = api.APIJudge(self.config)
        transient = HTTPError("https://example.test", 429, "rate limit", {}, BytesIO(b"sk-test-secret"))
        with patch.object(api, "urlopen", side_effect=[transient, URLError("unavailable"), response(completion())]) as request:
            decision = judge(row(0))
        self.assertEqual((decision["status"], request.call_count), ("passed", 3))
        self.assertNotIn("sk-test-secret", json.dumps(decision))
        broken_error = HTTPError("https://example.test", 502, "truncated error body", {}, BytesIO())
        with patch.object(broken_error, "read", side_effect=IncompleteRead(b"partial")), \
                patch.object(api, "urlopen", side_effect=[broken_error, response(completion())]):
            self.assertEqual(judge(row(0))["status"], "passed")
        for errors, expected in (
                ([URLError("offline")] * 3, 3),
                ([IncompleteRead(b"cut response")] * 3, 3),
                ([HTTPError("https://example.test", 401, "unauthorized", {}, BytesIO(b"error"))], 1)):
            with patch.object(api, "urlopen", side_effect=errors) as request:
                failed = judge(row(0))
            self.assertEqual(failed["status"], "pending")
            self.assertEqual(request.call_count, expected)

    def test_malformed_uncertain_and_truncated_judge_replies_are_pending(self):
        samples = [
            None, [], {"choices": []}, {"choices": [None]},
            {"choices": [{"finish_reason": "stop", "message": {"content": []}}]},
            completion("uncertain"), completion(finish="length"), completion(finish="content_filter"),
            completion("unsupported verdict"),
        ]
        bad_json = completion()
        bad_json["choices"][0]["message"]["content"] = "not a JSON verdict"
        samples.append(bad_json)
        for sample in samples:
            with self.subTest(sample=sample), patch.object(api, "urlopen", return_value=response(sample)) as request:
                result = api.APIJudge(self.config)(row(0))
                self.assertEqual(result["status"], "pending")
                self.assertEqual(request.call_count, 1)
        with patch.object(api, "urlopen", return_value=BytesIO(b"invalid API JSON")):
            self.assertEqual(api.APIJudge(self.config)(row(0))["reason"], "invalid_api_response")
        with patch.object(api, "urlopen", return_value=BytesIO(b'{"usage": NaN}')):
            self.assertEqual(api.APIJudge(self.config)(row(0))["reason"], "invalid_api_response")

    def test_corrupt_progress_cannot_trigger_new_calls_or_change_reports(self):
        self.write_input([row(0)])
        with patch.object(api, "urlopen", return_value=response(completion())):
            self.run_rescoring()
        ledger = self.output.with_suffix(".progress.jsonl")
        original = ledger.read_text()
        old_report = self.output.read_bytes()
        header, entry = [json.loads(line) for line in original.splitlines()]
        invalid_entries = []
        for value in (None, -1, float("inf")):
            changed = deepcopy(entry)
            changed["result"]["rescoring_runtime"] = value
            invalid_entries.append(changed)
        changed = deepcopy(entry)
        changed["result"]["stages"]["stage3"]["status"] = "not_run"
        invalid_entries.append(changed)
        changed = deepcopy(entry)
        changed["index"] = 999
        invalid_entries.append(changed)
        for changed in invalid_entries:
            ledger.write_text(json.dumps(header) + "\n" + json.dumps(changed) + "\n")
            with patch.object(api, "urlopen", side_effect=AssertionError("No new calls")), self.assertRaises(ValueError):
                self.run_rescoring()
            self.assertEqual(self.output.read_bytes(), old_report)
        ledger.write_text(original + json.dumps(entry) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_rescoring()

    def test_config_validation_and_cli_have_no_training_imports(self):
        for patch_config in (
                {"max_attempts": 0}, {"max_tokens": True}, {"timeout_seconds": float("nan")},
                {"api_key": "must not be accepted"}, {"base_url": "https://secret:password@example.test"},
                {"request_parameters": {"messages": []}}, {"request_parameters": {"max_tokens": 1}}):
            with self.subTest(config=patch_config), self.assertRaises(ValueError):
                api.validate_config({**self.config, **patch_config})
        script = Path(rescoring.__file__)
        completed = subprocess.run(
            [sys.executable, "-c", "import rescore_evaluation, sys; assert 'torch' not in sys.modules; rescore_evaluation.main()", "--help"],
            cwd=script.parent, capture_output=True, text=True, check=True)
        self.assertIn("--retry-pending", completed.stdout)
        self.assertIn("--input", completed.stdout)

    def test_cli_default_config_and_resume_outside_project_directory(self):
        self.write_input([
            row(0, answer=r"\boxed{2}"),
            row(1, reference=r"\frac{1}{12}", r1=r"\boxed{B}", answer=r"\boxed{B}"),
            row(2, answer=r"\boxed{2}", truncated=True),
        ])
        command = [sys.executable, str(Path(rescoring.__file__)), "--input", str(self.input),
                   "--output", str(self.output)]
        for _ in range(2):
            completed = subprocess.run(command, cwd=self.root, env={**os.environ, "ZAI_API_KEY": ""},
                                       capture_output=True, text=True, check=True)
            self.assertIn("66.67% (2/3)", completed.stdout)
            report = json.loads(self.output.read_text())
            self.assertEqual(report["metrics"]["api_samples"], 0)
            self.assertEqual(report["metrics"]["api_calls"], 0)


if __name__ == "__main__":
    unittest.main()
