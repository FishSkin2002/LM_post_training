"""Offline HTTP contracts, parser behavior and credential handling for the API judge."""

from http.client import IncompleteRead
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils.answer_judge as api
import summarize_evaluation as summarizing
import utils.math_scoring as scoring
import utils.rule_scoring as rules


def row(index, *, reference="2", r1=r"\boxed{3}", answer=r"\boxed{4}", truncated=False, correct=None):
    if correct is None:
        try:
            correct = scoring.score_answer(answer, reference, truncated)[0]
        except ValueError:
            correct = False
    return {"index": index, "source_index": index + 100, "generation_index": 1,
            "question": f"Problem {index}", "reference_final_answer": reference,
            "reference_answer": r1, "answer": answer, "correct": correct,
            "format_valid": True, "truncated": truncated, "generated_tokens": 25,
            "generation_runtime": 2, "generation_batch_size": 4}


def completion(verdict="correct", reason="Final conclusions match.", *, model="deepseek-flash", finish="stop"):
    return {"model": model, "choices": [{"finish_reason": finish, "message": {
        "role": "assistant", "content": json.dumps({"verdict": verdict, "reason": reason}),
        "reasoning_content": "Provider-specific reasoning is not the verdict.",
    }}], "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}}


def response(value):
    return BytesIO(json.dumps(value).encode())


class AnswerJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = api.load_judge_config()
        self.config.update(max_attempts=3, retry_delay_seconds=0)
        self.env = patch.dict(os.environ, {"ZAI_API_KEY": "sk-test-secret", "DEEPSEEK_API_KEY": "ds-test-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_credentials_are_validated_and_instance_secrets_and_prompt_do_not_change_environment(self):
        for key in ("", "bad\nkey", "非ASCII"):
            with self.subTest(key=key), patch.dict(os.environ, {"DEEPSEEK_API_KEY": key}), \
                    patch.object(api, "urlopen", side_effect=AssertionError("No request with invalid credentials")):
                decision = api.APIJudge(self.config)(row(0))
            self.assertEqual(decision["status"], "pending")
            self.assertEqual(decision["attempts"], [])
        key = "instance-only-key"
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
                patch.object(api, "urlopen", return_value=response(completion(reason="Matches " + key))) as request:
            decision = api.APIJudge(self.config, api_key=key, prompt="Saved judge prompt")(row(0))
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "")
        sent = json.loads(request.call_args.args[0].data)
        self.assertEqual(sent["messages"][0]["content"], "Saved judge prompt")
        self.assertNotIn(key, json.dumps(decision))

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
                self.assertEqual(rules.single_formal_box(text)[1], error)

    def test_api_contract_for_glm_and_deepseek_and_secret_redaction(self):
        for config in (
                {**self.config, "model": "glm-5.3", "base_url": "https://open.bigmodel.cn/api/paas/v4/",
                 "api_key_env": "ZAI_API_KEY", "request_parameters": {}},
                self.config):
            provider, base, env = (config[name] for name in ("model", "base_url", "api_key_env"))
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
                if env == "DEEPSEEK_API_KEY":
                    self.assertEqual(body["thinking"], {"type": "disabled"})
                    self.assertEqual(body["response_format"], {"type": "json_object"})
                else:
                    self.assertNotIn("thinking", body)
                    self.assertNotIn("response_format", body)
                self.assertEqual(decision["status"], "passed")
                self.assertNotIn(key, json.dumps(decision))
                self.assertEqual(decision["attempts"][0]["returned_model"], provider)
                self.assertEqual(decision["attempts"][0]["usage"]["total_tokens"], 125)

    def test_http_retry_exhaustion_and_nonretryable_errors(self):
        judge = api.APIJudge(self.config)
        transient = HTTPError("https://example.test", 429, "rate limit", {}, BytesIO(b"ds-test-secret"))
        with patch.object(api, "urlopen", side_effect=[transient, URLError("unavailable"), response(completion())]) as request:
            decision = judge(row(0))
        self.assertEqual((decision["status"], request.call_count), ("passed", 3))
        self.assertNotIn("ds-test-secret", json.dumps(decision))
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

    def test_config_validation_and_cli_have_no_training_imports(self):
        for patch_config in (
                {"max_attempts": 0}, {"max_tokens": True}, {"timeout_seconds": float("nan")},
                {"api_key": "must not be accepted"}, {"base_url": "https://secret:password@example.test"},
                {"request_parameters": {"messages": []}}, {"request_parameters": {"max_tokens": 1}}):
            with self.subTest(config=patch_config), self.assertRaises(ValueError):
                api.validate_config({**self.config, **patch_config})
        script = Path(summarizing.__file__)
        completed = subprocess.run(
            [sys.executable, "-c", "import summarize_evaluation, sys; assert 'torch' not in sys.modules; summarize_evaluation.main()", "--help"],
            cwd=script.parent, capture_output=True, text=True, check=True)
        self.assertIn("--retry-pending", completed.stdout)
        self.assertIn("--evaluation", completed.stdout)
        self.assertIn("--no-api", completed.stdout)


if __name__ == "__main__":
    unittest.main()
