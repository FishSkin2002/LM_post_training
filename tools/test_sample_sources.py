"""Offline sampling checks with random tiny models and synthetic questions."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from huggingface_hub import ResolvedRevision
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from transformers.models.auto import auto_factory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sample_checkpoint
from sft_runtime import attach_lora


class SampleSourceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.adapter = self.root / "adapter"
        self.model_name = "Qwen/Qwen3-4B"
        self.resolved_revision = "b" * 40
        torch.manual_seed(123)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
        ))
        model.save_pretrained(self.base)
        words = ["<pad>", "<eos>", "<unk>", "<think>", "</think>", "question",
                 "zero", "one", "two", "three", "four", "answer", "yes", "no", "3", "4"]
        backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
            additional_special_tokens=["<think>", "</think>"],
        )
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}{% if add_generation_prompt %}<think> {% endif %}"
        tokenizer.save_pretrained(self.base)
        self.model_loader = AutoModelForCausalLM.from_pretrained
        self.tokenizer_loader = AutoTokenizer.from_pretrained
        base = self.model_loader(self.base, local_files_only=True)
        attach_lora(base, {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0}).save_pretrained(self.adapter)
        tokenizer.save_pretrained(self.adapter)
        self.rows = [{"question": f"question {word}", "answer": f"reference {word}"}
                     for word in ("zero", "one", "two", "three", "four")]
        self.model_calls = []
        self.tokenizer_calls = []

    def load_model(self, source, **kwargs):
        if source == self.model_name:
            self.model_calls.append(kwargs)
            # Different loaders may consume RNG state; generation must reset it.
            torch.rand(37)
            local_kwargs = {key: value for key, value in kwargs.items() if key != "revision"}
            return self.model_loader(self.base, **local_kwargs)
        return self.model_loader(source, **kwargs)

    def load_tokenizer(self, source, **kwargs):
        if source == self.model_name:
            self.tokenizer_calls.append(kwargs)
            return self.tokenizer_loader(self.base, local_files_only=True)
        return self.tokenizer_loader(source, **kwargs)

    def sample(self, *args):
        with patch.object(sys, "argv", ["sample_checkpoint.py", "--device", "cpu",
                                       "--num-samples", "3", "--max-new-tokens", "4", *args]), \
                patch.object(sample_checkpoint, "__file__", str(self.root / "sample_checkpoint.py")), \
                patch.object(sample_checkpoint, "load_dataset", return_value=self.rows), \
                patch.object(sample_checkpoint, "resolve_revision", return_value=ResolvedRevision(self.resolved_revision, initial="baseline")) as resolver, \
                patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=self.load_model), \
                patch.object(AutoTokenizer, "from_pretrained", side_effect=self.load_tokenizer), \
                contextlib.redirect_stdout(io.StringIO()) as terminal:
            sample_checkpoint.main()
            if "--model-name" in args:
                resolver.assert_called_once_with(
                    self.model_name, "baseline" if "--revision" in args else None,
                    local_files_only="--local-files-only" in args,
                )
            else:
                resolver.assert_not_called()
        message = terminal.getvalue().strip()
        self.assertTrue(message.startswith("Saved report to "))
        self.assertEqual(len(message.splitlines()), 1)
        report = Path(message.removeprefix("Saved report to "))
        return report, report.read_text()

    def test_original_full_and_zero_adapter_use_same_questions_and_generation_seed(self):
        with patch.object(sample_checkpoint, "load_checkpoint_model", side_effect=AssertionError("Original model must not load an adapter.")):
            path, original = self.sample("--model-name", self.model_name, "--revision", "baseline")
        self.assertEqual(path.name, "original-Qwen--Qwen3-4B-test-seed42.md")
        self.assertIn("# 原始模型样本问答", original)
        self.assertIn(self.model_name, original)
        self.assertIn(self.resolved_revision, original)
        self.assertFalse(self.model_calls[0]["local_files_only"])
        self.assertEqual(self.model_calls[0]["revision"], self.resolved_revision)
        self.assertEqual(self.tokenizer_calls[0]["revision"], self.resolved_revision)
        _, full = self.sample("--checkpoint", str(self.base))
        _, adapter = self.sample("--checkpoint", str(self.adapter))
        for checkpoint_report in (full, adapter):
            self.assertEqual(original.split("## 样本 1", 1)[1], checkpoint_report.split("## 样本 1", 1)[1])
            self.assertIn("样本索引：[0, 4, 2]", checkpoint_report)
        self.assertTrue((self.root / "outputs/samples/base-test-seed42.md").is_file())
        self.assertTrue((self.root / "outputs/samples/adapter-test-seed42.md").is_file())

    def test_cache_only_custom_output_and_repeatability(self):
        custom = self.root / "custom/answers.md"
        path, first = self.sample("--model-name", self.model_name, "--local-files-only", "--output", str(custom))
        self.assertEqual(path, custom)
        self.assertTrue(self.model_calls[-1]["local_files_only"])
        self.assertTrue(self.tokenizer_calls[-1]["local_files_only"])
        _, repeated = self.sample("--model-name", self.model_name, "--local-files-only", "--output", str(custom))
        self.assertEqual(first, repeated)

    def test_invalid_source_selection_fails_before_loading(self):
        cases = [[], ["--model-name", self.model_name, "--checkpoint", str(self.base)],
                 ["--model-name", " "], ["--checkpoint", str(self.base), "--revision", "main"]]
        for args in cases:
            with self.subTest(args=args), patch.object(sys, "argv", ["sample_checkpoint.py", *args]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                sample_checkpoint.parse_args()
            self.assertEqual(error.exception.code, 2)

    def test_cache_only_reaches_real_transformers_adapter_probe(self):
        class ProbeReached(Exception):
            pass

        def inspect_probe(source, **kwargs):
            self.assertEqual(source, self.model_name)
            self.assertTrue(kwargs.get("local_files_only", False), "Adapter probe lost the cache-only flag.")
            raise ProbeReached

        revision = ResolvedRevision(self.resolved_revision, initial="main")
        adapter_config = self.adapter / "adapter_config.json"
        config = json.loads(adapter_config.read_text())
        config["base_model_name_or_path"] = self.model_name
        adapter_config.write_text(json.dumps(config))
        for source in (["--model-name", self.model_name, "--local-files-only"],
                       ["--checkpoint", str(self.adapter)]):
            with self.subTest(source=source), \
                    patch.object(sys, "argv", ["sample_checkpoint.py", "--device", "cpu", "--num-samples", "1", *source]), \
                    patch.object(sample_checkpoint, "load_dataset", return_value=self.rows), \
                    patch.object(sample_checkpoint, "resolve_revision", return_value=revision), \
                    patch.object(auto_factory, "resolve_revision", return_value=revision), \
                    patch.object(auto_factory, "find_adapter_config_file", side_effect=inspect_probe), \
                    self.assertRaises(ProbeReached):
                sample_checkpoint.main()

    def test_short_qwen_name_hint_preserves_existing_local_directories(self):
        for name in ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"):
            with self.subTest(name=name), patch.object(sys, "argv", ["sample_checkpoint.py", "--model-name", name]), \
                    contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
                sample_checkpoint.parse_args()
            self.assertIn(f"Qwen/{name}", error.getvalue())
        (self.root / "Qwen3-4B").mkdir()
        with contextlib.chdir(self.root), patch.object(sys, "argv", ["sample_checkpoint.py", "--model-name", "Qwen3-4B"]):
            args, _ = sample_checkpoint.parse_args()
        self.assertEqual(args.model_name, "Qwen3-4B")


if __name__ == "__main__":
    unittest.main()
