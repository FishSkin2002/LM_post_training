"""Offline integration checks using random tiny Qwen3 models, never pretrained weights."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM,
    Trainer, TrainerCallback,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sample_checkpoint
import train_sft
from sft_runtime import attach_lora, load_checkpoint_model, validate_resume


class StopAfterOne(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        control.should_training_stop = state.global_step == 1
        return control


class InterruptedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(StopAfterOne())


class LoRAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.options = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, tie_word_embeddings=True,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        ))
        model.save_pretrained(self.base)
        vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        vocabulary.update({f"t{index}": index for index in range(4, 64)})
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="<unk>")),
            pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
        )
        tokenizer.save_pretrained(self.base)
        rows = []
        for length in (5, 12, 8):
            ids = list(range(4, 4 + length))
            rows.append({"input_ids": ids, "attention_mask": [1] * length,
                         "labels": [-100] * 2 + ids[2:]})
        self.dataset = Dataset.from_list(rows)
        config = yaml.safe_load((train_sft.CONFIG_DIR / "development/sft.yaml").read_text())
        config.update(
            model_name=str(self.base), output_dir=str(self.root / "run"),
            use_cpu=True, dataloader_pin_memory=False, disable_tqdm=True,
            gradient_accumulation_steps=2, report_to="tensorboard",
        )
        self.config = self.root / "sft.yaml"
        self.config.write_text(yaml.safe_dump(config))

    def run_training(self, *args, trainer=Trainer):
        output = io.StringIO()
        with patch.object(sys, "argv", ["train_sft.py", "--config", str(self.config), *args]), \
                patch.object(train_sft, "load_training_data", return_value=self.dataset), \
                patch.object(train_sft, "Trainer", trainer), contextlib.redirect_stdout(output):
            train_sft.main()
        return output.getvalue()

    def test_low_precision_base_frozen_and_fp32_adapters_receive_gradients(self):
        base = AutoModelForCausalLM.from_pretrained(self.base, dtype=torch.bfloat16)
        model = attach_lora(base, self.options)
        model.config.use_cache = False
        model.gradient_checkpointing_enable({"use_reentrant": False})
        model.train()
        frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
        parameters = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        ids = torch.tensor([[4, 5, 6, 7, 8]])
        loss = model(input_ids=ids, labels=ids).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(value.dtype == torch.float32 for value in parameters))
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all() for value in parameters))
        self.assertTrue(any(value.grad.abs().sum() > 0 for value in parameters))
        optimizer.step()
        for name, value in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(value, frozen[name]), name)
        self.assertTrue(all("lora_" in name for name, value in model.named_parameters() if value.requires_grad))

    def test_real_trainer_resume_matches_uninterrupted_adapter_and_saves_small_artifacts(self):
        output = self.run_training("--memory-smoke-test", trainer=InterruptedTrainer)
        self.assertIn("repeating longest example (index 1)", output)
        run = self.root / "run-memory-smoke"
        checkpoint = run / "checkpoint-1"
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "tokenizer.json"):
            self.assertTrue((checkpoint / name).is_file(), name)
        self.assertFalse((checkpoint / "model.safetensors").exists())
        self.assertFalse((checkpoint / "config.json").exists())
        self.run_training("--memory-smoke-test", "--resume-from-checkpoint", str(checkpoint))
        state = json.loads((run / "checkpoint-2/trainer_state.json").read_text())
        self.assertEqual(state["global_step"], 2)
        self.assertEqual(state["log_history"][-1]["step"], 2)
        memory = json.loads((run / "memory_results.json").read_text())
        self.assertEqual(memory["longest_example_tokens"], 12)
        self.assertEqual(memory["training_mode"], "lora")
        self.assertNotIn("peak_allocated_gib", memory)  # CPU tests make no GPU claim.
        (self.root / "continuous").mkdir()
        marker = self.root / "continuous/old-result.txt"
        marker.write_text("keep")
        self.run_training("--memory-smoke-test", "--output-dir", str(self.root / "continuous"))
        self.assertEqual(marker.read_text(), "keep")
        resumed = load_checkpoint_model(run / "final", dtype=torch.float32, device="cpu", attention_implementation="sdpa")
        continuous = load_checkpoint_model(self.root / "continuous-1/final", dtype=torch.float32, device="cpu", attention_implementation="sdpa")
        for name, value in resumed.named_parameters():
            if "lora_" in name:
                torch.testing.assert_close(value, dict(continuous.named_parameters())[name], rtol=0, atol=0)
        self.assertTrue((run / "final/tokenizer.json").is_file())
        self.assertTrue(list((run / "runs").rglob("events.out.tfevents.*")))
        self.assertTrue(list((self.root / "continuous-1/runs").rglob("events.out.tfevents.*")))
        self.assertFalse((self.root / "continuous/runs").exists())
        with self.assertRaisesRegex(ValueError, "training mode"):
            validate_resume(checkpoint, "full", str(self.base), self.options)
        with self.assertRaisesRegex(ValueError, "base model"):
            validate_resume(checkpoint, "lora", "different-base", self.options)
        with self.assertRaisesRegex(ValueError, "lora_alpha"):
            validate_resume(checkpoint, "lora", str(self.base), {**self.options, "lora_alpha": 16})
        with self.assertRaisesRegex(ValueError, "not final"):
            validate_resume(run / "final", "lora", str(self.base), self.options)
        optimizer_path = checkpoint / "optimizer.pt"
        optimizer_path.rename(checkpoint / "temporarily-missing-optimizer.pt")
        with self.assertRaisesRegex(ValueError, "missing optimizer.pt"):
            validate_resume(checkpoint, "lora", str(self.base), self.options)
        report = self.root / "answers.md"
        with patch.object(sys, "argv", ["sample_checkpoint.py", "--checkpoint", str(run / "final"),
                                       "--device", "cpu", "--num-samples", "1", "--output", str(report)]), \
                patch.object(sample_checkpoint, "load_dataset", return_value=[{"question": "toy question", "answer": "toy reference"}]), \
                patch.object(sample_checkpoint, "generate_answer", return_value=("toy reasoning", "toy result", 3, False)), \
                contextlib.redirect_stdout(io.StringIO()) as terminal:
            sample_checkpoint.main()
        self.assertEqual(terminal.getvalue().strip(), f"Saved report to {report}")
        self.assertIn("toy reference", report.read_text())

    def test_full_mode_still_saves_and_loads_a_full_model(self):
        self.run_training("--training-mode", "full", "--learning-rate", "0.00002", "--memory-smoke-test", "--max-steps", "1")
        run = self.root / "run-memory-smoke"
        self.assertTrue((run / "final/config.json").is_file())
        self.assertTrue((run / "final/model.safetensors").is_file())
        self.assertFalse((run / "final/adapter_config.json").exists())
        model = load_checkpoint_model(run / "final", dtype=torch.float32, device="cpu", attention_implementation="sdpa")
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(input_ids=torch.tensor([[4, 5, 6]])).logits).all())
        with self.assertRaisesRegex(ValueError, "training mode"):
            validate_resume(run / "checkpoint-1", "lora", str(self.base), self.options)

    def test_precision_selection_and_all_profiles(self):
        for supported, expected in ((True, torch.bfloat16), (False, torch.float16)):
            args = SimpleNamespace(training_mode="lora", training_config={"bf16": "auto"})
            with patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(torch.cuda, "is_bf16_supported", return_value=supported), \
                    patch.object(train_sft, "TrainingArguments", side_effect=lambda **kwargs: kwargs):
                configured = train_sft.make_training_args(args)
            self.assertEqual(args.model_dtype, expected)
            self.assertEqual(configured["fp16"], not supported)
        for relative in ("development/sft.yaml", "training/sft.yaml", "training/sft-8b.yaml"):
            with patch.object(sys, "argv", ["train_sft.py", "--config", str(train_sft.CONFIG_DIR / relative)]):
                args = train_sft.parse_args()
            args.training_config.update(use_cpu=True, report_to="none")
            configured = train_sft.make_training_args(args)
            self.assertEqual(args.training_mode, "lora")
            self.assertEqual(configured.learning_rate, 1e-4)
            self.assertEqual(args.model_dtype, torch.float32)

    @unittest.skipUnless(os.environ.get("QWEN_TEST_CUDA") == "1", "Opt in to random tiny-model CUDA checks.")
    def test_cuda_bf16_and_fp16_training_and_memory_report(self):
        from accelerate.state import AcceleratorState

        self.assertTrue(torch.cuda.is_available())
        self.assertTrue(torch.cuda.is_bf16_supported())
        for bf16, dtype in (("auto", "torch.bfloat16"), (False, "torch.float16")):
            AcceleratorState._reset_state(reset_partial_state=True)
            config = yaml.safe_load(self.config.read_text())
            config.update(use_cpu=False, bf16=bf16, report_to="none")
            self.config.write_text(yaml.safe_dump(config))
            output = self.root / ("bf16" if bf16 else "fp16")
            self.run_training("--memory-smoke-test", "--output-dir", str(output))
            memory = json.loads((output / "memory_results.json").read_text())
            self.assertEqual(memory["base_dtype"], dtype)
            self.assertEqual(memory["fp16"], bf16 is False)
            self.assertEqual(memory["global_step"], 2)
            self.assertGreater(memory["peak_allocated_gib"], 0)
            self.assertGreaterEqual(memory["peak_reserved_gib"], memory["peak_allocated_gib"])
            self.assertGreater(memory["total_memory_gib"], memory["peak_reserved_gib"])
            trained = load_checkpoint_model(output / "final", dtype=torch.float32, device="cpu", attention_implementation="sdpa")
            self.assertTrue(all(torch.isfinite(value).all() for value in trained.parameters()))


if __name__ == "__main__":
    unittest.main()
