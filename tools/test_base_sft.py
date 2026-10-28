"""Offline Base LoRA integration checks with random tiny weights and synthetic data."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, LogitsProcessor,
    PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Trainer, TrainerCallback,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_sft_data as data
import sample_checkpoint
import train_sft


class StopAfterOne(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        control.should_training_stop = state.global_step == 1


class InterruptedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(StopAfterOne())


class ForceToken(LogitsProcessor):
    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, input_ids, scores):
        scores.fill_(-torch.inf)
        scores[:, self.token_id] = 0
        return scores


class BaseSFTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "tiny-base"
        self.model_name = "Qwen/Qwen3-0.6B-Base"
        words = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>",
                 "user", "assistant", "question", "one", "two", "three", "answer", "####",
                 "1", "2", "3", "5", "+", "=", "<<2+3=5>>5", "reference"]
        words += [f"token{i}" for i in range(len(words), 64)]
        backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
            unk_token="<unk>", additional_special_tokens=["<|im_start|>", "<|im_end|>"],
            chat_template=data.CHAT_TEMPLATE,
        )
        self.tokenizer.save_pretrained(self.base)
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=128, tie_word_embeddings=True, eos_token_id=0, pad_token_id=0,
        ))
        model.save_pretrained(self.base)
        self.initial_weights = {name: value.detach().clone() for name, value in model.named_parameters()}
        self.rows = [
            {"question": "question one", "answer": "2 + 3 = <<2+3=5>>5\n#### 5"},
            {"question": "question two", "answer": "answer 1 + 2\n#### 3"},
        ]
        self.config = data.load_config(data.CONFIG_DIR / "development/sft.yaml")
        self.config.update(num_train_epochs=1, use_cpu=True, disable_tqdm=True,
                           dataloader_pin_memory=False)
        self.model_loader = AutoModelForCausalLM.from_pretrained
        self.tokenizer_loader = AutoTokenizer.from_pretrained
        self.model_calls = []

    def load_model(self, source, **kwargs):
        self.assertEqual(source, self.model_name)
        self.model_calls.append(kwargs.copy())
        model = self.model_loader(self.base, **{**kwargs, "local_files_only": True})
        model.name_or_path = self.model_name
        model.config._name_or_path = self.model_name
        return model

    def load_tokenizer(self, source, **kwargs):
        return self.tokenizer_loader(self.base if source == self.model_name else source,
                                     **{**kwargs, "local_files_only": True})

    def run_training(self, name, *, resume=None, trainer=Trainer):
        output = self.root / name
        config = self.root / f"{name}.yaml"
        config.write_text(yaml.safe_dump({**self.config, "output_dir": str(output)}))
        argv = ["train_sft.py", "--config", str(config)]
        if resume:
            argv += ["--resume-from-checkpoint", str(resume)]
        with patch.object(sys, "argv", argv), \
                patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=self.load_model), \
                patch.object(AutoTokenizer, "from_pretrained", side_effect=self.load_tokenizer), \
                patch.object(train_sft, "load_dataset", return_value=Dataset.from_list(self.rows)), \
                patch.object(train_sft, "Trainer", trainer), contextlib.redirect_stdout(io.StringIO()):
            train_sft.main()
        return output

    def sample(self, *source):
        output = self.root / f"report-{len(list(self.root.glob('report-*.md')))}.md"
        argv = ["sample_checkpoint.py", *source, "--device", "cpu", "--num-samples", "2",
                "--max-new-tokens", "4", "--output", str(output)]
        with patch.object(sys, "argv", argv), \
                patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=self.load_model), \
                patch.object(AutoTokenizer, "from_pretrained", side_effect=self.load_tokenizer), \
                patch.object(sample_checkpoint, "load_dataset", return_value=Dataset.from_list(self.rows)), \
                contextlib.redirect_stdout(io.StringIO()) as terminal:
            sample_checkpoint.main()
        self.assertEqual(terminal.getvalue().strip(), f"Saved report to {output}")
        return output.read_text()

    def test_template_masks_and_padding_preserve_complete_reference(self):
        row = self.rows[0]
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]},
             {"role": "assistant", "content": row["answer"]}], tokenize=False,
        )
        self.assertEqual(text, f"<|im_start|>user\n{row['question']}<|im_end|>\n"
                               f"<|im_start|>assistant\n{row['answer']}<|im_end|>")
        examples = [data.build_sft_example(self.tokenizer, **row) for row in self.rows]
        for example, row in zip(examples, self.rows):
            answer_ids = self.tokenizer.encode(row["answer"], add_special_tokens=False) + [2]
            self.assertEqual([value for value in example["labels"] if value != -100], answer_ids)
            prefix_len = len(example["input_ids"]) - len(answer_ids)
            self.assertEqual(example["labels"][:prefix_len], [-100] * prefix_len)
            self.assertEqual(example["input_ids"][-1], 2)
        batch = DataCollatorForSeq2Seq(self.tokenizer)(examples)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)

    def test_training_updates_only_fp32_lora_and_resume_matches(self):
        continuous = self.run_training("continuous")
        interrupted = self.run_training("interrupted", trainer=InterruptedTrainer)
        checkpoint = interrupted / "checkpoint-1"
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                     "scheduler.pt", "rng_state.pth", "trainer_state.json", "tokenizer.json"):
            self.assertTrue((checkpoint / name).is_file(), name)
        self.run_training("interrupted", resume=checkpoint)
        expected = load_file(continuous / "final/adapter_model.safetensors")
        resumed = load_file(interrupted / "final/adapter_model.safetensors")
        self.assertEqual(expected.keys(), resumed.keys())
        for name, value in expected.items():
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all(), name)
            self.assertTrue(torch.equal(value, resumed[name]), name)
        self.assertTrue(any(value.abs().sum() > 0 for name, value in expected.items() if "lora_B" in name))
        base = self.model_loader(self.base, local_files_only=True, dtype=torch.float32)
        model = PeftModel.from_pretrained(base, continuous / "final", local_files_only=True)
        for name, value in model.get_base_model().named_parameters():
            if "lora_" not in name:
                self.assertTrue(torch.equal(value, self.initial_weights[name.replace(".base_layer.", ".")]), name)
        self.assertEqual(self.tokenizer_loader(continuous / "final").chat_template, data.CHAT_TEMPLATE)
        self.assertFalse((continuous / "final/model.safetensors").exists())
        state = json.loads((interrupted / "trainer_state.json").read_text())
        self.assertEqual(state["global_step"], 2)
        training_args = torch.load(checkpoint / "training_args.bin", weights_only=False)
        self.assertFalse(training_args.bf16 or training_args.fp16 or training_args.gradient_checkpointing)
        self.assertEqual(training_args.gradient_accumulation_steps, 1)
        self.assertEqual(training_args.report_to, [])
        with self.assertRaises(SystemExit):
            self.run_training("continuous")

    def test_original_and_zero_adapter_sample_same_answers_without_reference_input(self):
        base = self.load_model(self.model_name, dtype=torch.float32)
        adapter = self.root / "adapter"
        get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", r=2)).save_pretrained(adapter)
        self.tokenizer.save_pretrained(adapter)
        original = self.sample("--model-name", self.model_name, "--local-files-only")
        tuned = self.sample("--checkpoint", str(adapter))
        self.assertEqual(original[original.index("## 样本"):], tuned[tuned.index("## 样本"):])
        self.assertIn(self.rows[0]["answer"], original)
        self.assertTrue(all(call["local_files_only"] for call in self.model_calls[1:]))
        self.assertTrue(all(call["adapter_kwargs"]["local_files_only"] for call in self.model_calls[1:]))
        model = self.model_loader(self.base, local_files_only=True)
        generate = model.generate
        with patch.object(model, "generate", wraps=generate) as called:
            sample_checkpoint.generate_answer(model, self.tokenizer, self.rows[0]["question"], 4)
        prompt = self.tokenizer.decode(called.call_args.kwargs["input_ids"][0], skip_special_tokens=False)
        self.assertNotIn("####", prompt)
        self.assertTrue(prompt.endswith("assistant"))

    def test_real_generation_stops_at_assistant_end_and_reports_truncation(self):
        model = self.model_loader(self.base, local_files_only=True)
        generate = model.generate
        with patch.object(model, "generate", side_effect=lambda **kwargs: generate(
                **kwargs, logits_processor=[ForceToken(2)])):
            answer, count, truncated = sample_checkpoint.generate_answer(model, self.tokenizer, "question one", 4)
        self.assertEqual((answer, count, truncated), ("", 1, False))
        with patch.object(model, "generate", side_effect=lambda **kwargs: generate(
                **kwargs, logits_processor=[ForceToken(7)])):
            _, count, truncated = sample_checkpoint.generate_answer(model, self.tokenizer, "question one", 4)
        self.assertEqual((count, truncated), (4, True))
        with self.assertRaisesRegex(ValueError, "exceed"):
            sample_checkpoint.generate_answer(model, self.tokenizer, "question one", 128)

    def test_plain_qwen_and_full_weight_checkpoints_are_not_sources(self):
        with self.assertRaises(ValueError):
            data.require_base_model("Qwen/Qwen3-4B")
        for source in (("--model-name", "Qwen/Qwen3-4B"), ("--checkpoint", str(self.base))):
            with self.assertRaises(SystemExit):
                self.sample(*source)


if __name__ == "__main__":
    unittest.main()
