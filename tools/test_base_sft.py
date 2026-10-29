"""Offline Base LoRA integration checks with random tiny weights and synthetic data."""

import contextlib
import io
import json
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, LogitsProcessor,
    PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Trainer, TrainerCallback,
)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_sft_data as data
import evaluate_checkpoint as evaluation
import sample_checkpoint
import train_sft
import training_workflow as workflow


class TrainingInterrupted(RuntimeError):
    pass


class StopAfterSave(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.global_step == 2:
            raise TrainingInterrupted("Simulated interruption after a complete checkpoint.")


class InterruptedTrainer(workflow.WorkflowTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(StopAfterSave())


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
        self.enterContext(patch.object(evaluation, "MAX_NEW_TOKENS", 4))
        self.base = self.root / "tiny-base"
        self.model_name = "Qwen/Qwen3-0.6B-Base"
        words = ["<|endoftext|>", "<unk>", "Question:", "Answer:",
                 "question", "one", "two", "three", "answer", "####",
                 "1", "2", "3", "4", "5", "+", "=", "reference"]
        words += [f"token{i}" for i in range(len(words), 64)]
        backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
            unk_token="<unk>",
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
        self.training_rows = [{"question": f"question {i}", "answer": f"answer <<{i}+0={i}>>\n#### {i}"}
                              for i in range(20)]
        self.config = data.load_config(data.CONFIG_DIR / "development/sft.yaml")
        self.config.update(num_train_epochs=1, use_cpu=True, disable_tqdm=True,
                           dataloader_pin_memory=False, logging_steps=1, eval_steps=2, save_steps=2)
        self.model_loader = AutoModelForCausalLM.from_pretrained
        self.tokenizer_loader = AutoTokenizer.from_pretrained
        self.model_calls = []
        self.evaluation_count = 0

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

    def run_training(self, name, *, resume=None, continue_from=None, additional_epochs=None,
                     trainer=workflow.WorkflowTrainer):
        output = self.root / name
        config = self.root / f"{name}.yaml"
        config.write_text(yaml.safe_dump({**self.config, "output_dir": str(output)}))
        argv = ["train_sft.py", "--config", str(config)]
        if resume:
            argv += ["--resume-from-checkpoint", str(resume)]
        if continue_from:
            argv += ["--continue-from", str(continue_from), "--additional-epochs", str(additional_epochs)]
        with patch.object(sys, "argv", argv), \
                patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=self.load_model), \
                patch.object(AutoTokenizer, "from_pretrained", side_effect=self.load_tokenizer), \
                patch.object(train_sft, "load_dataset", return_value=Dataset.from_list(self.training_rows)), \
                patch.object(train_sft, "WorkflowTrainer", trainer), \
                contextlib.redirect_stdout(io.StringIO()) as terminal, \
                contextlib.redirect_stderr(io.StringIO()) as progress:
            train_sft.main()
        self.last_terminal, self.last_progress = terminal.getvalue(), progress.getvalue()
        line = next(line for line in terminal.getvalue().splitlines() if line.startswith("Output directory: "))
        return Path(line.removeprefix("Output directory: "))

    def evaluate(self, checkpoint=None, num_samples=None, rows=None, output=None, *, model_name=None):
        self.evaluation_count += 1
        output = output or self.root / f"evaluation-{self.evaluation_count}.json"
        argv = ["evaluate_checkpoint.py", "--device", "cpu", "--output", str(output)]
        if checkpoint is not None:
            argv += ["--checkpoint", str(checkpoint)]
        if model_name is not None:
            argv += ["--model-name", model_name]
        if num_samples is not None:
            argv += ["--num-samples", str(num_samples)]
        with patch.object(sys, "argv", argv), \
                patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=self.load_model), \
                patch.object(AutoTokenizer, "from_pretrained", side_effect=self.load_tokenizer), \
                patch.object(evaluation, "load_dataset", return_value=Dataset.from_list(self.training_rows if rows is None else rows)), \
                patch.object(torch.optim, "AdamW", side_effect=AssertionError("Evaluation must not create an optimizer.")), \
                contextlib.redirect_stdout(io.StringIO()) as terminal, contextlib.redirect_stderr(io.StringIO()):
            evaluation.main()
        self.last_evaluation_terminal = terminal.getvalue()
        return json.loads(output.read_text())

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

    def test_plain_text_masks_and_padding_preserve_cleaned_reference_and_eos(self):
        raw = "  2+3=<<2+3=5>>5; (6/2)=<<(6/2)=3>>3\n\n#### 5  "
        cleaned = "  2+3=5; (6/2)=3\n\n#### 5  "
        self.assertEqual(data.clean_answer(raw), cleaned)
        self.assertEqual(data.clean_answer(cleaned), cleaned)
        self.assertEqual(data.clean_answer(self.rows[1]["answer"]), self.rows[1]["answer"])
        prompt = data.build_prompt("question {one}")
        self.assertTrue(prompt.endswith("Question: question {one}\nAnswer:\n"))
        self.assertIn("if the final answer is 4, write #### 4.", prompt)
        self.assertNotIn("<<", prompt)
        examples = [data.build_sft_example(self.tokenizer, **row) for row in self.rows]
        for example, row in zip(examples, self.rows):
            answer_ids = self.tokenizer.encode(data.clean_answer(row["answer"]), add_special_tokens=False) + [0]
            prompt_ids = self.tokenizer.encode(data.build_prompt(row["question"]), add_special_tokens=False)
            self.assertEqual([value for value in example["labels"] if value != -100], answer_ids)
            self.assertEqual(example["input_ids"], prompt_ids + answer_ids)
            self.assertEqual(example["labels"][:len(prompt_ids)], [-100] * len(prompt_ids))
            self.assertEqual(example["input_ids"][-1], self.tokenizer.eos_token_id)
        batch = DataCollatorForSeq2Seq(self.tokenizer)(examples)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
        for index, example in enumerate(examples):
            self.assertEqual(batch["labels"][index, len(example["input_ids"]) - 1].item(), 0)
            self.assertEqual(batch["attention_mask"][index, len(example["input_ids"]) - 1].item(), 1)

    def test_saved_tokenizer_has_no_bundled_chat_metadata_or_vocab_changes(self):
        self.tokenizer.chat_template = "unused bundled metadata"
        self.tokenizer.save_pretrained(self.base)
        loaded = data.load_tokenizer(self.base, local_files_only=True)
        self.assertIsNone(loaded.chat_template)
        self.assertEqual(loaded.get_vocab(), self.tokenizer.get_vocab())
        destination = self.root / "saved-tokenizer"
        loaded.save_pretrained(destination)
        self.assertFalse((destination / "chat_template.jinja").exists())
        self.assertIsNone(json.loads((destination / "tokenizer_config.json").read_text()).get("chat_template"))

    def test_training_updates_only_fp32_lora_and_resume_matches(self):
        continuous = self.run_training("continuous")
        with self.assertRaises(TrainingInterrupted):
            self.run_training("interrupted", trainer=InterruptedTrainer)
        interrupted = self.root / "interrupted"
        checkpoint = interrupted / "checkpoint-2"
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                     "scheduler.pt", "rng_state.pth", "trainer_state.json", "tokenizer.json", "run.json"):
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
        self.assertIsNone(self.tokenizer_loader(continuous / "final").chat_template)
        for source in (continuous / "final", interrupted / "checkpoint-18"):
            self.assertFalse((source / "chat_template.jinja").exists())
            metadata = json.loads((source / "run.json").read_text())
            self.assertEqual(metadata["text_format"], data.TEXT_FORMAT)
            self.assertNotIn("template", metadata)
        self.assertFalse((continuous / "final/model.safetensors").exists())
        state = json.loads((interrupted / "trainer_state.json").read_text())
        self.assertEqual(state["global_step"], 18)
        training_args = torch.load(interrupted / "checkpoint-18/training_args.bin", weights_only=False)
        self.assertFalse(training_args.bf16 or training_args.fp16 or training_args.gradient_checkpointing)
        self.assertEqual(training_args.gradient_accumulation_steps, 1)
        self.assertEqual(training_args.report_to, [])
        marker = continuous / "keep.txt"
        marker.write_text("keep")
        rerun = self.run_training("continuous")
        self.assertEqual(rerun, continuous.with_name("continuous-1"))
        self.assertEqual(marker.read_text(), "keep")

    def test_strict_numeric_scores_and_atomic_run_names(self):
        for text, expected in (("reason\n#### -1,234.00", "-1234"), ("#### 5.0", "5"),
                               ("#### +0.005", "0.005")):
            self.assertEqual(evaluation.answer_number(text), evaluation.Decimal(expected))
        for text in ("5", "#### 1,23", "#### NaN", "#### Infinity", "#### 5 units",
                     "#### 1 #### 5", "#### 1e3", "####"):
            self.assertIsNone(evaluation.answer_number(text), text)
        scores = evaluation.answer_metrics([(True, True, False), (False, True, True), (False, False, False)])
        self.assertEqual(scores["answer_accuracy"], 1 / 3)
        self.assertEqual(scores["answer_format_rate"], 2 / 3)
        self.assertEqual(scores["generation_truncation_rate"], 1 / 3)
        target = self.root / "same"
        target.write_text("occupied")
        (self.root / "same-1").mkdir()
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda _: workflow.reserve_directory(target), range(8)))
        self.assertEqual(len(set(paths)), 8)
        self.assertEqual({path.name for path in paths}, {f"same-{i}" for i in range(2, 10)})
        self.assertEqual(target.read_text(), "occupied")

    def test_resume_rejects_missing_state_plan_and_data_changes_before_weights(self):
        with self.assertRaises(TrainingInterrupted):
            self.run_training("guard", trainer=InterruptedTrainer)
        checkpoint = self.root / "guard/checkpoint-2"
        for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "run.json", "trainer_state.json"):
            path = checkpoint / filename
            moved = path.with_suffix(path.suffix + ".missing")
            path.rename(moved)
            try:
                with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
                    workflow.read_source(checkpoint, self.config, resume=True)
            finally:
                moved.rename(path)
        changed = deepcopy(self.config)
        changed["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "original training configuration"):
            workflow.read_source(checkpoint, changed, resume=True)
        changed = deepcopy(self.config)
        changed["lora"]["r"] *= 2
        with self.assertRaisesRegex(ValueError, "LoRA r"):
            workflow.read_source(checkpoint, changed, resume=True)
        _, metadata, step = workflow.read_source(checkpoint, self.config, resume=True)
        self.assertEqual(step, 2)
        with self.assertRaisesRegex(ValueError, "data or validation split"):
            workflow.data_plan(Dataset.from_list(self.training_rows[1:]), metadata)
        with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("Must fail before weights.")):
            (checkpoint / "optimizer.pt").rename(checkpoint / "optimizer.missing")
            with self.assertRaises(SystemExit):
                self.run_training("guard", resume=checkpoint)

    def test_all_source_entrypoints_reject_missing_or_changed_text_format_before_weights(self):
        with self.assertRaises(TrainingInterrupted):
            self.run_training("format", trainer=InterruptedTrainer)
        checkpoint = self.root / "format/checkpoint-2"
        _, original, _ = workflow.read_source(checkpoint)
        variants = [None]
        for key in ("prompt", "answer_cleanup", "eos_token"):
            changed = deepcopy(data.TEXT_FORMAT)
            changed[key] += " changed"
            variants.append(changed)
        for text_format in variants:
            changed = deepcopy(original)
            if text_format is None:
                changed.pop("text_format")
            else:
                changed["text_format"] = text_format
            workflow.save_run(checkpoint, changed)
            with self.subTest(text_format=text_format), self.assertRaisesRegex(ValueError, "Source text format"):
                workflow.read_source(checkpoint, self.config, resume=True)
        calls = len(self.model_calls)
        with self.assertRaises(SystemExit):
            self.run_training("format", resume=checkpoint)
        with self.assertRaises(SystemExit):
            self.run_training("format", continue_from=checkpoint, additional_epochs=1)
        with self.assertRaises(SystemExit):
            self.evaluate(checkpoint)
        with self.assertRaises(SystemExit):
            self.sample("--checkpoint", str(checkpoint))
        self.assertEqual(len(self.model_calls), calls)

    def test_tensorboard_resume_purges_stale_tail_and_records_loss_only(self):
        with self.assertRaises(TrainingInterrupted):
            self.run_training("logged", trainer=InterruptedTrainer)
        run = self.root / "logged"
        # Order closed fixture files chronologically, independent of same-second writer UID suffixes.
        initial = run / "runs/events.out.tfevents.0000000001.initial"
        next((run / "runs").glob("events.out.tfevents.*")).rename(initial)
        with SummaryWriter(str(run / "runs")) as writer:
            writer.add_scalar("train/loss", 999, 777)
        next(path for path in (run / "runs").glob("events.out.tfevents.*") if path != initial).rename(
            run / "runs/events.out.tfevents.0000000002.stale"
        )
        self.run_training("logged", resume=run / "checkpoint-2")
        events = EventAccumulator(str(run / "runs")).Reload()
        for tag in ("train/loss", "train/grad_norm", "train/learning_rate", "eval/loss"):
            self.assertIn(tag, events.Tags()["scalars"], tag)
        steps = [event.step for event in events.Scalars("train/loss")]
        self.assertEqual(steps, list(range(1, 19)))
        self.assertFalse(any("answer_accuracy" in tag or "generation_" in tag for tag in events.Tags()["scalars"]))
        report = json.loads((run / "evaluations/final-n2.json").read_text())
        self.assertEqual(report["metrics"]["generation_samples"], 2)
        self.assertFalse((self.root / "logged-1").exists())

    def test_additional_stage_inherits_weights_but_resets_state_and_can_resume(self):
        original = self.run_training("stage")
        saved = load_file(original / "final/adapter_model.safetensors")
        captured = {}

        class AuditStart(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
                captured.update(step=state.global_step, optimizer_size=len(optimizer.state),
                                lr=optimizer.param_groups[0]["lr"], adapters=len(model.peft_config),
                                weights={key: value.detach().clone() for key, value in get_peft_model_state_dict(model).items()})

        class InspectStage(InterruptedTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.add_callback(AuditStart())

        with self.assertRaises(TrainingInterrupted):
            self.run_training("stage", continue_from=original / "final", additional_epochs=0.5, trainer=InspectStage)
        stage = self.root / "stage-1"
        self.assertEqual((captured["step"], captured["optimizer_size"], captured["adapters"]), (0, 0, 1))
        self.assertEqual(captured["lr"], self.config["learning_rate"])
        for name, value in saved.items():
            self.assertTrue(torch.equal(value, captured["weights"][name]), name)
        restored = self.run_training("stage", resume=stage / "checkpoint-2")
        self.assertEqual(restored, stage)
        source_meta = json.loads((original / "final/run.json").read_text())
        stage_meta = json.loads((stage / "final/run.json").read_text())
        self.assertEqual(stage_meta["data"], source_meta["data"])
        self.assertEqual(stage_meta["epochs"], 0.5)
        self.assertEqual(stage_meta["source_adapter"], str(original / "final"))
        self.assertEqual(json.loads((stage / "trainer_state.json").read_text())["global_step"], 9)
        last = json.loads((stage / "checkpoint-9/trainer_state.json").read_text())
        self.assertTrue(any(item.get("step") == 9 and "eval_loss" in item for item in last["log_history"]))
        self.assertEqual(json.loads((stage / "evaluations/final-n2.json").read_text())["metrics"]["generation_samples"], 2)
        self.assertEqual(sorted(event.step for event in EventAccumulator(str(stage / "runs")).Reload().Scalars("train/loss")),
                         list(range(1, 10)))
        with self.assertRaisesRegex(ValueError, "training plan is complete"):
            workflow.read_source(original / "checkpoint-18", self.config, resume=True)

    def test_generation_runs_once_after_final_save_and_scores_all_questions(self):
        plan = workflow.data_plan(Dataset.from_list(self.training_rows))
        expected = self.training_rows[plan["validation_indices"][0]]["answer"]

        def generate(model, tokenizer, question, budget):
            final = self.root / "scored/final"
            self.assertTrue((final / "adapter_model.safetensors").is_file())
            self.assertTrue((final / "run.json").is_file())
            self.assertEqual(json.loads((final.parent / "trainer_state.json").read_text())["global_step"], 18)
            self.assertFalse(model.training)
            self.assertEqual(budget, 4)
            return expected, 4, False

        with patch.object(evaluation, "generate_answer", side_effect=generate) as called:
            run = self.run_training("scored")
        self.assertEqual(called.call_count, 2)
        self.assertEqual(len(self.model_calls), 1)
        state = json.loads((run / "checkpoint-18/trainer_state.json").read_text())
        self.assertFalse(any("eval_answer_accuracy" in item for item in state["log_history"]))
        report = json.loads((run / "evaluations/final-n2.json").read_text())
        self.assertEqual(report["metrics"]["answer_accuracy"], 0.5)
        self.assertEqual(report["metrics"]["generation_samples"], 2)
        self.assertEqual(report["metrics"]["answer_format_rate"], 1)
        self.assertEqual(report["data"]["indices"], plan["validation_indices"])
        records = [json.loads(line) for line in (run / "evaluations/final-n2.progress.jsonl").read_text().splitlines()]
        self.assertEqual(records, report["predictions"])
        for record in records:
            row = self.training_rows[record["index"]]
            self.assertEqual(record["question"], row["question"])
            self.assertEqual(record["reference_answer"], data.clean_answer(row["answer"]))
        readable = (run / "evaluations/final-n2.md").read_text()
        self.assertEqual(readable.count("## Question "), len(records))
        self.assertIn("### Reference answer\n\n", readable)
        self.assertIn("### Model answer\n\n", readable)

    def test_evaluation_appends_each_answer_and_resumes_from_record_count(self):
        run = self.run_training("append")
        output = self.root / "append-report.json"
        progress_path = output.with_suffix(".progress.jsonl")
        readable_path = output.with_suffix(".md")
        indices = workflow.data_plan(Dataset.from_list(self.training_rows))["validation_indices"]

        def interrupt_after_first(model, tokenizer, question, budget):
            self.assertFalse(model.training)
            if question == self.training_rows[indices[1]]["question"]:
                first = [json.loads(line) for line in progress_path.read_text().splitlines()]
                self.assertEqual(len(first), 1)
                self.assertEqual(first[0]["question"], self.training_rows[indices[0]]["question"])
                self.assertFalse(first[0]["format_valid"])
                readable = readable_path.read_text()
                self.assertIn("## Question 1 (dataset index ", readable)
                self.assertNotIn("## Question 2 (dataset index ", readable)
                self.assertIn("    First line.\n    No final answer marker.", readable)
                raise KeyboardInterrupt
            return "First line.\nNo final answer marker.", 4, False

        with patch.object(evaluation, "generate_answer", side_effect=interrupt_after_first):
            with self.assertRaises(KeyboardInterrupt):
                self.evaluate(run / "final", output=output)
        self.assertFalse(output.exists())
        prefix = progress_path.read_bytes()
        readable_path.unlink()
        remaining = self.training_rows[indices[1]]
        with patch.object(evaluation, "generate_answer", return_value=(data.clean_answer(remaining["answer"]), 4, False)) as called:
            report = self.evaluate(run / "final", output=output)
        called.assert_called_once()
        self.assertEqual(called.call_args.args[2], remaining["question"])
        self.assertIn("Generation progress: 1/2 completed", self.last_evaluation_terminal)
        self.assertTrue(progress_path.read_bytes().startswith(prefix))
        records = [json.loads(line) for line in progress_path.read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records, report["predictions"])
        self.assertEqual([row["index"] for row in records], indices)
        self.assertEqual(report["metrics"]["answer_accuracy"], 0.5)
        self.assertEqual(report["metrics"]["answer_format_rate"], 0.5)
        self.assertEqual(report["metrics"]["generation_runtime"], sum(row["generation_runtime"] for row in records))
        self.assertEqual(report["metrics"]["generation_samples_per_second"],
                         2 / max(report["metrics"]["generation_runtime"], 1e-9))
        readable = readable_path.read_text()
        self.assertEqual(readable.count("## Question "), len(records))
        self.assertIn("    First line.\n    No final answer marker.", readable)
        self.assertIn("    #### ", readable)
        self.assertIn("Correct: False | Format valid: False | Truncated: False", readable)
        self.assertIn(f"Readable answers: {readable_path}", self.last_evaluation_terminal)
        for record in records:
            row = self.training_rows[record["index"]]
            self.assertEqual(record["question"], row["question"])
            self.assertEqual(record["reference_answer"], data.clean_answer(row["answer"]))
            self.assertNotIn("<<", record["reference_answer"])
        saved = progress_path.read_bytes()
        with patch.object(evaluation, "generate_answer", side_effect=AssertionError("Completed answers must be reused.")):
            repeated = self.evaluate(run / "final", output=output)
        self.assertEqual(repeated, report)
        self.assertEqual(progress_path.read_bytes(), saved)
        self.assertEqual(readable_path.read_text(), readable)
        self.assertIn("Generation progress: 2/2 completed", self.last_evaluation_terminal)

    def test_completed_records_survive_summary_failure_and_include_all_score_outcomes(self):
        self.training_rows *= 2
        run = self.run_training("summary-failure")
        output = self.root / "failed-summary.json"
        progress_path = output.with_suffix(".progress.jsonl")
        indices = workflow.data_plan(Dataset.from_list(self.training_rows))["validation_indices"]
        responses = [
            (data.clean_answer(self.training_rows[indices[0]]["answer"]), 4, False),
            ("#### -999999", 4, False),
            ("missing marker", 4, False),
            (data.clean_answer(self.training_rows[indices[3]]["answer"]), 4, True),
        ]
        write_text = Path.write_text

        def fail_summary(path, *args, **kwargs):
            if path == output:
                raise OSError("Simulated summary write failure.")
            return write_text(path, *args, **kwargs)

        with patch.object(evaluation, "generate_answer", side_effect=responses) as called, \
                patch.object(Path, "write_text", new=fail_summary):
            with self.assertRaises(SystemExit):
                self.evaluate(run / "final", output=output)
        self.assertEqual(called.call_count, 4)
        records = [json.loads(line) for line in progress_path.read_text().splitlines()]
        self.assertEqual(len(records), 4)
        saved = progress_path.read_bytes()
        with patch.object(evaluation, "generate_answer", side_effect=AssertionError("Completed answers must be reused.")):
            report = self.evaluate(run / "final", output=output)
        self.assertEqual(progress_path.read_bytes(), saved)
        self.assertEqual(report["predictions"], records)
        self.assertEqual(report["metrics"]["answer_accuracy"], 0.25)
        self.assertEqual(report["metrics"]["answer_format_rate"], 0.75)
        self.assertEqual(report["metrics"]["generation_truncation_rate"], 0.25)
        self.assertEqual(report["metrics"]["generation_samples"], 4)

    def test_original_base_and_zero_adapter_evaluation_match_without_training_run(self):
        before = {p.name: p.read_bytes() for p in self.base.iterdir() if p.is_file()}
        self.assertFalse(list(self.root.rglob("run.json")))
        with patch.object(evaluation, "read_source", side_effect=AssertionError("Base has no training run.")), \
                patch.object(PeftModel, "from_pretrained", side_effect=AssertionError("Base must not load an adapter.")):
            original = self.evaluate(model_name=self.model_name)
        self.assertEqual(original["source_type"], "base")
        self.assertIsNone(original["checkpoint"])
        self.assertEqual(original["model_name"], self.model_name)
        self.assertEqual(self.model_calls[-1]["dtype"], torch.float32)
        self.assertTrue(self.model_calls[-1]["local_files_only"])
        self.assertTrue(self.model_calls[-1]["adapter_kwargs"]["local_files_only"])
        adapter = self.root / "zero-adapter"
        base = self.load_model(self.model_name, dtype=torch.float32)
        get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                                      **self.config["lora"])).save_pretrained(adapter)
        self.tokenizer.save_pretrained(adapter)
        workflow.save_run(adapter, workflow.new_run(
            self.config, workflow.data_plan(Dataset.from_list(self.training_rows)), 1))
        tuned = self.evaluate(adapter)
        self.assertEqual(tuned["source_type"], "lora")
        self.assertEqual(tuned["checkpoint"], str(adapter))
        for key in ("model_name", "data", "text_format", "generation", "scoring"):
            self.assertEqual(original[key], tuned[key])
        for key in ("answer_accuracy", "answer_format_rate", "generation_truncation_rate", "generation_samples"):
            self.assertEqual(original["metrics"][key], tuned["metrics"][key])
        for first, second in zip(original["predictions"], tuned["predictions"], strict=True):
            self.assertEqual({k: v for k, v in first.items() if k != "generation_runtime"},
                             {k: v for k, v in second.items() if k != "generation_runtime"})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.base.iterdir() if p.is_file()})

    def test_original_base_evaluation_appends_and_resumes_all_report_formats(self):
        output = self.root / "custom-base-evaluation.json"
        progress_path = output.with_suffix(".progress.jsonl")
        readable_path = output.with_suffix(".md")
        indices = workflow.data_plan(Dataset.from_list(self.training_rows))["validation_indices"]
        first, second = [self.training_rows[index] for index in indices]

        def interrupt(model, tokenizer, question, budget):
            if question == second["question"]:
                records = [json.loads(line) for line in progress_path.read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["question"], first["question"])
                self.assertIn("### Model answer", readable_path.read_text())
                raise KeyboardInterrupt
            return data.clean_answer(first["answer"]), 4, False

        with patch.object(evaluation, "generate_answer", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.evaluate(model_name=self.model_name, output=output)
        self.assertFalse(output.exists())
        prefix = progress_path.read_bytes()
        with patch.object(evaluation, "generate_answer", return_value=(data.clean_answer(second["answer"]), 4, True)) as called:
            report = self.evaluate(model_name=self.model_name, output=output)
        called.assert_called_once()
        self.assertEqual(called.call_args.args[2], second["question"])
        self.assertTrue(progress_path.read_bytes().startswith(prefix))
        records = [json.loads(line) for line in progress_path.read_text().splitlines()]
        self.assertEqual(report["predictions"], records)
        self.assertEqual(report["metrics"]["answer_accuracy"], 0.5)
        self.assertEqual(report["metrics"]["answer_format_rate"], 1)
        self.assertEqual(report["metrics"]["generation_truncation_rate"], 0.5)
        self.assertEqual(readable_path.read_text().count("## Question "), 2)
        self.assertIn(f"Source: `{self.model_name}`", readable_path.read_text())
        with patch.object(evaluation, "generate_answer", side_effect=AssertionError("Completed Base answers must be reused.")):
            repeated = self.evaluate(model_name=self.model_name, output=output)
        self.assertEqual(repeated, report)
        self.assertEqual(readable_path.read_text().count("## Question "), 2)

    def test_original_base_evaluation_default_output_and_source_validation(self):
        for sources in ({}, {"checkpoint": self.base, "model_name": self.model_name}):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                evaluation.evaluate_checkpoint(**sources)
        for sources in ({}, {"checkpoint": self.base, "model_name": self.model_name},
                        {"model_name": "Qwen/Qwen3-4B"}, {"model_name": "Qwen3-4B-Base"}):
            with self.assertRaises(SystemExit):
                self.evaluate(**sources)
        for count in (0, -1, 3):
            with self.assertRaises(SystemExit):
                self.evaluate(model_name=self.model_name, num_samples=count)
        self.assertEqual(self.model_calls, [])
        dataset = Dataset.from_list(self.training_rows)
        model = self.model_loader(self.base, dtype=torch.float32, local_files_only=True)
        with patch.object(evaluation, "__file__", str(self.root / "evaluate_checkpoint.py")), \
                contextlib.redirect_stdout(io.StringIO()):
            report = evaluation.evaluate_checkpoint(model_name=self.model_name, num_samples=1, model=model,
                                                    tokenizer=self.tokenizer, dataset=dataset, disable_tqdm=True)
        output = self.root / "outputs/evaluations/base-Qwen3-0.6B-Base-n1.json"
        self.assertEqual(json.loads(output.read_text()), report)
        records = [json.loads(line) for line in output.with_suffix(".progress.jsonl").read_text().splitlines()]
        self.assertEqual(records, report["predictions"])
        self.assertIn("Source type: base", output.with_suffix(".md").read_text())
        self.assertEqual(report["data"]["indices"], workflow.data_plan(dataset)["validation_indices"][:1])

    def test_manual_evaluation_selects_fixed_questions_without_changing_checkpoint(self):
        run = self.run_training("manual")
        checkpoint = run / "checkpoint-16"
        weights = (checkpoint / "adapter_model.safetensors").read_bytes()
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            (checkpoint / name).unlink()
        before = {path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()}
        subset = self.evaluate(checkpoint, 1)
        full = self.evaluate(run / "final")
        self.assertEqual(subset["metrics"]["generation_samples"], 1)
        self.assertEqual(full["metrics"]["generation_samples"], 2)
        self.assertEqual(full["text_format"], data.TEXT_FORMAT)
        self.assertEqual(full["generation"]["eos_token_id"], self.tokenizer.eos_token_id)
        self.assertEqual(subset["data"]["indices"], full["data"]["indices"][:1])
        self.assertEqual(before, {path.name: path.read_bytes() for path in checkpoint.iterdir() if path.is_file()})
        self.assertEqual(weights, (checkpoint / "adapter_model.safetensors").read_bytes())
        self.assertTrue(self.model_calls[-1]["local_files_only"])
        self.assertTrue(self.model_calls[-1]["adapter_kwargs"]["local_files_only"])
        answers = [self.training_rows[index]["answer"] for index in full["data"]["indices"]]
        with patch.object(evaluation, "generate_answer", side_effect=[(answers[0], 4, False), (answers[1], 4, True)]):
            scored = self.evaluate(run / "final")
        self.assertEqual(scored["metrics"]["answer_accuracy"], 0.5)
        self.assertEqual(scored["metrics"]["answer_format_rate"], 1)
        self.assertEqual(scored["metrics"]["generation_truncation_rate"], 0.5)
        with patch.object(evaluation, "generate_answer", return_value=("missing final marker", 4, False)):
            scored = self.evaluate(run / "final")
        self.assertEqual(scored["metrics"]["answer_accuracy"], 0)
        self.assertEqual(scored["metrics"]["answer_format_rate"], 0)
        with patch.object(evaluation, "generate_answer", side_effect=[
                (f"2+3=6\n#### {evaluation.answer_number(answer)}", 4, False) for answer in answers]):
            scored = self.evaluate(run / "final")
        self.assertEqual(scored["metrics"]["answer_accuracy"], 1)
        calls = len(self.model_calls)
        for count in (0, -1, 3):
            with self.assertRaises(SystemExit):
                self.evaluate(checkpoint, count)
        with self.assertRaises(SystemExit):
            self.evaluate(checkpoint, rows=self.training_rows[1:])
        self.assertEqual(len(self.model_calls), calls)

    def test_interruption_skips_final_evaluation_and_failed_evaluation_can_be_retried(self):
        with patch.object(train_sft, "evaluate_checkpoint") as called:
            with self.assertRaises(TrainingInterrupted):
                self.run_training("pause", trainer=InterruptedTrainer)
            called.assert_not_called()
        with patch.object(train_sft, "evaluate_checkpoint", side_effect=RuntimeError("Simulated evaluation failure.")):
            with self.assertRaisesRegex(RuntimeError, "evaluation failure"):
                self.run_training("failed-evaluation")
        run = self.root / "failed-evaluation"
        saved = (run / "final/adapter_model.safetensors").read_bytes()
        state = (run / "trainer_state.json").read_bytes()
        self.assertEqual(json.loads(state)["global_step"], 18)
        self.assertEqual(self.evaluate(run / "final")["metrics"]["generation_samples"], 2)
        self.assertEqual(saved, (run / "final/adapter_model.safetensors").read_bytes())
        self.assertEqual(state, (run / "trainer_state.json").read_bytes())

    def test_english_terminal_stages_and_readable_logs_preserve_numeric_records(self):
        self.config["disable_tqdm"] = False
        run = self.run_training("terminal")
        for label in ("Training", "Validation loss", "Generate validation answers"):
            self.assertIn(label, self.last_progress)
        lines = [line for line in self.last_terminal.splitlines()
                 if line.startswith(("Training |", "Validation |", "Generation evaluation |"))]
        self.assertTrue(lines)
        self.assertTrue(any("loss=" in line and "grad_norm=" in line for line in lines))
        self.assertTrue(all("{" not in line and "}" not in line and "'" not in line for line in lines))
        history = json.loads((run / "trainer_state.json").read_text())["log_history"]
        losses = [(row["step"], row["loss"]) for row in history if "loss" in row]
        self.assertTrue(all(isinstance(value, float) for _, value in losses))
        events = EventAccumulator(str(run / "runs")).Reload().Scalars("train/loss")
        for event, (step, value) in zip(events, losses):
            self.assertEqual(event.step, step)
            self.assertAlmostEqual(event.value, value, places=5)

    def test_training_source_options_fail_before_data_loading(self):
        for options in (("--continue-from", "/missing"), ("--additional-epochs", "1"),
                        ("--continue-from", "/missing", "--additional-epochs", "0"),
                        ("--continue-from", "/missing", "--additional-epochs", "nan"),
                        ("--continue-from", "/missing", "--resume-from-checkpoint", "/missing")):
            with patch.object(sys, "argv", ["train_sft.py", *options]), \
                    patch.object(train_sft, "load_dataset", side_effect=AssertionError("Must reject arguments first.")), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_sft.main()

    def test_original_and_zero_adapter_sample_same_answers_without_reference_input(self):
        base = self.load_model(self.model_name, dtype=torch.float32)
        adapter = self.root / "adapter"
        get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                                      **self.config["lora"])).save_pretrained(adapter)
        self.tokenizer.save_pretrained(adapter)
        workflow.save_run(adapter, workflow.new_run(self.config, workflow.data_plan(Dataset.from_list(self.training_rows)), 1))
        original = self.sample("--model-name", self.model_name, "--local-files-only")
        tuned = self.sample("--checkpoint", str(adapter))
        self.assertEqual(original[original.index("## 样本"):], tuned[tuned.index("## 样本"):])
        self.assertIn(data.clean_answer(self.rows[0]["answer"]), original)
        self.assertNotIn("<<", original)
        self.assertTrue(all(call["local_files_only"] for call in self.model_calls[1:]))
        self.assertTrue(all(call["adapter_kwargs"]["local_files_only"] for call in self.model_calls[1:]))
        model = self.model_loader(self.base, local_files_only=True)
        generate = model.generate
        with patch.object(model, "generate", wraps=generate) as called:
            sample_checkpoint.generate_answer(model, self.tokenizer, self.rows[0]["question"], 4)
        prompt_ids = called.call_args.kwargs["input_ids"][0].tolist()
        example = data.build_sft_example(self.tokenizer, **self.rows[0])
        prefix_length = next(index for index, value in enumerate(example["labels"]) if value != -100)
        self.assertEqual(prompt_ids, example["input_ids"][:prefix_length])
        self.assertEqual(prompt_ids, self.tokenizer.encode(data.build_prompt(self.rows[0]["question"]),
                                                        add_special_tokens=False))
        self.assertTrue(self.tokenizer.decode(prompt_ids).endswith("Answer:"))

    def test_real_generation_stops_at_native_eos_and_reports_truncation(self):
        model = self.model_loader(self.base, local_files_only=True)
        generate = model.generate
        with patch.object(model, "generate", side_effect=lambda **kwargs: generate(
                **kwargs, logits_processor=[ForceToken(self.tokenizer.eos_token_id)])):
            for budget in (1, 4):
                answer, count, truncated = sample_checkpoint.generate_answer(model, self.tokenizer, "question one", budget)
                self.assertEqual((answer, count, truncated), ("", 1, False))
        with patch.object(model, "generate", side_effect=lambda **kwargs: generate(
                **kwargs, logits_processor=[ForceToken(self.tokenizer.convert_tokens_to_ids("one"))])):
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
