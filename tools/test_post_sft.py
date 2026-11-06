"""Offline DPO/GRPO checks using the project's tiny Qwen and native template."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import yaml
from datasets import Dataset
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, TrainerCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import test_instruct_sft as fixtures
import utils.evaluation as evaluation
import utils.math_data as math_data
import utils.post_training_data as data
import utils.training.post_sft as training
import train as training_cli
from utils.prepare_sft_data import CONFIG_DIR, SYSTEM_PROMPT, build_prompt


def skywork(question, references):
    return {"prompt": [{"role": "user", "content": question}], "ability": "math",
            "reward_model": {"ground_truth": json.dumps(references), "style": "rule"}}


class StopAtCheckpoint(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.global_step == 1:
            raise fixtures.Interrupted("simulated interruption")


class FixedAnswers(LogitsProcessor):
    """Make the real GRPO generator return two distinct, verifiable answers per group."""

    def __init__(self, tokenizer):
        self.answer_ids = [tokenizer.convert_tokens_to_ids(r"\boxed{2}"),
                           tokenizer.convert_tokens_to_ids(r"\boxed{3}")]
        self.eos = tokenizer.eos_token_id
        self.first = True

    def __call__(self, input_ids, scores):
        scores.fill_(-torch.inf)
        for index in range(len(scores)):
            scores[index, self.answer_ids[index % 2] if self.first else self.eos] = 0
        self.first = False
        return scores


class PostSFTTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InstructSFTTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.tokenizer = self.fixture.tokenizer

    def test_dataset_filtering_preserves_roles_and_preference_tokens(self):
        cases = [
            {"prompt": "question 2", "chosen": "\nCompute\n# Answer\n$2$",
             "rejected": "\nCompute\n# Answer\n3\n\n# Answer\n0.009259259..."},
            {"prompt": " question\t8\n", "chosen": "2", "rejected": "3"},
            {"prompt": "question 1", "chosen": "Compute " * 200 + "\n# Answer\n2", "rejected": r"\boxed{3}"},
            {"prompt": "question 3", "chosen": "# Answer\nC.\n\n119", "rejected": r"\boxed{3}"},
            {"prompt": "question 4", "chosen": r"\boxed{2}", "rejected": "# Answer\n" + r"\frac{3}{"},
            {"prompt": "question 5", "chosen": "# Answer\n100 cm$^2$", "rejected": r"\boxed{3}"},
            {"prompt": "question 6", "chosen": "# Answer\n0.009259259...", "rejected": r"\boxed{3}"},
            {"prompt": "question 7", "chosen": "# Answer\n$$\n2\n$$", "rejected": r"\boxed{3}"},
            {"prompt": "question 9", "chosen": fixtures.trace(2), "rejected": fixtures.trace(3)},
        ]
        source = yaml.safe_load((CONFIG_DIR / "training/dpo.yaml").read_text())["data"]
        with patch.object(data, "load_dataset", return_value=Dataset.from_list(cases)) as loader:
            prepared, info = data.prepare_post_training_data("dpo", source, self.tokenizer, ["question 8"], 128)
        loader.assert_called_once_with(source["name"], "solution", split="train", revision=source["revision"])
        self.assertEqual(info["answer_format"], data.DPO_ANSWER_FORMAT)
        self.assertEqual(info["system"], SYSTEM_PROMPT)
        self.assertEqual(info["filter_counts"], {"held_out_overlap": 1, "over_length": 1,
                                                 "invalid_chosen_format": 3, "unsupported_chosen_answer": 1})
        self.assertEqual(list(prepared["source_index"]), [0, 4, 8])
        row = prepared[0]
        self.assertEqual(row["prompt_ids"], self.tokenizer.encode(build_prompt(self.tokenizer, cases[0]["prompt"]),
                                                                 add_special_tokens=False))
        expected = {"chosen": "\nCompute\n" + r"\boxed{2}",
                    "rejected": "\nCompute\n# Answer\n3\n\n" + r"\boxed{0.009259259...}"}
        for side in ("chosen", "rejected"):
            self.assertEqual(row[side + "_ids"], self.tokenizer.encode(expected[side], add_special_tokens=False)
                             + [self.tokenizer.eos_token_id])
            self.assertEqual(prepared[2][side + "_ids"], self.tokenizer.encode(cases[-1][side], add_special_tokens=False)
                             + [self.tokenizer.eos_token_id])
        self.assertEqual(prepared[1]["rejected_ids"], self.tokenizer.encode(r"\boxed{\frac{3}{}", add_special_tokens=False)
                         + [self.tokenizer.eos_token_id])

        cases = [skywork(" question 2 ", ["1/4", "proof", "3"]), skywork("question\n8", ["2"]),
                 skywork("question 5", ["proof"]), skywork("question " * 200, ["2"])]
        source = yaml.safe_load((CONFIG_DIR / "training/grpo.yaml").read_text())["data"]
        with patch.object(data, "load_dataset", return_value=Dataset.from_list(cases)) as loader:
            prepared, info = data.prepare_post_training_data("grpo", source, self.tokenizer, ["question  8"], 128)
        loader.assert_called_once_with(source["name"], "default", split="math", revision=source["revision"])
        self.assertEqual(info["filter_counts"], {"held_out_overlap": 1, "over_length": 1, "unsupported_reference": 1})
        row = prepared[0]
        self.assertEqual(row["references"], ["1/4", "3"])
        self.assertEqual(self.tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True),
                         build_prompt(self.tokenizer, " question 2 "))

    def test_rejected_answer_preserves_content_and_removes_only_terminal_empty_headings(self):
        prefix = "Reasoning.\n\n"
        cases = {
            "# Answer\n\nC.\n\n119\n": r"\boxed{C." + "\n\n119}",
            "# Answer\n $21$  ": r"\boxed{$21$}",
            "# Answer\n\n": "",
            "# Answe \n\n": "",
            "# Answer\n3\n\n# Answe\n": "# Answer\n3\n\n",
            "An unmarked answer.\n": "An unmarked answer.\n",
            "# Another heading\n": "# Another heading\n",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(data._format_rejected_answer(prefix + original), prefix + expected)

    def test_boxed_reward_uses_alternatives_and_native_eos(self):
        texts = [r"\boxed{0.25}", r"\boxed{3}", r"\boxed{0.25}", r"\boxed{0.25}", "0.25",
                 r"\boxed{3} but \boxed{4}", r"\boxed{3} but \boxed{"]
        eos, pad = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        rewards = data.boxed_reward(
            [[{"role": "assistant", "content": text}] for text in texts],
            [[eos], [eos], [pad], [5], [eos], [eos], [eos]], [["1/4", "3"]] * len(texts), eos_token_id=eos,
        )
        self.assertEqual(rewards, [1, 1, 0, 0, 0, 0, 0])

    def test_stage_entrypoints_resume_and_share_saved_sft_evaluation(self):
        self.fixture.config.update(max_steps=2, eval_steps=1, save_steps=1)
        sft = self.fixture.run_training("sft") / "final"
        sft_metadata = json.loads((sft / "run.json").read_text())
        original_weights = load_file(sft / "adapter_model.safetensors")
        sft_report = json.loads(next((sft.parent / "evaluations").glob("*.json")).read_text())
        held_out = [row["question"] for row in sft_report["predictions"]]
        changed_whitespace = " \n ".join(held_out[0].split())

        for stage, base_trainer in (("dpo", training.PreferenceTrainer), ("grpo", training.MathGRPOTrainer)):
            with self.subTest(stage=stage):
                self.check_stage(stage, base_trainer, sft, sft_metadata, original_weights, sft_report, changed_whitespace)

        output = self.root / "original.json"
        with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=self.fixture.model_loader), \
                patch.object(AutoTokenizer, "from_pretrained", side_effect=self.fixture.load_tokenizer), \
                patch.object(math_data, "load_dataset", side_effect=self.fixture.raw_dataset), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            report = evaluation.evaluate_checkpoint(model_name=self.fixture.model_name, sft_source=sft,
                                                    output=output, device="cpu", disable_tqdm=True)
        self.assertEqual(report["stage"], "original")
        self.assertEqual(report["data"], sft_report["data"])
        self.assertEqual(report["generation"], sft_report["generation"])

    def check_stage(self, stage, base_trainer, sft, sft_metadata, original_weights, sft_report, overlap):
        trainers = []

        class ObserveTrainer(base_trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                trainers.append(self)

        class InterruptTrainer(ObserveTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.add_callback(StopAtCheckpoint())

        def load_model(*args, **kwargs):
            model = self.fixture.model_loader(*args, **kwargs)
            generate = model.generate

            def fixed_generate(*args, **kwargs):
                if "generation_config" in kwargs:
                    eos = kwargs["generation_config"].eos_token_id
                    self.assertIn(eos, (self.tokenizer.eos_token_id, [self.tokenizer.eos_token_id]))
                    self.assertTrue(kwargs["generation_config"].use_cache)
                    kwargs["logits_processor"] = [FixedAnswers(self.tokenizer)]
                return generate(*args, **kwargs)

            model.generate = fixed_generate
            return model

        cfg = yaml.safe_load((CONFIG_DIR / f"training/{stage}.yaml").read_text())
        cfg["trainer"].update(max_steps=2, use_cpu=True, save_steps=1, disable_tqdm=True, dataloader_pin_memory=False)
        path = self.root / f"{stage}.yaml"
        if stage == "dpo":
            cfg["max_length"] = 128
            rows = [{"prompt": question, "chosen": fixtures.trace(2), "rejected": fixtures.trace(3)}
                    for question in (overlap, "new question 1", "new question 2", "new question 3")]
        else:
            cfg["max_prompt_length"] = 128
            cfg["trainer"].update(use_vllm=False, max_completion_length=4, num_generations=2, gradient_accumulation_steps=2)
            rows = [skywork(question, ["2"]) for question in (overlap, "new question 1", "new question 2", "new question 3")]
        path.write_text(yaml.safe_dump(cfg))
        trainer_name = "PreferenceTrainer" if stage == "dpo" else "MathGRPOTrainer"

        with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=load_model) as model_loader, \
                patch.object(AutoTokenizer, "from_pretrained", side_effect=self.fixture.load_tokenizer), \
                patch.object(math_data, "load_dataset", side_effect=self.fixture.raw_dataset), \
                patch.object(data, "load_dataset", return_value=Dataset.from_list(rows)), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            continuous = self.root / f"{stage}-full" / "final"
            training_cli.main([stage, "--config", str(path), "--sft-source", str(sft),
                               "--output-dir", str(continuous.parent)])
            partial = self.root / f"{stage}-resume"
            with patch.object(training, trainer_name, InterruptTrainer), self.assertRaises(fixtures.Interrupted):
                training.train(stage, path, sft_source=sft, output_dir=partial)
            self.assertFalse((partial / "final").exists())
            checkpoint = partial / "checkpoint-1"
            self.assertFalse((checkpoint / "ref").exists())
            metadata_path = checkpoint / "run.json"
            if stage == "dpo":
                original_metadata = metadata_path.read_text()
                for version in ("boxed-v1", None):
                    old = json.loads(original_metadata)
                    if version is None:
                        del old["data"]["answer_format"]
                    else:
                        old["data"]["answer_format"] = version
                    metadata_path.write_text(json.dumps(old))
                    with patch.object(training, "prepare_dataset") as prepare, \
                            self.assertRaisesRegex(ValueError, "different answer format"):
                        training.train(stage, path, resume=checkpoint)
                    prepare.assert_not_called()
                metadata_path.write_text(original_metadata)
            with patch.object(training, trainer_name, ObserveTrainer):
                resumed = training.train(stage, path, resume=checkpoint)
            calls = model_loader.call_count
            with self.assertRaisesRegex(ValueError, "finished SFT"):
                training.train(stage, path, sft_source=resumed)
            with self.assertRaisesRegex(ValueError, "already has a saved final adapter"):
                training.train(stage, path, resume=checkpoint)
            self.assertEqual(model_loader.call_count, calls)
            broken = json.loads(metadata_path.read_text())
            broken["source_adapter_sha256"] = "different"
            metadata_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "source SFT artifact has changed"):
                training.train(stage, path, resume=checkpoint)
            self.assertEqual(model_loader.call_count, calls)

        reference = get_peft_model_state_dict(trainers[-1].model, adapter_name="ref")
        self.assertEqual(trainers[-1].model.active_adapter, "default")
        self.assertFalse(trainers[-1].model.config.use_cache)
        actual = load_file(resumed / "adapter_model.safetensors")
        expected = load_file(continuous / "adapter_model.safetensors")
        self.assertTrue(all(torch.equal(reference[key], original_weights[key]) for key in reference))
        self.assertTrue(all(torch.equal(actual[key], expected[key]) for key in actual))
        self.assertTrue(any(not torch.equal(actual[key], original_weights[key]) for key in actual))
        self.assertFalse((resumed / "ref").exists())
        metadata = json.loads((resumed / "run.json").read_text())
        self.assertEqual(metadata["stage"], stage)
        self.assertEqual(metadata["evaluation"]["data"], sft_metadata["data"])
        self.assertEqual(metadata["data"]["filter_counts"], {"held_out_overlap": 1})
        report = json.loads(next((resumed.parent / "evaluations").glob("*.json")).read_text())
        self.assertEqual(report["stage"], stage)
        self.assertEqual(report["data"], sft_report["data"])
        self.assertEqual(report["generation"], sft_report["generation"])
        self.assertEqual([row["question"] for row in report["predictions"]],
                         [row["question"] for row in sft_report["predictions"]])


if __name__ == "__main__":
    unittest.main()
