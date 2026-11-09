"""Offline public-interface and cross-environment cache checks; no Conda installs."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from peft import LoraConfig, get_peft_model
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate as evaluation
import train
from legacy.sft_data import migration, worker
from tools import test_instruct_sft as fixtures
from utils import data_preparation as preparation, math_data, post_training_data
from utils.prepare_sft_data import CONFIG_DIR, MODEL_NAME
from utils.training_workflow import data_plan, new_run, save_run


class ScriptInterfaceTests(unittest.TestCase):
    def test_training_dispatches_source_resume_and_environment(self):
        for stage in ("sft", "dpo", "grpo"):
            module = __import__(f"utils.training.{stage}", fromlist=["train"])
            for mode in ("source", "resume"):
                argv = [stage, f"--{mode}", "/tmp/source", "--env", "legacy",
                        "--generation-backend", "transformers"]
                if stage == "sft" and mode == "source":
                    argv += ["--additional-epochs", "0.5"]
                with self.subTest(stage=stage, mode=mode), \
                        patch.object(module, "train", return_value=Path("/tmp/final")) as run, \
                        patch.object(train, "exec_final_evaluation"):
                    train.main(argv)
                    options = run.call_args.kwargs
                    self.assertEqual(options["env"], "legacy")
                    self.assertNotIn("prepared_sft_data", options)
                    key = "resume" if mode == "resume" else "continue_from" if stage == "sft" else "sft_source"
                    self.assertEqual(options[key], Path("/tmp/source"))
        with patch("utils.training.sft.train", return_value=Path("/tmp/final")) as run:
            train.main(["sft", "--generation-backend", "transformers"])
            self.assertEqual(run.call_args.kwargs["env"], "current")

    def test_old_options_and_mixed_training_sources_are_rejected(self):
        for args in (["sft", "--resume-from-checkpoint", "x"], ["sft", "--continue-from", "x"],
                     ["dpo", "--sft-source", "x"], ["sft", "--prepared-sft-data", "x"],
                     ["sft", "--source", "x", "--resume", "y"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.main(args)

    def test_evaluation_accepts_one_checkpoint_and_only_original_accepts_source(self):
        with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", MODEL_NAME, "--source", "/tmp/sft/final"]), \
                patch.object(evaluation, "evaluate_checkpoint") as run:
            evaluation.main()
        self.assertEqual(run.call_args.args, (MODEL_NAME,))
        self.assertEqual(run.call_args.kwargs["sft_source"], Path("/tmp/sft/final"))
        with self.assertRaisesRegex(ValueError, "original Qwen"):
            evaluation.evaluate_checkpoint("/tmp/trained/final", sft_source=Path("/tmp/other/final"))
        for option in ("--model-name", "--prepared-sft-data"):
            with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", MODEL_NAME, option, "x"]), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "Incomplete"):
            evaluation.evaluate_checkpoint(Path(directory) / "missing")
        with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", "/tmp/trained/final", "--config", "x"]), \
                patch.object(evaluation, "evaluate_checkpoint") as run, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluation.main()
        run.assert_not_called()


class DataEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InstructSFTTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.tokenizer = self.fixture.root, self.fixture.tokenizer
        self.config = self.fixture.config

    def disk_dataset(self, name, dataset):
        path = self.root / f"{name}.arrow"
        writer = ArrowWriter(path=str(path), features=dataset.features)
        writer.write_table(dataset.data.table)
        writer.finalize()
        return Dataset.from_file(str(path))

    def test_current_never_starts_a_legacy_worker_for_any_stage(self):
        dataset, info = self.fixture.prepared()
        with patch.object(preparation, "prepare_in_legacy", side_effect=AssertionError("no legacy process")), \
                patch.object(migration, "ensure_compatibility_environment", side_effect=AssertionError("no Conda")):
            prepare = Mock(return_value=(dataset, info))
            result = preparation.prepare_sft(self.config, self.tokenizer, prepare=prepare)
            self.assertIs(result[0], dataset)
            prepare.assert_called_once_with(self.config, self.tokenizer)
            for stage in ("dpo", "grpo"):
                prepare = Mock(return_value=(dataset, info))
                result = preparation.prepare_stage(stage, {}, self.tokenizer, [], 128, prepare=prepare)
                self.assertIs(result[0], dataset)
                prepare.assert_called_once_with(stage, {}, self.tokenizer, [], 128)

    def test_native_arrow_reference_restores_filter_indices_and_projection_without_export(self):
        data = self.disk_dataset("native", self.fixture.prepared()[0])
        data = data.map(lambda row: {"extra": 1}).filter(lambda row: row["source_index"] % 2 == 0)
        data = data.remove_columns("extra")
        identity = preparation.recipe("sft", self.config, self.tokenizer)
        with patch.object(Dataset, "save_to_disk", side_effect=AssertionError("no duplicate dataset")):
            reference = preparation.cache_reference(data, identity)
            restored = preparation.load_reference(reference, identity)
        self.assertTrue(reference["indices_files"])
        self.assertEqual(restored.to_dict(), data.to_dict())
        self.assertEqual(restored._fingerprint, data._fingerprint)
        with self.assertRaisesRegex(ValueError, "different"):
            preparation.load_reference(reference, "changed-settings")
        in_memory_indices = data.select(list(reversed(range(len(data)))), keep_in_memory=True)
        self.assertIsNone(preparation.cache_reference(in_memory_indices, identity))

    def test_legacy_worker_failure_does_not_fall_back_to_current_preparation(self):
        prepare = Mock(side_effect=AssertionError("no implicit environment fallback"))
        with patch.object(migration, "ensure_compatibility_environment", return_value=Path(sys.executable)), \
                patch.object(migration, "run_child", side_effect=subprocess.CalledProcessError(1, "worker")), \
                self.assertRaisesRegex(ValueError, "Legacy SFT data preparation failed"):
            preparation.prepare_sft(self.config, self.tokenizer, env="legacy", prepare=prepare)
        prepare.assert_not_called()

    def simulated_child(self, command):
        self.assertIn("--request", command)
        request = Path(command[command.index("--request") + 1])
        result = Path(command[command.index("--result") + 1])
        with patch.object(worker, "check_environment"):
            worker.prepare_request(request, result)

    def test_legacy_fresh_sft_uses_native_cache_and_saved_reference_reuses_without_worker(self):
        raw = self.disk_dataset("openr1", Dataset.from_list(self.fixture.raw))
        legacy_environment = json.loads((migration.LEGACY / "profile.json").read_text())["data_environment"]
        with patch.object(migration, "ensure_compatibility_environment", return_value=Path(sys.executable)) as ensure, \
                patch.object(migration, "run_child", side_effect=self.simulated_child), \
                patch.object(math_data, "load_dataset", return_value=raw), \
                patch("utils.prepared_sft_data.data_environment", return_value=legacy_environment), \
                patch.object(Dataset, "save_to_disk", side_effect=AssertionError("no duplicate dataset")):
            data, info, environment, cache, prepared = preparation.prepare_sft(
                self.config, self.tokenizer, env="legacy", prepare=Mock(side_effect=AssertionError("not in main")))
        ensure.assert_called_once()
        self.assertIsNone(prepared)
        self.assertIsNotNone(cache)
        self.assertEqual(environment, legacy_environment)
        metadata = {"sft_data_cache": cache, "data_environment": environment, "data": {"source": info}}
        with patch.object(migration, "ensure_compatibility_environment", side_effect=AssertionError("no Conda for cache")), \
                patch.object(migration, "run_child", side_effect=AssertionError("no worker for cache")):
            restored = preparation.prepare_sft(self.config, self.tokenizer, metadata=metadata, env="current",
                                               prepare=Mock(side_effect=AssertionError("no repeat preprocessing")))
        self.assertEqual(restored[0].to_dict(), data.to_dict())
        self.assertEqual(restored[2], environment)

    def test_all_checkpoint_stages_read_their_recorded_native_data_without_preparing(self):
        dataset, info = self.fixture.prepared()
        dataset = self.disk_dataset("prepared-sft", dataset)
        reference = preparation.cache_reference(dataset, preparation.recipe("sft", self.config, self.tokenizer))
        source = self.root / "source" / "final"
        model = self.fixture.model_loader(MODEL_NAME)
        get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                                       **self.config["lora"])).save_pretrained(source)
        self.tokenizer.save_pretrained(source)
        metadata = new_run(self.config, data_plan(dataset, data_info=info), 1)
        metadata["sft_data_cache"] = reference
        save_run(source, metadata)
        expected = dataset.select(metadata["data"]["validation_indices"][:1])["question"]
        for stage in ("sft", "dpo", "grpo"):
            target = self.root / stage / "final"
            shutil.copytree(source, target)
            if stage != "sft":
                save_run(target, {"stage": stage, "config": {}, "evaluation": metadata,
                                  "source_adapter": str(source), "scoring": metadata["scoring"]})
            with self.subTest(stage=stage), \
                    patch.object(evaluation, "prepare_dataset", side_effect=AssertionError("no repeat preparation")), \
                    patch.object(evaluation, "ensure_prepared_sft_data", side_effect=AssertionError("use native cache")), \
                    patch.object(evaluation, "generate_answers", return_value=[(fixtures.trace(2), 1, False)]):
                report = evaluation.evaluate_checkpoint(target, model=model, tokenizer=self.tokenizer, num_samples=1,
                    output=self.root / f"eval-{stage}", device="cpu", generation_backend="transformers", disable_tqdm=True)
            self.assertEqual([row["question"] for row in report["predictions"]], expected)

    def test_legacy_dpo_and_grpo_workers_prepare_their_own_data(self):
        rows = {"dpo": [{"prompt": "question 1", "chosen": r"\boxed{2}", "rejected": r"\boxed{3}"}],
                "grpo": [{"prompt": [{"role": "user", "content": "question 1"}], "ability": "math",
                          "reward_model": {"ground_truth": '["2"]', "style": "rule"}}]}
        for stage in ("dpo", "grpo"):
            config = yaml.safe_load((CONFIG_DIR / f"training/{stage}.yaml").read_text())["data"]
            raw = self.disk_dataset(stage, Dataset.from_list(rows[stage]))
            with self.subTest(stage=stage), \
                    patch.object(migration, "ensure_compatibility_environment", return_value=Path(sys.executable)), \
                    patch.object(migration, "run_child", side_effect=self.simulated_child), \
                    patch.object(post_training_data, "load_dataset", return_value=raw), \
                    patch.object(Dataset, "save_to_disk", side_effect=AssertionError("no duplicate dataset")):
                data, info, environment, cache = preparation.prepare_stage(
                    stage, config, self.tokenizer, [], 256, env="legacy",
                    prepare=Mock(side_effect=AssertionError("not in main")))
            self.assertEqual(len(data), 1)
            self.assertIsNotNone(cache)
            self.assertNotIn("filter_reason", data.column_names)
            self.assertIn("fingerprint", info)


if __name__ == "__main__":
    unittest.main()
