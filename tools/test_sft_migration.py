"""Migration failure boundaries and automatic cache reuse; no package downloads."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peft import LoraConfig, get_peft_model

import evaluate as evaluation
from legacy.sft_data import migration
import utils.training.sft as train_sft
import train as training_cli
import utils.training.dpo as train_dpo
import utils.training.grpo as train_grpo
from utils.prepared_sft_data import (load_prepared_sft_data, ensure_prepared_sft_data,
                               resolve_prepared_sft_data, save_prepared_sft_data)
from utils.training_workflow import data_plan, new_run, save_run
from tools import test_instruct_sft as fixtures


class MigrationTests(unittest.TestCase):
    def setUp(self):
        fixture = self.fixture = fixtures.InstructSFTTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.source = self.root / "sft" / "final"
        model = fixture.model_loader(fixture.model_name)
        get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                                       **fixture.config["lora"])).save_pretrained(self.source)
        fixture.tokenizer.save_pretrained(self.source)
        self.dataset, self.info = fixture.prepared()
        self.metadata = new_run(fixture.config, data_plan(self.dataset, data_info=self.info), 1)
        self.metadata.pop("data_environment")  # An old artifact without runtime metadata.
        save_run(self.source, self.metadata)
        self.profile = json.loads((migration.LEGACY / "profile.json").read_text())
        self.target = self.source.parent / "prepared-sft"

    def export(self, command):
        target = Path(command[command.index("--output") + 1])
        save_prepared_sft_data(target, self.dataset, self.info, self.fixture.config, self.fixture.tokenizer)

    def assert_held_out_snapshot(self, report):
        expected = self.dataset.select(self.metadata["data"]["validation_indices"])
        actual = [(row["source_index"], row["generation_index"], row["question"],
                   row["reference_answer"], row["reference_final_answer"]) for row in report["predictions"]]
        self.assertEqual(actual, [(row["source_index"], row["generation_index"], row["question"],
                                   row["answer"], row["final_answer"]) for row in expected])

    def test_publish_then_reuse_without_conda_or_worker(self):
        original = {p.name: p.read_bytes() for p in self.source.iterdir() if p.is_file()}
        with patch.object(migration, "ensure_compatibility_environment", return_value=Path("/managed/python")) as ensure, \
                patch.object(migration, "run_child", side_effect=self.export) as run:
            self.assertEqual(migration.migrate_sft(self.source), self.target)
            ensure.assert_called_once()
            run.assert_called_once()
        with patch.object(migration, "find_conda", side_effect=AssertionError("no Conda needed")), \
                patch.object(migration, "ensure_compatibility_environment", side_effect=AssertionError("no interpreter needed")), \
                patch.object(migration, "run_child", side_effect=AssertionError("no child needed")):
            self.assertEqual(migration.migrate_sft(self.source), self.target)
        restored, info = load_prepared_sft_data(self.target, self.fixture.config, self.fixture.tokenizer)
        self.assertEqual(restored.to_list(), self.dataset.to_list())
        self.assertEqual(restored._fingerprint, self.dataset._fingerprint)
        self.assertEqual(info, self.info)
        self.assertEqual(data_plan(restored, self.metadata, info), self.metadata["data"])
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.source.iterdir() if p.is_file()})
        with self.assertRaises(ValueError):
            load_prepared_sft_data(self.target, {**self.fixture.config, "max_length": 1}, self.fixture.tokenizer)
        manifest_path = self.target / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["fingerprint"] = "changed"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity"):
            load_prepared_sft_data(self.target, self.fixture.config, self.fixture.tokenizer)

    def test_ordinary_evaluation_and_original_control_discover_cache(self):
        with patch.object(migration, "ensure_compatibility_environment", return_value=Path("/managed/python")), \
                patch.object(migration, "run_child", side_effect=self.export), \
                patch.object(evaluation, "prepare_dataset", side_effect=AssertionError("must use cache")), \
                patch.object(evaluation, "generate_answers", return_value=[("", 1, False)] * 2):
            for kwargs in ({"checkpoint": self.source},
                           {"model_name": self.fixture.model_name, "sft_source": self.source}):
                report = evaluation.evaluate_checkpoint(**kwargs, model=self.fixture.model_loader(self.fixture.model_name),
                    device="cpu", generation_backend="transformers", disable_tqdm=True,
                    output=self.root / ("original.json" if "model_name" in kwargs else "sft.json"))
                self.assertEqual(report["prepared_sft_data"], str(self.target))
                self.assert_held_out_snapshot(report)

    def test_missing_legacy_cache_automatically_prepares(self):
        with patch.object(migration, "migrate_sft", return_value=self.target) as prepare:
            self.assertEqual(ensure_prepared_sft_data(self.source, self.metadata), self.target)
        prepare.assert_called_once_with(self.source)
        self.assertIsNone(resolve_prepared_sft_data(self.source))
        override = self.root / "external"
        with patch.object(migration, "migrate_sft", side_effect=AssertionError("explicit overrides must be respected")):
            self.assertEqual(ensure_prepared_sft_data(self.source, self.metadata, override), override)

    def test_current_data_environment_and_new_sft_do_not_create_legacy_environment(self):
        current = new_run(self.fixture.config, self.metadata["data"], 1)
        with patch.object(migration, "find_conda", side_effect=AssertionError("no Conda for normal SFT")):
            self.assertIsNone(ensure_prepared_sft_data(None, None))
            self.assertIsNone(ensure_prepared_sft_data(self.source, current))

    def test_automatic_preparation_reports_captured_failure_and_keeps_source(self):
        original = (self.source / "run.json").read_bytes()
        error = subprocess.CalledProcessError(1, "conda", stderr="Conda could not read its configuration")
        with patch.object(migration, "migrate_sft", side_effect=error), \
                self.assertRaisesRegex(ValueError, "Conda could not read its configuration"):
            ensure_prepared_sft_data(self.source, self.metadata)
        self.assertEqual((self.source / "run.json").read_bytes(), original)

    def test_post_sft_current_preparation_never_creates_legacy_environment(self):
        class PreparedHeldOut(Exception):
            pass
        expected = self.dataset.select(self.metadata["data"]["validation_indices"])["question"]
        for stage in ("dpo", "grpo"):
            training = {"dpo": train_dpo, "grpo": train_grpo}[stage]
            source = self.root / stage / "source" / "final"
            shutil.copytree(self.source, source)
            with self.subTest(stage=stage), \
                    patch.object(migration, "ensure_compatibility_environment", side_effect=AssertionError("current must not create legacy")), \
                    patch.object(training, "load_config", return_value=({"trainer": {}, "data": {}, "output_dir": "unused"}, 2048)), \
                    patch.object(training, "training_arguments", return_value=SimpleNamespace(use_vllm=False)), \
                    patch.object(training, "prepare_dataset", return_value=(self.dataset, self.info)), \
                    patch.object(training, "prepare_post_training_data", side_effect=PreparedHeldOut) as prepare:
                with self.assertRaises(PreparedHeldOut):
                    training.train( "unused", sft_source=source)
                self.assertEqual(prepare.call_args.args[3], expected)
                self.assertFalse((source.parent / "prepared-sft").exists())

    def test_sft_continue_uses_current_preparation_before_loading_weights(self):
        class PreparedData(Exception):
            pass
        config = self.root / "sft.json"
        config.write_text(json.dumps(self.fixture.config))
        output = self.root / "continued"
        argv = ["train.py", "sft", "--config", str(config), "--source", str(self.source),
                "--additional-epochs", "1", "--output-dir", str(output)]
        with patch.object(migration, "ensure_compatibility_environment", side_effect=AssertionError("current must not create legacy")), \
                patch.object(train_sft, "prepare_dataset", return_value=(self.dataset, self.info)), \
                patch.object(train_sft, "load_training_model", side_effect=PreparedData), patch.object(sys, "argv", argv):
            with self.assertRaises(PreparedData):
                training_cli.main()
        from utils.prepared_sft_data import data_environment
        self.assertEqual(json.loads((output / "run.json").read_text())["data_environment"], data_environment())
        self.assertFalse(self.target.exists())

    def test_checkpoint_evaluation_prepares_without_requiring_final(self):
        self.source = self.source.rename(self.source.with_name("checkpoint-7"))
        with patch.object(migration, "ensure_compatibility_environment", return_value=Path("/managed/python")), \
                patch.object(migration, "run_child", side_effect=self.export), \
                patch.object(evaluation, "prepare_dataset", side_effect=AssertionError("must use automatic cache")), \
                patch.object(evaluation, "generate_answers", return_value=[("", 1, False)] * 2):
            report = evaluation.evaluate_checkpoint(self.source, model=self.fixture.model_loader(self.fixture.model_name),
                device="cpu", generation_backend="transformers", disable_tqdm=True, output=self.root / "checkpoint.json")
        self.assertEqual(report["prepared_sft_data"], str(self.target))

    def test_post_sft_training_and_evaluation_trace_original_sft_cache(self):
        self.export(["--output", self.target])
        expected = self.dataset.select(self.metadata["data"]["validation_indices"])["question"]
        class PreparedHeldOut(Exception):
            pass
        for stage in ("dpo", "grpo"):
            training = {"dpo": train_dpo, "grpo": train_grpo}[stage]
            with self.subTest(stage=stage):
                with patch.object(training, "load_config", return_value=({"trainer": {}, "data": {}, "output_dir": "unused"}, 2048)), \
                        patch.object(training, "training_arguments", return_value=SimpleNamespace(use_vllm=False)), \
                        patch.object(training, "prepare_dataset", side_effect=AssertionError("must use cache")), \
                        patch.object(training, "prepare_post_training_data", side_effect=PreparedHeldOut) as prepare:
                    with self.assertRaises(PreparedHeldOut):
                        training.train( "unused", sft_source=self.source)
                self.assertEqual(prepare.call_args.args[3], expected)
                post = self.root / stage / "final"
                shutil.copytree(self.source, post)
                metadata = {"stage": stage, "config": {}, "source_adapter": str(self.source),
                            "evaluation": self.metadata, "scoring": self.metadata["scoring"]}
                save_run(post, metadata)
                with patch.object(evaluation, "prepare_dataset", side_effect=AssertionError("must use cache")), \
                        patch.object(evaluation, "generate_answers", return_value=[("", 1, False)] * 2):
                    report = evaluation.evaluate_checkpoint(post, model=self.fixture.model_loader(self.fixture.model_name),
                        device="cpu", generation_backend="transformers", disable_tqdm=True, output=self.root / f"{stage}.json")
                self.assertEqual(report["prepared_sft_data"], str(self.target))
                self.assert_held_out_snapshot(report)

    def test_worker_failure_does_not_publish_and_can_retry(self):
        with patch.object(migration, "ensure_compatibility_environment", return_value=Path("/managed/python")), \
                patch.object(migration, "run_child", side_effect=subprocess.CalledProcessError(1, "worker")):
            with self.assertRaises(subprocess.CalledProcessError):
                migration.migrate_sft(self.source)
        self.assertFalse(self.target.exists())
        with patch.object(migration, "ensure_compatibility_environment", return_value=Path("/managed/python")), \
                patch.object(migration, "run_child", side_effect=self.export):
            self.assertEqual(migration.migrate_sft(self.source), self.target)

    def test_wrong_source_data_is_not_published(self):
        self.metadata["data"]["validation_indices"].reverse()
        save_run(self.source, self.metadata)
        with patch.object(migration, "ensure_compatibility_environment", return_value=Path("/managed/python")), \
                patch.object(migration, "run_child", side_effect=self.export), \
                self.assertRaisesRegex(ValueError, "split differs"):
            migration.migrate_sft(self.source)
        self.assertFalse(self.target.exists())

    def test_incomplete_cache_is_rejected_without_old_interpreter(self):
        self.target.mkdir()
        with patch.object(migration, "ensure_compatibility_environment", side_effect=AssertionError("must reject cache")), \
                self.assertRaises(OSError):
            migration.migrate_sft(self.source)

    def test_unsupported_profile_fails_before_environment_creation(self):
        self.metadata["data_environment"] = {"datasets": "unknown"}
        save_run(self.source, self.metadata)
        with patch.object(migration, "ensure_compatibility_environment") as ensure, \
                self.assertRaisesRegex(ValueError, "No supported legacy"):
            migration.migrate_sft(self.source)
        ensure.assert_not_called()

    def test_concurrent_migrations_publish_once(self):
        calls = Mock(side_effect=self.export)
        barrier = threading.Barrier(2)
        def migrate():
            barrier.wait(timeout=10)
            return migration.migrate_sft(self.source)
        with patch.object(migration, "ensure_compatibility_environment", return_value=Path("/managed/python")), \
                patch.object(migration, "run_child", calls), ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(migrate) for _ in range(2)]
            self.assertEqual([f.result(timeout=20) for f in futures], [self.target] * 2)
        calls.assert_called_once()

    def fake_conda(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.state_root = self.root / "state-cache"
        self.envs = self.root / "configured-conda-envs"
        info = {"root_prefix": str(self.root / "conda-root"), "envs_dirs": [str(self.envs)]}
        stack.enter_context(patch.object(migration, "managed_root", return_value=self.state_root))
        stack.enter_context(patch.object(migration, "find_conda", return_value="/conda"))
        stack.enter_context(patch.object(migration, "conda_info", return_value=info))
        def python(conda, name, prefix):
            executable = prefix / "bin" / "python"
            if not executable.is_file():
                raise ValueError("Conda creation was interrupted")
            return executable
        stack.enter_context(patch.object(migration, "environment_python", side_effect=python))
        return self.envs / migration.environment_name(self.profile)

    def create_environment(self, command):
        if "create" in command:
            self.assertNotIn("--prefix", command)
            name = command[command.index("-n") + 1]
            python = self.envs / name / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()

    def test_missing_conda_does_not_create_environment(self):
        with patch.object(migration.shutil, "which", return_value=None), \
                patch.dict(migration.os.environ, {}, clear=True), \
                self.assertRaisesRegex(ValueError, "Conda was not found"):
            migration.ensure_compatibility_environment(self.profile)

    def test_installation_failure_is_not_ready_and_retry_reuses_python(self):
        prefix = self.fake_conda()
        def execute(command):
            self.create_environment(command)
            if "install" in command:
                raise subprocess.CalledProcessError(1, "pip")
        with patch.object(migration, "run_child", side_effect=execute), self.assertRaises(subprocess.CalledProcessError):
            migration.ensure_compatibility_environment(self.profile)
        self.assertFalse((prefix / "sft-data-ready.json").exists())
        with patch.object(migration, "run_child") as run:
            self.assertEqual(migration.ensure_compatibility_environment(self.profile), prefix / "bin" / "python")
        self.assertFalse(any("create" in c.args[0] for c in run.call_args_list))
        self.assertTrue((prefix / "sft-data-ready.json").is_file())

    def test_interrupted_conda_creation_has_clear_recovery(self):
        prefix = self.fake_conda()
        def incomplete(command):
            prefix.mkdir(parents=True)
            raise subprocess.CalledProcessError(1, "conda")
        with patch.object(migration, "run_child", side_effect=incomplete), self.assertRaises(subprocess.CalledProcessError):
            migration.ensure_compatibility_environment(self.profile)
        with patch.object(migration, "run_child") as run, self.assertRaisesRegex(ValueError, "Repair or remove"):
            migration.ensure_compatibility_environment(self.profile)
        run.assert_not_called()

    def test_user_environment_is_not_modified(self):
        prefix = self.fake_conda()
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin" / "python").touch()
        with patch.object(migration, "run_child") as run, self.assertRaisesRegex(ValueError, "unmanaged existing Conda"):
            migration.ensure_compatibility_environment(self.profile)
        run.assert_not_called()

    def test_concurrent_environment_setup_installs_once(self):
        self.fake_conda()
        barrier = threading.Barrier(2)
        def setup():
            barrier.wait(timeout=10)
            return migration.ensure_compatibility_environment(self.profile)
        with patch.object(migration, "run_child", side_effect=self.create_environment) as run, ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(setup) for _ in range(2)]
            self.assertEqual(futures[0].result(timeout=20), futures[1].result(timeout=20))
        self.assertEqual(sum("create" in call.args[0] for call in run.call_args_list), 1)
        self.assertEqual(sum("install" in call.args[0] for call in run.call_args_list), 1)


if __name__ == "__main__":
    unittest.main()
