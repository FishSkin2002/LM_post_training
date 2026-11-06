"""Migration failure boundaries and automatic cache reuse; no package downloads."""

from concurrent.futures import ThreadPoolExecutor
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

import evaluate_checkpoint as evaluation
import migrate_sft as migration
import train_post_sft
from prepared_sft_data import (load_prepared_sft_data, require_data_environment,
                               resolve_prepared_sft_data, save_prepared_sft_data)
from training_workflow import data_plan, new_run, save_run
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
        self.assertEqual(data_plan(restored, self.metadata, info), self.metadata["data"])
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.source.iterdir() if p.is_file()})

    def test_ordinary_evaluation_and_original_control_discover_cache(self):
        self.export(["--output", self.target])
        with patch.object(evaluation, "prepare_dataset", side_effect=AssertionError("must use cache")), \
                patch.object(evaluation, "generate_answers", return_value=[("", 1, False)] * 2):
            for kwargs in ({"checkpoint": self.source},
                           {"model_name": self.fixture.model_name, "sft_source": self.source}):
                report = evaluation.evaluate_checkpoint(**kwargs, model=self.fixture.model_loader(self.fixture.model_name),
                    device="cpu", generation_backend="transformers", disable_tqdm=True,
                    output=self.root / ("original.json" if "model_name" in kwargs else "sft.json"))
                self.assertEqual(report["prepared_sft_data"], str(self.target))
                self.assertEqual(report["data"]["indices"], self.metadata["data"]["validation_indices"])

    def test_no_cache_guides_migration_without_installing(self):
        with patch("prepared_sft_data.version", return_value="4.4.2"), \
                self.assertRaisesRegex(ValueError, "python migrate_sft.py --sft-source"):
            require_data_environment(self.metadata, self.source)
        self.assertIsNone(resolve_prepared_sft_data(self.source))
        override = self.root / "external"
        self.assertEqual(resolve_prepared_sft_data(self.source, override, "ignored"), override)

    def test_post_sft_training_and_evaluation_trace_original_sft_cache(self):
        self.export(["--output", self.target])
        expected = self.dataset.select(self.metadata["data"]["validation_indices"])["question"]
        class PreparedHeldOut(Exception):
            pass
        for stage in ("dpo", "grpo"):
            with self.subTest(stage=stage):
                with patch.object(train_post_sft, "load_stage_config", return_value=({"trainer": {}, "data": {}, "output_dir": "unused"}, 2048)), \
                        patch.object(train_post_sft, "training_arguments", return_value=SimpleNamespace(use_vllm=False)), \
                        patch.object(train_post_sft, "prepare_dataset", side_effect=AssertionError("must use cache")), \
                        patch.object(train_post_sft, "prepare_post_training_data", side_effect=PreparedHeldOut) as prepare:
                    with self.assertRaises(PreparedHeldOut):
                        train_post_sft.train(stage, "unused", sft_source=self.source)
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
                self.assertEqual(report["data"]["indices"], self.metadata["data"]["validation_indices"])

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

    def test_missing_conda_does_not_create_environment(self):
        with patch.object(migration.shutil, "which", return_value=None), \
                patch.dict(migration.os.environ, {}, clear=True), \
                self.assertRaisesRegex(ValueError, "Conda was not found"):
            migration.ensure_compatibility_environment(self.profile, self.root / "managed")
        self.assertFalse((self.root / "managed" / "environments").exists())

    def test_installation_failure_is_not_ready_and_retry_reuses_python(self):
        root = self.root / "managed"
        prefix = root / "environments" / migration.compatibility_key(self.profile)
        python = prefix / "bin" / "python"
        def execute(command):
            if "create" in command:
                python.parent.mkdir(parents=True)
                python.touch()
            elif "install" in command:
                raise subprocess.CalledProcessError(1, "pip")
        with patch.object(migration, "find_conda", return_value="/conda"), \
                patch.object(migration, "run_child", side_effect=execute), self.assertRaises(subprocess.CalledProcessError):
            migration.ensure_compatibility_environment(self.profile, root)
        self.assertFalse((prefix / "sft-data-ready.json").exists())
        with patch.object(migration, "find_conda", return_value="/conda"), patch.object(migration, "run_child") as run:
            self.assertEqual(migration.ensure_compatibility_environment(self.profile, root), python)
        self.assertFalse(any("create" in c.args[0] for c in run.call_args_list))
        self.assertTrue((prefix / "sft-data-ready.json").is_file())

    def test_interrupted_conda_prefix_has_clear_recovery(self):
        root = self.root / "managed"
        (root / "environments" / migration.compatibility_key(self.profile)).mkdir(parents=True)
        with patch.object(migration, "find_conda", return_value="/conda"), \
                self.assertRaisesRegex(ValueError, "fresh --compat-root"):
            migration.ensure_compatibility_environment(self.profile, root)

    def test_user_environment_is_not_modified(self):
        root = self.root / "managed"
        prefix = root / "environments" / migration.compatibility_key(self.profile)
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin" / "python").touch()
        with patch.object(migration, "find_conda", return_value="/conda"), patch.object(migration, "run_child") as run, \
                self.assertRaisesRegex(ValueError, "unmanaged existing prefix"):
            migration.ensure_compatibility_environment(self.profile, root)
        run.assert_not_called()

    def test_concurrent_environment_setup_installs_once(self):
        root = self.root / "managed"
        barrier = threading.Barrier(2)
        def execute(command):
            if "create" in command:
                python = Path(command[command.index("--prefix") + 1]) / "bin" / "python"
                python.parent.mkdir(parents=True)
                python.touch()
        def setup():
            barrier.wait(timeout=10)
            return migration.ensure_compatibility_environment(self.profile, root)
        with patch.object(migration, "find_conda", return_value="/conda"), \
                patch.object(migration, "run_child", side_effect=execute) as run, ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(setup) for _ in range(2)]
            self.assertEqual(futures[0].result(timeout=20), futures[1].result(timeout=20))
        self.assertEqual(sum("create" in call.args[0] for call in run.call_args_list), 1)
        self.assertEqual(sum("install" in call.args[0] for call in run.call_args_list), 1)


if __name__ == "__main__":
    unittest.main()
