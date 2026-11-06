"""Real CPU migration acceptance using synthetic data and a random tiny SFT artifact.

Create a fixture with --make-fixture DIR under the historic data stack, then run
--fixture DIR --compat-root NEW_DIR from the main environment. The worker's raw
dataset loader is the only synthetic seam; Conda, pip, export and checks are real.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_fixture(root):
    from peft import LoraConfig, get_peft_model
    from tools import test_instruct_sft as fixtures
    from training_workflow import data_plan, new_run, save_run
    fixture = fixtures.InstructSFTTests()
    fixture.setUp()
    try:
        root.mkdir(parents=True, exist_ok=False)
        source = root / "sft" / "final"
        model = fixture.model_loader(fixture.model_name)
        get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                                       **fixture.config["lora"])).save_pretrained(source)
        fixture.tokenizer.save_pretrained(source)
        shutil.copytree(fixture.base, root / "base")
        dataset, info = fixture.prepared()
        plan = data_plan(dataset, data_info=info)
        metadata = new_run(fixture.config, plan, 1)
        metadata.pop("data_environment")  # Original artifacts predate the runtime marker.
        save_run(source, metadata)
        baseline = {"rows": fixture.raw, "prepared": dataset.to_list(), "plan": plan,
                    "vocab": fixture.tokenizer.get_vocab()}
        (root / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False), encoding="utf-8")
    finally:
        fixture.doCleanups()


def smoke(root, compat_root, conda_exe):
    import migrate_sft as migration
    from prepare_sft_data import load_tokenizer
    from prepared_sft_data import load_prepared_sft_data, resolve_prepared_sft_data
    from training_workflow import data_plan, read_source
    if compat_root.exists():
        raise ValueError("Acceptance requires a new, absent compatibility root.")
    source = root / "sft" / "final"
    original = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    baseline = json.loads((root / "baseline.json").read_text())
    # No public test switches: only replace the raw Hugging Face fetch in a private worker.
    with tempfile.TemporaryDirectory(prefix="migration-worker-") as directory:
        worker = Path(directory) / "synthetic_worker.py"
        worker.write_text(
            f"import sys, json\nsys.path.insert(0, {str(ROOT)!r})\n"
            "from datasets import Dataset\nimport math_data\n"
            f"rows = json.load(open({str(root / 'baseline.json')!r}))['rows']\n"
            "math_data.load_dataset = lambda *a, **k: Dataset.from_list(rows)\n"
            "from legacy.sft_data.worker import main\nmain()\n", encoding="utf-8")
        with patch.object(migration, "WORKER", worker):
            target = migration.migrate_sft(source, compat_root=compat_root, conda_exe=conda_exe)
    _, metadata, _ = read_source(source)
    tokenizer = load_tokenizer(source, local_files_only=True)
    dataset, info = load_prepared_sft_data(target, metadata["config"], tokenizer)
    assert dataset.to_list() == baseline["prepared"]
    assert data_plan(dataset, metadata, info) == baseline["plan"]
    assert tokenizer.get_vocab() == baseline["vocab"]
    assert original == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    with patch.object(migration, "run_child", side_effect=AssertionError("cache reuse invoked legacy Python")), \
            patch.object(migration, "find_conda", side_effect=AssertionError("cache reuse needs Conda")), \
            patch.object(migration, "ensure_compatibility_environment", side_effect=AssertionError("cache reuse needs environment")):
        assert migration.migrate_sft(source, compat_root=compat_root) == target
    assert resolve_prepared_sft_data(source) == target
    manifest = json.loads((target / "manifest.json").read_text())
    result = {"cpu_runtime": manifest["legacy_environment"], "fingerprint": dataset._fingerprint,
              "validation_indices": baseline["plan"]["validation_indices"],
              "rows_and_tokens_unchanged": True, "artifact_unchanged": True,
              "repeat_without_legacy_python": True, "prepared_sft_data": str(target)}
    (root / "migration-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-fixture", type=Path)
    mode.add_argument("--fixture", type=Path)
    parser.add_argument("--compat-root", type=Path)
    parser.add_argument("--conda-exe")
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture(args.make_fixture.resolve())
    else:
        if not args.compat_root:
            parser.error("--compat-root must point to a fresh location for acceptance.")
        smoke(args.fixture.resolve(), args.compat_root.resolve(), args.conda_exe)


if __name__ == "__main__":
    main()
