"""Real CPU migration acceptance using synthetic data and a random tiny SFT artifact.

Create a fixture with --make-fixture DIR under the historic data stack, then run
--fixture DIR from the main environment. A normal evaluation command automatically
creates a named Conda environment, prepares the data and generates a tiny answer.
"""

import argparse
import json
import os
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
    from utils.training_workflow import data_plan, new_run, save_run
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


def smoke(root, conda_exe):
    import torch
    import evaluate as evaluation
    import utils.generation as generation
    from legacy.sft_data import migration
    from utils.prepare_sft_data import load_tokenizer
    from utils.prepared_sft_data import load_prepared_sft_data, resolve_prepared_sft_data
    from utils.training_workflow import data_plan, read_source
    torch.set_num_threads(1)
    source = root / "sft" / "final"
    original = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    baseline = json.loads((root / "baseline.json").read_text())
    conda = migration.find_conda(conda_exe)
    _, metadata, _ = read_source(source)
    profile = migration.compatibility_profile(metadata)
    name = migration.environment_name(profile)
    if migration.named_prefix(migration.conda_info(conda), name) is not None:
        raise ValueError("Acceptance requires an absent named compatibility environment.")
    if resolve_prepared_sft_data(source):
        raise ValueError("Acceptance requires an absent prepared data cache.")
    output = root / "evaluation.json"
    argv = ["evaluate.py", "--checkpoint", str(source), "--device", "cpu",
            "--generation-backend", "transformers", "--num-samples", "1", "--output", str(output)]
    original_loader = generation.AutoModelForCausalLM.from_pretrained
    def load_tiny(model_name, **kwargs):
        assert model_name == metadata["config"]["model_name"]
        return original_loader(root / "base", **kwargs)
    # Only raw data and model size are synthetic. The normal entry, named Conda
    # creation, dependency install, worker, checks and actual generation are real.
    with tempfile.TemporaryDirectory(prefix="migration-worker-") as directory:
        worker = Path(directory) / "synthetic_worker.py"
        worker.write_text(
            f"import sys, json\nsys.path.insert(0, {str(ROOT)!r})\n"
            "from datasets import Dataset\nfrom utils import math_data\n"
            f"rows = json.load(open({str(root / 'baseline.json')!r}))['rows']\n"
            "math_data.load_dataset = lambda *a, **k: Dataset.from_list(rows)\n"
            "from legacy.sft_data.worker import main\nmain()\n", encoding="utf-8")
        with patch.object(migration, "WORKER", worker), patch.dict(os.environ, {"CONDA_EXE": conda}), \
                patch.object(sys, "argv", argv), \
                patch.object(generation.AutoModelForCausalLM, "from_pretrained", side_effect=load_tiny):
            evaluation.main()
    target = resolve_prepared_sft_data(source)
    tokenizer = load_tokenizer(source, local_files_only=True)
    dataset, info = load_prepared_sft_data(target, metadata["config"], tokenizer)
    assert dataset.to_list() == baseline["prepared"]
    assert data_plan(dataset, metadata, info) == baseline["plan"]
    assert tokenizer.get_vocab() == baseline["vocab"]
    assert original == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    saved = {suffix: output.with_suffix(suffix).read_bytes() for suffix in (".progress.jsonl", ".md")}
    with patch.object(migration, "find_conda", side_effect=AssertionError("cache reuse needs Conda")), \
            patch.object(migration, "run_child", side_effect=AssertionError("cache reuse invoked legacy Python")), \
            patch.object(evaluation, "create_generator", side_effect=AssertionError("completed answers regenerated")), \
            patch.object(evaluation, "score_answer", side_effect=AssertionError("completed answers rescored")), \
            patch.object(sys, "argv", argv):
        evaluation.main()
    assert saved == {suffix: output.with_suffix(suffix).read_bytes() for suffix in saved}
    manifest = json.loads((target / "manifest.json").read_text())
    prefix = migration.named_prefix(migration.conda_info(conda), name)
    result = {"cpu_runtime": manifest["legacy_environment"], "fingerprint": dataset._fingerprint,
              "validation_indices": baseline["plan"]["validation_indices"],
              "rows_and_tokens_unchanged": True, "artifact_unchanged": True,
              "repeat_without_legacy_python": True, "answers_and_markdown_unchanged": True,
              "conda_name": name, "conda_prefix": str(prefix), "prepared_sft_data": str(target)}
    (root / "migration-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-fixture", type=Path)
    mode.add_argument("--fixture", type=Path)
    parser.add_argument("--conda-exe")
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture(args.make_fixture.resolve())
    else:
        smoke(args.fixture.resolve(), args.conda_exe)


if __name__ == "__main__":
    main()
