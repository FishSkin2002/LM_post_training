"""Private CPU data worker, launched only by the main-environment migration entry."""

import argparse
from importlib.metadata import distributions, version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def check_environment():
    profile = json.loads(Path(__file__).with_name("profile.json").read_text(encoding="utf-8"))
    if f"{sys.version_info.major}.{sys.version_info.minor}" != profile["python"]:
        raise ValueError("Legacy data preparation requires Python 3.12.")
    original = {}
    for line in Path(__file__).with_name("requirements-original.txt").read_text().splitlines():
        name, pinned = line.split("==")
        original[name.split("[")[0]] = pinned
    expected = {**original, **profile["data_environment"], **profile["packages"]}
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise ValueError(f"Legacy data dependencies do not match the pinned profile: {actual}")
    import torch
    if torch.version.cuda is not None:
        raise ValueError("Legacy data preparation requires the managed CPU-only PyTorch build.")
    installed = {package.metadata["Name"]: package.version for package in distributions()}
    return {"profile": profile["id"], "python": sys.version, "packages": installed, "cuda_runtime": None}


def export_data(source, output):
    runtime = check_environment()
    from legacy.sft_data.migration import compatibility_profile
    from prepare_sft_data import load_tokenizer, prepare_dataset
    from prepared_sft_data import save_prepared_sft_data
    from training_workflow import data_plan, read_source
    source, metadata, _ = read_source(source)
    if metadata.get("stage", "sft") != "sft":
        raise ValueError("Legacy export requires an SFT artifact.")
    compatibility_profile(metadata)
    tokenizer = load_tokenizer(source, local_files_only=True)
    dataset, info = prepare_dataset(metadata["config"], tokenizer)
    data_plan(dataset, metadata, info)
    path = save_prepared_sft_data(output, dataset, info, metadata["config"], tokenizer)
    manifest = path / "manifest.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    saved.update(legacy_environment=runtime, source_plan=metadata["data"], source_artifact=str(source))
    manifest.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--sft-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.check_environment:
            runtime = check_environment()
            print(f"CPU compatibility environment ready: {runtime['profile']}; "
                  f"torch={runtime['packages']['torch']}; cuda_runtime=None", flush=True)
        elif args.sft_source and args.output:
            print(f"Exported verified SFT data: {export_data(args.sft_source, args.output)}", flush=True)
        else:
            parser.error("Provide --check-environment or both --sft-source and --output.")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
