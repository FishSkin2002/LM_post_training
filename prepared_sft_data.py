"""Transfer verified prepared SFT data between environments without rebuilding its identity."""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk

from prepare_sft_data import load_tokenizer, prepare_dataset, text_format


def tokenizer_backend(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    return backend


def data_settings(config):
    return {key: config[key] for key in ("model_name", "data", "max_length")}


def save_prepared_sft_data(path, dataset, info, config, tokenizer):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    dataset.save_to_disk(path / "dataset")
    manifest = {"version": 1, "settings": data_settings(config), "text_format": text_format(tokenizer),
                "tokenizer": tokenizer_backend(tokenizer), "fingerprint": dataset._fingerprint, "data_info": info}
    (path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_prepared_sft_data(path, config, tokenizer):
    path = Path(path).expanduser().resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest["settings"] != data_settings(config):
        raise ValueError("Prepared SFT data uses different model/data/length settings.")
    if manifest["text_format"] != text_format(tokenizer) or manifest["tokenizer"] != tokenizer_backend(tokenizer):
        raise ValueError("Prepared SFT data uses a different tokenizer or chat protocol.")
    state = json.loads((path / "dataset" / "state.json").read_text(encoding="utf-8"))
    if state["_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Prepared SFT data identity differs from its exported manifest.")
    dataset = load_from_disk(path / "dataset")
    if len(dataset) != manifest["data_info"]["kept_records"]:
        raise ValueError("Prepared SFT data has a different number of records.")
    # load_from_disk reapplies formatting and rehashes with the new library.
    # Restore the identity already persisted by the verified original-environment export.
    dataset._fingerprint = state["_fingerprint"]
    return dataset, manifest["data_info"]


def add_prepared_data_argument(parser):
    parser.add_argument("--prepared-sft-data", type=Path,
                        help="Prepared SFT data exported in the original environment; preserves old data fingerprints.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-source", type=Path, required=True, help="Existing SFT final/ artifact.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the transferable data cache.")
    args = parser.parse_args()
    from training_workflow import data_plan, read_source
    try:
        source, metadata, _ = read_source(args.sft_source)
        if metadata.get("stage", "sft") != "sft" or source.name != "final":
            raise ValueError("Export requires the original SFT final/ artifact.")
        config = metadata["config"]
        tokenizer = load_tokenizer(source, local_files_only=True)
        dataset, info = prepare_dataset(config, tokenizer)
        data_plan(dataset, metadata, info)
        target = save_prepared_sft_data(args.output, dataset, info, config, tokenizer)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Exported verified SFT data: {target}", flush=True)


if __name__ == "__main__":
    main()
