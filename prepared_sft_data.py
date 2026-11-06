"""Transfer verified prepared SFT data between environments without rebuilding its identity."""

import json
from importlib.metadata import version
from pathlib import Path
import subprocess

from datasets import load_from_disk

from prepare_sft_data import text_format


def data_environment():
    return {name: version(name) for name in ("datasets", "transformers", "tokenizers", "dill", "pyarrow", "xxhash")}


def resolve_prepared_sft_data(source, explicit=None, saved=None):
    """Resolve a cache without invoking Conda, subprocesses or package installation."""
    if explicit or saved:
        return Path(explicit or saved).expanduser().resolve()
    if source is not None:
        candidate = Path(source).expanduser().resolve().parent / "prepared-sft"
        if candidate.exists():
            return candidate
    return None


def ensure_prepared_sft_data(source, metadata, explicit=None, saved=None):
    """Reuse existing data, or prepare a known legacy source before continuing."""
    prepared = resolve_prepared_sft_data(source, explicit, saved)
    if metadata is None or prepared is not None:
        return prepared
    recorded = metadata.get("data_environment")
    # Unmarked artifacts were produced by the original datasets 5.1.0 stack.
    compatible = recorded == data_environment() if recorded else version("datasets") == "5.1.0"
    if compatible:
        return None
    from legacy.sft_data.migration import migrate_sft
    print(f"Preparing legacy SFT data automatically before continuing: {source}", flush=True)
    try:
        return migrate_sft(source)
    except subprocess.SubprocessError as error:
        detail = (getattr(error, "stderr", None) or str(error)).strip()[:2000]
        raise ValueError(f"Automatic legacy SFT data preparation failed: {detail}. Resolve the error and rerun "
                         "the same training/evaluation command; the original SFT artifact was preserved.") from error


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
                        help="Override prepared SFT data; default: reuse or automatically prepare it from the source.")
