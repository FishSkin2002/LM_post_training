"""Choose a data worker and reuse its existing Arrow caches without copying datasets."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from datasets import Dataset, concatenate_datasets
from datasets.table import list_table_cache_files

from utils.prepare_sft_data import text_format
from utils.prepared_sft_data import (data_environment, data_settings, load_prepared_sft_data,
                                     resolve_prepared_sft_data, tokenizer_backend)


def recipe(stage, config, tokenizer, held_out=None, length_limit=None):
    settings = data_settings(config) if stage == "sft" else {
        "source": config, "length_limit": length_limit,
        "held_out": sorted({" ".join(question.split()) for question in held_out}),
    }
    value = {"stage": stage, "settings": settings, "text_format": text_format(tokenizer),
             "tokenizer": tokenizer_backend(tokenizer)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cache_reference(dataset, identity):
    files = list_table_cache_files(dataset._data)
    indices = list_table_cache_files(dataset._indices) if dataset._indices is not None else []
    if not files or (dataset._indices is not None and not indices):
        return None
    return {"version": 1, "recipe": identity,
            "data_files": [str(Path(path).resolve()) for path in files],
            "indices_files": [str(Path(path).resolve()) for path in indices],
            "columns": dataset.column_names, "fingerprint": dataset._fingerprint, "rows": len(dataset)}


def load_reference(reference, identity):
    if reference.get("version") != 1 or reference["recipe"] != identity:
        raise ValueError("Prepared data cache uses different data/tokenizer/length settings.")
    def read_files(paths):
        parts = [Dataset.from_file(path) for path in paths]
        return parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    data = read_files(reference["data_files"])
    if reference["indices_files"]:
        indices = read_files(reference["indices_files"])
        data = Dataset(data._data, info=data.info, indices_table=indices._data)
    data = data.select_columns(reference["columns"])
    if len(data) != reference["rows"]:
        raise ValueError("Prepared data cache has a different number of rows.")
    data._fingerprint = reference["fingerprint"]
    return data


def prepare_in_legacy(stage, config, tokenizer, *, held_out=None, length_limit=None):
    from legacy.sft_data.migration import LEGACY, WORKER, ensure_compatibility_environment, run_child
    profile = json.loads((LEGACY / "profile.json").read_text())
    try:
        python = ensure_compatibility_environment(profile)
        with tempfile.TemporaryDirectory(prefix="qwen-data-task-") as directory:
            directory = Path(directory)
            tokenizer.save_pretrained(directory / "tokenizer")
            request = {"stage": stage, "config": config, "tokenizer": str(directory / "tokenizer"),
                       "held_out": held_out, "length_limit": length_limit}
            (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False))
            print(f"Preparing {stage.upper()} data in the legacy CPU environment", flush=True)
            run_child([python, WORKER, "--request", directory / "request.json", "--result", directory / "result.json"])
            result = json.loads((directory / "result.json").read_text())
    except subprocess.SubprocessError as error:
        raise ValueError(f"Legacy {stage.upper()} data preparation failed: {error}") from error

    identity = recipe(stage, config, tokenizer, held_out, length_limit)
    return load_reference(result["cache"], identity), result["info"], result["environment"]


def prepare_sft(config, tokenizer, *, source=None, metadata=None, env="current", prepared=None, prepare):
    if env not in ("current", "legacy"):
        raise ValueError("Data preparation env must be current or legacy.")
    identity = recipe("sft", config, tokenizer)
    prepared = resolve_prepared_sft_data(source, prepared, metadata.get("prepared_sft_data") if metadata else None)
    if prepared is not None:
        dataset, info = load_prepared_sft_data(prepared, config, tokenizer)
        environment = metadata.get("data_environment") if metadata else None
        if environment is None and metadata:
            from legacy.sft_data.migration import LEGACY
            environment = json.loads((LEGACY / "profile.json").read_text())["data_environment"]
        environment = environment or data_environment()
    elif metadata and metadata.get("sft_data_cache"):
        dataset = load_reference(metadata["sft_data_cache"], identity)
        info, environment = metadata["data"]["source"], metadata["data_environment"]
    elif env == "legacy":
        dataset, info, environment = prepare_in_legacy("sft", config, tokenizer)
    else:
        dataset, info = prepare(config, tokenizer)
        environment = data_environment()
    return dataset, info, environment, cache_reference(dataset, identity), prepared


def prepare_stage(stage, config, tokenizer, held_out, length_limit, *, env="current", metadata=None, prepare):
    if env not in ("current", "legacy"):
        raise ValueError("Data preparation env must be current or legacy.")
    identity = recipe(stage, config, tokenizer, held_out, length_limit)
    if metadata and metadata.get("training_data_cache"):
        dataset = load_reference(metadata["training_data_cache"], identity)
        info, environment = metadata["data"], metadata["data_environment"]
    elif env == "legacy":
        dataset, info, environment = prepare_in_legacy(stage, config, tokenizer,
                                                      held_out=held_out, length_limit=length_limit)
    else:
        dataset, info = prepare(stage, config, tokenizer, held_out, length_limit)
        environment = data_environment()
    return dataset, info, environment, cache_reference(dataset, identity)
