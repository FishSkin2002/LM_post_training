"""Run state, held-out validation and native Trainer integration."""

from copy import deepcopy
from decimal import Decimal
import json
import math
from pathlib import Path
import random
import re
import time

from transformers import Trainer

from prepare_sft_data import CHAT_TEMPLATE
from sample_checkpoint import generate_answer


RUN_FILE = "run.json"
GENERATION_SAMPLES = 128
GENERATION_MAX_NEW_TOKENS = 1024
NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")


def reserve_directory(path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        candidate = path if suffix == 0 else path.with_name(f"{path.name}-{suffix}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1


def answer_number(text):
    if text.count("####") != 1:
        return None
    value = text.split("####")[1].strip()
    return Decimal(value.replace(",", "")) if NUMBER.fullmatch(value) else None


def answer_metrics(results):
    count = len(results)
    return {
        "answer_accuracy": sum(correct for correct, _, _ in results) / count,
        "answer_format_rate": sum(valid for _, valid, _ in results) / count,
        "generation_truncation_rate": sum(cut for _, _, cut in results) / count,
        "generation_samples": count,
    }


def data_plan(dataset, source=None):
    if len(dataset) < 2:
        raise ValueError("At least two training examples are required for a held-out split.")
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    count = math.ceil(len(indices) * 0.1)
    plan = {
        "fingerprint": dataset._fingerprint,
        "validation_indices": indices[:count], "train_indices": indices[count:],
        "split_seed": 42,
    }
    if source is not None and plan != source["data"]:
        raise ValueError("Source data or validation split differs; keep the original dataset and split.")
    return source["data"] if source is not None else plan


def save_run(path, metadata):
    Path(path).mkdir(parents=True, exist_ok=True)
    (Path(path) / RUN_FILE).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")


def read_source(path, config, *, resume):
    path = Path(path).expanduser().resolve()
    required = [RUN_FILE, "adapter_config.json", "adapter_model.safetensors",
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    if resume:
        required += ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                     "training_args.bin"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete {'checkpoint' if resume else 'adapter source'}: missing {', '.join(missing)}")
    metadata = json.loads((path / RUN_FILE).read_text(encoding="utf-8"))
    saved = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if saved["base_model_name_or_path"] != config["model_name"]:
        raise ValueError("Source and configuration must use the same Qwen3 Base model.")
    if metadata["config"]["model_name"] != config["model_name"]:
        raise ValueError("Source run model differs from the configuration.")
    if metadata["template"] != CHAT_TEMPLATE:
        raise ValueError("Source chat template differs from the current training format.")
    for key, value in config["lora"].items():
        if saved[key] != value:
            raise ValueError(f"Source LoRA {key} differs from the configuration.")
    if metadata["generation"] != {"samples": GENERATION_SAMPLES, "max_new_tokens": GENERATION_MAX_NEW_TOKENS}:
        raise ValueError("Source generation validation settings differ from the current protocol.")
    step = None
    if resume:
        selected = {key: value for key, value in config.items() if key != "output_dir"}
        original = {key: value for key, value in metadata["config"].items() if key != "output_dir"}
        if selected != original:
            raise ValueError("Resume must keep the original training configuration; use a new stage for changes.")
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        step = state["global_step"]
        if step >= state["max_steps"]:
            raise ValueError("This training plan is complete; use --continue-from for additional training.")
    return path, metadata, step


def new_run(config, plan, epochs, source_path=None):
    return {
        "config": deepcopy(config), "data": plan, "epochs": epochs, "template": CHAT_TEMPLATE,
        "generation": {"samples": GENERATION_SAMPLES, "max_new_tokens": GENERATION_MAX_NEW_TOKENS},
        "source_adapter": str(source_path) if source_path else None,
    }


class WorkflowTrainer(Trainer):
    def __init__(self, *args, run_metadata, validation_rows, **kwargs):
        self.run_metadata = run_metadata
        self.validation_rows = validation_rows
        super().__init__(*args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        save_run(output_dir or self.args.output_dir, self.run_metadata)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix="eval"):
        was_training = self.model.training
        try:
            output = super().evaluation_loop(dataloader, description, prediction_loss_only,
                                              ignore_keys, metric_key_prefix)
            self.model.eval()
            full = self.state.max_steps > 0 and self.state.global_step >= self.state.max_steps
            subset_count = min(GENERATION_SAMPLES, len(self.validation_rows))
            rows = self.validation_rows if full else self.validation_rows.select(range(subset_count))
            started = time.perf_counter()
            results = []
            for row in rows:
                expected = answer_number(row["answer"])
                if expected is None:
                    raise ValueError("Validation reference answer has an invalid #### number.")
                text, _, truncated = generate_answer(
                    self.model, self.processing_class, row["question"], GENERATION_MAX_NEW_TOKENS,
                )
                predicted = answer_number(text)
                results.append((not truncated and predicted == expected, predicted is not None, truncated))
                if len(results) == subset_count:
                    metrics = answer_metrics(results)
                    elapsed = time.perf_counter() - started
                    metrics.update(generation_runtime=elapsed,
                                   generation_samples_per_second=len(results) / max(elapsed, 1e-9))
                    output.metrics.update({f"{metric_key_prefix}_{key}": value for key, value in metrics.items()})
            if full:
                metrics = answer_metrics(results)
                elapsed = time.perf_counter() - started
                metrics.update(generation_runtime=elapsed,
                               generation_samples_per_second=len(results) / max(elapsed, 1e-9))
                output.metrics.update({f"{metric_key_prefix}_full_{key}": value for key, value in metrics.items()})
            return output
        finally:
            self.model.train(was_training)
