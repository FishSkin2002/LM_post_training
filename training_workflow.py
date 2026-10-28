"""Run state, held-out validation and native Trainer integration."""

from copy import deepcopy
import json
import math
from pathlib import Path
import random

from transformers import Trainer
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.trainer_utils import has_length
from tqdm.auto import tqdm

from prepare_sft_data import TEXT_FORMAT, require_base_model


RUN_FILE = "run.json"


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


def read_source(path, config=None, *, resume=False):
    path = Path(path).expanduser().resolve()
    required = [RUN_FILE, "adapter_config.json", "adapter_model.safetensors",
                "tokenizer.json", "tokenizer_config.json"]
    if resume:
        required += ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                     "training_args.bin"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete {'checkpoint' if resume else 'adapter source'}: missing {', '.join(missing)}")
    metadata = json.loads((path / RUN_FILE).read_text(encoding="utf-8"))
    config = config if config is not None else metadata["config"]
    require_base_model(config["model_name"])
    saved = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if saved["base_model_name_or_path"] != config["model_name"]:
        raise ValueError("Source and configuration must use the same Qwen3 Base model.")
    if metadata["config"]["model_name"] != config["model_name"]:
        raise ValueError("Source run model differs from the configuration.")
    if metadata.get("text_format") != TEXT_FORMAT:
        raise ValueError("Source text format differs; start a new run with the current prompt and answer cleanup.")
    for key, value in config["lora"].items():
        if saved[key] != value:
            raise ValueError(f"Source LoRA {key} differs from the configuration.")
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
        "config": deepcopy(config), "data": plan, "epochs": epochs, "text_format": deepcopy(TEXT_FORMAT),
        "source_adapter": str(source_path) if source_path else None,
    }


def format_metrics(logs, label=None):
    label = label or ("Validation" if any(key.startswith("eval_") for key in logs) else "Training")
    values = (f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
              for key, value in logs.items() if key != "total_flos")
    return label + " | " + " | ".join(values)


class TerminalProgressCallback(ProgressCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, desc="Training", dynamic_ncols=True)
        self.current_step = 0

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader), desc="Validation loss",
                                           leave=self.training_bar is None, dynamic_ncols=True)
            self.prediction_bar.update(1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.write(format_metrics(logs))


class TerminalPrinterCallback(PrinterCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(format_metrics(logs))


class WorkflowTrainer(Trainer):
    def __init__(self, *args, run_metadata, **kwargs):
        self.run_metadata = run_metadata
        super().__init__(*args, **kwargs)
        self.remove_callback(ProgressCallback)
        self.remove_callback(PrinterCallback)
        self.add_callback(TerminalPrinterCallback if self.args.disable_tqdm else TerminalProgressCallback)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        save_run(output_dir or self.args.output_dir, self.run_metadata)
