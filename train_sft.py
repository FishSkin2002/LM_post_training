"""Fine-tune Qwen3 on GSM8K reference reasoning and final results."""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from prepare_sft_data import (
    COMMON_CONFIG,
    CONFIG_DIR,
    IGNORE_INDEX,
    build_sft_example,
    load_machine_config,
)
from sft_runtime import attach_lora, make_lora_config, validate_resume


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIG_DIR / "training" / "sft.yaml",
        help="Training configuration YAML (default: configs/training/sft.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        help="Run directory; new runs add -1, -2, ... on collision; resumes default to the checkpoint parent.",
    )
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--training-mode", choices=("lora", "full"))
    parser.add_argument("--max-steps", type=int, help="Stop after this many optimizer updates.")
    parser.add_argument(
        "--memory-smoke-test", action="store_true",
        help="Repeat the longest real training example; default to 2 updates, save each update and report GPU peaks.",
    )
    args = parser.parse_args()
    try:
        config = load_machine_config(args.config)
    except (OSError, yaml.YAMLError, ValueError) as error:
        parser.error(f"Cannot read configuration: {error}")
    args.model_name = config.pop("model_name")
    configured_mode = config.pop("training_mode", "full")
    args.training_mode = args.training_mode or configured_mode
    if args.training_mode not in ("lora", "full"):
        parser.error("training_mode must be lora or full.")
    args.lora_options = config.pop("lora", {})
    try:
        make_lora_config(args.lora_options)
        if args.resume_from_checkpoint:
            args.model_revision = validate_resume(
                args.resume_from_checkpoint, args.training_mode, args.model_name, args.lora_options,
            )
        else:
            args.model_revision = None
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    overrides = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive.")
    if args.memory_smoke_test:
        config.update(
            max_steps=args.max_steps or 2, save_strategy="steps", save_steps=1,
            logging_steps=1, logging_nan_inf_filter=False,
        )
        if not args.resume_from_checkpoint and args.output_dir is None:
            config["output_dir"] += "-memory-smoke"
    if args.resume_from_checkpoint and args.output_dir is None:
        config["output_dir"] = str(Path(args.resume_from_checkpoint).expanduser().resolve().parent)
    try:
        for key, value_type in {
            "num_train_epochs": float,
            "learning_rate": float,
            "per_device_train_batch_size": int,
            "gradient_accumulation_steps": int,
        }.items():
            config[key] = value_type(config[key])
    except (KeyError, TypeError, ValueError) as error:
        parser.error(f"Invalid training parameter: {error}")
    if min(
        config["num_train_epochs"], config["learning_rate"],
        config["per_device_train_batch_size"], config["gradient_accumulation_steps"],
    ) <= 0:
        parser.error("Epochs, learning rate, batch size and accumulation steps must be positive.")
    args.training_config = config
    args.output_dir = config["output_dir"]
    return args


def reserve_output_dir(output_dir: str, *, resume: bool) -> str:
    """Reserve a fresh run directory atomically, or keep an explicit resume path."""
    base = Path(output_dir).expanduser().resolve()
    if resume:
        return str(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        candidate = base if suffix == 0 else base.with_name(f"{base.name}-{suffix}")
        try:
            candidate.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return str(candidate)


def load_training_data(tokenizer):
    dataset = load_dataset(**COMMON_CONFIG["dataset"])
    return dataset.map(
        lambda row: build_sft_example(tokenizer, row["question"], row["answer"]),
        remove_columns=dataset.column_names,
        desc="Tokenizing GSM8K train",
    )


def make_collator(tokenizer):
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
    )


def make_training_args(args):
    config = dict(args.training_config)
    use_cuda = torch.cuda.is_available() and not config.get("use_cpu", False)
    if config.get("bf16") == "auto":
        config["bf16"] = use_cuda and torch.cuda.is_bf16_supported()
    args.model_dtype = getattr(torch, COMMON_CONFIG["model_dtype"])
    if args.training_mode == "lora":
        if use_cuda:
            args.model_dtype = torch.bfloat16 if config.get("bf16") else torch.float16
            config["fp16"] = not config.get("bf16", False)
        else:
            args.model_dtype = torch.float32
    return TrainingArguments(**config)


def main():
    args = parse_args()
    training_args = make_training_args(args)
    args.output_dir = reserve_output_dir(
        args.output_dir, resume=bool(args.resume_from_checkpoint),
    )
    args.training_config["output_dir"] = args.output_dir
    training_args.output_dir = args.output_dir
    print(f"Training configuration: {args.config}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Training mode: {args.training_mode} | Base dtype: {args.model_dtype}", flush=True)
    print(f"Thinking mode: {COMMON_CONFIG['enable_thinking']} | Loss: reasoning + final result", flush=True)
    print(
        f"Batch size: {training_args.per_device_train_batch_size} | "
        f"Gradient accumulation: {training_args.gradient_accumulation_steps} | "
        f"BF16: {training_args.bf16} | "
        f"FP16: {training_args.fp16} | "
        f"Gradient checkpointing: {training_args.gradient_checkpointing}",
        flush=True,
    )
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)
    tokenizer.padding_side = COMMON_CONFIG["padding_side"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_training_data(tokenizer)
    print(
        f"Training examples: {len(train_dataset)} | "
        f"Longest example: {max(map(len, train_dataset['input_ids']))} tokens",
        flush=True,
    )
    if args.memory_smoke_test:
        longest = max(range(len(train_dataset)), key=lambda index: len(train_dataset[index]["input_ids"]))
        repeats = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        train_dataset = train_dataset.select([longest] * repeats)
        print(
            f"Memory smoke test: repeating longest example (index {longest}), "
            f"{training_args.max_steps} optimizer updates; not an accuracy experiment.", flush=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        dtype=args.model_dtype,
        attn_implementation=COMMON_CONFIG["attention_implementation"],
    )
    if args.training_mode == "lora":
        model = attach_lora(model, args.lora_options)
        model.print_trainable_parameters()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collator(tokenizer),
        processing_class=tokenizer,
    )
    use_cuda = training_args.device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(training_args.device)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if not math.isfinite(result.training_loss):
        raise RuntimeError("Training loss is not finite; inspect the run before continuing.")
    memory = {
        "training_mode": args.training_mode, "model_name": args.model_name,
        "base_dtype": str(args.model_dtype), "memory_smoke_test": args.memory_smoke_test,
        "global_step": trainer.state.global_step,
        "batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "bf16": training_args.bf16, "fp16": training_args.fp16,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "longest_example_tokens": max(map(len, train_dataset["input_ids"])),
    }
    if use_cuda:
        torch.cuda.synchronize(training_args.device)
        memory.update(
            gpu_name=torch.cuda.get_device_name(training_args.device),
            peak_allocated_gib=torch.cuda.max_memory_allocated(training_args.device) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(training_args.device) / 1024**3,
            total_memory_gib=torch.cuda.get_device_properties(training_args.device).total_memory / 1024**3,
        )
        print(f"GPU memory: {json.dumps(memory)}", flush=True)
    (Path(args.output_dir) / "memory_results.json").write_text(
        json.dumps(memory, indent=2) + "\n", encoding="utf-8",
    )
    model.config.use_cache = use_cache
    final_dir = str(Path(args.output_dir) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    print(f"Saved model and tokenizer to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
