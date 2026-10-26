"""Fine-tune Qwen3-0.6B on the original GSM8K training answers."""

import argparse
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

from prepare_sft_data import COMMON_CONFIG, CONFIG_DIR, IGNORE_INDEX, MODEL_NAME, build_sft_example


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIG_DIR / "training" / "sft.yaml",
        help="Training configuration YAML (default: configs/training/sft.yaml).",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        parser.error(f"Cannot read configuration: {error}")
    if not isinstance(config, dict):
        parser.error("Training configuration must be a YAML mapping.")
    overrides = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
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
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_checkpoint:
        parser.error("Output directory is not empty; choose a new directory or resume a checkpoint.")
    return args


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
    if config.get("bf16") == "auto":
        config["bf16"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return TrainingArguments(**config)


def main():
    args = parse_args()
    training_args = make_training_args(args)
    print(f"Training configuration: {args.config}", flush=True)
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = COMMON_CONFIG["padding_side"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_training_data(tokenizer)
    print(
        f"Training examples: {len(train_dataset)} | "
        f"Longest example: {max(map(len, train_dataset['input_ids']))} tokens",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=getattr(torch, COMMON_CONFIG["model_dtype"]),
        attn_implementation=COMMON_CONFIG["attention_implementation"],
    )
    use_cache = model.config.use_cache
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.config.use_cache = use_cache
    final_dir = str(Path(args.output_dir) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    print(f"Saved model and tokenizer to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
