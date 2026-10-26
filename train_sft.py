"""Fine-tune Qwen3-0.6B on the original GSM8K training answers."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from prepare_sft_data import IGNORE_INDEX, MODEL_NAME, build_sft_example


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/gsm8k-sft")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    if min(args.epochs, args.learning_rate, args.batch_size, args.gradient_accumulation_steps) <= 0:
        parser.error("Epochs, learning rate, batch size and accumulation steps must be positive.")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_checkpoint:
        parser.error("Output directory is not empty; choose a new directory or resume a checkpoint.")
    return args


def load_training_data(tokenizer):
    dataset = load_dataset("openai/gsm8k", "main", split="train")
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
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
    )


def main():
    args = parse_args()
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_training_data(tokenizer)
    print(
        f"Training examples: {len(train_dataset)} | "
        f"Longest example: {max(map(len, train_dataset['input_ids']))} tokens",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="sdpa"
    )
    use_cache = model.config.use_cache
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=make_training_args(args),
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
