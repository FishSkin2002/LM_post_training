"""Train a Qwen3 Base LoRA adapter on complete GSM8K reference answers."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from prepare_sft_data import CONFIG_DIR, build_sft_example, load_config, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "training/sft.yaml")
    parser.add_argument("--output-dir", help="Use a new directory for each experiment.")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    model_name = config.pop("model_name")
    lora_options = config.pop("lora")
    if args.output_dir:
        config["output_dir"] = args.output_dir
    training_args = TrainingArguments(
        **config, save_strategy="epoch", save_total_limit=2,
        logging_steps=10, logging_first_step=True, report_to="none",
    )
    output_dir = Path(training_args.output_dir).expanduser().resolve()
    if not args.resume_from_checkpoint and output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir or explicitly resume.")
    training_args.output_dir = str(output_dir)
    print(f"Model: {model_name}\nOutput directory: {output_dir}", flush=True)
    set_seed(training_args.seed)
    tokenizer = load_tokenizer(model_name)
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(
        lambda row: build_sft_example(tokenizer, row["question"], row["answer"]),
        remove_columns=dataset.column_names,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    if args.resume_from_checkpoint:
        saved = PeftConfig.from_pretrained(args.resume_from_checkpoint, local_files_only=True)
        if saved.base_model_name_or_path != model_name:
            parser.error("Checkpoint and configuration must use the same Qwen3 Base model.")
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True,
                                          local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", target_modules="all-linear", **lora_options,
        ))
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = output_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    print(f"Saved adapter and tokenizer to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
