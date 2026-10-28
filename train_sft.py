"""Train a Qwen3 Base LoRA adapter on complete GSM8K reference answers."""

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, set_seed
from transformers.integrations import TensorBoardCallback

from prepare_sft_data import CONFIG_DIR, build_sft_example, load_config, load_tokenizer
from evaluate_checkpoint import evaluate_checkpoint
from training_workflow import WorkflowTrainer, data_plan, new_run, read_source, reserve_directory, save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "training/sft.yaml")
    parser.add_argument("--output-dir", type=Path, help="New run directory; collisions add -1, -2, ... .")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume-from-checkpoint", type=Path, help="Restore an unfinished training checkpoint.")
    source.add_argument("--continue-from", type=Path, help="Start a new stage from a saved adapter.")
    parser.add_argument("--additional-epochs", type=float, help="Extra epochs in the new stage; requires --continue-from.")
    args = parser.parse_args()
    config = load_config(args.config)
    if bool(args.continue_from) != (args.additional_epochs is not None):
        parser.error("--continue-from and --additional-epochs must be provided together.")
    if args.additional_epochs is not None and (not math.isfinite(args.additional_epochs) or args.additional_epochs <= 0):
        parser.error("additional-epochs must be a finite positive number.")
    source_path = args.resume_from_checkpoint or args.continue_from
    metadata = None
    step = None
    try:
        if source_path:
            source_path, metadata, step = read_source(source_path, config, resume=bool(args.resume_from_checkpoint))
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        plan = data_plan(dataset, metadata)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.resume_from_checkpoint:
        output_dir = source_path.parent
        if args.output_dir and args.output_dir.expanduser().resolve() != output_dir:
            parser.error("Resume uses the checkpoint's run directory; use --continue-from for a new stage.")
    else:
        output_dir = reserve_directory(args.output_dir or config["output_dir"])
        epochs = args.additional_epochs if args.continue_from else config["num_train_epochs"]
        metadata = new_run(config, plan, epochs, source_path)
    save_run(output_dir, metadata)
    settings = {key: value for key, value in config.items() if key not in ("model_name", "lora", "output_dir")}
    settings["num_train_epochs"] = metadata["epochs"]
    training_args = TrainingArguments(
        **settings, output_dir=str(output_dir), logging_first_step=True, report_to="none",
    )
    model_name = config["model_name"]
    print(f"Model: {model_name}\nOutput directory: {output_dir}", flush=True)
    set_seed(training_args.seed)
    tokenizer = load_tokenizer(source_path or model_name, local_files_only=bool(source_path))
    validation = dataset.select(plan["validation_indices"])

    def tokenize(rows, description):
        return rows.map(lambda row: build_sft_example(tokenizer, row["question"], row["answer"]),
                        remove_columns=rows.column_names, desc=description)

    train_dataset = tokenize(dataset.select(plan["train_indices"]), "Tokenize training split")
    eval_dataset = tokenize(validation, "Tokenize validation split")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    if source_path:
        model = PeftModel.from_pretrained(model, source_path, is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", target_modules="all-linear", **config["lora"],
        ))
    model.print_trainable_parameters()
    writer = SummaryWriter(log_dir=str(output_dir / "runs"), purge_step=step + 1 if step is not None else None)
    trainer = WorkflowTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
        processing_class=tokenizer, callbacks=[TensorBoardCallback(tb_writer=writer)],
        run_metadata=metadata,
    )
    try:
        result = trainer.train(resume_from_checkpoint=source_path if args.resume_from_checkpoint else None)
        if not math.isfinite(result.training_loss):
            raise RuntimeError("Training loss is not finite.")
        final_dir = output_dir / "final"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        trainer.save_state()
        print(f"Training complete. Saved adapter and tokenizer to {final_dir}", flush=True)
        evaluate_checkpoint(final_dir, model=trainer.model, tokenizer=tokenizer, dataset=dataset,
                            disable_tqdm=training_args.disable_tqdm)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
