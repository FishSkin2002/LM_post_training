"""Train an Instruct LoRA adapter on selected OpenR1 reasoning trajectories."""

import argparse
import math
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, set_seed
from transformers.integrations import TensorBoardCallback

from prepare_sft_data import CONFIG_DIR, load_config, load_tokenizer, model_dtype, prepare_dataset, validate_precision
from math_data import OPENR1
from evaluate_checkpoint import evaluate_checkpoint
from training_workflow import WorkflowTrainer, data_plan, new_run, read_source, reserve_directory, save_run
from generation import add_generation_arguments, exec_final_evaluation, load_generation_config
from prepared_sft_data import (add_prepared_data_argument, load_prepared_sft_data,
                               require_data_environment, resolve_prepared_sft_data)


def load_training_model(config, source_path=None):
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], dtype=model_dtype(config))
    model.config.use_cache = False
    if source_path:
        model = PeftModel.from_pretrained(model, source_path, is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", target_modules="all-linear", **config["lora"],
        ))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "training/sft.yaml")
    parser.add_argument("--output-dir", type=Path, help="New run directory; collisions add -1, -2, ... .")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume-from-checkpoint", type=Path, help="Restore an unfinished training checkpoint.")
    source.add_argument("--continue-from", type=Path, help="Start a new stage from a saved adapter.")
    parser.add_argument("--additional-epochs", type=float, help="Extra epochs in the new stage; requires --continue-from.")
    add_generation_arguments(parser)
    add_prepared_data_argument(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    generation_settings = load_generation_config(args.generation_config, args.generation_backend)
    if bool(args.continue_from) != (args.additional_epochs is not None):
        parser.error("--continue-from and --additional-epochs must be provided together.")
    if args.additional_epochs is not None and (not math.isfinite(args.additional_epochs) or args.additional_epochs <= 0):
        parser.error("additional-epochs must be a finite positive number.")
    if config["data"]["name"] != OPENR1["name"]:
        parser.error("The current SFT source is OpenR1; GSM8K remains a reading/evaluation interface.")
    source_path = args.resume_from_checkpoint or args.continue_from
    metadata = None
    step = None
    try:
        if source_path:
            source_path, metadata, step = read_source(source_path, config, resume=bool(args.resume_from_checkpoint))
        device = "cpu" if config.get("use_cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
        validate_precision(config, device)
        tokenizer = load_tokenizer(source_path or config["model_name"], local_files_only=bool(source_path))
        prepared = resolve_prepared_sft_data(source_path, args.prepared_sft_data,
                                             metadata.get("prepared_sft_data") if metadata else None)
        require_data_environment(metadata, source_path, prepared)
        dataset, data_info = (load_prepared_sft_data(prepared, config, tokenizer) if prepared
                              else prepare_dataset(config, tokenizer))
        plan = data_plan(dataset, metadata, data_info)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.resume_from_checkpoint:
        output_dir = source_path.parent
        if args.output_dir and args.output_dir.expanduser().resolve() != output_dir:
            parser.error("Resume uses the checkpoint's run directory; use --continue-from for a new stage.")
    else:
        output_dir = reserve_directory(args.output_dir or config["output_dir"])
        epochs = args.additional_epochs if args.continue_from else config["num_train_epochs"]
        source_environment = metadata.get("data_environment") if metadata else None
        metadata = new_run(config, plan, epochs, source_path)
        if prepared and source_path:
            metadata["data_environment"] = source_environment
    if prepared:
        metadata["prepared_sft_data"] = str(prepared)
    save_run(output_dir, metadata)
    settings = {key: value for key, value in config.items() if key not in ("model_name", "lora", "output_dir", "data", "max_length", "max_new_tokens")}
    settings["num_train_epochs"] = metadata["epochs"]
    training_args = TrainingArguments(
        **settings, output_dir=str(output_dir), logging_first_step=True, report_to="none",
    )
    model_name = config["model_name"]
    print(f"Model: {model_name}\nOutput directory: {output_dir}", flush=True)
    set_seed(training_args.seed)
    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset = dataset.select(plan["train_indices"]).select_columns(columns)
    eval_dataset = dataset.select(plan["validation_indices"]).select_columns(columns)
    model = load_training_model(config, source_path)
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
        if generation_settings["backend"] == "transformers":
            evaluate_checkpoint(final_dir, model=trainer.model, tokenizer=tokenizer, dataset=dataset,
                                disable_tqdm=training_args.disable_tqdm,
                                generation_config=args.generation_config, generation_backend="transformers")
    finally:
        writer.close()
    if generation_settings["backend"] == "vllm":
        exec_final_evaluation(final_dir, args.generation_config, args.generation_backend)


if __name__ == "__main__":
    main()
