"""Train an Instruct LoRA adapter on selected OpenR1 reasoning trajectories."""

import math

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorForSeq2Seq, TrainingArguments, set_seed
from transformers.integrations import TensorBoardCallback

from utils.prepare_sft_data import load_config, load_tokenizer, prepare_dataset, validate_precision
from utils.math_data import OPENR1
from utils.evaluation import evaluate_checkpoint
from utils.training_workflow import WorkflowTrainer, data_plan, new_run, read_source, reserve_directory, save_run
from utils.generation import load_generation_config
from utils.prepared_sft_data import ensure_prepared_sft_data, load_prepared_sft_data
from utils.training.common import load_training_model, save_final


def train(config_path, *, output_dir=None, resume=None, continue_from=None, additional_epochs=None,
          generation_config=None, generation_backend=None, prepared_sft_data=None):
    config = load_config(config_path)
    generation_settings = load_generation_config(generation_config, generation_backend)
    if bool(continue_from) != (additional_epochs is not None):
        raise ValueError("--continue-from and --additional-epochs must be provided together.")
    if additional_epochs is not None and (not math.isfinite(additional_epochs) or additional_epochs <= 0):
        raise ValueError("additional-epochs must be a finite positive number.")
    if config["data"]["name"] != OPENR1["name"]:
        raise ValueError("The current SFT source is OpenR1; GSM8K remains a reading/evaluation interface.")
    source_path = resume or continue_from
    metadata = None
    step = None
    if source_path:
        source_path, metadata, step = read_source(source_path, config, resume=bool(resume))
    device = "cpu" if config.get("use_cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    validate_precision(config, device)
    tokenizer = load_tokenizer(source_path or config["model_name"], local_files_only=bool(source_path))
    prepared = ensure_prepared_sft_data(source_path, metadata, prepared_sft_data,
                                       metadata.get("prepared_sft_data") if metadata else None)
    dataset, data_info = (load_prepared_sft_data(prepared, config, tokenizer) if prepared
                          else prepare_dataset(config, tokenizer))
    plan = data_plan(dataset, metadata, data_info)
    if resume:
        if output_dir and output_dir.expanduser().resolve() != source_path.parent:
            raise ValueError("Resume uses the checkpoint's run directory; use --continue-from for a new stage.")
        output_dir = source_path.parent
    else:
        output_dir = reserve_directory(output_dir or config["output_dir"])
        epochs = additional_epochs if continue_from else config["num_train_epochs"]
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
        result = trainer.train(resume_from_checkpoint=source_path if resume else None)
        if not math.isfinite(result.training_loss):
            raise RuntimeError("Training loss is not finite.")
        final_dir = save_final(trainer, tokenizer, output_dir)
        if generation_settings["backend"] == "transformers":
            evaluate_checkpoint(final_dir, model=trainer.model, tokenizer=tokenizer, dataset=dataset,
                                disable_tqdm=training_args.disable_tqdm,
                                generation_config=generation_config, generation_backend="transformers")
    finally:
        writer.close()
    return final_dir
