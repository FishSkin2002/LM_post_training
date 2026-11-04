"""Train a DPO or GRPO LoRA policy from a finished SFT adapter."""

import argparse
from functools import partial
import hashlib
import math
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import set_seed
from transformers.integrations import TensorBoardCallback
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
import yaml

from evaluate_checkpoint import evaluate_checkpoint
from post_training_data import SOURCES, boxed_reward, prepare_post_training_data
from prepare_sft_data import CONFIG_DIR, load_tokenizer, prepare_dataset, validate_precision
from train_sft import load_training_model
from training_workflow import (TerminalPrinterCallback, TerminalProgressCallback, data_plan,
                               read_source, reserve_directory, save_run)


class PostSFTTrainer:
    def __init__(self, *args, run_metadata, **kwargs):
        self.run_metadata = run_metadata
        super().__init__(*args, **kwargs)
        self.remove_callback(ProgressCallback)
        self.remove_callback(PrinterCallback)
        self.add_callback(TerminalPrinterCallback if self.args.disable_tqdm else TerminalProgressCallback)

    def _save(self, output_dir=None, state_dict=None):
        # A PEFT ref/ subdirectory would cause Transformers to skip the root policy on resume.
        path = Path(output_dir or self.args.output_dir)
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(path, state_dict=state_dict, selected_adapters=["default"])
        self.processing_class.save_pretrained(path)
        torch.save(self.args, path / TRAINING_ARGS_NAME)
        save_run(path, self.run_metadata)


class PreferenceTrainer(PostSFTTrainer, DPOTrainer):
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        # The pinned TRL text collator accepts these IDs without re-rendering an EOS suffix.
        return dataset


class MathGRPOTrainer(PostSFTTrainer, GRPOTrainer):
    pass


def load_stage_config(path, stage):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = config["data"]
    if tuple(source[key] for key in ("name", "config", "split")) != SOURCES[stage]:
        raise ValueError(f"{stage.upper()} uses {'/'.join(SOURCES[stage])}.")
    if not isinstance(source["revision"], str) or not source["revision"]:
        raise ValueError("Data revision must be a nonempty string.")
    length_key = "max_length" if stage == "dpo" else "max_prompt_length"
    if type(config[length_key]) is not int or config[length_key] <= 0:
        raise ValueError(f"{length_key} must be a positive integer.")
    beta = config["trainer"]["beta"]
    if not math.isfinite(beta) or beta < 0 or (stage == "dpo" and beta == 0):
        raise ValueError(f"Invalid beta for {stage.upper()}.")
    return config, config[length_key]


def training_arguments(stage, settings, output_dir, bf16, tokenizer):
    common = dict(output_dir=str(output_dir), bf16=bf16, logging_first_step=True,
                  report_to="none", eval_strategy="no", gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant": False})
    if stage == "dpo":
        return DPOConfig(**settings, **common, loss_type="sigmoid", max_length=None)
    if type(settings["max_completion_length"]) is not int or settings["max_completion_length"] <= 0:
        raise ValueError("max_completion_length must be a positive integer.")
    # The native EOS is the only terminator. Suppress PAD so TRL's clipped ratio uses the same boundary.
    args = GRPOConfig(**settings, **common, loss_type="grpo", scale_rewards="group", disable_dropout=True,
                      use_vllm=False, remove_unused_columns=False,
                      generation_kwargs={"use_cache": True, "suppress_tokens": [tokenizer.pad_token_id]})
    rollout_steps = args.steps_per_generation * args.num_iterations
    if args.gradient_accumulation_steps % rollout_steps:
        raise ValueError("Each GRPO optimizer step must contain whole rollout cycles for checkpoint resume.")
    return args


def train(stage, config_path, *, sft_source=None, resume=None, output_dir=None):
    config, length_limit = load_stage_config(config_path, stage)
    settings = config["trainer"]
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Post-SFT training uses one process.")
    saved = step = None
    if resume is not None:
        resume, saved, step = read_source(resume, resume=True)
        if saved.get("stage") != stage:
            raise ValueError(f"Resume requires a {stage.upper()} checkpoint.")
        if output_dir is not None and output_dir.expanduser().resolve() != resume.parent:
            raise ValueError("Resume uses the checkpoint's run directory.")
        for key in ("data", "trainer", "max_length", "max_prompt_length"):
            if config.get(key) != saved["config"].get(key):
                raise ValueError("Resume must keep the saved stage configuration.")
        sft_source = saved["source_adapter"]
    sft_source, sft, _ = read_source(sft_source)
    if sft.get("stage", "sft") != "sft" or sft_source.name != "final":
        raise ValueError("Use the final/ adapter from a finished SFT run.")
    evaluation = {key: sft[key] for key in ("config", "data", "text_format")}
    with (sft_source / "adapter_model.safetensors").open("rb") as weights:
        source_hash = hashlib.file_digest(weights, "sha256").hexdigest()
    if saved is not None and (evaluation != saved["evaluation"] or source_hash != saved["source_adapter_sha256"]):
        raise ValueError("The source SFT artifact has changed since this stage started.")
    if resume is not None and (resume.parent / "final" / "run.json").is_file():
        raise ValueError("This stage already has a saved final adapter.")

    sft_config = sft["config"]
    bf16 = sft_config.get("bf16", False)
    device = "cpu" if settings.get("use_cpu") or not torch.cuda.is_available() else "cuda"
    validate_precision(sft_config, device)
    tokenizer = load_tokenizer(sft_source, local_files_only=True)
    run_path = resume.parent if resume is not None else output_dir or config["output_dir"]
    args = training_arguments(stage, settings, run_path, bf16, tokenizer)
    sft_data, sft_info = prepare_dataset(sft_config, tokenizer)
    plan = data_plan(sft_data, sft, sft_info)
    held_out = sft_data.select(plan["validation_indices"])["question"]
    train_data, train_info = prepare_post_training_data(stage, config["data"], tokenizer, held_out, length_limit)
    if stage == "grpo" and len(train_data) < args.generation_batch_size // args.num_generations:
        raise ValueError("Too few eligible questions for a complete GRPO generation batch.")
    metadata = {"stage": stage, "config": config, "data": train_info, "evaluation": evaluation,
                "scoring": sft["scoring"],
                "source_adapter": str(sft_source), "source_adapter_sha256": source_hash}
    if saved is not None and train_info != saved["data"]:
        raise ValueError("Training data or filtering differs from the saved checkpoint.")

    set_seed(args.seed)
    model = load_training_model(sft_config, sft_source)
    if stage == "dpo":
        context_limit = length_limit
    else:
        context_limit = length_limit + args.max_completion_length
    if context_limit > model.config.max_position_embeddings:
        raise ValueError("The stage's token budget exceeds the model context.")
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    run_path = resume.parent if resume is not None else reserve_directory(run_path)
    args.output_dir = str(run_path)
    save_run(run_path, metadata)
    print(f"Stage: {stage.upper()}\nSFT source: {sft_source}\nOutput directory: {run_path}", flush=True)
    model.print_trainable_parameters()
    trainer_class = PreferenceTrainer if stage == "dpo" else MathGRPOTrainer
    options = {} if stage == "dpo" else {"reward_funcs": partial(boxed_reward, eos_token_id=tokenizer.eos_token_id)}
    # TRL uses reward function names for metric keys.
    if stage == "grpo":
        options["reward_funcs"].__name__ = boxed_reward.__name__
    with SummaryWriter(log_dir=str(run_path / "runs"), purge_step=step + 1 if step is not None else None) as writer:
        trainer = trainer_class(model=model, args=args, train_dataset=train_data, processing_class=tokenizer,
                                callbacks=[TensorBoardCallback(tb_writer=writer)], run_metadata=metadata, **options)
        result = trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
        if not math.isfinite(result.training_loss):
            raise RuntimeError("Training loss is not finite.")
        final_dir = run_path / "final"
        trainer.save_model(final_dir)
        trainer.save_state()
        print(f"Training complete. Saved adapter and tokenizer to {final_dir}", flush=True)
        evaluate_checkpoint(final_dir, model=trainer.model, tokenizer=tokenizer, dataset=sft_data,
                            data_info=sft_info, disable_tqdm=args.disable_tqdm)
    return final_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=SOURCES)
    parser.add_argument("--config", type=Path, help="Default: configs/training/<stage>.yaml.")
    parser.add_argument("--output-dir", type=Path, help="New run directory, with collision suffixes as needed.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sft-source", type=Path, help="Initialize from a completed SFT final/ adapter.")
    source.add_argument("--resume-from-checkpoint", type=Path, help="Resume an unfinished stage in its own run directory.")
    args = parser.parse_args()
    try:
        train(args.stage, args.config or CONFIG_DIR / f"training/{args.stage}.yaml",
              sft_source=args.sft_source, resume=args.resume_from_checkpoint, output_dir=args.output_dir)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
