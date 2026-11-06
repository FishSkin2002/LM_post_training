"""Train an SFT, DPO, or GRPO LoRA adapter."""

import argparse
import os
import sys
from pathlib import Path

from utils.arguments import add_generation_arguments, add_prepared_data_argument
from utils.paths import CONFIG_DIR, GENERATION_CONFIG, ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train an SFT, DPO, or GRPO LoRA adapter.")
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("sft", "dpo", "grpo"):
        command = stages.add_parser(stage, help=f"Train or resume the {stage.upper()} stage.")
        command.add_argument("--config", type=Path, default=CONFIG_DIR / f"training/{stage}.yaml")
        command.add_argument("--output-dir", type=Path, help="New run directory; collisions add -1, -2, ... .")
        source = command.add_mutually_exclusive_group(required=stage != "sft")
        source.add_argument("--resume-from-checkpoint", type=Path, help="Restore an unfinished training checkpoint.")
        if stage == "sft":
            source.add_argument("--continue-from", type=Path, help="Start a new stage from a saved SFT adapter.")
            command.add_argument("--additional-epochs", type=float,
                                 help="Extra epochs in the new SFT stage; requires --continue-from.")
        else:
            source.add_argument("--sft-source", type=Path, help="Initialize from a completed SFT final/ adapter.")
        add_generation_arguments(command)
        add_prepared_data_argument(command)
    args = parser.parse_args(argv)
    try:
        common = dict(output_dir=args.output_dir, resume=args.resume_from_checkpoint,
                      generation_config=args.generation_config, generation_backend=args.generation_backend,
                      prepared_sft_data=args.prepared_sft_data)
        from utils.generation import load_generation_config
        fresh_process = load_generation_config(args.generation_config, args.generation_backend)["backend"] == "vllm"
        if args.stage == "sft":
            from utils.training.sft import train
            final_dir = train(args.config, continue_from=args.continue_from,
                              additional_epochs=args.additional_epochs, **common)
        elif args.stage == "dpo":
            from utils.training.dpo import train
            final_dir = train(args.config, sft_source=args.sft_source, **common)
        else:
            from utils.training.grpo import load_config, train
            final_dir = train(args.config, sft_source=args.sft_source, **common)
            fresh_process |= load_config(args.config)[0]["trainer"].get("use_vllm", False)
        if fresh_process:
            exec_final_evaluation(final_dir, args.generation_config, args.generation_backend, args.prepared_sft_data)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))


def evaluate_after_training(final_dir, settings, *, model, tokenizer, dataset, data_info=None,
                            disable_tqdm=False, generation_config=None, training_uses_vllm=False):
    """Reuse the trained model for native evaluation; fresh-process evaluation follows main()."""
    if settings["backend"] == "transformers" and not training_uses_vllm:
        from evaluate_checkpoint import evaluate_checkpoint
        return evaluate_checkpoint(
            final_dir, model=model, tokenizer=tokenizer, dataset=dataset, data_info=data_info,
            disable_tqdm=disable_tqdm, generation_config=generation_config,
            generation_backend="transformers",
        )


def exec_final_evaluation(final_dir, generation_config=None, generation_backend=None, prepared_sft_data=None):
    """Replace the training process so no training CUDA context survives into vLLM."""
    args = [sys.executable, str(ROOT / "evaluate_checkpoint.py"),
            "--checkpoint", str(final_dir), "--generation-config", str(generation_config or GENERATION_CONFIG)]
    if generation_backend:
        args.extend(["--generation-backend", generation_backend])
    if prepared_sft_data:
        args.extend(["--prepared-sft-data", str(prepared_sft_data)])
    print(f"Starting saved-checkpoint evaluation: {final_dir}", flush=True)
    try:
        os.execv(sys.executable, args)
    except OSError as error:
        raise RuntimeError(f"Training is saved at {final_dir}; run evaluate_checkpoint.py --checkpoint {final_dir} separately.") from error


def load_training_model(config, source_path=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    from utils.prepare_sft_data import model_dtype

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


def save_final(trainer, tokenizer, run_dir):
    final_dir = run_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    print(f"Training complete. Saved adapter and tokenizer to {final_dir}", flush=True)
    return final_dir


class AdapterTrainerMixin:
    def __init__(self, *args, run_metadata, **kwargs):
        from transformers.trainer_callback import PrinterCallback, ProgressCallback
        from utils.training_workflow import TerminalPrinterCallback, TerminalProgressCallback

        self.run_metadata = run_metadata
        super().__init__(*args, **kwargs)
        self.remove_callback(ProgressCallback)
        self.remove_callback(PrinterCallback)
        self.add_callback(TerminalPrinterCallback if self.args.disable_tqdm else TerminalProgressCallback)

    def _save(self, output_dir=None, state_dict=None):
        import torch
        from transformers.trainer import TRAINING_ARGS_NAME
        from utils.training_workflow import save_run

        # A PEFT ref/ subdirectory would cause Transformers to skip the root policy on resume.
        path = Path(output_dir or self.args.output_dir)
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(path, state_dict=state_dict, selected_adapters=["default"])
        self.processing_class.save_pretrained(path)
        torch.save(self.args, path / TRAINING_ARGS_NAME)
        save_run(path, self.run_metadata)


def require_resume_engine(saved):
    from utils.generation import package_version

    engine = saved.get("generation_engine")
    if engine is None:
        if package_version("trl") != "1.16.0":
            raise ValueError("This legacy checkpoint requires its original TRL 1.16.0 environment; start a new run from SFT final/ to change environments.")
        return
    if engine["trl_version"] != package_version("trl"):
        raise ValueError("Resume requires the saved TRL version; use the original environment.")
    if engine["backend"] == "vllm" and engine["vllm_version"] != package_version("vllm"):
        raise ValueError("Resume requires the saved vLLM version; use the original environment.")


if __name__ == "__main__":
    sys.modules["train"] = sys.modules[__name__]
    main()
