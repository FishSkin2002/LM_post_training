"""Train an SFT, DPO, or GRPO LoRA adapter."""

import argparse
from pathlib import Path

from utils.arguments import add_generation_arguments, add_prepared_data_argument
from utils.paths import CONFIG_DIR


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
        else:
            from utils.training.post_sft import load_stage_config, train
            final_dir = train(args.stage, args.config, sft_source=args.sft_source, **common)
            if args.stage == "grpo":
                fresh_process |= load_stage_config(args.config, args.stage)[0]["trainer"].get("use_vllm", False)
        if fresh_process:
            from utils.generation import exec_final_evaluation
            exec_final_evaluation(final_dir, args.generation_config, args.generation_backend, args.prepared_sft_data)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
