"""Shared command-line options without loading model or training dependencies."""

from pathlib import Path

from utils.paths import GENERATION_CONFIG


def add_generation_arguments(parser):
    parser.add_argument("--generation-config", type=Path, default=GENERATION_CONFIG,
                        help="Shared generation settings (default: configs/generation.yaml).")
    parser.add_argument("--generation-backend", choices=("vllm", "transformers"),
                        help="Override the configured generation backend; never falls back automatically.")
