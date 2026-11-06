"""Repository paths shared by business modules and command-line entrypoints."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
GENERATION_CONFIG = CONFIG_DIR / "generation.yaml"
