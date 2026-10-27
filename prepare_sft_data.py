"""Build and inspect GSM8K SFT examples without loading model weights."""

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
COMMON_CONFIG = yaml.safe_load((CONFIG_DIR / "common.yaml").read_text(encoding="utf-8"))
IGNORE_INDEX = -100


def load_machine_config(config_path: Path) -> dict:
    """Read model selection and training parameters from one machine YAML."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Machine configuration must be a YAML mapping.")
    if not isinstance(config.get("model_name"), str) or not config["model_name"].strip():
        raise ValueError("Machine configuration must specify a nonempty model_name.")
    return config


def build_sft_example(tokenizer, question: str, answer: str) -> dict[str, list[int]]:
    """Return unpadded model inputs with loss restricted to the completion."""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=COMMON_CONFIG["enable_thinking"],
        return_dict=True,
    )
    encoded = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=COMMON_CONFIG["enable_thinking"],
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    prompt_ids = prompt["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("The prompt must be an exact token prefix of the full example.")
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIG_DIR / "development" / "sft.yaml",
        help="Machine configuration YAML (default: configs/development/sft.yaml).",
    )
    args = parser.parse_args()
    try:
        config = load_machine_config(args.config)
    except (OSError, yaml.YAMLError, ValueError) as error:
        parser.error(f"Cannot read configuration: {error}")
    print(f"Model: {config['model_name']} | Configuration: {args.config}", flush=True)
    dataset = load_dataset(**COMMON_CONFIG["dataset"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    row = dataset[0]
    example = build_sft_example(tokenizer, row["question"], row["answer"])
    supervised_ids = [label for label in example["labels"] if label != IGNORE_INDEX]
    print("【完整训练输入】")
    print(tokenizer.decode(example["input_ids"], skip_special_tokens=False))
    print("【参与损失的文本】")
    print(tokenizer.decode(supervised_ids, skip_special_tokens=False))
    print("input_ids:", example["input_ids"])
    print("labels:", example["labels"])


if __name__ == "__main__":
    main()
