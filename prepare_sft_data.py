"""Build and inspect GSM8K examples for Qwen3 Base LoRA SFT."""

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
IGNORE_INDEX = -100
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' }}"
    "{% if not loop.last or add_generation_prompt %}{{ '\\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def require_base_model(model_name):
    if not (model_name.startswith("Qwen/Qwen3-") and model_name.endswith("-Base")):
        raise ValueError("Use a Qwen3 Base repository ID, such as Qwen/Qwen3-4B-Base.")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    require_base_model(config["model_name"])
    return config


def load_tokenizer(source, *, local_files_only=False):
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def build_sft_example(tokenizer, question, answer):
    messages = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
    input_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        add_generation_prompt=False, return_dict=False,
    )
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("The question prompt must be an exact token prefix of the training example.")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids):],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "development/sft.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config["model_name"])
    row = load_dataset("openai/gsm8k", "main", split="train")[0]
    example = build_sft_example(tokenizer, row["question"], row["answer"])
    print("【完整训练输入】")
    print(tokenizer.decode(example["input_ids"], skip_special_tokens=False))
    print("【参与损失的文本】")
    print(tokenizer.decode([value for value in example["labels"] if value != IGNORE_INDEX],
                           skip_special_tokens=False))
    print("input_ids:", example["input_ids"])
    print("labels:", example["labels"])


if __name__ == "__main__":
    main()
