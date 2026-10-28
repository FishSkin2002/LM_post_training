"""Build and inspect GSM8K examples for Qwen3 Base LoRA SFT."""

import argparse
from pathlib import Path
import re

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
IGNORE_INDEX = -100
PROMPT = (
    "Solve the following math problem step by step.\n"
    "On the last line, write only #### followed by the final numeric answer, without units or other text. "
    "For example, if the final answer is 4, write #### 4.\n"
    "Use #### only once in your response. Do not write anything after the final answer.\n\n"
    "Question: {question}\nAnswer:\n"
)
CALCULATOR_ANNOTATION = re.compile(r"<<[^<>]*>>")
TEXT_FORMAT = {"prompt": PROMPT, "answer_cleanup": CALCULATOR_ANNOTATION.pattern,
               "eos_token": "<|endoftext|>"}


def build_prompt(question):
    return PROMPT.format(question=question)


def clean_answer(answer):
    return CALCULATOR_ANNOTATION.sub("", answer)


def require_base_model(model_name):
    if not (model_name.startswith("Qwen/Qwen3-") and model_name.endswith("-Base")):
        raise ValueError("Use a Qwen3 Base repository ID, such as Qwen/Qwen3-4B-Base.")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    require_base_model(config["model_name"])
    return config


def load_tokenizer(source, *, local_files_only=False):
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
    tokenizer.chat_template = None
    if tokenizer.eos_token != TEXT_FORMAT["eos_token"]:
        raise ValueError("The Base tokenizer must use <|endoftext|> as its EOS token.")
    return tokenizer


def build_sft_example(tokenizer, question, answer):
    prompt_ids = tokenizer.encode(build_prompt(question), add_special_tokens=False)
    answer_ids = tokenizer.encode(clean_answer(answer), add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = prompt_ids + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "development/sft.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config["model_name"])
    row = load_dataset("openai/gsm8k", "main", split="train")[0]
    example = build_sft_example(tokenizer, row["question"], row["answer"])
    print("Complete training input")
    print(tokenizer.decode(example["input_ids"], skip_special_tokens=False))
    print("Supervised text")
    print(tokenizer.decode([value for value in example["labels"] if value != IGNORE_INDEX],
                           skip_special_tokens=False))
    print("input_ids:", example["input_ids"])
    print("labels:", example["labels"])


if __name__ == "__main__":
    main()
