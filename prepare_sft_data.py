"""Inspect Instruct/OpenR1 chat examples, length filtering and assistant-only labels."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer
from datasets.fingerprint import Hasher

from math_data import OPENR1, clean_answer, load_math_data

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
IGNORE_INDEX = -100
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = ("Solve the following math problem step by step.\n"
                 "End your response with the final answer in \\boxed{...}, for example \\boxed{4}.")
TEXT_FORMAT = {"protocol": "qwen3-instruct-openr1-v1", "system": SYSTEM_PROMPT,
               "user": "unchanged problem", "assistant": "selected R1 generation",
               "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}


def require_model(model_name):
    if model_name != MODEL_NAME:
        raise ValueError(f"Use the current Instruct model: {MODEL_NAME}.")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    require_model(config["model_name"])
    for key in ("max_length", "max_new_tokens"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    return config


def load_tokenizer(source, *, local_files_only=False):
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
    if not tokenizer.chat_template or tokenizer.eos_token != TEXT_FORMAT["eos_token"] or tokenizer.pad_token != TEXT_FORMAT["pad_token"]:
        raise ValueError("The Instruct tokenizer requires its native chat template, EOS and PAD.")
    return tokenizer


def text_format(tokenizer):
    return {**deepcopy(TEXT_FORMAT), "chat_template": tokenizer.chat_template}


def messages(question, answer=None, *, system=SYSTEM_PROMPT):
    result = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    if answer is not None:
        result.append({"role": "assistant", "content": answer})
    return result


def build_prompt(tokenizer, question):
    return tokenizer.apply_chat_template(messages(question), tokenize=False, add_generation_prompt=True)


def build_sft_example(tokenizer, question, answer):
    prompt_ids = tokenizer.apply_chat_template(messages(question), tokenize=True, add_generation_prompt=True, return_dict=False)
    full_ids = tokenizer.apply_chat_template(messages(question, answer), tokenize=True, add_generation_prompt=False, return_dict=False)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Training and generation token prefixes do not align.")
    if tokenizer.eos_token_id not in full_ids[len(prompt_ids):]:
        raise ValueError("The assistant response has no EOS.")
    end = len(full_ids) - full_ids[::-1].index(tokenizer.eos_token_id)
    target = full_ids[len(prompt_ids):end]
    if target.count(tokenizer.eos_token_id) != 1:
        raise ValueError("Assistant content contains an unexpected message-ending token.")
    return {"input_ids": full_ids[:end], "attention_mask": [1] * end,
            "labels": [IGNORE_INDEX] * len(prompt_ids) + target}


def prepare_dataset(config, tokenizer):
    dataset, info = load_math_data(config["data"])
    def encode(row):
        example = build_sft_example(tokenizer, row["question"], row["answer"])
        return {**example, "sequence_length": len(example["input_ids"])}
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    fingerprint = Hasher.hash({"records": dataset._fingerprint, "format": text_format(tokenizer), "tokenizer": backend})
    encoded = dataset.map(encode, desc="Tokenize math records", new_fingerprint=fingerprint)
    lengths = sorted(encoded["sequence_length"])
    if not lengths:
        raise ValueError("No eligible math records remain.")
    stats = {"min": lengths[0], "p50": lengths[len(lengths)//2],
             "p90": lengths[min(len(lengths)-1, int(len(lengths)*0.9))], "max": lengths[-1]}
    limit = config["max_length"]
    kept = encoded.filter(lambda length: length <= limit, input_columns="sequence_length", desc="Filter sequence lengths",
                          new_fingerprint=Hasher.hash({"encoded": encoded._fingerprint, "max_length": limit}))
    groups = len(set(kept["group_id"]))
    info.update(question_groups=groups, duplicate_records=len(kept)-groups, max_length=config["max_length"], eligible_records=len(encoded), kept_records=len(kept),
                over_length=len(encoded)-len(kept), length_tokens=stats,
                text_format=text_format(tokenizer))
    print(f"Data preparation | raw={info['raw_records']} | eligible={len(encoded)} | kept={len(kept)} | "
          f"over_length={info['over_length']} | max_length={limit} | groups={groups} | duplicates={len(kept)-groups} | "
          + " | ".join(f"length_{key}={value}" for key, value in stats.items())
          + "".join(f" | filtered_{key}={value}" for key, value in info["filter_counts"].items()), flush=True)
    return kept, info


def model_dtype(config):
    import torch
    return torch.bfloat16 if config.get("bf16", False) else torch.float32


def validate_precision(config, device):
    import torch
    if config.get("bf16", False) and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 requires a supported CUDA device; use CPU/FP32 only for explicit tiny-model checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "development/sft.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config["model_name"])
    dataset, info = prepare_dataset(config, tokenizer)
    if not len(dataset):
        parser.error("No records fit max_length; inspect the length report and adjust the explicit budget.")
    row = dataset[0]
    source = config["data"]
    print(f"Data source: {source['name']}/{source['config']}/{source['split']} revision={source['revision']}")
    print("Complete training input")
    print(tokenizer.decode(row["input_ids"], skip_special_tokens=False))
    print("Supervised text")
    print(tokenizer.decode([value for value in row["labels"] if value != IGNORE_INDEX], skip_special_tokens=False))
    print("input_ids:", row["input_ids"])
    print("labels:", row["labels"])


if __name__ == "__main__":
    main()
