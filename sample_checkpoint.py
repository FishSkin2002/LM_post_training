"""Save math answers from original Instruct or a current LoRA checkpoint."""

import argparse
from pathlib import Path
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers.utils import logging
from tqdm.auto import tqdm

from prepare_sft_data import (CONFIG_DIR, build_prompt, load_config, load_tokenizer, model_dtype,
                              prepare_dataset, require_model, text_format, validate_precision)
from training_workflow import read_source
from math_data import dataset_label


@torch.inference_mode()
def generate_answer(model, tokenizer, question, max_new_tokens):
    inputs = tokenizer(build_prompt(tokenizer, question), add_special_tokens=False, return_tensors="pt").to(model.device)
    if inputs["input_ids"].shape[-1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Question and output budget exceed the model context; reduce max-new-tokens.")
    was_training, use_cache = model.training, model.config.use_cache
    model.eval()
    try:
        model.config.use_cache = True
        with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=model.dtype == torch.bfloat16):
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )[0, inputs["input_ids"].shape[-1]:].tolist()
    finally:
        model.config.use_cache = use_cache
        model.train(was_training)
    truncated = len(output) >= max_new_tokens and output[-1] != tokenizer.eos_token_id
    return tokenizer.decode(output, skip_special_tokens=True), len(output), truncated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-name", help="Original Instruct repository: Qwen/Qwen3-4B-Instruct-2507.")
    source.add_argument("--checkpoint", type=Path, help="Current LoRA checkpoint or final directory.")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "training/sft.yaml",
                        help="Data and precision settings for an original model; checkpoints use their saved config.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, help="Default: budget saved in the source/config.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.checkpoint:
            args.checkpoint, metadata, _ = read_source(args.checkpoint)
            config = metadata["config"]
            model_name = config["model_name"]
            label = f"lora-{args.checkpoint.parent.name}-{args.checkpoint.name}"
        else:
            metadata = None
            config = load_config(args.config)
            model_name = args.model_name
            require_model(model_name)
            if model_name != config["model_name"]:
                raise ValueError("Model and configuration differ.")
            label = f"original-{model_name}"
        budget = args.max_new_tokens if args.max_new_tokens is not None else config["max_new_tokens"]
        if min(args.num_samples, budget) <= 0:
            raise ValueError("num-samples and max-new-tokens must be positive.")
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        validate_precision(config, device)
        local_only = args.local_files_only or args.checkpoint is not None
        tokenizer = load_tokenizer(args.checkpoint or model_name, local_files_only=local_only)
        if metadata and text_format(tokenizer) != metadata["text_format"]:
            raise ValueError("Source tokenizer template differs.")
        dataset, info = prepare_dataset(config, tokenizer)
        if metadata:
            from training_workflow import data_plan
            data_plan(dataset, metadata, info)
        if args.num_samples > len(dataset):
            raise ValueError("num-samples exceeds the dataset size.")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    logging.disable_progress_bar()
    logging.set_verbosity_error()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=model_dtype(config), local_files_only=local_only,
        adapter_kwargs={"local_files_only": local_only},
    )
    if args.checkpoint:
        model = PeftModel.from_pretrained(model, args.checkpoint, local_files_only=True)
    model.to(device).eval()
    indices = random.Random(args.seed).sample(range(len(dataset)), args.num_samples)
    report = ["# 数学样本问答\n", f"- Model：`{model_name}`\n- 来源：`{args.checkpoint or model_name}`\n"
              f"- 数据：`{config['data']}`\n- 抽题种子：{args.seed}；索引：{indices}\n"
              f"- 贪心生成；max_new_tokens：{budget}；BF16：{config.get('bf16', False)}\n"]
    for index in tqdm(indices, desc="Generate sample answers", unit="question"):
        row = dataset[index]
        answer, count, truncated = generate_answer(model, tokenizer, row["question"], budget)
        report.append(f"## 样本 {row['source_index']} / 轨迹 {row['generation_index']}\n\n"
                      f"### 问题\n\n{row['question']}\n\n### 模型回答\n\n{answer}\n\n"
                      f"生成 {count} tokens；截断：{truncated}。\n\n"
                      f"### 数据集标准解答\n\n{row['answer']}\n\n### 参考最终答案\n\n{row['final_answer']}\n")
    label = re.sub(r"[^A-Za-z0-9._-]+", "--", label)
    data_label = dataset_label(config["data"])
    output = args.output or (Path(__file__).resolve().parent / "outputs/samples"
                            / f"{label}-{data_label}-seed{args.seed}.md")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved report to {output}", flush=True)


if __name__ == "__main__":
    main()
