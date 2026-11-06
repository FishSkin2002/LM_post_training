"""Save math answers from original Instruct or a current LoRA checkpoint."""

import argparse
from pathlib import Path
import random
import re

import torch
from transformers.utils import logging
from tqdm.auto import tqdm

from prepare_sft_data import (CONFIG_DIR, load_config, load_tokenizer,
                              prepare_dataset, require_model, text_format, validate_precision)
from training_workflow import read_source
from math_data import dataset_label
from generation import (add_generation_arguments, create_generator, generate_answers, load_generation_config)
from prepared_sft_data import (add_prepared_data_argument, ensure_prepared_sft_data, load_prepared_sft_data)


def generate_answer(model, tokenizer, question, max_new_tokens):
    return generate_answers(model, tokenizer, [question], max_new_tokens)[0]


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
    add_generation_arguments(parser)
    add_prepared_data_argument(parser)
    args = parser.parse_args()
    try:
        if args.checkpoint:
            args.checkpoint, metadata, _ = read_source(args.checkpoint)
            args.prepared_sft_data = args.prepared_sft_data or metadata.get("prepared_sft_data")
            data_source = metadata.get("source_adapter") if metadata.get("stage") in ("dpo", "grpo") else args.checkpoint
            metadata = metadata.get("evaluation", metadata)
            config = metadata["config"]
            model_name = config["model_name"]
            label = f"lora-{args.checkpoint.parent.name}-{args.checkpoint.name}"
        else:
            metadata = None
            data_source = None
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
        generation_settings = load_generation_config(args.generation_config, args.generation_backend)
        local_only = args.local_files_only or args.checkpoint is not None
        tokenizer = load_tokenizer(args.checkpoint or model_name, local_files_only=local_only)
        if metadata and text_format(tokenizer) != metadata["text_format"]:
            raise ValueError("Source tokenizer template differs.")
        args.prepared_sft_data = ensure_prepared_sft_data(data_source, metadata, args.prepared_sft_data)
        dataset, info = (load_prepared_sft_data(args.prepared_sft_data, config, tokenizer) if args.prepared_sft_data
                         else prepare_dataset(config, tokenizer))
        if metadata:
            from training_workflow import data_plan
            data_plan(dataset, metadata, info)
        if args.num_samples > len(dataset):
            raise ValueError("num-samples exceeds the dataset size.")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    logging.disable_progress_bar()
    logging.set_verbosity_error()
    generation_config = {**config, "max_new_tokens": budget}
    generator = create_generator(model_name, args.checkpoint, tokenizer, generation_config, generation_settings, device,
                                 local_files_only=local_only)
    indices = random.Random(args.seed).sample(range(len(dataset)), args.num_samples)
    report = ["# 数学样本问答\n", f"- Model：`{model_name}`\n- 来源：`{args.checkpoint or model_name}`\n"
              f"- 数据：`{config['data']}`\n- 抽题种子：{args.seed}；索引：{indices}\n"
              f"- 准备数据缓存：`{args.prepared_sft_data}`\n"
              f"- 贪心生成；max_new_tokens：{budget}；BF16：{config.get('bf16', False)}\n"
              f"- 生成引擎：`{generator.description}`；初始化：{generator.initialization_runtime:.3f} s\n"]
    try:
        for index in tqdm(indices, desc="Generate sample answers", unit="question"):
            row = dataset[index]
            answer, count, truncated = generator.generate([row["question"]], budget)[0]
            report.append(f"## 样本 {row['source_index']} / 轨迹 {row['generation_index']}\n\n"
                          f"### 问题\n\n{row['question']}\n\n### 模型回答\n\n{answer}\n\n"
                          f"生成 {count} tokens；截断：{truncated}。\n\n"
                          f"### 数据集标准解答\n\n{row['answer']}\n\n### 参考最终答案\n\n{row['final_answer']}\n")
    finally:
        generator.close()
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
