"""Save sample GSM8K answers from Qwen3 Base or a new LoRA checkpoint."""

import argparse
from pathlib import Path
import random
import re

import torch
from datasets import disable_progress_bars, load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM
from transformers.utils import logging

from prepare_sft_data import load_tokenizer, require_base_model


@torch.inference_mode()
def generate_answer(model, tokenizer, question, max_new_tokens):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    if inputs["input_ids"].shape[-1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Question and output budget exceed the model context; reduce --max-new-tokens.")
    stop_ids = list({tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id})
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id,
    )[0, inputs["input_ids"].shape[-1]:].tolist()
    truncated = len(output) >= max_new_tokens and output[-1] not in stop_ids
    return tokenizer.decode(output, skip_special_tokens=True), len(output), truncated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-name", help="Original Base repository, e.g. Qwen/Qwen3-4B-Base.")
    source.add_argument("--checkpoint", type=Path, help="New Base LoRA checkpoint or final directory.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.num_samples, args.max_new_tokens) <= 0:
        parser.error("num-samples and max-new-tokens must be positive.")
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not (args.checkpoint / "adapter_config.json").is_file():
            parser.error("Checkpoint must contain a Base LoRA adapter_config.json.")
        saved = PeftConfig.from_pretrained(args.checkpoint, local_files_only=True)
        model_name = saved.base_model_name_or_path
        label = f"lora-{args.checkpoint.parent.name}-{args.checkpoint.name}"
    else:
        model_name = args.model_name
        label = f"base-{model_name}"
    try:
        require_base_model(model_name)
    except ValueError as error:
        parser.error(str(error))
    disable_progress_bars()
    logging.disable_progress_bar()
    logging.set_verbosity_error()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    local_only = args.local_files_only or args.checkpoint is not None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, local_files_only=local_only,
        adapter_kwargs={"local_files_only": local_only},
    )
    tokenizer = load_tokenizer(args.checkpoint or model_name, local_files_only=local_only)
    if args.checkpoint:
        model = PeftModel.from_pretrained(model, args.checkpoint, local_files_only=True)
    model.to(device).eval()
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.num_samples > len(dataset):
        parser.error("num-samples exceeds the dataset size.")
    indices = random.Random(args.seed).sample(range(len(dataset)), args.num_samples)
    report = [
        "# GSM8K 样本问答\n",
        f"- Base：`{model_name}`\n- 来源：`{args.checkpoint or model_name}`\n"
        f"- 数据：`openai/gsm8k/main/{args.split}`\n- 抽题种子：{args.seed}；索引：{indices}\n"
        f"- 贪心生成；max_new_tokens：{args.max_new_tokens}；精度：FP32\n",
    ]
    for index in indices:
        row = dataset[index]
        answer, count, truncated = generate_answer(model, tokenizer, row["question"], args.max_new_tokens)
        status = "达到生成上限，回答可能未结束" if truncated else "正常结束"
        report.append(
            f"## 样本 {index}\n\n### 问题\n\n{row['question']}\n\n"
            f"### 模型回答\n\n{answer}\n\n生成 {count} tokens；{status}。\n\n"
            f"### 数据集标准解答\n\n{row['answer']}\n"
        )
    label = re.sub(r"[^A-Za-z0-9._-]+", "--", label)
    output = args.output or (Path(__file__).resolve().parent / "outputs/samples"
                             / f"{label}-{args.split}-seed{args.seed}.md")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved report to {output}", flush=True)


if __name__ == "__main__":
    main()
