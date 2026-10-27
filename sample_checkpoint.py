"""Sample GSM8K questions and inspect answers from a local SFT checkpoint."""

import argparse
from pathlib import Path
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from prepare_sft_data import COMMON_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not (args.checkpoint / "config.json").is_file():
        parser.error("Checkpoint must be a local model directory containing config.json.")
    if args.num_samples <= 0 or args.max_new_tokens <= 0:
        parser.error("Sample count and max-new-tokens must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or --device auto.")
    return args, parser


@torch.inference_mode()
def generate_answer(model, tokenizer, question, max_new_tokens):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=COMMON_CONFIG["enable_thinking"],
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[0, inputs["input_ids"].shape[-1]:].tolist()
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    truncated = len(generated) >= max_new_tokens and generated[-1] not in eos_ids
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated), truncated


def main():
    args, parser = parse_args()
    dataset_config = {**COMMON_CONFIG["dataset"], "split": args.split}
    dataset = load_dataset(**dataset_config)
    if args.num_samples > len(dataset):
        parser.error(f"Requested {args.num_samples} samples, but {args.split} has {len(dataset)} rows.")
    indices = random.Random(args.seed).sample(range(len(dataset)), args.num_samples)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Checkpoint: {args.checkpoint} | Device: {device} | Dtype: {dtype}", flush=True)
    print(f"Dataset: {dataset_config} | Indices: {indices}", flush=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=COMMON_CONFIG["attention_implementation"],
    ).to(device)
    model.eval()
    model.config.use_cache = True

    report = [
        "# Checkpoint 样本问答\n",
        f"- Checkpoint：`{args.checkpoint}`\n"
        f"- 数据集：`{dataset_config['path']}/{dataset_config['name']}`，split：`{args.split}`\n"
        f"- 种子：{args.seed}；样本索引：{indices}\n"
        f"- 设备：`{device}`；推理精度：`{dtype}`\n"
        f"- enable_thinking：`{COMMON_CONFIG['enable_thinking']}`；do_sample：`False`\n"
        f"- 最大新词元数：{args.max_new_tokens}\n",
    ]
    for number, index in enumerate(indices, 1):
        row = dataset[index]
        answer, token_count, truncated = generate_answer(
            model, tokenizer, row["question"], args.max_new_tokens,
        )
        status = "达到生成长度上限，回答可能未结束" if truncated else "正常结束"
        section = (
            f"## 样本 {number}（{args.split} 索引 {index}）\n\n"
            f"### 问题\n\n{row['question']}\n\n"
            f"### 模型回答\n\n{answer}\n\n"
            f"生成 {token_count} tokens；{status}。\n\n"
            f"### 数据集参考解答\n\n{row['answer']}\n"
        )
        print(f"\n{section}", flush=True)
        report.append(section)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(report), encoding="utf-8")
        print(f"Saved report to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
