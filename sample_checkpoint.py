"""Sample GSM8K answers from an original model or a local SFT checkpoint."""

import argparse
from pathlib import Path
import random
import re

import torch
from datasets import disable_progress_bars, load_dataset
from datasets.utils import logging as datasets_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.utils import logging as transformers_logging
from transformers.utils.hub import resolve_revision

from prepare_sft_data import COMMON_CONFIG
from sft_runtime import load_checkpoint_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Local full-model or LoRA adapter directory.")
    source.add_argument("--model-name", help="Original model repository, e.g. Qwen/Qwen3-4B.")
    parser.add_argument("--revision", help="Original model branch, tag or commit; requires --model-name.")
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Load the original model only from local cache; checkpoints always load locally.",
    )
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-new-tokens", type=int, default=COMMON_CONFIG["generation"]["max_new_tokens"],
        help="Output token budget including thinking (default: 32768).",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output", type=Path,
        help="Markdown report path (default: outputs/samples/<source>-<split>-seed<seed>.md).",
    )
    args = parser.parse_args()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not any((args.checkpoint / name).is_file() for name in ("config.json", "adapter_config.json")):
            parser.error("Checkpoint must contain config.json or adapter_config.json.")
        if args.revision is not None:
            parser.error("--revision requires --model-name; adapters use their saved base revision.")
        source_name = args.checkpoint.name
    else:
        args.model_name = args.model_name.strip()
        if not args.model_name:
            parser.error("--model-name must be nonempty.")
        if re.fullmatch(r"Qwen3-(?:0\.6|1\.7|4|8)B", args.model_name) and not Path(args.model_name).is_dir():
            parser.error(f"Use the full model repository ID: Qwen/{args.model_name} (or an existing local model directory).")
        source_name = "original-" + re.sub(r"[^A-Za-z0-9._-]+", "--", args.model_name).strip(".-")
    if args.num_samples <= 0 or args.max_new_tokens <= 0:
        parser.error("Sample count and max-new-tokens must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or --device auto.")
    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent / "outputs" / "samples"
            / f"{source_name}-{args.split}-seed{args.seed}.md"
        )
    args.output = args.output.expanduser().resolve()
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
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit and inputs["input_ids"].shape[-1] + max_new_tokens > context_limit:
        raise ValueError("Prompt and output budget exceed the model context; reduce --max-new-tokens.")
    generation_config = {**COMMON_CONFIG["generation"], "max_new_tokens": max_new_tokens}
    outputs = model.generate(
        **inputs,
        **generation_config,
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
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    thinking = None
    answer_ids = generated
    if think_end_id in generated:
        boundary = generated.index(think_end_id)
        thinking = tokenizer.decode(generated[:boundary], skip_special_tokens=True).strip()
        thinking = thinking.removeprefix("<think>").strip()
        answer_ids = generated[boundary + 1:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return thinking, answer, len(generated), truncated


def main():
    args, parser = parse_args()
    disable_progress_bars()
    datasets_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    transformers_logging.set_verbosity_error()
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
    if args.model_name is not None:
        resolved = resolve_revision(
            args.model_name, args.revision, local_files_only=args.local_files_only,
        )
        revision = getattr(resolved, "resolved", resolved)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, revision=revision, local_files_only=args.local_files_only,
            adapter_kwargs={"local_files_only": args.local_files_only},
            dtype=dtype, attn_implementation=COMMON_CONFIG["attention_implementation"],
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, revision=revision, local_files_only=args.local_files_only,
        )
        title = "原始模型样本问答"
        source_description = f"- 原始模型：`{args.model_name}`\n- Revision：`{revision or 'default'}`\n"
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        model = load_checkpoint_model(
            args.checkpoint, dtype=dtype, device=device,
            attention_implementation=COMMON_CONFIG["attention_implementation"],
        )
        title = "Checkpoint 样本问答"
        source_description = f"- Checkpoint：`{args.checkpoint}`\n"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    # Model/adapter initialization must not consume the generation seed.
    set_seed(args.seed)

    generation_config = {**COMMON_CONFIG["generation"], "max_new_tokens": args.max_new_tokens}
    report = [
        f"# {title}\n",
        source_description + f"- 数据集：`{dataset_config['path']}/{dataset_config['name']}`，split：`{args.split}`\n"
        f"- 种子：{args.seed}；样本索引：{indices}\n"
        f"- 设备：`{device}`；推理精度：`{dtype}`\n"
        f"- enable_thinking：`{COMMON_CONFIG['enable_thinking']}`；生成设置：`{generation_config}`\n"
        f"- 最大新词元数：{args.max_new_tokens}\n",
    ]
    for number, index in enumerate(indices, 1):
        row = dataset[index]
        thinking, answer, token_count, truncated = generate_answer(
            model, tokenizer, row["question"], args.max_new_tokens,
        )
        status = "达到生成长度上限，回答可能未结束" if truncated else "正常结束"
        if thinking is None:
            model_response = (
                "### 模型生成内容\n\n"
                "未检测到思考结束标记，以下保留生成内容，不将其判定为最终输出。\n\n"
                f"````text\n{answer}\n````\n\n"
            )
        else:
            model_response = (
                f"### 模型思考\n\n{thinking}\n\n"
                f"### 模型最终输出\n\n{answer}\n\n"
            )
        section = (
            f"## 样本 {number}（{args.split} 索引 {index}）\n\n"
            f"### 问题\n\n{row['question']}\n\n"
            f"{model_response}"
            f"生成 {token_count} tokens；{status}。\n\n"
            f"### 数据集参考解答\n\n{row['answer']}\n"
        )
        report.append(section)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
