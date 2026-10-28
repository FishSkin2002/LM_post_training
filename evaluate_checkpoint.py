"""Measure generated-answer metrics on a checkpoint's saved GSM8K validation split."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import time

from datasets import load_dataset
from peft import PeftModel
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from prepare_sft_data import load_tokenizer
from sample_checkpoint import generate_answer
from training_workflow import data_plan, format_metrics, read_source


MAX_NEW_TOKENS = 1024
NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")


def answer_number(text):
    if text.count("####") != 1:
        return None
    value = text.split("####")[1].strip()
    return Decimal(value.replace(",", "")) if NUMBER.fullmatch(value) else None


def answer_metrics(results):
    count = len(results)
    return {
        "answer_accuracy": sum(correct for correct, _, _ in results) / count,
        "answer_format_rate": sum(valid for _, valid, _ in results) / count,
        "generation_truncation_rate": sum(cut for _, _, cut in results) / count,
        "generation_samples": count,
    }


def evaluate_checkpoint(checkpoint, *, num_samples=None, output=None, device=None,
                        model=None, tokenizer=None, dataset=None, disable_tqdm=False):
    checkpoint, metadata, _ = read_source(checkpoint)
    if dataset is None:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    plan = data_plan(dataset, metadata)
    indices = plan["validation_indices"]
    if num_samples is not None:
        if num_samples <= 0 or num_samples > len(indices):
            raise ValueError(f"num-samples must be between 1 and {len(indices)}; omit it for the full validation set.")
        indices = indices[:num_samples]
    rows = dataset.select(indices)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            metadata["config"]["model_name"], dtype=torch.float32, local_files_only=True,
            adapter_kwargs={"local_files_only": True},
        )
        model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True)
        model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if tokenizer is None:
        tokenizer = load_tokenizer(checkpoint, local_files_only=True)
    results, predictions = [], []
    was_training = model.training
    started = time.perf_counter()
    model.eval()
    try:
        with tqdm(rows, desc="Generate validation answers", unit="question", disable=disable_tqdm) as progress:
            for position, row in enumerate(progress):
                index = indices[position]
                expected = answer_number(row["answer"])
                if expected is None:
                    raise ValueError("Validation reference answer has an invalid #### number.")
                text, count, truncated = generate_answer(model, tokenizer, row["question"], MAX_NEW_TOKENS)
                predicted = answer_number(text)
                correct = not truncated and predicted == expected
                results.append((correct, predicted is not None, truncated))
                predictions.append({"index": index, "answer": text, "generated_tokens": count,
                                    "correct": correct, "format_valid": predicted is not None,
                                    "truncated": truncated})
    finally:
        model.train(was_training)
    elapsed = time.perf_counter() - started
    metrics = answer_metrics(results)
    metrics.update(generation_runtime=elapsed, generation_samples_per_second=len(results) / max(elapsed, 1e-9))
    output = Path(output).expanduser().resolve() if output else (
        checkpoint.parent / "evaluations" / f"{checkpoint.name}-n{len(rows)}.json"
    )
    report = {
        "checkpoint": str(checkpoint), "model_name": metadata["config"]["model_name"],
        "data": {"dataset": "openai/gsm8k", "split": "validation", "indices": indices,
                 "fingerprint": plan["fingerprint"], "split_seed": plan["split_seed"]},
        "text_format": metadata["text_format"],
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                       "eos_token_id": tokenizer.eos_token_id},
        "scoring": "One #### followed only by a decimal number; exact numeric equality; truncated answers are incorrect.",
        "metrics": metrics, "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(format_metrics(metrics, "Generation evaluation"), flush=True)
    print(f"Saved evaluation report to {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Base LoRA checkpoint or final directory.")
    parser.add_argument("--num-samples", type=int, help="Evaluate the first N validation questions; default: all.")
    parser.add_argument("--output", type=Path, help="JSON report path; default: run/evaluations/<checkpoint>-n<N>.json.")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: CUDA when available, otherwise CPU.")
    args = parser.parse_args()
    try:
        evaluate_checkpoint(args.checkpoint, num_samples=args.num_samples, output=args.output, device=args.device)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
