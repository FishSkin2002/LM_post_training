"""Evaluate original Qwen3 Base or a LoRA checkpoint on the fixed GSM8K validation split."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
from textwrap import indent
import time

from datasets import load_dataset
from peft import PeftModel
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from prepare_sft_data import TEXT_FORMAT, clean_answer, load_tokenizer, require_base_model
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


def readable_prediction(position, prediction):
    return (
        f"## Question {position + 1} (dataset index {prediction['index']})\n\n"
        f"Correct: {prediction['correct']} | Format valid: {prediction['format_valid']} | "
        f"Truncated: {prediction['truncated']}\n\n"
        f"Generated tokens: {prediction['generated_tokens']} | "
        f"Generation time: {prediction['generation_runtime']:.3f} s\n\n"
        f"### Question\n\n{indent(prediction['question'], '    ')}\n\n"
        f"### Reference answer\n\n{indent(prediction['reference_answer'], '    ')}\n\n"
        f"### Model answer\n\n{indent(prediction['answer'], '    ')}\n\n---\n\n"
    )


def evaluate_checkpoint(checkpoint=None, *, model_name=None, num_samples=None, output=None, device=None,
                        model=None, tokenizer=None, dataset=None, disable_tqdm=False):
    if (checkpoint is None) == (model_name is None):
        raise ValueError("Specify exactly one of checkpoint or model-name.")
    metadata = None
    if checkpoint is not None:
        checkpoint, metadata, _ = read_source(checkpoint)
        model_name = metadata["config"]["model_name"]
    else:
        require_base_model(model_name)
    source_type = "lora" if checkpoint is not None else "base"
    if dataset is None:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    plan = data_plan(dataset, metadata)
    indices = plan["validation_indices"]
    if num_samples is not None:
        if num_samples <= 0 or num_samples > len(indices):
            raise ValueError(f"num-samples must be between 1 and {len(indices)}; omit it for the full validation set.")
        indices = indices[:num_samples]
    rows = dataset.select(indices)
    if checkpoint is not None:
        directory = checkpoint.parent / "evaluations"
        label = checkpoint.name
    else:
        directory = Path(__file__).resolve().parent / "outputs" / "evaluations"
        label = f"base-{model_name.rsplit('/', 1)[-1]}"
    output = Path(output).expanduser().resolve() if output else directory / f"{label}-n{len(rows)}.json"
    progress_path = output.with_suffix(".progress.jsonl")
    readable_path = output.with_suffix(".md")
    predictions = []
    if progress_path.is_file():
        with progress_path.open(encoding="utf-8") as saved:
            predictions = [json.loads(line) for line in saved]
    print(f"Generation progress: {len(predictions)}/{len(rows)} completed; {progress_path}", flush=True)
    print(f"Readable answers: {readable_path}", flush=True)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, local_files_only=True,
            adapter_kwargs={"local_files_only": True},
        )
        if checkpoint is not None:
            model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True)
        model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if tokenizer is None:
        tokenizer = load_tokenizer(checkpoint or model_name, local_files_only=True)
    was_training = model.training
    model.eval()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as saved, \
                readable_path.open("w", encoding="utf-8") as readable, tqdm(
                total=len(rows), initial=len(predictions), desc="Generate validation answers",
                unit="question", disable=disable_tqdm) as progress:
            readable.write(f"# GSM8K generation evaluation\n\nSource type: {source_type}\n\n"
                           f"Model: `{model_name}`\n\nSource: `{checkpoint or model_name}`\n\n")
            for position, prediction in enumerate(predictions):
                readable.write(readable_prediction(position, prediction))
            readable.flush()
            for position in range(len(predictions), len(rows)):
                row = rows[position]
                index = indices[position]
                expected = answer_number(row["answer"])
                if expected is None:
                    raise ValueError("Validation reference answer has an invalid #### number.")
                started = time.perf_counter()
                text, count, truncated = generate_answer(model, tokenizer, row["question"], MAX_NEW_TOKENS)
                predicted = answer_number(text)
                correct = not truncated and predicted == expected
                prediction = {"index": index, "question": row["question"],
                              "reference_answer": clean_answer(row["answer"]), "answer": text,
                              "generated_tokens": count, "correct": correct,
                              "format_valid": predicted is not None, "truncated": truncated,
                              "generation_runtime": time.perf_counter() - started}
                saved.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                saved.flush()
                readable.write(readable_prediction(position, prediction))
                readable.flush()
                predictions.append(prediction)
                progress.update(1)
    finally:
        model.train(was_training)
    results = [(row["correct"], row["format_valid"], row["truncated"]) for row in predictions]
    elapsed = sum(row["generation_runtime"] for row in predictions)
    metrics = answer_metrics(results)
    metrics.update(generation_runtime=elapsed, generation_samples_per_second=len(results) / max(elapsed, 1e-9))
    report = {
        "source_type": source_type, "model_name": model_name,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "data": {"dataset": "openai/gsm8k", "split": "validation", "indices": indices,
                 "fingerprint": plan["fingerprint"], "split_seed": plan["split_seed"]},
        "text_format": TEXT_FORMAT,
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                       "eos_token_id": tokenizer.eos_token_id},
        "scoring": "One #### followed only by a decimal number; exact numeric equality; truncated answers are incorrect.",
        "metrics": metrics, "predictions": predictions,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(format_metrics(metrics, "Generation evaluation"), flush=True)
    print(f"Saved evaluation report to {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Base LoRA checkpoint or final directory.")
    source.add_argument("--model-name", help="Original Base repository, e.g. Qwen/Qwen3-4B-Base.")
    parser.add_argument("--num-samples", type=int, help="Evaluate the first N validation questions; default: all.")
    parser.add_argument("--output", type=Path, help="JSON report path; default: run/evaluations for LoRA, outputs/evaluations for Base.")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: CUDA when available, otherwise CPU.")
    args = parser.parse_args()
    try:
        evaluate_checkpoint(args.checkpoint, model_name=args.model_name, num_samples=args.num_samples,
                            output=args.output, device=args.device)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
