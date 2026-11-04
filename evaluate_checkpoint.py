"""Evaluate original Instruct or LoRA on the same held-out math questions."""

import argparse
import json
from pathlib import Path
from textwrap import indent
import time

from datasets.fingerprint import Hasher
from peft import PeftModel
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from prepare_sft_data import (CONFIG_DIR, load_config, load_tokenizer, model_dtype, prepare_dataset,
                              require_model, text_format, validate_precision)
from math_scoring import SCORING, score_answer
from math_data import dataset_label
from sample_checkpoint import generate_answer
from training_workflow import data_plan, format_metrics, read_source


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
        f"### Reference final answer\n\n{indent(prediction['reference_final_answer'], '    ')}\n\n"
        f"### Model answer\n\n{indent(prediction['answer'], '    ')}\n\n---\n\n"
    )


def evaluate_checkpoint(checkpoint=None, *, model_name=None, num_samples=None, output=None, device=None,
                        model=None, tokenizer=None, dataset=None, data_info=None, config=None, disable_tqdm=False,
                        sft_source=None):
    if (checkpoint is None) == (model_name is None):
        raise ValueError("Specify exactly one of checkpoint or model-name.")
    if sft_source is not None and (checkpoint is not None or config is not None):
        raise ValueError("Use sft-source only with model-name and without a separate config.")
    metadata = None
    stage = "original"
    if checkpoint is not None:
        checkpoint, metadata, _ = read_source(checkpoint)
        stage = metadata.get("stage", "sft")
        sft_source = metadata.get("source_adapter") if stage in ("dpo", "grpo") else None
        metadata = metadata.get("evaluation", metadata)
        config = metadata["config"]
        model_name = config["model_name"]
    else:
        if sft_source is not None:
            sft_source, metadata, _ = read_source(sft_source)
            if metadata.get("stage", "sft") != "sft":
                raise ValueError("sft-source must be an SFT artifact.")
            config = metadata["config"]
        config = config or load_config(CONFIG_DIR / "training/sft.yaml")
        require_model(model_name)
        if model_name != config["model_name"]:
            raise ValueError("Model and configuration differ.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    validate_precision(config, device)
    if tokenizer is None:
        tokenizer = load_tokenizer(checkpoint or sft_source or model_name, local_files_only=True)
    if metadata and text_format(tokenizer) != metadata["text_format"]:
        raise ValueError("Source tokenizer template differs.")
    source_type = "lora" if checkpoint is not None else "original"
    if dataset is None:
        dataset, data_info = prepare_dataset(config, tokenizer)
    elif data_info is None and metadata is not None:
        data_info = metadata["data"]["source"]
    plan = data_plan(dataset, metadata, data_info)
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
        label = f"original-{model_name.rsplit('/', 1)[-1]}"
        if sft_source is not None:
            identity = Hasher.hash((plan["fingerprint"], indices, config["max_new_tokens"], config.get("bf16", False)))
            label += f"-{identity}"
    data_label = dataset_label(config["data"])
    output = Path(output).expanduser().resolve() if output else directory / f"{label}-{data_label}-n{len(rows)}.json"
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
            model_name, dtype=model_dtype(config), local_files_only=True,
            adapter_kwargs={"local_files_only": True},
        )
        if checkpoint is not None:
            model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True)
        model.to(device)
    was_training = model.training
    model.eval()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as saved, \
                readable_path.open("w", encoding="utf-8") as readable, tqdm(
                total=len(rows), initial=len(predictions), desc="Generate validation answers",
                unit="question", disable=disable_tqdm) as progress:
            readable.write(f"# Math generation evaluation ({data_label})\n\nSource type: {source_type}\n\nStage: {stage}\n\n"
                           f"Model: `{model_name}`\n\nSource: `{checkpoint or model_name}`\n\n")
            for position, prediction in enumerate(predictions):
                readable.write(readable_prediction(position, prediction))
            readable.flush()
            for position in range(len(predictions), len(rows)):
                row = rows[position]
                index = indices[position]
                started = time.perf_counter()
                text, count, truncated = generate_answer(model, tokenizer, row["question"], config["max_new_tokens"])
                correct, valid, truncated = score_answer(text, row["final_answer"], truncated)
                prediction = {"index": index, "question": row["question"],
                              "source_index": row["source_index"], "generation_index": row["generation_index"],
                              "reference_answer": row["answer"], "reference_final_answer": row["final_answer"],
                              "answer": text,
                              "generated_tokens": count, "correct": correct,
                              "format_valid": valid, "truncated": truncated,
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
        "source_type": source_type, "stage": stage, "model_name": model_name,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "sft_source": str(sft_source) if sft_source is not None else None,
        "data": {"source": plan["source"], "split": "validation", "indices": indices,
                 "fingerprint": plan["fingerprint"], "split_seed": plan["split_seed"]},
        "text_format": text_format(tokenizer), "precision": "bf16" if config.get("bf16") else "fp32",
        "generation": {"do_sample": False, "max_new_tokens": config["max_new_tokens"],
                       "eos_token_id": tokenizer.eos_token_id},
        "scoring": SCORING,
        "metrics": metrics, "predictions": predictions,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(format_metrics(metrics, "Generation evaluation"), flush=True)
    print(f"Saved evaluation report to {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Current Instruct LoRA checkpoint or final directory.")
    source.add_argument("--model-name", help="Original Instruct repository: Qwen/Qwen3-4B-Instruct-2507.")
    evaluation_source = parser.add_mutually_exclusive_group()
    evaluation_source.add_argument("--config", type=Path,
                                   help="Original-model settings; default: configs/training/sft.yaml.")
    evaluation_source.add_argument("--sft-source", type=Path,
                                   help="Anchor original-model evaluation to the exact held-out split of an SFT artifact.")
    parser.add_argument("--num-samples", type=int, help="Evaluate the first N validation questions; default: all.")
    parser.add_argument("--output", type=Path, help="JSON report path; default: run/evaluations for LoRA, outputs/evaluations for original Instruct.")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: CUDA when available, otherwise CPU.")
    args = parser.parse_args()
    try:
        evaluate_checkpoint(args.checkpoint, model_name=args.model_name, num_samples=args.num_samples,
                            output=args.output, device=args.device,
                            config=load_config(args.config) if args.checkpoint is None and args.config else None,
                            sft_source=args.sft_source)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
