"""Evaluate original Instruct or LoRA on the same held-out math questions."""

import argparse
import json
from pathlib import Path
from textwrap import indent
import time

from datasets.fingerprint import Hasher
import torch
from tqdm.auto import tqdm

from utils.prepare_sft_data import (CONFIG_DIR, load_config, load_tokenizer, prepare_dataset,
                              require_model, text_format, validate_precision)
from utils.math_scoring import SCORING, score_answer
from utils.math_data import OPENR1, dataset_label
from utils.generation import add_generation_arguments, create_generator, generate_answers, load_generation_config
from utils.paths import ROOT
from utils.rule_scoring import apply_rule_scoring
from utils.prepared_sft_data import (add_prepared_data_argument, ensure_prepared_sft_data, load_prepared_sft_data)
from utils.training_workflow import data_plan, format_metrics, read_source

DEFAULT_BATCH_SIZES = {"transformers": 4, "vllm": 64}


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
        f"Generated tokens: {prediction['generated_tokens']} | Batch size: {prediction['generation_batch_size']} | "
        f"Generation time share: {prediction['generation_runtime']:.3f} s\n\n"
        f"### Question\n\n{indent(prediction['question'], '    ')}\n\n"
        f"### Reference answer\n\n{indent(prediction['reference_answer'], '    ')}\n\n"
        f"### Reference final answer\n\n{indent(prediction['reference_final_answer'], '    ')}\n\n"
        f"### Model answer\n\n{indent(prediction['answer'], '    ')}\n\n---\n\n"
    )


def evaluate_checkpoint(checkpoint=None, *, model_name=None, num_samples=None, output=None, device=None,
                        model=None, tokenizer=None, dataset=None, data_info=None, config=None, disable_tqdm=False,
                        sft_source=None, batch_size=None, auto_batch_size=False,
                        generation_config=None, generation_backend=None, prepared_sft_data=None):
    generation_settings = load_generation_config(generation_config, generation_backend)
    if auto_batch_size and generation_settings["backend"] != "transformers":
        raise ValueError("--auto-batch-size requires --generation-backend transformers; vLLM uses fixed chunks.")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZES[generation_settings["backend"]]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch-size must be a positive integer.")
    if (checkpoint is None) == (model_name is None):
        raise ValueError("Specify exactly one of checkpoint or model-name.")
    if sft_source is not None and (checkpoint is not None or config is not None):
        raise ValueError("Use sft-source only with model-name and without a separate config.")
    metadata = None
    stage = "original"
    if checkpoint is not None:
        checkpoint, metadata, _ = read_source(checkpoint)
        prepared_sft_data = prepared_sft_data or metadata.get("prepared_sft_data")
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
    data_source = sft_source or checkpoint
    if dataset is None:
        prepared_sft_data = ensure_prepared_sft_data(data_source, metadata, prepared_sft_data,
                                                    metadata.get("prepared_sft_data") if metadata else None)
        dataset, data_info = (load_prepared_sft_data(prepared_sft_data, config, tokenizer) if prepared_sft_data
                              else prepare_dataset(config, tokenizer))
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
        directory = ROOT / "outputs" / "evaluations"
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
        for prediction in predictions:
            prediction.setdefault("generation_batch_size", 1)
    print(f"Generation progress: {len(predictions)}/{len(rows)} completed; {progress_path}", flush=True)
    batch_mode = "adaptive" if auto_batch_size else "fixed"
    print(f"Generation batch size: {batch_size} ({batch_mode})", flush=True)
    print(f"Readable answers: {readable_path}", flush=True)
    generator = None
    if len(predictions) < len(rows):
        generator = create_generator(model_name, checkpoint, tokenizer, config, generation_settings, device, model=model)
        if generation_settings["backend"] == "transformers":
            model = generator.model
        print(f"Generation engine: {generator.description}", flush=True)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as saved, \
                readable_path.open("a", encoding="utf-8") as readable, tqdm(
                total=len(rows), initial=len(predictions), desc="Generate validation answers",
                unit="question", disable=disable_tqdm) as progress:
            if readable.tell() == 0:
                readable.write(f"# Math generation evaluation ({data_label})\n\nSource type: {source_type}\n\nStage: {stage}\n\n"
                               f"Model: `{model_name}`\n\nSource: `{checkpoint or model_name}`\n\n")
            readable.flush()
            next_batch_size = batch_size
            last_successful_size = 0
            probing = auto_batch_size
            while len(predictions) < len(rows):
                start = len(predictions)
                while True:
                    actual_batch_size = min(next_batch_size, len(rows) - start)
                    batch = rows.select(range(start, start + actual_batch_size))
                    started = time.perf_counter()
                    initialization_before = generator.initialization_runtime
                    try:
                        if generation_settings["backend"] == "transformers":
                            answers = generate_answers(model, tokenizer, batch["question"], config["max_new_tokens"])
                        else:
                            answers = generator.generate(batch["question"], config["max_new_tokens"])
                        if len(answers) != actual_batch_size:
                            raise RuntimeError(f"Generation returned {len(answers)} answers for {actual_batch_size} questions.")
                    except Exception as error:
                        if not auto_batch_size or actual_batch_size == 1:
                            raise
                        if probing and 0 < last_successful_size < actual_batch_size:
                            next_batch_size = last_successful_size
                        else:
                            next_batch_size = actual_batch_size // 2
                        probing = False
                        progress.write(f"Generation batch of {actual_batch_size} failed ({type(error).__name__}: {error}). "
                                       f"Retrying question {start + 1} with batch size {next_batch_size}.")
                    else:
                        break
                    # Release the failed call's traceback before collecting its tensors.
                    generator.recover()
                initialization_delta = generator.initialization_runtime - initialization_before
                time_share = max(0, time.perf_counter() - started - initialization_delta) / actual_batch_size
                if probing:
                    last_successful_size = actual_batch_size
                    next_batch_size = actual_batch_size * 2
                for position, (row, answer) in enumerate(zip(batch, answers, strict=True), start=start):
                    text, count, truncated = answer
                    started = time.perf_counter()
                    correct, valid, truncated = score_answer(text, row["final_answer"], truncated)
                    score_time = time.perf_counter() - started
                    prediction = {"index": indices[position], "question": row["question"],
                                  "source_index": row["source_index"], "generation_index": row["generation_index"],
                                  "reference_answer": row["answer"], "reference_final_answer": row["final_answer"],
                                  "answer": text, "generated_tokens": count, "correct": correct,
                                  "format_valid": valid, "truncated": truncated,
                                  "generation_batch_size": actual_batch_size,
                                  "generation_runtime": time_share + score_time,
                                  "generation_engine": generator.description}
                    saved.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    saved.flush()
                    readable.write(readable_prediction(position, prediction))
                    readable.flush()
                    predictions.append(prediction)
                    progress.update(1)
    finally:
        if generator is not None:
            generator.close()
    results = [(row["correct"], row["format_valid"], row["truncated"]) for row in predictions]
    elapsed = sum(row["generation_runtime"] for row in predictions)
    metrics = answer_metrics(results)
    metrics.update(generation_runtime=elapsed, generation_samples_per_second=len(results) / max(elapsed, 1e-9))
    metrics["generation_initialization_runtime"] = generator.initialization_runtime if generator else 0.0
    report = {
        "source_type": source_type, "stage": stage, "model_name": model_name,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "sft_source": str(sft_source) if sft_source is not None else None,
        "prepared_sft_data": str(prepared_sft_data) if prepared_sft_data else None,
        "data": {"source": plan["source"], "split": "validation", "indices": indices,
                 "fingerprint": plan["fingerprint"], "split_seed": plan["split_seed"]},
        "text_format": text_format(tokenizer), "precision": "bf16" if config.get("bf16") else "fp32",
        "generation": {"do_sample": False, "max_new_tokens": config["max_new_tokens"],
                       "eos_token_id": tokenizer.eos_token_id, "batch_size": batch_size,
                       "auto_batch_size": auto_batch_size, "requested_backend": generation_settings["backend"],
                       "engine": generator.description if generator else None},
        "scoring": SCORING,
        "metrics": metrics, "predictions": predictions,
    }
    try:
        if config["data"]["name"] == OPENR1["name"]:
            apply_rule_scoring(report, output, disable_tqdm=disable_tqdm)
    finally:
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
    parser.add_argument("--batch-size", type=int,
                        help=f"Maximum questions per call (default: vLLM {DEFAULT_BATCH_SIZES['vllm']}, "
                             f"Transformers {DEFAULT_BATCH_SIZES['transformers']}); starting size in adaptive mode.")
    parser.add_argument("--auto-batch-size", action="store_true",
                        help="Transformers only: double successful batches until the first failure, then retry smaller batches.")
    parser.add_argument("--output", type=Path, help="JSON report path; default: run/evaluations for LoRA, outputs/evaluations for original Instruct.")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: CUDA when available, otherwise CPU.")
    add_generation_arguments(parser)
    add_prepared_data_argument(parser)
    args = parser.parse_args()
    try:
        evaluate_checkpoint(args.checkpoint, model_name=args.model_name, num_samples=args.num_samples,
                            output=args.output, device=args.device,
                            config=load_config(args.config) if args.checkpoint is None and args.config else None,
                            sft_source=args.sft_source, batch_size=args.batch_size, auto_batch_size=args.auto_batch_size,
                            generation_config=args.generation_config, generation_backend=args.generation_backend,
                            prepared_sft_data=args.prepared_sft_data)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
