"""Evaluate original Instruct or LoRA on the same held-out math questions."""

import argparse
import json
from pathlib import Path
import time

from datasets.fingerprint import Hasher
import torch
from tqdm.auto import tqdm

from utils.prepare_sft_data import (CONFIG_DIR, MODEL_NAME, load_config, load_tokenizer, prepare_dataset,
                              require_model, text_format, validate_precision)
from utils.math_scoring import SCORING, score_answer
from utils.math_data import OPENR1, dataset_label
from utils.generation import add_generation_arguments, create_generator, generate_answers, load_generation_config
from utils.paths import ROOT
from utils.evaluation_artifacts import EvaluationStore
from utils.rule_scoring import PROTOCOL, apply_rule_scoring, saved_rule_result
from utils.prepared_sft_data import ensure_prepared_sft_data, load_prepared_sft_data
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


def saved_evaluation(store):
    """Return an in-memory view for existing callers, without writing a summary file."""
    predictions = store.predictions()
    metrics = answer_metrics([(row["correct"], row["format_valid"], row["truncated"]) for row in predictions])
    elapsed = sum(row["generation_runtime"] for row in predictions)
    metrics.update(generation_runtime=elapsed,
                   generation_samples_per_second=len(predictions) / max(elapsed, 1e-9),
                   generation_initialization_runtime=store.metadata["generation_initialization_runtime"])
    report = {**store.metadata, "directory": str(store.directory),
              "data": {**store.metadata["data"], "indices": [row["index"] for row in store.samples]},
              "metrics": metrics, "predictions": predictions}
    if report["data"]["source"]["name"] == OPENR1["name"]:
        completed = 0
        for row in predictions:
            judgments = store.judgments.get(row["index"], {})
            row["rule_scoring"] = saved_rule_result(row, judgments)
            completed += row["correct"] or row["truncated"] or 2 in judgments
        report["rule_scoring"] = {"protocol": PROTOCOL, "completed_samples": completed,
                                  "total_samples": len(store.samples), "complete": completed == len(store.samples),
                                  "progress_file": str(store.judgments_path)}
    return report


def save_first_stage(store, answer):
    sample = store.samples[answer["index"]]
    started = time.perf_counter()
    correct, valid, truncated = score_answer(answer["answer"], sample["reference_final_answer"], answer["truncated"])
    reason = ("truncated" if truncated else "saved_reference_pass" if correct else
              "saved_reference_failure" if valid else "saved_invalid_model_box")
    store.save_judgment(answer["index"], 1,
                        {"status": "passed" if correct else "failed", "reason": reason,
                         "format_valid": valid, "runtime": time.perf_counter() - started})


def evaluate_checkpoint(checkpoint=None, *, model_name=None, num_samples=None, output=None, device=None,
                        model=None, tokenizer=None, dataset=None, data_info=None, config=None, disable_tqdm=False,
                        sft_source=None, batch_size=None, auto_batch_size=False,
                        generation_config=None, generation_backend=None, prepared_sft_data=None):
    if str(checkpoint) == MODEL_NAME and model_name is None:
        model_name, checkpoint = MODEL_NAME, None
    generation_settings = load_generation_config(generation_config, generation_backend)
    if auto_batch_size and generation_settings["backend"] != "transformers":
        raise ValueError("--auto-batch-size requires --generation-backend transformers; vLLM uses fixed chunks.")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZES[generation_settings["backend"]]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch-size must be a positive integer.")
    if (checkpoint is None) == (model_name is None):
        raise ValueError("Specify exactly one model source.")
    if sft_source is not None and (checkpoint is not None or config is not None):
        raise ValueError("Use --source only with the original Qwen checkpoint and without a separate --config.")
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
                raise ValueError("--source must be an SFT artifact.")
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
    if num_samples is not None and (type(num_samples) is not int or num_samples <= 0):
        raise ValueError("num-samples must be a positive integer; omit it for the full validation set.")
    if checkpoint is not None:
        directory = checkpoint.parent / "evaluations"
        label = checkpoint.name
    else:
        directory = ROOT / "outputs" / "evaluations"
        label = f"original-{model_name.rsplit('/', 1)[-1]}"
        if sft_source is not None:
            selected = metadata["data"]["validation_indices"]
            if num_samples is not None:
                selected = selected[:num_samples]
            identity = Hasher.hash((metadata["data"]["fingerprint"], selected,
                                    config["max_new_tokens"], config.get("bf16", False)))
            label += f"-{identity}"
    data_label = dataset_label(config["data"])
    prefix = f"{label}-{data_label}-n"
    if output is None and num_samples is not None:
        output = directory / f"{prefix}{num_samples}"
    if output is None:
        # Locate a previous full evaluation by its metadata, without reopening the source dataset.
        for path in sorted(directory.glob(f"{prefix}*/metadata.json")):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if candidate["data"].get("requested_num_samples") is None:
                output = path.parent
                break
    store = EvaluationStore(output, root=ROOT) if output is not None else None
    if store is not None and store.metadata is not None:
        rows = store.samples
        plan = store.metadata["data"]
    else:
        if dataset is None:
            if metadata and metadata.get("sft_data_cache") and prepared_sft_data is None:
                from utils.data_preparation import load_reference, recipe
                dataset = load_reference(metadata["sft_data_cache"], recipe("sft", config, tokenizer))
                data_info = metadata["data"]["source"]
            else:
                prepared_sft_data = ensure_prepared_sft_data(data_source, metadata, prepared_sft_data,
                                                            metadata.get("prepared_sft_data") if metadata else None)
                dataset, data_info = (load_prepared_sft_data(prepared_sft_data, config, tokenizer) if prepared_sft_data
                                      else prepare_dataset(config, tokenizer))
        elif data_info is None and metadata is not None:
            data_info = metadata["data"]["source"]
        plan = data_plan(dataset, metadata, data_info)
        indices = plan["validation_indices"]
        if num_samples is not None:
            if num_samples > len(indices):
                raise ValueError(f"num-samples must be between 1 and {len(indices)}; omit it for the full validation set.")
            indices = indices[:num_samples]
        rows = list(dataset.select(indices))
        store = store or EvaluationStore(directory / f"{prefix}{len(rows)}", root=ROOT)
    print(f"Generation progress: {len(store.answers)}/{len(rows)} completed; {store.answers_path}", flush=True)
    batch_mode = "adaptive" if auto_batch_size else "fixed"
    print(f"Generation batch size: {batch_size} ({batch_mode})", flush=True)
    generator = None
    try:
        # A previous interruption may have saved an answer before saving its first decision.
        for answer in store.answers:
            if 1 not in store.judgments.get(answer["index"], {}):
                save_first_stage(store, answer)
        if len(store.answers) < len(rows):
            generator = create_generator(model_name, checkpoint, tokenizer, config, generation_settings, device, model=model)
            if generation_settings["backend"] == "transformers":
                model = generator.model
            print(f"Generation engine: {generator.description}", flush=True)
        store.create(rows, {
            "source_type": source_type, "stage": stage, "model_name": model_name,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "sft_source": str(sft_source) if sft_source is not None else None,
            "prepared_sft_data": str(prepared_sft_data) if prepared_sft_data else None,
            "data": {"source": plan["source"], "split": "validation", "fingerprint": plan["fingerprint"],
                     "split_seed": plan["split_seed"], "requested_num_samples": num_samples},
            "text_format": text_format(tokenizer), "precision": "bf16" if config.get("bf16") else "fp32",
            "generation": {"do_sample": False, "max_new_tokens": config["max_new_tokens"],
                           "eos_token_id": tokenizer.eos_token_id, "batch_size": batch_size,
                           "auto_batch_size": auto_batch_size, "requested_backend": generation_settings["backend"],
                           "engine": generator.description if generator else None},
            "generation_initialization_runtime": generator.initialization_runtime if generator else 0,
            "scoring": SCORING,
        })
        with tqdm(total=len(rows), initial=len(store.answers), desc="Generate validation answers",
                unit="question", disable=disable_tqdm) as progress:
            next_batch_size = batch_size
            last_successful_size = 0
            probing = auto_batch_size
            while len(store.answers) < len(rows):
                start = len(store.answers)
                while True:
                    actual_batch_size = min(next_batch_size, len(rows) - start)
                    batch = store.samples[start:start + actual_batch_size]
                    questions = [row["question"] for row in batch]
                    started = time.perf_counter()
                    initialization_before = generator.initialization_runtime
                    try:
                        if generation_settings["backend"] == "transformers":
                            answers = generate_answers(model, tokenizer, questions, config["max_new_tokens"])
                        else:
                            answers = generator.generate(questions, config["max_new_tokens"])
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
                for row, answer in zip(batch, answers, strict=True):
                    text, count, truncated = answer
                    prediction = {"index": row["index"], "answer": text, "generated_tokens": count,
                                  "truncated": truncated,
                                  "generation_batch_size": actual_batch_size,
                                  "generation_runtime": time_share}
                    store.save_answer(prediction)
                    save_first_stage(store, prediction)
                    progress.update(1)
    finally:
        if generator is not None:
            generator.close()
    if config["data"]["name"] == OPENR1["name"]:
        apply_rule_scoring(store, disable_tqdm=disable_tqdm)
    report = saved_evaluation(store)
    print(format_metrics(report["metrics"], "Generation evaluation"), flush=True)
    print(f"Saved evaluation records to {store.directory}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Qwen/Qwen3-4B-Instruct-2507 for the original model, or a project checkpoint/final directory.")
    evaluation_source = parser.add_mutually_exclusive_group()
    evaluation_source.add_argument("--config", type=Path,
                                   help="Original-model settings; default: configs/training/sft.yaml.")
    evaluation_source.add_argument("--source", type=Path,
                                   help="Anchor original-model evaluation to the exact held-out split of an SFT artifact.")
    parser.add_argument("--num-samples", type=int, help="Evaluate the first N validation questions; default: all.")
    parser.add_argument("--batch-size", type=int,
                        help=f"Maximum questions per call (default: vLLM {DEFAULT_BATCH_SIZES['vllm']}, "
                             f"Transformers {DEFAULT_BATCH_SIZES['transformers']}); starting size in adaptive mode.")
    parser.add_argument("--auto-batch-size", action="store_true",
                        help="Transformers only: double successful batches until the first failure, then retry smaller batches.")
    parser.add_argument("--output", type=Path, help="Evaluation directory; default: run/evaluations for LoRA, outputs/evaluations for original Instruct.")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: CUDA when available, otherwise CPU.")
    add_generation_arguments(parser)
    args = parser.parse_args()
    if args.config is not None and args.checkpoint != MODEL_NAME:
        parser.error("Use --config only with the original Qwen checkpoint.")
    try:
        evaluate_checkpoint(args.checkpoint, num_samples=args.num_samples,
                            output=args.output, device=args.device,
                            config=load_config(args.config) if args.config else None,
                            sft_source=args.source, batch_size=args.batch_size, auto_batch_size=args.auto_batch_size,
                            generation_config=args.generation_config, generation_backend=args.generation_backend)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
