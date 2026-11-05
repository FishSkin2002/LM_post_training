"""Rescore a completed OpenR1 evaluation JSON without generating answers."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from textwrap import indent
import time

from tqdm.auto import tqdm

from answer_judge import APIJudge, DEFAULT_CONFIG, JUDGE_PROMPT, PROMPT_VERSION, decode_json, load_judge_config
from math_scoring import SCORING, boxed_answer, parse_answer, score_answer, verify

PROTOCOL = "openr1-three-stage-v1"


def load_evaluation(path):
    raw = path.read_bytes()
    try:
        report = decode_json(raw)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Input must be a completed evaluation JSON report, not a progress JSONL file.") from error
    if not isinstance(report, dict) or "rescoring" in report:
        raise ValueError("Input must be an original generation evaluation report.")
    rows = report.get("predictions")
    data = report.get("data", {})
    if (not isinstance(rows, list) or not rows or not isinstance(data, dict)
            or not isinstance(data.get("source"), dict)
            or data["source"].get("name") != "open-r1/OpenR1-Math-220k"):
        raise ValueError("Input must contain nonempty OpenR1 predictions and data metadata.")
    indices = data.get("indices")
    if (not isinstance(indices, list) or len(indices) != len(rows)
            or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Input evaluation indices must be unique and match the prediction count.")
    metrics = report.get("metrics")
    if (not isinstance(metrics, dict) or type(metrics.get("generation_samples")) is not int
            or metrics["generation_samples"] != len(rows)):
        raise ValueError("Input generation_samples must match all saved predictions.")
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("index") != indices[position]:
            raise ValueError(f"Prediction {position} does not match the evaluation indices.")
        for key in ("index", "source_index", "generation_index"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"Prediction {position} has an invalid {key}.")
        for key in ("question", "reference_final_answer", "reference_answer", "answer"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"Prediction {position} requires text field {key}.")
        if not row["question"].strip():
            raise ValueError(f"Prediction {position} has an empty question.")
        for key in ("correct", "format_valid", "truncated"):
            if type(row.get(key)) is not bool:
                raise ValueError(f"Prediction {position} requires boolean field {key}.")
    return report, hashlib.sha256(raw).hexdigest()


def single_formal_box(text):
    formal = text.rsplit("</think>", 1)[-1]
    count = len(re.findall(r"\\boxed\b", formal))
    if count != 1:
        return None, "missing_box" if count == 0 else "multiple_boxes"
    answer = boxed_answer(formal)
    if answer is None:
        return None, "empty_or_incomplete_box"
    if not parse_answer(answer):
        return None, "unparseable_box"
    return answer, None


def grade_rules(row):
    stages = {f"stage{i}": {"status": "not_run"} for i in (1, 2, 3)}
    result = {"correct": None, "decision_stage": 1, "stages": stages}
    if row["truncated"]:
        stages["stage1"] = {"status": "failed", "reason": "truncated"}
        result["correct"] = False
        return result
    try:
        correct, valid, _ = score_answer(row["answer"], row["reference_final_answer"])
        reason = "equivalent" if correct else "different_from_reference" if valid else "invalid_model_box"
    except ValueError:
        correct, reason = False, "unsupported_reference"
    stages["stage1"] = {"status": "passed" if correct else "failed", "reason": reason}
    if correct:
        result["correct"] = True
        return result
    result["decision_stage"] = 2
    expected, reference_error = single_formal_box(row["reference_answer"])
    predicted, model_error = single_formal_box(row["answer"])
    stage2 = {"status": "failed", "reference_box": expected, "model_box": predicted}
    stages["stage2"] = stage2
    if reference_error or model_error:
        stage2["reason"] = "; ".join(
            f"{side}_{error}" for side, error in (("r1", reference_error), ("model", model_error)) if error)
    elif verify(parse_answer(expected), parse_answer(predicted), **SCORING["verification"]):
        stage2.update(status="passed", reason="equivalent")
        result["correct"] = True
    else:
        stage2["reason"] = "different_from_r1"
    return result


def grade_prediction(row, judge, previous=None):
    started = time.perf_counter()
    result = grade_rules(row)
    if result["correct"] is None:
        api = judge(row)
        if previous:
            api["attempts"] = previous["stages"]["stage3"].get("attempts", []) + api["attempts"]
        result["stages"]["stage3"] = api
        result["decision_stage"] = 3
        result["correct"] = {"passed": True, "failed": False, "pending": None}[api["status"]]
    result["rescoring_runtime"] = (
        time.perf_counter() - started + (previous["rescoring_runtime"] if previous else 0))
    return result


def summarize(results, total):
    values = list(results.values())
    correct = sum(row["correct"] is True for row in values)
    api = [row["stages"]["stage3"] for row in values if row["decision_stage"] == 3]
    return {
        "answer_accuracy": correct / total, "confirmed_correct": correct,
        "incorrect": sum(row["correct"] is False for row in values),
        "pending_review": sum(row["correct"] is None for row in values),
        "generation_samples": total, "processed_samples": len(values),
        "unprocessed_samples": total - len(values),
        **{f"stage{i}_correct": sum(row["correct"] is True and row["decision_stage"] == i for row in values)
           for i in (1, 2, 3)},
        "api_samples": len(api), "api_calls": sum(len(row.get("attempts", [])) for row in api),
        "rescoring_runtime": sum(row["rescoring_runtime"] for row in values),
        "api_runtime": sum(attempt["runtime"] for row in api for attempt in row.get("attempts", [])),
    }


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def readable_row(row):
    lines = [f"## Dataset index {row['index']}",
             f"Correct: {row['correct']} | Decision stage: {row['decision_stage']} | Truncated: {row['truncated']}",
             f"Source index: {row['source_index']} | Generation index: {row['generation_index']}"]
    for name, stage in row["stages"].items():
        lines.append(f"{name}: {stage['status']} | {stage.get('reason', '')}")
    for label, key in (("Question", "question"), ("Reference final answer", "reference_final_answer"),
                       ("R1 reference solution", "reference_answer"), ("Model answer", "answer")):
        lines.extend((f"### {label}", indent(row[key], "    ")))
    api = row["stages"]["stage3"]
    if api["status"] != "not_run":
        lines.extend(("### API audit",
                      indent(json.dumps(api, ensure_ascii=False, indent=2), "    ")))
    return "\n\n".join(lines) + "\n\n---\n\n"


def save_reports(source, results, metadata, output):
    metrics = summarize(results, len(source["predictions"]))
    predictions = []
    for row in source["predictions"]:
        decision = results.get(row["index"], {
            "correct": None, "decision_stage": None,
            "stages": {f"stage{i}": {"status": "not_run"} for i in (1, 2, 3)},
            "rescoring_runtime": 0,
        })
        predictions.append({**row, "original_correct": row["correct"], **decision})
    pending = [row["index"] for row in predictions
               if row["decision_stage"] == 3 and row["correct"] is None]
    report = {**source, "original_metrics": source["metrics"], "original_scoring": source.get("scoring"),
              "scoring": metadata["protocol"], "rescoring": metadata,
              "metrics": {**source["metrics"], **metrics}, "pending_review_indices": pending,
              "predictions": predictions}
    atomic_write(output, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    header = (f"# Answer rescoring\n\nConfirmed accuracy: {metrics['answer_accuracy']:.2%} "
              f"({metrics['confirmed_correct']}/{metrics['generation_samples']})\n\n"
              f"Pending review: {metrics['pending_review']} | Unprocessed: {metrics['unprocessed_samples']}\n\n"
              f"Pending dataset indices: {', '.join(map(str, pending)) or 'none'}\n\n")
    atomic_write(output.with_suffix(".md"), header + "".join(readable_row(row) for row in predictions))
    audit = header + "# API review\n\n"
    for title, verdict in (("API correct", True), ("API incorrect", False), ("Pending review", None)):
        selected = [row for row in predictions if row["decision_stage"] == 3 and row["correct"] is verdict]
        audit += f"## {title} ({len(selected)})\n\n"
        audit += "".join(readable_row(row) for row in selected) or "None.\n\n"
    atomic_write(output.with_suffix(".api-review.md"), audit)
    return report


def load_progress(path, metadata, indices):
    if not path.exists():
        return {}
    results = {}
    with path.open(encoding="utf-8") as saved:
        try:
            header = decode_json(next(saved))
            if header != {"type": "run", **metadata}:
                raise ValueError("Rescoring input, judge configuration, prompt, or protocol differs; use a new output.")
            for line in saved:
                entry = decode_json(line)
                index, result = entry["index"], entry["result"]
                if (entry.get("type") != "decision" or index not in indices or type(index) is not int
                        or type(result["decision_stage"]) is not int or result["decision_stage"] not in (1, 2, 3)
                        or (result["correct"] is not None and type(result["correct"]) is not bool)):
                    raise ValueError("Invalid rescoring progress record.")
                runtime = result["rescoring_runtime"]
                if type(runtime) not in (int, float) or not math.isfinite(runtime) or runtime < 0:
                    raise ValueError("Invalid rescoring runtime.")
                stages = result["stages"]
                if (not isinstance(stages, dict) or set(stages) != {"stage1", "stage2", "stage3"}
                        or any(not isinstance(stage, dict) for stage in stages.values())):
                    raise ValueError("Invalid rescoring stages.")
                expected = {
                    1: ("passed" if result["correct"] else "failed", "not_run", "not_run"),
                    2: ("failed", "passed", "not_run"),
                    3: ("failed", "failed", "pending" if result["correct"] is None
                        else "passed" if result["correct"] else "failed"),
                }[result["decision_stage"]]
                if (tuple(stages[f"stage{i}"]["status"] for i in (1, 2, 3)) != expected
                        or (result["decision_stage"] in (1, 2) and result["correct"] is None)
                        or (result["decision_stage"] == 2 and result["correct"] is not True)):
                    raise ValueError("Inconsistent rescoring stage results.")
                attempts = stages["stage3"].get("attempts", [])
                if not isinstance(attempts, list) or any(
                        not isinstance(attempt, dict) or type(attempt.get("runtime")) not in (int, float)
                        or not math.isfinite(attempt["runtime"]) or attempt["runtime"] < 0 for attempt in attempts):
                    raise ValueError("Invalid saved API attempts.")
                if index in results and results[index]["correct"] is not None:
                    raise ValueError("Duplicate completed rescoring decision.")
                results[index] = result
        except (StopIteration, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid rescoring progress file; the original generation file is not a rescoring ledger.") from error
    return results


def rescore_evaluation(input_path, *, output=None, config_path=DEFAULT_CONFIG,
                       retry_pending=False, disable_tqdm=False):
    input_path = Path(input_path).expanduser().resolve()
    output = (Path(output).expanduser().resolve() if output else
              input_path.with_name(input_path.stem + "-rescored.json"))
    artifacts = [output, output.with_suffix(".md"), output.with_suffix(".api-review.md"),
                 output.with_suffix(".progress.jsonl")]
    if input_path in artifacts or output.suffix != ".json":
        raise ValueError("Use a new .json output path; the input report is preserved.")
    source, digest = load_evaluation(input_path)
    config = load_judge_config(config_path)
    metadata = {"input_sha256": digest,
                "protocol": {"name": PROTOCOL, "math_scoring": SCORING,
                             "prompt_version": PROMPT_VERSION,
                             "prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()},
                "judge": config}
    progress_path = output.with_suffix(".progress.jsonl")
    results = load_progress(progress_path, metadata, set(source["data"]["indices"]))
    if not progress_path.exists() and any(path.exists() for path in artifacts):
        raise ValueError("Output artifacts exist without their rescoring progress file; use a new output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    judge = APIJudge(config)
    rows = source["predictions"]
    work = [row for row in rows if row["index"] not in results or
            (retry_pending and results[row["index"]]["correct"] is None)]
    print(f"Rescoring input: {input_path}\nReport: {output}\nAPI review: {output.with_suffix('.api-review.md')}", flush=True)
    try:
        with progress_path.open("a", encoding="utf-8") as saved:
            if saved.tell() == 0:
                saved.write(json.dumps({"type": "run", **metadata}, ensure_ascii=False, allow_nan=False) + "\n")
                saved.flush()
            counts = summarize(results, len(rows))
            confirmed, pending = counts["confirmed_correct"], counts["pending_review"]
            with tqdm(work, desc="Rescore saved answers", unit="question", disable=disable_tqdm) as progress:
                for row in progress:
                    index = row["index"]
                    previous = results.get(index)
                    decision = grade_prediction(row, judge, previous)
                    saved.write(json.dumps({"type": "decision", "index": index, "result": decision},
                                           ensure_ascii=False, allow_nan=False) + "\n")
                    saved.flush()
                    results[index] = decision
                    confirmed += (decision["correct"] is True) - (previous is not None and previous["correct"] is True)
                    pending += (decision["correct"] is None) - (previous is not None and previous["correct"] is None)
                    progress.set_postfix_str(f"correct={confirmed}/{len(rows)}, pending={pending}", refresh=False)
    finally:
        report = save_reports(source, results, metadata, output)
        metrics = report["metrics"]
        print(f"Confirmed accuracy: {metrics['answer_accuracy']:.2%} "
              f"({metrics['confirmed_correct']}/{len(rows)}) | Pending review: {metrics['pending_review']} "
              f"| Unprocessed: {metrics['unprocessed_samples']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Completed OpenR1 generation evaluation JSON.")
    parser.add_argument("--output", type=Path, help="New JSON output; default: INPUT-rescored.json.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Judge YAML; default: configs/evaluation/glm-5.3.yaml.")
    parser.add_argument("--retry-pending", action="store_true", help="Retry saved pending judgments once in this run.")
    args = parser.parse_args()
    try:
        rescore_evaluation(args.input, output=args.output, config_path=args.config, retry_pending=args.retry_pending)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
