"""Saved first-level decisions and local R1 comparison, without API or model imports."""

import json
from pathlib import Path
import re
import time

from tqdm.auto import tqdm

from utils.math_scoring import SCORING, boxed_answer, parse_answer, verify

PROTOCOL = "openr1-two-stage-v1"


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
    # Generation has already executed stage one. Use its saved decision.
    correct = row["correct"]
    reason = ("saved_reference_pass" if correct else
              "saved_reference_failure" if row["format_valid"] else "saved_invalid_model_box")
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


def readable_rule_result(row, result):
    lines = [f"## Dataset index {row['index']}",
             f"Correct: {result['correct']} | Decision stage: {result['decision_stage']}"]
    if result["correct"] is None:
        lines.append("Awaiting independent stage 3 scoring.")
    for name, stage in result["stages"].items():
        lines.append(f"{name}: {stage['status']} | {stage.get('reason', '')}")
    stage2 = result["stages"]["stage2"]
    if stage2["status"] != "not_run":
        lines.extend((f"R1 box: {stage2['reference_box']}", f"Model box: {stage2['model_box']}"))
    return "\n\n".join(lines) + "\n\n---\n\n"


def apply_rule_scoring(report, output, *, disable_tqdm=False):
    """Append local decisions after generation, restoring only saved indices/results."""
    output = Path(output)
    progress_path = output.with_suffix(".stage2.progress.jsonl")
    readable_path = output.with_suffix(".stage2.md")
    results = {}
    if progress_path.is_file():
        with progress_path.open(encoding="utf-8") as saved:
            for line in saved:
                entry = json.loads(line)
                results[entry["index"]] = entry["result"]
    predictions = report["predictions"]
    work = [row for row in predictions if row["index"] not in results]
    completed = len(predictions) - len(work)
    print(f"Rule scoring progress: {completed}/{len(predictions)} completed; {progress_path}", flush=True)
    try:
        if work:
            with progress_path.open("a", encoding="utf-8") as saved, \
                    readable_path.open("a", encoding="utf-8") as readable, tqdm(
                    total=len(predictions), initial=completed, desc="Grade validation answers (stage 2)",
                    unit="question", disable=disable_tqdm) as progress:
                if readable.tell() == 0:
                    readable.write("# Local answer rules (stage 2)\n\n")
                    readable.flush()
                for row in work:
                    started = time.perf_counter()
                    result = grade_rules(row)
                    result["rule_runtime"] = time.perf_counter() - started
                    saved.write(json.dumps({"index": row["index"], "result": result}, ensure_ascii=False) + "\n")
                    saved.flush()
                    results[row["index"]] = result
                    readable.write(readable_rule_result(row, result))
                    readable.flush()
                    progress.update(1)
    finally:
        completed = sum(row["index"] in results for row in predictions)
        report["rule_scoring"] = {"protocol": PROTOCOL, "completed_samples": completed,
                                  "total_samples": len(predictions), "complete": completed == len(predictions),
                                  "progress_file": str(progress_path), "readable_file": str(readable_path)}
        report["predictions"] = [
            {**row, "rule_scoring": results[row["index"]]} if row["index"] in results else row
            for row in predictions
        ]
