"""Saved first-level decisions and local R1 comparison, without API or model imports."""

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
    stage2 = grade_second_stage(row)
    stages["stage2"] = stage2
    if stage2["status"] == "passed":
        result["correct"] = True
    return result


def grade_second_stage(row):
    expected, reference_error = single_formal_box(row["reference_answer"])
    predicted, model_error = single_formal_box(row["answer"])
    stage2 = {"status": "failed", "reference_box": expected, "model_box": predicted}
    if reference_error or model_error:
        stage2["reason"] = "; ".join(
            f"{side}_{error}" for side, error in (("r1", reference_error), ("model", model_error)) if error)
    elif verify(parse_answer(expected), parse_answer(predicted), **SCORING["verification"]):
        stage2.update(status="passed", reason="equivalent")
    else:
        stage2["reason"] = "different_from_r1"
    return stage2


def saved_rule_result(row, judgments):
    """Reconstruct the legacy in-memory result without re-running either rule."""
    stages = {f"stage{i}": {"status": "not_run"} for i in (1, 2, 3)}
    for level in (1, 2):
        if level in judgments:
            stages[f"stage{level}"] = {key: value for key, value in judgments[level].items()
                                      if key not in ("index", "stage", "runtime", "format_valid")}
    terminal = row["correct"] or row["truncated"]
    correct = row["correct"] if terminal else True if stages["stage2"]["status"] == "passed" else None
    return {"correct": correct, "decision_stage": 1 if terminal else 2, "stages": stages,
            "rule_runtime": judgments.get(2, {}).get("runtime", 0)}


def apply_rule_scoring(store, *, disable_tqdm=False):
    """Append only outstanding second-level decisions after every first-level decision."""
    rows = [row for row in store.predictions() if not row["correct"] and not row["truncated"]]
    work = [row for row in rows if 2 not in store.judgments.get(row["index"], {})]
    print(f"Rule scoring progress: {len(rows) - len(work)}/{len(rows)} completed; {store.judgments_path}", flush=True)
    with tqdm(work, desc="Grade validation answers (stage 2)", unit="question", disable=disable_tqdm) as progress:
        for row in progress:
            started = time.perf_counter()
            result = grade_second_stage(row)
            store.save_judgment(row["index"], 2, {**result, "runtime": time.perf_counter() - started})
