"""One entry point for natural math records; no chat formatting or model code."""

from collections import Counter
import hashlib
import re
import sys

from datasets import load_dataset

from utils.math_scoring import boxed_answer, parse_answer

# datasets serializes callable module names into existing data fingerprints.
# Keep that identity while the public import and source live under utils/.
sys.modules["math_data"] = sys.modules[__name__]
__name__ = "math_data"

OPENR1 = {"name": "open-r1/OpenR1-Math-220k", "config": "default", "split": "train", "revision": "main"}
GSM8K = {"name": "openai/gsm8k", "config": "main", "split": "train", "revision": "main"}
CALCULATOR_ANNOTATION = re.compile(r"<<[^<>]*>>")


def dataset_label(source):
    return "-".join((source["name"].rsplit("/", 1)[-1], source["config"], source["split"]))


def clean_answer(answer):
    return CALCULATOR_ANNOTATION.sub("", answer)


def _record(question, answer, final_answer, index, generation_index=-1, reason=""):
    return {"question": question, "answer": answer, "final_answer": final_answer,
            "group_id": hashlib.sha256(question.encode()).hexdigest(),
            "source_index": index, "generation_index": generation_index, "filter_reason": reason}


def _openr1(row, index):
    question, reference = row["problem"], row["answer"]
    fields = ("generations", "correctness_math_verify", "correctness_llama", "is_reasoning_complete")
    if any(field not in row for field in fields) or not isinstance(row["generations"], list):
        raise ValueError(f"OpenR1 row {index} has missing or invalid generation annotations.")
    count = len(row["generations"])
    annotations = []
    for field in fields[1:]:
        values = row[field]
        # The published default split uses null when a judge was not applied.
        if values is None and field != "is_reasoning_complete":
            values = [None] * count
        if not isinstance(values, list):
            raise ValueError(f"OpenR1 row {index} has invalid {field} annotations.")
        if len(values) != count:
            raise ValueError(f"OpenR1 row {index} has misaligned generation annotations.")
        if any(value is not None and type(value) is not bool for value in values):
            raise ValueError(f"OpenR1 row {index} has non-boolean annotations.")
        annotations.append(values)
    if not parse_answer(reference):
        return _record(question, "", reference, index, reason="unsupported_reference")
    for position, (answer, math_ok, llama_ok, complete) in enumerate(zip(row["generations"], *annotations)):
        if (math_ok is True or llama_ok is True) and complete is True and isinstance(answer, str):
            final = boxed_answer(answer)
            if final is not None and parse_answer(final):
                return _record(question, answer, reference, index, position)
    return _record(question, "", reference, index, reason="no_eligible_generation")


def _gsm8k(row, index):
    answer = clean_answer(row["answer"])
    reference = row["answer"].rsplit("####", 1)[-1].strip() if row["answer"].count("####") == 1 else ""
    reason = "" if parse_answer(reference) else "unsupported_reference"
    return _record(row["question"], answer, reference, index, reason=reason)


def load_math_data(source=None):
    source = dict(source or OPENR1)
    if source["name"] not in (OPENR1["name"], GSM8K["name"]):
        raise ValueError("Supported datasets: OpenR1-Math-220k and GSM8K.")
    if source["name"] == OPENR1["name"] and (source["config"], source["split"]) != ("default", "train"):
        raise ValueError("OpenR1 currently uses default/train.")
    raw = load_dataset(source["name"], source["config"], split=source["split"], revision=source["revision"])
    convert = _openr1 if source["name"] == OPENR1["name"] else _gsm8k
    normalized = raw.map(convert, with_indices=True, remove_columns=raw.column_names, desc="Normalize math records")
    reasons = Counter(normalized["filter_reason"])
    info = {**source, "raw_fingerprint": raw._fingerprint, "raw_records": len(raw),
            "filter_counts": {key: value for key, value in sorted(reasons.items()) if key},
            "selection": ("first correct complete parseable boxed generation in source order"
                          if convert is _openr1 else "original GSM8K solution without calculator annotations")}
    return normalized.filter(lambda reason: not reason, input_columns="filter_reason", desc="Filter math records"), info
