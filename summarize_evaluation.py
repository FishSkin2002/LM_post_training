"""Optionally judge remaining OpenR1 answers, then publish paginated reading reports."""

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from textwrap import indent
import time

from tqdm.auto import tqdm

from utils.answer_judge import APIJudge, DEFAULT_CONFIG, JUDGE_PROMPT, PROMPT_VERSION, load_judge_config
from utils.evaluation_artifacts import EvaluationStore, json_text, write_once
from utils.paths import ROOT

PROTOCOL = "openr1-three-stage-v3"
OPENR1 = "open-r1/OpenR1-Math-220k"


def atomic_write(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as saved:
            temporary = Path(saved.name)
            saved.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_evaluation(path):
    store = EvaluationStore(path, root=ROOT)
    if store.metadata is None or not store.samples or len(store.answers) != len(store.samples):
        raise ValueError("Complete generation and both local rule stages with evaluate.py first.")
    is_openr1 = store.metadata["data"]["source"]["name"] == OPENR1
    for row in store.answers:
        levels = store.judgments.get(row["index"], {})
        if 1 not in levels or (is_openr1 and not row["truncated"]
                               and levels[1]["status"] != "passed" and 2 not in levels):
            raise ValueError("Complete both local rule stages with evaluate.py first.")
    return store


def select_run(evaluation, resume_from):
    if (evaluation is None) == (resume_from is None):
        raise ValueError("Choose exactly one of evaluation or resume-from.")
    if resume_from is not None:
        directory = Path(resume_from).expanduser().resolve()
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        store = load_evaluation(directory / metadata["evaluation"])
        return store, directory, metadata
    store = load_evaluation(evaluation)
    parent = store.directory / "scorings"
    parent.mkdir(exist_ok=True)
    suffix = 0
    while True:
        name = "scoring" + (f"-{suffix}" if suffix else "")
        directory = parent / name
        try:
            directory.mkdir()
            break
        except FileExistsError:
            suffix += 1
    metadata = {"version": 1, "scoring_run": name, "evaluation": "../..", "judge": None}
    write_once(directory / "metadata.json", metadata)
    return store, directory, metadata


def combined_predictions(store, scoring_run):
    api = store.api_results(scoring_run)
    is_openr1 = store.metadata["data"]["source"]["name"] == OPENR1
    rows = []
    for row in store.predictions():
        local = store.judgments[row["index"]]
        stages = {f"stage{i}": {"status": "not_run"} for i in (1, 2, 3)}
        stages["stage1"] = local[1]
        if 2 in local:
            stages["stage2"] = local[2]
        if row["correct"] or row["truncated"] or not is_openr1:
            correct, level, state = row["correct"], 1, "decided"
        elif stages["stage2"]["status"] == "passed":
            correct, level, state = True, 2, "decided"
        elif row["index"] not in api:
            correct, level, state = None, 2, "unprocessed"
        else:
            decision = api[row["index"]]
            stages["stage3"] = decision
            correct = {"passed": True, "failed": False, "pending": None}[decision["status"]]
            level, state = 3, "pending_review" if correct is None else "decided"
        rows.append({**row, "correct": correct, "decision_stage": level, "state": state, "stages": stages})
    return rows


def summarize(store, rows, scoring_run):
    total = len(store.samples)
    counts = {f"stage{i}_correct": sum(row["correct"] is True and row["decision_stage"] == i for row in rows)
              for i in (1, 2, 3)}
    correct = sum(counts.values())
    history = [record for record in store.judgment_records
               if record["stage"] == 3 and record.get("scoring_run") == scoring_run]
    elapsed = sum(row["generation_runtime"] for row in store.answers)
    return {
        **counts, "generation_samples": total, "confirmed_correct": correct, "answer_accuracy": correct / total,
        "stage1_accuracy": counts["stage1_correct"] / total,
        "stage2_accuracy": (counts["stage1_correct"] + counts["stage2_correct"]) / total,
        "stage3_accuracy": correct / total,
        "incorrect": sum(row["correct"] is False for row in rows),
        "pending_review": sum(row["state"] == "pending_review" for row in rows),
        "unprocessed_samples": sum(row["state"] == "unprocessed" for row in rows),
        "processed_samples": sum(row["state"] != "unprocessed" for row in rows),
        "api_samples": sum(row["decision_stage"] == 3 for row in rows),
        "api_calls": sum(len(record.get("attempts", [])) for record in history),
        "api_runtime": sum(attempt["runtime"] for record in history for attempt in record.get("attempts", [])),
        "generation_runtime": elapsed, "generation_samples_per_second": total / max(elapsed, 1e-9),
        "answer_format_rate": sum(store.judgments[row["index"]][1]["format_valid"] for row in rows) / total,
        "generation_truncation_rate": sum(row["truncated"] for row in rows) / total,
    }


def api_key(config):
    key = os.environ.get(config["api_key_env"], "").strip()
    if not key:
        if not sys.stdin.isatty():
            raise ValueError(f"Set {config['api_key_env']} or use an interactive terminal to enter the API key; progress is saved.")
        try:
            key = getpass.getpass(f"{config['api_key_env']}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise ValueError("API key input cancelled; progress is saved.") from None
    if not key or not key.isascii() or any(ord(char) < 32 or ord(char) == 127 for char in key):
        raise ValueError("A nonempty, valid API key is required; progress is saved.")
    return key


def summary_text(metrics):
    total = metrics["generation_samples"]
    first = metrics["stage1_correct"]
    second = first + metrics["stage2_correct"]
    return (f"## Summary\n\nConfirmed accuracy: {metrics['answer_accuracy']:.2%} "
            f"({metrics['confirmed_correct']}/{total})\n\n"
            f"Stage 1 cumulative: {first}/{total} ({metrics['stage1_accuracy']:.2%}) | "
            f"Stage 2 cumulative: {second}/{total} ({metrics['stage2_accuracy']:.2%}) | "
            f"Stage 3 cumulative: {metrics['confirmed_correct']}/{total} ({metrics['stage3_accuracy']:.2%})\n\n"
            f"Incorrect: {metrics['incorrect']} | Pending review: {metrics['pending_review']} | "
            f"Unprocessed: {metrics['unprocessed_samples']}\n\n"
            f"Format valid: {metrics['answer_format_rate']:.2%} | Truncated: {metrics['generation_truncation_rate']:.2%}\n\n"
            f"Generation time: {metrics['generation_runtime']:.3f} s | "
            f"Generation throughput: {metrics['generation_samples_per_second']:.3f} samples/s\n\n"
            f"API cases: {metrics['api_samples']} | API calls: {metrics['api_calls']} | "
            f"API time: {metrics['api_runtime']:.3f} s\n\n"
            "[API decided](api-decided.md) | [API pending review](api-pending.md)\n\n")


def readable_row(row):
    result = str(row["correct"]).lower() if row["correct"] is not None else row["state"]
    stages = " | ".join(f"Stage {i}: {row['stages'][f'stage{i}']['status']}" for i in (1, 2, 3))
    lines = [f'<a id="sample-{row["index"]}"></a>', f"## Question {row['index'] + 1}",
             f"Correct: {result}", stages]
    for label, key in (("Question", "question"), ("Reference final answer", "reference_final_answer"),
                       ("R1 reference solution", "reference_answer"), ("Model answer", "answer")):
        lines.extend((f"### {label}", indent(row[key], "    ")))
    return "\n\n".join(lines) + "\n\n---\n\n"


def review_text(rows, pending, questions_per_file):
    title = "API pending review" if pending else "API decided"
    entries = []
    for position, row in enumerate(rows):
        if row["decision_stage"] != 3 or (row["state"] == "pending_review") != pending:
            continue
        page = position // questions_per_file + 1
        outcome = "pending" if pending else "correct" if row["correct"] else "incorrect"
        entries.append(f"## Sample {row['index'] + 1}\n\nDecision: {outcome}\n\n"
                       + indent(row["stages"]["stage3"]["reason"], "    ") + "\n\n"
                       + f"[Question and answers](answers-{page:04d}.md#sample-{row['index']})\n\n")
    return f"# {title}\n\n" + ("".join(entries) if entries else "None.\n")


def render_reports(directory, report, questions_per_file):
    """Prepare all pages before replacing the shared reading view, rolling back on failure."""
    with tempfile.TemporaryDirectory(prefix=".reading-", dir=directory) as temporary:
        staging = Path(temporary)
        prepared = staging / "prepared"
        backup = staging / "previous"
        prepared.mkdir()
        backup.mkdir()
        rows = report["predictions"]
        for start in range(0, len(rows), questions_per_file):
            page = start // questions_per_file + 1
            content = f"# Answers {page}\n\n"
            if start == 0:
                content += summary_text(report["metrics"])
            content += "".join(readable_row(row) for row in rows[start:start + questions_per_file])
            (prepared / f"answers-{page:04d}.md").write_text(content, encoding="utf-8")
        for pending, name in ((False, "api-decided.md"), (True, "api-pending.md")):
            (prepared / name).write_text(review_text(rows, pending, questions_per_file), encoding="utf-8")
        previous = [path for path in directory.iterdir()
                    if path.is_file() and (re.fullmatch(r"answers-\d{4,}\.md", path.name)
                                           or path.name in ("api-decided.md", "api-pending.md"))]
        published = []
        try:
            for path in previous:
                path.replace(backup / path.name)
            for path in prepared.iterdir():
                target = directory / path.name
                path.replace(target)
                published.append(target)
        except BaseException:
            for path in published:
                path.unlink()
            for path in backup.iterdir():
                path.replace(directory / path.name)
            raise


def summarize_evaluation(evaluation=None, *, resume_from=None, config_path=DEFAULT_CONFIG,
                         no_api=False, retry_pending=False, questions_per_file=100, disable_tqdm=False):
    if type(questions_per_file) is not int or questions_per_file < 1:
        raise ValueError("questions-per-file must be a positive integer.")
    store, directory, metadata = select_run(evaluation, resume_from)
    run = metadata["scoring_run"]
    rows = combined_predictions(store, run)
    work = [row for row in rows if row["state"] == "unprocessed"
            or (retry_pending and row["state"] == "pending_review")]
    print(f"Evaluation: {store.directory}\nScoring run: {directory}", flush=True)
    if not no_api and work:
        if metadata["judge"] is None:
            metadata.update(judge=load_judge_config(config_path), protocol=PROTOCOL,
                            prompt_version=PROMPT_VERSION, prompt=JUDGE_PROMPT)
            atomic_write(directory / "metadata.json", json_text(metadata))
        judge = APIJudge(metadata["judge"], api_key=api_key(metadata["judge"]), prompt=metadata["prompt"])
        with tqdm(work, desc="Judge saved answers (stage 3)", unit="question", disable=disable_tqdm) as progress:
            for row in progress:
                started = time.perf_counter()
                decision = judge(row)
                store.save_judgment(row["index"], 3,
                                    {**decision, "scoring_run": run, "runtime": time.perf_counter() - started})
    rows = combined_predictions(store, run)
    report = {"evaluation": str(store.directory), "scoring_run": str(directory),
              "metrics": summarize(store, rows, run), "predictions": rows}
    render_reports(store.directory, report, questions_per_file)
    metrics = report["metrics"]
    print(f"Confirmed accuracy: {metrics['answer_accuracy']:.2%} "
          f"({metrics['confirmed_correct']}/{metrics['generation_samples']}) | "
          f"Pending review: {metrics['pending_review']} | Unprocessed: {metrics['unprocessed_samples']}", flush=True)
    print(f"Readable answers: {store.directory / 'answers-0001.md'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evaluation", type=Path, help="Completed local evaluation directory; create a new scoring run.")
    source.add_argument("--resume", type=Path, help="Existing scoring run; reuse saved source, judge and decisions.")
    parser.add_argument("--no-api", action="store_true", help="Summarize saved decisions only; do not request API or credentials.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Judge YAML used when first enabling API scoring.")
    parser.add_argument("--retry-pending", action="store_true", help="With API enabled, retry saved pending decisions once.")
    parser.add_argument("--questions-per-file", type=int, default=100, help="Maximum questions per reading volume (default: 100).")
    args = parser.parse_args()
    try:
        summarize_evaluation(args.evaluation, resume_from=args.resume, config_path=args.config,
                             no_api=args.no_api, retry_pending=args.retry_pending,
                             questions_per_file=args.questions_per_file)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
