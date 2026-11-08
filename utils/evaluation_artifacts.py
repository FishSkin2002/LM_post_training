"""Shared frozen samples and append-only machine records for an evaluation."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from utils.paths import ROOT


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def write_once(path, value):
    """Publish a complete JSON file without replacing an existing snapshot/config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as saved:
            temporary = Path(saved.name)
            saved.write(json_text(value))
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink()


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as saved:
        return [json.loads(line) for line in saved]


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as saved:
        saved.write(json_text(value))
        saved.flush()


class EvaluationStore:
    def __init__(self, output, *, root=ROOT):
        self.root = Path(root).resolve()
        output = Path(output).expanduser().resolve()
        if output.is_file():
            raise ValueError("Existing evaluation files are preserved; choose a new output directory.")
        # Accept the former --output basename, but write the new format in a directory.
        self.directory = output.with_suffix("") if output.suffix == ".json" else output
        self.metadata_path = self.directory / "metadata.json"
        self.answers_path = self.directory / "answers.jsonl"
        self.judgments_path = self.directory / "judgments.jsonl"
        self.metadata = None
        self.samples = None
        if not self.metadata_path.exists() and any(
                self.directory.with_name(self.directory.name + suffix).is_file()
                for suffix in (".json", ".progress.jsonl")):
            raise ValueError("Legacy evaluation artifacts exist; choose a new output directory.")
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.snapshot_path = self.root / self.metadata["snapshot"]
            if not self.snapshot_path.is_file():
                raise ValueError(f"Saved evaluation snapshot is missing: {self.snapshot_path}")
            self.samples = json.loads(self.snapshot_path.read_text(encoding="utf-8"))["samples"]
        elif self.answers_path.exists() or self.judgments_path.exists():
            raise ValueError("Evaluation records exist without metadata; choose a new output directory.")
        self.answers = read_jsonl(self.answers_path)
        self.judgments = {}
        for record in read_jsonl(self.judgments_path):
            self.judgments.setdefault(record["index"], {})[record["stage"]] = record

    def create(self, rows, metadata):
        if self.metadata is not None:
            return
        samples = [
            {"index": position, "question": row["question"],
             "reference_answer": row["answer"], "reference_final_answer": row["final_answer"],
             "source_index": row["source_index"], "generation_index": row["generation_index"]}
            for position, row in enumerate(rows)
        ]
        snapshot = {"version": 1, "samples": samples}
        identity = hashlib.sha256(json_text(snapshot).encode("utf-8")).hexdigest()
        self.snapshot_path = self.root / "outputs" / "evaluation-datasets" / f"{identity}.json"
        write_once(self.snapshot_path, snapshot)
        reference = self.snapshot_path.relative_to(self.root).as_posix()
        write_once(self.metadata_path, {"version": 1, "snapshot": reference, **metadata})
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.samples = samples

    def save_answer(self, answer):
        append_jsonl(self.answers_path, answer)
        self.answers.append(answer)

    def save_judgment(self, index, stage, result):
        record = {"index": index, "stage": stage, **result}
        append_jsonl(self.judgments_path, record)
        self.judgments.setdefault(index, {})[stage] = record

    def predictions(self):
        """Join saved facts in memory; this view is never persisted as another report."""
        samples = {row["index"]: row for row in self.samples}
        values = []
        for answer in self.answers:
            first = self.judgments.get(answer["index"], {}).get(1)
            row = {**samples[answer["index"]], **answer}
            row.setdefault("generation_batch_size", 1)
            if first is not None:
                row.update(correct=first["status"] == "passed", format_valid=first["format_valid"])
            values.append(row)
        return values
