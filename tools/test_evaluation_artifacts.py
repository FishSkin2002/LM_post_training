"""Offline checks for shared sample publication and evaluation storage boundaries."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.evaluation_artifacts import EvaluationStore


class EvaluationArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = [{"question": "A question\nwith original whitespace", "answer": r"Derivation: \boxed{1/4}",
                      "final_answer": "0.25", "source_index": 207, "generation_index": 2}]

    def create(self, name, rows=None):
        store = EvaluationStore(self.root / name, root=self.root)
        store.create(self.rows if rows is None else rows, {"model_name": name, "generation": {"engine": name}})
        return store

    def test_models_share_a_snapshot_without_duplicating_references_or_configuration(self):
        first = self.create("training/sft/evaluation")
        original = first.snapshot_path.read_bytes()
        metadata = first.metadata_path.read_bytes()
        rows = [{**self.rows[0], "input_ids": [1, 2, 3]}]
        second = self.create("evaluations/original", rows)
        self.assertEqual(first.snapshot_path, second.snapshot_path)
        self.assertEqual(first.snapshot_path.read_bytes(), original)
        self.assertEqual(len(list((self.root / "outputs/evaluation-datasets").iterdir())), 1)
        self.assertFalse(list(first.directory.glob("*snapshot*")))
        self.assertFalse(Path(first.metadata["snapshot"]).is_absolute())
        first.create(rows, {"model_name": "changed"})
        self.assertEqual(first.metadata_path.read_bytes(), metadata)
        self.assertEqual(first.samples[0]["source_index"], 207)
        self.assertEqual(first.samples[0]["reference_answer"], self.rows[0]["answer"])
        self.assertNotIn("input_ids", first.samples[0])

    def test_changed_reference_or_sample_order_creates_a_new_snapshot(self):
        first = self.create("first")
        changed = deepcopy(self.rows)
        changed[0]["answer"] = "A different selected reference"
        second = self.create("second", changed)
        third = self.create("third", self.rows + changed)
        fourth = self.create("fourth", changed + self.rows)
        self.assertEqual(len({store.snapshot_path for store in (first, second, third, fourth)}), 4)
        self.assertEqual(first.samples[0]["reference_answer"], self.rows[0]["answer"])

    def test_parallel_creators_publish_only_one_complete_snapshot(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            stores = list(executor.map(self.create, ("one", "two")))
        self.assertEqual(stores[0].snapshot_path, stores[1].snapshot_path)
        self.assertEqual(len(list(stores[0].snapshot_path.parent.iterdir())), 1)
        self.assertEqual(json.loads(stores[0].snapshot_path.read_text())["samples"], stores[1].samples)

    def test_snapshot_location_is_independent_of_working_directory_and_missing_snapshot_fails(self):
        location = Path.cwd()
        self.addCleanup(os.chdir, location)
        os.chdir(self.root)
        store = self.create("external-training/evaluation")
        os.chdir(self.root / "outputs")
        loaded = EvaluationStore(store.directory, root=self.root)
        self.assertEqual(loaded.snapshot_path, store.snapshot_path)
        self.assertEqual(loaded.samples, store.samples)
        store.snapshot_path.unlink()
        with self.assertRaisesRegex(ValueError, "snapshot is missing"):
            EvaluationStore(store.directory, root=self.root)

    def test_existing_legacy_report_is_preserved_and_orphan_records_are_not_repaired(self):
        old = self.root / "legacy.json"
        old.write_text('{"predictions": []}\n')
        content = old.read_bytes()
        with self.assertRaisesRegex(ValueError, "choose a new output directory"):
            EvaluationStore(old, root=self.root)
        with self.assertRaisesRegex(ValueError, "Legacy evaluation artifacts"):
            EvaluationStore(old.with_suffix(""), root=self.root)
        self.assertEqual(old.read_bytes(), content)
        progress = self.root / "partial.progress.jsonl"
        progress.write_text('{"index":207,"answer":"saved"}\n')
        with self.assertRaisesRegex(ValueError, "Legacy evaluation artifacts"):
            EvaluationStore(self.root / "partial.json", root=self.root)
        self.assertFalse((self.root / "partial").exists())
        orphan = self.root / "orphan"
        orphan.mkdir()
        (orphan / "answers.jsonl").write_text('{"index":0}\n')
        with self.assertRaisesRegex(ValueError, "without metadata"):
            EvaluationStore(orphan, root=self.root)


if __name__ == "__main__":
    unittest.main()
