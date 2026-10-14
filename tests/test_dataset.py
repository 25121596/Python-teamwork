"""Synthetic checks for checksum-verified split dataset loading."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spam_classifier.dataset import load_dataset


def load_split_fixture_module():
    path = REPOSITORY / "tests" / "test_split_pipeline.py"
    spec = importlib.util.spec_from_file_location("dataset_test_split_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split_fixture = load_split_fixture_module()


class DatasetLoaderTests(unittest.TestCase):
    def prepare_fixture(self):
        # Reuse the existing all-synthetic corpus and cleaning setup. Each test
        # receives a fresh fixture so that tampering remains isolated.
        fixture = split_fixture.SplitPipelineTests(
            "test_pipeline_accounts_duplicates_and_public_outputs_are_metadata_only"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)

        checker_source = REPOSITORY / "scripts" / "check_splits.py"
        (fixture.root / "scripts" / "check_splits.py").write_bytes(checker_source.read_bytes())
        self.assertEqual(fixture.run_main(split_fixture.splitter), 0)
        self.assertEqual(fixture.run_main(split_fixture.split_checker), 0)
        return fixture

    def test_loaded_records_align_with_frozen_split_ids(self):
        fixture = self.prepare_fixture()

        bundle = load_dataset(fixture.root)
        all_ids = [row["mail_id"] for row in bundle.all_records]
        split_ids = {
            split: [row["mail_id"] for row in rows]
            for split, rows in bundle.splits.items()
        }

        self.assertEqual(len(all_ids), len(set(all_ids)))
        ledger_rows = split_fixture.split_checker.read_csv_file(
            fixture.root / "artifacts" / "processed" / "dedup_ledger.csv"
        )[1]
        retained_ids = {row["mail_id"] for row in ledger_rows if row["disposition"] == "retained"}
        assigned_ids = set(split_ids["train"] + split_ids["validation"] + split_ids["test"])
        self.assertTrue(retained_ids.issubset(set(all_ids)))
        self.assertEqual(assigned_ids, retained_ids)
        self.assertFalse(set(split_ids["train"]) & set(split_ids["validation"]))
        self.assertFalse(set(split_ids["train"]) & set(split_ids["test"]))
        self.assertFalse(set(split_ids["validation"]) & set(split_ids["test"]))
        for split, rows in bundle.splits.items():
            self.assertTrue(all(row["split"] == split for row in rows))

        development_ids = {
            row["mail_id"] for row in split_fixture.split_checker.read_csv_file(
                fixture.root / "artifacts" / "splits" / "development_sample.csv"
            )[1]
        }
        self.assertTrue(development_ids.issubset(set(split_ids["train"])))

    def test_changed_split_csv_checksum_is_rejected(self):
        fixture = self.prepare_fixture()
        target = fixture.root / "artifacts" / "splits" / "test.csv"
        target.write_bytes(target.read_bytes() + b"\n")

        with self.assertRaises(ValueError):
            load_dataset(fixture.root)

    def test_changed_cleaning_cache_is_rejected(self):
        fixture = self.prepare_fixture()
        target = fixture.root / "artifacts" / "processed" / "cleaned_mail.jsonl"
        target.write_bytes(target.read_bytes() + b" ")

        with self.assertRaises(ValueError):
            load_dataset(fixture.root)


if __name__ == "__main__":
    unittest.main()
