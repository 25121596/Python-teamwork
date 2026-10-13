"""Small synthetic corpora exercise failures that are absent in the real corpus."""

from __future__ import annotations

import builtins
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spam_classifier.index_audit import audit_index


class IndexAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="trec-index-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "corpus"
        (self.root / "full").mkdir(parents=True)
        (self.root / "data").mkdir()

    def mail(self, name: str, content: bytes = b"PRIVATE_MAIL_SENTINEL") -> Path:
        path = self.root / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def index(self, lines: list[str]):
        (self.root / "full" / "index").write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))

    def assert_ledger(self, result):
        index = result.summary["index"]
        self.assertEqual(index["physical_lines"], sum(index[key] for key in (
            "blank_lines", "malformed_rows", "unknown_label_rows", "unsafe_path_rows",
            "path_resolution_error_rows", "auxiliary_path_rows", "accepted_rows",
        )))
        self.assertEqual(index["accepted_rows"], index["unique_paths"] + index["duplicate_extra_rows"])
        self.assertEqual(index["unique_paths"], sum(result.summary["files"]["unique_status_counts"].values()))

    def test_clean_corpus_and_no_mail_body_reads(self):
        self.mail("000/001")
        self.mail("000/002", b"another private body")
        self.index(["ham ../data/000/001", "spam ../data/000/002"])
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path != self.root / "full" / "index":
                raise AssertionError("Audit tried to open a mail message.")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch.object(builtins, "open", side_effect=AssertionError("Unexpected file read")):
            result = audit_index(self.root)
        self.assertTrue(result.summary["passed"])
        self.assertEqual(result.summary["files"]["eligible_label_counts"], {"ham": 1, "spam": 1})
        self.assertEqual(result.summary["development_sample"]["actual_size"], 2)
        self.assertNotIn("PRIVATE_MAIL_SENTINEL", json.dumps(result.__dict__))
        self.assertNotIn(str(self.root), json.dumps(result.__dict__))
        self.assert_ledger(result)

    def test_duplicate_aliases_and_label_conflicts(self):
        self.mail("000/001")
        self.mail("000/002")
        self.index([
            "ham ../data/000/001", "ham ../data/000/../000/001",
            "spam ../data/000/002", "ham ../data/000/002",
        ])
        result = audit_index(self.root)
        self.assertFalse(result.summary["passed"])
        self.assertEqual(result.summary["index"]["duplicate_extra_rows"], 2)
        self.assertEqual(result.summary["index"]["label_conflict_paths"], 1)
        self.assertEqual(result.summary["index"]["label_conflict_rows"], 2)
        self.assertEqual(result.summary["files"]["eligible_unique_files"], 1)
        self.assertEqual(result.development_sample[0]["index_occurrences"], 2)
        conflict = next(row for row in result.records if row["status"] == "label_conflict")
        self.assertEqual(conflict["label"], "")
        self.assert_ledger(result)

    def test_bad_rows_missing_empty_directory_and_escaping_paths(self):
        self.mail("000/001")
        self.mail("000/empty", b"")
        self.index([
            "ham ../data/000/001", "", "broken", "Spam ../data/000/001",
            "ham ../data/000/missing", "spam ../data/000/empty", "spam ../data/000",
            "ham ../../outside", "spam /absolute/path",
        ])
        result = audit_index(self.root)
        index = result.summary["index"]
        self.assertEqual(index["blank_lines"], 1)
        self.assertEqual(index["malformed_rows"], 1)
        self.assertEqual(index["unknown_label_rows"], 1)
        self.assertEqual(index["unsafe_path_rows"], 2)
        self.assertEqual(result.summary["files"]["unique_status_counts"], {
            "eligible": 1, "empty": 1, "missing": 1, "non_regular": 1,
        })
        self.assertEqual(len(result.development_sample), 1)
        self.assert_ledger(result)

    def test_inventory_excludes_helper_files_but_detects_orphans(self):
        self.mail("000/001")
        self.mail("000/orphan")
        self.mail("000/._helper")
        self.mail(".DS_Store")
        self.mail("__MACOSX/hidden")
        self.index(["ham ../data/000/001"])
        result = audit_index(self.root)
        inventory = result.summary["inventory"]
        self.assertEqual(inventory["candidate_files"], 2)
        self.assertEqual(inventory["unindexed_files"], 1)
        self.assertEqual(inventory["helper_files"], 2)
        self.assertEqual(inventory["helper_directories"], 1)
        self.assertEqual(result.summary["issue_counts"], {"unindexed_file": 1})

    def test_indexed_helper_files_are_never_eligible(self):
        self.mail("000/001")
        self.mail("000/._helper")
        self.mail(".DS_Store")
        self.mail("__MACOSX/hidden")
        self.index([
            "ham ../data/000/001", "spam ../data/000/._helper",
            "spam ../data/.DS_Store", "spam ../data/__MACOSX/hidden",
        ])
        result = audit_index(self.root)
        self.assertFalse(result.summary["passed"])
        self.assertEqual(result.summary["index"]["auxiliary_path_rows"], 3)
        self.assertEqual(result.summary["issue_counts"], {"auxiliary_index_path": 3})
        self.assertEqual(result.summary["files"]["eligible_unique_files"], 1)
        self.assertEqual(len(result.development_sample), 1)
        self.assert_ledger(result)

    def test_seeded_sampling_and_ids_survive_index_reordering(self):
        lines = []
        for number in range(12):
            self.mail(f"000/{number:03d}")
            lines.append(f"{'ham' if number < 6 else 'spam'} ../data/000/{number:03d}")
        self.index(lines)
        first = audit_index(self.root, development_size=8, seed=42)
        self.index(list(reversed(lines)))
        second = audit_index(self.root, development_size=8, seed=42)
        self.assertEqual({row["relative_path"]: row["mail_id"] for row in first.records},
                         {row["relative_path"]: row["mail_id"] for row in second.records})
        self.assertEqual([row["mail_id"] for row in first.development_sample],
                         [row["mail_id"] for row in second.development_sample])
        self.assertEqual(first.summary["development_sample"]["label_counts"], {"ham": 4, "spam": 4})

    def test_relocated_corpus_keeps_mail_ids(self):
        self.mail("000/001")
        self.index(["ham ../data/000/001"])
        first = audit_index(self.root)
        moved = self.root.with_name("relocated")
        self.root.rename(moved)
        second = audit_index(moved)
        self.assertEqual(first.records[0]["mail_id"], second.records[0]["mail_id"])

    def test_symlink_outside_data_is_rejected(self):
        self.mail("000/001")
        target = Path(self.temp.name) / "outside"
        target.write_bytes(b"PRIVATE_OUTSIDE_SENTINEL")
        try:
            os.symlink(target, self.root / "data" / "link")
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is unavailable on this host.")
        self.index(["ham ../data/000/001", "spam ../data/link"])
        result = audit_index(self.root)
        self.assertEqual(result.summary["index"]["unsafe_path_rows"], 1)
        self.assertEqual(result.summary["inventory"]["symlink_entries"], 1)
        self.assertNotIn("PRIVATE_OUTSIDE_SENTINEL", json.dumps(result.__dict__))

    def test_no_eligible_mail_and_invalid_sample_size(self):
        self.index([])
        self.assertFalse(audit_index(self.root).summary["passed"])
        with self.assertRaises(ValueError):
            audit_index(self.root, development_size=0)


if __name__ == "__main__":
    unittest.main()
