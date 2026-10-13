"""End-to-end synthetic checks for cache privacy and per-message failure handling."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
from spam_classifier.index_audit import MANIFEST_FIELDS, audit_index

spec = importlib.util.spec_from_file_location("clean_mail_cli", REPOSITORY / "scripts/clean_mail.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
check_spec = importlib.util.spec_from_file_location("check_cleaning_cli", REPOSITORY / "scripts/check_cleaning.py")
checker = importlib.util.module_from_spec(check_spec)
check_spec.loader.exec_module(checker)


class CleanMailIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="clean-mail-test-")
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.project = base / "project"
        self.corpus = base / "corpus"
        (self.project / "artifacts/audit").mkdir(parents=True)
        (self.project / "artifacts/processed").mkdir()
        (self.project / "artifacts/splits").mkdir()
        (self.project / "src/spam_classifier").mkdir(parents=True)
        (self.project / "src/spam_classifier/email_parser.py").write_bytes(
            (REPOSITORY / "src/spam_classifier/email_parser.py").read_bytes()
        )
        (self.corpus / "full").mkdir(parents=True)
        (self.corpus / "data/000").mkdir(parents=True)
        for number, label in enumerate(("ham", "spam")):
            raw = b"Subject: synthetic subject\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nPRIVATE_RAW_BODY_SENTINEL person@example.test"
            (self.corpus / f"data/000/{number:03d}").write_bytes(raw)
        (self.corpus / "full/index").write_text("ham ../data/000/000\nspam ../data/000/001\n", encoding="utf-8")
        result = audit_index(self.corpus, development_size=2)
        (self.project / "artifacts/audit/index_summary.json").write_text(json.dumps(result.summary), encoding="utf-8")
        for name in ("processed/mail_manifest.csv", "splits/development_sample.csv"):
            with (self.project / "artifacts" / name).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(result.records)
        config = {"data_root": "../corpus", "index_file": "full/index", "labels": {"ham": 0, "spam": 1}, "artifacts_dir": "artifacts"}
        (self.project / "config.example.json").write_text(json.dumps(config), encoding="utf-8")

    def run_cli(self):
        with patch.object(cli, "PROJECT_ROOT", self.project), patch.object(sys, "argv", ["clean_mail.py"]), patch.dict(os.environ):
            os.environ.pop("TREC06C_ROOT", None)
            with contextlib.redirect_stdout(io.StringIO()):
                return cli.main()

    def test_local_cache_contains_source_text_but_public_summary_does_not(self):
        self.assertEqual(self.run_cli(), 0)
        cache = (self.project / "artifacts/processed/cleaned_development.jsonl").read_text(encoding="utf-8")
        summary = (self.project / "artifacts/audit/development_cleaning_summary.json").read_text(encoding="utf-8")
        self.assertIn("PRIVATE_RAW_BODY_SENTINEL", cache)
        self.assertNotIn("PRIVATE_RAW_BODY_SENTINEL", summary)
        self.assertNotIn("person@example.test", summary)
        self.assertNotIn(str(self.project), summary)
        parsed = [json.loads(line) for line in cache.splitlines()]
        self.assertTrue(all("person@example.test" not in row["text_clean"] for row in parsed))
        self.assertEqual(json.loads(summary)["processed_records"], 2)

    def test_changed_index_is_rejected_before_new_cache_is_written(self):
        with (self.corpus / "full/index").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        self.assertEqual(self.run_cli(), 2)
        self.assertFalse((self.project / "artifacts/processed/cleaned_development.jsonl").exists())

    def test_generation_summary_records_actual_parser_source_fingerprint(self):
        self.assertEqual(self.run_cli(), 0)
        summary = json.loads((self.project / "artifacts/audit/development_cleaning_summary.json").read_text(encoding="utf-8"))
        expected = hashlib.sha256((self.project / "src/spam_classifier/email_parser.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        self.assertEqual(summary["parser_source_sha256"], expected)

    def run_checker(self):
        with patch.object(checker, "PROJECT_ROOT", self.project), patch.object(sys, "argv", ["check_cleaning.py"]):
            with contextlib.redirect_stdout(io.StringIO()):
                return checker.main()

    def test_public_validation_is_aggregate_and_per_mail_review_stays_local(self):
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.run_checker(), 0)
        public = (self.project / "artifacts/audit/development_cleaning_validation.json").read_text(encoding="utf-8")
        private = (self.project / "artifacts/processed/development_cleaning_rule_review.json").read_text(encoding="utf-8")
        self.assertNotIn('"mail_id"', public)
        self.assertNotIn("PRIVATE_RAW_BODY_SENTINEL", public)
        self.assertIn('"mail_id"', private)
        self.assertTrue(json.loads(public)["passed"])

    def test_validation_rejects_cache_after_parser_source_changes(self):
        self.assertEqual(self.run_cli(), 0)
        source = self.project / "src/spam_classifier/email_parser.py"
        source.write_bytes(source.read_bytes() + b'\n# synthetic source change\n')
        self.assertEqual(self.run_checker(), 1)
        public = json.loads((self.project / "artifacts/audit/development_cleaning_validation.json").read_text(encoding="utf-8"))
        self.assertFalse(public["checks"]["parser_source_matches_generation"])

    def test_a_missing_mail_is_accounted_without_aborting_other_messages(self):
        (self.corpus / "data/000/000").unlink()
        self.assertEqual(self.run_cli(), 1)
        summary = json.loads((self.project / "artifacts/audit/development_cleaning_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["processed_records"], 2)
        self.assertEqual(summary["status_counts"]["error"], 1)
        self.assertEqual(summary["usable_records"], 1)
        self.assertFalse(summary["passed"])


if __name__ == "__main__":
    unittest.main()
