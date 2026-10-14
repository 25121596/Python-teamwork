"""Synthetic full-pipeline checks for frozen metadata and leakage detection."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))
from spam_classifier.index_audit import MANIFEST_FIELDS, audit_index
from spam_classifier.email_parser import clean_text


def module(name, script):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / "scripts" / script)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


cleaner = module("split_test_cleaner", "clean_mail.py")
clean_checker = module("split_test_clean_checker", "check_cleaning.py")
splitter = module("split_test_splitter", "split_data.py")
split_checker = module("split_test_checker", "check_splits.py")


class SplitPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="split-test-")
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.corpus = base / "project", base / "corpus"
        for path in ("artifacts/audit", "artifacts/processed", "artifacts/splits", "src/spam_classifier", "scripts"):
            (self.root / path).mkdir(parents=True, exist_ok=True)
        for path in ("src/spam_classifier/email_parser.py", "src/spam_classifier/data_split.py", "scripts/split_data.py"):
            (self.root / path).write_bytes((REPOSITORY / path).read_bytes())
        (self.corpus / "full").mkdir(parents=True)
        (self.corpus / "data/000").mkdir(parents=True)
        index_lines = []
        for label_index, label in enumerate(("ham", "spam")):
            for index in range(16):
                number = label_index * 16 + index
                suffix = max(1, index)
                raw = (f"Subject: {label} subject {suffix}\r\nContent-Type: text/plain; charset=us-ascii\r\n\r\n"
                       f"SYNTHETIC_PRIVATE_SENTINEL {label} message number {suffix}").encode("ascii")
                (self.corpus / f"data/000/{number:03d}").write_bytes(raw)
                index_lines.append(f"{label} ../data/000/{number:03d}\n")
        (self.corpus / "full/index").write_text("".join(index_lines), encoding="utf-8")
        audit = audit_index(self.corpus, development_size=2)
        (self.root / "artifacts/audit/index_summary.json").write_text(json.dumps(audit.summary), encoding="utf-8")
        for path, rows in (("processed/mail_manifest.csv", audit.records), ("splits/development_sample.csv", audit.development_sample)):
            with (self.root / "artifacts" / path).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        self.config = {"data_root": "../corpus", "index_file": "full/index", "labels": {"ham": 0, "spam": 1},
                       "artifacts_dir": "artifacts", "random_state": 42,
                       "split_ratios": {"train": .7, "validation": .15, "test": .15}}
        self.config_path = self.root / "config.example.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        for scope in ("full", "development"):
            self.assertEqual(self.run_main(cleaner, ["--scope", scope]), 0)
            self.assertEqual(self.run_main(clean_checker, ["--scope", scope]), 0)

    def run_main(self, target, args=None):
        with patch.object(target, "PROJECT_ROOT", self.root), patch.object(sys, "argv", ["script"] + (args or [])), patch.dict(os.environ):
            os.environ.pop("TREC06C_ROOT", None)
            with contextlib.redirect_stdout(io.StringIO()):
                return target.main()

    def frozen_files(self):
        paths = [self.root / f"artifacts/splits/{name}.csv" for name in ("train", "validation", "test")]
        paths += [self.root / "artifacts/processed/dedup_ledger.csv", self.root / "artifacts/audit/split_summary.json"]
        return {path: path.read_bytes() for path in paths}

    def test_pipeline_accounts_duplicates_and_public_outputs_are_metadata_only(self):
        self.assertEqual(self.run_main(splitter), 0)
        self.assertEqual(self.run_main(split_checker), 0)
        summary = json.loads((self.root / "artifacts/audit/split_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["deduplication"]["disposition_counts"], {"duplicate": 2, "retained": 30})
        for path in list((self.root / "artifacts/audit").glob("split*.json")) + list((self.root / "artifacts/splits").glob("*.csv")):
            self.assertNotIn("SYNTHETIC_PRIVATE_SENTINEL", path.read_text(encoding="utf-8-sig"))
        fields, _ = split_checker.read_csv_file(self.root / "artifacts/splits/test.csv")
        self.assertEqual(fields, ["mail_id", "label_name", "label", "split"])

    def test_rerun_is_byte_identical_and_different_seed_cannot_overwrite_freeze(self):
        self.assertEqual(self.run_main(splitter), 0)
        original = self.frozen_files()
        self.assertEqual(self.run_main(splitter), 0)
        self.assertEqual(self.frozen_files(), original)
        self.config["random_state"] = 43
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.assertEqual(self.run_main(splitter), 2)
        self.assertEqual(self.frozen_files(), original)

    def test_changed_cleaning_cache_fails_without_overwriting_freeze(self):
        self.assertEqual(self.run_main(splitter), 0)
        original = self.frozen_files()
        cache = self.root / "artifacts/processed/cleaned_mail.jsonl"
        cache.write_bytes(cache.read_bytes() + b"\n")
        self.assertEqual(self.run_main(splitter), 2)
        self.assertEqual(self.frozen_files(), original)

    def test_changed_parser_source_is_rejected_before_publication(self):
        parser = self.root / "src/spam_classifier/email_parser.py"
        parser.write_bytes(parser.read_bytes() + b"\n# modified\n")
        self.assertEqual(self.run_main(splitter), 2)
        self.assertFalse((self.root / "artifacts/splits/train.csv").exists())

    def test_public_label_tampering_fails_independent_validation(self):
        self.assertEqual(self.run_main(splitter), 0)
        target = self.root / "artifacts/splits/test.csv"
        fields, rows = split_checker.read_csv_file(target)
        rows[0]["label"] = str(1 - int(rows[0]["label"]))
        target.write_bytes(splitter.csv_bytes(rows, tuple(fields)))
        self.assertEqual(self.run_main(split_checker), 1)
        report = json.loads((self.root / "artifacts/audit/split_validation.json").read_text(encoding="utf-8"))
        self.assertFalse(report["checks"]["all_frozen_output_checksums_match"])
        self.assertGreater(report["row_violation_counts"]["public_metadata_mismatch"], 0)

    def test_graph_detects_split_duplicate_even_with_recomputed_checksum(self):
        self.assertEqual(self.run_main(splitter), 0)
        path = self.root / "artifacts/processed/dedup_ledger.csv"
        fields, rows = split_checker.read_csv_file(path)
        duplicate = next(row for row in rows if row["disposition"] == "duplicate")
        duplicate["split"] = "test" if duplicate["split"] != "test" else "train"
        content = splitter.csv_bytes(rows, tuple(fields))
        path.write_bytes(content)
        summary_path = self.root / "artifacts/audit/split_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["output_sha256"]["artifacts/processed/dedup_ledger.csv"] = hashlib.sha256(content).hexdigest()
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        report = split_checker.check(self.root)
        self.assertTrue(report["checks"]["all_frozen_output_checksums_match"])
        self.assertFalse(report["passed"])
        self.assertGreater(report["row_violation_counts"]["duplicate_assignment_mismatch"], 0)

    def test_ledger_raw_hash_tampering_fails_cleaning_digest_check_with_recomputed_output_hash(self):
        self.assertEqual(self.run_main(splitter), 0)
        ledger_path = self.root / "artifacts/processed/dedup_ledger.csv"
        fields, rows = split_checker.read_csv_file(ledger_path)
        rows[0]["raw_sha256"] = "a" * 64
        content = splitter.csv_bytes(rows, tuple(fields))
        ledger_path.write_bytes(content)

        summary_path = self.root / "artifacts/audit/split_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["output_sha256"]["artifacts/processed/dedup_ledger.csv"] = hashlib.sha256(content).hexdigest()
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        report = split_checker.check(self.root)
        self.assertFalse(report["checks"]["ledger_raw_signatures_match_cleaning_digest"])
        self.assertTrue(report["checks"]["all_frozen_output_checksums_match"])
        self.assertFalse(report["passed"])

    def test_dedup_and_component_summary_counts_are_independently_recomputed(self):
        self.assertEqual(self.run_main(splitter), 0)
        summary_path = self.root / "artifacts/audit/split_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["deduplication"]["by_label"]["ham"]["retained"] += 1
        summary["deduplication"]["leakage_groups_before_dedup"]["groups"] += 1
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        report = split_checker.check(self.root)
        self.assertFalse(report["checks"]["dedup_class_ledger_matches"])
        self.assertFalse(report["checks"]["component_summary_matches"])
        self.assertFalse(report["passed"])

    def test_cache_body_empty_flag_must_match_body_content(self):
        self.assertEqual(self.run_main(splitter), 0)
        cache_path = self.root / "artifacts/processed/cleaned_mail.jsonl"
        rows = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
        candidate = next(row for row in rows if row["body_clean"] and any(char.isalnum() for char in row["body_clean"]))
        candidate["body_empty"] = True
        cache_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

        report = split_checker.check(self.root)
        self.assertGreater(report["row_violation_counts"].get("cache_signal_flag_mismatch", 0), 0)
        self.assertFalse(report["passed"])

    def test_contact_collision_audit_distinguishes_unmasked_surfaces(self):
        first, second = "Reach a@example.test now", "Reach b@example.test now"
        one = {"body_empty": False, "body_selection": "plain", "body_decoded": first, "body_clean": clean_text(first)}
        two = dict(one, body_decoded=second, body_clean=clean_text(second))
        self.assertEqual(one["body_clean"], two["body_clean"])
        self.assertTrue(splitter.precontact_body_signature(one))
        self.assertNotEqual(splitter.precontact_body_signature(one), splitter.precontact_body_signature(two))
        self.assertEqual(splitter.precontact_body_signature(dict(one, body_selection="mixed")), "")

    def test_validator_requires_complete_checksum_manifest(self):
        self.assertEqual(self.run_main(splitter), 0)
        path = self.root / "artifacts/audit/split_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        del summary["output_sha256"]["artifacts/processed/dedup_ledger.csv"]
        path.write_text(json.dumps(summary), encoding="utf-8")
        report = split_checker.check(self.root)
        self.assertFalse(report["checks"]["all_frozen_output_checksums_match"])
        self.assertFalse(report["passed"])

    def test_validator_rejects_other_seed_as_this_frozen_contract(self):
        self.assertEqual(self.run_main(splitter), 0)
        path = self.root / "artifacts/audit/split_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["random_state"] = 43
        path.write_text(json.dumps(summary), encoding="utf-8")
        report = split_checker.check(self.root)
        self.assertFalse(report["checks"]["frozen_seed_and_ratios_match_contract"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
