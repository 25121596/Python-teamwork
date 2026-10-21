"""Synthetic end-to-end run, with no real corpus or reported course scores."""

import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import check_ablation
import check_final_results
import compare_cleaning
import evaluate_final
from test_cleaning_ablation import records_fixture
from spam_classifier.cleaning_ablation import train_validation_only
from spam_classifier.dataset import file_sha256
from spam_classifier.experiment_io import EXPERIMENT_SOURCES, read_json, verify_lock, write_json
from spam_classifier.final_evaluation import public_coefficients, score_locked_models
from spam_classifier.text_features import PUBLIC_TERMS


class FinalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = records_fixture()
        for name in EXPERIMENT_SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        for directory in ("audit", "metrics", "splits", "processed/stage06"):
            (self.root / "artifacts" / directory).mkdir(parents=True, exist_ok=True)
        for split, rows in self.data.items():
            with (self.root / f"artifacts/splits/{split}.csv").open("w", newline="", encoding="utf-8") as stream:
                fields = ["mail_id", "label_name", "label", "split"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
                writer.writeheader();writer.writerows(rows)
        cache = self.root / "artifacts/processed/cleaned_mail.jsonl"
        cache.write_text("".join(json.dumps(row) + "\n" for rows in self.data.values() for row in rows), encoding="utf-8")
        self.summary, arms, _, predictions = train_validation_only({k: self.data[k] for k in ("train", "validation")})
        self.inputs = SimpleNamespace(train_rows=self.data["train"], validation_rows=self.data["validation"], feature_pipeline=arms["text_clean"])
        self.baseline = {"experiments": [{"candidate_id": "lr_C_2", "metrics": self.summary["comparisons"][0]["metrics"]}]}
        np.savez_compressed(self.root / "artifacts/processed/stage06/validation_predictions.npz",
                            candidate_ids=np.array(["lr_C_2"]), mail_ids=np.array([r["mail_id"] for r in self.data["validation"]]),
                            predictions=predictions[:, [0]])
        frozen = {"input_provenance": {"cleaning_cache_sha256": file_sha256(cache)},
                  "allocation": {"test": {"records": len(self.data["test"])}},
                  "output_sha256": {f"artifacts/splits/{split}.csv": file_sha256(self.root / f"artifacts/splits/{split}.csv") for split in self.data}}
        write_json(self.root / "artifacts/audit/split_summary.json", frozen)
        for name in ("metrics/model_selection.json", "audit/model_validation.json", "audit/split_validation.json", "audit/cleaning_summary.json"):
            write_json(self.root / "artifacts" / name, {"fixture": "synthetic upstream boundary stub"})

    def build_lock(self):
        # Upstream stages have their own 110 regression tests; this isolates the
        # new production run from unavailable private stage-six real artifacts.
        with patch.object(compare_cleaning, "verified_stage06", return_value=(self.inputs, self.baseline)):
            result = compare_cleaning.run(self.root)
        self.assertEqual(result["status"], "locked")
        report = check_ablation.check(self.root)
        self.assertTrue(report["passed"])
        write_json(self.root / "artifacts/audit/ablation_validation.json", report)

    def test_full_roundtrip_independent_metrics_replay_and_idempotence(self):
        self.build_lock()
        result = evaluate_final.run(self.root)
        self.assertEqual(result["status"], "evaluated")
        self.assertTrue(check_final_results.check(self.root)["passed"])
        with patch.object(evaluate_final, "score_locked_models", side_effect=AssertionError("must not predict again")):
            self.assertEqual(evaluate_final.run(self.root)["status"], "existing_evaluation_verified")
        self.assertEqual(compare_cleaning.run(self.root)["status"], "existing_lock_verified")
        for path in (self.root / "artifacts/metrics").glob("*.json"):
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("testonlyprobe", content)
            self.assertNotIn('"text_clean":', content)

    def test_stage_seven_never_opens_test_csv_and_retains_no_test_rows(self):
        original = Path.open
        def guard(path, *args, **kwargs):
            if path.name in {"test.csv", "test.npz"}:
                raise AssertionError("test access before lock")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            self.build_lock()

    def test_missing_lock_blocks_before_test_read(self):
        with patch.object(evaluate_final, "read_test_rows", side_effect=AssertionError("must not read")):
            with self.assertRaises(FileNotFoundError):
                evaluate_final.run(self.root)

    def test_changed_model_rejected_before_deserialization(self):
        self.build_lock()
        (self.root / "artifacts/models/stage07/final_models.joblib").write_bytes(b"changed")
        with patch.object(evaluate_final.joblib, "load", side_effect=AssertionError("must not load")):
            with self.assertRaises(ValueError):
                evaluate_final.run(self.root)

    def test_changed_test_labels_rejected_before_predictions(self):
        self.build_lock()
        path = self.root / "artifacts/splits/test.csv"
        path.write_text(path.read_text().replace("test-0,ham,0", "test-0,spam,1"))
        with patch.object(evaluate_final, "score_locked_models", side_effect=AssertionError("must not predict")):
            with self.assertRaises(ValueError):
                evaluate_final.run(self.root)

    def test_changed_lock_source_rejected(self):
        self.build_lock()
        source = self.root / "scripts/compare_cleaning.py"
        source.write_bytes(source.read_bytes() + b"\n# changed\n")
        with self.assertRaises(ValueError):
            verify_lock(self.root)

    def test_test_access_prevents_reselection_even_without_final_report(self):
        write_json(self.root / "artifacts/processed/stage08/test_access.json", {"lock_sha256": "earlier"})
        with self.assertRaisesRegex(ValueError, "already exists"):
            compare_cleaning.run(self.root)

    def test_failed_ablation_audit_blocks_before_test_access(self):
        self.build_lock()
        path = self.root / "artifacts/audit/ablation_validation.json"
        audit = read_json(path);audit["passed"] = False;write_json(path, audit)
        with patch.object(evaluate_final, "read_test_rows", side_effect=AssertionError("must not read")):
            with self.assertRaises(ValueError):
                evaluate_final.run(self.root)

    def test_wrong_clean_arm_cannot_publish_lock(self):
        self.baseline["experiments"][0]["metrics"]["accuracy"] = .123
        with patch.object(compare_cleaning, "verified_stage06", return_value=(self.inputs, self.baseline)):
            with self.assertRaisesRegex(ValueError, "did not reproduce"):
                compare_cleaning.run(self.root)
        self.assertFalse((self.root / "artifacts/metrics/final_lock.json").exists())

    def test_private_cases_cover_fp_fn_and_public_coefficients_use_whitelist(self):
        _, _, models, _ = train_validation_only({k: self.data[k] for k in ("train", "validation")})
        records = [dict(row, label=1-row["label"]) for row in self.data["test"]]
        metrics, _, reviews, diagnostics = score_locked_models(models, records, "text_clean")
        for name in ("nb", "lr"):
            self.assertEqual(len(reviews[name]["FP"]), metrics[name]["confusion_matrix"]["fp"])
            self.assertEqual(len(reviews[name]["FN"]), metrics[name]["confusion_matrix"]["fn"])
            self.assertFalse(any(r["reviewed"] for cases in reviews[name].values() for r in cases))
            self.assertEqual(diagnostics[name]["FP"]["all_errors"], metrics[name]["confusion_matrix"]["fp"])
        public = public_coefficients(models["lr"])
        self.assertTrue(all(row["term"] in PUBLIC_TERMS for label in ("ham", "spam") for row in public[label]))


if __name__ == "__main__":
    unittest.main()
