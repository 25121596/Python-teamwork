"""Independently recompute final metrics and replay the locked predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np

from check_models import independent_metrics, _metrics_match
from evaluate_final import EVALUATION_SOURCES
from spam_classifier.dataset import file_sha256, source_sha256
from spam_classifier.experiment_io import read_json, read_records, read_test_rows, require_ablation_validation, verify_lock, verified_file, write_json


def check(root: Path) -> dict:
    lock = verify_lock(root)
    require_ablation_validation(root, file_sha256(root / "artifacts/metrics/final_lock.json"))
    report_path = root / "artifacts/metrics/final_evaluation.json"
    report = read_json(report_path)
    if (report.get("passed") is not True or report["lock_sha256"] != file_sha256(root / "artifacts/metrics/final_lock.json")
            or report["evaluation_source_sha256"] != source_sha256(root, EVALUATION_SOURCES)):
        raise ValueError("Final evaluation provenance changed.")
    expected_paths = {"artifacts/processed/stage08/test_predictions.npz", "artifacts/metrics/error_analysis.json",
                      "artifacts/metrics/stage08_test.csv", "artifacts/figures/stage08_nb_confusion.png",
                      "artifacts/figures/stage08_lr_confusion.png", "artifacts/figures/stage08_lr_public_terms.png"}
    if set(report["output_sha256"]) != expected_paths:
        raise ValueError("Final evaluation manifest is incomplete.")
    for name, digest in report["output_sha256"].items():
        verified_file(root, name, digest, prefix="artifacts/")
    rows = read_test_rows(root, lock)
    records = read_records(root, {"test": rows}, lock["cleaning_cache_sha256"])["test"]
    truth = np.array([r["label"] for r in records])
    with np.load(root / "artifacts/processed/stage08/test_predictions.npz", allow_pickle=False) as saved:
        predictions = saved["predictions"]
        alignment = (saved["model_names"].tolist() == lock["model_names"]
                     and saved["mail_ids"].tolist() == [r["mail_id"] for r in records]
                     and np.array_equal(saved["y_true"], truth)
                     and predictions.shape == (len(rows), 3))
    if not alignment:
        raise ValueError("Saved final predictions do not align with frozen IDs/labels.")
    models = joblib.load(root / "artifacts/models/stage07/final_models.joblib")
    texts = [r[lock["representation"]] for r in records]
    checks = {"prediction_alignment": alignment, "pretest_model_selection_preserved": report["selected_model_before_test"] == lock["selected_model"]}
    for col, name in enumerate(lock["model_names"]):
        checks[name + "_metrics"] = _metrics_match(report["metrics"][name], independent_metrics(truth, predictions[:, col]))
        checks[name + "_reloaded_predictions"] = bool(np.array_equal(models[name].predict(texts), predictions[:, col]))
    return {"scope": "Independent arithmetic and locked model replay; no fitting", "checks": checks,
            "final_evaluation_sha256": file_sha256(report_path), "passed": all(checks.values())}


def main() -> int:
    try:
        report = check(ROOT)
        write_json(ROOT / "artifacts/audit/final_evaluation_validation.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "message": "Final evaluation is missing or failed verification; no old result was overwritten."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
