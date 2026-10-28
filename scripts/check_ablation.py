"""Independently replay stage-seven validation predictions without test access."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from check_models import independent_metrics, _metrics_match
from spam_classifier.dataset import file_sha256, read_csv
from spam_classifier.experiment_io import REPRESENTATIONS, read_json, read_records, verify_lock, verified_file, write_json
from spam_classifier.model_inputs import fitted_feature_state


def check(root: Path) -> dict:
    lock = verify_lock(root)
    rows = {}
    for split in ("train", "validation"):
        relative = f"artifacts/splits/{split}.csv"
        path = verified_file(root, relative, lock["split_csv_sha256"][split], prefix="artifacts/splits/")
        rows[split] = read_csv(path)[1]
    data = read_records(root, rows, lock["cleaning_cache_sha256"])
    comparisons = joblib.load(root / "artifacts/models/stage07/ablation_models.joblib")
    models = joblib.load(root / "artifacts/models/stage07/final_models.joblib")
    summary = read_json(root / "artifacts/metrics/cleaning_ablation.json")
    with np.load(root / "artifacts/processed/stage07/validation_predictions.npz", allow_pickle=False) as saved:
        predictions = saved["predictions"]
        alignment = (saved["columns"].tolist() == [*REPRESENTATIONS, "majority", "nb", "lr"]
                     and saved["mail_ids"].tolist() == [r["mail_id"] for r in data["validation"]]
                     and np.array_equal(saved["y_true"], [r["label"] for r in data["validation"]])
                     and predictions.shape == (len(data["validation"]), 5))
    if not alignment or set(comparisons) != set(REPRESENTATIONS) or set(models) != {"majority", "nb", "lr"}:
        raise ValueError("Ablation artifacts do not match the frozen contract.")
    checks = {"alignment": alignment, "representation_matches_lock": summary["representation"] == lock["representation"]}
    y_train = [r["label"] for r in data["train"]]
    y_val = [r["label"] for r in data["validation"]]
    with threadpool_limits(limits=1):
        for col, representation in enumerate(REPRESENTATIONS):
            pipeline = comparisons[representation]
            train = [r[representation] for r in data["train"]]
            validation = [r[representation] for r in data["validation"]]
            refit = clone(pipeline).fit(train, y_train)
            checks[representation + "_train_only_state"] = (fitted_feature_state(refit.named_steps["tfidf"]) == fitted_feature_state(pipeline.named_steps["tfidf"])
                    and bool(np.allclose(refit.named_steps["classifier"].coef_, pipeline.named_steps["classifier"].coef_, rtol=0, atol=1e-12)))
            checks[representation + "_predictions"] = bool(np.array_equal(pipeline.predict(validation), predictions[:, col]) and np.array_equal(refit.predict(validation), predictions[:, col]))
            record = next(row for row in summary["comparisons"] if row["representation"] == representation)
            checks[representation + "_metrics"] = _metrics_match(record["metrics"], independent_metrics(y_val, predictions[:, col]))
        for col, name in enumerate(lock["model_names"], 2):
            pipeline = models[name]
            train = [r[lock["representation"]] for r in data["train"]]
            validation = [r[lock["representation"]] for r in data["validation"]]
            refit = clone(pipeline).fit(train, y_train)
            checks[name + "_selected_representation"] = fitted_feature_state(pipeline.named_steps["tfidf"]) == fitted_feature_state(comparisons[lock["representation"]].named_steps["tfidf"])
            checks[name + "_train_only_replay"] = bool(np.array_equal(pipeline.predict(validation), predictions[:, col]) and np.array_equal(refit.predict(validation), predictions[:, col]))
            checks[name + "_metrics"] = _metrics_match(summary["final_validation_metrics"][name], independent_metrics(y_val, predictions[:, col]))
    ranked = sorted((row for row in summary["comparisons"] if row["converged"]), key=lambda row: (
        -row["metrics"]["macro_f1"], row["metrics"]["ham_false_positive_rate"], -row["metrics"]["spam_recall"], REPRESENTATIONS.index(row["representation"])))
    checks["independent_representation_ranking"] = bool(ranked and ranked[0]["representation"] == lock["representation"])
    return {"scope": "Train-only refits, validation-only metrics; no test CSV or matrix opened", "checks": checks,
            "lock_sha256": file_sha256(root / "artifacts/metrics/final_lock.json"),
            "validator_source_sha256": hashlib.sha256((root / "scripts/check_ablation.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "passed": all(checks.values())}


def main() -> int:
    try:
        report = check(ROOT)
        write_json(ROOT / "artifacts/audit/ablation_validation.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "message": "Ablation evidence unavailable or inconsistent; existing audit preserved."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
