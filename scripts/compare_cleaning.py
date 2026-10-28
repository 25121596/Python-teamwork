"""Run stage seven and freeze the three models before final-test evaluation."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np

from spam_classifier.cleaning_ablation import train_validation_only
from spam_classifier.dataset import file_sha256, source_sha256
from spam_classifier.experiment_io import EXPERIMENT_SOURCES, REPRESENTATIONS, SCOPE, read_json, read_records, verified_stage06, verify_lock, write_json
from spam_classifier.model_inputs import fitted_feature_state, runtime_versions


def run(root: Path) -> dict:
    lock_path = root / "artifacts/metrics/final_lock.json"
    if lock_path.exists():
        verify_lock(root)
        return {"status": "existing_lock_verified", "test_metrics_computed": False}
    if ((root / "artifacts/metrics/final_evaluation.json").exists()
            or (root / "artifacts/processed/stage08/test_access.json").exists()):
        raise ValueError("Final-test evidence already exists; do not reselect after seeing test results.")
    inputs, baseline = verified_stage06(root)
    frozen = read_json(root / "artifacts/audit/split_summary.json")
    cache_sha = frozen["input_provenance"]["cleaning_cache_sha256"]
    metadata = {"train": inputs.train_rows, "validation": inputs.validation_rows}
    records = read_records(root, metadata, cache_sha)
    source_before = source_sha256(root, EXPERIMENT_SOURCES)
    summary, comparisons, models, predictions = train_validation_only(records)
    # The clean arm must reproduce stage six on the same inputs and configuration.
    clean = summary["comparisons"][0]
    original = next(row for row in baseline["experiments"] if row["candidate_id"] == "lr_C_2")
    with np.load(root / "artifacts/processed/stage06/validation_predictions.npz", allow_pickle=False) as saved:
        col = saved["candidate_ids"].tolist().index("lr_C_2")
        aligned = saved["mail_ids"].tolist() == [row["mail_id"] for row in inputs.validation_rows]
        identical = np.array_equal(saved["predictions"][:, col], predictions[:, 0])
    if (not aligned or not identical or clean["metrics"] != original["metrics"]
            or clean["fitted_state"] != fitted_feature_state(inputs.feature_pipeline.named_steps["tfidf"])
            or source_before != source_sha256(root, EXPERIMENT_SOURCES)):
        raise ValueError("Clean comparison did not reproduce the validated stage-six model.")
    for dirname in ("artifacts/models/stage07", "artifacts/processed/stage07"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    outputs = []
    for relative, value in (("artifacts/models/stage07/final_models.joblib", models),
                            ("artifacts/models/stage07/ablation_models.joblib", comparisons)):
        target = root / relative
        temporary = target.with_name(target.name + ".tmp")
        joblib.dump(value, temporary, compress=3)
        temporary.replace(target)
        outputs.append(relative)
    relative = "artifacts/processed/stage07/validation_predictions.npz"
    np.savez_compressed(root / relative, mail_ids=np.array([r["mail_id"] for r in records["validation"]]),
                        y_true=np.array([r["label"] for r in records["validation"]]), predictions=predictions,
                        columns=np.array([*REPRESENTATIONS, "majority", "nb", "lr"]))
    outputs.append(relative)
    summary.update(schema_version=1, clean_arm_reproduced_stage06=True, test_metrics_computed=False)
    relative = "artifacts/metrics/cleaning_ablation.json"
    write_json(root / relative, summary)
    outputs.append(relative)
    stream = io.StringIO(newline="")
    fields = ("representation", "accuracy", "spam_precision", "spam_recall", "macro_f1", "ham_false_positive_rate", "vocabulary_size", "training_markup_token_fraction")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in summary["comparisons"]:
        combined = dict(row, **row["metrics"])
        writer.writerow({key: combined[key] for key in fields})
    relative = "artifacts/metrics/stage07_validation.csv"
    (root / relative).write_bytes(stream.getvalue().encode("utf-8"))
    outputs.append(relative)
    upstream = ["artifacts/metrics/model_selection.json", "artifacts/audit/model_validation.json",
                "artifacts/audit/split_summary.json", "artifacts/audit/split_validation.json", "artifacts/audit/cleaning_summary.json"]
    lock = {"schema_version": 1, "scope": SCOPE, "versions": runtime_versions(), "source_sha256": source_before,
            "representation": summary["representation"], "model_names": ["majority", "nb", "lr"], "selected_model": "lr",
            "parameters": {"nb": {"alpha": 0.1}, "lr": {"C": 2.0}},
            "decision_rule": "estimator.predict; no threshold tuning; no train+validation refit",
            "test_rows": frozen["allocation"]["test"]["records"], "cleaning_cache_sha256": cache_sha,
            "split_csv_sha256": {name: frozen["output_sha256"][f"artifacts/splits/{name}.csv"] for name in ("train", "validation", "test")},
            "upstream_sha256": {name: file_sha256(root / name) for name in upstream},
            "output_sha256": {name: file_sha256(root / name) for name in outputs}}
    write_json(lock_path, lock)  # Commit marker written last; never silently overwritten.
    verify_lock(root)
    return {"status": "locked", "representation": lock["representation"], "comparisons": summary["comparisons"], "test_metrics_computed": False}


def main() -> int:
    try:
        result = run(ROOT)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__, "message": "Restore and validate stages 2–6 first. No final-test result was computed."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
