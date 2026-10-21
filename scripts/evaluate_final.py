"""Evaluate the immutable stage-seven lock; retain predictions/text locally."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from threadpoolctl import threadpool_limits

from spam_classifier.dataset import file_sha256, source_sha256
from spam_classifier.experiment_io import read_json, read_records, read_test_rows, require_ablation_validation, verify_lock, verified_file, write_json
from spam_classifier.final_evaluation import public_coefficients, score_locked_models

EVALUATION_SOURCES = ("src/spam_classifier/final_evaluation.py", "scripts/evaluate_final.py")


def figures(root: Path, metrics: dict, words: dict) -> list[str]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    outputs = []
    for name in ("nb", "lr"):
        counts = metrics[name]["confusion_matrix"]
        matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
        fig, ax = plt.subplots(figsize=(6.8, 5.5), layout="constrained")
        ax.imshow(matrix, cmap="Blues")
        for (row, col), count in np.ndenumerate(matrix):
            ax.text(col, row, str(count), ha="center", va="center", fontsize=23,
                    color="white" if count > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], xticklabels=["HAM", "SPAM"], yticks=[0, 1], yticklabels=["HAM", "SPAM"],
               xlabel="Predicted label", ylabel="True label", title=f"{name.upper()} frozen test set, n={metrics[name]['n']:,}")
        relative = f"artifacts/figures/stage08_{name}_confusion.png"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(root / relative, dpi=180)
        plt.close(fig)
        outputs.append(relative)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6), layout="constrained")
    for label, axis in zip(("ham", "spam"), axes):
        entries = words[label][::-1]
        if entries:
            axis.barh([r["term"] for r in entries], [r["coefficient"] for r in entries], color="#347a9f" if label == "ham" else "#d98535")
        else:
            axis.text(.5, .5, "No whitelist term with this sign", transform=axis.transAxes, ha="center")
        axis.set(title=f"LR association with {label.upper()}", xlabel="Coefficient (whitelist only)")
    relative = "artifacts/figures/stage08_lr_public_terms.png"
    fig.savefig(root / relative, dpi=180)
    plt.close(fig)
    outputs.append(relative)
    return outputs


def run(root: Path) -> dict:
    lock = verify_lock(root)  # Must precede all test access.
    lock_sha = file_sha256(root / "artifacts/metrics/final_lock.json")
    require_ablation_validation(root, lock_sha)
    report_path = root / "artifacts/metrics/final_evaluation.json"
    source = source_sha256(root, EVALUATION_SOURCES)
    if report_path.exists():
        existing = read_json(report_path)
        if existing["lock_sha256"] != lock_sha or existing["evaluation_source_sha256"] != source:
            raise ValueError("Final evaluation already exists under a different lock/source.")
        for relative, digest in existing["output_sha256"].items():
            verified_file(root, relative, digest, prefix="artifacts/")
        return {"status": "existing_evaluation_verified", "metrics": existing["metrics"]}
    private = root / "artifacts/processed/stage08"
    private.mkdir(parents=True, exist_ok=True)
    access = private / "test_access.json"
    if access.exists() and read_json(access)["lock_sha256"] != lock_sha:
        raise ValueError("Test set was already opened under a different lock.")
    write_json(access, {"lock_sha256": lock_sha, "purpose": "Final evaluation only; cannot reselect using test outcomes."})
    rows = read_test_rows(root, lock)
    records = read_records(root, {"test": rows}, lock["cleaning_cache_sha256"])["test"]
    # Deserialisation happens only after verify_lock checks the exact artifact set.
    models = joblib.load(root / "artifacts/models/stage07/final_models.joblib")
    with threadpool_limits(limits=1):
        metrics, predictions, reviews, diagnostics = score_locked_models(models, records, lock["representation"])
    words = public_coefficients(models["lr"])
    outputs = figures(root, metrics, words)
    relative = "artifacts/processed/stage08/test_predictions.npz"
    np.savez_compressed(root / relative, model_names=np.array(lock["model_names"]),
                        mail_ids=np.array([r["mail_id"] for r in records]),
                        y_true=np.array([r["label"] for r in records]), predictions=predictions)
    outputs.append(relative)
    # Review is a mutable local work file and deliberately excluded from the
    # immutable prediction manifest. Reviewed statements require human scrutiny.
    review_path = private / "error_review.json"
    if not review_path.exists():
        write_json(review_path, {"scope": "Private review candidates, real mail text; never commit this file.",
                                 "selection": "Up to 15 highest-absolute-score FP and FN per formal model; deterministic mail ID tie break.",
                                 "lock_sha256": lock_sha, "cases": reviews})
    relative = "artifacts/metrics/error_analysis.json"
    write_json(root / relative, {"diagnostics": diagnostics, "public_coefficients": words,
                               "semantic_review_status": "pending",
                               "note": "Quality flags are descriptive observations, not established causes. Review private candidates before making semantic claims."})
    outputs.append(relative)
    stream = io.StringIO(newline="")
    fields = ("model", "n", "accuracy", "spam_precision", "spam_recall", "ham_f1", "spam_f1", "macro_f1", "ham_false_positive_rate", "tn", "fp", "fn", "tp")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for name in lock["model_names"]:
        row = dict(metrics[name], **metrics[name]["confusion_matrix"], model=name)
        writer.writerow({key: row[key] for key in fields})
    relative = "artifacts/metrics/stage08_test.csv"
    (root / relative).write_bytes(stream.getvalue().encode("utf-8"))
    outputs.append(relative)
    if verify_lock(root) != lock or source_sha256(root, EVALUATION_SOURCES) != source:
        raise ValueError("Experiment source or lock changed during evaluation.")
    report = {"schema_version": 1, "scope": "Frozen final test; no fit or parameter selection", "lock_sha256": lock_sha,
              "evaluation_source_sha256": source, "representation": lock["representation"],
              "selected_model_before_test": lock["selected_model"], "metrics": metrics,
              "output_sha256": {name: file_sha256(root / name) for name in outputs},
              "semantic_review_status": "pending", "passed": True}
    write_json(report_path, report)
    return {"status": "evaluated", "metrics": metrics, "semantic_review_status": "pending"}


def main() -> int:
    try:
        print(json.dumps(run(ROOT), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__, "message": "A verified stage-seven lock and original cached texts are required; existing evidence is preserved."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
