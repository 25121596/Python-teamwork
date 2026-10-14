"""Train a frozen seven-run comparison and select on validation data only."""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from spam_classifier.dataset import file_sha256, source_sha256
from spam_classifier.model_inputs import MODEL_SOURCES, fitted_feature_state, load_model_inputs, runtime_versions
from spam_classifier.model_selection import CANDIDATES, SELECTION_POLICY, make_estimator, select_candidate, validation_metrics


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def write_comparison_csv(path: Path, results: list[dict]) -> None:
    metric_names = ("n", "ham_support", "spam_support", "accuracy", "spam_precision", "spam_recall", "ham_f1", "spam_f1", "macro_f1", "ham_false_positive_rate")
    fields = ("candidate_id", "family", "parameters", "converged", "n_iter", "fit_seconds", "predict_seconds") + metric_names + ("tn", "fp", "fn", "tp")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in results:
        item = {key: row[key] for key in fields[:7]}
        item["parameters"] = json.dumps(item["parameters"], sort_keys=True)
        item["n_iter"] = json.dumps(item["n_iter"])
        item.update({key: row["metrics"][key] for key in metric_names})
        item.update(row["metrics"]["confusion_matrix"])
        writer.writerow(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stream.getvalue().encode("utf-8"))


def plot_comparison(path: Path, results: list[dict], selected_id: str) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    positions = np.arange(len(results))
    labels = ["多数类" if row["family"] == "majority" else
              f"NB α={row['parameters']['alpha']:g}" if row["family"] == "multinomial_nb" else
              f"LR C={row['parameters']['C']:g}" for row in results]
    colors = ["#d98535" if row["candidate_id"] == selected_id else "#347a9f" for row in results]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    values = [100 * row["metrics"]["macro_f1"] for row in results]
    axes[0].barh(positions, values, color=colors)
    axes[0].set(yticks=positions, yticklabels=labels, xlabel="macro-F1（%）", xlim=(0, 108), title="验证集 macro-F1：越高越好")
    for number, value in enumerate(values):
        axes[0].text(value + .6, number, f"{value:.2f}", va="center", fontsize=9)
    values = [100 * row["metrics"]["ham_false_positive_rate"] for row in results]
    axes[1].barh(positions, values, color=colors)
    axes[1].set(yticks=positions, yticklabels=labels, xlabel="正常邮件误报率 FP / 3,212（%）", title="验证集误报率：越低越好")
    axes[1].set_xlim(0, max(values) * 1.3 + .05)
    for number, value in enumerate(values):
        axes[1].text(value + .015, number, f"{value:.3f}", va="center", fontsize=9)
    for axis in axes:
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("固定验证集 5,099 封：七次预设实验（橙色为所选候选）", fontsize=13)
    fig.text(.5, .025, "训练 23,792 封；相同 50,000 维训练侧 TF-IDF；默认 predict 决策；未计算测试集分类成绩", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "Python-teamwork"})
    plt.close(fig)


def run(root: Path) -> dict:
    inputs = load_model_inputs(root)
    source_before = source_sha256(root, MODEL_SOURCES)
    feature_before = fitted_feature_state(inputs.feature_pipeline.named_steps["tfidf"])
    y_train = np.array([row["label"] for row in inputs.train_rows], dtype=np.int8)
    y_validation = np.array([row["label"] for row in inputs.validation_rows], dtype=np.int8)
    results, models, predictions = [], {}, []
    # Limit BLAS/OpenMP threads for reproducible small, sparse experiments.
    with threadpool_limits(limits=1):
        for spec in CANDIDATES:
            estimator = make_estimator(spec)
            print(json.dumps({"action": "fit_train_predict_validation", "candidate_id": spec["candidate_id"]}), flush=True)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                start = time.perf_counter()
                estimator.fit(inputs.train_matrix, y_train)
                fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            prediction = estimator.predict(inputs.validation_matrix)
            predict_seconds = time.perf_counter() - start
            iterations = [int(value) for value in np.asarray(getattr(estimator, "n_iter_", []))]
            converged = not any(issubclass(item.category, ConvergenceWarning) for item in captured)
            if spec["family"] == "logistic_regression":
                converged &= all(value < estimator.max_iter for value in iterations)
            results.append({**spec, "estimator_parameters": estimator.get_params(deep=False),
                            "converged": bool(converged), "warning_categories": sorted({item.category.__name__ for item in captured}),
                            "n_iter": iterations, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                            "metrics": validation_metrics(y_validation, prediction)})
            models[spec["candidate_id"]] = estimator
            predictions.append(prediction.astype(np.int8))
    selected_id = select_candidate(results)
    family_winners = {family: select_candidate([row for row in results if row["family"] == family])
                      for family in ("multinomial_nb", "logistic_regression")}
    if (source_before != source_sha256(root, MODEL_SOURCES)
            or feature_before != fitted_feature_state(inputs.feature_pipeline.named_steps["tfidf"])):
        raise ValueError("Source or fitted feature state changed during training.")
    model_dir = root / "artifacts/models/stage06"
    model_dir.mkdir(parents=True, exist_ok=True)
    private_outputs = {}
    for identity, estimator in models.items():
        path = model_dir / f"{identity}.joblib"
        temporary = path.with_name(path.name + ".tmp")
        joblib.dump(estimator, temporary, compress=3)
        temporary.replace(path)
        private_outputs[path.relative_to(root).as_posix()] = file_sha256(path)
    pipeline = Pipeline([("tfidf", inputs.feature_pipeline.named_steps["tfidf"]), ("classifier", models[selected_id])])
    path = model_dir / "selected_pipeline.joblib"
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(pipeline, temporary, compress=3)
    temporary.replace(path)
    private_outputs[path.relative_to(root).as_posix()] = file_sha256(path)
    path = root / "artifacts/processed/stage06/validation_predictions.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("validation_predictions.tmp.npz")
    np.savez_compressed(temporary, candidate_ids=np.array([row["candidate_id"] for row in results]),
                        mail_ids=np.array([row["mail_id"] for row in inputs.validation_rows]),
                        y_true=y_validation, predictions=np.column_stack(predictions))
    temporary.replace(path)
    private_outputs[path.relative_to(root).as_posix()] = file_sha256(path)
    csv_path = root / "artifacts/metrics/stage06_validation.csv"
    figure_path = root / "artifacts/figures/stage06_validation_comparison.png"
    write_comparison_csv(csv_path, results)
    plot_comparison(figure_path, results, selected_id)
    report = {
        "schema_version": 1, "model_source_sha256": source_before, "input_provenance": inputs.provenance,
        "versions": runtime_versions(),
        "scope": {"fit": "train_only", "selection": "validation_only", "test_predictions_computed": False, "test_metrics_computed": False},
        "decision_rule": "Unmodified estimator.predict; binary LR decision score cutoff 0 (probability 0.5), NB maximum posterior; no threshold tuning.",
        "thread_limit": 1, "train_rows": len(y_train), "validation_rows": len(y_validation), "feature_columns": inputs.train_matrix.shape[1],
        "label_mapping": {"ham": 0, "spam": 1}, "training_label_counts": {"ham": int(np.sum(y_train == 0)), "spam": int(np.sum(y_train == 1))},
        "selection_policy": SELECTION_POLICY, "experiments": results,
        "selected_candidate_id": selected_id, "family_winners": family_winners,
        "selection_status": "Validation candidate for stage seven; final cleaning/model choice remains pending, no train+validation refit.",
        "private_output_sha256": private_outputs,
        "public_output_sha256": {path.relative_to(root).as_posix(): file_sha256(path) for path in (csv_path, figure_path)},
        "timing_scope": "Per-estimator fit and validation prediction wall time; excludes input verification, TF-IDF building, serialization and plotting; machine-specific.",
        "privacy": "Only parameters, aggregate metrics and checksums are public; fitted models, vocabulary and per-mail validation predictions stay local.",
        "passed": True,
    }
    # Summary is the last write; consumers must verify all referenced checksums.
    write_json(root / "artifacts/metrics/model_selection.json", report)
    return report


def main() -> int:
    try:
        report = run(PROJECT_ROOT)
        print(json.dumps({"passed": report["passed"], "selected_candidate_id": report["selected_candidate_id"],
                          "family_winners": report["family_winners"], "experiments": report["experiments"]}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "message": "Model training failed; previous artifacts may remain."}))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
