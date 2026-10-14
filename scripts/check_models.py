"""Independently verify the validation-only model comparison artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from spam_classifier.dataset import file_sha256, source_sha256
from spam_classifier.model_selection import CANDIDATES, SELECTION_POLICY, make_estimator


PRIVATE_OUTPUTS = {
    *(f"artifacts/models/stage06/{spec['candidate_id']}.joblib" for spec in CANDIDATES),
    "artifacts/models/stage06/selected_pipeline.joblib",
    "artifacts/processed/stage06/validation_predictions.npz",
}
PUBLIC_OUTPUTS = {
    "artifacts/metrics/stage06_validation.csv",
    "artifacts/figures/stage06_validation_comparison.png",
}
METRIC_KEYS = (
    "n", "ham_support", "spam_support", "confusion_matrix", "accuracy",
    "spam_precision", "spam_recall", "ham_f1", "spam_f1", "macro_f1",
    "ham_false_positive_rate",
)


def independent_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, Any]:
    """Calculate binary metrics directly from labels, without model-selection helpers."""
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.size == 0 or truth.size != prediction.size:
        raise ValueError("truth and prediction must be nonempty one-dimensional vectors of equal length")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(np.isin(prediction, (0, 1))):
        raise ValueError("truth and prediction must be binary")
    truth = truth.astype(np.int64, copy=False)
    prediction = prediction.astype(np.int64, copy=False)

    tn = int(np.count_nonzero((truth == 0) & (prediction == 0)))
    fp = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((truth == 1) & (prediction == 0)))
    tp = int(np.count_nonzero((truth == 1) & (prediction == 1)))
    ham_n, spam_n = tn + fp, fn + tp

    def divide(a: int | float, b: int | float) -> float:
        return float(a / b) if b else 0.0

    # F1 comes directly from confusion counts rather than reusing precision/recall.
    ham_precision = divide(tn, tn + fn)
    spam_precision = divide(tp, tp + fp)
    spam_recall = divide(tp, spam_n)
    ham_f1 = divide(2 * tn, 2 * tn + fp + fn)
    spam_f1 = divide(2 * tp, 2 * tp + fp + fn)
    return {
        "n": int(truth.size),
        "ham_support": int(ham_n),
        "spam_support": int(spam_n),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": divide(tn + tp, truth.size),
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "ham_f1": ham_f1,
        "spam_f1": spam_f1,
        "macro_f1": (ham_f1 + spam_f1) / 2.0,
        "ham_false_positive_rate": divide(fp, ham_n),
    }


def rank_candidate_ids(rows: Sequence[Mapping[str, Any]], *, family: str | None = None) -> str:
    """Apply the frozen ranking policy independently to candidate result rows."""
    eligible = []
    for row in rows:
        if row.get("family") == "majority" or row.get("converged") is not True:
            continue
        if family is not None and row.get("family") != family:
            continue
        metrics = row["metrics"]
        values = (float(metrics["macro_f1"]), float(metrics["ham_false_positive_rate"]), float(metrics["spam_recall"]))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate ranking metrics must be finite")
        eligible.append(((-values[0], values[1], -values[2], str(row["candidate_id"])), str(row["candidate_id"])))
    if not eligible:
        raise ValueError("no eligible candidates for independent ranking")
    return min(eligible, key=lambda item: item[0])[1]


def validate_prediction_arrays(
    candidate_ids: Sequence[Any],
    mail_ids: Sequence[Any],
    y_true: Sequence[Any],
    predictions: Any,
    expected_candidate_ids: Sequence[str],
    expected_mail_ids: Sequence[str],
    expected_y_true: Sequence[int],
) -> np.ndarray:
    """Validate archive shape, labels and frozen row/column order; return binary matrix."""
    ids = np.asarray(candidate_ids)
    mail = np.asarray(mail_ids)
    truth = np.asarray(y_true)
    predicted = np.asarray(predictions)
    if ids.ndim != 1 or mail.ndim != 1 or truth.ndim != 1 or predicted.ndim != 2:
        raise ValueError("prediction archive arrays have invalid dimensions")
    if ids.tolist() != list(expected_candidate_ids):
        raise ValueError("prediction candidate order does not match the frozen catalog")
    if mail.tolist() != list(expected_mail_ids):
        raise ValueError("prediction mail ID order does not match validation metadata")
    expected_truth = np.asarray(expected_y_true, dtype=np.int8)
    if not np.array_equal(truth, expected_truth):
        raise ValueError("prediction truth labels do not match validation metadata order")
    if predicted.shape != (len(expected_mail_ids), len(expected_candidate_ids)):
        raise ValueError("prediction matrix shape does not match the frozen validation set")
    if not np.all(np.isin(predicted, (0, 1))):
        raise ValueError("prediction matrix contains non-binary values")
    return predicted.astype(np.int8, copy=False)


def _close_array(left: Any, right: Any) -> bool:
    try:
        return np.asarray(left).shape == np.asarray(right).shape and bool(
            np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=1e-12, equal_nan=False)
        )
    except (TypeError, ValueError):
        return False


def _vectorizer_state(pipeline: Any) -> dict[str, Any]:
    vectorizer = pipeline.named_steps["tfidf"]
    vocabulary = sorted((str(term), int(index)) for term, index in vectorizer.vocabulary_.items())
    return {
        "vocabulary_sha256": hashlib.sha256(
            json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
        "feature_count": int(len(vocabulary)),
        "params": vectorizer.get_params(deep=False),
    }


def _classifier_state(estimator: Any) -> dict[str, Any]:
    state: dict[str, Any] = {"class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
                             "classes": np.asarray(estimator.classes_).tolist(),
                             "n_features_in_": np.asarray(estimator.n_features_in_)}
    if hasattr(estimator, "feature_count_"):
        state["feature_count_"] = np.asarray(estimator.feature_count_)
        state["class_count_"] = np.asarray(estimator.class_count_)
        state["feature_log_prob_"] = np.asarray(estimator.feature_log_prob_)
        state["class_log_prior_"] = np.asarray(estimator.class_log_prior_)
    if hasattr(estimator, "coef_"):
        state["coef_"] = np.asarray(estimator.coef_)
        state["intercept_"] = np.asarray(estimator.intercept_)
    if hasattr(estimator, "class_prior_"):
        state["class_prior_"] = np.asarray(estimator.class_prior_)
    return state


def reported_n_iter_matches(estimator: Any, record: Mapping[str, Any]) -> bool:
    actual = [int(value) for value in np.asarray(getattr(estimator, "n_iter_", [])).reshape(-1)]
    reported = record.get("n_iter")
    return isinstance(reported, list) and all(type(value) is int for value in reported) and reported == actual


def timing_metadata_valid(record: Mapping[str, Any]) -> bool:
    try:
        values = (float(record["fit_seconds"]), float(record["predict_seconds"]))
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) and value >= 0.0 for value in values)


def warning_metadata_valid(
    record: Mapping[str, Any], family: str, max_iter: int | None,
) -> bool:
    categories = record.get("warning_categories")
    if not isinstance(categories, list) or any(not isinstance(value, str) for value in categories):
        return False
    if categories != sorted(set(categories)):
        return False
    expected_converged = "ConvergenceWarning" not in categories
    if family == "logistic_regression" and max_iter is not None:
        iterations = record.get("n_iter")
        if not isinstance(iterations, list) or not all(type(value) is int for value in iterations):
            return False
        expected_converged &= all(value < max_iter for value in iterations)
    return record.get("converged") is expected_converged


def construction_passed(summary: Mapping[str, Any]) -> bool:
    return summary.get("passed") is True


def _states_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        if isinstance(left[key], np.ndarray) or isinstance(right[key], np.ndarray):
            if not _close_array(left[key], right[key]):
                return False
        elif left[key] != right[key]:
            return False
    return True


def _metrics_match(saved: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    if set(saved) != set(METRIC_KEYS):
        return False
    for key in METRIC_KEYS:
        if key == "confusion_matrix":
            if saved[key] != actual[key]:
                return False
        elif key in {"n", "ham_support", "spam_support"}:
            if saved[key] != actual[key]:
                return False
        else:
            try:
                if not math.isclose(float(saved[key]), float(actual[key]), rel_tol=0.0, abs_tol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def check(root: Path) -> dict[str, Any]:
    # Imported inside the checker so pure audit helpers remain testable while the
    # upstream model-input stage is still being developed.
    from spam_classifier.model_inputs import MODEL_SOURCES, load_model_inputs, runtime_versions

    root = Path(root).resolve()
    artifact_root = root / "artifacts"
    summary_path = artifact_root / "metrics/model_selection.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1:
        raise ValueError("Unsupported model selection summary schema.")
    if not construction_passed(summary):
        raise ValueError("Model selection construction summary is not marked passed.")

    # Verify every declared output before opening any untrusted joblib artifact.
    private_hashes = summary.get("private_output_sha256")
    public_hashes = summary.get("public_output_sha256")
    if not isinstance(private_hashes, dict) or set(private_hashes) != PRIVATE_OUTPUTS:
        raise ValueError("Private model output path set does not match the frozen contract.")
    if not isinstance(public_hashes, dict) or set(public_hashes) != PUBLIC_OUTPUTS:
        raise ValueError("Public model output path set does not match the frozen contract.")
    private_checksums = {name: file_sha256(root / name) == digest for name, digest in private_hashes.items()}
    public_checksums = {name: file_sha256(root / name) == digest for name, digest in public_hashes.items()}
    if not all(private_checksums.values()):
        raise ValueError("Private model artifact checksum mismatch; saved models were not loaded.")

    model_inputs = load_model_inputs(root)
    train_x, validation_x = model_inputs.train_matrix, model_inputs.validation_matrix
    if not sparse.isspmatrix_csr(train_x) or not sparse.isspmatrix_csr(validation_x):
        raise ValueError("Model input matrices must be CSR.")
    train_y = np.asarray([row["label"] for row in model_inputs.train_rows], dtype=np.int8)
    validation_y = np.asarray([row["label"] for row in model_inputs.validation_rows], dtype=np.int8)
    validation_ids = [str(row["mail_id"]) for row in model_inputs.validation_rows]
    expected_ids = [spec["candidate_id"] for spec in CANDIDATES]

    if summary.get("input_provenance") != model_inputs.provenance:
        raise ValueError("Model input provenance differs from generation.")
    if summary.get("model_source_sha256") != source_sha256(root, MODEL_SOURCES):
        raise ValueError("Model source fingerprint differs from generation.")
    if summary.get("versions") != runtime_versions():
        raise ValueError("Runtime versions differ from model generation.")
    expected_scope = {"fit": "train_only", "selection": "validation_only",
                      "test_predictions_computed": False, "test_metrics_computed": False}
    if summary.get("scope") != expected_scope:
        raise ValueError("Model selection scope does not match the validation-only contract.")
    if summary.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("Selection policy differs from the frozen policy.")
    if len(summary.get("experiments", [])) != len(CANDIDATES):
        raise ValueError("Model summary does not contain the frozen candidate catalog.")

    npz_path = root / "artifacts/processed/stage06/validation_predictions.npz"
    with np.load(npz_path, allow_pickle=False) as saved_predictions:
        predictions = validate_prediction_arrays(
            saved_predictions["candidate_ids"], saved_predictions["mail_ids"],
            saved_predictions["y_true"], saved_predictions["predictions"], expected_ids,
            validation_ids, validation_y,
        )

    experiments = summary["experiments"]
    if [row.get("candidate_id") for row in experiments] != expected_ids:
        raise ValueError("Summary candidate order does not match the frozen catalog.")
    candidate_checks: dict[str, dict[str, bool]] = {}
    independent_results = []
    loaded_models = {}
    for column, (spec, record) in enumerate(zip(CANDIDATES, experiments, strict=True)):
        candidate_id = spec["candidate_id"]
        path = root / f"artifacts/models/stage06/{candidate_id}.joblib"
        estimator = joblib.load(path)
        loaded_models[candidate_id] = estimator
        expected_estimator = make_estimator(spec)
        observed = estimator.predict(validation_x)
        truth = validation_y
        metrics = independent_metrics(truth, observed)
        stored_metrics = record.get("metrics", {})
        params_match = estimator.get_params(deep=False) == record.get("estimator_parameters")
        frozen_params_match = estimator.get_params(deep=False) == expected_estimator.get_params(deep=False)
        catalog_match = record.get("family") == spec["family"] and record.get("parameters") == spec["parameters"]
        prediction_match = np.array_equal(np.asarray(observed, dtype=np.int8), predictions[:, column])
        summary_prediction_match = prediction_match and _metrics_match(stored_metrics, metrics)
        class_match = np.array_equal(estimator.classes_, np.asarray([0, 1]))
        expected_type_match = type(estimator) is type(expected_estimator)
        if spec["family"] == "majority":
            expected_type_match &= estimator.get_params(deep=False) == expected_estimator.get_params(deep=False)

        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as refit_warnings:
            warnings.simplefilter("always")
            refit = make_estimator(spec).fit(train_x, train_y)
        refit_warning_categories = sorted({item.category.__name__ for item in refit_warnings})
        refit_has_convergence_warning = any(
            issubclass(item.category, ConvergenceWarning) for item in refit_warnings
        )
        refit_prediction = refit.predict(validation_x)
        refit_state_match = _states_match(_classifier_state(estimator), _classifier_state(refit))
        refit_prediction_match = np.array_equal(np.asarray(refit_prediction, dtype=np.int8), predictions[:, column])
        state = _classifier_state(estimator)
        state_numeric_match = _states_match(state, _classifier_state(refit))
        converged_actual = True
        if hasattr(estimator, "n_iter_"):
            n_iter = np.asarray(estimator.n_iter_, dtype=np.int64).reshape(-1)
            converged_actual = bool(np.all(n_iter < int(estimator.get_params(deep=False)["max_iter"])))
        convergence_match = record.get("converged") is converged_actual
        max_iter = int(estimator.get_params(deep=False)["max_iter"]) if hasattr(estimator, "n_iter_") else None
        candidate_checks[candidate_id] = {
            "catalog_and_spec_match": bool(catalog_match),
            "estimator_class_and_parameters_match_summary": bool(expected_type_match and params_match and class_match),
            "estimator_parameters_match_frozen_candidate": bool(frozen_params_match),
            "stored_validation_predictions_match": bool(prediction_match),
            "independent_metrics_match_summary": bool(_metrics_match(stored_metrics, metrics)),
            "summary_prediction_and_metrics_match": bool(summary_prediction_match),
            "train_only_refit_predictions_match": bool(refit_prediction_match),
            "train_only_refit_fitted_state_matches": bool(refit_state_match and state_numeric_match),
            "convergence_matches_summary": bool(convergence_match),
            "reported_n_iter_matches_loaded_estimator": reported_n_iter_matches(estimator, record),
            "timings_are_finite_and_nonnegative": timing_metadata_valid(record),
            "warning_categories_match_refit": record.get("warning_categories") == refit_warning_categories
                and ("ConvergenceWarning" in record.get("warning_categories", [])) is refit_has_convergence_warning,
            "warning_categories_and_convergence_are_consistent": warning_metadata_valid(record, spec["family"], max_iter),
        }
        independent_results.append({"candidate_id": candidate_id, "family": spec["family"],
                                    "converged": converged_actual, "metrics": metrics})

    try:
        winner = rank_candidate_ids(independent_results)
        family_winners = {
            family: rank_candidate_ids(independent_results, family=family)
            for family in ("multinomial_nb", "logistic_regression")
        }
    except ValueError as exc:
        raise ValueError("Independent selection could not identify all required winners.") from exc
    selected_id = summary.get("selected_candidate_id")
    selected_path = root / "artifacts/models/stage06/selected_pipeline.joblib"
    selected_pipeline = joblib.load(selected_path)
    selected_classifier = selected_pipeline.named_steps["classifier"]
    original_classifier = loaded_models.get(winner)
    tfidf_equal = _vectorizer_state(selected_pipeline) == _vectorizer_state(model_inputs.feature_pipeline)
    selected_classifier_equal = (
        original_classifier is not None
        and type(selected_classifier) is type(original_classifier)
        and selected_classifier.get_params(deep=False) == original_classifier.get_params(deep=False)
        and _states_match(_classifier_state(selected_classifier), _classifier_state(original_classifier))
    )
    selected_prediction_equal = np.array_equal(
        np.asarray(selected_classifier.predict(validation_x), dtype=np.int8), predictions[:, expected_ids.index(winner)]
    )
    synthetic_texts = [
        "这是用于分类流程检查的合成中文文本。",
        "请确认下周会议时间和项目进度。",
        "",
    ]
    synthetic_matrix = model_inputs.feature_pipeline.transform(synthetic_texts).tocsr()
    expected_synthetic_predictions = original_classifier.predict(synthetic_matrix)
    selected_synthetic_predictions = selected_pipeline.predict(synthetic_texts)
    selected_synthetic_prediction_equal = np.array_equal(
        np.asarray(selected_synthetic_predictions, dtype=np.int8),
        np.asarray(expected_synthetic_predictions, dtype=np.int8),
    )
    actual_training_label_counts = {
        "ham": int(np.count_nonzero(train_y == 0)),
        "spam": int(np.count_nonzero(train_y == 1)),
    }

    checks = {
        "private_output_checksums_match_before_model_load": all(private_checksums.values()),
        "public_output_checksums_match": all(public_checksums.values()),
        "input_provenance_matches": summary["input_provenance"] == model_inputs.provenance,
        "model_source_fingerprint_matches": summary["model_source_sha256"] == source_sha256(root, MODEL_SOURCES),
        "runtime_versions_match": summary["versions"] == runtime_versions(),
        "validation_only_scope_is_explicit": summary["scope"] == expected_scope,
        "selection_policy_matches_frozen_contract": summary["selection_policy"] == SELECTION_POLICY,
        "selected_candidate_matches_independent_ranking": selected_id == winner,
        "family_winners_match_independent_ranking": summary.get("family_winners") == family_winners,
        "selected_pipeline_tfidf_state_matches_training_pipeline": bool(tfidf_equal),
        "selected_pipeline_classifier_matches_winning_candidate": bool(selected_classifier_equal),
        "selected_pipeline_predicts_stored_validation_column": bool(selected_prediction_equal),
        "selected_pipeline_matches_training_pipeline_on_synthetic_chinese_text": bool(selected_synthetic_prediction_equal),
        "summary_dimensions_match_loaded_inputs": summary.get("train_rows") == len(model_inputs.train_rows)
            and summary.get("validation_rows") == len(model_inputs.validation_rows)
            and summary.get("feature_columns") == train_x.shape[1] == validation_x.shape[1],
        "summary_label_mapping_and_training_counts_match": summary.get("label_mapping") == {"ham": 0, "spam": 1}
            and summary.get("training_label_counts") == actual_training_label_counts,
        "summary_thread_limit_is_one": summary.get("thread_limit") == 1,
        "all_candidate_checks_pass": all(all(items.values()) for items in candidate_checks.values()),
    }
    return {
        "schema_version": 1,
        "checks": checks,
        "per_candidate_checks": candidate_checks,
        "validator_source_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "model_selection_summary_sha256": file_sha256(summary_path),
        "scope": {
            "fit": "train_only",
            "selection": "validation_only",
            "test_predictions_computed": False,
            "test_metrics_computed": False,
        },
        "passed": all(checks.values()),
    }


def main() -> int:
    target = PROJECT_ROOT / "artifacts/audit/model_validation.json"
    try:
        report = check(PROJECT_ROOT)
        status = 0 if report["passed"] else 1
    except Exception as exc:
        # Keep failures useful while avoiding model or message contents in public output.
        report = {
            "schema_version": 1,
            "checks": {},
            "per_candidate_checks": {},
            "validator_source_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "model_selection_summary_sha256": None,
            "scope": {"fit": "train_only", "selection": "validation_only",
                      "test_predictions_computed": False, "test_metrics_computed": False},
            "passed": False,
            "error_type": type(exc).__name__,
            "error": "Model validation failed; review inputs and regenerate stage 06 artifacts.",
        }
        status = 2
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return status


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
