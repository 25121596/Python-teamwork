"""Frozen candidate definitions and validation metrics for model selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# Keep the candidate order stable for reproducible reports and downstream runs.
CANDIDATES = (
    {"candidate_id": "majority", "family": "majority", "parameters": {}},
    {"candidate_id": "nb_alpha_1", "family": "multinomial_nb", "parameters": {"alpha": 1.0}},
    {"candidate_id": "lr_C_1", "family": "logistic_regression", "parameters": {"C": 1.0}},
    {"candidate_id": "nb_alpha_0.1", "family": "multinomial_nb", "parameters": {"alpha": 0.1}},
    {"candidate_id": "nb_alpha_0.5", "family": "multinomial_nb", "parameters": {"alpha": 0.5}},
    {"candidate_id": "lr_C_0.5", "family": "logistic_regression", "parameters": {"C": 0.5}},
    {"candidate_id": "lr_C_2", "family": "logistic_regression", "parameters": {"C": 2.0}},
)


SELECTION_POLICY = {
    "eligible_rows": "exclude family='majority' and rows where converged is not true",
    "ranking": [
        {"metric": "macro_f1", "order": "descending"},
        {"metric": "ham_false_positive_rate", "order": "ascending"},
        {"metric": "spam_recall", "order": "descending"},
        {"metric": "candidate_id", "order": "ascending_lexical"},
    ],
    "precision": "use exact unrounded metric values",
}


def make_estimator(spec: Mapping[str, Any]):
    """Build one estimator from a candidate specification."""
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a mapping")
    family = spec.get("family")
    parameters = spec.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise TypeError("spec parameters must be a mapping")

    if family == "majority":
        if parameters:
            raise ValueError("majority candidate does not accept parameters")
        return DummyClassifier(strategy="most_frequent")
    if family == "multinomial_nb":
        try:
            alpha = float(parameters["alpha"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("nb candidate requires a numeric alpha") from exc
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        return MultinomialNB(alpha=alpha, force_alpha=True, fit_prior=True)
    if family == "logistic_regression":
        try:
            c_value = float(parameters["C"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("lr candidate requires a numeric C") from exc
        if not np.isfinite(c_value) or c_value <= 0:
            raise ValueError("C must be finite and positive")
        return LogisticRegression(
            C=c_value,
            solver="liblinear",
            l1_ratio=0.0,
            max_iter=2000,
            tol=1e-4,
            random_state=42,
            fit_intercept=True,
            class_weight=None,
        )
    raise ValueError(f"unknown estimator family: {family!r}")


def _binary_vector(values: Sequence[Any], name: str) -> np.ndarray:
    """Convert a one-dimensional sequence of binary labels to integer labels."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    # Comparing before conversion rejects strings and fractional/non-binary values.
    try:
        valid = np.isin(array, (0, 1))
    except (TypeError, ValueError):
        valid = np.zeros(array.shape, dtype=bool)
    if not bool(np.all(valid)):
        raise ValueError(f"{name} must contain only binary labels 0 and 1")
    return array.astype(np.int64, copy=False)


def validation_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, Any]:
    """Return JSON-safe binary metrics for ham=0 and spam=1 predictions."""
    truth = _binary_vector(y_true, "y_true")
    prediction = _binary_vector(y_pred, "y_pred")
    if truth.size != prediction.size:
        raise ValueError("y_true and y_pred must have equal length")

    tn = int(np.count_nonzero((truth == 0) & (prediction == 0)))
    fp = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((truth == 1) & (prediction == 0)))
    tp = int(np.count_nonzero((truth == 1) & (prediction == 1)))

    ham_support = tn + fp
    spam_support = fn + tp
    n = int(truth.size)

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    accuracy = ratio(tn + tp, n)
    spam_precision = ratio(tp, tp + fp)
    spam_recall = ratio(tp, tp + fn)
    ham_precision = ratio(tn, tn + fn)
    ham_recall = ratio(tn, tn + fp)
    ham_f1 = ratio(2 * ham_precision * ham_recall, ham_precision + ham_recall)
    spam_f1 = ratio(2 * spam_precision * spam_recall, spam_precision + spam_recall)

    return {
        "n": n,
        "ham_support": int(ham_support),
        "spam_support": int(spam_support),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": accuracy,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "ham_f1": ham_f1,
        "spam_f1": spam_f1,
        "macro_f1": float((ham_f1 + spam_f1) / 2.0),
        "ham_false_positive_rate": ratio(fp, fp + tn),
    }


def select_candidate(results: Sequence[Mapping[str, Any]]) -> str:
    """Choose the eligible candidate using the frozen exact-value policy."""
    eligible = []
    for row in results:
        if not isinstance(row, Mapping):
            raise TypeError("each result must be a mapping")
        if row.get("family") == "majority" or row.get("converged") is not True:
            continue
        candidate_id = row.get("candidate_id")
        metrics = row.get("metrics")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("eligible results require a nonempty candidate_id")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"eligible result {candidate_id!r} requires metrics")
        try:
            macro_f1 = float(metrics["macro_f1"])
            ham_fpr = float(metrics["ham_false_positive_rate"])
            spam_recall = float(metrics["spam_recall"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"eligible result {candidate_id!r} has incomplete metrics") from exc
        if not all(np.isfinite(value) for value in (macro_f1, ham_fpr, spam_recall)):
            raise ValueError(f"eligible result {candidate_id!r} has non-finite metrics")
        eligible.append(((-macro_f1, ham_fpr, -spam_recall, candidate_id), candidate_id))

    if not eligible:
        raise ValueError("no converged non-majority candidate is eligible")
    return min(eligible, key=lambda item: item[0])[1]
