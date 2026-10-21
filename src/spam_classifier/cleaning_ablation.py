"""Controlled HTML comparison using training and validation records only."""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .experiment_io import REPRESENTATIONS
from .model_inputs import fitted_feature_state
from .model_selection import make_estimator, validation_metrics
from .text_features import make_feature_pipeline, tokenize_text

HTML_TERMS = frozenset("html body div table tr td font href width height nbsp script style meta align color border span size class head title cellpadding cellspacing".split())


def choose_representation(results: list[dict]) -> str:
    if {row["representation"] for row in results} != set(REPRESENTATIONS) or len(results) != 2:
        raise ValueError("Exactly the two predeclared representations are required.")
    eligible = [row for row in results if row["converged"] is True]
    if not eligible:
        raise ValueError("Neither comparison model converged.")
    def key(row):
        metrics = row["metrics"]
        values = (metrics["macro_f1"], metrics["ham_false_positive_rate"], metrics["spam_recall"])
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Invalid comparison metrics.")
        return (-values[0], values[1], -values[2], REPRESENTATIONS.index(row["representation"]))
    return min(eligible, key=key)["representation"]


def train_validation_only(records: dict[str, list[dict]]) -> tuple[dict, dict, dict, np.ndarray]:
    if set(records) != {"train", "validation"}:
        raise ValueError("Ablation accepts train and validation only, never test.")
    if not all(records.values()):
        raise ValueError("Training and validation must be nonempty.")
    identities = [row["mail_id"] for rows in records.values() for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Training and validation IDs overlap.")
    y_train = np.asarray([row["label"] for row in records["train"]])
    y_val = np.asarray([row["label"] for row in records["validation"]])
    if set(y_train) != {0, 1} or set(y_val) != {0, 1}:
        raise ValueError("Both classes are required in each split.")
    results, comparisons, predictions = [], {}, []
    with threadpool_limits(limits=1):
        for representation in REPRESENTATIONS:
            train = [row[representation] for row in records["train"]]
            validation = [row[representation] for row in records["validation"]]
            pipeline = Pipeline(make_feature_pipeline().steps + [
                ("classifier", make_estimator({"family": "logistic_regression", "parameters": {"C": 2.0}}))])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                pipeline.fit(train, y_train)
                elapsed = time.perf_counter() - start
            vectorizer = pipeline.named_steps["tfidf"]
            before = fitted_feature_state(vectorizer)
            predicted = pipeline.predict(validation)
            if fitted_feature_state(vectorizer) != before:
                raise ValueError("Validation transform changed the fitted vocabulary/IDF.")
            estimator = pipeline.named_steps["classifier"]
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught) and bool(np.all(estimator.n_iter_ < estimator.max_iter))
            vocabulary = set(vectorizer.vocabulary_)
            noise = sorted(vocabulary & HTML_TERMS)
            token_total = markup_total = 0
            for text in train:
                tokens = tokenize_text(text)
                token_total += len(tokens)
                markup_total += sum(t in HTML_TERMS for t in tokens)
            results.append({"representation": representation, "C": 2.0, "fit_seconds": elapsed,
                            "converged": converged, "n_iter": [int(x) for x in estimator.n_iter_],
                            "warning_categories": sorted({w.category.__name__ for w in caught}),
                            "metrics": validation_metrics(y_val, predicted), "fitted_state": before,
                            "vocabulary_size": len(vocabulary), "predeclared_markup_terms_present": noise,
                            "training_markup_token_fraction": markup_total / max(1, token_total),
                            "training_mean_characters": sum(map(len, train)) / len(train)})
            comparisons[representation] = pipeline
            predictions.append(predicted)
        chosen = choose_representation(results)
        vectorizer = comparisons[chosen].named_steps["tfidf"]
        train_x = vectorizer.transform([row[chosen] for row in records["train"]])
        val_x = vectorizer.transform([row[chosen] for row in records["validation"]])
        models = {"lr": comparisons[chosen]}
        for name, family, parameters in (("majority", "majority", {}), ("nb", "multinomial_nb", {"alpha": 0.1})):
            estimator = make_estimator({"family": family, "parameters": parameters}).fit(train_x, y_train)
            models[name] = Pipeline([("tfidf", vectorizer), ("classifier", estimator)])
        model_order = ["majority", "nb", "lr"]
        final_predictions = np.column_stack([models[name].named_steps["classifier"].predict(val_x) for name in model_order])
    summary = {
        "scope": "train/validation only; both representations fit their own training vocabulary and IDF",
        "train_rows": len(y_train), "validation_rows": len(y_val), "comparisons": results,
        "representation": chosen,
        "selection_rule": "validation macro-F1 descending, ham FPR ascending, spam recall descending, exact tie prefers text_clean",
        "final_validation_metrics": {name: validation_metrics(y_val, final_predictions[:, i]) for i, name in enumerate(model_order)},
        "changed_text_rows": {name: sum(row["text_clean"] != row["text_unstripped_html"] for row in rows) for name, rows in records.items()},
        "markup_note": "Predeclared HTML-like words are a noise proxy; some also occur in legitimate plain text. No full vocabulary is published.",
    }
    return summary, comparisons, models, np.column_stack(predictions + [final_predictions[:, i] for i in range(3)])
