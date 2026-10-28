"""Final metrics, private review candidates and public whitelist coefficients."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .model_selection import validation_metrics
from .text_features import PUBLIC_TERMS


def score_locked_models(models: dict, records: list[dict], representation: str):
    if set(models) != {"majority", "nb", "lr"} or not records:
        raise ValueError("Exactly three locked models and a nonempty test set are required.")
    truth = np.asarray([r["label"] for r in records])
    texts = [r[representation] for r in records]
    predictions, metrics, reviews, diagnostics = [], {}, {}, {}
    for name in ("majority", "nb", "lr"):
        pipeline = models[name]
        predicted = pipeline.predict(texts)
        metrics[name] = validation_metrics(truth, predicted)
        predictions.append(predicted)
        if name == "majority":
            continue
        scores = (pipeline.decision_function(texts) if name == "lr" else
                  np.diff(pipeline.predict_log_proba(texts), axis=1).ravel())
        groups, group_diagnostics = {}, {}
        for label, actual, predicted_value in (("FP", 0, 1), ("FN", 1, 0)):
            indices = [i for i, (a, p) in enumerate(zip(truth, predicted)) if a == actual and p == predicted_value]
            # Fixed selection rule, independent of semantic narratives.
            selected = sorted(indices, key=lambda i: (-abs(float(scores[i])), records[i]["mail_id"]))[:15]
            groups[label] = [{
                "mail_id": records[i]["mail_id"], "true_label": int(truth[i]), "prediction": int(predicted[i]),
                "score_spam_direction": float(scores[i]), "subject_clean": records[i].get("subject_clean", ""),
                "text_clean": records[i]["text_clean"], "has_html": bool(records[i].get("has_html")),
                "parse_status": records[i].get("parse_status", "unknown"),
                "encoding_warning": bool(records[i].get("encoding_warnings")),
                "reviewed": False, "reason_code": "", "safe_paraphrase": "", "reviewer_note": "",
            } for i in selected]
            group_diagnostics[label] = {
                "all_errors": len(indices), "review_candidates": len(selected),
                "parse_status_counts": dict(Counter(records[i].get("parse_status", "unknown") for i in indices)),
                "html_count": sum(bool(records[i].get("has_html")) for i in indices),
                "encoding_warning_count": sum(bool(records[i].get("encoding_warnings")) for i in indices),
                "short_text_le_30_characters_count": sum(len(records[i]["text_clean"]) <= 30 for i in indices),
            }
        reviews[name], diagnostics[name] = groups, group_diagnostics
    return metrics, np.column_stack(predictions), reviews, diagnostics


def public_coefficients(lr_pipeline) -> dict:
    vectorizer = lr_pipeline.named_steps["tfidf"]
    classifier = lr_pipeline.named_steps["classifier"]
    if np.asarray(classifier.classes_).tolist() != [0, 1] or classifier.coef_.shape != (1, len(vectorizer.vocabulary_)):
        raise ValueError("Expected a binary LR whose positive class is spam.")
    candidates = [{"term": term, "coefficient": float(classifier.coef_[0, index])}
                  for term, index in vectorizer.vocabulary_.items() if term in PUBLIC_TERMS]
    return {"scope": "Top signed LR coefficients within the predeclared public whitelist, not the full vocabulary; association is not causation.",
            "ham": sorted((r for r in candidates if r["coefficient"] < 0), key=lambda r: (r["coefficient"], r["term"]))[:12],
            "spam": sorted((r for r in candidates if r["coefficient"] > 0), key=lambda r: (-r["coefficient"], r["term"]))[:12]}
