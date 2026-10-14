"""Fit TF-IDF on frozen training mail; publish aggregates, keep features local."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy import sparse

from spam_classifier.dataset import file_sha256, load_dataset, source_sha256
from spam_classifier.text_features import PUBLIC_TERMS, make_feature_pipeline, tokenize_text, tokenizer_fingerprint

FEATURE_SOURCES = ("src/spam_classifier/dataset.py", "src/spam_classifier/text_features.py", "scripts/build_features.py")


def feature_state(vectorizer) -> dict:
    vocabulary = json.dumps(sorted((term, int(index)) for term, index in vectorizer.vocabulary_.items()),
                            ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "vocabulary_sha256": hashlib.sha256(vocabulary).hexdigest(),
        "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
    }


def matrix_summary(matrix) -> dict:
    matrix = matrix.tocsr()
    count = int(np.count_nonzero(np.diff(matrix.indptr) == 0))
    return {
        "rows": matrix.shape[0], "columns": matrix.shape[1], "nnz": matrix.nnz,
        "density": matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1]),
        "zero_rows": count, "zero_row_fraction": count / max(1, matrix.shape[0]),
        "csr_bytes": int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
        "format": matrix.format, "dtype": str(matrix.dtype),
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    temporary.replace(path)


def word_coverage(training_records: list[dict]) -> tuple[list[dict], dict]:
    counts = {label: Counter() for label in ("ham", "spam")}
    totals = Counter(row["label_name"] for row in training_records)
    tokenless = numeric_documents = 0
    for number, row in enumerate(training_records, 1):
        tokens = set(tokenize_text(row["text_clean"]))
        counts[row["label_name"]].update(tokens)
        tokenless += not tokens
        numeric_documents += any(token.isdecimal() for token in tokens)
        if number % 5000 == 0:
            print(json.dumps({"training_coverage_processed": number, "total": len(training_records)}), flush=True)
    terms = sorted(set(counts["ham"]) | set(counts["spam"]),
                   key=lambda token: (-counts["ham"][token] - counts["spam"][token], token))
    rows = [{"term": term, "ham_documents": counts["ham"][term], "spam_documents": counts["spam"][term],
             "ham_coverage": counts["ham"][term] / totals["ham"], "spam_coverage": counts["spam"][term] / totals["spam"]}
            for term in terms]
    # Whitelist and minimum coverage threshold fixed before inspecting counts.
    public = [row for row in rows if row["term"] in PUBLIC_TERMS and row["ham_documents"] + row["spam_documents"] >= 20]
    return rows, {
        "scope": "Training representatives only; each term counted once per mail.",
        "training_label_counts": dict(totals), "unique_token_types_before_tfidf_filter": len(terms),
        "numeric_token_types": sum(term.isdecimal() for term in terms),
        "training_documents_with_numeric_token": numeric_documents, "training_tokenless_documents": tokenless,
        "public_term_policy": "Predeclared generic whitelist, at least 20 training documents overall; no full vocabulary or numeric terms published.",
        "public_term_coverage": public,
    }


def plot_public_words(coverage: dict, path: Path) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.7), sharex=True, layout="constrained")
    colors = {"ham": "#326b99", "spam": "#d98748"}
    for ax, label, title in zip(axes, ("ham", "spam"), ("正常邮件（ham）", "垃圾邮件（spam）")):
        rows = sorted(coverage["public_term_coverage"], key=lambda row: (-row[label + "_coverage"], row["term"]))[:12]
        rows.reverse()
        values = [row[label + "_coverage"] * 100 for row in rows]
        ax.barh([row["term"] for row in rows], values, color=colors[label])
        ax.set_title(f"{title} | n={coverage['training_label_counts'][label]:,}")
        ax.set_xlabel("该类训练邮件中包含此词的比例（%）")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("训练集通用词覆盖率：预设公开词表内的前 12 个词", fontsize=15)
    fig.text(.5, -.025, "每封邮件每词最多计一次；仅公开预设通用词且总覆盖 ≥20 封。完整词频留在本地；高频不等于模型关联词。", ha="center", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "Python-teamwork"})
    plt.close(fig)


def main() -> int:
    try:
        bundle = load_dataset(PROJECT_ROOT)
        fingerprint = tokenizer_fingerprint()
        source_sha = source_sha256(PROJECT_ROOT, FEATURE_SOURCES)
        pipeline = make_feature_pipeline()
        training_text = [row["text_clean"] for row in bundle.splits["train"]]
        print(json.dumps({"action": "fit_training_tfidf", "documents": len(training_text)}), flush=True)
        matrices = {"train": pipeline.fit_transform(training_text).tocsr()}
        state = feature_state(pipeline.named_steps["tfidf"])
        for name in ("validation", "test"):
            print(json.dumps({"action": "transform_only", "split": name, "documents": len(bundle.splits[name])}), flush=True)
            matrices[name] = pipeline.transform([row["text_clean"] for row in bundle.splits[name]]).tocsr()
        if feature_state(pipeline.named_steps["tfidf"]) != state:
            raise ValueError("Heldout transformation changed fitted state.")
        frequencies, coverage = word_coverage(bundle.splits["train"])
        if fingerprint != tokenizer_fingerprint() or source_sha != source_sha256(PROJECT_ROOT, FEATURE_SOURCES):
            raise ValueError("Feature source or dictionary changed while running.")
        private = PROJECT_ROOT / "artifacts/processed/features"
        private.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for name, matrix in matrices.items():
            path = private / f"{name}.npz"
            temporary = path.with_name(name + ".tmp.npz")
            sparse.save_npz(temporary, matrix, compressed=True)
            temporary.replace(path)
            outputs[path.relative_to(PROJECT_ROOT).as_posix()] = file_sha256(path)
        ordering = {name: [{"mail_id": row["mail_id"], "label": row["label"]} for row in rows]
                    for name, rows in bundle.splits.items()}
        order_path = private / "row_order.json"
        write_json(order_path, ordering)
        outputs[order_path.relative_to(PROJECT_ROOT).as_posix()] = file_sha256(order_path)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=("term", "ham_documents", "spam_documents", "ham_coverage", "spam_coverage"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(frequencies)
        full_words = private / "training_word_coverage.csv"
        full_words.write_bytes(stream.getvalue().encode("utf-8"))
        outputs[full_words.relative_to(PROJECT_ROOT).as_posix()] = file_sha256(full_words)
        model_path = PROJECT_ROOT / "artifacts/models/tfidf_pipeline.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_name(model_path.name + ".tmp")
        joblib.dump(pipeline, temporary, compress=3)
        temporary.replace(model_path)
        outputs[model_path.relative_to(PROJECT_ROOT).as_posix()] = file_sha256(model_path)
        plot_public_words(coverage, PROJECT_ROOT / "artifacts/figures/stage05_training_word_coverage.png")
        vectorizer = pipeline.named_steps["tfidf"]
        summary = {
            "schema_version": 1, "input_provenance": bundle.provenance, "feature_source_sha256": source_sha,
            "tokenizer": fingerprint,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
            "configuration": {key: vectorizer.get_params()[key] for key in ("min_df", "max_df", "max_features", "sublinear_tf", "smooth_idf", "use_idf", "norm", "lowercase", "token_pattern")},
            "ngram_range": list(vectorizer.ngram_range), "dtype": "float64", "fit_scope": "train_only",
            "matrices": {name: matrix_summary(matrix) for name, matrix in matrices.items()},
            "fitted_state": state, "heldout_transform_preserved_fitted_state": True,
            "training_word_coverage": coverage, "private_output_sha256": outputs,
            "privacy": "All texts, full vocabulary, word frequencies, fitted pipeline and sparse matrices remain local; public words use the predeclared generic whitelist.",
            "passed": True,
        }
        write_json(PROJECT_ROOT / "artifacts/audit/feature_summary.json", summary)
        print(json.dumps({"matrices": summary["matrices"], "token_types": coverage["unique_token_types_before_tfidf_filter"], "passed": True}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "message": "Check verified inputs, feature source and runtime; previous outputs may remain."}))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
