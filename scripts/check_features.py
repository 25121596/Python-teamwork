"""Verify training-only IDF, matrix alignment and local feature artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import scipy
import sklearn
from scipy import sparse

from spam_classifier.dataset import file_sha256, load_dataset, source_sha256
from spam_classifier.text_features import PUBLIC_TERMS, make_feature_pipeline, tokenize_text, tokenizer_fingerprint


def state(vectorizer):
    vocabulary = sorted((term, int(index)) for term, index in vectorizer.vocabulary_.items())
    return {
        "vocabulary_sha256": hashlib.sha256(json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
    }


def check(root: Path) -> dict:
    art = root / "artifacts"
    summary = json.loads((art / "audit/feature_summary.json").read_text(encoding="utf-8"))
    bundle = load_dataset(root)
    sources = ("src/spam_classifier/dataset.py", "src/spam_classifier/text_features.py", "scripts/build_features.py")
    expected_outputs = {f"artifacts/processed/features/{name}.npz" for name in bundle.splits}
    expected_outputs |= {"artifacts/processed/features/row_order.json", "artifacts/processed/features/training_word_coverage.csv", "artifacts/models/tfidf_pipeline.joblib"}
    output_checksums = set(summary["private_output_sha256"]) == expected_outputs and all(
        file_sha256(root / name) == summary["private_output_sha256"][name] for name in expected_outputs)
    if not output_checksums:
        raise ValueError("Private feature outputs changed; regenerate before loading the local pipeline.")
    if source_sha256(root, sources) != summary["feature_source_sha256"] or tokenizer_fingerprint() != summary["tokenizer"]:
        raise ValueError("Feature rules or tokenizer dictionary changed.")
    pipeline = joblib.load(art / "models/tfidf_pipeline.joblib")
    vectorizer = pipeline.named_steps["tfidf"]
    fitted_before = state(vectorizer)
    order = json.loads((art / "processed/features/row_order.json").read_text(encoding="utf-8"))
    coverage = {label: Counter() for label in ("ham", "spam")}
    term_frequencies = Counter()
    tokenless = numeric_documents = 0
    training = bundle.splits["train"]
    for number, row in enumerate(training, 1):
        tokens = tokenize_text(row["text_clean"])
        unique = set(tokens)
        coverage[row["label_name"]].update(unique)
        term_frequencies.update(tokens)
        tokenless += not unique
        numeric_documents += any(token.isdecimal() for token in unique)
        if number % 5000 == 0:
            print(json.dumps({"verified_training_document_frequencies": number, "total": len(training)}), flush=True)
    all_terms = set(coverage["ham"]) | set(coverage["spam"])
    names = vectorizer.get_feature_names_out()
    dfs = np.array([coverage["ham"][term] + coverage["spam"][term] for term in names])
    expected_idf = np.log((1 + len(training)) / (1 + dfs)) + 1
    eligible = {term for term in all_terms if coverage["ham"][term] + coverage["spam"][term] >= 2}
    # When the cap binds, all selected terms must at least satisfy the training
    # frequency cutoff; ties at the cutoff may use the library's fixed ordering.
    selection_valid = len(names) == min(50000, len(eligible)) and set(names) <= eligible
    if len(eligible) <= 50000:
        selection_valid &= set(names) == eligible
    elif selection_valid:
        kept_min = min(term_frequencies[term] for term in names)
        discarded_max = max(term_frequencies[term] for term in eligible - set(names))
        selection_valid &= kept_min >= discarded_max
    label_counts = Counter(row["label_name"] for row in training)
    word_table_valid = True
    with (art / "processed/features/training_word_coverage.csv").open(encoding="utf-8", newline="") as stream:
        word_rows = list(csv.DictReader(stream))
    word_table_valid &= len(word_rows) == len({row["term"] for row in word_rows}) == len(all_terms)
    word_table_valid &= {row["term"] for row in word_rows} == all_terms
    for row in word_rows:
        for label in ("ham", "spam"):
            count = coverage[label][row["term"]]
            word_table_valid &= int(row[label + "_documents"]) == count
            word_table_valid &= abs(float(row[label + "_coverage"]) - count / label_counts[label]) < 1e-12
    public_expected = []
    for term in sorted(all_terms, key=lambda term: (-coverage["ham"][term] - coverage["spam"][term], term)):
        if term in PUBLIC_TERMS and coverage["ham"][term] + coverage["spam"][term] >= 20:
            public_expected.append({"term": term, "ham_documents": coverage["ham"][term], "spam_documents": coverage["spam"][term],
                                    "ham_coverage": coverage["ham"][term] / label_counts["ham"], "spam_coverage": coverage["spam"][term] / label_counts["spam"]})
    matrix_checks = {}
    for name, records in bundle.splits.items():
        matrix = sparse.load_npz(art / f"processed/features/{name}.npz")
        transformed = pipeline.transform([row["text_clean"] for row in records]).tocsr()
        difference = matrix - transformed
        norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
        nonempty = np.diff(matrix.indptr) > 0
        details = summary["matrices"][name]
        current = {
            "rows": matrix.shape[0], "columns": matrix.shape[1], "nnz": matrix.nnz,
            "density": matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1]),
            "zero_rows": int(np.count_nonzero(~nonempty)), "zero_row_fraction": float(np.mean(~nonempty)),
            "csr_bytes": int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
            "format": matrix.format, "dtype": str(matrix.dtype),
        }
        matrix_checks[name] = {
            "id_and_label_order_matches_frozen_csv": order[name] == [{"mail_id": row["mail_id"], "label": row["label"]} for row in records],
            "shape_matches_split_and_vocabulary": matrix.shape == (len(records), len(names)),
            "summary_matches_matrix": details == current,
            "finite_nonnegative_csr_float64": sparse.isspmatrix_csr(matrix) and str(matrix.dtype) == "float64"
                and bool(np.all(np.isfinite(matrix.data))) and bool(np.all(matrix.data >= 0)),
            "nonzero_rows_have_unit_l2_norm": bool(np.allclose(norms[nonempty], 1, atol=1e-10, rtol=0)),
            "zero_rows_have_zero_norm": bool(np.all(norms[~nonempty] == 0)),
            "saved_features_equal_fresh_transform": difference.nnz == 0 or bool(np.max(np.abs(difference.data)) <= 1e-12),
        }
    default = make_feature_pipeline().named_steps["tfidf"]
    expected_configuration = {key: default.get_params()[key] for key in ("min_df", "max_df", "max_features", "sublinear_tf", "smooth_idf", "use_idf", "norm", "lowercase", "token_pattern")}
    actual_configuration = {key: vectorizer.get_params()[key] for key in expected_configuration}
    coverage_summary = summary["training_word_coverage"]
    checks = {
        "verified_upstream_inputs_match_generation": bundle.provenance == summary["input_provenance"],
        "feature_source_and_dictionary_match": source_sha256(root, sources) == summary["feature_source_sha256"] and tokenizer_fingerprint() == summary["tokenizer"],
        "private_output_checksums_match": output_checksums,
        "training_only_fit_declared": summary["fit_scope"] == "train_only",
        "fitted_state_matches_generation": fitted_before == summary["fitted_state"],
        "vocabulary_selected_only_from_training_frequencies": bool(selection_valid),
        "idf_matches_independently_counted_training_documents": bool(np.allclose(vectorizer.idf_, expected_idf, rtol=0, atol=1e-12)),
        "saved_pipeline_matches_fixed_configuration": actual_configuration == expected_configuration == summary["configuration"]
            and vectorizer.ngram_range == (1, 1) and summary["ngram_range"] == [1, 1],
        "full_private_word_table_matches_training_counts": bool(word_table_valid),
        "public_terms_exactly_match_whitelist_and_training_df": coverage_summary["public_term_coverage"] == public_expected,
        "token_aggregates_match_training_recount": coverage_summary["unique_token_types_before_tfidf_filter"] == len(all_terms)
            and coverage_summary["numeric_token_types"] == sum(term.isdecimal() for term in all_terms)
            and coverage_summary["training_documents_with_numeric_token"] == numeric_documents
            and coverage_summary["training_tokenless_documents"] == tokenless
            and coverage_summary["training_label_counts"] == dict(label_counts),
        "all_matrix_and_alignment_checks_pass": all(all(values.values()) for values in matrix_checks.values()),
        "all_transform_calls_preserve_vocabulary_and_idf": state(vectorizer) == fitted_before,
        "runtime_versions_match_generation": summary["versions"] == {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    return {"checks": checks, "matrix_checks": matrix_checks,
            "feature_source_sha256": summary["feature_source_sha256"],
            "validator_source_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "scope": "Fresh process loads local joblib, recounts training-only document frequencies and reproduces saved transforms; no classifier fitting or prediction.",
            "passed": all(checks.values())}


def main() -> int:
    try:
        report = check(PROJECT_ROOT)
        target = PROJECT_ROOT / "artifacts/audit/feature_validation.json"
        target.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "message": "Feature checks failed; previous validation may remain."}))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
