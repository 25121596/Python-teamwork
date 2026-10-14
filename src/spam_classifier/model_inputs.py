"""Load verified train/validation features without opening final-test inputs."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from scipy import sparse

from .dataset import file_sha256, read_csv, source_sha256
from .text_features import make_feature_pipeline, tokenizer_fingerprint

FEATURE_SOURCES = ("src/spam_classifier/dataset.py", "src/spam_classifier/text_features.py", "scripts/build_features.py")
MODEL_SOURCES = ("src/spam_classifier/model_selection.py", "src/spam_classifier/model_inputs.py", "scripts/train_models.py")
CONSUMED_SPLITS = ("train", "validation")


def runtime_versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__}


def fitted_feature_state(vectorizer) -> dict:
    vocabulary = sorted((term, int(index)) for term, index in vectorizer.vocabulary_.items())
    return {
        "vocabulary_sha256": hashlib.sha256(json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
    }


@dataclass
class ModelInputs:
    train_matrix: sparse.csr_matrix
    validation_matrix: sparse.csr_matrix
    train_rows: list[dict]
    validation_rows: list[dict]
    feature_pipeline: object
    provenance: dict


def require_passed(report: dict, checks_key: str = "checks") -> None:
    checks = report[checks_key]
    if report["passed"] is not True or not checks or not all(value is True for value in checks.values()):
        raise ValueError("Resolve upstream validation failures first.")


def load_model_inputs(root: Path) -> ModelInputs:
    root = Path(root).resolve()
    art = root / "artifacts"
    summary_path = art / "audit/feature_summary.json"
    validation_path = art / "audit/feature_validation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    frozen = json.loads((art / "audit/split_summary.json").read_text(encoding="utf-8"))
    split_validation = json.loads((art / "audit/split_validation.json").read_text(encoding="utf-8"))
    cleaning = json.loads((art / "audit/cleaning_summary.json").read_text(encoding="utf-8"))
    require_passed(validation)
    require_passed(split_validation)
    if summary["passed"] is not True or frozen["passed"] is not True or cleaning["passed"] is not True:
        raise ValueError("Upstream construction is incomplete.")
    if frozen["random_state"] != 42 or frozen["target_ratios"] != {"train": .7, "validation": .15, "test": .15}:
        raise ValueError("Frozen split contract changed.")
    policy = source_sha256(root, ("src/spam_classifier/data_split.py", "scripts/split_data.py"))
    if (policy != frozen["split_policy_source_sha256"] or policy != split_validation["split_policy_source_sha256"]
            or policy != summary["input_provenance"]["split_policy_source_sha256"]
            or file_sha256(art / "audit/split_summary.json") != summary["input_provenance"]["split_summary_sha256"]
            or file_sha256(art / "audit/split_validation.json") != summary["input_provenance"]["split_validation_sha256"]):
        raise ValueError("Frozen split provenance changed.")
    parser_sha = hashlib.sha256((root / "src/spam_classifier/email_parser.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    split_checker_sha = hashlib.sha256((root / "scripts/check_splits.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if (parser_sha != cleaning["parser_source_sha256"] or parser_sha != frozen["input_provenance"]["parser_source_sha256"]
            or split_checker_sha != split_validation["validator_source_sha256"]
            or cleaning["cache_sha256"] != summary["input_provenance"]["cleaning_cache_sha256"]
            or cleaning["cache_sha256"] != frozen["input_provenance"]["cleaning_cache_sha256"]):
        raise ValueError("Parser or cleaning provenance changed.")
    feature_sha = source_sha256(root, FEATURE_SOURCES)
    checker_sha = hashlib.sha256((root / "scripts/check_features.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    feature_versions = {key: value for key, value in runtime_versions().items() if key != "threadpoolctl"}
    if (feature_sha != summary["feature_source_sha256"] or feature_sha != validation["feature_source_sha256"]
            or checker_sha != validation["validator_source_sha256"] or tokenizer_fingerprint() != summary["tokenizer"]
            or feature_versions != summary["versions"] or summary["fit_scope"] != "train_only"):
        raise ValueError("Feature rules, validation or runtime changed; rerun feature verification.")
    expected_paths = {f"artifacts/processed/features/{name}.npz" for name in ("train", "validation", "test")}
    expected_paths |= {"artifacts/processed/features/row_order.json", "artifacts/processed/features/training_word_coverage.csv", "artifacts/models/tfidf_pipeline.joblib"}
    if set(summary["private_output_sha256"]) != expected_paths:
        raise ValueError("Unexpected feature artifact paths.")
    consumed = {f"artifacts/processed/features/{name}.npz" for name in CONSUMED_SPLITS}
    consumed |= {"artifacts/processed/features/row_order.json", "artifacts/models/tfidf_pipeline.joblib"}
    input_sha = {name: file_sha256(root / name) for name in sorted(consumed)}
    if any(digest != summary["private_output_sha256"][name] for name, digest in input_sha.items()):
        raise ValueError("Consumed feature artifacts changed.")
    ordering = json.loads((art / "processed/features/row_order.json").read_text(encoding="utf-8"))
    rows, matrices, assigned = {}, {}, set()
    for name in CONSUMED_SPLITS:
        matrix_checks = validation["matrix_checks"][name]
        if not matrix_checks or not all(value is True for value in matrix_checks.values()):
            raise ValueError("Upstream matrix verification failed.")
        csv_path = art / f"splits/{name}.csv"
        digest = file_sha256(csv_path)
        if (digest != frozen["output_sha256"][f"artifacts/splits/{name}.csv"]
                or digest != summary["input_provenance"]["split_csv_sha256"][name]):
            raise ValueError("Train/validation CSV changed.")
        fields, current_rows = read_csv(csv_path)
        if fields != ["mail_id", "label_name", "label", "split"]:
            raise ValueError("Unexpected split schema.")
        for row in current_rows:
            if (row["mail_id"] in assigned or row["split"] != name or row["label_name"] not in {"ham", "spam"}
                    or row["label"] != str({"ham": 0, "spam": 1}[row["label_name"]])):
                raise ValueError("Split IDs or labels are invalid.")
            assigned.add(row["mail_id"])
            row["label"] = int(row["label"])
        if ordering[name] != [{"mail_id": row["mail_id"], "label": row["label"]} for row in current_rows]:
            raise ValueError("Feature rows do not match frozen ID/label order.")
        if (len(current_rows) != frozen["allocation"][name]["records"]
                or dict(Counter(row["label_name"] for row in current_rows)) != frozen["allocation"][name]["label_counts"]):
            raise ValueError("Frozen row and class totals do not match.")
        matrix = sparse.load_npz(art / f"processed/features/{name}.npz")
        details = summary["matrices"][name]
        if (not sparse.isspmatrix_csr(matrix) or matrix.dtype != np.dtype("float64")
                or matrix.shape != (len(current_rows), details["columns"]) or matrix.nnz != details["nnz"]
                or not np.all(np.isfinite(matrix.data)) or not np.all(matrix.data >= 0)):
            raise ValueError("Invalid sparse feature matrix.")
        rows[name], matrices[name] = current_rows, matrix
    # Only deserialize this project's checksum-verified local pipeline.
    pipeline = joblib.load(art / "models/tfidf_pipeline.joblib")
    vectorizer = pipeline.named_steps["tfidf"]
    default = make_feature_pipeline().named_steps["tfidf"]
    config_keys = tuple(summary["configuration"])
    if (fitted_feature_state(vectorizer) != summary["fitted_state"] or summary["dtype"] != "float64"
            or {key: vectorizer.get_params()[key] for key in config_keys} != summary["configuration"]
            or {key: default.get_params()[key] for key in config_keys} != summary["configuration"]
            or vectorizer.ngram_range != (1, 1) or vectorizer.dtype != np.float64
            or any(matrix.shape[1] != len(vectorizer.vocabulary_) for matrix in matrices.values())):
        raise ValueError("Fitted training feature pipeline changed.")
    provenance = {
        "feature_summary_sha256": file_sha256(summary_path), "feature_validation_sha256": file_sha256(validation_path),
        "feature_source_sha256": feature_sha, "fitted_feature_state": summary["fitted_state"],
        "split_csv_sha256": {name: file_sha256(art / f"splits/{name}.csv") for name in CONSUMED_SPLITS},
        "private_input_sha256": input_sha,
        "consumed_splits": list(CONSUMED_SPLITS),
    }
    return ModelInputs(matrices["train"], matrices["validation"], rows["train"], rows["validation"], pipeline, provenance)
