"""Verified inputs and immutable run contracts for the final two experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dataset import file_sha256, read_csv, source_sha256
from .model_inputs import MODEL_SOURCES, load_model_inputs, require_passed, runtime_versions

EXPERIMENT_SOURCES = (
    "src/spam_classifier/experiment_io.py",
    "src/spam_classifier/cleaning_ablation.py",
    "src/spam_classifier/model_selection.py",
    "src/spam_classifier/text_features.py",
    "config/stopwords_zh.txt", "config/public_terms_zh.txt",
    "scripts/compare_cleaning.py", "scripts/check_ablation.py",
)
SCOPE = {"fit": "train_only", "selection": "validation_only", "test_metrics_computed": False}
REPRESENTATIONS = ("text_clean", "text_unstripped_html")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def verified_file(root: Path, relative: str, expected: str, *, prefix: str) -> Path:
    path = (root / relative).resolve()
    if not relative.startswith(prefix) or not path.is_relative_to((root / prefix).resolve()):
        raise ValueError("Artifact path escapes its declared scope.")
    if file_sha256(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {relative}")
    return path


def verified_stage06(root: Path):
    """No test CSV/matrix or mail text is opened by this upstream check."""
    inputs = load_model_inputs(root)
    summary = read_json(root / "artifacts/metrics/model_selection.json")
    audit = read_json(root / "artifacts/audit/model_validation.json")
    require_passed(audit)
    validator = root / "scripts/check_models.py"
    if (summary.get("passed") is not True
            or audit["model_selection_summary_sha256"] != file_sha256(root / "artifacts/metrics/model_selection.json")
            or audit["validator_source_sha256"] != hashlib.sha256(validator.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            or summary["model_source_sha256"] != source_sha256(root, MODEL_SOURCES)
            or summary["input_provenance"] != inputs.provenance
            or summary["versions"] != runtime_versions()):
        raise ValueError("Stage-six evidence is stale; rebuild and verify it first.")
    if (summary["family_winners"] != {"multinomial_nb": "nb_alpha_0.1", "logistic_regression": "lr_C_2"}
            or summary["selected_candidate_id"] != "lr_C_2"):
        raise ValueError("Stage-six winner changed; review the predeclared stage-seven contract.")
    for relative, digest in summary["private_output_sha256"].items():
        verified_file(root, relative, digest, prefix="artifacts/")
    for relative, digest in summary["public_output_sha256"].items():
        verified_file(root, relative, digest, prefix="artifacts/")
    return inputs, summary


def read_records(root: Path, rows_by_split: dict[str, list[dict]], expected_cache_sha: str) -> dict[str, list[dict]]:
    """Hash the complete cache but retain only explicitly requested frozen IDs.

    Stage seven requests train/validation only. Full-cache bytes are read to
    verify their fingerprint, never used as a source of heldout statistics.
    """
    wanted = {row["mail_id"]: row for rows in rows_by_split.values() for row in rows}
    if len(wanted) != sum(map(len, rows_by_split.values())):
        raise ValueError("Repeated IDs in requested splits.")
    found, seen, digest = {}, set(), hashlib.sha256()
    with (root / "artifacts/processed/cleaned_mail.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            identity = record["mail_id"]
            if identity in seen:
                raise ValueError("Repeated ID in cleaning cache.")
            seen.add(identity)
            if identity not in wanted:
                continue
            metadata = wanted[identity]
            if (record["label"] != int(metadata["label"]) or record["label_name"] != metadata["label_name"]
                    or record["usable"] is not True or any(not isinstance(record[key], str) for key in REPRESENTATIONS)):
                raise ValueError("Requested cache rows differ from frozen metadata.")
            found[identity] = record
    if digest.hexdigest() != expected_cache_sha or set(found) != set(wanted):
        raise ValueError("Cleaning cache checksum or requested IDs differ.")
    return {name: [found[row["mail_id"]] for row in rows] for name, rows in rows_by_split.items()}


def read_test_rows(root: Path, lock: dict) -> list[dict]:
    relative = "artifacts/splits/test.csv"
    path = verified_file(root, relative, lock["split_csv_sha256"]["test"], prefix="artifacts/splits/")
    fields, rows = read_csv(path)
    if fields != ["mail_id", "label_name", "label", "split"] or not rows:
        raise ValueError("Unexpected test metadata schema.")
    if len({row["mail_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated test IDs.")
    for row in rows:
        if row["split"] != "test" or row["label_name"] not in {"ham", "spam"} or row["label"] != str({"ham": 0, "spam": 1}[row["label_name"]]):
            raise ValueError("Invalid test labels or split.")
        row["label"] = int(row["label"])
    if len(rows) != lock["test_rows"]:
        raise ValueError("Test population changed.")
    return rows


def verify_lock(root: Path) -> dict:
    """Check the immutable pre-test lock before reading final-test examples."""
    lock = read_json(root / "artifacts/metrics/final_lock.json")
    if (lock.get("schema_version") != 1 or lock.get("scope") != SCOPE
            or lock.get("representation") not in REPRESENTATIONS
            or lock.get("model_names") != ["majority", "nb", "lr"]
            or lock.get("selected_model") != "lr"
            or lock.get("parameters") != {"nb": {"alpha": 0.1}, "lr": {"C": 2.0}}
            or lock.get("decision_rule") != "estimator.predict; no threshold tuning; no train+validation refit"
            or lock["versions"] != runtime_versions()
            or lock["source_sha256"] != source_sha256(root, EXPERIMENT_SOURCES)):
        raise ValueError("Final lock contract, runtime or source changed.")
    for relative, digest in lock["upstream_sha256"].items():
        verified_file(root, relative, digest, prefix="artifacts/")
    expected = {
        "artifacts/models/stage07/final_models.joblib",
        "artifacts/models/stage07/ablation_models.joblib",
        "artifacts/processed/stage07/validation_predictions.npz",
        "artifacts/metrics/cleaning_ablation.json",
        "artifacts/metrics/stage07_validation.csv",
    }
    if set(lock["output_sha256"]) != expected:
        raise ValueError("Incomplete final lock artifact manifest.")
    for relative, digest in lock["output_sha256"].items():
        verified_file(root, relative, digest, prefix="artifacts/")
    return lock


def require_ablation_validation(root: Path, lock_sha: str) -> None:
    audit = read_json(root / "artifacts/audit/ablation_validation.json")
    require_passed(audit)
    expected = hashlib.sha256((root / "scripts/check_ablation.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if audit["lock_sha256"] != lock_sha or audit["validator_source_sha256"] != expected:
        raise ValueError("Ablation validation is stale.")
