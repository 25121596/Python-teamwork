"""Freeze exact deduplication and grouped holdouts; publish metadata only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spam_classifier.data_split import (
    POLICY_VERSION, SPLITS, allocation_summary, assign_splits, has_signal,
    prepare_records, signature, validate_parameters,
)
from spam_classifier.email_parser import PARSER_VERSION, clean_text, html_to_text
from spam_classifier.index_audit import balanced_sample, mail_id

PUBLIC_FIELDS = ("mail_id", "label_name", "label", "split")
LEDGER_FIELDS = (
    "mail_id", "label_name", "label", "usable", "is_development", "disposition", "reason",
    "representative_id", "leakage_group_id", "forced_train", "split",
    "raw_sha256", "text_sha256", "body_sha256", "ablation_sha256", "precontact_body_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("src/spam_classifier/data_split.py", "scripts/split_data.py"):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows: list[dict], fields: tuple) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def precontact_body_signature(row: dict) -> str:
    """Attribute body collisions to masks only when the cached view reconstructs."""
    if row["body_empty"] or row["body_selection"] not in {"plain", "html"}:
        return ""
    text = row["body_decoded"]
    if row["body_selection"] == "html":
        text = html_to_text(text)
    text = unicodedata.normalize("NFKC", text).lower().replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(char for char in text if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf", "Cs"})
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not has_signal(text) or clean_text(text) != row["body_clean"]:
        return ""
    return signature(text, "precontact-body")


def load_inputs(root: Path, config_path: Path) -> tuple[list[dict], set[str], dict, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    validate_parameters(config["random_state"], config["split_ratios"])
    if config["labels"] != {"ham": 0, "spam": 1} or config["index_file"] != "full/index":
        raise ValueError("Unexpected index or labels.")
    output = Path(config["artifacts_dir"])
    if not output.is_absolute():
        output = root / output
    if output.resolve() != root / "artifacts":
        raise ValueError("Keep the fixed ignored cache layout.")
    corpus = Path(os.environ.get("TREC06C_ROOT") or config["data_root"]).expanduser()
    if not corpus.is_absolute():
        corpus = config_path.parent / corpus
    corpus = corpus.resolve()
    if corpus.is_relative_to(root) or not (corpus / "data").resolve().is_relative_to(corpus):
        raise ValueError("Raw corpus must be external and contained.")
    audit_dir = root / "artifacts/audit"
    index = json.loads((audit_dir / "index_summary.json").read_text(encoding="utf-8"))
    summary = json.loads((audit_dir / "cleaning_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((audit_dir / "cleaning_validation.json").read_text(encoding="utf-8"))
    development = json.loads((audit_dir / "development_cleaning_summary.json").read_text(encoding="utf-8"))
    if not all(record["passed"] for record in (index, summary, validation, development)):
        raise ValueError("Resolve preceding audit and cleaning failures first.")
    parser_sha = hashlib.sha256((root / "src/spam_classifier/email_parser.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if (summary["parser_source_sha256"] != parser_sha or validation["parser_source_sha256"] != parser_sha
            or summary["parser_version"] != PARSER_VERSION or validation["cache_sha256"] != summary["cache_sha256"]):
        raise ValueError("Cleaning provenance no longer matches source.")
    if sha256_file(corpus / "full/index") != index["index_sha256"]:
        raise ValueError("Index changed; repeat earlier stages.")
    manifest_path = root / "artifacts/processed/mail_manifest.csv"
    if sha256_file(manifest_path) != summary["input_manifest_sha256"]:
        raise ValueError("Canonical manifest checksum changed.")
    manifest_rows = read_csv(manifest_path)
    manifest = {row["mail_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows) or len(manifest) != summary["input_records"]:
        raise ValueError("Canonical IDs or input counts changed.")
    dev_path = root / "artifacts/splits/development_sample.csv"
    dev_rows = read_csv(dev_path)
    sampling = index["development_sample"]
    if (sha256_file(dev_path) != development["input_manifest_sha256"]
            or dev_rows != balanced_sample(manifest_rows, sampling["requested_size"], sampling["seed"])):
        raise ValueError("Development sample changed.")
    dev_ids = {row["mail_id"] for row in dev_rows}
    cache_digest, raw_digest = hashlib.sha256(), hashlib.sha256()
    rows, seen = [], set()
    with (root / "artifacts/processed/cleaned_mail.jsonl").open("rb") as stream:
        for number, line in enumerate(stream, 1):
            cache_digest.update(line)
            parsed = json.loads(line)
            original = manifest.get(parsed["mail_id"])
            if (original is None or parsed["mail_id"] in seen or original["status"] != "eligible"
                    or parsed["relative_path"] != original["relative_path"]
                    or parsed["label_name"] != original["label_name"] or parsed["label"] != int(original["label"])
                    or parsed["mail_id"] != mail_id(parsed["relative_path"]) or parsed["parser_version"] != PARSER_VERSION):
                raise ValueError("Cache metadata differs from the canonical manifest.")
            seen.add(parsed["mail_id"])
            target = (corpus / parsed["relative_path"]).resolve()
            if not target.is_relative_to((corpus / "data").resolve()):
                raise ValueError("Raw mail path escaped the corpus.")
            raw = target.read_bytes()
            if len(raw) != int(original["size_bytes"]):
                raise ValueError("Raw mail size changed.")
            raw_sha = hashlib.sha256(raw).hexdigest()
            raw_digest.update(parsed["mail_id"].encode("ascii") + b"\0" + bytes.fromhex(raw_sha))
            row = {field: parsed[field] for field in (
                "mail_id", "label_name", "label", "text_clean", "body_clean", "body_empty", "usable", "text_unstripped_html"
            )}
            row.update(raw_sha256=raw_sha, precontact_body_sha256=precontact_body_signature(parsed))
            rows.append(row)
            if number % 10000 == 0:
                print(json.dumps({"checked_inputs": number, "total": len(manifest)}), flush=True)
    if (seen != set(manifest) or cache_digest.hexdigest() != summary["cache_sha256"]
            or raw_digest.hexdigest() != summary["raw_source_digest"]):
        raise ValueError("Input cache or raw content differs from the validated cleaning run.")
    provenance = {
        "cleaning_cache_sha256": cache_digest.hexdigest(), "raw_source_digest": raw_digest.hexdigest(),
        "canonical_manifest_sha256": sha256_file(manifest_path), "development_sample_sha256": sha256_file(dev_path),
        "parser_source_sha256": parser_sha,
    }
    return rows, dev_ids, config, provenance


def build_outputs(root: Path, config_path: Path) -> tuple[dict[str, bytes], dict]:
    source_sha = policy_source_sha256(root)
    rows, development_ids, config, provenance = load_inputs(root, config_path)
    prepared = prepare_records(rows, development_ids)
    assigned = assign_splits(prepared.representatives, seed=config["random_state"], ratios=config["split_ratios"])
    allocation = allocation_summary(assigned, config["split_ratios"])
    if any(not part["label_counts"][label] for part in allocation.values() for label in ("ham", "spam")):
        raise ValueError("Group constraints could not produce nonempty two-class holdouts.")
    mapping = {row["mail_id"]: row["split"] for row in assigned}
    ledger = [dict(row, split=mapping.get(row["representative_id"], "")) for row in prepared.ledger]
    outputs = {
        f"artifacts/splits/{name}.csv": csv_bytes([row for row in assigned if row["split"] == name], PUBLIC_FIELDS)
        for name in SPLITS
    }
    outputs["artifacts/processed/dedup_ledger.csv"] = csv_bytes(ledger, LEDGER_FIELDS)
    if policy_source_sha256(root) != source_sha:
        raise ValueError("Split source changed while running.")
    summary = {
        "schema_version": 1, "policy_version": POLICY_VERSION,
        "split_policy_source_sha256": source_sha, "input_provenance": provenance,
        "random_state": config["random_state"], "target_ratios": config["split_ratios"],
        "deduplication": prepared.summary, "allocation": allocation,
        "output_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in outputs.items()},
        "policy": {
            "remove": "One representative per exact final text; prefer development membership, then lexicographic mail_id.",
            "conflicts": "Isolate every row sharing a cross-label raw-byte or final-text key; do not treat body-only mixed labels as a conflict.",
            "group": "Connected components of equal raw bytes, final text, usable body or unstripped-HTML ablation text, with field-specific keys.",
            "split": "Force development-exposed components into train, then descending group size and seeded SHA ties; greedily minimize normalized per-class count error.",
            "public": "ID, label and split only; content signatures, linkage groups and duplicate mappings remain private.",
        },
        "limitations": ["Exact equality does not detect all near-duplicate advertising templates.",
                        "Holdouts describe unique normalized texts in a historical corpus, not natural inbox traffic.",
                        "Fixed group and development constraints may cause target-ratio deviations."],
        "passed": True,
    }
    outputs["artifacts/audit/split_summary.json"] = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return outputs, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    try:
        chosen = args.config or ("config.local.json" if (PROJECT_ROOT / "config.local.json").is_file() else "config.example.json")
        config_path = Path(chosen)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        outputs, summary = build_outputs(PROJECT_ROOT, config_path.resolve())
        from check_splits import check
        validation = check(PROJECT_ROOT, candidates=outputs)
        if not validation["passed"]:
            raise ValueError("Candidate split failed independent validation; no outputs published.")
        # Reproduction may restore missing private metadata, but never silently
        # replace a different frozen split or its provenance.
        if any((PROJECT_ROOT / name).exists() and (PROJECT_ROOT / name).read_bytes() != value
               for name, value in outputs.items()):
            raise ValueError("Existing frozen outputs differ; preserve the old version and deliberately version a new split.")
        outputs["artifacts/audit/split_validation.json"] = (json.dumps(validation, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        for name, value in outputs.items():
            target = PROJECT_ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(value)
            temporary.replace(target)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "fatal_error_type": type(exc).__name__,
                          "message": "Input provenance, configuration or frozen outputs failed checks; previous outputs may remain."}))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
