"""Load only checksum-verified cached mail and frozen split metadata."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


@dataclass
class DatasetBundle:
    all_records: list[dict]
    splits: dict[str, list[dict]]
    provenance: dict


def load_dataset(root: Path) -> DatasetBundle:
    root = Path(root).resolve()
    art = root / "artifacts"
    cleaning = json.loads((art / "audit/cleaning_summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((art / "audit/split_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((art / "audit/split_validation.json").read_text(encoding="utf-8"))
    if not all(item["passed"] for item in (cleaning, frozen, validation)) or not all(validation["checks"].values()):
        raise ValueError("Resolve cleaning and frozen-split verification first.")
    if frozen["random_state"] != 42 or frozen["target_ratios"] != {"train": .7, "validation": .15, "test": .15}:
        raise ValueError("Unexpected frozen split contract.")
    policy_sha = source_sha256(root, ("src/spam_classifier/data_split.py", "scripts/split_data.py"))
    parser_sha = hashlib.sha256((root / "src/spam_classifier/email_parser.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    checker_sha = hashlib.sha256((root / "scripts/check_splits.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    source = frozen["input_provenance"]
    if (policy_sha != frozen["split_policy_source_sha256"] or policy_sha != validation["split_policy_source_sha256"]
            or parser_sha != cleaning["parser_source_sha256"] or parser_sha != source["parser_source_sha256"]
            or checker_sha != validation["validator_source_sha256"]):
        raise ValueError("Upstream source fingerprint no longer matches validated inputs.")
    expected_paths = {f"artifacts/splits/{name}.csv" for name in SPLITS} | {"artifacts/processed/dedup_ledger.csv"}
    if set(frozen["output_sha256"]) != expected_paths or any(file_sha256(root / name) != digest for name, digest in frozen["output_sha256"].items()):
        raise ValueError("Frozen metadata checksums changed.")
    canonical_path = art / "processed/mail_manifest.csv"
    dev_path = art / "splits/development_sample.csv"
    if (file_sha256(canonical_path) != source["canonical_manifest_sha256"]
            or file_sha256(dev_path) != source["development_sample_sha256"]):
        raise ValueError("Canonical or development metadata changed.")
    _, canonical_rows = read_csv(canonical_path)
    canonical = {row["mail_id"]: row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise ValueError("Duplicate canonical IDs.")
    _, ledger_rows = read_csv(art / "processed/dedup_ledger.csv")
    ledger = {row["mail_id"]: row for row in ledger_rows}
    retained = {row["mail_id"] for row in ledger_rows if row["disposition"] == "retained"}
    if len(ledger) != len(ledger_rows) or set(ledger) != set(canonical):
        raise ValueError("Ledger does not account for every canonical ID.")
    all_records, records = [], {}
    digest = hashlib.sha256()
    with (art / "processed/cleaned_mail.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            identity = row["mail_id"]
            original = canonical.get(identity)
            if (identity in records or original is None or row["relative_path"] != original["relative_path"]
                    or row["label_name"] != original["label_name"] or str(row["label"]) != original["label"]
                    or row["label_name"] not in {"ham", "spam"}
                    or row["label"] != {"ham": 0, "spam": 1}[row["label_name"]]
                    or row["parser_version"] != cleaning["parser_version"]):
                raise ValueError("Cache ID, label or parser version mismatch.")
            records[identity] = row
            all_records.append(row)
    if (set(records) != set(canonical) or len(records) != cleaning["processed_records"]
            or digest.hexdigest() != cleaning["cache_sha256"] or digest.hexdigest() != source["cleaning_cache_sha256"]):
        raise ValueError("Cleaning cache changed.")
    splits, assigned = {}, set()
    for name in SPLITS:
        fields, rows = read_csv(art / f"splits/{name}.csv")
        if fields != ["mail_id", "label_name", "label", "split"]:
            raise ValueError("Unexpected public metadata schema.")
        splits[name] = []
        for item in rows:
            identity = item["mail_id"]
            if identity in assigned or identity not in retained or item["split"] != name:
                raise ValueError("Repeated or unknown representative ID.")
            record, entry = records[identity], ledger[identity]
            if (not record["usable"] or item["label_name"] != record["label_name"]
                    or item["label"] != str(record["label"]) or entry["split"] != name):
                raise ValueError("Frozen split differs from cached text or ledger.")
            assigned.add(identity)
            splits[name].append(dict(record, split=name))
    if assigned != retained or len(assigned) != validation["public_representatives"]:
        raise ValueError("Split representatives do not match the validated partition.")
    return DatasetBundle(all_records, splits, {
        "cleaning_cache_sha256": digest.hexdigest(), "split_policy_source_sha256": policy_sha,
        "split_summary_sha256": file_sha256(art / "audit/split_summary.json"),
        "split_validation_sha256": file_sha256(art / "audit/split_validation.json"),
        "split_csv_sha256": {name: frozen["output_sha256"][f"artifacts/splits/{name}.csv"] for name in SPLITS},
    })
