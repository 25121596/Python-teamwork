"""Validate private cleaning caches and publish metadata-only quality evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from spam_classifier.email_parser import EMAIL, PHONE, URL, PARSER_VERSION, clean_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("development", "full"), default="development")
    args = parser.parse_args()
    prefix = "development_" if args.scope == "development" else ""
    artifacts = PROJECT_ROOT / "artifacts"
    cache = artifacts / "processed" / ("cleaned_development.jsonl" if prefix else "cleaned_mail.jsonl")
    summary = json.loads((artifacts / "audit" / (prefix + "cleaning_summary.json")).read_text(encoding="utf-8"))
    manifest = artifacts / ("splits/development_sample.csv" if prefix else "processed/mail_manifest.csv")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        expected = {row["mail_id"]: row for row in csv.DictReader(stream)}
    statuses, labels, failures = Counter(), Counter(), Counter()
    seen = set()
    digest = hashlib.sha256()
    retained = replacements = usable = attempted = attempted_replacements = 0
    # Keep a bounded deterministic priority sample, without storing extra text.
    qa = {"ham": [], "spam": []}
    rng = random.Random(42)
    with cache.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            identity = row["mail_id"]
            original = expected.get(identity)
            if identity in seen:
                failures["duplicate_ids"] += 1
            seen.add(identity)
            if (original is None or row["relative_path"] != original["relative_path"]
                    or row["label"] != int(original["label"]) or row["label_name"] != original["label_name"]):
                failures["manifest_mismatch"] += 1
            text = row["text_clean"]
            if row["parser_version"] != summary["parser_version"] or row["parser_version"] != PARSER_VERSION:
                failures["parser_version_mismatch"] += 1
            if clean_text(text) != text:
                failures["normalization_not_idempotent"] += 1
            if any(pattern.search(text) for pattern in (URL, EMAIL, PHONE)):
                failures["unmasked_contact_patterns"] += 1
            if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} and char not in "\n\t" for char in text):
                failures["unexpected_control_characters"] += 1
            if bool(any(char.isalnum() for char in text)) != row["usable"]:
                failures["usable_flag_mismatch"] += 1
            if row["retained_characters"] != len(text) or row["retained_replacement_characters"] != text.count("\ufffd"):
                failures["retained_character_count_mismatch"] += 1
            if abs(row["retained_replacement_ratio"] - text.count("\ufffd") / max(1, len(text))) > 1e-12:
                failures["retained_replacement_ratio_mismatch"] += 1
            if (row["decoded_characters"] < 0 or not 0 <= row["replacement_characters"] <= row["decoded_characters"]
                    or abs(row["replacement_ratio"] - row["replacement_characters"] / max(1, row["decoded_characters"])) > 1e-12):
                failures["attempt_character_ratio_mismatch"] += 1
            statuses[row["parse_status"]] += 1
            labels[row["label_name"]] += 1
            retained += len(text)
            replacements += text.count("\ufffd")
            usable += int(row["usable"])
            attempted += row["decoded_characters"]
            attempted_replacements += row["replacement_characters"]
            priority = (row["parse_status"] not in {"ok", "recovered"},
                        bool(row["encoding_warnings"]), bool(row["recovery_actions"]),
                        bool(row["mime_defects"]), row["replacement_ratio"], rng.random())
            selected = qa[row["label_name"]]
            selected.append((priority, {
                "mail_id": identity, "parse_status": row["parse_status"],
                "body_length": len(row["body_clean"]), "mime_defects": row["mime_defects"],
                "recovery_actions": row["recovery_actions"], "replacement_ratio": row["replacement_ratio"],
            }))
            selected.sort(key=lambda item: item[0], reverse=True)
            del selected[20:]
    parser_source_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "src/spam_classifier/email_parser.py").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    checks = {
        "parser_source_matches_generation": summary.get("parser_source_sha256") == parser_source_sha256,
        "all_input_ids_accounted": seen == set(expected),
        "input_output_counts_match": len(expected) == summary["input_records"] and sum(statuses.values()) == summary["processed_records"],
        "cache_checksum_matches_summary": digest.hexdigest() == summary["cache_sha256"],
        "manifest_checksum_matches_summary": hashlib.sha256(manifest.read_bytes()).hexdigest() == summary["input_manifest_sha256"],
        "status_partition_matches_summary": dict(statuses) == summary["status_counts"],
        "class_counts_match_summary": all(labels[label] == summary["by_label"][label]["total"] for label in labels),
        "usable_isolated_ledger": usable == summary["usable_records"] and len(seen) - usable == summary["isolated_records"],
        "retained_character_totals_match": retained == summary["retained_characters"] and replacements == summary["retained_replacement_characters"],
        "attempt_character_totals_match": attempted == summary["attempted_decoded_characters"] and attempted_replacements == summary["replacement_characters"],
        "attempt_aggregate_ratio_matches": abs(summary["replacement_ratio"] - attempted_replacements / max(1, attempted)) <= 1e-12,
        "no_parser_errors": statuses["error"] == 0,
        "all_row_checks_pass": not failures,
    }
    report = {
        "scope": args.scope, "cache_sha256": digest.hexdigest(), "checks": checks,
        "parser_source_sha256": parser_source_sha256,
        "row_violation_counts": dict(failures),
        "rule_review": {
            "method": "Top 20 per class by status/warnings/recovery/defects/replacement ratio, fixed random tie breaks.",
            "limitation": "Rule-based metadata review; not a claim of semantic correctness or comprehensive anonymization.",
            "selected_counts": {label: len(selected) for label, selected in qa.items()},
            "selected_status_counts": {
                label: dict(Counter(item[1]["parse_status"] for item in selected))
                for label, selected in qa.items()
            },
        },
        "passed": all(checks.values()),
    }
    # Real per-mail QA metadata stays beside the ignored text caches.
    private_review = artifacts / "processed" / (prefix + "cleaning_rule_review.json")
    private_review.write_text(json.dumps({
        "scope": args.scope, "cache_sha256": digest.hexdigest(),
        "selected_metadata": {label: [item[1] for item in selected] for label, selected in qa.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = artifacts / "audit" / (prefix + "cleaning_validation.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": args.scope, "checks": checks, "row_violation_counts": dict(failures), "passed": report["passed"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
