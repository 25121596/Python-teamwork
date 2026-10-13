"""Parse development or full mail corpus into a Git-ignored local JSONL cache."""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spam_classifier.email_parser import PARSER_VERSION, parse_email
from spam_classifier.index_audit import balanced_sample, mail_id


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def charset_bucket(value: str) -> str:
    try:
        return codecs.lookup(value).name
    except (LookupError, ValueError):
        return "invalid_or_unknown"


def distribution(values: list[int]) -> dict:
    values = sorted(values)
    return {
        "min": values[0] if values else None,
        "p50": values[(len(values) - 1) // 2] if values else None,
        "p95": values[max(0, (95 * len(values) + 99) // 100 - 1)] if values else None,
        "max": values[-1] if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("development", "full"), default="development")
    parser.add_argument("--config")
    args = parser.parse_args()
    try:
        chosen = args.config or ("config.local.json" if (PROJECT_ROOT / "config.local.json").is_file() else "config.example.json")
        config_path = project_path(chosen).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if config["index_file"] != "full/index" or config["labels"] != {"ham": 0, "spam": 1}:
            raise ValueError("Unexpected canonical index or labels.")
        data_root = Path(os.environ.get("TREC06C_ROOT") or config["data_root"]).expanduser()
        if not data_root.is_absolute():
            data_root = config_path.parent / data_root
        data_root = data_root.resolve()
        if data_root.is_relative_to(PROJECT_ROOT):
            raise ValueError("Keep raw mail outside the repository.")
        output_root = project_path(config["artifacts_dir"]).resolve()
        if output_root != PROJECT_ROOT / "artifacts":
            raise ValueError("Use the Git-ignored project cache layout.")
        audit = json.loads((output_root / "audit/index_summary.json").read_text(encoding="utf-8"))
        if not audit["passed"]:
            raise ValueError("Run and resolve the index audit before cleaning.")
        if hashlib.sha256((data_root / "full/index").read_bytes()).hexdigest() != audit["index_sha256"]:
            raise ValueError("The source index changed; repeat the index audit.")
        full_manifest = output_root / "processed/mail_manifest.csv"
        with full_manifest.open(encoding="utf-8-sig", newline="") as stream:
            canonical = {row["mail_id"]: row for row in csv.DictReader(stream)}
        if len(canonical) != audit["files"]["eligible_unique_files"]:
            raise ValueError("Full manifest does not match the latest audit.")
        source = full_manifest if args.scope == "full" else output_root / "splits/development_sample.csv"
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or len({row["mail_id"] for row in rows}) != len(rows):
            raise ValueError("Expected a nonempty manifest with unique IDs.")
        for row in rows:
            if (row != canonical.get(row["mail_id"]) or row["status"] != "eligible"
                    or row["label"] != str(config["labels"].get(row["label_name"]))
                    or row["mail_id"] != mail_id(row["relative_path"])):
                raise ValueError("Manifest record failed canonical metadata checks.")
        if args.scope == "development":
            sampling = audit["development_sample"]
            expected_rows = balanced_sample(list(canonical.values()), sampling["requested_size"], sampling["seed"])
            if rows != expected_rows:
                raise ValueError("Development sample changed; regenerate it with the audit script.")

        prefix = "development_" if args.scope == "development" else ""
        cache = output_root / "processed" / ("cleaned_development.jsonl" if prefix else "cleaned_mail.jsonl")
        issues = output_root / "processed" / (prefix + "cleaning_issues.jsonl")
        summary_path = output_root / "audit" / (prefix + "cleaning_summary.json")
        parser_source_sha256 = hashlib.sha256(
            (PROJECT_ROOT / "src/spam_classifier/email_parser.py").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        cache_temp = cache.with_name(cache.name + ".tmp")
        issues_temp = issues.with_name(issues.name + ".tmp")
        cache.parent.mkdir(parents=True, exist_ok=True)
        status = Counter()
        by_label = {label: Counter() for label in ("ham", "spam")}
        mime_types, defects, actions, declared, used, warnings = (Counter() for _ in range(6))
        lengths = []
        replacements = characters = usable_count = retained_characters = retained_replacements = 0
        digest = hashlib.sha256()
        raw_digest = hashlib.sha256()
        known_mime = {"text/plain", "text/html", "multipart/alternative", "multipart/mixed", "multipart/related", "message/rfc822", "application/octet-stream"}
        with cache_temp.open("wb") as cache_stream, issues_temp.open("w", encoding="utf-8") as issue_stream:
            for number, row in enumerate(rows, 1):
                target = (data_root / row["relative_path"]).resolve()
                try:
                    if not target.is_relative_to((data_root / "data").resolve()):
                        raise ValueError("Mail target escaped the data directory.")
                    raw = target.read_bytes()
                    if len(raw) != int(row["size_bytes"]):
                        raise ValueError("Source size changed; repeat index audit.")
                    raw_digest.update(row["mail_id"].encode("ascii") + b"\0" + hashlib.sha256(raw).digest())
                    parsed = parse_email(raw)
                except Exception as exc:
                    parsed = parse_email(b"")
                    parsed.update(parse_status="error", usable=False, error_type=type(exc).__name__)
                parsed.update(mail_id=row["mail_id"], relative_path=row["relative_path"],
                              label_name=row["label_name"], label=int(row["label"]))
                encoded = (json.dumps(parsed, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                cache_stream.write(encoded)
                digest.update(encoded)
                label_stats = by_label[row["label_name"]]
                label_stats["total"] += 1
                status[parsed["parse_status"]] += 1
                label_stats["status_" + parsed["parse_status"]] += 1
                usable_count += int(parsed["usable"])
                label_stats["usable"] += int(parsed["usable"])
                for key in ("decode_fallback", "has_html", "has_html_part_anywhere", "subject_missing", "subject_empty", "body_empty"):
                    label_stats[key] += int(parsed[key])
                label_stats["mime_defect_mail"] += bool(parsed["mime_defects"])
                label_stats["encoding_warning_mail"] += bool(parsed["encoding_warnings"])
                label_stats["replacement_mail"] += parsed["replacement_characters"] > 0
                label_stats["attachment_skipped_mail"] += parsed["skipped_attachments"] > 0
                label_stats["forwarded_skipped_mail"] += parsed["skipped_forwarded"] > 0
                replacements += parsed["replacement_characters"]
                characters += parsed["decoded_characters"]
                retained_characters += parsed["retained_characters"]
                retained_replacements += parsed["retained_replacement_characters"]
                lengths.append(len(parsed["text_clean"]))
                mime_types[parsed["top_mime_type"] if parsed["top_mime_type"] in known_mime else "other"] += 1
                defects.update(parsed["mime_defects"])
                actions.update(parsed["recovery_actions"])
                declared.update(charset_bucket(value) for value in set(parsed["declared_charsets"]))
                used.update(set(parsed["used_charsets"]))
                warnings.update(parsed["encoding_warnings"])
                if parsed["parse_status"] != "ok":
                    issue_stream.write(json.dumps({
                        "mail_id": row["mail_id"], "label_name": row["label_name"],
                        "parse_status": parsed["parse_status"], "mime_defects": parsed["mime_defects"],
                        "recovery_actions": parsed["recovery_actions"], "error_type": parsed["error_type"],
                    }, ensure_ascii=False) + "\n")
                if args.scope == "full" and number % 5000 == 0:
                    print(json.dumps({"processed": number, "total": len(rows)}), flush=True)
        cache_temp.replace(cache)
        issues_temp.replace(issues)
        summary = {
            "schema_version": 1, "parser_version": PARSER_VERSION, "scope": args.scope,
            "parser_source_sha256": parser_source_sha256,
            "input_records": len(rows), "processed_records": sum(status.values()),
            "input_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "cache_sha256": digest.hexdigest(), "status_counts": dict(sorted(status.items())),
            "raw_source_digest": raw_digest.hexdigest(),
            "usable_records": usable_count, "isolated_records": len(rows) - usable_count,
            "by_label": {label: dict(sorted(values.items())) for label, values in by_label.items()},
            "top_mime_types": dict(sorted(mime_types.items())),
            "mime_defect_mail_counts": dict(sorted(defects.items())),
            "recovery_action_mail_counts": dict(sorted(actions.items())),
            "declared_charset_mail_counts": dict(sorted(declared.items())),
            "used_charset_mail_counts": dict(sorted(used.items())),
            "encoding_warning_mail_counts": dict(sorted(warnings.items())),
            "attempted_decoded_characters": characters, "replacement_characters": replacements,
            "replacement_ratio": replacements / max(1, characters),
            "decoding_statistics_scope": "One decoded outcome per examined subject segment/body candidate, including rejected candidates; not internal codec probes.",
            "retained_characters": retained_characters, "retained_replacement_characters": retained_replacements,
            "retained_replacement_ratio": retained_replacements / max(1, retained_characters),
            "text_length_characters": distribution(lengths),
            "quality_note": "Zero replacement characters do not prove absence of semantic mojibake. Review real samples.",
            "passed": sum(status.values()) == len(rows) and status["error"] == 0,
        }
        temporary = summary_path.with_name(summary_path.name + ".tmp")
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(summary_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "fatal_error_type": type(exc).__name__, "message": "Check input audit, configuration and file access; previous outputs may remain."}))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
