"""Independently check frozen splits, exact relations and development exposure."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from spam_classifier.email_parser import clean_text, html_to_text

SIGNATURE_FIELDS = ("raw_sha256", "text_sha256", "body_sha256", "ablation_sha256")
SPLITS = ("train", "validation", "test")


def has_signal(text):
    return any(char.isalnum() for char in text)


def signature(text, field):
    return hashlib.sha256(("trec06c-content-v1\0" + field + "\0" + text).encode("utf-8")).hexdigest()


def distribution(sizes):
    values = sorted(sizes)
    result = {"groups": len(values), "max": max(values, default=0)}
    for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99)):
        result[name] = values[math.ceil(len(values) * fraction) - 1] if values else 0
    return result


def precontact_signature(row):
    if row["body_empty"] or row["body_selection"] not in {"plain", "html"}:
        return ""
    text = html_to_text(row["body_decoded"]) if row["body_selection"] == "html" else row["body_decoded"]
    text = unicodedata.normalize("NFKC", text).lower().replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(char for char in text if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf", "Cs"})
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return signature(text, "precontact-body") if has_signal(text) and clean_text(text) == row["body_clean"] else ""


def read_csv_file(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(root: Path, *, candidates: dict[str, bytes] | None = None) -> dict:
    candidates = candidates or {}

    def read_csv(path):
        content = candidates.get(path.relative_to(root).as_posix())
        if content is None:
            return read_csv_file(path)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        return reader.fieldnames, list(reader)

    def read_json(path):
        content = candidates.get(path.relative_to(root).as_posix())
        return json.loads(content.decode("utf-8") if content is not None else path.read_text(encoding="utf-8"))

    def sha(path):
        content = candidates.get(path.relative_to(root).as_posix())
        return hashlib.sha256(content).hexdigest() if content is not None else sha_file(path)

    art = root / "artifacts"
    summary = read_json(art / "audit/split_summary.json")
    cleaning = read_json(art / "audit/cleaning_summary.json")
    _, canonical_rows = read_csv(art / "processed/mail_manifest.csv")
    canonical = {row["mail_id"]: row for row in canonical_rows}
    _, development_rows = read_csv(art / "splits/development_sample.csv")
    development = {row["mail_id"] for row in development_rows}
    _, ledger_rows = read_csv(art / "processed/dedup_ledger.csv")
    ledger = {row["mail_id"]: row for row in ledger_rows}
    violations = Counter()
    public_rows = []
    public_schema = True
    for name in SPLITS:
        fields, rows = read_csv(art / f"splits/{name}.csv")
        public_schema &= fields == ["mail_id", "label_name", "label", "split"]
        public_rows.extend(rows)
        if any(row["split"] != name for row in rows):
            violations["wrong_split_file"] += 1
    public = {row["mail_id"]: row for row in public_rows}
    signatures = {field: defaultdict(list) for field in SIGNATURE_FIELDS}
    cache_ids = set()
    usable = {}
    with (art / "processed/cleaned_mail.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["mail_id"]
            if identity in cache_ids:
                violations["duplicate_cache_id"] += 1
            cache_ids.add(identity)
            stored, original = ledger.get(identity), canonical.get(identity)
            if stored is None or original is None:
                violations["missing_canonical_or_ledger_row"] += 1
                continue
            if (row["label_name"] != original["label_name"] or str(row["label"]) != original["label"]
                    or stored["label_name"] != row["label_name"] or stored["label"] != str(row["label"])
                    or stored["usable"] != str(row["usable"])
                    or stored["is_development"] != str(identity in development)):
                violations["metadata_mismatch"] += 1
            usable[identity] = row["usable"]
            if row["usable"] != has_signal(row["text_clean"]) or row["body_empty"] != (not has_signal(row["body_clean"])):
                violations["cache_signal_flag_mismatch"] += 1
            if stored["precontact_body_sha256"] != precontact_signature(row):
                violations["precontact_signature_mismatch"] += 1
            computed = {
                "raw_sha256": stored["raw_sha256"],
                "text_sha256": signature(row["text_clean"], "text") if row["usable"] else "",
                "body_sha256": signature(row["body_clean"], "body") if not row["body_empty"] and has_signal(row["body_clean"]) else "",
                "ablation_sha256": signature(row["text_unstripped_html"], "ablation")
                    if row["usable"] and has_signal(row["text_unstripped_html"]) else "",
            }
            for field, value in computed.items():
                if stored[field] != value:
                    violations["signature_mismatch"] += 1
                if value:
                    signatures[field][value].append(identity)
    # Reconstruct relations as an undirected graph rather than reusing the
    # producer's disjoint-set or representative-selection functions.
    adjacency = {identity: set() for identity in ledger}
    conflicts = set()
    cross_split = Counter()
    for field, buckets in signatures.items():
        for members in buckets.values():
            for first, second in zip(members, members[1:]):
                adjacency[first].add(second)
                adjacency[second].add(first)
            if field in {"raw_sha256", "text_sha256"} and len({ledger[m]["label"] for m in members}) > 1:
                conflicts.update(members)
            occupied = {ledger[m]["split"] for m in members if ledger[m]["split"]}
            if len(occupied) > 1:
                cross_split[field] += 1
    visited = set()
    component_count = forced_components = 0
    component_sizes, retained_sizes = [], []
    retained_forced_groups = retained_forced_records = 0
    mixed_retained_components = 0
    for identity in sorted(adjacency):
        if identity in visited:
            continue
        stack, component = [identity], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        visited.update(component)
        component_count += 1
        component_sizes.append(len(component))
        expected_group = "group_" + signature(min(component), "leakage-group")
        exposed = bool(component & development)
        forced_components += exposed
        retained_members = [m for m in component if ledger[m]["disposition"] == "retained"]
        if retained_members:
            retained_sizes.append(len(retained_members))
            retained_forced_groups += exposed
            retained_forced_records += len(retained_members) if exposed else 0
            mixed_retained_components += len({ledger[m]["label"] for m in retained_members}) > 1
        occupied = {ledger[m]["split"] for m in component if ledger[m]["split"]}
        if len(occupied) > 1:
            violations["component_crosses_sets"] += 1
        if exposed and occupied - {"train"}:
            violations["development_component_in_holdout"] += 1
        for member in component:
            if ledger[member]["leakage_group_id"] != expected_group or ledger[member]["forced_train"] != str(exposed):
                violations["component_or_exposure_flag_mismatch"] += 1
    expected_representatives = set()
    for members in signatures["text_sha256"].values():
        eligible = [m for m in members if m not in conflicts and usable[m]]
        if not eligible:
            continue
        representative = sorted(eligible, key=lambda m: (m not in development, m))[0]
        expected_representatives.add(representative)
        for member in eligible:
            stored = ledger[member]
            expected_status = "retained" if member == representative else "duplicate"
            if stored["disposition"] != expected_status or stored["representative_id"] != representative:
                violations["incorrect_dedup_representative"] += 1
            if stored["split"] != ledger[representative]["split"]:
                violations["duplicate_assignment_mismatch"] += 1
    for identity, row in ledger.items():
        if identity in conflicts:
            if row["disposition"] != "conflict" or row["split"] or row["representative_id"]:
                violations["conflict_not_isolated"] += 1
        elif not usable.get(identity, False):
            if row["disposition"] != "unusable" or row["split"] or row["representative_id"]:
                violations["unusable_not_isolated"] += 1
        elif row["split"] not in SPLITS:
            violations["eligible_without_split"] += 1
        if identity in public:
            if any(public[identity][field] != row[field] for field in ("label_name", "label", "split")):
                violations["public_metadata_mismatch"] += 1
    dispositions = Counter(row["disposition"] for row in ledger_rows)
    allocation_matches = True
    allocation_targets_match = True
    all_labels = Counter(row["label_name"] for row in public_rows)
    for name in SPLITS:
        rows = [row for row in public_rows if row["split"] == name]
        counts = Counter(row["label_name"] for row in rows)
        part = summary["allocation"][name]
        allocation_matches &= part["records"] == len(rows) and part["label_counts"] == {label: counts[label] for label in ("ham", "spam")}
        allocation_matches &= part["groups"] == len({ledger[row["mail_id"]]["leakage_group_id"] for row in rows})
        allocation_matches &= abs(part["fraction"] - len(rows) / max(1, len(public_rows))) < 1e-12
        ratio = summary["target_ratios"][name]
        allocation_targets_match &= part["target_fraction"] == ratio
        allocation_targets_match &= abs(part["fraction_deviation"] - (part["fraction"] - ratio)) < 1e-12
        allocation_targets_match &= abs(part["spam_fraction"] - counts["spam"] / max(1, len(rows))) < 1e-12
        allocation_targets_match &= all(abs(part["target_label_counts"][label] - all_labels[label] * ratio) < 1 for label in ("ham", "spam"))
    allocation_targets_match &= all(sum(summary["allocation"][name]["target_label_counts"][label] for name in SPLITS) == all_labels[label] for label in ("ham", "spam"))
    by_label_matches = True
    for label in ("ham", "spam"):
        counts = Counter(row["disposition"] for row in ledger_rows if row["label_name"] == label)
        expected = {name: counts[name] for name in ("retained", "duplicate", "conflict", "unusable")}
        expected["input"] = sum(counts.values())
        by_label_matches &= summary["deduplication"]["by_label"][label] == expected
    signature_counts_match = True
    for field, buckets in signatures.items():
        part = summary["deduplication"]["signature_audits"][field]
        duplicate_groups = [members for members in buckets.values() if len(members) > 1]
        mixed_groups = [members for members in buckets.values() if len({ledger[m]["label"] for m in members}) > 1]
        expected = {
            "records_with_signature": sum(map(len, buckets.values())), "unique_groups": len(buckets),
            "duplicate_groups": len(duplicate_groups), "records_in_duplicate_groups": sum(map(len, duplicate_groups)),
            "duplicate_extra_records": sum(len(members) - 1 for members in duplicate_groups),
            "mixed_label_groups": len(mixed_groups), "records_in_mixed_label_groups": sum(map(len, mixed_groups)),
        }
        signature_counts_match &= all(part[key] == value for key, value in expected.items())
        signature_counts_match &= part["duplicate_group_sizes"] == distribution([len(m) for m in duplicate_groups])
    dedup = summary["deduplication"]
    raw_digest = hashlib.sha256()
    for original in canonical_rows:
        identity = original["mail_id"]
        try:
            value = bytes.fromhex(ledger[identity]["raw_sha256"])
            if len(value) != 32:
                raise ValueError("Invalid SHA256 length.")
            raw_digest.update(identity.encode("ascii") + b"\0" + value)
        except (KeyError, ValueError):
            violations["invalid_raw_signature"] += 1
    contact_groups = [members for members in signatures["body_sha256"].values()
                      if len({ledger[m]["precontact_body_sha256"] for m in members if ledger[m]["precontact_body_sha256"]}) > 1]
    contact = dedup["contact_mask_audit"]
    provenance = summary["input_provenance"]
    policy_digest = hashlib.sha256()
    for name in ("src/spam_classifier/data_split.py", "scripts/split_data.py"):
        policy_digest.update(name.encode("utf-8") + b"\0")
        policy_digest.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
        policy_digest.update(b"\0")
    cache_sha = sha(art / "processed/cleaned_mail.jsonl")
    expected_output_paths = {f"artifacts/splits/{name}.csv" for name in SPLITS} | {"artifacts/processed/dedup_ledger.csv"}
    checks = {
        "preceding_cleaning_passed": cleaning["passed"] is True,
        "input_cache_provenance_matches": cache_sha == provenance["cleaning_cache_sha256"] == cleaning["cache_sha256"],
        "raw_provenance_matches_cleaning_run": provenance["raw_source_digest"] == cleaning["raw_source_digest"],
        "ledger_raw_signatures_match_cleaning_digest": raw_digest.hexdigest() == provenance["raw_source_digest"],
        "parser_source_matches": provenance["parser_source_sha256"] == cleaning["parser_source_sha256"]
            == hashlib.sha256((root / "src/spam_classifier/email_parser.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "split_policy_source_matches": policy_digest.hexdigest() == summary["split_policy_source_sha256"],
        "canonical_manifest_checksum_matches": sha(art / "processed/mail_manifest.csv") == provenance["canonical_manifest_sha256"],
        "development_checksum_matches": sha(art / "splits/development_sample.csv") == provenance["development_sample_sha256"],
        "all_frozen_output_checksums_match": set(summary["output_sha256"]) == expected_output_paths
            and all(sha(root / name) == summary["output_sha256"][name] for name in expected_output_paths),
        "frozen_seed_and_ratios_match_contract": summary["random_state"] == 42
            and summary["target_ratios"] == {"train": .7, "validation": .15, "test": .15},
        "unique_canonical_cache_and_ledger_ids": len(ledger) == len(ledger_rows) == len(canonical) == len(canonical_rows)
            and set(ledger) == set(canonical) == cache_ids,
        "disposition_partition_matches": dict(dispositions) == summary["deduplication"]["disposition_counts"]
            and sum(dispositions.values()) == summary["deduplication"]["input_records"],
        "unique_public_representatives": len(public) == len(public_rows) and set(public) == expected_representatives,
        "public_schema_contains_only_id_label_split": public_schema,
        "two_classes_in_every_split": all({row["label_name"] for row in public_rows if row["split"] == name} == {"ham", "spam"} for name in SPLITS),
        "allocation_counts_and_fractions_match": allocation_matches,
        "allocation_targets_and_class_fractions_match": allocation_targets_match,
        "dedup_class_ledger_matches": by_label_matches,
        "signature_audit_counts_match": signature_counts_match,
        "component_summary_matches": dedup["leakage_groups_before_dedup"]["groups"] == component_count
            and dedup["leakage_groups_before_dedup"] == distribution(component_sizes)
            and dedup["leakage_groups_after_dedup"] == distribution(retained_sizes)
            and dedup["mixed_label_retained_leakage_groups"] == mixed_retained_components
            and dedup["forced_training_groups"] == retained_forced_groups
            and dedup["forced_training_representatives"] == retained_forced_records
            and dedup["development_dispositions"] == dict(Counter(ledger[m]["disposition"] for m in development)),
        "contact_mask_audit_counts_match": contact["reconstructable_body_records"] == sum(bool(row["precontact_body_sha256"]) for row in ledger_rows)
            and contact["body_groups_merging_distinct_precontact_surfaces"] == len(contact_groups)
            and contact["records_in_those_groups"] == sum(map(len, contact_groups)),
        "no_raw_text_body_or_ablation_key_crosses_sets": not cross_split,
        "all_independent_row_and_graph_checks_pass": not violations,
        "development_subset_accounted": len(development) == len(development_rows) == summary["deduplication"]["development_input_records"]
            and development <= set(ledger),
    }
    return {
        "checks": checks, "row_violation_counts": dict(violations), "cross_split_signature_groups": dict(cross_split),
        "independently_reconstructed_components": component_count,
        "development_exposed_components_including_isolated": forced_components,
        "public_representatives": len(public_rows), "input_records": len(ledger_rows),
        "split_policy_source_sha256": policy_digest.hexdigest(),
        "validator_source_sha256": hashlib.sha256((Path(__file__)).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "verification_scope": "Graph, text signatures and contact-mask audit reconstructed from cache. Ledger raw signatures are aggregated in canonical manifest order and compared with the cleaning digest. Raw bytes were read by the generator; this validator does not reread all original files.",
        "passed": all(checks.values()),
    }


def main() -> int:
    try:
        report = check(PROJECT_ROOT)
        target = PROJECT_ROOT / "artifacts/audit/split_validation.json"
        target.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"passed": False, "fatal_error_type": type(exc).__name__, "message": "Check split artifacts and provenance; previous validation may remain."}))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
