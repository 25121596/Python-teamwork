"""Deterministic exact deduplication and exposure-aware grouped holdouts."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

LABELS = {"ham": 0, "spam": 1}
SPLITS = ("train", "validation", "test")
POLICY_VERSION = "1.0"
SIGNATURE_FIELDS = ("raw_sha256", "text_sha256", "body_sha256", "ablation_sha256")


def signature(text: str, field: str) -> str:
    return hashlib.sha256(("trec06c-content-v1\0" + field + "\0" + text).encode("utf-8")).hexdigest()


def has_signal(text: str) -> bool:
    return any(char.isalnum() for char in text)


def size_distribution(sizes: list[int]) -> dict:
    values = sorted(sizes)
    result = {"groups": len(values), "max": max(values, default=0)}
    for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99)):
        result[name] = values[max(0, math.ceil(len(values) * fraction) - 1)] if values else 0
    return result


def signature_audit(groups: dict[str, list[dict]]) -> dict:
    duplicates = [rows for rows in groups.values() if len(rows) > 1]
    conflicts = [rows for rows in groups.values() if len({row["label"] for row in rows}) > 1]
    return {
        "records_with_signature": sum(map(len, groups.values())),
        "unique_groups": len(groups), "duplicate_groups": len(duplicates),
        "records_in_duplicate_groups": sum(map(len, duplicates)),
        "duplicate_extra_records": sum(len(rows) - 1 for rows in duplicates),
        "mixed_label_groups": len(conflicts),
        "records_in_mixed_label_groups": sum(map(len, conflicts)),
        "duplicate_group_sizes": size_distribution([len(rows) for rows in duplicates]),
    }


@dataclass
class SplitPreparation:
    ledger: list[dict]
    representatives: list[dict]
    summary: dict


def prepare_records(rows: list[dict], development_ids: set[str]) -> SplitPreparation:
    """Do not collapse differing subjects or mistake body groups for label errors."""
    ids = [row["mail_id"] for row in rows]
    if len(ids) != len(set(ids)) or not set(development_ids) <= set(ids):
        raise ValueError("Unique input IDs and a development subset are required.")
    metadata = []
    for row in rows:
        if row["label_name"] not in LABELS or row["label"] != LABELS[row["label_name"]]:
            raise ValueError("Unexpected labels.")
        usable = row["usable"]
        if not isinstance(usable, bool) or usable != has_signal(row["text_clean"]):
            raise ValueError("Inconsistent usable text flag.")
        body_valid = not row["body_empty"] and has_signal(row["body_clean"])
        ablation = row.get("text_unstripped_html", row["text_clean"])
        metadata.append({
            "mail_id": row["mail_id"], "label_name": row["label_name"], "label": row["label"],
            "usable": usable, "is_development": row["mail_id"] in development_ids,
            "raw_sha256": row["raw_sha256"],
            "text_sha256": signature(row["text_clean"], "text") if usable else "",
            "body_sha256": signature(row["body_clean"], "body") if body_valid else "",
            "ablation_sha256": signature(ablation, "ablation") if usable and has_signal(ablation) else "",
            "precontact_body_sha256": row.get("precontact_body_sha256", ""),
        })
    metadata.sort(key=lambda row: row["mail_id"])
    parent = {row["mail_id"]: row["mail_id"] for row in metadata}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        a, b = sorted((find(first), find(second)))
        parent[b] = a

    buckets = {field: defaultdict(list) for field in SIGNATURE_FIELDS}
    for row in metadata:
        for field in SIGNATURE_FIELDS:
            if row[field]:
                buckets[field][row[field]].append(row)
    for groups in buckets.values():
        for group in groups.values():
            for row in group[1:]:
                union(group[0]["mail_id"], row["mail_id"])
    components = defaultdict(list)
    for row in metadata:
        components[find(row["mail_id"])].append(row)
    for members in components.values():
        component_id = "group_" + signature(min(row["mail_id"] for row in members), "leakage-group")
        forced = any(row["is_development"] for row in members)
        for row in members:
            row.update(leakage_group_id=component_id, forced_train=forced)

    conflicts = defaultdict(set)
    for field in ("raw_sha256", "text_sha256"):
        for group in buckets[field].values():
            if len({row["label"] for row in group}) > 1:
                for row in group:
                    conflicts[row["mail_id"]].add(field)
    eligible = defaultdict(list)
    for row in metadata:
        if row["mail_id"] in conflicts:
            row.update(disposition="conflict", representative_id="", reason="+".join(sorted(conflicts[row["mail_id"]])))
        elif not row["usable"]:
            row.update(disposition="unusable", representative_id="", reason="no_usable_text")
        else:
            eligible[row["text_sha256"]].append(row)
    for group in eligible.values():
        representative = min(group, key=lambda row: (not row["is_development"], row["mail_id"]))
        for row in group:
            retained = row is representative
            row.update(disposition="retained" if retained else "duplicate",
                       representative_id=representative["mail_id"],
                       reason="canonical" if retained else "same_label_exact_text")
    reps = [dict(row) for row in metadata if row["disposition"] == "retained"]
    retained_components = defaultdict(list)
    for row in reps:
        retained_components[row["leakage_group_id"]].append(row)
    by_label = {}
    for label in LABELS:
        counts = Counter(row["disposition"] for row in metadata if row["label_name"] == label)
        by_label[label] = {name: counts[name] for name in ("retained", "duplicate", "conflict", "unusable")}
        by_label[label]["input"] = sum(counts.values())
    contact_groups = [group for group in buckets["body_sha256"].values()
                      if len({row["precontact_body_sha256"] for row in group if row["precontact_body_sha256"]}) > 1]
    summary = {
        "input_records": len(metadata), "disposition_counts": dict(sorted(Counter(row["disposition"] for row in metadata).items())),
        "by_label": by_label,
        "signature_audits": {field: signature_audit(groups) for field, groups in buckets.items()},
        "leakage_groups_before_dedup": size_distribution([len(group) for group in components.values()]),
        "leakage_groups_after_dedup": size_distribution([len(group) for group in retained_components.values()]),
        "mixed_label_retained_leakage_groups": sum(len({row["label"] for row in group}) > 1 for group in retained_components.values()),
        "forced_training_groups": sum(group[0]["forced_train"] for group in retained_components.values()),
        "forced_training_representatives": sum(row["forced_train"] for row in reps),
        "development_input_records": len(development_ids),
        "development_dispositions": dict(sorted(Counter(row["disposition"] for row in metadata if row["is_development"]).items())),
        "contact_mask_audit": {
            "reconstructable_body_records": sum(bool(row["precontact_body_sha256"]) for row in metadata),
            "body_groups_merging_distinct_precontact_surfaces": len(contact_groups),
            "records_in_those_groups": sum(map(len, contact_groups)),
            "scope": "Body only; precontact surfaces already normalize HTML, Unicode and whitespace. Not a causal count of all final-text duplicate removals.",
        },
    }
    return SplitPreparation(metadata, reps, summary)


def integer_targets(total: int, ratios: dict) -> dict[str, int]:
    exact = {name: total * ratios[name] for name in SPLITS}
    result = {name: math.floor(exact[name]) for name in SPLITS}
    order = sorted(SPLITS, key=lambda name: (-(exact[name] - result[name]), SPLITS.index(name)))
    for name in order[:total - sum(result.values())]:
        result[name] += 1
    return result


def validate_parameters(seed: int, ratios: dict) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")
    if set(ratios) != set(SPLITS) or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or not 0 < v < 1 for v in ratios.values()):
        raise ValueError("Use positive finite train/validation/test ratios.")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Ratios must sum to one.")


def assign_splits(representatives: list[dict], *, seed: int, ratios: dict) -> list[dict]:
    """Greedy whole-group allocation minimizing normalized class count error."""
    validate_parameters(seed, ratios)
    if len({row["mail_id"] for row in representatives}) != len(representatives):
        raise ValueError("Representative IDs must be unique.")
    groups = defaultdict(list)
    for row in representatives:
        groups[row["leakage_group_id"]].append(row)
    totals = Counter(row["label_name"] for row in representatives)
    targets = {label: integer_targets(totals[label], ratios) for label in LABELS}
    counts = {name: Counter() for name in SPLITS}
    assignments = {}
    for group_id, group in groups.items():
        if any(row["forced_train"] for row in group):
            assignments[group_id] = "train"
            counts["train"].update(row["label_name"] for row in group)
    remaining = sorted((group_id for group_id in groups if group_id not in assignments),
                       key=lambda key: (-len(groups[key]), signature(f"{seed}\0{key}", "split-order")))
    for group_id in remaining:
        group_counts = Counter(row["label_name"] for row in groups[group_id])

        def cost(name):
            # Other sets' terms stay constant; compare only this set's delta.
            delta = sum(((counts[name][label] + group_counts[label] - targets[label][name]) ** 2
                         - (counts[name][label] - targets[label][name]) ** 2)
                        / max(1, targets[label][name]) for label in LABELS)
            return delta, signature(f"{seed}\0{group_id}\0{name}", "split-tie")

        chosen = min(SPLITS, key=cost)
        assignments[group_id] = chosen
        counts[chosen].update(group_counts)
    return [dict(row, split=assignments[row["leakage_group_id"]])
            for row in sorted(representatives, key=lambda row: row["mail_id"])]


def allocation_summary(assigned: list[dict], ratios: dict) -> dict:
    total = len(assigned)
    labels = Counter(row["label_name"] for row in assigned)
    output = {}
    for name in SPLITS:
        members = [row for row in assigned if row["split"] == name]
        counts = Counter(row["label_name"] for row in members)
        output[name] = {
            "records": len(members), "fraction": len(members) / max(1, total),
            "target_fraction": ratios[name], "fraction_deviation": len(members) / max(1, total) - ratios[name],
            "groups": len({row["leakage_group_id"] for row in members}),
            "label_counts": {label: counts[label] for label in LABELS},
            "target_label_counts": {label: integer_targets(labels[label], ratios)[name] for label in LABELS},
            "spam_fraction": counts["spam"] / max(1, len(members)),
        }
    return output
