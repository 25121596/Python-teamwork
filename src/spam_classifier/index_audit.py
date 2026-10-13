"""Audit TREC index metadata without opening any mail message."""

from __future__ import annotations

import hashlib
import math
import os
import random
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


LABELS = {"ham": 0, "spam": 1}
MANIFEST_FIELDS = (
    "mail_id", "relative_path", "index_order", "label_name", "label",
    "size_bytes", "status", "index_occurrences",
)


@dataclass
class AuditResult:
    records: list[dict]
    issues: list[dict]
    development_sample: list[dict]
    summary: dict


def mail_id(relative_path: str) -> str:
    """An ID for a path, not a content hash or a deduplication signature."""
    value = ("trec06c-path-v1\0" + relative_path).encode("utf-8")
    return "trec06c_" + hashlib.sha256(value).hexdigest()


def is_auxiliary(relative_path: str) -> bool:
    return any(
        part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
        for part in PurePosixPath(relative_path).parts
    )


def balanced_sample(records: list[dict], size: int, seed: int) -> list[dict]:
    """Sample reproducibly, giving both labels equal coverage where possible."""
    if size <= 0:
        raise ValueError("Development sample size must be positive.")
    groups = {
        label: sorted(
            (row for row in records if row["status"] == "eligible" and row["label_name"] == label),
            key=lambda row: row["relative_path"],
        )
        for label in sorted(LABELS)
    }
    target = min(size, sum(map(len, groups.values())))
    quotas = {label: min(len(rows), target // 2) for label, rows in groups.items()}
    while sum(quotas.values()) < target:
        for label, rows in groups.items():
            if quotas[label] < len(rows) and sum(quotas.values()) < target:
                quotas[label] += 1
    rng = random.Random(seed)
    selected = [row for label, rows in groups.items() for row in rng.sample(rows, quotas[label])]
    return sorted(selected, key=lambda row: (row["label_name"], row["relative_path"]))


def size_statistics(sizes: list[int]) -> dict:
    ordered = sorted(sizes)
    result = {"count": len(sizes), "total_bytes": sum(sizes), "percentile_method": "nearest_rank_ceil_p_times_n"}
    if not ordered:
        return dict(result, min_bytes=None, p50_bytes=None, p90_bytes=None, p95_bytes=None, p99_bytes=None, max_bytes=None)
    result.update(min_bytes=ordered[0], max_bytes=ordered[-1])
    for name, probability in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
        result[name + "_bytes"] = ordered[max(0, math.ceil(probability * len(ordered)) - 1)]
    return result


def audit_index(data_root: Path, *, development_size: int = 400, seed: int = 42) -> AuditResult:
    """Read only full/index and stat/scan metadata under data_root/data."""
    if development_size <= 0:
        raise ValueError("Development sample size must be positive.")
    data_root = Path(data_root).resolve()
    data_dir = (data_root / "data").resolve()
    if not data_dir.is_dir() or not data_dir.is_relative_to(data_root):
        raise ValueError("Expected a data directory within the corpus root.")
    index_path = data_root / "full" / "index"
    issues: list[dict] = []
    counters = Counter()
    label_counts = Counter()
    candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)
    digest = hashlib.sha256()

    def issue(kind: str, *, line: int | None = None, path: str | None = None, **detail) -> None:
        item = {"kind": kind}
        if line is not None:
            item["index_order"] = line
        if path is not None:
            item["relative_path"] = path
        item.update(detail)
        issues.append(item)

    with index_path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            counters["physical_lines"] += 1
            try:
                line = raw.decode("utf-8").strip()
                if number == 1:
                    line = line.lstrip("\ufeff")
            except UnicodeDecodeError:
                counters["malformed_rows"] += 1
                issue("invalid_index_encoding", line=number)
                continue
            if not line:
                counters["blank_lines"] += 1
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                counters["malformed_rows"] += 1
                issue("malformed_row", line=number)
                continue
            label, raw_path = fields
            if label not in LABELS:
                counters["unknown_label_rows"] += 1
                issue("unknown_label", line=number)
                continue
            # ../data is valid; absolute paths and resolved escapes are not.
            raw_path = raw_path.replace("\\", "/")
            if PurePosixPath(raw_path).is_absolute() or PureWindowsPath(raw_path).drive:
                counters["unsafe_path_rows"] += 1
                issue("absolute_index_path", line=number)
                continue
            try:
                target = (index_path.parent / raw_path).resolve()
                if not target.is_relative_to(data_dir):
                    counters["unsafe_path_rows"] += 1
                    issue("path_outside_data", line=number)
                    continue
                relative = target.relative_to(data_root).as_posix()
            except (OSError, RuntimeError, ValueError):
                counters["path_resolution_error_rows"] += 1
                issue("path_resolution_error", line=number)
                continue
            if is_auxiliary(relative):
                counters["auxiliary_path_rows"] += 1
                issue("auxiliary_index_path", line=number, path=relative)
                continue
            candidates[relative].append((number, label))
            counters["accepted_rows"] += 1
            label_counts[label] += 1

    records = []
    duplicate_paths = conflict_paths = conflict_rows = 0
    sizes = []
    for relative, occurrences in sorted(candidates.items()):
        labels = {label for _, label in occurrences}
        if len(occurrences) > 1:
            duplicate_paths += 1
            issue("duplicate_path", line=occurrences[0][0], path=relative, occurrences=len(occurrences))
        conflict = len(labels) > 1
        if conflict:
            conflict_paths += 1
            conflict_rows += len(occurrences)
            issue("label_conflict", line=occurrences[0][0], path=relative)
        label = next(iter(labels)) if not conflict else ""
        row = {
            "mail_id": mail_id(relative), "relative_path": relative,
            "index_order": occurrences[0][0], "label_name": label,
            "label": LABELS[label] if label else "", "size_bytes": "",
            "status": "eligible", "index_occurrences": len(occurrences),
        }
        try:
            metadata = (data_root / relative).stat()
            if not stat.S_ISREG(metadata.st_mode):
                row["status"] = "non_regular"
            else:
                row["size_bytes"] = metadata.st_size
                sizes.append(metadata.st_size)
                if metadata.st_size == 0:
                    row["status"] = "empty"
                elif conflict:
                    row["status"] = "label_conflict"
        except FileNotFoundError:
            row["status"] = "missing"
        except OSError:
            row["status"] = "stat_error"
        if row["status"] not in {"eligible", "label_conflict"}:
            issue(row["status"], line=row["index_order"], path=relative)
        records.append(row)

    inventory: set[str] = set()
    inventory_counts = Counter()
    pending = [data_dir]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    relative = Path(entry.path).relative_to(data_root).as_posix()
                    try:
                        if entry.name == "__MACOSX" and entry.is_dir(follow_symlinks=False):
                            inventory_counts["helper_directories"] += 1
                        elif entry.name.startswith("._") or entry.name == ".DS_Store":
                            inventory_counts["helper_files"] += 1
                        elif entry.is_symlink():
                            inventory_counts["symlink_entries"] += 1
                            issue("inventory_symlink", path=relative)
                        elif entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            inventory.add(relative)
                        else:
                            inventory_counts["scan_errors"] += 1
                            issue("inventory_non_regular", path=relative)
                    except OSError:
                        inventory_counts["scan_errors"] += 1
                        issue("inventory_entry_error", path=relative)
        except OSError:
            inventory_counts["scan_errors"] += 1
            issue("inventory_directory_error", path=directory.relative_to(data_root).as_posix())
    unindexed = sorted(inventory - candidates.keys())
    for relative in unindexed:
        issue("unindexed_file", path=relative)
    absent_from_inventory = sorted(candidates.keys() - inventory)
    for relative in absent_from_inventory:
        issue("indexed_path_not_in_inventory", path=relative)
    eligible = [row for row in records if row["status"] == "eligible"]
    if not eligible:
        issue("no_eligible_mail")
    sample = balanced_sample(records, development_size, seed)
    summary = {
        "schema_version": 1,
        "scope": "Index and filesystem metadata only; mail bodies are not read.",
        "canonical_index": "full/index", "index_sha256": digest.hexdigest(),
        "mail_id_scheme": "trec06c_ + SHA256(UTF8(trec06c-path-v1 + NUL + normalized relative_path))",
        "index": {
            **{key: counters[key] for key in (
                "physical_lines", "blank_lines", "malformed_rows", "unknown_label_rows",
                "unsafe_path_rows", "path_resolution_error_rows", "auxiliary_path_rows", "accepted_rows",
            )},
            "accepted_label_counts": {label: label_counts[label] for label in LABELS},
            "unique_paths": len(records), "duplicate_paths": duplicate_paths,
            "duplicate_extra_rows": counters["accepted_rows"] - len(records),
            "label_conflict_paths": conflict_paths, "label_conflict_rows": conflict_rows,
        },
        "files": {
            "unique_status_counts": dict(sorted(Counter(row["status"] for row in records).items())),
            "eligible_unique_files": len(eligible),
            "eligible_label_counts": {label: sum(row["label_name"] == label for row in eligible) for label in LABELS},
            "size_bytes": size_statistics(sizes),
        },
        "inventory": {
            "candidate_files": len(inventory), "unindexed_files": len(unindexed),
            "indexed_paths_not_in_inventory": len(absent_from_inventory),
            **{key: inventory_counts[key] for key in (
                "helper_directories", "helper_files", "symlink_entries", "scan_errors",
            )},
        },
        "development_sample": {
            "purpose": "Parser development only; not a train/validation/test split.",
            "strategy": "Equal class quotas, capped by availability; sorted paths, then random.Random(seed).sample.",
            "requested_size": development_size, "actual_size": len(sample), "seed": seed,
            "label_counts": {label: sum(row["label_name"] == label for row in sample) for label in LABELS},
        },
        "issue_counts": dict(sorted(Counter(item["kind"] for item in issues).items())),
        "issue_records": len(issues), "passed": not issues,
    }
    return AuditResult(records, issues, sample, summary)
