"""Create aggregate-only EDA tables and report figures for the frozen corpus.

The validated dataset loader is the only source of message records. This script
uses labels and derived scalar metadata, and computes text lengths without
printing or serializing message content.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABELS = ("ham", "spam")
SPLITS = ("train", "validation", "test")
EXPECTED_SPLIT_COUNTS = {"train": 23792, "validation": 5099, "test": 5097}


def _label(record: dict[str, Any]) -> str:
    value = record.get("label_name")
    if value in LABELS:
        return value
    value = record.get("label")
    if value in (0, "0"):
        return "ham"
    if value in (1, "1"):
        return "spam"
    raise ValueError("Dataset record is missing a recognized class label")


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _get(record: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def _quantiles(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)

    def q(fraction: float) -> int:
        if not ordered:
            return 0
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "n": len(ordered),
        "mean": round(mean(ordered), 2) if ordered else 0,
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": max(ordered, default=0),
    }


def _has_encoding_warning(record: dict[str, Any]) -> bool:
    value = _get(record, "encoding_warnings", "encoding_warning", default=False)
    return _truth(value)


def _has_html(record: dict[str, Any]) -> bool:
    return _truth(_get(record, "has_html", "html_present", default=False))


def _is_fallback(record: dict[str, Any]) -> bool:
    return _truth(_get(record, "decode_fallback", "used_decode_fallback", default=False))


def _is_subject_only(record: dict[str, Any]) -> bool:
    status = str(_get(record, "parse_status", "status", default="")).lower()
    return status == "subject_only"


def _set_chinese_font() -> str:
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    else:
        name = "DejaVu Sans"
        plt.rcParams["font.sans-serif"] = [name]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    return name


def _save_category_figure(raw: Counter, cleaned: Counter, representatives: Counter, output: Path) -> None:
    stages = (("原始索引", raw), ("清洗缓存", cleaned), ("去重代表", representatives))
    fig, (ax, rate_ax) = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = {"ham": "#4C78A8", "spam": "#E45756"}
    x = list(range(len(stages)))
    width = 0.34
    for offset, label in enumerate(LABELS):
        vals = [counts[label] for _, counts in stages]
        bars = ax.bar([i + (offset - 0.5) * width for i in x], vals, width, label=label.upper(), color=colors[label])
        ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_xticks(x, [name for name, _ in stages])
    ax.set_ylabel("邮件数（封）")
    ax.set_title("类别数量：原始 → 清洗 → 去重")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    raw_total = sum(raw.values())
    rates = [100 * representatives[label] / raw[label] if raw[label] else 0 for label in LABELS]
    bars = rate_ax.bar(["HAM", "SPAM"], rates, color=[colors[x] for x in LABELS], width=0.58)
    rate_ax.bar_label(bars, labels=[f"{v:.1f}%" for v in rates], padding=4)
    rate_ax.set_ylim(0, max(100, max(rates, default=0) * 1.18))
    rate_ax.set_ylabel("代表数 / 原始数（%）")
    rate_ax.set_title(f"按类保留率（总计 {sum(representatives.values()):,}/{raw_total:,}）")
    rate_ax.grid(axis="y", alpha=0.22)
    rate_ax.set_axisbelow(True)
    fig.suptitle("TREC06C 数据规模与去重保留", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_length_figure(lengths: dict[str, dict[str, list[int]]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    data = [lengths[split][label] for split in SPLITS for label in LABELS]
    data.extend([[size for split in SPLITS for size in lengths[split][label]] for label in LABELS])
    labels = [f"{split}\n{label.upper()}" for split in SPLITS for label in LABELS]
    labels.extend([f"全部集合\n{label.upper()}" for label in LABELS])
    boxes = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, whis=(5, 95))
    palette = {"ham": "#4C78A8", "spam": "#E45756"}
    for box, label in zip(boxes["boxes"], [label for split in SPLITS for label in LABELS] + list(LABELS)):
        box.set_facecolor(palette[label])
        box.set_alpha(0.7)
    ax.set_ylabel("text_clean 字符数（Unicode 字符）")
    ax.set_title("固定划分代表邮件的清洗文本长度分布")
    ax.text(0.01, 0.98, "箱体：四分位范围；中线：中位数；须：第 5–95 百分位；隐藏离群点仅为可读性。", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_quality_figure(quality: dict[str, dict[str, dict[str, float | int]]], output: Path) -> None:
    features = (("subject_only", "仅主题"), ("has_html", "所选正文HTML"), ("decode_fallback", "解码回退"), ("encoding_warning", "编码警告"))
    groups = [(split, label) for split in SPLITS for label in LABELS]
    x = list(range(len(groups)))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    palette = ("#72B7B2", "#F2CF5B", "#B279A2", "#FF9DA6")
    for i, (key, title) in enumerate(features):
        values = [quality[split][label][key + "_rate"] * 100 for split, label in groups]
        bars = ax.bar([v + (i - 1.5) * width for v in x], values, width, label=title, color=palette[i])
        for bar, value in zip(bars, values):
            if value >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x, [f"{sp}\n{lab.upper()}" for sp, lab in groups])
    ax.set_ylabel("该类别邮件比例（%）")
    ax.set_ylim(0, 105)
    ax.set_title("解析与编码质量特征（按固定集合及类别）")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_split_figure(split_counts: dict[str, Counter], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    x = list(range(len(SPLITS)))
    ham = [split_counts[s]["ham"] for s in SPLITS]
    spam = [split_counts[s]["spam"] for s in SPLITS]
    bars_h = ax.bar(x, ham, label="HAM", color="#4C78A8")
    bars_s = ax.bar(x, spam, bottom=ham, label="SPAM", color="#E45756")
    ax.bar_label(bars_h, fmt="%d", label_type="center", color="white", fontsize=9)
    ax.bar_label(bars_s, fmt="%d", label_type="center", color="white", fontsize=9)
    for i, split in enumerate(SPLITS):
        total = ham[i] + spam[i]
        ax.text(i, total + max(ham + spam) * 0.015, f"N={total:,}", ha="center", fontsize=9)
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("邮件数（封）")
    ax.set_title("冻结数据划分的类别数量")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(bundle: Any, output_root: Path) -> dict[str, Any]:
    all_records = bundle.all_records
    split_records = bundle.splits
    if len(all_records) != 64620:
        raise ValueError(f"Expected 64,620 complete records, got {len(all_records)}")
    if set(split_records) != set(SPLITS):
        raise ValueError("Dataset loader must expose train, validation and test splits")

    raw_counts = Counter(_label(record) for record in all_records)
    cleaned_counts = Counter()
    cleaned_status_counts = Counter()
    for record in all_records:
        label = _label(record)
        usable = _get(record, "usable", "is_usable", default=None)
        if usable is None:
            usable = bool(_get(record, "text_clean", default=""))
        if _truth(usable):
            cleaned_counts[label] += 1
        cleaned_status_counts[str(_get(record, "parse_status", "status", default="unknown"))] += 1

    representatives: dict[str, list[dict[str, Any]]] = {sp: list(split_records[sp]) for sp in SPLITS}
    split_counts = {sp: Counter(_label(record) for record in representatives[sp]) for sp in SPLITS}
    if {sp: len(representatives[sp]) for sp in SPLITS} != EXPECTED_SPLIT_COUNTS:
        raise ValueError("Frozen split counts differ from expected stage-04 counts")
    representative_counts = Counter()
    for sp in SPLITS:
        representative_counts.update(split_counts[sp])

    lengths: dict[str, dict[str, list[int]]] = {sp: {label: [] for label in LABELS} for sp in SPLITS}
    quality: dict[str, dict[str, dict[str, Any]]] = {sp: {} for sp in SPLITS}
    for split in SPLITS:
        for label in LABELS:
            group = [r for r in representatives[split] if _label(r) == label]
            lengths[split][label] = [len(str(_get(r, "text_clean", default="") or "")) for r in group]
            n = len(group)
            feature_counts = {
                "subject_only": sum(_is_subject_only(r) for r in group),
                "has_html": sum(_has_html(r) for r in group),
                "decode_fallback": sum(_is_fallback(r) for r in group),
                "encoding_warning": sum(_has_encoding_warning(r) for r in group),
            }
            quality[split][label] = {
                key: count for key, count in feature_counts.items()
            } | {
                key + "_rate": (count / n if n else 0.0) for key, count in feature_counts.items()
            }

    output_root.mkdir(parents=True, exist_ok=True)
    figures = {
        "category_counts": output_root / "stage05_category_counts.png",
        "length_distribution": output_root / "stage05_length_distribution.png",
        "quality_features": output_root / "stage05_quality_features.png",
        "fixed_split_class_counts": output_root / "stage05_fixed_split_class_counts.png",
    }
    _set_chinese_font()
    _save_category_figure(raw_counts, cleaned_counts, representative_counts, figures["category_counts"])
    _save_length_figure(lengths, figures["length_distribution"])
    _save_quality_figure(quality, figures["quality_features"])
    _save_split_figure(split_counts, figures["fixed_split_class_counts"])

    length_summary = {
        split: {label: _quantiles(lengths[split][label]) for label in LABELS}
        for split in SPLITS
    }
    length_summary["all"] = {
        label: _quantiles([size for split in SPLITS for size in lengths[split][label]])
        for label in LABELS
    }
    raw_total = sum(raw_counts.values())
    representative_total = sum(representative_counts.values())
    summary = {
        "scope": "Aggregate-only EDA from validated dataset loader; no message text serialized or printed.",
        "interpretation": "Descriptive corpus and fixed-split statistics only; these results must not be used to select vocabulary, token filters, thresholds or model features.",
        "counts": {
            "raw_indexed": {label: raw_counts[label] for label in LABELS},
            "cleaned_usable": {label: cleaned_counts[label] for label in LABELS},
            "cleaned_parse_status": dict(sorted(cleaned_status_counts.items())),
            "deduplicated_representatives": {label: representative_counts[label] for label in LABELS},
            "split_counts": {
                split: {"total": len(representatives[split]), **{label: split_counts[split][label] for label in LABELS}}
                for split in SPLITS
            },
            "overall_retention_rate": representative_total / raw_total if raw_total else 0.0,
            "retention_rate_by_label": {
                label: representative_counts[label] / raw_counts[label] if raw_counts[label] else 0.0
                for label in LABELS
            },
        },
        "text_clean_character_lengths": {
            "unit": "Unicode code points in text_clean; subject and body if present in text_clean",
            "quantiles": length_summary,
        },
        "quality_features": quality,
        "quality_feature_rate_denominator": "Representatives within each split and class.",
        "figures": {key: f"artifacts/figures/{path.name}" for key, path in figures.items()},
        "passed": len(all_records) == 64620 and representative_total == 33988,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project directory containing config and artifacts")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "src"))
    try:
        from spam_classifier.dataset import load_dataset, source_sha256
    except ImportError as exc:
        raise SystemExit("Validated loader is unavailable; expected src/spam_classifier/dataset.py") from exc

    bundle = load_dataset(project_root)
    summary = analyze(bundle, project_root / "artifacts" / "figures")
    summary["input_provenance"] = bundle.provenance
    summary["eda_source_sha256"] = source_sha256(
        project_root,
        ("src/spam_classifier/dataset.py", "scripts/analyze_corpus.py"),
    )
    summary_path = project_root / "artifacts" / "audit" / "eda_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    # Print only safe aggregate outcome and project-relative artifact names.
    print(json.dumps({"passed": summary["passed"], "counts": summary["counts"], "summary": "artifacts/audit/eda_summary.json", "figures": summary["figures"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
