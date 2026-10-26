"""Check that the project guide and the walkthrough deck still match the evidence.

The guide and the deck are hand-authored narrative, but every number they
quote comes from artifacts/. This script re-derives the headline figures
from the published aggregates and asserts that the same figures still
appear in the guide and in the built deck, so a stale document fails
loudly instead of misleading a reader.

The deck is checked as the generated .pptx rather than as its source, so
the check covers what a reader actually sees. Rebuild the deck first when
the generator changes.

Run from the repository root:

    python scripts/check_guide_consistency.py

Exit code 0 means every checked figure was found, 1 means a figure went
missing, and 2 means the check could not run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "项目说明文档.md"
DECK = ROOT / "deliverables" / "项目演示.pptx"


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def group_count(split_summary: dict) -> str:
    """Number of duplicate-linkage groups, read from the recorded audit."""
    audits = split_summary["deduplication"]["signature_audits"]
    for audit in audits.values():
        groups = audit.get("unique_groups")
        if groups:
            return f"{groups:,}"
    raise KeyError("no group count found in the deduplication signature audits")


def expected_facts() -> dict[str, str]:
    """Headline figures that the guide and the deck must both quote."""
    eda = load("artifacts/audit/eda_summary.json")
    cleaning = load("artifacts/audit/cleaning_summary.json")
    split = load("artifacts/audit/split_summary.json")
    final = load("artifacts/metrics/final_evaluation.json")
    ablation = load("artifacts/metrics/cleaning_ablation.json")
    selection = load("artifacts/metrics/model_selection.json")

    metrics = final["metrics"]
    lr = metrics["lr"]
    nb = metrics["nb"]
    majority = metrics["majority"]
    counts = eda["counts"]

    clean_arm = next(c for c in ablation["comparisons"] if c["representation"] == "text_clean")
    html_arm = next(c for c in ablation["comparisons"] if c["representation"] == "text_unstripped_html")
    best_lr = next(e for e in selection["experiments"] if e["candidate_id"] == "lr_C_2")["metrics"]

    return {
        "total_mail": f"{counts['raw_indexed']['ham'] + counts['raw_indexed']['spam']:,}",
        "representatives": f"{counts['deduplicated_representatives']['ham'] + counts['deduplicated_representatives']['spam']:,}",
        "train_rows": f"{counts['split_counts']['train']['total']:,}",
        "validation_rows": f"{counts['split_counts']['validation']['total']:,}",
        "test_rows": f"{counts['split_counts']['test']['total']:,}",
        "ok_count": f"{cleaning['status_counts']['ok']:,}",
        "recovered_count": f"{cleaning['status_counts']['recovered']:,}",
        "subject_only_count": f"{cleaning['status_counts']['subject_only']:,}",
        "duplicates_removed": f"{split['deduplication']['disposition_counts']['duplicate']:,}",
        "linkage_groups": group_count(split),
        "lr_accuracy": percent(lr["accuracy"]),
        "lr_macro_f1": percent(lr["macro_f1"]),
        "lr_spam_precision": percent(lr["spam_precision"]),
        "lr_spam_recall": percent(lr["spam_recall"]),
        "lr_ham_f1": percent(lr["ham_f1"]),
        "lr_fp": str(lr["confusion_matrix"]["fp"]),
        "lr_fn": str(lr["confusion_matrix"]["fn"]),
        "lr_tp": f"{lr['confusion_matrix']['tp']:,}",
        "lr_tn": f"{lr['confusion_matrix']['tn']:,}",
        "nb_accuracy": percent(nb["accuracy"]),
        "nb_macro_f1": percent(nb["macro_f1"]),
        "nb_fp": str(nb["confusion_matrix"]["fp"]),
        "nb_fn": str(nb["confusion_matrix"]["fn"]),
        "majority_accuracy": percent(majority["accuracy"]),
        "majority_fn": f"{majority['confusion_matrix']['fn']:,}",
        "validation_lr_macro_f1": percent(best_lr["macro_f1"]),
        "ablation_clean_macro_f1": percent(clean_arm["metrics"]["macro_f1"], 4),
        "ablation_html_macro_f1": percent(html_arm["metrics"]["macro_f1"], 4),
    }


# The guide is the full written account and must carry every headline figure.
# The deck is a summary: it shows the linkage group count rather than the
# number of removed duplicates, and the confusion counts rather than the
# per-class ham F1. Those two are deliberately not required of the deck.
DECK_OMITTED = ("duplicates_removed", "lr_ham_f1")


def deck_text() -> str:
    """All visible text of the built deck, including table cells."""
    from pptx import Presentation

    presentation = Presentation(DECK)
    parts = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        parts.append(cell.text)
    return "\n".join(parts)


def missing_from(text: str, facts: dict[str, str]) -> list[str]:
    """Keys whose figure is absent from the text."""
    return sorted(key for key, value in facts.items() if value not in text)


def run() -> dict:
    guide_text = GUIDE.read_text(encoding="utf-8")
    facts = expected_facts()

    missing_guide = missing_from(guide_text, facts)
    try:
        deck = deck_text()
    except (ImportError, KeyError) as exc:
        return {
            "schema_version": 1,
            "scope": "Compare headline figures in the project guide and the deck against published aggregates.",
            "guide": GUIDE.relative_to(ROOT).as_posix(),
            "deck": DECK.relative_to(ROOT).as_posix(),
            "checked_facts": len(facts),
            "missing_in_guide": missing_guide,
            "missing_in_deck": [],
            "error": f"the deck could not be read: {type(exc).__name__}",
            "passed": False,
        }
    missing_deck = missing_from(deck, facts)
    omitted_from_deck = [key for key in DECK_OMITTED if key not in missing_deck]
    missing_deck = [key for key in missing_deck if key not in DECK_OMITTED]

    return {
        "schema_version": 1,
        "scope": "Compare headline figures in the project guide and the deck against published aggregates.",
        "guide": GUIDE.relative_to(ROOT).as_posix(),
        "deck": DECK.relative_to(ROOT).as_posix(),
        "checked_facts": len(facts),
        "missing_in_guide": missing_guide,
        "missing_in_deck": missing_deck,
        "deck_summary_omissions": sorted(DECK_OMITTED),
        "checks": {
            "artifacts_readable": True,
            "guide_quotes_every_headline_figure": not missing_guide,
            "deck_quotes_every_headline_figure": not missing_deck,
        },
        "passed": not missing_guide and not missing_deck,
    }


def main() -> int:
    try:
        report = run()
    except (OSError, KeyError, ValueError, StopIteration) as exc:
        print(json.dumps({
            "passed": False,
            "error_type": type(exc).__name__,
            "message": "Guide consistency check could not read the published evidence.",
        }, ensure_ascii=False, indent=2))
        return 2
    target = ROOT / "artifacts" / "audit" / "guide_consistency.json"
    target.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
