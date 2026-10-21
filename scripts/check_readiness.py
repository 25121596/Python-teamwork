"""Read-only preflight for continuing stages 7–10, without loading email text."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_INPUTS = (
    "artifacts/processed/mail_manifest.csv",
    "artifacts/processed/cleaned_mail.jsonl",
    "artifacts/processed/dedup_ledger.csv",
    "artifacts/processed/features/train.npz",
    "artifacts/processed/features/validation.npz",
    "artifacts/processed/features/row_order.json",
    "artifacts/models/tfidf_pipeline.joblib",
    "artifacts/processed/stage06/validation_predictions.npz",
)


def check(root: Path) -> dict:
    selection = json.loads((root / "artifacts/metrics/model_selection.json").read_text(encoding="utf-8"))
    expected = set(PRIVATE_INPUTS) | set(selection["private_output_sha256"])
    missing = sorted(name for name in expected if not (root / name).is_file())
    versions, differences = {}, {}
    for key, package in (("numpy", "numpy"), ("scipy", "scipy"),
                         ("scikit_learn", "scikit-learn"), ("joblib", "joblib"),
                         ("threadpoolctl", "threadpoolctl")):
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
        if versions[key] != selection["versions"][key]:
            differences[key] = {"recorded": selection["versions"][key], "current": versions[key]}
    return {
        "scope": "File existence and runtime metadata only; no corpus experiment executed.",
        "missing_private_inputs": missing,
        "runtime_differences_from_stage06": differences,
        "can_reuse_stage06_caches": not missing and not differences,
        "required_for_course_submission": [
            "Teacher's original Word template", "Confirmed group number, leader, member names and student IDs",
            "Actual member contributions", "Recorded five-minute presentation and course upload confirmation",
        ],
        "next_action": "Restore the original corpus outside this repository and rebuild stages 2–6 in a compatible environment; retain frozen split CSVs.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(ROOT)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["can_reuse_stage06_caches"] else 2


if __name__ == "__main__":
    sys.exit(main())
