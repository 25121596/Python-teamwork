"""Audit full/index, write local manifests and a public metadata-only summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spam_classifier.index_audit import MANIFEST_FIELDS, audit_index


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Config path relative to the repository unless absolute.")
    parser.add_argument("--development-size", type=int, default=400)
    args = parser.parse_args()
    try:
        chosen = args.config or ("config.local.json" if (PROJECT_ROOT / "config.local.json").is_file() else "config.example.json")
        config_path = project_path(chosen).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if config["index_file"] != "full/index" or config["labels"] != {"ham": 0, "spam": 1}:
            raise ValueError("Expected canonical full/index with ham=0 and spam=1.")
        seed = config["random_state"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("random_state must be an integer.")
        data_root = Path(os.environ.get("TREC06C_ROOT") or config["data_root"]).expanduser()
        if not data_root.is_absolute():
            data_root = config_path.parent / data_root
        data_root = data_root.resolve()
        if data_root.is_relative_to(PROJECT_ROOT):
            raise ValueError("Keep the raw dataset outside the repository.")
        output_root = project_path(config["artifacts_dir"]).resolve()
        # processed/ is ignored specifically under the fixed project artifacts directory.
        if output_root != PROJECT_ROOT / "artifacts":
            raise ValueError("Use artifacts as output root so local manifests stay Git-ignored.")
        result = audit_index(data_root, development_size=args.development_size, seed=seed)
        write_csv(output_root / "processed" / "mail_manifest.csv", result.records)
        write_csv(output_root / "splits" / "development_sample.csv", result.development_sample)
        write_text(output_root / "processed" / "index_issues.jsonl", "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in result.issues))
        write_text(output_root / "audit" / "index_summary.json", json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result.summary, ensure_ascii=False, indent=2))
        return 0 if result.summary["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        # Avoid filesystem paths or arbitrary source-line text in public output.
        print(json.dumps({"passed": False, "fatal_error_type": type(exc).__name__, "message": "Audit could not run; check configuration, file access and CLI arguments."}))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
