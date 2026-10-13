"""Check the project interpreter, imports, configuration and index readability.

This is an environment check, not a dataset audit or model experiment.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "jieba": "jieba",
    "beautifulsoup4": "bs4",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "ipykernel": "ipykernel",
}


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="JSON config path, relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/audit/environment_check.json",
        help="Safe JSON summary; relative paths are based on the project root.",
    )
    args = parser.parse_args()

    report = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "virtual_environment": sys.prefix != sys.base_prefix,
        "supported_python": (3, 11) <= sys.version_info[:2] < (3, 13),
        "imports": {},
        "dataset": {},
        "errors": [],
    }
    if not report["virtual_environment"]:
        report["errors"].append("Run with the project's virtual-environment Python.")
    if not report["supported_python"]:
        report["errors"].append("The project currently targets Python 3.11 or 3.12.")

    for distribution, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            report["imports"][distribution] = {
                "ok": True,
                "version": importlib.metadata.version(distribution),
            }
        except Exception as exc:
            report["imports"][distribution] = {
                "ok": False,
                "error_type": type(exc).__name__,
            }
            report["errors"].append(f"Cannot import {module}: {type(exc).__name__}")

    selected = args.config or (
        "config.local.json"
        if (PROJECT_ROOT / "config.local.json").is_file()
        else "config.example.json"
    )
    try:
        config_path = project_path(selected).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        configured_root = os.environ.get("TREC06C_ROOT") or config["data_root"]
        data_root = Path(configured_root).expanduser()
        if not data_root.is_absolute():
            data_root = config_path.parent / data_root
        data_root = data_root.resolve()
        if data_root.is_relative_to(PROJECT_ROOT):
            report["errors"].append("Keep the raw dataset outside the repository directory.")
            raise ValueError("Dataset inside the repository is not allowed.")
        index_relative = Path(config["index_file"])
        if index_relative.as_posix() != "full/index":
            raise ValueError("Use full/index as the canonical index.")
        index_path = data_root / index_relative
        report["dataset"] = {
            "index_file": index_relative.as_posix(),
            "data_root_exists": data_root.is_dir(),
            "data_directory_exists": (data_root / "data").is_dir(),
            "index_readable": False,
            "first_record_valid": False,
            "first_mail_exists": False,
        }
        with index_path.open("r", encoding="ascii") as stream:
            line = next((line.strip() for line in stream if line.strip()), "")
        report["dataset"]["index_readable"] = True
        fields = line.split(maxsplit=1)
        valid = len(fields) == 2 and fields[0] in {"ham", "spam"}
        report["dataset"]["first_record_valid"] = valid
        if valid:
            # Paths in full/index are relative to the full directory.
            first_mail = (index_path.parent / fields[1]).resolve()
            report["dataset"]["first_mail_exists"] = (
                first_mail.is_relative_to((data_root / "data").resolve())
                and first_mail.is_file()
            )
        if not all(
            report["dataset"][key]
            for key in (
                "data_root_exists",
                "data_directory_exists",
                "index_readable",
                "first_record_valid",
                "first_mail_exists",
            )
        ):
            report["errors"].append("Dataset location or the first index record is invalid.")
        if config["labels"] != {"ham": 0, "spam": 1}:
            raise ValueError("Expected ham=0 and spam=1.")
        ratios = config["split_ratios"]
        values = [ratios[key] for key in ("train", "validation", "test")]
        if not all(isinstance(value, (int, float)) and 0 < value < 1 for value in values):
            raise ValueError("Split ratios must be between zero and one.")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to one.")
    except Exception as exc:
        # Do not publish personal filesystem paths or mail content in the report.
        report["errors"].append(f"Configuration/dataset check failed: {type(exc).__name__}")

    report["passed"] = not report["errors"]
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
