"""Synthetic provenance and held-out-boundary tests for model input loading."""

from __future__ import annotations

import csv
import hashlib
import builtins
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spam_classifier.dataset import file_sha256, source_sha256  # noqa: E402
from spam_classifier.model_inputs import (  # noqa: E402
    FEATURE_SOURCES,
    fitted_feature_state,
    load_model_inputs,
    runtime_versions,
)
from spam_classifier.text_features import make_feature_pipeline, tokenizer_fingerprint  # noqa: E402


class ModelInputsFixture:
    """Build a complete, tiny frozen input bundle without any email corpus."""

    source_paths = (
        "src/spam_classifier/dataset.py",
        "src/spam_classifier/text_features.py",
        "scripts/build_features.py",
        "src/spam_classifier/data_split.py",
        "scripts/split_data.py",
        "src/spam_classifier/email_parser.py",
        "scripts/check_splits.py",
        "scripts/check_features.py",
    )

    def __init__(self, root: Path):
        self.root = root
        for relative in self.source_paths:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY / relative, destination)

        self.art = root / "artifacts"
        (self.art / "audit").mkdir(parents=True)
        (self.art / "splits").mkdir(parents=True)
        feature_dir = self.art / "processed/features"
        feature_dir.mkdir(parents=True)
        model_dir = self.art / "models"
        model_dir.mkdir(parents=True)

        self.rows = {
            "train": [
                self.row("train-ham-1", "ham", 0, "train"),
                self.row("train-spam-1", "spam", 1, "train"),
                self.row("train-ham-2", "ham", 0, "train"),
                self.row("train-spam-2", "spam", 1, "train"),
            ],
            "validation": [
                self.row("validation-ham", "ham", 0, "validation"),
                self.row("validation-spam", "spam", 1, "validation"),
            ],
        }
        self.write_splits()
        # Held-out files exist as sentinels. Their contents are deliberately not valid
        # artifacts, so a successful load also proves the loader does not deserialize them.
        self.test_csv = self.art / "splits/test.csv"
        self.test_csv.write_text("DO NOT READ test.csv\n", encoding="utf-8")
        self.cleaned_cache = self.art / "processed/cleaned_mail.jsonl"
        self.cleaned_cache.parent.mkdir(parents=True, exist_ok=True)
        self.cleaned_cache.write_text("DO NOT READ cleaned_mail.jsonl\n", encoding="utf-8")
        self.test_matrix_path = feature_dir / "test.npz"
        self.test_matrix_path.write_bytes(b"DO NOT READ test.npz")

        texts = [
            "sharedanchor hamalpha",
            "sharedanchor spambeta",
            "sharedanchor hamgamma",
            "sharedanchor spamdelta",
        ]
        pipeline = make_feature_pipeline()
        train_matrix = pipeline.fit_transform(texts)
        validation_matrix = pipeline.transform(["sharedanchor hamalpha", "sharedanchor spamdelta"])
        sparse.save_npz(feature_dir / "train.npz", train_matrix)
        sparse.save_npz(feature_dir / "validation.npz", validation_matrix)
        joblib.dump(pipeline, model_dir / "tfidf_pipeline.joblib")
        (feature_dir / "row_order.json").write_text(
            json.dumps({name: [{"mail_id": row["mail_id"], "label": row["label"]} for row in rows]
                        for name, rows in self.rows.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
        coverage = self.art / "processed/features/training_word_coverage.csv"
        coverage.write_text("aggregate-only\n", encoding="utf-8")

        self.write_upstream_reports(train_matrix, validation_matrix, pipeline)

    @staticmethod
    def row(mail_id: str, label_name: str, label: int, split: str) -> dict[str, str | int]:
        return {"mail_id": mail_id, "label_name": label_name, "label": label, "split": split}

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def write_splits(self) -> None:
        for name, rows in self.rows.items():
            path = self.art / f"splits/{name}.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("mail_id", "label_name", "label", "split"), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

    def write_upstream_reports(self, train_matrix, validation_matrix, pipeline) -> None:
        parser_sha = hashlib.sha256((self.root / "src/spam_classifier/email_parser.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        split_policy_sha = source_sha256(self.root, ("src/spam_classifier/data_split.py", "scripts/split_data.py"))
        split_checker_sha = hashlib.sha256((self.root / "scripts/check_splits.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        feature_sha = source_sha256(self.root, FEATURE_SOURCES)
        feature_checker_sha = hashlib.sha256((self.root / "scripts/check_features.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        cache_sha = hashlib.sha256(b"synthetic-cleaning-cache-fingerprint").hexdigest()
        self.cache_sha = cache_sha
        self.write_json(self.art / "audit/cleaning_summary.json", {
            "passed": True, "parser_source_sha256": parser_sha, "cache_sha256": cache_sha,
        })

        split_hashes = {
            f"artifacts/splits/{name}.csv": file_sha256(self.art / f"splits/{name}.csv")
            for name in ("train", "validation", "test")
        }
        frozen = {
            "passed": True, "random_state": 42,
            "target_ratios": {"train": .7, "validation": .15, "test": .15},
            "split_policy_source_sha256": split_policy_sha,
            "input_provenance": {"parser_source_sha256": parser_sha, "cleaning_cache_sha256": cache_sha},
            "output_sha256": split_hashes,
            "allocation": {
                name: {"records": len(rows), "label_counts": {
                    "ham": sum(row["label_name"] == "ham" for row in rows),
                    "spam": sum(row["label_name"] == "spam" for row in rows),
                }}
                for name, rows in self.rows.items()
            },
        }
        # The real loader consumes these two allocation entries; test metadata stays opaque.
        self.write_json(self.art / "audit/split_summary.json", frozen)
        split_validation = {
            "passed": True, "checks": {"synthetic_upstream_check": True},
            "split_policy_source_sha256": split_policy_sha,
            "validator_source_sha256": split_checker_sha,
        }
        self.write_json(self.art / "audit/split_validation.json", split_validation)

        row_order = self.art / "processed/features/row_order.json"
        private_paths = (
            "artifacts/processed/features/train.npz",
            "artifacts/processed/features/validation.npz",
            "artifacts/processed/features/test.npz",
            "artifacts/processed/features/row_order.json",
            "artifacts/processed/features/training_word_coverage.csv",
            "artifacts/models/tfidf_pipeline.joblib",
        )
        default = make_feature_pipeline().named_steps["tfidf"]
        config_keys = ("min_df", "max_df", "max_features", "sublinear_tf", "smooth_idf", "use_idf", "norm", "lowercase", "token_pattern")
        feature_summary = {
            "passed": True,
            "input_provenance": {
                "split_policy_source_sha256": split_policy_sha,
                "split_summary_sha256": file_sha256(self.art / "audit/split_summary.json"),
                "split_validation_sha256": file_sha256(self.art / "audit/split_validation.json"),
                "cleaning_cache_sha256": cache_sha,
                "split_csv_sha256": {name: split_hashes[f"artifacts/splits/{name}.csv"] for name in ("train", "validation")},
            },
            "feature_source_sha256": feature_sha,
            "tokenizer": tokenizer_fingerprint(),
            "versions": {key: value for key, value in runtime_versions().items() if key != "threadpoolctl"},
            "configuration": {key: default.get_params()[key] for key in config_keys},
            "dtype": "float64", "fit_scope": "train_only",
            "matrices": {
                "train": {"columns": train_matrix.shape[1], "nnz": train_matrix.nnz},
                "validation": {"columns": validation_matrix.shape[1], "nnz": validation_matrix.nnz},
            },
            "fitted_state": fitted_feature_state(pipeline.named_steps["tfidf"]),
            "private_output_sha256": {name: file_sha256(self.root / name) for name in private_paths},
        }
        self.write_json(self.art / "audit/feature_summary.json", feature_summary)
        feature_validation = {
            "passed": True, "checks": {"synthetic_upstream_check": True},
            "feature_source_sha256": feature_sha,
            "validator_source_sha256": feature_checker_sha,
            "matrix_checks": {
                name: {"synthetic_matrix_check": True} for name in ("train", "validation")
            },
        }
        self.write_json(self.art / "audit/feature_validation.json", feature_validation)

    def refresh_split_input_provenance(self) -> None:
        """Recompute upstream metadata hashes after a deliberate split CSV mutation."""
        frozen_path = self.art / "audit/split_summary.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        for name in ("train", "validation"):
            frozen["output_sha256"][f"artifacts/splits/{name}.csv"] = file_sha256(self.art / f"splits/{name}.csv")
        self.write_json(frozen_path, frozen)
        summary_path = self.art / "audit/feature_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["input_provenance"]["split_summary_sha256"] = file_sha256(frozen_path)
        summary["input_provenance"]["split_csv_sha256"] = {
            name: frozen["output_sha256"][f"artifacts/splits/{name}.csv"] for name in ("train", "validation")
        }
        summary["private_output_sha256"]["artifacts/processed/features/row_order.json"] = file_sha256(self.art / "processed/features/row_order.json")
        self.write_json(summary_path, summary)


class ModelInputLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="model-inputs-synthetic-")
        self.root = Path(self.temporary.name)
        self.fixture = ModelInputsFixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_loads_train_and_validation_with_frozen_row_alignment(self):
        result = load_model_inputs(self.root)

        self.assertTrue(sparse.isspmatrix_csr(result.train_matrix))
        self.assertTrue(sparse.isspmatrix_csr(result.validation_matrix))
        self.assertEqual(result.train_matrix.shape[0], len(result.train_rows))
        self.assertEqual(result.validation_matrix.shape[0], len(result.validation_rows))
        self.assertEqual([row["mail_id"] for row in result.train_rows], ["train-ham-1", "train-spam-1", "train-ham-2", "train-spam-2"])
        self.assertEqual([row["label"] for row in result.train_rows], [0, 1, 0, 1])
        self.assertEqual([row["mail_id"] for row in result.validation_rows], ["validation-ham", "validation-spam"])
        self.assertEqual(result.provenance["consumed_splits"], ["train", "validation"])
        self.assertEqual(set(result.provenance["private_input_sha256"]), {
            "artifacts/processed/features/train.npz",
            "artifacts/processed/features/validation.npz",
            "artifacts/processed/features/row_order.json",
            "artifacts/models/tfidf_pipeline.joblib",
        })

    def test_never_opens_test_or_cleaning_cache_inputs(self):
        forbidden = {"test.npz", "test.csv", "cleaned_mail.jsonl"}
        original_builtin_open = builtins.open
        original_io_open = io.open
        original_path_open = Path.open
        attempted = []

        def forbidden_name(file):
            # These APIs also accept file descriptors and some wrappers may be
            # handed an already-open file object. Preserve their normal handling.
            if isinstance(file, int) or hasattr(file, "read") or hasattr(file, "write"):
                return None
            try:
                return Path(os.fsdecode(os.fspath(file))).name
            except TypeError:
                return None

        def guarded(api_name, original):
            def open_checked(file, *args, **kwargs):
                name = forbidden_name(file)
                if name in forbidden:
                    attempted.append((api_name, name))
                    raise AssertionError(f"{api_name}: forbidden path read blocked: {name}")
                return original(file, *args, **kwargs)
            return open_checked

        guarded_builtin = guarded("builtins.open", original_builtin_open)
        guarded_io = guarded("io.open", original_io_open)
        guarded_path = guarded("Path.open", original_path_open)
        api_cases = (
            ("builtins.open", patch("builtins.open", guarded_builtin), lambda path: builtins.open(path, "rb")),
            ("io.open", patch("io.open", guarded_io), lambda path: io.open(path, "rb")),
            ("Path.open", patch.object(Path, "open", guarded_path), lambda path: path.open("rb")),
        )

        # Trip each patched entry point directly so a broken guard cannot pass
        # merely because today's loader happens not to use that particular API.
        for api_name, patcher, open_path in api_cases:
            with self.subTest(api=api_name):
                with patcher:
                    with self.assertRaisesRegex(AssertionError, f"{api_name}: forbidden path read blocked"):
                        open_path(self.fixture.test_matrix_path)
                self.assertEqual(attempted[-1], (api_name, "test.npz"))

        # Exercise pass-through of file descriptors before watching the loader.
        safe_path = self.root / "descriptor.bin"
        with original_builtin_open(safe_path, "wb") as stream:
            stream.write(b"fd")
        descriptor = original_builtin_open(safe_path, "rb")
        try:
            fd = descriptor.fileno()
            for api_name, opener in (("builtins.open", guarded_builtin), ("io.open", guarded_io)):
                with self.subTest(file_descriptor_api=api_name):
                    descriptor.seek(0)
                    with opener(fd, "rb", closefd=False) as stream:
                        self.assertEqual(stream.read(), b"fd")
            filelike = io.BytesIO(b"handle")
            for api_name, opener in (("builtins.open", guarded_builtin), ("io.open", guarded_io)):
                with self.subTest(filelike_api=api_name):
                    with self.assertRaises(TypeError):
                        opener(filelike)
        finally:
            descriptor.close()

        attempted.clear()
        with patch("builtins.open", guarded_builtin), patch("io.open", guarded_io), patch.object(Path, "open", guarded_path):
            loaded = load_model_inputs(self.root)
        self.assertEqual(attempted, [])
        self.assertEqual(loaded.provenance["consumed_splits"], ["train", "validation"])

    def test_rejects_row_order_change_even_when_new_order_file_hash_is_recorded(self):
        path = self.root / "artifacts/processed/features/row_order.json"
        order = json.loads(path.read_text(encoding="utf-8"))
        order["train"][0], order["train"][1] = order["train"][1], order["train"][0]
        path.write_text(json.dumps(order, ensure_ascii=False), encoding="utf-8")
        summary_path = self.root / "artifacts/audit/feature_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["private_output_sha256"]["artifacts/processed/features/row_order.json"] = file_sha256(path)
        ModelInputsFixture.write_json(summary_path, summary)

        with self.assertRaisesRegex(ValueError, "Feature rows do not match frozen ID/label order"):
            load_model_inputs(self.root)

    def test_rejects_modified_consumed_sparse_matrix(self):
        path = self.root / "artifacts/processed/features/train.npz"
        matrix = sparse.load_npz(path).tocsr()
        matrix.data[0] += 0.25
        sparse.save_npz(path, matrix)

        with self.assertRaisesRegex(ValueError, "Consumed feature artifacts changed"):
            load_model_inputs(self.root)

    def test_rejects_changed_feature_source(self):
        source = self.root / "src/spam_classifier/text_features.py"
        source.write_bytes(source.read_bytes() + b"\n# synthetic source drift\n")
        with self.assertRaisesRegex(ValueError, "Feature rules, validation or runtime changed"):
            load_model_inputs(self.root)

    def test_rejects_changed_vectorizer_configuration(self):
        summary_path = self.root / "artifacts/audit/feature_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["configuration"]["min_df"] = 9
        ModelInputsFixture.write_json(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "Fitted training feature pipeline changed"):
            load_model_inputs(self.root)

    def test_rejects_invalid_label_after_csv_provenance_is_refreshed(self):
        path = self.root / "artifacts/splits/train.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        rows[0]["label"] = "2"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        self.fixture.refresh_split_input_provenance()

        with self.assertRaisesRegex(ValueError, "Split IDs or labels are invalid"):
            load_model_inputs(self.root)


if __name__ == "__main__":
    unittest.main()
