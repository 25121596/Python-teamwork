"""Synthetic leakage, comparison, row-alignment and private-cache guards."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spam_classifier.cleaning_ablation import choose_representation, train_validation_only
from spam_classifier.experiment_io import read_records, verified_file
from spam_classifier.model_inputs import fitted_feature_state


def records_fixture():
    records = {}
    for split, count in (("train", 8), ("validation", 4), ("test", 4)):
        records[split] = []
        for i in range(count):
            label = i % 2
            text = "免费 优惠 会员 公司 联系" if label else "会议 学校 项目 通知 讨论"
            unique = " validationonlyprobe" if split == "validation" else " testonlyprobe" if split == "test" else ""
            records[split].append({"mail_id": f"{split}-{i}", "label_name": "spam" if label else "ham", "label": label,
                                   "split": split, "usable": True, "text_clean": text + unique,
                                   "text_unstripped_html": f'<div class="advertising">{text}{unique}</div>' if label else text + unique,
                                   "subject_clean": "", "parse_status": "ok", "has_html": bool(label), "encoding_warnings": []})
    return records


class AblationTests(unittest.TestCase):
    def test_fits_only_training_vocabulary_and_both_models_share_selected_representation(self):
        data = records_fixture()
        result, arms, models, predictions = train_validation_only({k: data[k] for k in ("train", "validation")})
        self.assertEqual(predictions.shape, (4, 5))
        self.assertEqual(result["train_rows"], 8)
        self.assertEqual(result["changed_text_rows"], {"train": 4, "validation": 2})
        for pipeline in arms.values():
            self.assertNotIn("validationonlyprobe", pipeline.named_steps["tfidf"].vocabulary_)
            self.assertNotIn("testonlyprobe", pipeline.named_steps["tfidf"].vocabulary_)
        self.assertNotIn("advertising", arms["text_clean"].named_steps["tfidf"].vocabulary_)
        self.assertIn("advertising", arms["text_unstripped_html"].named_steps["tfidf"].vocabulary_)
        state = fitted_feature_state(arms[result["representation"]].named_steps["tfidf"])
        for pipeline in models.values():
            self.assertEqual(fitted_feature_state(pipeline.named_steps["tfidf"]), state)

    def test_test_records_cannot_enter_ablation(self):
        with self.assertRaisesRegex(ValueError, "never test"):
            train_validation_only(records_fixture())

    def test_overlapping_ids_rejected(self):
        data = records_fixture()
        data["validation"][0]["mail_id"] = data["train"][0]["mail_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            train_validation_only({k: data[k] for k in ("train", "validation")})

    def test_fractional_labels_are_not_silently_cast_to_binary(self):
        data = records_fixture()
        data["train"][0]["label"] = .5
        with self.assertRaises(ValueError):
            train_validation_only({k: data[k] for k in ("train", "validation")})

    def test_exact_tie_prefers_clean_and_excludes_nonconvergence(self):
        rows = [{"representation": name, "converged": True, "metrics": {"macro_f1": .9, "ham_false_positive_rate": .03, "spam_recall": .91}}
                for name in ("text_unstripped_html", "text_clean")]
        self.assertEqual(choose_representation(rows), "text_clean")
        rows[1]["converged"] = False
        self.assertEqual(choose_representation(rows), "text_unstripped_html")
        rows[0]["converged"] = False
        with self.assertRaises(ValueError):
            choose_representation(rows)

    def test_selection_uses_validation_f1_then_fpr_then_recall(self):
        rows = [{"representation": name, "converged": True, "metrics": {"macro_f1": .9, "ham_false_positive_rate": .03, "spam_recall": .91}}
                for name in ("text_clean", "text_unstripped_html")]
        rows[1]["metrics"]["macro_f1"] = .91
        self.assertEqual(choose_representation(rows), "text_unstripped_html")
        rows[1]["metrics"]["macro_f1"] = .9
        rows[1]["metrics"]["ham_false_positive_rate"] = .02
        self.assertEqual(choose_representation(rows), "text_unstripped_html")

    def test_cache_read_aligns_by_id_and_does_not_return_unrequested_rows(self):
        data = records_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifacts/processed/cleaned_mail.jsonl"
            path.parent.mkdir(parents=True)
            payload = "".join(json.dumps(row) + "\n" for rows in reversed(list(data.values())) for row in reversed(rows)).encode()
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            requested = {k: data[k] for k in ("train", "validation")}
            self.assertEqual(read_records(root, requested, digest), requested)
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_records(root, requested, "0" * 64)
            tampered = copy.deepcopy(requested)
            tampered["train"][0]["label"] = 1
            with self.assertRaises(ValueError):
                read_records(root, tampered, digest)

    def test_artifact_path_and_checksum_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            (root / "artifacts/model").write_bytes(b"model")
            digest = hashlib.sha256(b"model").hexdigest()
            verified_file(root, "artifacts/model", digest, prefix="artifacts/")
            with self.assertRaises(ValueError):
                verified_file(root, "artifacts/../outside", digest, prefix="artifacts/")
            with self.assertRaises(ValueError):
                verified_file(root, "artifacts/model", "0" * 64, prefix="artifacts/")


if __name__ == "__main__":
    unittest.main()
