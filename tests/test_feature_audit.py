"""Regression checks for meaningful tokens and NumPy fitted-state metadata."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spam_classifier.text_features import make_feature_pipeline, tokenize_text

spec = importlib.util.spec_from_file_location("feature_audit_cli", ROOT / "scripts/build_features.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class FeatureAuditTests(unittest.TestCase):
    def test_single_chinese_negations_and_action_words_are_retained(self):
        tokens = tokenize_text("不 没 买 送 a b 3 的")
        for value in ("不", "没", "买", "送", "3"):
            self.assertIn(value, tokens)
        for value in ("a", "b", "的"):
            self.assertNotIn(value, tokens)

    def test_numpy_vocabulary_indices_have_stable_serializable_fingerprint(self):
        pipeline = make_feature_pipeline()
        pipeline.fit(["sharedword alphaword", "sharedword betaword"])
        vectorizer = pipeline.named_steps["tfidf"]
        vectorizer.vocabulary_ = {term: np.int64(index) for term, index in vectorizer.vocabulary_.items()}
        first = cli.feature_state(vectorizer)
        vectorizer.vocabulary_ = dict(reversed(list(vectorizer.vocabulary_.items())))
        self.assertEqual(first, cli.feature_state(vectorizer))
        self.assertEqual(len(first["vocabulary_sha256"]), 64)

    def test_sparse_summary_accounts_for_zero_rows_without_densifying(self):
        matrix = sparse.csr_matrix(([1.0, 2.0], ([0, 2], [1, 4])), shape=(3, 100000))
        result = cli.matrix_summary(matrix)
        self.assertEqual(result["zero_rows"], 1)
        self.assertEqual(result["nnz"], 2)
        self.assertLess(result["csr_bytes"], 100)


if __name__ == "__main__":
    unittest.main()
