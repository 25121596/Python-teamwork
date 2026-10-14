"""Synthetic checks for deterministic tokenization and TF-IDF features."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spam_classifier.text_features import (
    STOPWORDS,
    make_feature_pipeline,
    tokenize_text,
    tokenizer_fingerprint,
)


class TokenizerTests(unittest.TestCase):
    def test_tokenizer_fingerprint_is_stable_and_does_not_expose_paths(self):
        import json

        first = tokenizer_fingerprint()
        second = tokenizer_fingerprint()

        self.assertEqual(first, second)
        self.assertRegex(first["dictionary_sha256"], r"\A[0-9a-f]{64}\Z")
        self.assertRegex(first["stopwords_sha256"], r"\A[0-9a-f]{64}\Z")
        self.assertRegex(first["public_terms_sha256"], r"\A[0-9a-f]{64}\Z")
        serialized = json.dumps(first, ensure_ascii=False).lower()
        self.assertNotIn("path", serialized)
        self.assertNotIn(str(REPOSITORY).lower(), serialized)

    def test_normalizes_unicode_and_english_and_keeps_special_tokens(self):
        tokens = tokenize_text("ＦＲＡＵＤ MixedCase 中文内容 urltoken emailtoken phonetoken")

        self.assertIn("fraud", tokens)
        self.assertIn("mixedcase", tokens)
        for placeholder in ("urltoken", "emailtoken", "phonetoken"):
            self.assertIn(placeholder, tokens)
        self.assertTrue(any("中文" in token or "内容" in token for token in tokens))

    def test_removes_stopwords_punctuation_and_invisible_characters(self):
        self.assertIsInstance(STOPWORDS, frozenset)
        candidates = [word for word in STOPWORDS if len(word) >= 2 and word.isalnum()]
        self.assertTrue(candidates, "the fixed stopword list should contain token-like entries")
        stopword = sorted(candidates)[0]
        tokens = tokenize_text(f"{stopword} 保留内容 !!! \u200b\ue000\x00")

        self.assertNotIn(stopword, tokens)
        self.assertFalse(any(token in STOPWORDS for token in tokens))
        self.assertNotIn("!!!", tokens)
        self.assertFalse(any("\ue000" in token or "\x00" in token or "\u200b" in token for token in tokens))
        self.assertTrue(tokens, "ordinary content should remain after filtering")

    def test_keeps_numbers_but_drops_single_english_letter_and_punctuation(self):
        tokens = tokenize_text("7 42 2026 x Q !? 中文")

        self.assertIn("7", tokens)
        self.assertIn("42", tokens)
        self.assertIn("2026", tokens)
        self.assertNotIn("x", tokens)
        self.assertNotIn("q", tokens)
        self.assertNotIn("!?", tokens)


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = make_feature_pipeline()
        self.assertIsInstance(self.pipeline, Pipeline)
        self.assertIn("tfidf", self.pipeline.named_steps)
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.assertIsInstance(self.vectorizer, TfidfVectorizer)

    def test_vectorizer_has_the_publicly_required_configuration(self):
        self.assertIs(self.vectorizer.tokenizer, tokenize_text)
        self.assertIsNone(self.vectorizer.token_pattern)
        self.assertFalse(self.vectorizer.lowercase)
        self.assertEqual(self.vectorizer.ngram_range, (1, 1))
        self.assertEqual(self.vectorizer.min_df, 2)
        self.assertEqual(self.vectorizer.max_features, 50000)
        self.assertTrue(self.vectorizer.sublinear_tf)
        self.assertEqual(self.vectorizer.norm, "l2")
        self.assertTrue(self.vectorizer.smooth_idf)
        self.assertEqual(self.vectorizer.dtype, np.float64)

    def test_fit_is_train_only_and_heldout_transform_does_not_change_idf(self):
        train = [
            "sharedanchor alphaone",
            "sharedanchor alphatwo",
            "sharedanchor alphathree",
        ]
        self.pipeline.fit(train)
        vocabulary = self.vectorizer.vocabulary_
        self.assertIn("sharedanchor", vocabulary)

        idf_before = self.vectorizer.idf_.copy()
        heldout = self.pipeline.transform(["heldoutzebraquux"])

        self.assertNotIn("heldoutzebraquux", self.vectorizer.vocabulary_)
        np.testing.assert_array_equal(self.vectorizer.idf_, idf_before)
        self.assertEqual(heldout.shape[1], len(vocabulary))
        self.assertEqual(heldout.nnz, 0)

    def test_min_df_filters_singletons_and_output_is_normalized_csr(self):
        documents = [
            "sharedanchor commonword rareonlyonce",
            "sharedanchor commonword secondcopy",
            "sharedanchor commonword thirdcopy",
            "!!!",
        ]
        matrix = self.pipeline.fit_transform(documents)

        self.assertIn("commonword", self.vectorizer.vocabulary_)
        self.assertNotIn("rareonlyonce", self.vectorizer.vocabulary_)
        self.assertLessEqual(len(self.vectorizer.vocabulary_), 50000)
        self.assertTrue(sparse.isspmatrix_csr(matrix))
        self.assertTrue(np.all(matrix.data >= 0))
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        np.testing.assert_allclose(norms[:3], np.ones(3), rtol=1e-6, atol=1e-7)
        self.assertEqual(norms[3], 0.0)

    def test_fitted_pipeline_joblib_roundtrip_preserves_features(self):
        documents = [
            "sharedanchor oneword",
            "sharedanchor twoword",
            "sharedanchor threeword",
        ]
        self.pipeline.fit(documents)
        probe = ["sharedanchor probeonly", "oneword"]
        expected = self.pipeline.transform(probe)

        with tempfile.TemporaryDirectory(prefix="text-features-test-") as directory:
            saved = Path(directory) / "pipeline.joblib"
            joblib.dump(self.pipeline, saved)
            restored = joblib.load(saved)
            actual = restored.transform(probe)

        self.assertTrue(sparse.isspmatrix_csr(actual))
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_allclose(actual.data, expected.data)


if __name__ == "__main__":
    unittest.main()
