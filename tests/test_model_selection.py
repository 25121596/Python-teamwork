"""Synthetic checks for the frozen model candidates and selection metrics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spam_classifier.model_selection import (  # noqa: E402
    CANDIDATES,
    SELECTION_POLICY,
    make_estimator,
    select_candidate,
    validation_metrics,
)


class ValidationMetricsTests(unittest.TestCase):
    def test_confusion_matrix_and_metrics_match_independent_hand_calculation(self):
        # tn=2, fp=1, fn=1, tp=2; each class has support three.
        result = validation_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])

        self.assertEqual(result["n"], 6)
        self.assertEqual(result["ham_support"], 3)
        self.assertEqual(result["spam_support"], 3)
        self.assertEqual(result["confusion_matrix"], {"tn": 2, "fp": 1, "fn": 1, "tp": 2})
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertAlmostEqual(result["spam_precision"], 2 / 3)
        self.assertAlmostEqual(result["spam_recall"], 2 / 3)
        self.assertAlmostEqual(result["ham_f1"], 2 / 3)
        self.assertAlmostEqual(result["spam_f1"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertAlmostEqual(result["ham_false_positive_rate"], 1 / 3)
        # Output can be serialized without NumPy scalar extensions.
        encoded = json.dumps(result, allow_nan=False)
        self.assertIsInstance(encoded, str)

    def test_false_positive_and_false_negative_are_counted_in_correct_direction(self):
        result = validation_metrics([0, 1], [1, 0])
        self.assertEqual(result["confusion_matrix"], {"tn": 0, "fp": 1, "fn": 1, "tp": 0})
        self.assertEqual(result["ham_false_positive_rate"], 1.0)
        self.assertEqual(result["spam_recall"], 0.0)

    def test_missing_class_uses_zero_division_zero(self):
        ham_only = validation_metrics([0, 0], [0, 0])
        self.assertEqual(ham_only["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 0, "tp": 0})
        self.assertEqual(ham_only["spam_precision"], 0.0)
        self.assertEqual(ham_only["spam_recall"], 0.0)
        self.assertEqual(ham_only["spam_f1"], 0.0)
        self.assertEqual(ham_only["ham_f1"], 1.0)
        self.assertEqual(ham_only["macro_f1"], 0.5)

        spam_only = validation_metrics([1, 1], [1, 0])
        self.assertEqual(spam_only["ham_f1"], 0.0)
        self.assertEqual(spam_only["ham_false_positive_rate"], 0.0)
        self.assertEqual(spam_only["spam_precision"], 1.0)
        self.assertEqual(spam_only["spam_recall"], 0.5)

    def test_rejects_empty_mismatched_non_vector_and_non_binary_inputs(self):
        cases = (
            ([], []),
            ([0, 1], [0]),
            ([[0], [1]], [[0], [1]]),
            ([0, 2], [0, 1]),
            ([0.0, 0.5], [0, 1]),
            (["ham", "spam"], [0, 1]),
        )
        for truth, prediction in cases:
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaises(ValueError):
                    validation_metrics(truth, prediction)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def row(candidate_id, macro_f1, ham_fpr, spam_recall, *, family="multinomial_nb", converged=True):
        return {
            "candidate_id": candidate_id,
            "family": family,
            "converged": converged,
            "metrics": {
                "macro_f1": macro_f1,
                "ham_false_positive_rate": ham_fpr,
                "spam_recall": spam_recall,
            },
        }

    def test_ranking_uses_macro_f1_then_false_positive_rate_then_recall_then_id(self):
        rows = [
            self.row("zeta", 0.90, 0.10, 0.80),
            self.row("alpha", 0.90, 0.10, 0.80),
            self.row("lower_recall", 0.90, 0.10, 0.70),
            self.row("higher_fpr", 0.90, 0.11, 1.00),
            self.row("lower_f1", 0.899999999999, 0.00, 1.00),
        ]
        self.assertEqual(select_candidate(rows), "alpha")

    def test_excludes_majority_and_unconverged_rows(self):
        rows = [
            self.row("majority", 1.0, 0.0, 1.0, family="majority"),
            self.row("diverged", 0.99, 0.0, 1.0, converged=False),
            self.row("eligible", 0.70, 0.20, 0.75),
        ]
        self.assertEqual(select_candidate(rows), "eligible")

    def test_raises_if_no_candidate_is_eligible(self):
        rows = [
            self.row("majority", 1.0, 0.0, 1.0, family="majority"),
            self.row("diverged", 0.99, 0.0, 1.0, converged=False),
        ]
        with self.assertRaisesRegex(ValueError, "no converged non-majority"):
            select_candidate(rows)


class CandidateEstimatorTests(unittest.TestCase):
    def test_candidate_catalog_and_estimator_parameters_are_frozen(self):
        self.assertEqual(
            [candidate["candidate_id"] for candidate in CANDIDATES],
            ["majority", "nb_alpha_1", "lr_C_1", "nb_alpha_0.1", "nb_alpha_0.5", "lr_C_0.5", "lr_C_2"],
        )
        majority = make_estimator(CANDIDATES[0])
        self.assertIsInstance(majority, DummyClassifier)
        self.assertEqual(majority.strategy, "most_frequent")

        nb = make_estimator(CANDIDATES[1])
        self.assertIsInstance(nb, MultinomialNB)
        self.assertEqual(nb.alpha, 1.0)
        self.assertTrue(nb.force_alpha)
        self.assertTrue(nb.fit_prior)

        lr = make_estimator(CANDIDATES[2])
        self.assertIsInstance(lr, LogisticRegression)
        self.assertEqual(lr.C, 1.0)
        self.assertEqual(lr.solver, "liblinear")
        self.assertEqual(lr.l1_ratio, 0.0)
        self.assertEqual(lr.max_iter, 2000)
        self.assertEqual(lr.tol, 1e-4)
        self.assertEqual(lr.random_state, 42)
        self.assertTrue(lr.fit_intercept)
        self.assertIsNone(lr.class_weight)

    def test_sparse_models_fit_and_joblib_roundtrip(self):
        matrix = sparse.csr_matrix(
            np.array([[3, 0, 0], [2, 0, 0], [0, 0, 3], [0, 0, 2]], dtype=np.float64)
        )
        labels = np.array([0, 0, 1, 1])
        for spec in (CANDIDATES[1], CANDIDATES[2]):
            with self.subTest(candidate=spec["candidate_id"]):
                estimator = make_estimator(spec).fit(matrix, labels)
                expected = estimator.predict(matrix)
                with tempfile.TemporaryDirectory(prefix="model-selection-test-") as directory:
                    path = Path(directory) / "model.joblib"
                    joblib.dump(estimator, path)
                    restored = joblib.load(path)
                np.testing.assert_array_equal(restored.predict(matrix), expected)


if __name__ == "__main__":
    unittest.main()
