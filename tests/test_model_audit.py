"""Tests for independent validation arithmetic and prediction archive guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from check_models import (  # noqa: E402
    construction_passed,
    independent_metrics,
    reported_n_iter_matches,
    timing_metadata_valid,
    validate_prediction_arrays,
    warning_metadata_valid,
)


class IndependentMetricTests(unittest.TestCase):
    def test_counts_and_metrics_are_recomputed_from_predictions(self):
        # tn=2, fp=1, fn=1, tp=2, with three examples per class.
        observed = independent_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])
        self.assertEqual(observed["confusion_matrix"], {"tn": 2, "fp": 1, "fn": 1, "tp": 2})
        self.assertEqual((observed["n"], observed["ham_support"], observed["spam_support"]), (6, 3, 3))
        self.assertAlmostEqual(observed["accuracy"], 4 / 6)
        self.assertAlmostEqual(observed["spam_precision"], 2 / 3)
        self.assertAlmostEqual(observed["spam_recall"], 2 / 3)
        self.assertAlmostEqual(observed["ham_f1"], 2 / 3)
        self.assertAlmostEqual(observed["spam_f1"], 2 / 3)
        self.assertEqual(observed["ham_f1"], 2 * 2 / (2 * 2 + 1 + 1))
        self.assertEqual(observed["spam_f1"], 2 * 2 / (2 * 2 + 1 + 1))
        self.assertAlmostEqual(observed["macro_f1"], 2 / 3)
        self.assertAlmostEqual(observed["ham_false_positive_rate"], 1 / 3)

    def test_missing_class_uses_zero_division_zero(self):
        observed = independent_metrics([0, 0], [0, 0])
        self.assertEqual(observed["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 0, "tp": 0})
        self.assertEqual(observed["spam_precision"], 0.0)
        self.assertEqual(observed["spam_recall"], 0.0)
        self.assertEqual(observed["ham_f1"], 1.0)
        self.assertEqual(observed["macro_f1"], 0.5)

    def test_rejects_bad_vectors(self):
        for truth, prediction in (([], []), ([0, 1], [0]), ([0, 2], [0, 1]), ([[0]], [[0]])):
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaises(ValueError):
                    independent_metrics(truth, prediction)


class PredictionOrderTests(unittest.TestCase):
    def setUp(self):
        self.candidates = ["majority", "nb_alpha_1"]
        self.mail_ids = ["validation-a", "validation-b"]
        self.truth = [0, 1]
        self.matrix = [[0, 0], [1, 1]]

    def test_accepts_exact_frozen_prediction_order(self):
        matrix = validate_prediction_arrays(
            self.candidates, self.mail_ids, self.truth, self.matrix,
            self.candidates, self.mail_ids, self.truth,
        )
        self.assertEqual(matrix.tolist(), self.matrix)

    def test_rejects_candidate_order_tampering(self):
        with self.assertRaisesRegex(ValueError, "candidate order"):
            validate_prediction_arrays(
                list(reversed(self.candidates)), self.mail_ids, self.truth, self.matrix,
                self.candidates, self.mail_ids, self.truth,
            )

    def test_rejects_mail_order_tampering_even_if_labels_match(self):
        with self.assertRaisesRegex(ValueError, "mail ID order"):
            validate_prediction_arrays(
                self.candidates, list(reversed(self.mail_ids)), self.truth, self.matrix,
                self.candidates, self.mail_ids, self.truth,
            )


class SummaryGuardTests(unittest.TestCase):
    def test_rejects_wrong_reported_iterations(self):
        estimator = SimpleNamespace(n_iter_=np.asarray([17]))
        self.assertTrue(reported_n_iter_matches(estimator, {"n_iter": [17]}))
        self.assertFalse(reported_n_iter_matches(estimator, {"n_iter": [16]}))
        self.assertFalse(reported_n_iter_matches(estimator, {"n_iter": [17.0]}))

    def test_timings_must_be_finite_and_nonnegative(self):
        self.assertTrue(timing_metadata_valid({"fit_seconds": 0.0, "predict_seconds": 0.25}))
        for record in (
            {"fit_seconds": -0.1, "predict_seconds": 0.25},
            {"fit_seconds": float("nan"), "predict_seconds": 0.25},
            {"fit_seconds": 0.1},
        ):
            with self.subTest(record=record):
                self.assertFalse(timing_metadata_valid(record))

    def test_warning_categories_must_agree_with_convergence(self):
        self.assertTrue(warning_metadata_valid(
            {"warning_categories": [], "converged": True, "n_iter": [17]},
            "logistic_regression", 2000,
        ))
        self.assertFalse(warning_metadata_valid(
            {"warning_categories": ["ConvergenceWarning"], "converged": True, "n_iter": [17]},
            "logistic_regression", 2000,
        ))
        self.assertFalse(warning_metadata_valid(
            {"warning_categories": [], "converged": True, "n_iter": [2000]},
            "logistic_regression", 2000,
        ))

    def test_construction_summary_must_be_explicitly_passed(self):
        self.assertTrue(construction_passed({"passed": True}))
        self.assertFalse(construction_passed({"passed": False}))
        self.assertFalse(construction_passed({}))

if __name__ == "__main__":
    unittest.main()
