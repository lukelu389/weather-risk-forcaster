"""Independent coverage calculations, calibration scores, and grid bootstrap checks."""

import json
import math
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import scipy
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "scipy"}:
        raise
    raise unittest.SkipTest("Phase 5 tests require the optional baseline dependencies") from exc

from weather_risk.evaluation import (
    coverage_tests, es_identification_residual, fz0_score, holm_adjust,
    moving_block_bootstrap, stable_seed,
)


class CoverageTests(unittest.TestCase):
    def test_hand_calculated_likelihood_and_transitions(self):
        result = coverage_tests([-1, 2, -1, 2, 2, -1, -1], [0] * 7, .5)
        self.assertEqual(result["transitions"], {"n00": 1, "n01": 2, "n10": 2, "n11": 1})
        expected_uc = 2 * (3 * math.log(3 / 7) + 4 * math.log(4 / 7) - 7 * math.log(.5))
        expected_ind = 2 * (4 * math.log(2 / 3) + 2 * math.log(1 / 3) - 6 * math.log(.5))
        self.assertAlmostEqual(result["unconditional"]["statistic"], expected_uc)
        self.assertAlmostEqual(result["independence"]["statistic"], expected_ind)
        self.assertAlmostEqual(result["conditional"]["statistic"], expected_uc + expected_ind)
        # The chi-square(2) survival function has this simple closed form.
        self.assertAlmostEqual(result["conditional"]["pvalue"], math.exp(-(expected_uc + expected_ind) / 2))
        self.assertEqual(result["exceedances"], 3)
        self.assertEqual(result["exact"]["pvalue"], 1.)
        json.dumps(result, allow_nan=False)

    def test_zero_and_all_exceedances_have_finite_uc_and_exact_boundary_intervals(self):
        for hit in (0, 1):
            with self.subTest(hit=hit):
                result = coverage_tests([hit] * 10, [.5] * 10, .95)
                probability = .05 if hit else .95
                self.assertAlmostEqual(result["unconditional"]["statistic"], -20 * math.log(probability))
                self.assertFalse(result["independence"]["available"])
                self.assertFalse(result["conditional"]["available"])
                self.assertIn("unidentified", result["independence"]["reason"])
                if hit:
                    self.assertAlmostEqual(result["exact"]["ci_low"], .025 ** .1)
                    self.assertEqual(result["exact"]["ci_high"], 1.)
                else:
                    self.assertEqual(result["exact"]["ci_low"], 0.)
                    self.assertAlmostEqual(result["exact"]["ci_high"], 1 - .025 ** .1)
                json.dumps(result, allow_nan=False)

    def test_exact_two_sided_binomial_uses_probability_ordering(self):
        # Binomial(4, .25): P(k) = [81,108,54,12,1]/256. For observed k=3,
        # only k=3 and k=4 are no more likely, so two-sided p=13/256.
        result = coverage_tests([1, 1, 1, 0], [.5] * 4, .75)
        self.assertAlmostEqual(result["exact"]["pvalue"], 13 / 256)

    def test_ties_are_non_exceedances_and_incomplete_pairs_are_excluded(self):
        result = coverage_tests([1, 2, 3, np.nan, 7], [1, 1, np.nan, 2, 7], .95)
        self.assertEqual(result["observations"], 3)
        self.assertEqual(result["exceedances"], 1)
        self.assertEqual(result["adjacent_pairs"], 1)
        self.assertEqual(result["transitions"], {"n00": 0, "n01": 1, "n10": 0, "n11": 0})

    def test_missing_holes_are_not_bridged_and_indices_match_uncompressed_sample(self):
        full = coverage_tests([0, 1, np.nan, 1, 0, 0, np.nan, 1], [.5] * 8, .75)
        subset = coverage_tests([0, 1, 1, 0, 0, 1], [.5] * 6, .75, grid_indices=[0, 1, 3, 4, 5, 7])
        self.assertEqual(full["transitions"], {"n00": 1, "n01": 1, "n10": 1, "n11": 0})
        self.assertEqual(full["adjacent_pairs"], 3)
        for key in ("transitions", "unconditional", "independence", "conditional", "exact"):
            self.assertEqual(full[key], subset[key])
        compressed = coverage_tests([0, 1, 1, 0, 0, 1], [.5] * 6, .75)
        self.assertNotEqual(full["transitions"], compressed["transitions"])

    def test_last_row_only_exceedance_does_not_identify_previous_hit_transition(self):
        result = coverage_tests([0] * 19 + [1], [.5] * 20, .95)
        self.assertEqual(result["exceedances"], 1)
        self.assertFalse(result["independence"]["available"])

    def test_no_valid_pairs_returns_explicit_unavailable_json(self):
        for losses, var in [([], []), ([np.nan], [1]), ([1], [np.nan])]:
            result = coverage_tests(losses, var, .95)
            self.assertEqual(result["observations"], 0)
            self.assertIsNone(result["exceedance_rate"])
            for key in ("unconditional", "independence", "conditional", "exact"):
                self.assertFalse(result[key]["available"])
                self.assertIsNone(result[key]["pvalue"])
            json.dumps(result, allow_nan=False)

    def test_invalid_arrays_and_grid_indices_raise(self):
        for losses, var in [([1], [1, 2]), ([[1]], [[1]]), ([np.inf], [1]), (["1"], [1])]:
            with self.subTest(losses=losses), self.assertRaises(ValueError):
                coverage_tests(losses, var, .95)
        for indices in [[0, 0], [2, 1], [-1, 0], [0., 1.], [0], [True, False]]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                coverage_tests([0, 1], [.5, .5], .95, grid_indices=indices)
        for confidence in [0, 1, np.nan, True]:
            with self.assertRaises(ValueError):
                coverage_tests([0], [1], confidence)


class JointScoreTests(unittest.TestCase):
    def test_hand_calculated_tail_non_tail_tie_and_missing(self):
        losses, var, es = [1, 2, 6, np.nan], [2] * 4, [4] * 4
        np.testing.assert_allclose(fz0_score(losses, var, es, .75),
                                   [math.log(4) - .5, math.log(4) - .5, math.log(4) + 3.5, np.nan])
        np.testing.assert_allclose(es_identification_residual(losses, var, es, .75),
                                   [-.5, -.5, 3.5, np.nan])

    def test_uniform_quadrature_calibrates_residual_and_minimizes_expected_score(self):
        losses = (np.arange(1000) + .5) / 1000
        var, es = np.full(1000, .5), np.full(1000, .75)
        residual = es_identification_residual(losses, var, es, .5)
        self.assertAlmostEqual(float(residual.mean()), 0., places=14)
        optimum = float(fz0_score(losses, var, es, .5).mean())
        self.assertAlmostEqual(optimum, math.log(.75), places=14)
        for alternative_var, alternative_es in [(.4, .75), (.6, .75), (.5, .65), (.5, .85)]:
            altered = fz0_score(losses, np.full(1000, alternative_var), np.full(1000, alternative_es), .5)
            self.assertGreater(float(altered.mean()), optimum)

    def test_residual_scale_invariance_and_score_log_scaling(self):
        losses, var, es = np.array([-1, 2, 8]), np.array([2, 2, 2]), np.array([4, 4, 4])
        np.testing.assert_allclose(es_identification_residual(losses * 3, var * 3, es * 3, .95),
                                   es_identification_residual(losses, var, es, .95))
        np.testing.assert_allclose(fz0_score(losses * 3, var * 3, es * 3, .95),
                                   fz0_score(losses, var, es, .95) + math.log(3))

    def test_nonpositive_es_rejected_and_partial_rows_remain_nan(self):
        for function in (fz0_score, es_identification_residual):
            for es in [0, -1, np.inf]:
                with self.subTest(function=function.__name__, es=es), self.assertRaises(ValueError):
                    function([1], [.5], [es], .95)
            result = function([np.nan, 1, 1, 1], [.5, np.nan, .5, .5], [1, 1, np.nan, 1], .95)
            self.assertTrue(np.isnan(result[:3]).all())
            self.assertTrue(np.isfinite(result[-1]))


class BootstrapTests(unittest.TestCase):
    def test_constant_zero_and_nonzero_series_have_known_intervals_and_pvalues(self):
        for constant, expected_p in [(0., 1.), (2., 1 / 1001)]:
            result = moving_block_bootstrap(np.full(120, constant))
            self.assertTrue(result["available"])
            self.assertEqual(result["valid_replications"], 1000)
            self.assertEqual((result["mean"], result["ci_low"], result["ci_high"]), (constant,) * 3)
            self.assertEqual(result["pvalue"], expected_p)
            json.dumps(result, allow_nan=False)

    def test_reproducible_independent_streams_without_mutating_input_or_global_rng(self):
        values = np.sin(np.arange(180) / 11)
        values[40:65] = np.nan
        original = values.copy()
        np.random.seed(314)
        expected_random = np.random.random()
        np.random.seed(314)
        first = moving_block_bootstrap(values, seed=27, key="asset/model")
        self.assertEqual(first, moving_block_bootstrap(values, seed=27, key="asset/model"))
        self.assertNotEqual(first, moving_block_bootstrap(values, seed=27, key="asset/other"))
        self.assertEqual(np.random.random(), expected_random)
        np.testing.assert_array_equal(values, original)

    def test_sampler_keeps_nan_holes_uses_overlapping_non_circular_blocks_and_truncates(self):
        # Two three-row blocks [0,NaN,4] and [4,6,8] produce five rows
        # [0,NaN,4,4,6], with mean 3.5. Compressing first or circular starts
        # would change both the admissible starts and the result.
        values = np.array([0., np.nan, 4., 6., 8.])
        with patch("weather_risk.evaluation.np.random.default_rng") as factory:
            factory.return_value.integers.return_value = np.tile([0, 2], (20, 1))
            result = moving_block_bootstrap(values, block_length=3, replications=20, min_observations=4)
        factory.return_value.integers.assert_called_once_with(0, 3, size=(20, 2))
        self.assertEqual(result["mean"], 4.5)
        self.assertEqual(result["ci_low"], 3.5)
        self.assertEqual(result["ci_high"], 3.5)
        self.assertEqual(result["pvalue"], 1 / 21)

    def test_insufficient_observations_and_short_grid_are_explicit(self):
        for values in [[], np.full(200, np.nan), np.arange(99.)]:
            result = moving_block_bootstrap(values)
            self.assertFalse(result["available"])
            self.assertIn("fewer than 100", result["reason"])
            self.assertIsNone(result["pvalue"])
            json.dumps(result, allow_nan=False)
        short = moving_block_bootstrap([1, 2], min_observations=2, block_length=3)
        self.assertFalse(short["available"])
        self.assertIn("shorter than block", short["reason"])

    def test_requires_95_percent_valid_bootstrap_means(self):
        values = [1., np.nan, np.nan, np.nan]
        # At length one, starts of zero provide a finite row and starts of
        # one provide NaN. One invalid replication out of 20 is acceptable;
        # two are not. The valid means always equal one.
        for invalid_count, available in [(1, True), (2, False)]:
            starts = np.zeros((20, 4), dtype=int)
            starts[:invalid_count] = 1
            with patch("weather_risk.evaluation.np.random.default_rng") as factory:
                factory.return_value.integers.return_value = starts
                result = moving_block_bootstrap(values, block_length=1, replications=20, min_observations=1)
            self.assertEqual(result["available"], available)
            self.assertEqual(result["valid_replications"], 20 - invalid_count)
            if available:
                self.assertEqual(result["pvalue"], 1 / 20)
            else:
                self.assertIsNone(result["pvalue"])

    def test_invalid_parameters_raise(self):
        for keyword in ("block_length", "replications", "min_observations"):
            for invalid in (0, -1, True, 2.5):
                with self.subTest(keyword=keyword, invalid=invalid), self.assertRaises(ValueError):
                    moving_block_bootstrap([1, 2], **{keyword: invalid})
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                stable_seed(seed)
        with self.assertRaises(ValueError):
            stable_seed(1, None)
        with self.assertRaises(ValueError):
            moving_block_bootstrap([np.inf])


class HolmTests(unittest.TestCase):
    def test_ordering_monotonicity_and_family_size_with_unavailable_hypotheses(self):
        np.testing.assert_allclose(holm_adjust([.03, .01, .04]), [.06, .03, .06])
        self.assertEqual(holm_adjust([.01, None, .04, np.nan]), [.04, None, .12, None])
        np.testing.assert_allclose(holm_adjust([.2, .2, .9]), [.6, .6, .9])
        self.assertEqual(holm_adjust([0., 1.]), [0., 1.])
        self.assertEqual(holm_adjust([None, np.nan]), [None, None])
        self.assertEqual(holm_adjust([]), [])

    def test_invalid_pvalues_raise(self):
        for invalid in [-.01, 1.01, np.inf, True, "0.1"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                holm_adjust([invalid])


if __name__ == "__main__":
    unittest.main()
