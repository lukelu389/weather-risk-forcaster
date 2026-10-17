"""Numerical risk conventions, finite-sample tails, and validation checks."""

import math
import unittest

from weather_risk.risk import empirical_var_es, gaussian_var_es, quantile_loss


class EmpiricalRiskTests(unittest.TestCase):
    def test_loss_sign_and_inverted_cdf_quantile(self):
        # Ordered losses are [-4, -1, 2, 8], and F(2) = 0.75 exactly.
        self.assertEqual(empirical_var_es([4, 1, -2, -8], .75), (2, 8))
        self.assertEqual(empirical_var_es([4, 1, -2, -8], .5), (-1, 5))

    def test_fractional_boundary_mass_is_included(self):
        # The upper 37.5% consists of one full weight of 8 and half a weight of 2.
        var, es = empirical_var_es([4, 1, -2, -8], .625)
        self.assertEqual(var, 2)
        self.assertEqual(es, 6)

    def test_ties_do_not_drop_probability_mass(self):
        # Including all losses >= VaR would give 3; excluding ties would give 5.
        var, es = empirical_var_es([0, -2, -2, -5], .625)
        self.assertEqual(var, 2)
        self.assertEqual(es, 4)

    def test_tail_smaller_than_one_observation_and_singleton(self):
        self.assertEqual(empirical_var_es([3, -7], .99), (7, 7))
        self.assertEqual(empirical_var_es([3], .95), (-3, -3))
        self.assertEqual(empirical_var_es([3, 3, 3], math.nextafter(1, 0)), (-3, -3))

    def test_negative_risk_is_not_clipped(self):
        self.assertEqual(empirical_var_es([1, 2, 3, 4], .625), (-2, -4 / 3))

    def test_scaling_translation_and_input_order(self):
        targets = [4, 1, -2, -8]
        original = targets.copy()
        baseline = empirical_var_es(targets, .625)
        self.assertEqual(targets, original)
        self.assertEqual(empirical_var_es(reversed(targets), .625), baseline)
        self.assertEqual(empirical_var_es([3 * x for x in targets], .625), tuple(3 * x for x in baseline))
        self.assertEqual(empirical_var_es([x + 10 for x in targets], .625), tuple(x - 10 for x in baseline))

    def test_risk_increases_with_confidence_and_es_is_at_least_var(self):
        targets = [2, -3, 1, -5, -3]
        previous_var, previous_es = -math.inf, -math.inf
        for confidence in [.1, .3, .5, .8, .95, .99]:
            var, es = empirical_var_es(targets, confidence)
            self.assertGreaterEqual(es, var - 1e-12)
            self.assertGreaterEqual(var, previous_var)
            self.assertGreaterEqual(es, previous_es - 1e-12)
            previous_var, previous_es = var, es

    def test_large_finite_losses_do_not_overflow_while_averaging(self):
        var, es = empirical_var_es([-1e308] * 4, .5)
        self.assertEqual((var, es), (1e308, 1e308))

    def test_invalid_samples_raise(self):
        for targets in [[], None, [True], [False], [math.nan], [math.inf], [-math.inf], ["2"], [None], [1 + 2j]]:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                empirical_var_es(targets, .95)


class GaussianRiskTests(unittest.TestCase):
    def test_known_standard_normal_values(self):
        for confidence, expected_var, expected_es in [
            (.95, 1.6448536269514722, 2.0627128075074253),
            (.99, 2.3263478740408408, 2.665214220345805),
        ]:
            with self.subTest(confidence=confidence):
                var, es = gaussian_var_es(0, 1, confidence)
                self.assertAlmostEqual(var, expected_var, places=12)
                self.assertAlmostEqual(es, expected_es, places=12)

    def test_mean_sign_scaling_translation_and_es_order(self):
        var, es = gaussian_var_es(2, 3, .95)
        standard_var, standard_es = gaussian_var_es(0, 1, .95)
        self.assertAlmostEqual(var, -2 + 3 * standard_var)
        self.assertAlmostEqual(es, -2 + 3 * standard_es)
        self.assertGreater(es, var)
        self.assertEqual(gaussian_var_es(5, 0, .99), (-5, -5))
        shifted = gaussian_var_es(12, 3, .95)
        self.assertAlmostEqual(shifted[0], var - 10)
        self.assertAlmostEqual(shifted[1], es - 10)

    def test_positive_sigma_can_still_have_negative_risk(self):
        var, es = gaussian_var_es(10, 1, .95)
        self.assertLess(var, 0)
        self.assertLess(es, 0)

    def test_invalid_distribution_parameters_raise(self):
        for invalid in [True, False, math.nan, math.inf, -math.inf, "1", None, 1j]:
            for index in [0, 1]:
                parameters = [0, 1, .95]
                parameters[index] = invalid
                with self.subTest(invalid=invalid, index=index), self.assertRaises(ValueError):
                    gaussian_var_es(*parameters)
        with self.assertRaises(ValueError):
            gaussian_var_es(0, -1, .95)


class QuantileLossTests(unittest.TestCase):
    def test_upper_loss_tail_has_asymmetric_penalty(self):
        self.assertAlmostEqual(quantile_loss(12, 10, .95), 1.9)
        self.assertAlmostEqual(quantile_loss(8, 10, .95), .1)
        self.assertEqual(quantile_loss(10, 10, .95), 0)

    def test_translation_and_positive_scaling(self):
        value = quantile_loss(-2, -1, .95)
        self.assertAlmostEqual(quantile_loss(8, 9, .95), value)
        self.assertAlmostEqual(quantile_loss(-6, -3, .95), 3 * value)

    def test_large_finite_endpoints_with_representable_penalty(self):
        self.assertEqual(quantile_loss(1e308, -1e308, .25), 5e307)
        self.assertEqual(quantile_loss(-1e308, 1e308, .75), 5e307)

    def test_empirical_var_minimizes_sample_pinball_loss(self):
        losses = [-4, -1, 2, 8]
        var, _ = empirical_var_es([-x for x in losses], .625)
        optimum = sum(quantile_loss(loss, var, .625) for loss in losses)
        for alternative in [-10, -4, 0, 1.9, 2.1, 4, 8, 10]:
            self.assertLess(optimum, sum(quantile_loss(loss, alternative, .625) for loss in losses))

    def test_invalid_loss_and_forecast_raise(self):
        for invalid in [True, False, math.nan, math.inf, -math.inf, "1", None, 1j]:
            for index in [0, 1]:
                parameters = [0, 1, .95]
                parameters[index] = invalid
                with self.subTest(invalid=invalid, index=index), self.assertRaises(ValueError):
                    quantile_loss(*parameters)


class ConfidenceTests(unittest.TestCase):
    def test_all_functions_reject_invalid_confidences(self):
        for confidence in [0, 1, -.1, 1.1, True, False, math.nan, math.inf, -math.inf, "0.95", None, 1j]:
            for function, arguments in [
                (empirical_var_es, [[-1, 2]]),
                (gaussian_var_es, [0, 1]),
                (quantile_loss, [0, 1]),
            ]:
                with self.subTest(function=function.__name__, confidence=confidence), self.assertRaises(ValueError):
                    function(*arguments, confidence)


if __name__ == "__main__":
    unittest.main()
