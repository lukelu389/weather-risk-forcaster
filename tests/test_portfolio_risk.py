"""Exact fixed-notional CAD risk, nonlinear tail conversion, and budget shifts."""

import json
import math
import unittest

try:
    import numpy as np
    from scipy.integrate import quad
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "scipy"}:
        raise
    raise unittest.SkipTest("Phase 6 tests require the optional baseline dependencies") from exc

from weather_risk.portfolio_risk import (
    empirical_cad_var_es, gaussian_cad_var_es, log_returns_to_cad_losses,
)
from weather_risk.risk import empirical_var_es


class LossConversionTests(unittest.TestCase):
    def test_exact_simple_returns_and_negative_losses(self):
        targets = 100 * np.log([1.1, .8, 1.])
        original = targets.copy()
        np.testing.assert_allclose(log_returns_to_cad_losses(targets, notional=1000), [-100, 200, 0])
        np.testing.assert_array_equal(targets, original)

    def test_small_return_keeps_precision_and_losses_do_not_exceed_notional(self):
        result = log_returns_to_cad_losses([1e-14, -1e6])
        self.assertAlmostEqual(result[0], -1e-10, delta=1e-25)
        self.assertEqual(result[1], 1_000_000.)

    def test_generator_and_empty_conversion(self):
        np.testing.assert_allclose(log_returns_to_cad_losses(x for x in [0, -100]),
                                   [0, 1_000_000 * (1 - math.exp(-1))])
        self.assertEqual(log_returns_to_cad_losses([]).shape, (0,))

    def test_tiny_notional_can_offset_large_positive_log_return(self):
        expected = 1e-300 - math.exp(math.log(1e-300) + 710)
        self.assertEqual(log_returns_to_cad_losses([71000], notional=1e-300)[0], expected)

    def test_nonfinite_nonnumeric_and_nested_targets_rejected(self):
        for values in (None, [np.nan], [np.inf], [-np.inf], [True], ["1"], [[1]], [1j]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                log_returns_to_cad_losses(values)
        with self.assertRaisesRegex(ValueError, "finite numeric range"):
            log_returns_to_cad_losses([1e9])


class GaussianCadRiskTests(unittest.TestCase):
    def test_var_matches_known_normal_quantiles(self):
        for confidence, z in [(.5, 0.), (.95, 1.6448536269514722), (.99, 2.3263478740408408)]:
            with self.subTest(confidence=confidence):
                result = gaussian_cad_var_es(.2, 3, confidence)
                expected = 1_000_000 * (1 - math.exp((.2 - 3 * z) / 100))
                self.assertAlmostEqual(result["gross_var_cad"], expected, delta=1e-8)
                self.assertGreater(result["gross_es_cad"], result["gross_var_cad"])
                json.dumps(result, allow_nan=False)

    def test_es_matches_direct_gaussian_tail_integration(self):
        # Integrate the actual nonlinear loss against a standard normal density,
        # independently of the closed-form shifted-CDF expression in production.
        for mean, sigma, confidence, z in [
            (.2, 3, .95, 1.6448536269514722),
            (-2., 30., .99, 2.3263478740408408),
            (8., 2., .5, 0.),
        ]:
            with self.subTest(mean=mean, sigma=sigma, confidence=confidence):
                def integrand(standard_normal):
                    loss = -1_000_000 * math.expm1((mean + sigma * standard_normal) / 100)
                    return loss * math.exp(-standard_normal ** 2 / 2) / math.sqrt(2 * math.pi)
                integral, error = quad(integrand, -np.inf, -z, epsabs=1e-6, epsrel=1e-11)
                expected = integral / (1 - confidence)
                actual = gaussian_cad_var_es(mean, sigma, confidence)["gross_es_cad"]
                self.assertLess(error / (1 - confidence), 1e-4)
                self.assertAlmostEqual(actual, expected, delta=1e-6)

    def test_zero_variance_is_deterministic_at_all_confidences(self):
        for mean in (-3., 0., 5.):
            expected = -1_000_000 * math.expm1(mean / 100)
            for confidence in (.5, .95, .99):
                result = gaussian_cad_var_es(mean, 0, confidence)
                self.assertEqual(result["gross_var_cad"], expected)
                self.assertEqual(result["gross_es_cad"], expected)

    def test_negative_risk_is_retained_and_notional_scales_both_measures(self):
        first = gaussian_cad_var_es(10, 1, .95, notional=1000)
        scaled = gaussian_cad_var_es(10, 1, .95, notional=3000)
        for key in ("gross_var_cad", "gross_es_cad"):
            self.assertLess(first[key], 0)
            self.assertAlmostEqual(scaled[key], first[key] * 3)

    def test_extreme_variance_avoids_overflow_times_zero(self):
        # exp(sigma²/20000)*Phi(-z-sigma/100) overflows times underflows
        # naively. The exact upper loss tail still approaches the notional.
        for sigma in (10000., 1e308):
            result = gaussian_cad_var_es(0, sigma, .99)
            self.assertTrue(math.isfinite(result["gross_es_cad"]))
            self.assertAlmostEqual(result["gross_es_cad"], 1_000_000.)
            self.assertGreaterEqual(result["gross_es_cad"], result["gross_var_cad"])

    def test_scaled_error_function_branch_agrees_with_direct_integration(self):
        sigma, confidence, z = 1200., .5, 0.
        def integrand(standard_normal):
            return (1 - math.exp(sigma * standard_normal / 100)) * math.exp(-standard_normal ** 2 / 2) / math.sqrt(2 * math.pi)
        expected = quad(integrand, -np.inf, -z, epsabs=1e-12)[0] / (1 - confidence)
        actual = gaussian_cad_var_es(0, sigma, confidence, notional=1)["gross_es_cad"]
        self.assertAlmostEqual(actual, expected, places=12)

    def test_invalid_distribution_parameters_raise(self):
        for invalid in (True, np.nan, np.inf, -np.inf, "1", None, 1j):
            for name in ("mean", "sigma"):
                parameters = dict(mean=0, sigma=1, confidence=.95)
                parameters[name] = invalid
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    gaussian_cad_var_es(**parameters)
        with self.assertRaises(ValueError):
            gaussian_cad_var_es(0, -1, .95)
        with self.assertRaisesRegex(ValueError, "finite numeric range"):
            gaussian_cad_var_es(1e9, 1, .95)


class EmpiricalCadRiskTests(unittest.TestCase):
    def test_fractional_tail_is_averaged_after_nonlinear_loss_conversion(self):
        losses = np.array([-400., -100., 200., 800.])
        targets = 100 * np.log1p(-losses / 1000)
        result = empirical_cad_var_es(targets, .625, notional=1000)
        self.assertAlmostEqual(result["gross_var_cad"], 200)
        self.assertAlmostEqual(result["gross_es_cad"], 600)
        _, log_percent_es = empirical_var_es(targets, .625)
        naive_es = -1000 * math.expm1(-log_percent_es / 100)
        self.assertAlmostEqual(naive_es, 1000 * (1 - .2 ** (2 / 3) * .8 ** (1 / 3)))
        self.assertGreater(naive_es, result["gross_es_cad"] + 50)

    def test_ties_fractional_mass_and_tiny_tail(self):
        targets = 100 * np.log1p(-np.array([0., 200., 200., 500.]) / 1000)
        fractional = empirical_cad_var_es(targets, .625, notional=1000)
        self.assertAlmostEqual(fractional["gross_var_cad"], 200)
        self.assertAlmostEqual(fractional["gross_es_cad"], 400)
        tiny = empirical_cad_var_es(targets, .99, notional=1000)
        self.assertAlmostEqual(tiny["gross_var_cad"], 500)
        self.assertAlmostEqual(tiny["gross_es_cad"], 500)

    def test_all_positive_returns_can_have_negative_var_and_es(self):
        result = empirical_cad_var_es([1., 2., 3., 4.], .625)
        self.assertLess(result["gross_var_cad"], 0)
        self.assertLess(result["gross_es_cad"], 0)
        self.assertGreaterEqual(result["gross_es_cad"], result["gross_var_cad"])

    def test_empty_empirical_sample_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            empirical_cad_var_es([], .95)


class BudgetTests(unittest.TestCase):
    def test_explicit_budget_shifts_var_and_es_for_both_distributions(self):
        for function, arguments in [(gaussian_cad_var_es, (0, 2, .95)),
                                    (empirical_cad_var_es, ([1, -2, 3, -4], .625))]:
            with self.subTest(function=function.__name__):
                gross = function(*arguments, notional=1_000_000)
                adjusted = function(*arguments, notional=1_000_000, cost_bps=25)
                self.assertEqual(adjusted["cost_budget_cad"], 2500)
                self.assertEqual(adjusted["cost_bps"], 25)
                self.assertEqual(adjusted["notional_cad"], 1_000_000)
                self.assertEqual(adjusted["gross_var_cad"], gross["gross_var_cad"])
                self.assertEqual(adjusted["gross_es_cad"], gross["gross_es_cad"])
                self.assertEqual(adjusted["var_cad"], gross["var_cad"] + 2500)
                self.assertEqual(adjusted["es_cad"], gross["es_cad"] + 2500)
                json.dumps(adjusted, allow_nan=False)

    def test_invalid_notional_cost_and_confidence_rejected(self):
        for function, arguments in [(gaussian_cad_var_es, (0, 2)),
                                    (empirical_cad_var_es, ([1, -2, 3, -4],))]:
            for keyword, invalids in [("notional", [0, -1, True, np.inf, "1000"]),
                                      ("cost_bps", [-1, True, np.nan, np.inf])]:
                for invalid in invalids:
                    with self.subTest(function=function.__name__, keyword=keyword, invalid=invalid), self.assertRaises(ValueError):
                        function(*arguments, .95, **{keyword: invalid})
            for confidence in (0, 1, np.nan, True):
                with self.assertRaises(ValueError):
                    function(*arguments, confidence)
        with self.assertRaisesRegex(ValueError, "cost budget"):
            gaussian_cad_var_es(0, 0, .95, notional=1e308, cost_bps=1e308)


if __name__ == "__main__":
    unittest.main()
