"""Numerical, missing-grid, and frozen-preprocessing baseline checks."""

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import scipy
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "scipy"}:
        raise
    raise unittest.SkipTest("Phase 3 tests require the optional baseline dependencies") from exc

from weather_risk.baseline_models import (
    _garch_objective, fit_garch, fit_log_variance, garch_variances, predict_log_variance,
)


class GarchTests(unittest.TestCase):
    def test_hand_calculated_asymmetric_recursion_and_missing_step(self):
        params = dict(omega=.2, alpha=.1, beta=.6, gamma=.2, initial_variance=2.0)
        values = np.array([2.0, -2.0, np.nan, 1.0, np.nan])
        # Positive and negative shocks differ, then the unobserved step uses E[y²|past]=h.
        np.testing.assert_allclose(garch_variances(values, params), [2, 1.8, 2.48, 2.184, 1.6104])
        changed = values.copy()
        changed[-1] = 10000
        np.testing.assert_array_equal(garch_variances(values, params), garch_variances(changed, params))

    def test_missing_steps_cannot_be_dropped_or_zero_filled(self):
        params = dict(omega=.1, alpha=.2, beta=.5, gamma=0, initial_variance=1.)
        with_gap = garch_variances([3, np.nan, 2], params)
        self.assertAlmostEqual(with_gap[-1], .1 + .7 * 2.4)
        self.assertNotAlmostEqual(with_gap[-1], garch_variances([3, 2], params)[-1])
        self.assertNotAlmostEqual(with_gap[-1], garch_variances([3, 0, 2], params)[-1])

    def test_likelihood_omits_missing_values_and_gradient_matches_finite_difference(self):
        values = np.array([1., -2., np.nan, np.nan, .3, -.4])
        observed = np.isfinite(values)
        squares = np.where(observed, values * values, 0.)
        for asymmetric, coefficients in [(False, np.array([.1, .1, .7])),
                                          (True, np.array([.1, .1, .6, .2]))]:
            with self.subTest(asymmetric=asymmetric):
                args = (squares, values < 0, observed, asymmetric)
                loss, gradient = _garch_objective(coefficients, *args)
                params = dict(zip(["omega", "alpha", "beta", "gamma"], coefficients))
                params["initial_variance"] = 1.
                h = garch_variances(values, params)[observed]
                self.assertAlmostEqual(loss, .5 * np.mean(np.log(h) + values[observed] ** 2 / h))
                numerical = []
                for index in range(len(coefficients)):
                    delta = np.zeros(len(coefficients))
                    delta[index] = 1e-6
                    numerical.append((_garch_objective(coefficients + delta, *args)[0]
                                      - _garch_objective(coefficients - delta, *args)[0]) / 2e-6)
                np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)

    def test_fits_are_deterministic_stationary_and_restore_original_units(self):
        random = np.random.default_rng(1984)
        values = random.standard_normal(180) * np.tile([1., 2., 3.], 60)
        values[[20, 21, 50]] = np.nan
        for asymmetric in (False, True):
            with self.subTest(asymmetric=asymmetric):
                first = fit_garch(values, asymmetric=asymmetric)
                again = fit_garch(values, asymmetric=asymmetric)
                self.assertEqual(first, again)
                json.dumps(first, allow_nan=False)
                self.assertGreater(first["omega"], 0)
                self.assertLess(first["alpha"] + first["beta"] + first["gamma"] / 2, 1)
                self.assertTrue(np.isfinite(garch_variances(values, first)).all())
                self.assertEqual(first["diagnostics"]["observations"], 177)
                self.assertEqual(first["diagnostics"]["grid_rows"], 180)
                scaled = fit_garch(values * 10, asymmetric=asymmetric)
                np.testing.assert_allclose(garch_variances(values * 10, scaled),
                                           100 * garch_variances(values, first), rtol=1e-4)
                if not asymmetric:
                    self.assertEqual(first["gamma"], 0)

    def test_optimizer_failure_is_explicit_without_fallback(self):
        failure = SimpleNamespace(success=False, status=9, message="Iteration limit reached",
                                  nit=200, fun=1., x=np.array([.1, .1, .7]))
        with patch("weather_risk.baseline_models.minimize", return_value=failure) as optimizer:
            with self.assertRaisesRegex(RuntimeError, "every deterministic start.*Iteration limit"):
                fit_garch(np.arange(30, dtype=float))
        self.assertEqual(optimizer.call_count, 3)

    def test_invalid_training_and_parameters_fail(self):
        for values in (np.zeros(30), np.ones(19), np.array([1., np.inf]), np.ones((2, 20))):
            with self.subTest(values=values.shape), self.assertRaises(ValueError):
                fit_garch(values)
        with self.assertRaisesRegex(ValueError, "persistence"):
            garch_variances([1], dict(omega=.1, alpha=.1, beta=.9, initial_variance=1))


class LogVarianceTests(unittest.TestCase):
    def test_intercept_smearing_recovers_mean_square_and_is_unpenalized(self):
        y = np.array([0., 1., -2., 4., 7.])
        for penalty in (0., 1., 1e8):
            params = fit_log_variance(np.zeros((len(y), 1)), y, ridge=penalty)
            prediction = predict_log_variance(np.zeros((3, 1)), params)
            np.testing.assert_allclose(prediction, np.mean(y * y), rtol=1e-12)
            json.dumps(params, allow_nan=False)

    def test_learns_log_squared_relationship(self):
        X = np.linspace(-2, 2, 40).reshape(-1, 1)
        y = np.exp(.5 + .3 * X[:, 0])
        params = fit_log_variance(X, y, ridge=0)
        np.testing.assert_allclose(predict_log_variance(X, params), y * y, rtol=1e-5)

    def test_missing_values_and_all_missing_features_remain_positive(self):
        X = np.array([[1., np.nan], [3., np.nan], [np.nan, np.nan], [8., np.nan]])
        params = fit_log_variance(X, np.array([0., 2., -3., 1.]))
        self.assertEqual(params["medians"], [3., 0.])
        self.assertEqual(len(params["coefficients"]), 5)
        prediction = predict_log_variance(np.array([[np.nan, np.nan], [100., 7.]]), params)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())
        zero_params = fit_log_variance(X, np.zeros(4))
        self.assertTrue((predict_log_variance(X, zero_params) > 0).all())

    def test_prediction_does_not_refit_or_depend_on_future_rows(self):
        X = np.array([[1., 3.], [2., np.nan], [3., 8.], [4., 11.]])
        params = fit_log_variance(X, np.array([1., -2., 3., -4.]))
        original = copy.deepcopy(params)
        current = np.array([[5., np.nan]])
        single = predict_log_variance(current, params)
        batch = predict_log_variance(np.vstack((current, [[1e6, -1e6], [np.nan, np.nan]])), params)
        np.testing.assert_array_equal(single, batch[:1])
        self.assertEqual(params, original)
        self.assertEqual(params["medians"], [2.5, 8.])

    def test_shapes_nonfinite_targets_and_penalties_are_rejected(self):
        for X, y, ridge in [(np.ones(4), np.ones(4), 1), (np.ones((4, 1)), np.ones(3), 1),
                            (np.ones((4, 1)), np.array([1., 2., np.nan, 4.]), 1),
                            (np.ones((4, 1)), np.ones(4), -1)]:
            with self.assertRaises(ValueError):
                fit_log_variance(X, y, ridge=ridge)
        params = fit_log_variance(np.ones((4, 1)), np.ones(4))
        with self.assertRaisesRegex(ValueError, "feature count"):
            predict_log_variance(np.ones((2, 2)), params)


if __name__ == "__main__":
    unittest.main()
