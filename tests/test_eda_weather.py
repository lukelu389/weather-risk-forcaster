"""Weather cutoffs, past-only climatology and paired block resampling checks."""

import hashlib
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "pandas"}:
        raise
    raise unittest.SkipTest("Phase 2 tests require the optional EDA dependencies") from exc

from weather_risk.eda_weather import rank_correlation, relationships, weather_features


SPEC = {"series_id": "toronto_temp", "provider": "eccc", "timezone": "America/Toronto", "stale_days": 5}


def observation(day, value, available=None, **overrides):
    row = {"series_id": "toronto_temp", "observation_date": pd.Timestamp(day),
           "value": value, "unit": "degC", "source": "test", "quality_flag": "",
           "available_at": pd.Timestamp(available or day + "T12:00:00Z"),
           "retrieved_at": pd.Timestamp("2026-01-01T12:00:00Z"),
           "availability_basis": "assumed_lag", "vintage": "latest"}
    return {**row, **overrides}


def origins(*timestamps):
    return pd.DataFrame({"origin_at": pd.to_datetime(list(timestamps), utc=True)})


class WeatherFeaturesTests(unittest.TestCase):
    def test_availability_equal_to_cutoff_is_included_future_is_excluded(self):
        observations = pd.DataFrame([
            observation("2020-01-01", 1, "2020-01-03T17:00:00Z"),
            observation("2020-01-02", 2, "2020-01-03T18:00:00Z"),
            observation("2020-01-03", 999, "2020-01-03T18:00:01Z"),
        ])
        targets = origins("2020-01-03T17:59:59Z", "2020-01-03T18:00:00Z")
        features = weather_features(observations, targets, [SPEC], allow_latest_vintage=True)
        base = features.loc[features.feature.eq("toronto_temp")]
        self.assertEqual(base.value.tolist(), [1, 2])
        self.assertEqual(base.source_observation_date.tolist(), ["2020-01-01", "2020-01-02"])
        observations.loc[2, "value"] = -1_000_000
        assert_frame_equal(features, weather_features(observations, targets, [SPEC], allow_latest_vintage=True))

    def test_latest_vintage_is_unavailable_without_explicit_override(self):
        observations = pd.DataFrame([observation("2020-01-02", 2)])
        features = weather_features(observations, origins("2020-01-03T18:00:00Z"), [SPEC])
        self.assertTrue(features.value.isna().all())
        self.assertTrue(features.missing_reason.eq("strict_vintage").all())

    def test_source_missing_newest_value_does_not_reuse_older_weather(self):
        observations = pd.DataFrame([
            observation("2020-01-01", 1), observation("2020-01-02", np.nan),
        ])
        features = weather_features(observations, origins("2020-01-03T18:00:00Z"), [SPEC], allow_latest_vintage=True)
        self.assertTrue(features.value.isna().all())
        self.assertTrue(features.missing_reason.eq("source_missing").all())
        self.assertTrue(features.source_observation_date.eq("2020-01-02").all())

    def reference_history(self):
        rows = [observation(f"2019-01-{day:02d}", float(day)) for day in range(1, 21)]
        rows += [
            observation("2019-01-21", 888, "2020-01-12T18:00:01Z"),
            observation("2019-01-22", np.nan),
            observation("2019-12-15", 1000),
            observation("2020-01-09", 100),
            observation("2020-01-10", 30, "2020-01-12T18:00:00Z"),
            observation("2021-01-01", -1000),
        ]
        return pd.DataFrame(rows)

    def test_anomaly_uses_20_earlier_year_same_month_available_nonmissing_rows(self):
        observations = self.reference_history()
        features = weather_features(observations, origins("2020-01-12T18:00:00Z"), [SPEC], allow_latest_vintage=True)
        anomaly = features.loc[features.feature.eq("toronto_temp_anomaly")].iloc[0]
        self.assertEqual(anomaly.reference_n, 20)
        self.assertAlmostEqual(anomaly.reference_mean, 10.5)
        self.assertAlmostEqual(anomaly.value, 19.5)
        self.assertEqual(anomaly.missing_reason, "")
        # Same-year, wrong-month, unavailable, and future-year changes cannot alter the reference.
        excluded = observations.observation_date.isin(pd.to_datetime([
            "2019-01-21", "2019-12-15", "2020-01-09", "2021-01-01",
        ]))
        observations.loc[excluded, "value"] = 1_000_000
        assert_frame_equal(features, weather_features(observations, origins("2020-01-12T18:00:00Z"), [SPEC], allow_latest_vintage=True))

    def test_anomaly_requires_at_least_20_valid_reference_observations(self):
        observations = self.reference_history()
        observations.loc[0, "value"] = np.nan
        features = weather_features(observations, origins("2020-01-12T18:00:00Z"), [SPEC], allow_latest_vintage=True)
        anomaly = features.loc[features.feature.eq("toronto_temp_anomaly")].iloc[0]
        self.assertEqual(anomaly.reference_n, 19)
        self.assertTrue(pd.isna(anomaly.value))
        self.assertTrue(pd.isna(anomaly.reference_mean))
        self.assertEqual(anomaly.missing_reason, "insufficient_prior_year_reference")


class RelationshipTests(unittest.TestCase):
    def sample(self):
        timestamps = pd.date_range("2020-01-01", periods=12, tz="UTC")
        targets = pd.DataFrame({"asset": "gas", "date": timestamps.tz_localize(None),
                                "origin_at": timestamps, "target": [-4, 2, np.nan, 1, 5, -3, 2, -6, 4, 1, -5, 3]})
        features = pd.DataFrame({"origin_at": timestamps, "feature": "weather",
                                "value": [1, 4, 2, 3, np.nan, 5, 8, 4, 6, 2, 7, 9]})
        return targets, features

    def test_rank_correlation_uses_tied_average_ranks_and_pairwise_missingness(self):
        x = np.array([1, 1, 3, 4, np.nan])
        y = np.array([2, 3, 3, 1, 5])
        expected = np.corrcoef([1.5, 1.5, 3, 4], [2, 3.5, 3.5, 1])[0, 1]
        self.assertAlmostEqual(rank_correlation(x, y), expected)
        self.assertTrue(np.isnan(rank_correlation(np.ones(4), np.arange(4))))
        self.assertTrue(np.isnan(rank_correlation(np.array([1., np.nan, 2]), np.arange(3))))

    def test_paired_bootstrap_retains_original_grid_and_missing_rows(self):
        targets, features = self.sample()
        calls = []
        def capture(x, y):
            calls.append((np.array(x, copy=True), np.array(y, copy=True)))
            return rank_correlation(x, y)
        with patch("weather_risk.eda_weather.rank_correlation", side_effect=capture):
            result = relationships(targets, features, seed=1729, block_length=2,
                                   bootstrap_repetitions=40, minimum_pairs=3).iloc[0]
        self.assertEqual(result.n_grid_rows, 12)
        self.assertEqual(result.n_pairs, 10)
        self.assertEqual(len(calls), 42)  # point estimate, 40 resamples, signed correlation
        self.assertTrue(all(len(x) == len(y) == 12 for x, y in calls))
        x = features.value.to_numpy()
        y = np.abs(targets.target.to_numpy())
        np.testing.assert_equal(calls[0][0], x)
        np.testing.assert_equal(calls[0][1], y)
        stable = hashlib.sha256(b"1729:gas:weather").digest()
        rng = np.random.default_rng(int.from_bytes(stable[:8], "big"))
        starts = rng.integers(0, 11, size=6)
        indices = (starts[:, None] + np.arange(2)).ravel()
        np.testing.assert_equal(calls[1][0], x[indices])
        np.testing.assert_equal(calls[1][1], y[indices])
        self.assertGreaterEqual(result.valid_bootstrap_repetitions, 20)
        self.assertTrue(np.isfinite(result.ci_low))
        self.assertTrue(np.isfinite(result.ci_high))

    def test_results_repeat_and_do_not_depend_on_other_feature_order(self):
        targets, features = self.sample()
        kwargs = {"seed": 71, "block_length": 2, "bootstrap_repetitions": 40, "minimum_pairs": 3}
        first = relationships(targets, features, **kwargs)
        assert_frame_equal(first, relationships(targets, features, **kwargs))
        additional = features.assign(feature="another_feature", value=features.value * -1)
        enlarged = pd.concat([additional, features.iloc[::-1]], ignore_index=True)
        actual = relationships(targets.iloc[::-1], enlarged, **kwargs)
        actual = actual.loc[actual.feature.eq("weather")].reset_index(drop=True)
        assert_frame_equal(first, actual)

    def test_unavailable_feature_kept_with_zero_pairs_and_no_interval(self):
        targets, features = self.sample()
        features["value"] = np.nan
        result = relationships(targets, features, block_length=2, bootstrap_repetitions=20,
                               minimum_pairs=3).iloc[0]
        self.assertEqual(result.n_pairs, 0)
        self.assertEqual(result.n_grid_rows, len(targets))
        self.assertEqual(result.valid_bootstrap_repetitions, 0)
        self.assertTrue(np.isnan(result.rho))
        self.assertTrue(np.isnan(result.ci_low))
        self.assertTrue(np.isnan(result.ci_high))


if __name__ == "__main__":
    unittest.main()
