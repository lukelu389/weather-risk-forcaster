"""Leakage guards tested at release boundaries and under future-data changes."""

import unittest
from dataclasses import replace

from weather_risk.schema import Observation
from weather_risk.timing import build_feature_panel, walk_forward_splits


SPEC = {
    "series_id": "x", "unit": "USD", "frequency": "daily",
    "timezone": "America/Toronto", "stale_days": 5,
}


def row(day="2024-01-01", value=10.0, available="2024-01-02T17:00:00+00:00", **changes):
    result = Observation(
        series_id="x", observation_date=day, value=value, unit="USD", source="test",
        available_at=available, retrieved_at="2024-06-01T00:00:00+00:00",
        availability_basis="release_timestamp", vintage="point_in_time",
    )
    return replace(result, **changes)


class FeatureTimingTests(unittest.TestCase):
    def panel(self, records, times, specifications=None, **options):
        return build_feature_panel(records, times, specifications or [SPEC], **options)

    def test_exact_cutoff_included_one_second_before_excluded(self):
        panel = self.panel([row()], ["2024-01-02T16:59:59Z", "2024-01-02T17:00:00Z"])
        self.assertEqual(panel[0]["missing_reason"], "absent")
        self.assertEqual(panel[1]["value"], 10)
        self.assertEqual(panel[1]["age_days"], 1)

    def test_no_older_nonmissing_fallback(self):
        records = [row(), row("2024-01-02", None, "2024-01-03T17:00:00Z")]
        result = self.panel(records, ["2024-01-03T17:00:00Z"])[0]
        self.assertEqual(result["observation_date"], "2024-01-02")
        self.assertIsNone(result["value"])
        self.assertEqual(result["missing_reason"], "source_missing")

    def test_stale_age_is_from_observation_not_release(self):
        records = [row(available="2024-02-01T17:00:00Z")]
        result = self.panel(records, ["2024-02-01T17:00:00Z"])[0]
        self.assertEqual(result["age_days"], 31)
        self.assertEqual(result["missing_reason"], "stale")
        self.assertIsNone(result["value"])

    def test_stale_boundary_is_inclusive(self):
        result = self.panel([row()], ["2024-01-06T17:00:00Z", "2024-01-07T17:00:00Z"])
        self.assertEqual(result[0]["value"], 10)
        self.assertEqual(result[1]["missing_reason"], "stale")

    def test_timezone_and_daylight_saving_boundary(self):
        records = [row("2024-03-09", available="2024-03-10T07:00:00Z")]
        # Toronto moves from EST to EDT on March 10. Both cutoffs are March 10 locally.
        result = self.panel(records, ["2024-03-10T03:00:00-04:00", "2024-03-11T03:59:59Z"])
        self.assertEqual([entry["age_days"] for entry in result], [1, 1])
        self.assertEqual(result[0]["forecast_at"], "2024-03-10T07:00:00+00:00")

    def test_strict_provenance_and_explicit_research_override(self):
        timestamp = ["2024-01-02T17:00:00Z"]
        for changes in ({"vintage": "latest"}, {"availability_basis": "assumed_lag"}):
            with self.subTest(changes=changes):
                records = [row(**changes)]
                self.assertEqual(self.panel(records, timestamp)[0]["missing_reason"], "strict_vintage")
                self.assertEqual(self.panel(records, timestamp, allow_latest_vintage=True)[0]["value"], 10)

    def test_newest_untrusted_row_does_not_resurrect_old_value(self):
        records = [row(), row("2024-01-02", 20, "2024-01-03T17:00:00Z", vintage="latest")]
        result = self.panel(records, ["2024-01-03T17:00:00Z"])[0]
        self.assertEqual(result["missing_reason"], "strict_vintage")
        self.assertEqual(result["observation_date"], "2024-01-02")

    def test_synthetic_records_remain_explicitly_labeled(self):
        records = [row(vintage="synthetic", availability_basis="synthetic", quality_flag="demo")]
        result = self.panel(records, ["2024-01-02T17:00:00Z"])[0]
        self.assertEqual(result["value"], 10)
        self.assertEqual(result["vintage"], "synthetic")
        self.assertEqual(result["quality_flag"], "demo")

    def test_future_values_cannot_change_past_features(self):
        past = row()
        future = row("2024-01-02", 20, "2024-01-03T17:00:00Z")
        times = ["2024-01-02T17:00:00Z"]
        original = self.panel([past, future], times)
        changed = self.panel([replace(future, value=999999), past], times)
        self.assertEqual(original, changed)
        self.assertEqual(original, self.panel([past], times))

    def test_late_release_of_older_observation_does_not_displace_newer(self):
        records = [row(available="2024-01-04T17:00:00Z"), row("2024-01-02", 20, "2024-01-03T17:00:00Z")]
        result = self.panel(records, ["2024-01-04T17:00:00Z"])[0]
        self.assertEqual(result["value"], 20)

    def test_invalid_cutoffs_duplicates_and_future_observation_reject(self):
        with self.assertRaises(ValueError):
            self.panel([row()], ["2024-01-02T17:00:00"])
        with self.assertRaises(ValueError):
            self.panel([row(), row()], ["2024-01-02T17:00:00Z"])
        with self.assertRaises(ValueError):
            self.panel([row("2024-01-03")], ["2024-01-02T17:00:00Z"])


class WalkForwardTests(unittest.TestCase):
    def test_expanding_indices_gap_and_partial_final_test(self):
        result = walk_forward_splits(10, min_train=3, test_size=2, gap=1)
        self.assertEqual(result[0], {"train_indices": [0, 1, 2], "test_indices": [4, 5]})
        self.assertEqual(result[-1]["test_indices"], [8, 9])
        for split in result:
            self.assertEqual(max(split["train_indices"]) + 2, min(split["test_indices"]))
            self.assertFalse(set(split["train_indices"]) & set(split["test_indices"]))
        self.assertEqual(walk_forward_splits(8, 3, test_size=2)[-1]["test_indices"], [7])

    def test_rolling_window_has_no_future_train_indices(self):
        result = walk_forward_splits(12, min_train=4, test_size=2, train_window=4)
        self.assertEqual(result[-1]["train_indices"], [6, 7, 8, 9])
        for split in result:
            self.assertEqual(len(split["train_indices"]), 4)
            self.assertLess(max(split["train_indices"]), min(split["test_indices"]))

    def test_small_sample_yields_no_splits_and_bad_parameters_fail(self):
        self.assertEqual(walk_forward_splits(3, 3), [])
        for options in ({"min_train": 0}, {"test_size": 0}, {"gap": -1}, {"train_window": 2}, {"n": 3.5}):
            kwargs = {"n": 10, "min_train": 3, **options}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                walk_forward_splits(**kwargs)


if __name__ == "__main__":
    unittest.main()
