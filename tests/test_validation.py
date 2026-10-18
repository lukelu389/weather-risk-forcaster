"""Validation fails on data corruption, but retains legitimate stress moves."""

import json
import unittest
from dataclasses import replace

from weather_risk.schema import Observation
from weather_risk.validation import validate_observations


def spec(**changes):
    result = {
        "series_id": "weather", "unit": "degC", "frequency": "daily",
        "timezone": "America/Toronto", "lag_days": 2, "available_hour": 12,
        "stale_days": 5, "minimum": -80, "maximum": 60, "required": True,
    }
    result.update(changes)
    return result


def observation(day="2024-01-01", value=1.0, **changes):
    row = Observation(
        series_id="weather", observation_date=day, value=value, unit="degC",
        source="test", available_at=day + "T18:00:00+00:00",
        retrieved_at="2024-03-01T18:00:00+00:00",
        availability_basis="release_timestamp", vintage="point_in_time",
    )
    return replace(row, **changes)


class ValidationTests(unittest.TestCase):
    def report(self, rows, specifications=None, start="2024-01-01", end="2024-01-01"):
        return validate_observations(rows, specifications or [spec()], start, end)

    def test_clean_data_and_json_serializable_report(self):
        result = self.report([observation()])
        self.assertTrue(result["passed"])
        self.assertEqual(result["series"]["weather"]["missing_fraction"], 0)
        json.dumps(result, allow_nan=False)

    def test_duplicate_and_unsorted_data_fail(self):
        self.assertFalse(self.report([observation(), observation()])["passed"])
        result = self.report([observation("2024-01-02"), observation()], end="2024-01-02")
        self.assertTrue(any("unsorted" in error for error in result["errors"]))

    def test_nonfinite_and_nonnumeric_values_fail(self):
        for value in (float("inf"), float("-inf"), float("nan"), True, "1"):
            with self.subTest(value=value):
                result = self.report([observation(value=value)])
                self.assertFalse(result["passed"])
                json.dumps(result, allow_nan=False)

    def test_explicit_null_preserved_and_exceeds_default_threshold(self):
        row = observation(value=None)
        result = self.report([row])
        self.assertFalse(result["passed"])
        self.assertEqual(result["series"]["weather"]["null_count"], 1)
        self.assertIsNone(row.value)

    def test_daily_calendar_gap_enforced(self):
        result = self.report([observation()], end="2024-01-02")
        self.assertFalse(result["passed"])
        stats = result["series"]["weather"]
        self.assertEqual(stats["missing_periods_sample"], ["2024-01-02"])
        self.assertEqual(stats["missing_fraction"], 0.5)

    def test_user_missing_threshold_and_optional_series(self):
        result = self.report([observation()], [spec(max_missing_fraction=0.5)], end="2024-01-02")
        self.assertTrue(result["passed"])
        self.assertTrue(self.report([], [spec(required=False)])["passed"])
        self.assertFalse(self.report([])["passed"])

    def test_business_calendar_gaps_warn_without_inventing_holidays(self):
        # Jan 1 is a holiday; this phase deliberately has no exchange calendar.
        result = self.report([observation("2024-01-02")], [spec(frequency="business_daily")], end="2024-01-02")
        self.assertTrue(result["passed"])
        self.assertTrue(any("market holidays" in warning for warning in result["warnings"]))

    def test_weekly_and_monthly_coverage_use_periods(self):
        for frequency, expected in (("weekly", 9), ("monthly", 2)):
            with self.subTest(frequency=frequency):
                result = self.report([observation()], [spec(frequency=frequency)], end="2024-02-29")
                self.assertTrue(result["passed"])
                self.assertEqual(result["series"]["weather"]["expected_period_count"], expected)
                self.assertEqual(result["series"]["weather"]["missing_period_count"], expected - 1)

    def test_sparse_business_series_fails_independent_coverage_ceiling(self):
        result = self.report([observation()],
                             [spec(frequency="business_daily", max_missing_fraction=1,
                                   max_coverage_missing_fraction=0.12)], end="2024-01-31")
        stats = result["series"]["weather"]
        self.assertFalse(result["passed"])
        self.assertEqual(stats["expected_period_count"], 23)
        self.assertEqual(stats["missing_period_count"], 22)
        self.assertEqual(stats["missing_fraction"], 0)
        self.assertTrue(any("coverage missing fraction" in error for error in result["errors"]))

    def test_coverage_ceiling_applies_to_weekly_and_monthly_periods(self):
        for frequency in ("weekly", "monthly"):
            with self.subTest(frequency=frequency):
                result = self.report([observation(), observation("2024-01-02")],
                                     [spec(frequency=frequency, max_coverage_missing_fraction=0.1)],
                                     end="2024-02-29")
                self.assertFalse(result["passed"])
                self.assertEqual(result["series"]["weather"]["observed_period_count"], 1)

    def test_optional_coverage_failure_warns_and_preserves_rows(self):
        rows = [observation()]
        result = self.report(rows, [spec(frequency="business_daily", required=False,
                                        max_coverage_missing_fraction=0.12)], end="2024-01-31")
        self.assertTrue(result["passed"])
        self.assertTrue(any("coverage missing fraction" in warning for warning in result["warnings"]))
        self.assertEqual(result["series"]["weather"]["row_count"], len(rows))

    def test_coverage_ceiling_allows_holiday_and_accepts_exact_boundary(self):
        # Nine of ten weekdays: one closure remains below the configured 12% ceiling.
        days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
        for ceiling in (0.12, 0.1):
            with self.subTest(ceiling=ceiling):
                result = self.report([observation(day) for day in days],
                                     [spec(frequency="business_daily", max_coverage_missing_fraction=ceiling)],
                                     end="2024-01-12")
                self.assertTrue(result["passed"])
                self.assertEqual(result["series"]["weather"]["coverage_missing_fraction"], 0.1)

    def test_invalid_coverage_ceiling_is_rejected(self):
        for value in (True, False, None, float("nan"), float("inf"), float("-inf"), -0.1, 1.1, "bad"):
            with self.subTest(value=value):
                result = self.report([observation()], [spec(max_coverage_missing_fraction=value)])
                self.assertFalse(result["passed"])
                self.assertTrue(any("invalid configuration" in error for error in result["errors"]))
                json.dumps(result, allow_nan=False)

    def test_units_and_physical_bounds_reject(self):
        for row in (observation(unit="degF"), observation(value=100), observation(value=-100)):
            with self.subTest(row=row):
                self.assertFalse(self.report([row])["passed"])

    def test_negative_price_allowed_by_config_and_spike_not_clipped(self):
        rows = [observation(value=20), observation("2024-01-02", value=-37.63)]
        result = self.report(rows, [spec(minimum=None, maximum=None, positive=False, jump_warning=30)], end="2024-01-02")
        self.assertTrue(result["passed"])
        self.assertEqual(result["series"]["weather"]["outlier_count"], 1)
        self.assertEqual(rows[1].value, -37.63)
        self.assertFalse(self.report([observation(value=0)], [spec(positive=True)])["passed"])

    def test_timestamp_awareness_and_local_midnight(self):
        for field in ("available_at", "retrieved_at"):
            for timestamp in ("2024-01-01T18:00:00", "2024-01-01T04:59:59+00:00"):
                with self.subTest(field=field, timestamp=timestamp):
                    self.assertFalse(self.report([observation(**{field: timestamp})])["passed"])
        self.assertTrue(self.report([observation(available_at="2024-01-01T05:00:00+00:00")])["passed"])

    def test_future_assumed_availability_after_retrieval_is_allowed(self):
        result = self.report([observation(available_at="2024-03-02T18:00:00+00:00", availability_basis="assumed_lag", vintage="latest")])
        self.assertTrue(result["passed"])

    def test_bad_dates_unknown_series_and_duplicate_specs(self):
        self.assertFalse(self.report([observation("2024-01-02")])["passed"])
        self.assertFalse(self.report([observation(observation_date="20240101")])["passed"])
        self.assertFalse(self.report([observation(series_id="unknown")])["passed"])
        self.assertFalse(self.report([observation()], [spec(), spec()])["passed"])
        self.assertFalse(self.report([], start="2024-02-01", end="2024-01-01")["passed"])


if __name__ == "__main__":
    unittest.main()
