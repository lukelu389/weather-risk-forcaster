"""Checks for development isolation, target definitions, and missing-session handling."""

import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "pandas"}:
        raise
    raise unittest.SkipTest("Phase 2 tests require the optional EDA dependencies") from exc

from weather_risk.eda_core import (
    acf_table, describe_targets, load_development_observations, make_targets,
)


def prices(asset, values, dates=None):
    if dates is None:
        dates = pd.date_range("2020-04-17", periods=len(values), freq="B")
    return pd.DataFrame({"series_id": asset, "observation_date": pd.to_datetime(dates), "value": values})


def spec(asset="gas", transform="log_pct", equity=False):
    return {"series_id": asset, "label": asset, "transform": transform,
            "unit": "USD/barrel change" if transform == "difference" else "log return (%)", "equity": equity}


class TargetTests(unittest.TestCase):
    def test_negative_wti_remains_valid_dollar_change(self):
        targets = make_targets(prices("wti", [18.31, -36.98, 8.91]), [spec("wti", "difference")])
        np.testing.assert_allclose(targets["target"].iloc[1:], [-55.29, 45.89])
        self.assertTrue(targets["invalid_reason"].iloc[1:].eq("").all())

    def test_log_requires_strictly_positive_both_endpoints(self):
        targets = make_targets(prices("gas", [10, 0, 10, -1, 10, 11]), [spec()])
        self.assertTrue(targets["target"].iloc[1:5].isna().all())
        self.assertTrue(targets["invalid_reason"].iloc[1:5].eq("nonpositive_endpoint").all())
        self.assertAlmostEqual(targets["target"].iloc[5], 100 * math.log(1.1))

    def test_explicit_missing_quote_cannot_be_bridged(self):
        targets = make_targets(prices("gas", [10, None, 20, 22]), [spec()])
        self.assertEqual(targets["invalid_reason"].tolist(), [
            "no_previous_quote", "missing_current_price", "missing_previous_price", "",
        ])
        self.assertAlmostEqual(targets["target"].iloc[3], 100 * math.log(1.1))

    def test_gap_boundary_weekends_and_holidays(self):
        dates = ["2020-12-24", "2020-12-28", "2021-01-02", "2021-01-04"]
        targets = make_targets(prices("gas", [10, 11, 12, 13], dates), [spec()])
        self.assertEqual(targets["gap_days"].iloc[1:].tolist(), [4, 5, 2])
        self.assertTrue(np.isfinite(targets["target"].iloc[1]))
        self.assertEqual(targets["invalid_reason"].iloc[2], "excessive_gap")
        self.assertTrue(np.isfinite(targets["target"].iloc[3]))
        # An observed Saturday is retained; this function does not invent a calendar.
        self.assertEqual(targets["date"].iloc[2].day_name(), "Saturday")

    def test_forecast_origin_precedes_both_price_dates_and_handles_dst(self):
        dates = ["2024-03-08", "2024-03-11", "2024-03-12"]
        targets = make_targets(prices("gas", [10, 11, 12], dates), [spec()])
        for row in targets.iloc[1:].itertuples():
            self.assertLess(row.origin_at, row.interval_start.tz_localize("America/Toronto"))
            self.assertLess(row.origin_at, row.date.tz_localize("America/Toronto"))
            self.assertEqual(row.origin_at.tz_convert("America/Toronto").hour, 18)
        self.assertEqual(targets["origin_at"].iloc[1], pd.Timestamp("2024-03-07T23:00Z"))
        self.assertEqual(targets["origin_at"].iloc[2], pd.Timestamp("2024-03-10T22:00Z"))

    def test_basket_averages_simple_returns_before_taking_log(self):
        obs = pd.concat([prices(f"e{i}", [100, end]) for i, end in enumerate([110, 90, 130, 110])])
        targets = make_targets(obs, [spec(f"e{i}", equity=True) for i in range(4)])
        basket = targets.loc[targets["asset"].eq("canadian_basket")]
        self.assertAlmostEqual(basket["target"].iloc[1], 100 * math.log(1.1))
        self.assertNotAlmostEqual(basket["target"].iloc[1], np.log([1.1, .9, 1.3, 1.1]).mean() * 100)

    def test_basket_never_redistributes_a_missing_constituent(self):
        obs = pd.concat([prices(f"e{i}", [100, None if i == 3 else 110, 120]) for i in range(4)])
        targets = make_targets(obs, [spec(f"e{i}", equity=True) for i in range(4)])
        basket = targets.loc[targets["asset"].eq("canadian_basket")]
        self.assertTrue(basket["target"].isna().all())
        self.assertTrue(basket["invalid_reason"].eq("invalid_constituent_return").all())

    def test_basket_missing_date_and_start_mismatch_are_rejected(self):
        obs = pd.concat([
            prices(f"e{i}", [100, 110, 120], ["2024-01-02", "2024-01-03", "2024-01-04"])
            for i in range(3)
        ] + [prices("e3", [100, 120], ["2024-01-02", "2024-01-04"])])
        targets = make_targets(obs, [spec(f"e{i}", equity=True) for i in range(4)])
        basket = targets.loc[targets["asset"].eq("canadian_basket")].reset_index(drop=True)
        self.assertEqual(basket["invalid_reason"].iloc[1], "missing_constituent")
        self.assertEqual(basket["invalid_reason"].iloc[2], "constituent_interval_mismatch")
        self.assertTrue(basket["target"].isna().all())

    def test_duplicate_price_dates_fail(self):
        obs = prices("gas", [10, 11], ["2024-01-02", "2024-01-02"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            make_targets(obs, [spec()])

    def test_future_mutations_do_not_change_prior_targets(self):
        obs = prices("gas", [10, 11, 12, 13, 14])
        first = make_targets(obs, [spec()])
        obs.loc[obs.index[-1], "value"] = 10000
        changed = make_targets(obs, [spec()])
        assert_frame_equal(first.iloc[:-1], changed.iloc[:-1])


class SummaryTests(unittest.TestCase):
    def test_lag_correlations_preserve_missing_positions(self):
        # Only three adjacent pairs survive, rather than joining 2 to 10.
        targets = pd.DataFrame({"asset": "gas", "date": pd.date_range("2020-01-01", periods=6),
                                "target": [1, 2, np.nan, 10, 3, 4], "unit": "log return (%)"})
        acf = acf_table(targets, max_lag=1)
        row = acf.loc[acf["transform"].eq("return")].iloc[0]
        self.assertEqual(row["n_pairs"], 3)
        self.assertAlmostEqual(row["acf"], np.corrcoef([2, 3, 4], [1, 10, 3])[0, 1])
        summary = describe_targets(targets).iloc[0]
        self.assertEqual(summary["n"], 5)
        self.assertAlmostEqual(summary["acf1"], row["acf"])
        self.assertAlmostEqual(summary["std"], np.std([1, 2, 10, 3, 4], ddof=1))
        self.assertAlmostEqual(summary["gaussian_3sigma_expected"], 5 * math.erfc(3 / math.sqrt(2)))

    def test_constant_and_empty_series_return_undefined_correlations(self):
        targets = pd.DataFrame({"asset": ["constant"] * 4 + ["empty"] * 4,
                                "date": list(pd.date_range("2020-01-01", periods=4)) * 2,
                                "target": [0] * 4 + [np.nan] * 4, "unit": "log return (%)"})
        self.assertTrue(acf_table(targets)["acf"].isna().all())
        summary = describe_targets(targets).set_index("asset")
        self.assertEqual(summary.loc["empty", "n"], 0)
        self.assertEqual(summary.loc["constant", "count_abs_z_gt3"], 0)


class SnapshotTests(unittest.TestCase):
    def write_snapshot(self, folder, future_value="100000", validation_passed=True):
        config = {"start": "2020-01-01", "end": "2025-12-31"}
        (folder / "config.json").write_text(json.dumps(config))
        (folder / "validation.json").write_text(json.dumps({"passed": validation_passed, "errors": []}))
        fields = ["series_id", "observation_date", "value", "unit", "source", "available_at",
                  "retrieved_at", "availability_basis", "vintage", "quality_flag"]
        with (folder / "observations.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for day, value in [("2020-01-02", "10"), ("2020-01-03", "11"), ("2025-01-02", future_value)]:
                writer.writerow(dict(zip(fields, ["gas", day, value, "USD/MMBtu", "source", day + "T23:00:00Z",
                                                 "2026-01-01T12:00:00Z", "assumed_lag", "latest", "E"])))
        hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                  for name in ("config.json", "validation.json", "observations.csv")}
        manifest = {"phase": 1, "status": "passed", "mode": "live", "artifacts": hashes,
                    "config_sha256": hashes["config.json"], "historical_point_in_time_certified": False}
        (folder / "manifest.json").write_text(json.dumps(manifest))

    def test_holdout_is_not_numerically_parsed_and_flags_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.write_snapshot(folder)
            obs, provenance = load_development_observations(folder, "2020-01-01", "2020-12-31")
            initial_targets = make_targets(obs, [spec()])
            self.write_snapshot(folder, future_value="NOT_A_NUMBER_IN_HELDOUT")
            changed, _ = load_development_observations(folder, "2020-01-01", "2020-12-31")
            assert_frame_equal(obs, changed)
            assert_frame_equal(describe_targets(initial_targets), describe_targets(make_targets(changed, [spec()])))
            self.assertEqual(obs["quality_flag"].tolist(), ["E", "E"])
            self.assertEqual(provenance["excluded_rows"], 1)
            self.assertFalse(provenance["numeric_values_read_outside_development"])
            self.assertEqual(str(obs["available_at"].dt.tz), "UTC")
            self.assertIsNone(obs["observation_date"].dt.tz)

    def test_tampered_data_config_validation_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("observations.csv", "config.json", "validation.json"):
                with self.subTest(name=name):
                    self.write_snapshot(folder)
                    with (folder / name).open("a") as stream:
                        stream.write(" ")
                    with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                        load_development_observations(folder, "2020-01-01", "2020-12-31")

    def test_failed_validation_rejected_even_if_manifest_says_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.write_snapshot(folder, validation_passed=False)
            with self.assertRaisesRegex(ValueError, "validation must have passed"):
                load_development_observations(folder, "2020-01-01", "2020-12-31")

    def test_cannot_claim_an_interval_beyond_configured_history(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.write_snapshot(folder)
            with self.assertRaisesRegex(ValueError, "outside"):
                load_development_observations(folder, "2008-01-01", "2020-12-31")


if __name__ == "__main__":
    unittest.main()
