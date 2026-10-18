"""Phase 3 availability, forecast isolation, scoring, and artifact lifecycle."""

from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
    import scipy
    from pandas.testing import assert_frame_equal
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "pandas", "scipy"}:
        raise
    raise unittest.SkipTest("Phase 3 tests require the optional baseline dependencies") from exc

from weather_risk import baselines
from weather_risk.baseline_models import garch_variances


ROOT = baselines.ROOT


def config():
    result = json.loads((ROOT / "config/baselines.json").read_text())
    result.update(assets=["gas"], train_window=40, minimum_train=30, recent_window=2,
                  weather_columns=["weather"], prices=[{
                      "series_id": "gas", "label": "Gas", "transform": "log_pct", "unit": "log return (%)",
                  }])
    return result


def panel():
    dates = pd.date_range("2021-11-20", periods=90)
    origins = (dates - pd.Timedelta(days=2) + pd.Timedelta(hours=23)).tz_localize("UTC")
    return pd.DataFrame({
        "asset": "gas", "label": "Gas", "date": dates,
        "interval_start": dates - pd.Timedelta(days=1), "origin_at": origins,
        "label_available_at": (dates + pd.Timedelta(hours=23)).tz_localize("UTC"),
        "target": np.sin(np.arange(len(dates)) / 3) + .2,
        "invalid_reason": "", "unit": "log return (%)", "weather": np.arange(len(dates)) % 7,
        "log_recent_second_moment": .1, "annual_sin": .2, "annual_cos": .3,
    })


def features_for(targets):
    origins = targets.origin_at.dropna().unique()
    return pd.DataFrame({"origin_at": origins, "feature": "weather", "value": 4.0})


def fixed_garch(values, asymmetric=False):
    return {"omega": .1, "alpha": .1, "beta": .75,
            "gamma": .1 if asymmetric else 0.0, "initial_variance": 1.0}


@contextmanager
def mocked_fits():
    with patch.object(baselines, "fit_garch", side_effect=fixed_garch) as garch, \
         patch.object(baselines, "fit_log_variance", return_value={"variance": 1.0}) as regression, \
         patch.object(baselines, "predict_log_variance", side_effect=lambda x, p: np.full(len(x), p["variance"])):
        yield garch, regression


class AvailabilityTests(unittest.TestCase):
    def test_maturity_requires_both_endpoints_and_all_eight_basket_prices(self):
        targets = panel().iloc[[2]].copy()
        targets["asset"] = "e0"
        basket = targets.copy()
        basket["asset"] = "canadian_basket"
        targets = pd.concat([targets, basket], ignore_index=True)
        start, end = targets.iloc[0][["interval_start", "date"]]
        specs = [{"series_id": f"e{i}", "equity": True} for i in range(4)]
        records = [{"series_id": f"e{i}", "observation_date": day,
                    "available_at": day.tz_localize("UTC") + pd.Timedelta(hours=23)}
                   for i in range(4) for day in (start, end)]
        observations = pd.DataFrame(records)
        # Delaying the initial endpoint must delay the movement too.
        delayed_start = end.tz_localize("UTC") + pd.Timedelta(days=3)
        observations.loc[0, "available_at"] = delayed_start
        result = baselines.target_availability(observations, targets, specs)
        self.assertEqual(result.label_available_at.tolist(), [delayed_start, delayed_start])
        for index in range(8):
            changed = observations.copy()
            delayed = delayed_start + pd.Timedelta(days=index + 1)
            changed.loc[index, "available_at"] = delayed
            with self.subTest(endpoint=index):
                self.assertEqual(baselines.target_availability(changed, targets, specs).label_available_at.iloc[1], delayed)
                missing = baselines.target_availability(changed.drop(index), targets, specs)
                self.assertTrue(pd.isna(missing.label_available_at.iloc[1]))
                if index < 2:
                    self.assertTrue(pd.isna(missing.label_available_at.iloc[0]))

    def test_regression_features_use_each_rows_original_origin(self):
        targets = panel().iloc[:12].drop(columns=baselines.CONTROL_COLUMNS + ["weather"])
        targets["target"] = np.arange(1, len(targets) + 1, dtype=float)
        targets.loc[5, "label_available_at"] += pd.Timedelta(days=20)
        baseline = baselines.regression_design(targets, features_for(targets), ["weather"], 2)
        # At row 7's origin, rows 3 and 4 are the latest two matured labels.
        self.assertAlmostEqual(baseline.log_recent_second_moment.iloc[7], np.log((4 ** 2 + 5 ** 2) / 2))
        self.assertTrue(baseline.log_recent_second_moment.iloc[:3].isna().all())
        changed = targets.copy()
        changed.loc[5, "target"] = 1e9  # Past date, but still unavailable.
        changed.loc[8:, "target"] = 1e12
        rebuilt = baselines.regression_design(changed, features_for(changed), ["weather"], 2)
        assert_frame_equal(baseline.loc[:7, baselines.CONTROL_COLUMNS + ["weather"]],
                           rebuilt.loc[:7, baselines.CONTROL_COLUMNS + ["weather"]])
        # Seasonal controls are tied to Toronto's origin date.
        day = targets.origin_at.iloc[7].tz_convert("America/Toronto").dayofyear
        self.assertAlmostEqual(baseline.annual_sin.iloc[7], np.sin(2 * np.pi * (day - 1) / 365.25))

    def test_unconfigured_weather_column_fails(self):
        targets = panel().drop(columns=baselines.CONTROL_COLUMNS + ["weather"])
        with self.assertRaisesRegex(ValueError, "Unconfigured weather"):
            baselines.regression_design(targets, features_for(targets), ["absent"], 2)


class ForecastTests(unittest.TestCase):
    def test_monthly_fits_are_reused_and_training_maturity_precedes_fit(self):
        data, settings = panel(), config()
        with mocked_fits() as (garch, regression):
            forecasts, fits = baselines.forecast_baselines(data, settings)
        months = {pd.Timestamp(origin).tz_convert("America/Toronto").strftime("%Y-%m")
                  for origin in forecasts.origin_at}
        self.assertEqual(len(fits), 4 * len(months))
        self.assertEqual(garch.call_count, 2 * len(months))
        self.assertEqual(regression.call_count, 2 * len(months))
        self.assertTrue(forecasts.status.eq("passed").all())
        self.assertEqual(len(forecasts), 48 * len(baselines.MODELS) * 2)
        for fit in fits:
            self.assertLessEqual(pd.Timestamp(fit["max_label_available_at"]), pd.Timestamp(fit["fit_origin"]))
            self.assertEqual(fit["n_train"], 40)
        self.assertTrue(forecasts.groupby(["model", forecasts.origin_at.str[:7]]).fit_id.nunique().le(1).all())

    def test_current_future_and_missing_target_mutations_cannot_change_prior_forecasts(self):
        data, settings = panel(), config()
        with mocked_fits():
            baseline, _ = baselines.forecast_baselines(data, settings)
        columns = [name for name in baseline.columns if name not in {"target", "target_invalid_reason"}]
        cutoff = data.date.iloc[50].date().isoformat()
        for replacement in [1e9, np.nan]:
            changed = data.copy()
            changed.loc[50:, "target"] = replacement
            changed.loc[50:, "invalid_reason"] = "missing_current_price" if pd.isna(replacement) else ""
            with mocked_fits():
                rebuilt, _ = baselines.forecast_baselines(changed, settings)
            with self.subTest(replacement=replacement):
                assert_frame_equal(baseline.loc[baseline.date.le(cutoff), columns].reset_index(drop=True),
                                   rebuilt.loc[rebuilt.date.le(cutoff), columns].reset_index(drop=True))

    def test_unavailable_and_missing_labels_preserve_original_grid(self):
        data, settings = panel(), config()
        data.loc[5, "target"] = np.nan
        data.loc[35, "label_available_at"] = pd.Timestamp("2022-04-01T00:00Z")
        with mocked_fits() as (fit, _), patch.object(baselines, "garch_variances", wraps=garch_variances) as filtering:
            forecasts, _ = baselines.forecast_baselines(data, settings)
        first_training = fit.call_args_list[0].args[0]
        self.assertEqual(len(first_training), 40)
        self.assertTrue(np.isnan(first_training[[4, 34]]).all())
        first_known = filtering.call_args_list[0].args[0]
        self.assertEqual(len(first_known), 42)
        self.assertTrue(np.isnan(first_known[[4, 34, -2, -1]]).all())
        self.assertEqual(forecasts.n_available.iloc[0], 38)
        self.assertEqual(forecasts.unobserved_grid_rows.iloc[0], 1)
        changed = data.copy()
        changed.loc[35, "target"] = 1e9
        with mocked_fits():
            rebuilt, _ = baselines.forecast_baselines(changed, settings)
        assert_frame_equal(forecasts, rebuilt)

    def test_cutoff_is_inclusive_and_current_target_never_enters_training(self):
        data, settings = panel(), config()
        first = data.loc[data.date.eq("2022-01-01")].iloc[0]
        data.loc[35, "label_available_at"] = first.origin_at
        # A malformed early release on the current label must not bypass the date guard.
        data.loc[data.date.eq(first.date), "label_available_at"] = first.origin_at
        with mocked_fits() as (fit, _):
            forecasts, _ = baselines.forecast_baselines(data, settings)
        self.assertEqual(forecasts.n_available.iloc[0], 40)
        self.assertIn(data.target.iloc[35], fit.call_args_list[0].args[0])
        self.assertNotIn(first.target, fit.call_args_list[0].args[0])
        data.loc[35, "label_available_at"] += pd.Timedelta(seconds=1)
        with mocked_fits() as (fit, _):
            late, _ = baselines.forecast_baselines(data, settings)
        self.assertEqual(late.n_available.iloc[0], 39)
        self.assertTrue(np.isnan(fit.call_args_list[0].args[0][34]))

    def test_fit_failures_and_insufficient_history_remain_in_output(self):
        data, settings = panel(), config()
        def failing_gjr(values, asymmetric=False):
            if asymmetric:
                raise RuntimeError("optimizer did not converge")
            return fixed_garch(values)
        with mocked_fits(), patch.object(baselines, "fit_garch", side_effect=failing_gjr):
            forecasts, fits = baselines.forecast_baselines(data, settings)
        failed = forecasts.loc[forecasts.model.eq("gjr_garch")]
        self.assertTrue(failed.status.eq("failed").all())
        self.assertTrue(failed.error.str.contains("optimizer did not converge").all())
        self.assertTrue(failed[["var", "es"]].isna().all().all())
        self.assertEqual(sum(fit["status"] == "failed" for fit in fits), failed.fit_id.nunique())
        data.loc[:20, "target"] = np.nan
        with mocked_fits():
            short, _ = baselines.forecast_baselines(data, settings)
        first = short.loc[short.date.eq("2022-01-01")]
        self.assertEqual(len(first), 10)
        self.assertTrue(first.status.eq("skipped").all())
        self.assertTrue(first.error.eq("insufficient_matured_labels").all())

    def test_prediction_failures_remain_in_output(self):
        with mocked_fits(), patch.object(baselines, "predict_log_variance", return_value=np.array([np.nan])):
            forecasts, _ = baselines.forecast_baselines(panel(), config())
        failed = forecasts.loc[forecasts.model.isin(["seasonal_variance", "weather_variance"])]
        self.assertTrue(failed.status.eq("failed").all())
        self.assertTrue(failed.error.str.contains("Nonpositive/nonfinite").all())

    def test_scores_use_common_dates_and_exclude_missing_realizations(self):
        with mocked_fits():
            forecasts, _ = baselines.forecast_baselines(panel(), config())
        days = forecasts.date.unique()
        forecasts.loc[forecasts.date.eq(days[0]) & forecasts.model.eq("garch"), "status"] = "failed"
        forecasts.loc[forecasts.date.eq(days[1]), "target"] = np.nan
        scores = baselines.score_forecasts(forecasts)
        self.assertTrue(scores.common_scored_rows.eq(len(days) - 2).all())
        self.assertTrue(scores.loc[scores.model.eq("garch"), "failed_rows"].eq(1).all())
        self.assertTrue(scores[["mean_quantile_loss", "mean_qlike"]].notna().all().all())


class ConfigTests(unittest.TestCase):
    def test_frozen_protocol_rejects_changes_even_without_overlap(self):
        settings = config()
        baselines.validate_baseline_config(settings)
        for changes in [{"analysis_start": "2019-01-01"}, {"analysis_end": "2021-12-30"},
                        {"tuning_start": "2022-01-02"}, {"tuning_end": "2023-12-30"},
                        {"test_start": "2024-01-02"}, {"test_end": "2026-12-31"}]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "frozen"):
                baselines.validate_baseline_config({**settings, **changes})

    def test_invalid_training_and_risk_settings_fail(self):
        settings = config()
        for changes in [{"train_window": 29}, {"minimum_train": 41}, {"recent_window": True},
                        {"refit": "daily"}, {"confidences": [.90, .99]}, {"ridge": 0},
                        {"assets": ["gas", "gas"]}, {"weather_columns": []}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                baselines.validate_baseline_config({**settings, **changes})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.settings = config()
        self.config_path = self.root / "baselines.json"
        shutil.copyfile(ROOT / "requirements-baselines.lock.txt", self.root / "requirements-baselines.lock.txt")
        self.write_snapshot()
        root_patch = patch.object(baselines, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write_snapshot(self, poisoned=False):
        (self.source / "config.json").write_text(json.dumps({"start": "2018-01-01", "end": "2025-12-31", "series": []}))
        (self.source / "validation.json").write_text(json.dumps({"passed": True, "errors": []}))
        columns = ["series_id", "observation_date", "value", "unit", "source", "available_at",
                   "retrieved_at", "availability_basis", "vintage", "quality_flag"]
        with (self.source / "observations.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index, day in enumerate(pd.date_range("2021-11-20", periods=90)):
                day = day.date().isoformat()
                writer.writerow(dict(zip(columns, ["gas", day, 50 + index + np.sin(index), "USD/MMBtu", "fixture",
                                                     day + "T23:00:00Z", "2026-01-01T00:00:00Z", "assumed_lag", "latest", ""])))
            writer.writerow(dict(zip(columns, ["gas", "2024-01-02", "UNREAD_HOLDOUT" if poisoned else 1e9,
                                                 "USD/MMBtu", "fixture", "UNREAD_TIMESTAMP" if poisoned else "2024-01-02T23:00:00Z",
                                                 "UNREAD_TIMESTAMP" if poisoned else "2026-01-01T00:00:00Z", "assumed_lag", "latest", ""])))
        hashes = {name: baselines.sha(self.source / name) for name in ["config.json", "observations.csv", "validation.json"]}
        (self.source / "manifest.json").write_text(json.dumps({
            "phase": 1, "status": "passed", "mode": "live", "config_sha256": hashes["config.json"],
            "artifacts": hashes, "historical_point_in_time_certified": False,
        }))
        self.settings.update(input_run="input", input_observations_sha256=hashes["observations.csv"])
        self.config_path.write_text(json.dumps(self.settings))

    def run_fixture(self, **kwargs):
        with mocked_fits(), patch.object(baselines, "weather_features", side_effect=lambda obs, targets, specs, **kw: features_for(targets)):
            return baselines.run_baselines(self.config_path, allow_latest_vintage=True, publish=False, **kwargs)

    def test_final_holdout_values_and_timestamps_are_unread_and_artifacts_repeat(self):
        first = self.run_fixture()
        self.write_snapshot(poisoned=True)
        second = self.run_fixture()
        for name in ["design_panel.csv", "forecasts.csv", "tuning_scores.csv", "model_fits.json", "baseline_report.md"]:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
        manifest = json.loads((second / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "passed")
        self.assertFalse(manifest["final_test_numeric_values_read"])
        self.assertFalse(manifest["source"]["numeric_values_read_outside_modeling_interval"])
        self.assertEqual(manifest["source"]["excluded_rows"], 1)
        for name, expected in manifest["artifacts"].items():
            self.assertEqual(baselines.sha(second / name), expected, name)
        self.assertFalse((self.root / "reports").exists())

    def test_latest_vintage_acknowledgment_and_pinned_snapshot_are_required(self):
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            baselines.run_baselines(self.config_path)
        self.settings["input_observations_sha256"] = "wrong"
        self.config_path.write_text(json.dumps(self.settings))
        with self.assertRaisesRegex(ValueError, "Pinned input differs"):
            self.run_fixture()
        self.assertFalse((self.root / "runs").exists())

    def test_failure_records_manifest_without_publishing(self):
        with patch.object(baselines, "make_targets", side_effect=ValueError("target failure")), \
             self.assertRaisesRegex(ValueError, "target failure"):
            self.run_fixture()
        output, = (self.root / "runs").iterdir()
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure"], "ValueError: target failure")
        self.assertFalse((self.root / "reports").exists())


    def test_missing_package_metadata_creates_no_run(self):
        with patch.object(baselines.importlib.metadata, "version",
                          side_effect=baselines.importlib.metadata.PackageNotFoundError("optional dependency")), \
             self.assertRaises(baselines.importlib.metadata.PackageNotFoundError):
            self.run_fixture()
        self.assertEqual(list((self.root / "runs").glob("baselines-*")), [])
        self.assertFalse((self.root / "reports").exists())

    def test_initial_artifact_write_failure_records_failed_run(self):
        write_json = baselines.write_json

        def fail_initial_artifact(path, value):
            if Path(path).name == "baseline_config.json":
                raise OSError("initial artifact write failed")
            return write_json(path, value)

        with patch.object(baselines, "write_json", side_effect=fail_initial_artifact), \
             self.assertRaisesRegex(OSError, "initial artifact write failed"):
            self.run_fixture()
        output, = (self.root / "runs").glob("baselines-*")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure"], "OSError: initial artifact write failed")
        self.assertEqual(manifest["artifacts"], {})
        self.assertFalse((self.root / "reports").exists())

    def test_missing_pinned_input_explains_public_reproduction_boundary(self):
        (self.source / "observations.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "make verify-results"):
            self.run_fixture()
        self.assertEqual(list((self.root / "runs").glob("baselines-*")), [])


if __name__ == "__main__":
    unittest.main()
