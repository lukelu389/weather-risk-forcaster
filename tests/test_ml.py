"""Phase 4 sequence timing, matched ablations, ensemble isolation, and runner audit."""

from contextlib import contextmanager
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
    import torch
    from pandas.testing import assert_frame_equal
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "pandas", "scipy", "torch"}:
        raise
    raise unittest.SkipTest("Phase 4 tests require the optional ML dependencies") from exc

# Import fixtures by module alias so unittest does not rediscover their classes.
import test_baselines as baseline_fixtures
from weather_risk import baselines, ml
from weather_risk.sequence_models import fit_sequence_variance, predict_sequence_variance


ROOT = ml.ROOT
COLUMNS = baselines.CONTROL_COLUMNS + ["weather"]


def config():
    settings = json.loads((ROOT / "config/ml.json").read_text())
    settings.update(baseline_fixtures.config())
    settings.update(sequence_length=3, epochs=2, gru_hidden_size=2, ssm_hidden_size=2)
    return settings


def fixed_fit(values, targets, *, seed, **kwargs):
    # Distinct variances test averaging variances rather than standard deviations.
    variance = 1.0 if seed == 1729 else 9.0
    return {"seed": seed, "variance": variance}, {"variance": torch.tensor(variance, dtype=torch.float64)}


@contextmanager
def mocked_fits():
    with patch.object(ml, "fit_sequence_variance", side_effect=fixed_fit) as fitting, \
         patch.object(ml, "predict_sequence_variance",
                      side_effect=lambda values, metadata, checkpoint: np.full(len(values), metadata["variance"])) as predicting:
        yield fitting, predicting


class SequenceInputsTests(unittest.TestCase):
    def test_padding_and_missing_labels_preserve_original_grid_steps(self):
        data = baseline_fixtures.panel().iloc[:5].copy()
        data["weather"] = np.arange(5, dtype=float)
        data.loc[2, "target"] = np.nan
        original = data.copy(deep=True)
        windows = ml.sequence_inputs(data, COLUMNS, 3)
        self.assertEqual(windows.shape, (5, 3, 4))
        self.assertTrue(np.isnan(windows[0, :2]).all())
        np.testing.assert_array_equal(windows[0, -1], data.loc[0, COLUMNS].to_numpy(dtype=float))
        # A missing outcome neither deletes its feature snapshot nor shifts time.
        np.testing.assert_array_equal(windows[3, :, -1], [1, 2, 3])
        assert_frame_equal(data, original)

    def test_snapshots_without_an_origin_or_from_later_origins_are_masked(self):
        data = baseline_fixtures.panel().iloc[:5].copy()
        data.loc[1, "origin_at"] = data.origin_at.iloc[4] + pd.Timedelta(days=1)
        data.loc[2, "origin_at"] = pd.NaT
        windows = ml.sequence_inputs(data, COLUMNS, 4)
        self.assertTrue(np.isnan(windows[2]).all())
        self.assertTrue(np.isnan(windows[3, 1:3]).all())
        np.testing.assert_array_equal(windows[3, 0], data.loc[0, COLUMNS].to_numpy(dtype=float))
        np.testing.assert_array_equal(windows[3, -1], data.loc[3, COLUMNS].to_numpy(dtype=float))

    def test_control_weather_values_and_missingness_are_removed_before_fitting(self):
        data = baseline_fixtures.panel().iloc[:12].copy()
        reference = ml.sequence_inputs(data, COLUMNS, 3, use_weather=False)
        self.assertTrue(np.isnan(reference[:, :, -1]).all())
        for architecture in ("gru", "ssm"):
            metadata, checkpoint = fit_sequence_variance(reference, data.target.to_numpy(),
                                                         architecture=architecture, hidden_size=2, epochs=2, seed=1729)
            expected = predict_sequence_variance(reference[-2:], metadata, checkpoint)
            for replacement in (np.full(len(data), 1e10), np.full(len(data), np.nan),
                                np.where(np.arange(len(data)) % 2, np.nan, -1e8)):
                with self.subTest(architecture=architecture, replacement=replacement.tolist()):
                    changed = data.copy()
                    changed["weather"] = replacement
                    actual = ml.sequence_inputs(changed, COLUMNS, 3, use_weather=False)
                    np.testing.assert_array_equal(actual, reference)
                    other_metadata, other_checkpoint = fit_sequence_variance(
                        actual, changed.target.to_numpy(), architecture=architecture,
                        hidden_size=2, epochs=2, seed=1729)
                    self.assertEqual(other_metadata, metadata)
                    np.testing.assert_array_equal(
                        predict_sequence_variance(actual[-2:], other_metadata, other_checkpoint), expected)

    def test_rebuilding_design_after_current_future_and_unreleased_mutations_is_causal(self):
        data = baseline_fixtures.panel().iloc[:16].drop(columns=COLUMNS).copy()
        data["target"] = np.arange(1, len(data) + 1, dtype=float)
        data.loc[5, "label_available_at"] += pd.Timedelta(days=30)
        features = baseline_fixtures.features_for(data)
        design = baselines.regression_design(data, features, ["weather"], 2)
        expected = ml.sequence_inputs(design, COLUMNS, 3)
        for index, replacement in ((5, 1e12), (7, 1e12), (7, np.nan), (12, 1e12)):
            with self.subTest(index=index, replacement=replacement):
                changed = data.copy()
                changed.loc[index, "target"] = replacement
                rebuilt = baselines.regression_design(changed, features, ["weather"], 2)
                np.testing.assert_array_equal(ml.sequence_inputs(rebuilt, COLUMNS, 3)[:8], expected[:8])
        # This assertion also confirms that legitimately matured targets do matter.
        changed = data.copy()
        changed.loc[4, "target"] = 100
        rebuilt = baselines.regression_design(changed, features, ["weather"], 2)
        self.assertNotEqual(ml.sequence_inputs(rebuilt, COLUMNS, 3)[7, -1, 0], expected[7, -1, 0])

    def test_duplicate_reversed_and_empty_length_grids_fail(self):
        data = baseline_fixtures.panel().iloc[:5]
        for invalid, length in ((data, 0), (data.iloc[::-1], 3), (pd.concat([data, data.iloc[-1:]]), 3)):
            with self.subTest(length=length), self.assertRaises(ValueError):
                ml.sequence_inputs(invalid, COLUMNS, length)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)

    def test_monthly_cache_uses_all_fixed_seeds_and_identical_optimizer_budgets(self):
        settings = config()
        with mocked_fits() as (fitting, predicting):
            forecasts, fits = ml.forecast_sequences(baseline_fixtures.panel(), settings, self.output)
        months = {pd.Timestamp(origin).tz_convert("America/Toronto").strftime("%Y-%m")
                  for origin in forecasts.origin_at}
        self.assertEqual(len(forecasts), 48 * len(ml.MODELS) * 2)
        self.assertEqual(len(fits), len(months) * len(ml.MODELS))
        self.assertEqual(fitting.call_count, len(fits) * len(settings["seeds"]))
        self.assertEqual(predicting.call_count, len(forecasts) // 2 * len(settings["seeds"]))
        self.assertTrue(forecasts.status.eq("passed").all())
        np.testing.assert_array_equal(forecasts.variance, np.full(len(forecasts), 5.0))
        for fit in fits:
            self.assertEqual([member["metadata"]["seed"] for member in fit["members"]], settings["seeds"])
            self.assertLessEqual(pd.Timestamp(fit["max_label_available_at"]), pd.Timestamp(fit["fit_origin"]))
            self.assertEqual(fit["n_train"], 40)
            for member in fit["members"]:
                path = self.output / member["checkpoint"]
                self.assertEqual(ml.sha(path), member["checkpoint_sha256"])
        for call in fitting.call_args_list:
            self.assertEqual(call.kwargs["epochs"], settings["epochs"])
            self.assertEqual(call.kwargs["learning_rate"], settings["learning_rate"])
            self.assertEqual(call.kwargs["hidden_size"], settings[call.kwargs["architecture"] + "_hidden_size"])
        for (_, month), group in forecasts.groupby(["model", forecasts.origin_at.str[:7]]):
            self.assertEqual(group.fit_id.nunique(), 1, month)

    def test_matured_training_mask_keeps_missing_grid_context_and_excludes_current_label(self):
        data, settings = baseline_fixtures.panel(), config()
        data["weather"] = np.arange(len(data), dtype=float)
        data.loc[5, "target"] = np.nan
        data.loc[35, "label_available_at"] = pd.Timestamp("2022-04-01T00:00Z")
        # A malformed early release cannot admit the current target.
        data.loc[42, "label_available_at"] = data.origin_at.iloc[42]
        with mocked_fits() as (fitting, _):
            forecasts, fits = ml.forecast_sequences(data, settings, self.output)
        first_weather = fitting.call_args_list[2]
        X, y = first_weather.args
        expected_indices = [i for i in range(1, 41) if i not in (5, 35)]
        np.testing.assert_array_equal(X[:, -1, -1], expected_indices)
        np.testing.assert_array_equal(y, data.target.iloc[expected_indices].to_numpy())
        np.testing.assert_array_equal(X[expected_indices.index(6), :, -1], [4, 5, 6])
        self.assertEqual((fits[0]["grid_rows"], fits[0]["n_train"]), (40, 38))
        self.assertEqual(forecasts.n_available.iloc[0], 38)
        self.assertEqual(forecasts.unobserved_grid_rows.iloc[0], 1)

    def test_member_fit_failure_invalidates_ensemble_and_is_cached(self):
        settings = config()
        def failing_member(X, y, **kwargs):
            if kwargs["seed"] == settings["seeds"][1]:
                raise RuntimeError("second seed failed")
            return fixed_fit(X, y, **kwargs)
        with mocked_fits() as (_, predicting), patch.object(ml, "fit_sequence_variance", side_effect=failing_member) as fitting:
            forecasts, fits = ml.forecast_sequences(baseline_fixtures.panel(), settings, self.output)
        self.assertTrue(forecasts.status.eq("failed").all())
        self.assertTrue(forecasts.error.str.contains("second seed failed").all())
        self.assertTrue(forecasts[["var", "es", "variance"]].isna().all().all())
        self.assertEqual(predicting.call_count, 0)
        self.assertEqual(fitting.call_count, 2 * len(fits))
        self.assertTrue(all(fit["status"] == "failed" and len(fit["members"]) == 1 for fit in fits))

    def test_member_prediction_failure_is_not_dropped_from_ensemble(self):
        with mocked_fits(), patch.object(ml, "predict_sequence_variance", side_effect=lambda X, m, p:
                                       np.full(len(X), np.nan if m["seed"] == 2718 else 1.0)):
            forecasts, fits = ml.forecast_sequences(baseline_fixtures.panel(), config(), self.output)
        self.assertTrue(all(fit["status"] == "passed" for fit in fits))
        self.assertTrue(forecasts.status.eq("failed").all())
        self.assertTrue(forecasts.error.str.contains("Nonpositive/nonfinite").all())
        self.assertTrue(forecasts[["var", "es"]].isna().all().all())

    def test_missing_origins_and_insufficient_history_remain_explicit_rows(self):
        data = baseline_fixtures.panel()
        data.loc[:20, "target"] = np.nan
        data.loc[43, "origin_at"] = pd.NaT
        with mocked_fits():
            forecasts, _ = ml.forecast_sequences(data, config(), self.output)
        first = forecasts.loc[forecasts.date.eq("2022-01-01")]
        missing = forecasts.loc[forecasts.date.eq("2022-01-02")]
        self.assertEqual(len(first), len(ml.MODELS) * 2)
        self.assertTrue(first.status.eq("skipped").all())
        self.assertTrue(first.error.eq("insufficient_matured_labels").all())
        self.assertTrue(missing.status.eq("skipped").all())
        self.assertTrue(missing.error.eq("missing_origin").all())

    def test_nine_model_comparison_scores_only_dates_shared_by_every_model(self):
        with baseline_fixtures.mocked_fits():
            baseline, _ = baselines.forecast_baselines(baseline_fixtures.panel(), config())
        with mocked_fits():
            neural, _ = ml.forecast_sequences(baseline_fixtures.panel(), config(), self.output)
        combined = pd.concat([baseline, neural], ignore_index=True)
        dates = combined.date.unique()
        combined.loc[combined.date.eq(dates[0]) & combined.model.eq("gru_weather"), "status"] = "failed"
        combined.loc[combined.date.eq(dates[1]) & combined.model.eq("garch"), "target"] = np.nan
        scores = baselines.score_forecasts(combined, models=ml.ALL_MODELS)
        self.assertEqual(set(scores.model), set(ml.ALL_MODELS))
        self.assertEqual(len(scores), 18)
        self.assertTrue(scores.common_scored_rows.eq(len(dates) - 2).all())
        self.assertTrue(scores[["mean_quantile_loss", "mean_qlike"]].notna().all().all())
        self.assertTrue(scores.loc[scores.model.eq("gru_weather"), "failed_rows"].eq(1).all())


class ConfigTests(unittest.TestCase):
    def test_frozen_dates_and_risk_protocol_cannot_be_changed(self):
        settings = config()
        ml.validate_ml_config(settings)
        for changes in ({"analysis_start": "2019-01-01"}, {"analysis_end": "2021-12-30"},
                        {"tuning_start": "2022-01-02"}, {"tuning_end": "2023-12-30"},
                        {"test_start": "2024-01-02"}, {"test_end": "2026-12-31"},
                        {"refit": "daily"}, {"confidences": [.90, .99]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ml.validate_ml_config({**settings, **changes})

    def test_invalid_neural_budgets_and_seed_sets_fail(self):
        settings = config()
        for changes in ({"sequence_length": 0}, {"epochs": True}, {"gru_hidden_size": 1.5},
                        {"ssm_hidden_size": -1}, {"learning_rate": 0}, {"learning_rate": np.inf},
                        {"seeds": []}, {"seeds": [1729, 1729]}, {"seeds": [-1]}, {"seeds": [True]},
                        {"seeds": [1.5]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ml.validate_ml_config({**settings, **changes})


class PreservedBaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "baselines"
        self.folder.mkdir()
        self.settings = config()
        self.settings["baseline_run"] = "baselines"
        with baseline_fixtures.mocked_fits():
            self.forecasts, _ = baselines.forecast_baselines(baseline_fixtures.panel(), self.settings)
        self.write_preserved_baseline()
        root_patch = patch.object(ml, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write_preserved_baseline(self, *, baseline_config=None, forecasts=None, phase=3, status="passed"):
        baseline_config = baseline_config if baseline_config is not None else self.settings
        forecasts = forecasts if forecasts is not None else self.forecasts
        (self.folder / "baseline_config.json").write_text(json.dumps(baseline_config))
        forecasts.to_csv(self.folder / "forecasts.csv", index=False)
        hashes = {name: ml.sha(self.folder / name) for name in ("baseline_config.json", "forecasts.csv")}
        (self.folder / "manifest.json").write_text(json.dumps({
            "phase": phase, "status": status, "source": {"observations_sha256": "snapshot"}, "artifacts": hashes,
        }))
        self.settings["baseline_forecasts_sha256"] = hashes["forecasts.csv"]

    def test_intact_baseline_returns_forecasts_and_verifiable_provenance(self):
        forecasts, provenance = ml.load_baseline_forecasts(self.settings, "snapshot")
        assert_frame_equal(forecasts, pd.read_csv(self.folder / "forecasts.csv"))
        self.assertEqual(provenance["run_dir"], str(self.folder.resolve()))
        self.assertEqual(provenance["forecasts_sha256"], ml.sha(self.folder / "forecasts.csv"))
        self.assertEqual(provenance["manifest_sha256"], ml.sha(self.folder / "manifest.json"))

    def test_source_snapshot_and_passed_phase_are_checked_before_parsing_forecasts(self):
        with patch.object(ml.pd, "read_csv") as reading, \
             self.assertRaisesRegex(ValueError, "input snapshots differ"):
            ml.load_baseline_forecasts(self.settings, "different_snapshot")
        reading.assert_not_called()
        for changes in ({"phase": 2}, {"status": "completed_with_failures"}, {"status": "failed"}):
            self.write_preserved_baseline(**changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "passed Phase 3"):
                ml.load_baseline_forecasts(self.settings, "snapshot")

    def test_tampered_forecasts_or_config_fail_checksum_verification(self):
        for name in ("forecasts.csv", "baseline_config.json"):
            self.write_preserved_baseline()
            path = self.folder / name
            path.write_text(path.read_text() + "\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Baseline checksum mismatch"):
                ml.load_baseline_forecasts(self.settings, "snapshot")

    def test_baseline_forecasts_must_also_match_the_independent_config_pin(self):
        self.settings["baseline_forecasts_sha256"] = "different_forecasts"
        with self.assertRaisesRegex(ValueError, "Pinned baseline forecasts differ"):
            ml.load_baseline_forecasts(self.settings, "snapshot")

    def test_verified_baseline_cannot_use_an_unmatched_comparison_protocol(self):
        for changes in ({"train_window": 41}, {"weather_columns": ["different_weather"]},
                        {"assets": ["different_asset"]}, {"max_gap_days": 5},
                        {"tuning_end": "2024-12-31"}, {"confidences": [.90, .99]},
                        {"prices": [{"series_id": "gas", "transform": "difference"}]}):
            self.write_preserved_baseline(baseline_config={**self.settings, **changes})
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Unmatched baseline protocol"):
                ml.load_baseline_forecasts(self.settings, "snapshot")

    def test_even_a_verified_file_must_contain_all_five_baseline_models(self):
        for missing in ml.BASELINES:
            self.write_preserved_baseline(forecasts=self.forecasts.loc[self.forecasts.model.ne(missing)])
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "Incomplete baseline model set"):
                ml.load_baseline_forecasts(self.settings, "snapshot")


class RunnerTests(unittest.TestCase):
    write_snapshot = baseline_fixtures.RunnerTests.write_snapshot

    def setUp(self):
        baseline_fixtures.RunnerTests.setUp(self)
        # Use a baseline forecast with the same fixture targets, retaining the ML
        # runner's real source loader, target construction, design, and scoring.
        baseline_run = baseline_fixtures.RunnerTests.run_fixture(self)
        self.baseline_forecasts = pd.read_csv(baseline_run / "forecasts.csv")
        self.settings.update({k: v for k, v in config().items() if k not in self.settings})
        self.config_path.write_text(json.dumps(self.settings))
        shutil.copyfile(ROOT / "requirements-ml.lock.txt", self.root / "requirements-ml.lock.txt")
        root_patch = patch.object(ml, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def run_fixture(self, **kwargs):
        with mocked_fits(), patch.object(ml, "load_baseline_forecasts", return_value=(self.baseline_forecasts, {"fixture": True})), \
             patch.object(ml, "weather_features", side_effect=lambda obs, targets, specs, **kw: baseline_fixtures.features_for(targets)):
            return ml.run_ml(self.config_path, allow_latest_vintage=True, publish=False, **kwargs)

    def test_poisoned_final_holdout_stays_unread_and_forecast_artifacts_repeat(self):
        first = self.run_fixture()
        self.write_snapshot(poisoned=True)
        second = self.run_fixture()
        names = ["design_panel.csv", "forecasts.csv", "tuning_scores.csv", "model_fits.json", "ml_report.md"]
        names.extend(str(p.relative_to(first)) for p in sorted((first / "checkpoints").glob("*.pt")))
        for name in names:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
        manifest = json.loads((second / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "passed")
        self.assertFalse(manifest["final_test_numeric_values_read"])
        self.assertFalse(manifest["source"]["numeric_values_read_outside_modeling_interval"])
        self.assertEqual(manifest["source"]["excluded_rows"], 1)
        for name, expected in manifest["artifacts"].items():
            self.assertEqual(ml.sha(second / name), expected, name)
        self.assertFalse((self.root / "reports").exists())

    def test_acknowledgment_and_snapshot_pin_are_enforced_before_fitting(self):
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            ml.run_ml(self.config_path)
        self.settings["input_observations_sha256"] = "wrong"
        self.config_path.write_text(json.dumps(self.settings))
        with self.assertRaisesRegex(ValueError, "Pinned input snapshot differs"):
            self.run_fixture()

    def test_runner_failure_records_manifest_without_publication(self):
        with patch.object(ml, "make_targets", side_effect=ValueError("target failure")), \
             self.assertRaisesRegex(ValueError, "target failure"):
            self.run_fixture()
        output, = (self.root / "runs").glob("ml-*")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure"], "ValueError: target failure")
        self.assertFalse((self.root / "reports").exists())


    def test_missing_package_metadata_creates_no_run(self):
        with patch.object(ml.importlib.metadata, "version",
                          side_effect=ml.importlib.metadata.PackageNotFoundError("optional dependency")), \
             self.assertRaises(ml.importlib.metadata.PackageNotFoundError):
            self.run_fixture()
        self.assertEqual(list((self.root / "runs").glob("ml-*")), [])
        self.assertFalse((self.root / "reports").exists())

    def test_initial_artifact_write_failure_records_failed_run(self):
        write_json = ml.write_json

        def fail_initial_artifact(path, value):
            if Path(path).name == "ml_config.json":
                raise OSError("initial artifact write failed")
            return write_json(path, value)

        with patch.object(ml, "write_json", side_effect=fail_initial_artifact), \
             self.assertRaisesRegex(OSError, "initial artifact write failed"):
            self.run_fixture()
        output, = (self.root / "runs").glob("ml-*")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure"], "OSError: initial artifact write failed")
        self.assertEqual(manifest["artifacts"], {})
        self.assertFalse((self.root / "reports").exists())

    def test_missing_pinned_input_explains_public_reproduction_boundary(self):
        (self.source / "observations.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "make verify-results"):
            self.run_fixture()
        self.assertEqual(list((self.root / "runs").glob("ml-*")), [])


if __name__ == "__main__":
    unittest.main()
