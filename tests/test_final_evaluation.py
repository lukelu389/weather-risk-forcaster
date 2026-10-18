"""Synthetic checks for the final-period access gate, scoring and artifacts."""

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
    import torch
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "pandas", "scipy", "torch"}:
        raise
    raise unittest.SkipTest("Phase 5 tests require the optional ML dependencies") from exc

import test_baselines as baseline_fixtures
import test_ml as ml_fixtures
from weather_risk import baselines, final_evaluation as final, ml
from weather_risk.risk import gaussian_var_es


ROOT = final.ROOT


def evaluation_config():
    return json.loads((ROOT / "config/evaluation.json").read_text())


def forecast_fixture(length=130):
    """All nine models, three synthetic assets and both fixed risk levels."""
    records = []
    for asset in ("henry_hub_spot", "wti_spot", "canadian_basket"):
        for i, date in enumerate(pd.date_range("2024-01-01", periods=length)):
            for model_index, model in enumerate(final.ALL_MODELS):
                variance = 1 + model_index / 10
                for confidence in (.95, .99):
                    var, es = gaussian_var_es(0, np.sqrt(variance), confidence)
                    records.append({
                        "asset": asset, "date": date.date().isoformat(), "model": model,
                        "confidence": confidence, "status": "passed",
                        "interval_start": (date - pd.Timedelta(days=1)).date().isoformat(),
                        "target": 3 * np.sin(i / 7), "var": var, "es": es,
                        "second_moment": variance,
                    })
    return pd.DataFrame(records)


class ScoringTests(unittest.TestCase):
    def test_exact_planned_family_has_165_unique_comparisons_and_no_duplicate_qlike(self):
        scores, paired = final.evaluate_forecasts(forecast_fixture(), evaluation_config())
        self.assertEqual(len(scores), 54)
        self.assertEqual(len(paired), 165)
        self.assertFalse(paired.duplicated(["asset", "candidate", "reference", "confidence", "metric"]).any())
        self.assertEqual(paired.metric.value_counts().to_dict(), {"quantile_loss": 66, "fz0": 66, "qlike": 33})
        self.assertTrue(paired.loc[paired.metric.eq("qlike"), "confidence"].eq(.95).all())
        self.assertTrue(paired.available.all())
        self.assertTrue(paired.pvalue_holm.ge(paired.pvalue).all())
        for family in ("exact", "independence", "conditional", "es"):
            self.assertTrue(scores[f"{family}_pvalue_holm"].dropna().ge(scores[f"{family}_pvalue"].dropna()).all())
        # Independently calculate candidate-minus-reference QLIKE on the same dates.
        target = 3 * np.sin(np.arange(130) / 7)
        expected = np.mean(np.log(1.1) + target ** 2 / 1.1 - target ** 2)
        selected = paired.loc[paired.candidate.eq("garch") & paired.metric.eq("qlike")]
        np.testing.assert_allclose(selected["mean"], expected)

    def test_all_nine_models_share_valid_dates_and_missing_rows_break_transitions(self):
        data = forecast_fixture(8)
        dates = sorted(data.date.unique())
        data.loc[data.date.eq(dates[2]) & data.model.eq("gru_weather"), "status"] = "failed"
        data.loc[data.date.eq(dates[5]), "target"] = np.nan
        original = data.copy(deep=True)
        scores, paired = final.evaluate_forecasts(data, evaluation_config())
        self.assertTrue(scores.grid_rows.eq(8).all())
        self.assertTrue(scores.common_scored_rows.eq(6).all())
        self.assertTrue(scores.excluded_rows.eq(2).all())
        self.assertTrue(scores.missing_realized_rows.eq(1).all())
        # Segments [0,1], [3,4], [6,7] have only three adjacent transitions.
        self.assertTrue(scores.adjacent_pairs.eq(3).all())
        self.assertTrue(scores[["n00", "n01", "n10", "n11"]].sum(axis=1).eq(3).all())
        self.assertTrue(paired.grid_rows.eq(8).all())
        self.assertTrue(paired.observations.eq(6).all())
        self.assertTrue(paired.reason.str.contains("fewer than 100").all())
        pd.testing.assert_frame_equal(data, original)

    def test_disagreed_targets_intervals_duplicates_and_missing_model_are_rejected(self):
        source = forecast_fixture(4)
        variants = []
        targets = source.copy()
        targets.loc[2, "target"] += 1
        variants.append((targets, "targets"))
        intervals = source.copy()
        intervals.loc[2, "interval_start"] = "2023-12-29"
        variants.append((intervals, "intervals"))
        variants.append((pd.concat([source, source.iloc[:1]], ignore_index=True), "Duplicate"))
        variants.append((source.loc[source.model.ne("ssm_weather")], "incomplete model set"))
        variants.append((source.loc[source.asset.ne("wti_spot")], "asset/confidence"))
        variants.append((source.loc[source.confidence.ne(.99)], "asset/confidence"))
        variants.append((source.drop(index=2), "model/confidence grid"))
        outside = source.copy()
        outside.loc[outside.date.eq("2024-01-01"), "date"] = "2023-12-31"
        variants.append((outside, "outside final evaluation interval"))
        for data, error in variants:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                final.evaluate_forecasts(data, evaluation_config())

    def test_nonpositive_es_keeps_coverage_but_marks_joint_score_unavailable(self):
        data = forecast_fixture(5)
        mask = data.model.eq("gru_weather")
        data.loc[mask, ["var", "es"]] = [-2.0, -1.0]
        scores, paired = final.evaluate_forecasts(data, evaluation_config())
        candidate = scores.loc[scores.model.eq("gru_weather")]
        self.assertTrue(candidate.common_scored_rows.eq(5).all())
        self.assertTrue(candidate.quantile_loss_rows.eq(5).all())
        self.assertTrue(candidate.fz0_rows.eq(0).all())
        self.assertTrue(candidate.es_score_rows.eq(0).all())
        self.assertTrue(candidate.mean_fz0.isna().all())
        comparisons = paired.loc[paired.candidate.eq("gru_weather") & paired.metric.eq("fz0")]
        self.assertTrue(comparisons.observations.eq(0).all())
        self.assertTrue(comparisons.pvalue.isna().all())
        self.assertEqual(len(paired), 165)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "config").mkdir()
        (self.root / "docs").mkdir()
        self.source = self.root / "input"
        self.source.mkdir()
        shutil.copytree(ROOT / "src/weather_risk", self.root / "src/weather_risk",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "requirements-ml.lock.txt", self.root / "requirements-ml.lock.txt")
        self.config = evaluation_config()
        self.model = ml_fixtures.config()
        self.baseline = baseline_fixtures.config()
        self.config["assets"] = self.model["assets"]
        self.config_path = self.root / "config/evaluation.json"
        self.config_path.write_text(json.dumps(self.config))
        (self.root / self.config["protocol_document"]).write_text("Synthetic test protocol; no empirical values.\n")
        self.write_snapshot()
        root_patch = patch.object(final, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write_snapshot(self):
        source_config = {"start": "2018-01-01", "end": "2026-01-01", "series": [
            {"series_id": "weather", "provider": "eccc", "timezone": "UTC", "stale_days": 7},
        ]}
        (self.source / "config.json").write_text(json.dumps(source_config))
        (self.source / "validation.json").write_text(json.dumps({"passed": True, "errors": []}))
        fields = ["series_id", "observation_date", "value", "unit", "source", "available_at",
                  "retrieved_at", "availability_basis", "vintage", "quality_flag"]
        with (self.source / "observations.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            dates = list(pd.date_range("2023-11-20", periods=90)) + [pd.Timestamp("2025-12-31")]
            for index, date in enumerate(dates):
                day = date.date().isoformat()
                for series, value, unit in (("gas", 50 + 3 * np.sin(index / 3), "USD/MMBtu"),
                                            ("weather", index % 8, "degC_day")):
                    writer.writerow(dict(zip(fields, [series, day, value, unit, "synthetic fixture",
                        day + "T23:00:00Z", "2026-01-01T00:00:00Z", "synthetic", "synthetic", ""])))
            # The final evaluation must still filter values after its fixed end date.
            writer.writerow(dict(zip(fields, ["gas", "2026-01-01", "UNREAD_AFTER_FINAL_WINDOW", "USD/MMBtu",
                "synthetic fixture", "UNREAD_TIMESTAMP", "UNREAD_TIMESTAMP", "synthetic", "synthetic", ""])))
        hashes = {name: final.sha(self.source / name) for name in ("config.json", "validation.json", "observations.csv")}
        (self.source / "manifest.json").write_text(json.dumps({
            "phase": 1, "status": "passed", "mode": "live", "config_sha256": hashes["config.json"],
            "artifacts": hashes, "historical_point_in_time_certified": False,
        }))
        for settings, key in ((self.model, "model_config"), (self.baseline, "baseline_config")):
            settings.update(input_run="input", input_observations_sha256=hashes["observations.csv"])
            (self.root / self.config[key]).write_text(json.dumps(settings))

    @contextmanager
    def mock_expensive_work(self):
        # Keep actual source parsing, targets, weather alignment, training eligibility,
        # forecast loops, diagnostics, checkpoint writing and manifest publication.
        with baseline_fixtures.mocked_fits(), ml_fixtures.mocked_fits(), \
             patch.object(final, "plot_results", side_effect=self.write_plot_placeholders):
            yield

    @staticmethod
    def write_plot_placeholders(scores, paired, output):
        for stem in ("01_var_coverage", "02_weather_comparisons"):
            for extension in ("png", "pdf"):
                (output / f"{stem}.{extension}").write_bytes(b"deterministic test plot placeholder\n")

    def run_fixture(self, *, publish=False):
        with self.mock_expensive_work():
            return final.run_evaluation(self.config_path, allow_latest_vintage=True, publish=publish)

    def test_seal_is_exclusive_and_access_cannot_be_relabelled_as_untouched(self):
        seal_path = final.freeze_protocol(self.config_path)
        seal_bytes = seal_path.read_bytes()
        with self.assertRaises(FileExistsError):
            final.freeze_protocol(self.config_path)
        self.assertEqual(seal_path.read_bytes(), seal_bytes)
        config, model, seal = final.verify_seal(self.config_path)
        self.assertFalse(seal["final_test_numeric_values_read_at_freeze"])
        self.assertEqual(seal["input_observations_sha256"], self.model["input_observations_sha256"])
        self.assertIn("src/weather_risk/final_evaluation.py", seal["files"])
        first = final.record_access(config)
        self.assertEqual(first, final.record_access(config))
        seal_path.unlink()
        with self.assertRaisesRegex(ValueError, "already recorded"):
            final.freeze_protocol(self.config_path)

    def test_mutated_sealed_files_or_input_are_rejected_before_access(self):
        final.freeze_protocol(self.config_path)
        names = [self.config["model_config"], self.config["protocol_document"], "requirements-ml.lock.txt",
                 "src/weather_risk/final_evaluation.py", "input/observations.csv", "input/config.json",
                 "input/manifest.json", "input/validation.json"]
        for name in names:
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            try:
                with self.subTest(name=name), patch.object(final, "record_access") as access, \
                     self.assertRaisesRegex(ValueError, "Sealed"):
                    final.run_evaluation(self.config_path, allow_latest_vintage=True)
                access.assert_not_called()
            finally:
                path.write_bytes(original)

    def test_unrelated_new_code_does_not_change_sealed_experiment(self):
        final.freeze_protocol(self.config_path)
        expected = final.verify_seal(self.config_path)
        (self.root / "src/weather_risk/new_module.py").write_text("# added after freeze\n")
        self.assertEqual(final.verify_seal(self.config_path), expected)

    def test_no_acknowledgment_never_verifies_or_records_access(self):
        with patch.object(final, "verify_seal") as verify, patch.object(final, "record_access") as access, \
             patch.object(final, "load_development_observations") as loader, \
             self.assertRaisesRegex(ValueError, "acknowledge"):
            final.run_evaluation(self.config_path)
        verify.assert_not_called()
        access.assert_not_called()
        loader.assert_not_called()
        self.assertFalse((self.root / self.config["first_access_audit"]).exists())
        self.assertFalse((self.root / "runs").exists())

    def test_audit_is_written_before_loader_even_when_first_parse_fails(self):
        final.freeze_protocol(self.config_path)
        def fail_loader(source, start, end):
            audit_path = self.root / self.config["first_access_audit"]
            self.assertTrue(audit_path.is_file())
            audit = json.loads(audit_path.read_text())
            self.assertEqual(audit["seal_sha256"], final.sha(self.root / self.config["seal"]))
            self.assertEqual((start, end), ("2018-01-01", "2025-12-31"))
            raise ValueError("synthetic parse failure")
        with patch.object(final, "load_development_observations", side_effect=fail_loader), \
             self.assertRaisesRegex(ValueError, "synthetic parse failure"):
            final.run_evaluation(self.config_path, allow_latest_vintage=True)
        output, = (self.root / "runs").glob("evaluation-*")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(manifest["final_test_numeric_values_read"])
        self.assertIn("synthetic parse failure", manifest["failure"])
        self.assertFalse((self.root / "reports").exists())
        with self.assertRaisesRegex(ValueError, "already recorded"):
            final.freeze_protocol(self.config_path)

    def test_replays_match_all_artifacts_and_use_only_final_forecast_window(self):
        final.freeze_protocol(self.config_path)
        first = self.run_fixture()
        audit_path = self.root / self.config["first_access_audit"]
        original_audit = audit_path.read_bytes()
        second = self.run_fixture()
        first_manifest = json.loads((first / "manifest.json").read_text())
        second_manifest = json.loads((second / "manifest.json").read_text())
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(audit_path.read_bytes(), original_audit)
        self.assertEqual(first_manifest["status"], "passed")
        self.assertTrue(first_manifest["final_test_numeric_values_read"])
        self.assertGreater(first_manifest["member_fits"], 0)
        for name, digest in first_manifest["artifacts"].items():
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
            self.assertEqual(final.sha(second / name), digest, name)
        forecasts = pd.read_csv(first / "forecasts.csv")
        self.assertTrue(forecasts.date.between("2024-01-01", "2025-12-31").all())
        self.assertEqual(forecasts.date.min(), "2024-01-01")
        self.assertEqual(forecasts.date.max(), "2025-12-31")
        for _, group in forecasts.groupby(["asset", "model", "confidence"]):
            self.assertTrue(group.date.is_monotonic_increasing)
        self.assertEqual(set(forecasts.model), set(final.ALL_MODELS))
        fits = json.loads((first / "model_fits.json").read_text())
        panel = pd.read_csv(first / "design_panel.csv")
        self.assertTrue(any(panel.date.iloc[fit["train_end_index"]] >= "2024-01-01" for fit in fits["neural"]))
        for fit in fits["baselines"] + fits["neural"]:
            self.assertLessEqual(pd.Timestamp(fit["max_label_available_at"]), pd.Timestamp(fit["fit_origin"]))
        self.assertFalse((self.root / "reports").exists())

    def test_failed_forecasts_are_retained_and_never_published(self):
        final.freeze_protocol(self.config_path)
        with self.mock_expensive_work(), patch.object(ml, "fit_sequence_variance", side_effect=RuntimeError("member failed")):
            output = final.run_evaluation(self.config_path, allow_latest_vintage=True, publish=True)
        manifest = json.loads((output / "manifest.json").read_text())
        forecasts = pd.read_csv(output / "forecasts.csv")
        self.assertEqual(manifest["status"], "completed_with_failures")
        self.assertGreater(manifest["failed_forecast_rows"], 0)
        failed = forecasts.loc[forecasts.model.isin(ml.MODELS)]
        self.assertTrue(failed.status.eq("failed").all())
        self.assertTrue(failed.error.str.contains("member failed").all())
        self.assertEqual(len(failed), manifest["failed_forecast_rows"])
        self.assertFalse((self.root / "reports").exists())


    def test_missing_package_metadata_creates_no_run(self):
        final.freeze_protocol(self.config_path)
        with patch.object(final.importlib.metadata, "version",
                          side_effect=final.importlib.metadata.PackageNotFoundError("optional dependency")), \
             self.assertRaises(final.importlib.metadata.PackageNotFoundError):
            self.run_fixture()
        self.assertEqual(list((self.root / "runs").glob("evaluation-*")), [])
        self.assertFalse((self.root / "reports").exists())

    def test_initial_artifact_write_failure_records_failed_run(self):
        final.freeze_protocol(self.config_path)
        write_json = final.write_json

        def fail_initial_artifact(path, value):
            if Path(path).name == "evaluation_config.json":
                raise OSError("initial artifact write failed")
            return write_json(path, value)

        with patch.object(final, "write_json", side_effect=fail_initial_artifact), \
             self.assertRaisesRegex(OSError, "initial artifact write failed"):
            self.run_fixture()
        output, = (self.root / "runs").glob("evaluation-*")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure"], "OSError: initial artifact write failed")
        self.assertEqual(manifest["artifacts"], {})
        self.assertFalse((self.root / "reports").exists())

    def test_missing_pinned_input_explains_public_reproduction_boundary(self):
        final.freeze_protocol(self.config_path)
        (self.source / "observations.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "make verify-results"):
            self.run_fixture()
        self.assertEqual(list((self.root / "runs").glob("evaluation-*")), [])


if __name__ == "__main__":
    unittest.main()
