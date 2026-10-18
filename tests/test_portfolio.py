"""Synthetic provenance, maturity and fixed-CAD-basket sensitivity tests."""

import json
from pathlib import Path
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
    raise unittest.SkipTest("Phase 6 runner tests require the optional model dependencies") from exc

from weather_risk import portfolio
from weather_risk.portfolio_risk import empirical_cad_var_es, log_returns_to_cad_losses
from weather_risk.risk import empirical_var_es, gaussian_var_es


def settings():
    return {"input_run": "final", "input_manifest_sha256": "0" * 64,
            "asset": "canadian_basket", "notional_cad": 1_000_000,
            "cost_bps": [0, 5, 10], "calendar_subperiods": [2024, 2025]}


def synthetic_inputs():
    dates = pd.date_range("2023-11-20", periods=90).append(pd.date_range("2025-01-01", periods=4))
    target = 3 * np.sin(np.arange(len(dates)) / 4)
    target[20:22] = [-60, -20]
    target[10] = np.nan
    target[45] = np.nan
    panel = pd.DataFrame({"asset": "canadian_basket", "date": dates,
        "origin_at": (dates - pd.Timedelta(days=2) + pd.Timedelta(hours=23)).tz_localize("UTC"),
        "interval_start": dates - pd.Timedelta(days=1),
        "label_available_at": (dates + pd.Timedelta(hours=23)).tz_localize("UTC"),
        "target": target, "unit": "log return (%)"})
    panel.loc[39, "label_available_at"] = pd.Timestamp("2026-01-01T00:00Z")
    model = {"test_start": "2024-01-01", "test_end": "2025-12-31", "confidences": [.95, .99],
        "train_window": 40, "minimum_train": 30,
        "prices": [{"series_id": f"equity_{i}", "equity": True, "transform": "log_pct"} for i in range(4)]}
    forecasts = []
    for i, row in enumerate(panel.itertuples(index=False)):
        if row.date < pd.Timestamp("2024-01-01"):
            continue
        # Independent fixture construction keeps eligibility explicit.
        eligible = [j for j in range(i) if np.isfinite(target[j]) and panel.label_available_at.iloc[j] <= row.origin_at]
        stop = eligible[-1] + 1 if eligible else 0
        sample = [target[j] for j in eligible if j >= max(0, stop - 40)]
        for name in portfolio.ALL_MODELS:
            for confidence in (.95, .99):
                mean, variance = (.3, 6.25) if name != "historical" else (float(np.mean(sample)), float(np.var(sample)))
                var, es = (empirical_var_es(sample, confidence) if name == "historical"
                           else gaussian_var_es(mean, np.sqrt(variance), confidence))
                forecasts.append({**row._asdict(), "model": name, "confidence": confidence,
                    "status": "passed", "error": "", "target_invalid_reason": "missing_current_price"
                    if not np.isfinite(row.target) else "", "n_available": len(sample),
                    "mean": mean, "variance": variance, "var": var, "es": es})
    return pd.DataFrame(forecasts), panel, model


class SensitivityTests(unittest.TestCase):
    def test_maturity_preserves_missing_grid_rows_and_excludes_current_future_labels(self):
        dates = pd.date_range("2024-01-01", periods=8, tz="UTC")
        panel = pd.DataFrame({"target": [1, np.nan, 3, 4, 5, 6, 7, 8], "label_available_at": dates})
        panel.loc[3, "label_available_at"] = pd.Timestamp("2026-01-01", tz="UTC")
        panel.loc[6:, "label_available_at"] = dates[0]
        result = portfolio.matured_sample(panel, 6, dates[4], 4)
        np.testing.assert_array_equal(result, [3, 5])
        panel.loc[[3, 6, 7], "target"] = 1e8
        np.testing.assert_array_equal(portfolio.matured_sample(panel, 6, dates[4], 4), result)

    def test_historical_es_transforms_each_observation_before_tail_averaging(self):
        forecasts, panel, model = synthetic_inputs()
        rows = portfolio.sensitivity_rows(forecasts, panel, settings(), model)
        row = rows.loc[rows.date.eq("2024-01-01") & rows.model.eq("historical")
                       & rows.confidence.eq(.95) & rows.cost_bps.eq(0)].iloc[0]
        cutoff = pd.Timestamp("2023-12-30T23:00Z")
        sample = [panel.target.iloc[j] for j in range(1, 41)
                  if np.isfinite(panel.target.iloc[j]) and panel.label_available_at.iloc[j] <= cutoff]
        expected = empirical_cad_var_es(sample, .95)
        self.assertAlmostEqual(row.es_cad, expected["es_cad"])
        log_es = empirical_var_es(sample, .95)[1]
        naive_es = log_returns_to_cad_losses([-log_es])[0]
        self.assertGreater(abs(row.es_cad - naive_es), 1000)

    def test_costs_shift_both_loss_and_risk_and_leave_exceedances_unchanged(self):
        forecasts, panel, model = synthetic_inputs()
        rows = portfolio.sensitivity_rows(forecasts, panel, settings(), model)
        self.assertEqual(len(rows), 3 * len(forecasts))
        self.assertEqual(set(rows.model), set(portfolio.ALL_MODELS))
        for _, group in rows.groupby(["date", "model", "confidence"]):
            self.assertEqual(group.cost_bps.tolist(), [0, 5, 10])
            np.testing.assert_allclose(group.var_cad - group.gross_var_cad, [0, 500, 1000])
            np.testing.assert_allclose(group.es_cad - group.gross_es_cad, [0, 500, 1000])
            np.testing.assert_array_equal(group.var_exceedance, group.gross_var_exceedance)
            if np.isfinite(group.gross_realized_loss_cad.iloc[0]):
                np.testing.assert_allclose(group.realized_loss_cad - group.gross_realized_loss_cad, [0, 500, 1000])
                self.assertEqual(group.var_exceedance.nunique(), 1)

    def test_missing_realizations_and_skipped_forecasts_remain_explicit(self):
        forecasts, panel, model = synthetic_inputs()
        mask = forecasts.date.eq(pd.Timestamp("2024-01-02")) & forecasts.model.eq("gru_weather")
        forecasts.loc[mask, "status"] = "skipped"
        forecasts.loc[mask, "error"] = "synthetic skipped forecast"
        forecasts.loc[mask, ["mean", "variance", "var", "es"]] = np.nan
        rows = portfolio.sensitivity_rows(forecasts, panel, settings(), model)
        skipped = rows.loc[rows.status.eq("skipped")]
        self.assertEqual(len(skipped), 6)
        self.assertTrue(skipped[["var_cad", "es_cad", "var_exceedance"]].isna().all().all())
        self.assertTrue(skipped.realized_loss_cad.notna().all())
        missing = rows.loc[rows.date.eq("2024-01-04")]
        self.assertEqual(len(missing), 54)
        self.assertTrue(missing[["realized_loss_cad", "var_exceedance"]].isna().all().all())
        self.assertTrue(missing.var_cad.notna().all())
        summary = portfolio.summarize(rows, settings())
        self.assertEqual(len(summary), 9 * 2 * 3 * 3)
        self.assertEqual(set(summary.period), {"all", "2024", "2025"})
        selected = summary.loc[summary.model.eq("gru_weather") & summary.confidence.eq(.95)
                               & summary.cost_bps.eq(0)].set_index("period")
        self.assertEqual(selected.loc["all", "skipped_rows"], 1)
        self.assertEqual(selected.loc["all", "missing_realized_rows"], 1)
        self.assertEqual(selected.loc["all", "forecast_rows"], selected.loc["2024", "forecast_rows"] + selected.loc["2025", "forecast_rows"])

    def test_changed_sample_counts_risks_units_grid_and_targets_fail(self):
        forecasts, panel, model = synthetic_inputs()
        changed = []
        counts = forecasts.copy()
        counts.loc[0, "n_available"] += 1
        changed.append((counts, "n_available"))
        risks = forecasts.copy()
        risks.loc[risks.model.eq("historical"), "es"] += 1
        changed.append((risks, "log-return VaR/ES"))
        units = forecasts.copy()
        units.loc[0, "unit"] = "USD/barrel change"
        changed.append((units, "log return"))
        changed.append((forecasts.drop(index=0), "Incomplete basket forecast grid"))
        changed.append((pd.concat([forecasts, forecasts.iloc[:1]], ignore_index=True), "Duplicate"))
        targets = forecasts.copy()
        targets.loc[0, "target"] += .1
        changed.append((targets, "targets"))
        failed = forecasts.copy()
        failed.loc[0, "status"] = "failed"
        changed.append((failed, "failed or unknown"))
        for data, error in changed:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                portfolio.sensitivity_rows(data, panel, settings(), model)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "final"
        self.source.mkdir()
        self.config = settings()
        self.config_path = self.root / "portfolio.json"
        self.forecasts, self.panel, self.model = synthetic_inputs()
        self.write_source()
        root_patch = patch.object(portfolio, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write_source(self, *, phase=5, status="passed"):
        self.forecasts.to_csv(self.source / "forecasts.csv", index=False, float_format="%.12g")
        self.panel.to_csv(self.source / "design_panel.csv", index=False, float_format="%.12g")
        (self.source / "model_config.json").write_text(json.dumps(self.model))
        artifacts = {name: portfolio.sha(self.source / name) for name in ("forecasts.csv", "design_panel.csv", "model_config.json")}
        (self.source / "manifest.json").write_text(json.dumps({"phase": phase, "status": status, "artifacts": artifacts}))
        self.config["input_manifest_sha256"] = portfolio.sha(self.source / "manifest.json")
        self.config_path.write_text(json.dumps(self.config))

    def test_repeated_runs_match_every_artifact_and_publish_expected_reports(self):
        first = portfolio.run_portfolio(self.config_path, publish=False)
        self.assertFalse((self.root / "reports").exists())
        second = portfolio.run_portfolio(self.config_path, publish=True)
        manifest = json.loads((first / "manifest.json").read_text())
        self.assertEqual(manifest, json.loads((second / "manifest.json").read_text()))
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["sensitivity_rows"], 3 * len(self.forecasts))
        self.assertTrue(manifest["cost_exceedance_invariant"])
        for name, digest in manifest["artifacts"].items():
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
            self.assertEqual(portfolio.sha(second / name), digest)
        for name in ("summary.csv", "portfolio_report.md", "manifest.json", "portfolio_config.json"):
            self.assertEqual((second / name).read_bytes(), (self.root / "reports/phase6" / name).read_bytes())

    def test_pins_and_all_consumed_hashes_are_verified_before_csv_parsing(self):
        for name in ("manifest.json", "forecasts.csv", "design_panel.csv", "model_config.json"):
            path = self.source / name
            saved = path.read_bytes()
            path.write_bytes(saved + b"\n")
            try:
                with self.subTest(name=name), patch.object(portfolio.pd, "read_csv") as reading, self.assertRaises(ValueError):
                    portfolio.run_portfolio(self.config_path)
                reading.assert_not_called()
            finally:
                path.write_bytes(saved)
        self.assertFalse((self.root / "runs").exists())

    def test_nonpassed_or_wrong_phase_input_is_rejected_before_csv_parsing(self):
        for changes in ({"status": "failed"}, {"status": "completed_with_failures"}, {"phase": 4}):
            self.write_source(**changes)
            with self.subTest(changes=changes), patch.object(portfolio.pd, "read_csv") as reading, \
                 self.assertRaisesRegex(ValueError, "passed Phase 5"):
                portfolio.run_portfolio(self.config_path)
            reading.assert_not_called()

    def test_fixed_sensitivity_scope_cannot_be_silently_changed(self):
        for changes in ({"asset": "wti_spot"}, {"notional_cad": 2_000_000}, {"cost_bps": [0, 10]},
                        {"calendar_subperiods": [2008, 2024]}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "fixed portfolio"):
                portfolio.validate_config({**self.config, **changes})

    def test_reconstruction_failure_saves_manifest_without_publication(self):
        self.forecasts.loc[self.forecasts.model.eq("historical"), "es"] += 1
        self.write_source()
        with self.assertRaisesRegex(ValueError, "log-return VaR/ES"):
            portfolio.run_portfolio(self.config_path)
        output, = (self.root / "runs").glob("portfolio-*")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("log-return VaR/ES", manifest["failure"])
        self.assertFalse((self.root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
