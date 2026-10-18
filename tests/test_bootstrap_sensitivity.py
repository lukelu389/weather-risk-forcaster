"""Synthetic provenance, complete-family, and reproducibility correction checks."""

import copy
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
    raise unittest.SkipTest("Bootstrap sensitivity tests require optional ML dependencies") from exc

import test_final_evaluation as fixtures
from weather_risk import bootstrap_sensitivity as sensitivity


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        original_root = sensitivity.ROOT
        self.source = self.root / "runs/primary"
        self.source.mkdir(parents=True)
        (self.root / "config").mkdir()
        (self.root / "docs").mkdir()
        (self.root / "src/weather_risk").mkdir(parents=True)
        snapshot, code = {}, {}
        for name in sensitivity.SEALED_MODULES:
            relative = f"src/weather_risk/{name}"
            shutil.copyfile(original_root / relative, self.root / relative)
            snapshot[relative] = (self.root / relative).read_text()
            code[name] = sensitivity.sha(self.root / relative)
        shutil.copyfile(original_root / "requirements-ml.lock.txt", self.root / "requirements-ml.lock.txt")
        shutil.copyfile(self.root / "requirements-ml.lock.txt", self.source / "requirements-ml.lock.txt")
        self.evaluation = fixtures.evaluation_config()
        self.forecasts = fixtures.forecast_fixture(8)
        self.scores, self.paired = sensitivity.evaluate_forecasts(self.forecasts, self.evaluation)
        for name, frame in (("forecasts", self.forecasts), ("calibration_scores", self.scores),
                            ("paired_comparisons", self.paired)):
            frame.to_csv(self.source / f"{name}.csv", index=False, float_format="%.12g")
        sensitivity.write_json(self.source / "evaluation_config.json", self.evaluation)
        sensitivity.write_json(self.source / "source_snapshot.json", snapshot)
        sensitivity.write_json(self.source / "protocol_seal.json", {
            "files": {f"src/weather_risk/{name}": digest for name, digest in code.items()},
        })
        self.manifest = {
            "phase": 5, "status": "passed", "code_sha256": code,
            "protocol_seal_sha256": sensitivity.sha(self.source / "protocol_seal.json"),
            "artifacts": {name: sensitivity.sha(self.source / name) for name in sensitivity._PRIMARY_ARTIFACTS},
        }
        sensitivity.write_json(self.source / "manifest.json", self.manifest)
        protocol = self.root / "docs/addendum.md"
        protocol.write_text("Synthetic post-exposure correction protocol; 19999 resamples, all families.\n")
        self.config = {
            "input_run": "runs/primary", "input_manifest_sha256": sensitivity.sha(self.source / "manifest.json"),
            "protocol_document": "docs/addendum.md", "protocol_document_sha256": sensitivity.sha(protocol),
            "reason": "Synthetic post-exposure correction test", "replications": 19999,
        }
        self.config_path = self.root / "config/sensitivity.json"
        sensitivity.write_json(self.config_path, self.config)
        root_patch = patch.object(sensitivity, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_complete_replay_changes_only_bootstrap_budget_and_preserves_primary(self):
        before = {name: sensitivity.sha(self.source / name) for name in self.manifest["artifacts"]}
        with patch.object(sensitivity, "evaluate_forecasts", wraps=sensitivity.evaluate_forecasts) as evaluate:
            first = sensitivity.run_sensitivity(self.config_path)
        settings = evaluate.call_args.args[1]
        self.assertEqual(settings, {**self.evaluation, "bootstrap_replications": 19999})
        second = sensitivity.run_sensitivity(self.config_path, publish=False)
        first_manifest = json.loads((first / "manifest.json").read_text())
        second_manifest = json.loads((second / "manifest.json").read_text())
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["status"], "passed")
        self.assertEqual((first_manifest["calibration_tests"], first_manifest["paired_comparisons"]), (54, 165))
        self.assertFalse(first_manifest["untouched_confirmatory_analysis"])
        self.assertFalse(first_manifest["models_refitted"])
        for name in first_manifest["artifacts"]:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        self.assertEqual(before, {name: sensitivity.sha(self.source / name) for name in before})
        self.assertFalse((self.root / "reports/phase5").exists())
        self.assertTrue((self.root / "reports/phase5-sensitivity/sensitivity_report.md").exists())
        provenance = json.loads((first / "provenance.json").read_text())
        self.assertEqual(provenance["source_csv_significant_digits"], 12)
        self.assertTrue(provenance["invariance_checks"]["calibration"]["integer_counts_match_exactly"])

    def test_artifact_protocol_and_code_tampering_rejected_before_csv_parse(self):
        for relative in ["runs/primary/manifest.json", "runs/primary/forecasts.csv",
                         "runs/primary/evaluation_config.json", "runs/primary/calibration_scores.csv",
                         "runs/primary/paired_comparisons.csv", "runs/primary/source_snapshot.json",
                         "docs/addendum.md", "src/weather_risk/evaluation.py", "requirements-ml.lock.txt"]:
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            try:
                with self.subTest(relative=relative), patch.object(sensitivity.pd, "read_csv") as parse, \
                     self.assertRaisesRegex(ValueError, "differs"):
                    sensitivity.run_sensitivity(self.config_path, publish=False)
                parse.assert_not_called()
            finally:
                path.write_bytes(original)

    def test_missing_artifact_hash_or_nonpassed_manifest_is_rejected(self):
        for change in ("missing_hash", "failed_status", "different_phase"):
            manifest = copy.deepcopy(self.manifest)
            if change == "missing_hash":
                manifest["artifacts"].pop("forecasts.csv")
            elif change == "failed_status":
                manifest["status"] = "failed"
            else:
                manifest["phase"] = 4
            sensitivity.write_json(self.source / "manifest.json", manifest)
            config = {**self.config, "input_manifest_sha256": sensitivity.sha(self.source / "manifest.json")}
            sensitivity.write_json(self.config_path, config)
            with self.subTest(change=change), patch.object(sensitivity.pd, "read_csv") as parse, self.assertRaises(ValueError):
                sensitivity.run_sensitivity(self.config_path, publish=False)
            parse.assert_not_called()

    def test_other_replication_budgets_or_undisclosed_reason_are_rejected(self):
        for updates in ({"replications": 1000}, {"replications": True}, {"reason": ""}):
            sensitivity.write_json(self.config_path, {**self.config, **updates})
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                sensitivity.run_sensitivity(self.config_path, publish=False)

    def test_missing_hypothesis_or_changed_sample_counts_prevents_publication(self):
        variants = []
        altered = self.scores.copy()
        altered.loc[0, "exceedances"] += 1
        variants.append((altered, self.paired, "exceedances"))
        variants.append((self.scores.iloc[:-1], self.paired, "54-row"))
        variants.append((self.scores, self.paired.iloc[:-1], "165-row"))
        altered = self.paired.copy()
        altered.loc[0, "observations"] -= 1
        variants.append((self.scores, altered, "observations"))
        for scores, paired, expected in variants:
            with self.subTest(expected=expected), \
                 patch.object(sensitivity, "evaluate_forecasts", return_value=(scores, paired)), \
                 self.assertRaisesRegex(ValueError, expected):
                sensitivity.run_sensitivity(self.config_path)
        self.assertFalse((self.root / "reports/phase5-sensitivity").exists())
        for path in (self.root / "runs").glob("bootstrap-sensitivity-*/manifest.json"):
            self.assertEqual(json.loads(path.read_text())["status"], "failed")

    def test_small_numeric_rounding_tolerated_but_changed_means_rejected(self):
        updated = self.scores.copy()
        updated.loc[0, "mean_quantile_loss"] += 5e-9
        check = sensitivity._check_invariants(self.scores, updated, keys=sensitivity._SCORE_KEYS,
            changed_columns=sensitivity._CHANGED_SCORE_COLUMNS, expected_rows=54)
        self.assertLess(check["maximum_absolute_numeric_difference"], 1e-8)
        updated.loc[0, "mean_quantile_loss"] += 1e-6
        with self.assertRaisesRegex(ValueError, "mean_quantile_loss"):
            sensitivity._check_invariants(self.scores, updated, keys=sensitivity._SCORE_KEYS,
                changed_columns=sensitivity._CHANGED_SCORE_COLUMNS, expected_rows=54)

    def test_rejection_resolution_and_unavailable_hypotheses_are_explicit(self):
        scores, paired = self.scores.copy(), self.paired.copy()
        scores["es_pvalue_holm"] = np.nan
        paired["pvalue_holm"] = .1
        paired.loc[:2, "pvalue_holm"] = .01
        result = sensitivity._rejection_summary(self.scores, self.paired, scores, paired)
        self.assertEqual(result[0]["sensitivity_unavailable_tests"], 54)
        self.assertEqual(result[1]["sensitivity_adjusted_rejections_5pct"], 3)
        for row in result:
            self.assertGreater(row["primary_minimum_raw_pvalue"], row["first_holm_threshold_5pct"])
            self.assertLess(row["sensitivity_minimum_raw_pvalue"], row["first_holm_threshold_5pct"])


if __name__ == "__main__":
    unittest.main()
