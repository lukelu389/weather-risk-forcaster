"""Phase 2 protocol, input gates, and run artifact lifecycle checks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy
    import pandas as pd
    import matplotlib
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "pandas", "matplotlib"}:
        raise
    raise unittest.SkipTest("EDA pipeline tests require numpy, pandas and matplotlib") from exc

import test_eda_core as fixtures
from weather_risk import eda


class ProtocolTests(unittest.TestCase):
    def test_protocol_rejects_overlap_reversal_and_noncalendar_bounds(self):
        config = json.loads((eda.ROOT / "config/analysis.json").read_text())
        eda.validate_protocol(config)
        for changes in (
            {"tuning_start": config["analysis_end"]},
            {"test_start": config["tuning_end"]},
            {"analysis_end": "2017-01-01"},
            {"analysis_start": "2018-01-01T12:00:00"},
            {"analysis_start": "2018-01-01T00:00:00Z"},
            {"test_end": "NaT"},
        ):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                eda.validate_protocol({**config, **changes})


class AnalysisRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "input"
        self.run.mkdir()
        fixtures.SnapshotTests().write_snapshot(self.run)
        source_config = self.run / "config.json"
        source_config.write_text(json.dumps({"start": "2020-01-01", "end": "2025-12-31", "series": []}))
        manifest_path = self.run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config_sha256"] = eda.sha(source_config)
        manifest["artifacts"]["config.json"] = eda.sha(source_config)
        manifest_path.write_text(json.dumps(manifest))
        self.config = json.loads((eda.ROOT / "config/analysis.json").read_text())
        self.config.update(input_run="input", input_observations_sha256=eda.sha(self.run / "observations.csv"),
                           analysis_start="2020-01-01", prices=[fixtures.spec()])
        self.path = self.root / "analysis.json"
        self.save_config()
        self.root_patch = patch.object(eda, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def save_config(self):
        self.path.write_text(json.dumps(self.config))

    def test_pinned_input_change_rejected(self):
        self.config["input_observations_sha256"] = "wrong"
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Pinned input differs"):
            eda.run_analysis(self.path, allow_latest_vintage=True)
        self.assertFalse((self.root / "runs").exists())

    def test_latest_vintage_requires_acknowledgement(self):
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            eda.run_analysis(self.path)
        self.assertFalse((self.root / "runs").exists())

    def test_synthetic_snapshot_rejected(self):
        path = self.run / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["mode"] = "demo"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "live-data snapshot"):
            eda.run_analysis(self.path, allow_latest_vintage=True)

    def test_success_hashes_outputs_and_no_publish_preserves_existing_report(self):
        published = self.root / "reports/phase2"
        published.mkdir(parents=True)
        sentinel = published / "analysis_report.md"
        sentinel.write_text("existing report")
        with patch.object(eda, "weather_features", return_value=pd.DataFrame({"value": [1]})), \
             patch.object(eda, "relationships", return_value=pd.DataFrame({"rho": [0.1]})), \
             patch("weather_risk.eda_plots.make_plots", return_value=[]):
            out = eda.run_analysis(self.path, allow_latest_vintage=True, publish=False)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["models_fitted"], 0)
        self.assertFalse(manifest["source"]["numeric_values_read_outside_development"])
        self.assertEqual(sentinel.read_text(), "existing report")
        for name, expected in manifest["artifacts"].items():
            self.assertEqual(eda.sha(out / name), expected, name)
        self.assertIn("targets.csv", manifest["artifacts"])
        self.assertIn("source_snapshot.json", manifest["artifacts"])

    def test_missing_pinned_input_explains_public_reproduction_boundary(self):
        (self.run / "observations.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "make verify-results"):
            eda.run_analysis(self.path, allow_latest_vintage=True)
        self.assertFalse((self.root / "runs").exists())

    def test_missing_package_metadata_creates_no_run(self):
        with patch.object(eda.importlib.metadata, "version",
                          side_effect=eda.importlib.metadata.PackageNotFoundError("matplotlib")), \
             self.assertRaises(eda.importlib.metadata.PackageNotFoundError):
            eda.run_analysis(self.path, allow_latest_vintage=True)
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "reports").exists())

    def test_initial_artifact_write_failure_records_failed_run(self):
        write_json = eda.write_json

        def fail_initial_artifact(path, value):
            if Path(path).name == "analysis_config.json":
                raise OSError("initial artifact write failed")
            return write_json(path, value)

        with patch.object(eda, "write_json", side_effect=fail_initial_artifact), \
             self.assertRaisesRegex(OSError, "initial artifact write failed"):
            eda.run_analysis(self.path, allow_latest_vintage=True)
        out, = (self.root / "runs").iterdir()
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure"], "OSError: initial artifact write failed")
        self.assertEqual(manifest["artifacts"], {})
        self.assertFalse((self.root / "reports").exists())

    def test_failed_run_records_failure_without_publishing(self):
        with patch.object(eda, "make_targets", side_effect=ValueError("target failure")), \
             self.assertRaisesRegex(ValueError, "target failure"):
            eda.run_analysis(self.path, allow_latest_vintage=True)
        out, = (self.root / "runs").iterdir()
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure"], "ValueError: target failure")
        self.assertFalse((self.root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
