from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weather_risk.demo import synthetic_observations
from weather_risk.pipeline import ROOT, digest, load_config, main, run


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config/demo.json")
        self.config["start"] = "2024-01-01"
        self.config["end"] = "2024-03-31"

    def test_same_seed_reproduces_data_and_alignment_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run(self.config, "demo", root)
            second = run(self.config, "demo", root)
            for name in ("observations.csv", "feature_panel.csv", "validation.json", "config.json"):
                self.assertEqual(digest(first / name), digest(second / name))
            manifest = json.loads((second / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "passed")
            self.assertFalse(manifest["historical_point_in_time_certified"])
            for name, checksum in manifest["artifacts"].items():
                self.assertEqual(digest(second / name), checksum)

    def test_derived_weather_and_month_end_contract(self):
        rows = synthetic_observations(self.config)
        values = {(r.series_id, r.observation_date): r.value for r in rows}
        for r in rows:
            if r.series_id.endswith("_temp"):
                base = r.series_id.removesuffix("_temp")
                self.assertAlmostEqual(values[(base + "_hdd18", r.observation_date)], max(18 - r.value, 0), places=7)
                self.assertAlmostEqual(values[(base + "_cdd18", r.observation_date)], max(r.value - 18, 0), places=7)
        cpi_dates = [r.observation_date for r in rows if r.series_id == "us_cpi"]
        self.assertEqual(cpi_dates, ["2024-01-31", "2024-02-29", "2024-03-31"])

    def test_synthetic_history_unchanged_by_future_extension_or_subset(self):
        original = synthetic_observations(self.config)
        extended = synthetic_observations({**self.config, "end": "2024-06-30"})
        by_key = {(r.series_id, r.observation_date): r.value for r in extended}
        self.assertTrue(all(r.value == by_key[(r.series_id, r.observation_date)] for r in original))
        subset = synthetic_observations({**self.config, "series": self.config["series"][3:5]})
        self.assertTrue(all(r.value == by_key[(r.series_id, r.observation_date)] for r in subset))

    def test_replay_command_verifies_exact_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run(self.config, "demo", root)
            status = main(["--replay", str(first / "manifest.json"), "--output-root", str(root)])
            self.assertEqual(status, 0)
            other = [p for p in (root / "runs").iterdir() if p != first][0]
            self.assertEqual(digest(first / "feature_panel.csv"), digest(other / "feature_panel.csv"))

    def test_live_never_falls_back_to_synthetic_and_records_failure(self):
        config = {**self.config, "series": self.config["series"][:1]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("weather_risk.pipeline.fetch_series", side_effect=ValueError("upstream unavailable")):
                with self.assertRaisesRegex(ValueError, "Data validation failed"):
                    run(config, "live", root)
            folder = next((root / "runs").iterdir())
            manifest = json.loads((folder / "manifest.json").read_text())
            report = json.loads((folder / "validation.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(any("upstream unavailable" in e for e in report["errors"]))
            with (folder / "observations.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_live_replay_preserves_optional_source_skips(self):
        required = self.config["series"][0]
        optional = next(s for s in self.config["series"] if s["series_id"] == "us_gas_storage")
        config = {**self.config, "series": [required, optional]}
        rows = [replace(r, vintage="latest", availability_basis="assumed_lag", source="test")
                for r in synthetic_observations({**config, "series": [required]})]
        def initial_fetch(spec, *_):
            if spec["series_id"] == optional["series_id"]:
                raise ValueError("API key is not configured")
            return rows
        def replay_fetch(spec, *_):
            self.assertEqual(spec["series_id"], required["series_id"])
            return rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("weather_risk.pipeline.fetch_series", side_effect=initial_fetch):
                first = run(config, "live", root)
            with patch("weather_risk.pipeline.fetch_series", side_effect=replay_fetch) as fetch:
                self.assertEqual(main(["--replay", str(first / "manifest.json"), "--output-root", str(root)]), 0)
                self.assertEqual(fetch.call_count, 1)

    def test_live_latest_vintage_is_blocked_unless_explicitly_opted_in(self):
        config = {**self.config, "series": self.config["series"][:1]}
        data = [replace(r, vintage="latest", availability_basis="assumed_lag", source="test")
                for r in synthetic_observations(config)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("weather_risk.pipeline.fetch_series", return_value=data):
                strict = run(config, "live", root)
                research = run(config, "live", root, allow_latest_vintage=True)
            with (strict / "feature_panel.csv").open() as stream:
                self.assertTrue(all(r["value"] == "" for r in csv.DictReader(stream)))
            with (research / "feature_panel.csv").open() as stream:
                self.assertTrue(any(r["value"] != "" for r in csv.DictReader(stream)))


if __name__ == "__main__":
    unittest.main()
