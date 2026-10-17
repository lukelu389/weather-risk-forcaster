"""Synthetic tests of the future-only protocol, snapshot and numeric access gates."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from weather_risk import prospective as gate


FREEZE_TIME = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
ACCESS_TIME = datetime(2028, 1, 9, 12, tzinfo=timezone.utc)


class ProspectiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "src/weather_risk").mkdir(parents=True)
        shutil.copyfile(gate.__file__, self.root / gate.GATE_MODULE)
        (self.root / "config").mkdir()
        (self.root / "config/models.json").write_text('{"model":"synthetic"}\n')
        (self.root / "input.csv").write_text("date,value\n2023-01-01,1\n")
        self.path = self.root / "config/prospective.json"
        self.protocol = {
            "schema_version": 1, "status": "ready", "protocol_id": "synthetic-future-test",
            "evaluation": {"start": "2027-01-01", "end": "2027-12-31", "access_not_before": "2028-01-08"},
            "hypotheses": [{"id": "weather", "family": "comparison", "statement": "Synthetic paired mean equals zero"}],
            "models": {"synthetic": {"constant_variance": 1}},
            "refit": {"schedule": "monthly", "window_rows": 756, "minimum_training_rows": 252},
            "maturity": {"target_delay_days": 8, "forecast_origin": "18:00 America/Toronto",
                         "availability_policy": "Synthetic availability timestamps",
                         "prior_holdout_labels_in_later_fits": True},
            "inference": {"alpha": .05, "adjustment": "holm", "family_sizes": {"comparison": 1},
                          "bootstrap_families": ["comparison"], "missing_policy": "Retain grid gaps",
                          "bootstrap": {"replications": 19999, "minimum_valid_fraction": .95,
                                        "block_length": 20, "minimum_observations": 100, "seed": 1729}},
            "files": {"code": [gate.GATE_MODULE], "config": ["config/models.json"], "input": ["input.csv"]},
            "holdout": {"path": "future.csv", "date_column": "observation_date", "availability_column": "available_at",
                        "series_column": "series_id", "series_ids": ["gas"], "numeric_columns": ["value"]},
            "outputs": {"seal": "records/seal.json", "first_access": "records/access.json"},
        }
        self.write_protocol()
        self.clock = patch.object(gate, "_utc_now", return_value=FREEZE_TIME)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)

    def write_protocol(self):
        self.path.write_text(json.dumps(self.protocol, indent=2) + "\n")

    def freeze(self):
        self.write_protocol()
        return gate.freeze_protocol(self.path, root=self.root)

    def acquire(self, rows="2027-05-01,2027-05-08T12:00:00Z,gas,2.5\n"):
        (self.root / "future.csv").write_text("observation_date,available_at,series_id,value\n" + rows)
        self.now.return_value = ACCESS_TIME

    def test_freeze_is_exclusive_and_verifies_exact_dependency_bytes(self):
        config_bytes = self.path.read_bytes()
        seal_path = self.freeze()
        seal_bytes = seal_path.read_bytes()
        protocol, seal = gate.verify_protocol(self.path, root=self.root)
        self.assertEqual(protocol, self.protocol)
        self.assertEqual(set(seal["files"]), {gate.GATE_MODULE, "config/models.json", "input.csv", "config/prospective.json"})
        self.assertFalse(seal["holdout_numeric_values_parsed"])
        self.assertEqual(self.path.read_bytes(), config_bytes)
        with self.assertRaises(FileExistsError):
            self.freeze()
        self.assertEqual(seal_path.read_bytes(), seal_bytes)
        (self.root / "unrelated.txt").write_text("Unrelated later work is allowed")
        gate.verify_protocol(self.path, root=self.root)

    def test_consumed_dates_and_nonprospective_dates_are_rejected(self):
        for start, end, clock, error in [
            ("2024-01-01", "2025-12-31", datetime(2023, 1, 1, tzinfo=timezone.utc), "consumed"),
            ("2023-12-31", "2024-01-01", datetime(2023, 1, 1, tzinfo=timezone.utc), "consumed"),
            ("2026-10-15", "2027-12-31", FREEZE_TIME, "after the UTC freeze date"),
            ("2026-10-14", "2027-12-31", FREEZE_TIME, "after the UTC freeze date"),
        ]:
            with self.subTest(start=start):
                self.protocol["evaluation"].update(start=start, end=end)
                self.now.return_value = clock
                with self.assertRaisesRegex(ValueError, error):
                    self.freeze()
        self.assertFalse((self.root / "records/seal.json").exists())

    def test_access_waits_for_complete_period_and_declared_maturity(self):
        for access in ("2027-12-31", "2028-01-02"):
            self.protocol["evaluation"]["access_not_before"] = access
            with self.assertRaisesRegex(ValueError, "complete evaluation period|maturity delay"):
                self.freeze()

    def test_no_holdout_file_is_opened_before_calendar_gate(self):
        self.freeze()
        self.acquire()
        self.now.return_value = datetime(2028, 1, 7, 23, 59, tzinfo=timezone.utc)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Payload was opened")), \
                patch.object(gate, "_numeric", side_effect=AssertionError("Numeric parse occurred")):
            with self.assertRaisesRegex(ValueError, "calendar gate"):
                gate.load_holdout_csv(self.path, root=self.root)
        self.assertFalse((self.root / "records/access.json").exists())

    def test_input_and_code_changes_are_rejected_before_any_access(self):
        self.freeze()
        self.acquire()
        for name in (gate.GATE_MODULE, "input.csv", "config/models.json"):
            original = (self.root / name).read_bytes()
            (self.root / name).write_bytes(original + b"changed\n")
            with self.subTest(name=name), patch.object(gate, "_numeric") as parse:
                with self.assertRaisesRegex(ValueError, "Sealed file changed"):
                    gate.load_holdout_csv(self.path, root=self.root)
                parse.assert_not_called()
            (self.root / name).write_bytes(original)
        self.assertFalse((self.root / "records/access.json").exists())

    def test_seal_missing_or_extra_dependency_is_rejected(self):
        path = self.freeze()
        original = json.loads(path.read_text())
        for extra in (False, True):
            seal = json.loads(json.dumps(original))
            if extra:
                seal["files"]["unexpected.txt"] = "0" * 64
            else:
                del seal["files"]["input.csv"]
            path.write_text(json.dumps(seal))
            with self.assertRaisesRegex(ValueError, "exact protocol dependency set"):
                gate.verify_protocol(self.path, root=self.root)

    def test_filtered_values_and_unneeded_timestamps_are_never_parsed(self):
        self.freeze()
        self.acquire(
            "2024-06-01,BAD_TIMESTAMP,gas,CONSUMED_UNREAD\n"
            "2028-01-01,BAD_TIMESTAMP,gas,AFTER_END_UNREAD\n"
            "2027-05-01,BAD_TIMESTAMP,other,OTHER_UNREAD\n"
            "2027-05-01,2029-01-01T00:00:00Z,gas,UNAVAILABLE_UNREAD\n"
            "2027-05-01,2027-05-08T12:00:00Z,gas,2.5\n"
            "2027-05-02,2027-05-09T12:00:00Z,gas,\n")
        parse = gate._numeric
        def audited_parse(value):
            self.assertTrue((self.root / "records/access.json").exists())
            self.assertIn(value, ("2.5", ""))
            return parse(value)
        with patch.object(gate, "_numeric", side_effect=audited_parse) as parser:
            result = gate.load_holdout_csv(self.path, root=self.root)
        self.assertEqual(parser.call_count, 2)
        self.assertEqual([row["value"] for row in result["rows"]], [2.5, None])
        self.assertEqual(result["excluded"], {"outside_evaluation": 2, "unselected_series": 1, "not_yet_available": 1})

    def test_reproduction_binds_first_snapshot_and_preserves_access_record(self):
        self.freeze()
        self.acquire("2027-05-01,2027-05-08T12:00:00Z,gas,2.5\n"
                     "2027-05-02,2028-01-20T00:00:00Z,gas,STILL_UNREAD_ON_REPRODUCTION\n")
        first = gate.load_holdout_csv(self.path, root=self.root)
        saved = (self.root / "records/access.json").read_bytes()
        self.now.return_value = datetime(2028, 2, 1, tzinfo=timezone.utc)
        self.assertEqual(first, gate.load_holdout_csv(self.path, root=self.root))
        self.assertEqual(saved, (self.root / "records/access.json").read_bytes())
        self.acquire("2027-05-01,2027-05-08T12:00:00Z,gas,9.0\n")
        with patch.object(gate, "_numeric") as parser:
            with self.assertRaisesRegex(ValueError, "snapshot"):
                gate.load_holdout_csv(self.path, root=self.root)
            parser.assert_not_called()
        self.assertEqual(saved, (self.root / "records/access.json").read_bytes())

    def test_failed_numeric_parse_still_records_consumed_access(self):
        self.freeze()
        self.acquire("2027-05-01,2027-05-08T12:00:00Z,gas,BAD_ELIGIBLE_VALUE\n")
        with self.assertRaises(ValueError):
            gate.load_holdout_csv(self.path, root=self.root)
        self.assertTrue((self.root / "records/access.json").exists())
        (self.root / "records/seal.json").unlink()
        self.now.return_value = FREEZE_TIME
        with self.assertRaisesRegex(ValueError, "Access already recorded"):
            self.freeze()

    def test_parallel_access_publishes_one_complete_audit(self):
        self.freeze()
        self.acquire()
        with ThreadPoolExecutor(max_workers=4) as pool:
            audits = list(pool.map(lambda _: gate.record_first_access(self.path, root=self.root), range(4)))
        self.assertTrue(all(audit == audits[0] for audit in audits))
        self.assertEqual(json.loads((self.root / "records/access.json").read_text()), audits[0])

    def test_family_resolution_accounts_for_minimum_valid_replications(self):
        self.protocol["inference"]["family_sizes"]["comparison"] = 165
        self.protocol["hypotheses"] = [{"id": str(i), "family": "comparison", "statement": f"Synthetic null {i}"} for i in range(165)]
        for replications in (1000, 3299):
            self.protocol["inference"]["bootstrap"]["replications"] = replications
            with self.subTest(replications=replications), self.assertRaisesRegex(ValueError, "resolution"):
                self.freeze()
        self.protocol["inference"]["bootstrap"]["replications"] = 19999
        self.freeze()

    def test_family_size_cannot_hide_undeclared_comparisons(self):
        self.protocol["inference"]["family_sizes"]["comparison"] = 2
        with self.assertRaisesRegex(ValueError, "hypothesis counts"):
            self.freeze()

    def test_draft_incomplete_and_unknown_fields_are_rejected(self):
        self.protocol["status"] = "draft"
        with self.assertRaisesRegex(ValueError, "explicitly ready"):
            self.freeze()
        self.protocol["status"] = "ready"
        self.protocol["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            self.freeze()
        del self.protocol["unexpected"]
        del self.protocol["models"]
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            self.freeze()

    def test_preexisting_holdout_and_overlapping_file_roles_are_rejected(self):
        self.acquire()
        self.now.return_value = FREEZE_TIME
        with self.assertRaisesRegex(ValueError, "acquired after"):
            self.freeze()
        (self.root / "future.csv").unlink()
        self.protocol["holdout"]["path"] = "input.csv"
        with self.assertRaisesRegex(ValueError, "distinct paths"):
            self.freeze()

    def test_path_traversal_absolute_paths_and_symlinks_are_rejected(self):
        for name in ("../outside", "/tmp/outside", "config/../outside", "config//outside"):
            self.protocol["outputs"]["seal"] = name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "canonical relative path"):
                self.freeze()
        self.protocol["outputs"]["seal"] = "records/seal.json"
        (self.root / "linked.csv").symlink_to(self.root / "input.csv")
        self.protocol["files"]["input"] = ["linked.csv"]
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.freeze()
        self.protocol["files"]["input"] = ["input.csv"]
        (self.root / "linked-dir").symlink_to(self.root / "config", target_is_directory=True)
        self.protocol["outputs"]["seal"] = "linked-dir/seal.json"
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.freeze()

    def test_symlink_replacement_after_freeze_is_rejected(self):
        self.freeze()
        (self.root / "future.csv").symlink_to(self.root / "input.csv")
        self.now.return_value = ACCESS_TIME
        with self.assertRaisesRegex(ValueError, "Symlink"):
            gate.load_holdout_csv(self.path, root=self.root)
        self.assertFalse((self.root / "records/access.json").exists())

    def test_json_duplicate_keys_and_nonfinite_settings_are_rejected(self):
        for text in ('{"status":"ready","status":"draft"}', '{"value":NaN}', '{"value":1e999}'):
            self.path.write_text(text)
            with self.assertRaisesRegex(ValueError, "Duplicate JSON|Non-finite JSON"):
                gate.freeze_protocol(self.path, root=self.root)

    def test_bad_access_audit_cannot_be_silently_replaced(self):
        self.freeze()
        self.acquire()
        gate.record_first_access(self.path, root=self.root)
        path = self.root / "records/access.json"
        saved = json.loads(path.read_text())
        saved["first_access_attempt_at_utc"] = "2026-10-15T12:00:00+00:00"
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "First-access audit"):
            gate.load_holdout_csv(self.path, root=self.root)


if __name__ == "__main__":
    unittest.main()
