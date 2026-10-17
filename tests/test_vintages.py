"""Revision leakage, uncertain release days, and lossless archive storage."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from weather_risk.vintages import (
    VersionedObservation, build_vintage_feature_panel, read_archive_csv,
    read_archive_json, select_vintages_as_of, validate_archive,
    write_archive_csv, write_archive_json,
)


def row(**changes):
    record = VersionedObservation(
        series_id="x", observation_date="2024-01-01", value=10.0,
        unit="USD", source="test:x", retrieved_at="2024-06-01T00:00:00Z",
        source_timezone="America/Toronto", availability_basis="release_timestamp",
        effective_at="2024-01-02T17:00:00Z", release_evidence="fixture:release-log",
    )
    return replace(record, **changes)


def day_row(**changes):
    return row(effective_at=None, availability_basis="vintage_date",
               vintage_date="2024-01-04", release_evidence="", **changes)


class VintageSelectionTests(unittest.TestCase):
    def test_exact_release_boundary_and_future_revision_mutation(self):
        old = row()
        future = row(value=20, effective_at="2024-02-02T17:00:00Z")
        self.assertEqual(select_vintages_as_of([old], "2024-01-02T16:59:59Z"), [])
        early = select_vintages_as_of([old, future], "2024-01-02T17:00:00Z")
        self.assertEqual(early[0]["value"], 10)
        self.assertEqual(early, select_vintages_as_of([replace(future, value=999), old], "2024-01-02T17:00:00Z"))
        self.assertEqual(select_vintages_as_of([old, future], "2024-02-02T17:00:00Z")[0]["value"], 20)

    def test_null_revision_never_revives_older_value(self):
        records = [row(), row(value=None, effective_at="2024-01-03T17:00:00Z")]
        selected = select_vintages_as_of(records, "2024-01-03T17:00:00Z")[0]
        self.assertIsNone(selected["value"])
        self.assertEqual(selected["missing_reason"], "source_missing")

    def test_date_only_uncertainty_blocks_older_revision(self):
        records = [row(), day_row(value=12)]
        before = select_vintages_as_of(records, "2024-01-04T04:59:59Z")[0]
        self.assertEqual(before["value"], 10)
        for allow in (False, True):
            selected = select_vintages_as_of(records, "2024-01-04T23:00:00Z", allow_date_only=allow)[0]
            self.assertIsNone(selected["value"])
            self.assertIsNone(selected["available_at"])
            self.assertEqual(selected["missing_reason"], "uncertain_vintage_day")
        strict = select_vintages_as_of(records, "2024-01-05T05:00:00Z")[0]
        self.assertIsNone(strict["value"])
        self.assertEqual(strict["missing_reason"], "unverified_release_time")
        conservative = select_vintages_as_of(records, "2024-01-05T05:00:00Z", allow_date_only=True)[0]
        self.assertEqual(conservative["value"], 12)
        self.assertEqual(conservative["availability_basis"], "vintage_date")
        self.assertEqual(conservative["vintage"], "date_only")
        self.assertEqual(conservative["available_at"], "2024-01-05T05:00:00+00:00")

    def test_date_only_midnight_uses_dst_not_fixed_24_hours(self):
        records = [replace(day_row(), vintage_date="2024-03-10")]
        self.assertIsNone(select_vintages_as_of(records, "2024-03-11T03:59:59Z", allow_date_only=True)[0]["value"])
        self.assertEqual(select_vintages_as_of(records, "2024-03-11T04:00:00Z", allow_date_only=True)[0]["value"], 10)

    def test_date_only_null_and_expired_interval_stay_missing(self):
        missing = select_vintages_as_of([day_row(value=None)], "2024-01-06T05:00:00Z", allow_date_only=True)[0]
        self.assertEqual(missing["missing_reason"], "source_missing")
        bounded = day_row(valid_through_date="2024-01-06")
        self.assertEqual(select_vintages_as_of([bounded], "2024-01-07T04:59:59Z", allow_date_only=True)[0]["value"], 10)
        expired = select_vintages_as_of([bounded], "2024-01-07T05:00:00Z", allow_date_only=True)[0]
        self.assertIsNone(expired["value"])
        self.assertEqual(expired["missing_reason"], "outside_vintage_period")

    def test_newest_observation_then_age_and_missingness(self):
        records = [row(), row(observation_date="2024-01-02", value=None, effective_at="2024-01-03T17:00:00Z")]
        specs = [{"series_id": "x", "timezone": "America/Toronto", "stale_days": 2}]
        panel = build_vintage_feature_panel(records, ["2024-01-03T18:00:00Z"], specs)
        self.assertEqual(panel[0]["observation_date"], "2024-01-02")
        self.assertEqual(panel[0]["missing_reason"], "source_missing")
        late_old_revision = row(value=100, effective_at="2024-01-04T17:00:00Z")
        self.assertEqual(build_vintage_feature_panel(records + [late_old_revision], ["2024-01-04T18:00:00Z"], specs)[0]["observation_date"], "2024-01-02")
        panel = build_vintage_feature_panel([row()], ["2024-01-03T18:00:00Z", "2024-01-04T18:00:00Z"], specs)
        self.assertEqual(panel[0]["value"], 10)
        self.assertEqual(panel[1]["missing_reason"], "stale")

    def test_strict_timestamp_timezone_and_evidence_validation(self):
        for change in ({"effective_at": "2024-01-02T17:00:00"}, {"retrieved_at": "2024-06-01"}, {"source_timezone": "No/SuchZone"}, {"release_evidence": ""}, {"effective_at": "2024-01-02"}, {"vintage_date": "2024-01-02"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_archive([row(**change)])
        with self.assertRaises(ValueError):
            validate_archive([replace(day_row(), effective_at="2024-01-04T17:00:00Z")])
        with self.assertRaises(ValueError):
            select_vintages_as_of([row()], "2024-01-03T17:00:00")

    def test_observation_date_and_retrieval_causality(self):
        for change in ({"observation_date": "2024-01-03"}, {"retrieved_at": "2024-01-02T16:59:59Z"}, {"observation_date": "20240101"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_archive([row(**change)])
        # UTC previous date is valid when the observation's local date has begun.
        valid = row(observation_date="2024-01-02", effective_at="2024-01-01T23:00:00Z", source_timezone="Asia/Tokyo")
        self.assertEqual(len(validate_archive([valid])), 1)

    def test_duplicate_offsets_and_ambiguous_day_overlap_rejected(self):
        for other in (row(), row(value=11), row(effective_at="2024-01-02T12:00:00-05:00")):
            with self.assertRaises(ValueError):
                validate_archive([row(), other])
        with self.assertRaises(ValueError):
            validate_archive([day_row(), row(effective_at="2024-01-04T17:00:00Z")])
        with self.assertRaises(ValueError):
            validate_archive([day_row(valid_through_date="2024-01-10"), replace(day_row(), vintage_date="2024-01-08")])

    def test_conflicting_units_invalid_values_and_bad_specs_rejected(self):
        with self.assertRaises(ValueError):
            validate_archive([row(), row(observation_date="2024-01-02", unit="CAD")])
        for value in (True, float("nan"), float("inf"), "12"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_archive([row(value=value)])
        for specs in ([{"series_id": "y"}], [{"series_id": "x", "stale_days": True}], [{"series_id": "x"}, {"series_id": "x"}]):
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                build_vintage_feature_panel([row()], [], specs)


class VintageArchiveStorageTests(unittest.TestCase):
    def test_roundtrip_and_deterministic_order_and_no_overwrite(self):
        records = [row(), day_row(value=None, quality_flag="missing, revised\nsource flag")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix, write, read in (("json", write_archive_json, read_archive_json), ("csv", write_archive_csv, read_archive_csv)):
                first, second = root / f"first.{suffix}", root / f"second.{suffix}"
                write(first, records)
                write(second, records[::-1])
                self.assertEqual(first.read_bytes(), second.read_bytes())
                self.assertEqual(read(first), validate_archive(records))
                with self.assertRaises(FileExistsError):
                    write(first, records)

    def test_malformed_imports_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.json"
            for payload in ({"schema_version": True, "records": []}, {"schema_version": 1, "records": [None]}, {"schema_version": 1, "records": {}}):
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    read_archive_json(path)
            path.write_text("series_id,value\nx,1\n")
            with self.assertRaises(ValueError):
                read_archive_csv(path)


if __name__ == "__main__":
    unittest.main()
