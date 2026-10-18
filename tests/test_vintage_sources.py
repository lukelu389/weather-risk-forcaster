"""ALFRED schema/pagination fixtures; no live credentials or requests."""

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weather_risk.cache import Cache
from weather_risk.vintage_sources import fetch_alfred, main, parse_alfred_observations
from weather_risk.vintages import read_archive_csv, read_archive_json, select_vintages_as_of, write_archive_json


SPEC = {
    "series_id": "henry_hub_spot", "source_series": "DHHNGSP", "unit": "USD/MMBtu",
    "source_unit": "Dollars per Million BTU", "frequency": "daily", "timezone": "America/Chicago",
}
START, END = "2024-01-01", "2024-03-01"
STAMP = "2024-06-01T00:00:00Z"
ROWS = [
    {"date": "2024-01-01", "value": "2.1", "realtime_start": "2024-01-03", "realtime_end": "2024-02-01"},
    {"date": "2024-01-01", "value": ".", "realtime_start": "2024-02-02", "realtime_end": END},
    {"date": "2024-01-02", "value": "2.2", "realtime_start": "2024-01-04", "realtime_end": END},
]


def payload(rows=None, **changes):
    return {"output_type": 1, "units": "lin", "observations": ROWS if rows is None else rows, **changes}


class FakeCache:
    offline = False

    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    def get(self, url, params):
        self.calls.append((url, params))
        if url.endswith("/series"):
            result = {"seriess": [{"id": SPEC["source_series"], "units": SPEC["source_unit"], "frequency_short": "D"}]}
        else:
            offset, limit = params["offset"], params["limit"]
            result = payload(ROWS[offset:offset + limit], count=len(ROWS), offset=offset, limit=limit,
                             realtime_start=START, realtime_end=END, observation_start=START,
                             observation_end="2024-01-02", sort_order="asc")
        if self.mutate:
            result = self.mutate(url, params, result)
        return json.dumps(result).encode(), {"retrieved_at": STAMP}


class AlfredSourceTests(unittest.TestCase):
    def test_parser_preserves_missing_revisions_and_validity(self):
        rows = parse_alfred_observations(payload(), SPEC, STAMP)
        self.assertEqual(len(rows), 3)
        self.assertIsNone(rows[1].value)
        self.assertEqual(rows[1].valid_through_date, END)
        self.assertTrue(all(row.availability_basis == "vintage_date" and row.effective_at is None for row in rows))
        selected = select_vintages_as_of(rows, "2024-02-03T06:00:00Z", allow_date_only=True)
        self.assertIsNone(selected[0]["value"])
        self.assertEqual(selected[0]["missing_reason"], "source_missing")
        self.assertEqual(select_vintages_as_of(rows, "2024-03-02T06:00:00Z", allow_date_only=True)[0]["missing_reason"], "outside_vintage_period")

    def test_monthly_period_normalization(self):
        spec = {**SPEC, "frequency": "monthly"}
        raw = {"date": "2024-01-01", "value": "3", "realtime_start": "2024-02-15", "realtime_end": END}
        self.assertEqual(parse_alfred_observations(payload([raw]), spec, STAMP)[0].observation_date, "2024-01-31")

    def test_malformed_transform_and_revision_conflicts_fail(self):
        for document in (payload(output_type=2), payload(units="chg"), payload(output_type=True), payload([ROWS[0], ROWS[0]]), payload([{**ROWS[0], "value": "NaN"}]), payload([{**ROWS[0], "value": True}]), payload([{**ROWS[0], "realtime_end": "2024-01-01"}])):
            with self.subTest(document=document), self.assertRaises(ValueError):
                parse_alfred_observations(document, SPEC, STAMP)

    @patch.dict("os.environ", {"FRED_API_KEY": "fixture-key"}, clear=True)
    def test_pagination_requests_all_revisions_and_explicit_ranges(self):
        cache = FakeCache()
        rows = fetch_alfred(SPEC, START, "2024-01-02", START, END, cache, page_size=2)
        self.assertEqual(len(rows), 3)
        self.assertEqual([params["offset"] for url, params in cache.calls if url.endswith("/observations")], [0, 2])
        for _, params in cache.calls[1:]:
            self.assertEqual(params["output_type"], 1)
            self.assertEqual(params["units"], "lin")
            self.assertEqual(params["realtime_start"], START)
            self.assertEqual(params["realtime_end"], END)

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_live_credential_fails_before_request(self):
        cache = FakeCache()
        with self.assertRaisesRegex(ValueError, "FRED_API_KEY"):
            fetch_alfred(SPEC, START, "2024-01-02", START, END, cache)
        self.assertEqual(cache.calls, [])

    @patch.dict("os.environ", {}, clear=True)
    def test_offline_replay_does_not_need_credential(self):
        cache = FakeCache()
        cache.offline = True
        self.assertEqual(len(fetch_alfred(SPEC, START, "2024-01-02", START, END, cache)), 3)

    @patch.dict("os.environ", {"FRED_API_KEY": "fixture-key"}, clear=True)
    def test_inconsistent_pagination_and_metadata_fail(self):
        mutations = [
            lambda url, params, result: {**result, "count": 4} if params.get("offset") == 2 else result,
            lambda url, params, result: {**result, "offset": 0} if params.get("offset") == 2 else result,
            lambda url, params, result: {**result, "observations": []} if url.endswith("/observations") else result,
            lambda url, params, result: {**result, "realtime_start": "2024-01-02"} if url.endswith("/observations") else result,
            lambda url, params, result: {"seriess": [{"id": "WRONG", "units": SPEC["source_unit"], "frequency_short": "D"}]} if url.endswith("/series") else result,
            lambda url, params, result: {"seriess": [{"id": SPEC["source_series"], "units": "wrong", "frequency_short": "D"}]} if url.endswith("/series") else result,
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                fetch_alfred(SPEC, START, "2024-01-02", START, END, FakeCache(mutate), page_size=2)

    @patch.dict("os.environ", {"FRED_API_KEY": "fixture-key"}, clear=True)
    def test_duplicate_cross_page_revision_fails(self):
        def duplicate(url, params, result):
            if params.get("offset") == 2:
                result["observations"] = [ROWS[0]]
            return result
        with self.assertRaises(ValueError):
            fetch_alfred(SPEC, START, "2024-01-02", START, END, FakeCache(duplicate), page_size=2)

    def test_invalid_fetch_options_fail_without_network(self):
        cache = FakeCache()
        for page_size in (0, 100001, True, 1.5):
            with self.assertRaises(ValueError):
                fetch_alfred(SPEC, START, "2024-01-02", START, END, cache, page_size=page_size)
        with self.assertRaises(ValueError):
            fetch_alfred(SPEC, END, START, START, END, cache)
        self.assertEqual(cache.calls, [])


class VintageCommandTests(unittest.TestCase):
    def test_fetch_command_binds_output_source_code_and_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = root / 'spec.json'
            spec.write_text(json.dumps(SPEC))
            output = root / 'archive.json'
            rows = parse_alfred_observations(payload(), SPEC, STAMP)
            args = ['fetch-alfred', '--spec', str(spec), '--output', str(output),
                    '--cache', str(root / 'cache'), '--observation-start', START,
                    '--observation-end', '2024-01-02', '--realtime-start', START,
                    '--realtime-end', END, '--offline']
            with patch('weather_risk.vintage_sources.fetch_alfred', return_value=rows):
                self.assertEqual(main(args), 0)
            manifest = json.loads(output.with_suffix('.json.manifest.json').read_text())
            self.assertEqual(manifest['archive_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(manifest['requested_ranges']['realtime_end'], END)
            self.assertEqual(set(manifest['code_sha256']), {'vintage_sources.py', 'vintages.py', 'cache.py'})
            self.assertFalse(manifest['actual_release_times_verified'])
            with self.assertRaises(FileExistsError):
                main(args)

    def test_import_and_select_commands_preserve_strict_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, converted = root / "archive.json", root / "archive.csv"
            rows = parse_alfred_observations(payload(), SPEC, STAMP)
            write_archive_json(source, rows)
            self.assertEqual(main(["import", "--input", str(source), "--output", str(converted)]), 0)
            self.assertEqual(read_archive_csv(converted), read_archive_json(source))
            selected = root / "selected.json"
            self.assertEqual(main(["select", "--archive", str(converted), "--forecast-at", "2024-01-05T06:00:00Z", "--output", str(selected)]), 0)
            self.assertTrue(all(record["value"] is None for record in json.loads(selected.read_text())))
            with self.assertRaises(FileExistsError):
                main(["import", "--input", str(source), "--output", str(converted)])


if __name__ == "__main__":
    unittest.main()
