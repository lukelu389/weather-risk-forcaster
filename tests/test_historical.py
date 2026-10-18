"""Offline source-format fixtures and pinned staging replay, without model fits."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weather_risk import historical as h
from weather_risk.cache import Cache


STAMP = {'retrieved_at': '2026-10-15T12:00:00+00:00'}
STATION = {'series_id': 'test_airport', 'provider': 'noaa_ghcn',
           'source_series': 'USW00094846', 'station_name': 'TEST AIRPORT',
           'timezone': 'Etc/GMT+6', 'lag_days': 2, 'available_hour': 12,
           'required': True}


def dly(element, value, *, year=2008, month=1, changes=None):
    cells = [f'{value:5d}  S'] * 31
    for day, cell in (changes or {}).items():
        cells[day - 1] = cell
    return (f"{STATION['source_series']}{year:04d}{month:02d}{element}" + ''.join(cells)).encode()


def inventory():
    return (f"{STATION['source_series']} {41.9603:8.4f} {-87.9317:9.4f} {204.8:6.1f} IL "
            f"{'TEST AIRPORT':30s}    \n").encode()


def eia_spec(identifier):
    _, _, _, unit, frequency = h.EIA_TABLES[identifier]
    return {'series_id': identifier, 'provider': 'eia_history', 'source_series': 'fixture',
            'unit': unit, 'frequency': frequency, 'timezone': 'America/Toronto',
            'lag_days': 8, 'available_hour': 18}


def table(identifier, rows):
    _, title, units, _, _ = h.EIA_TABLES[identifier]
    cells = ''.join('<tr>' + ''.join(f'<td class="{cls}">{text}</td>' for cls, text in row)
                    + '</tr>' for row in rows)
    return f'<html><title>{title} ({units})</title><table>{cells}</table></html>'.encode()


class SourceFormatTests(unittest.TestCase):
    def parse(self, payload, start='2008-01-01', end='2008-01-02'):
        return {(r.observation_date, r.series_id.removeprefix('test_airport_')): r
                for r in h.parse_ghcn(payload, STATION, start, end, STAMP)}

    def test_fixed_units_midrange_degree_days_and_no_tavg_substitution(self):
        rows = self.parse(b'\n'.join([dly('TMIN', 100), dly('TMAX', 300),
                                     dly('PRCP', 25), dly('TAVG', 999)]))
        self.assertEqual(rows['2008-01-01', 'midrange_temp'].value, 20)
        self.assertEqual(rows['2008-01-01', 'precip'].value, 2.5)
        self.assertEqual(rows['2008-01-01', 'hdd18'].value, 0)
        self.assertEqual(rows['2008-01-01', 'cdd18'].value, 2)
        row = rows['2008-01-01', 'midrange_temp']
        self.assertEqual(row.available_at, '2008-01-03T18:00:00+00:00')
        self.assertEqual((row.vintage, row.availability_basis), ('latest', 'assumed_lag'))

    def test_quality_rejections_and_missing_days_propagate_without_backfill(self):
        rows = self.parse(b'\n'.join([dly('TMIN', 100, changes={1: '  100 QX', 2: '-9999  S'}),
                                     dly('TMAX', 300)]))
        for day in ('2008-01-01', '2008-01-02'):
            for field in ('tmin', 'midrange_temp', 'hdd18', 'cdd18', 'precip'):
                self.assertIsNone(rows[day, field].value)
        self.assertIn('Q=Q', rows['2008-01-01', 'midrange_temp'].quality_flag)
        self.assertEqual(rows['2008-01-01', 'tmax'].value, 30)

    def test_outside_dates_are_filtered_before_numeric_parsing(self):
        poison = dly('TMIN', 0, year=2026, changes={1: 'xxxxx   '})
        rows = self.parse(dly('TMIN', 100, changes={3: 'xxxxx   '}) + b'\n' + poison)
        self.assertEqual(rows['2008-01-01', 'tmin'].value, 10)

    def test_bad_station_length_duplicates_and_impossible_temperatures_fail(self):
        samples = [b'short', dly('TMIN', 100).replace(b'USW00094846', b'USW00000000'),
                   dly('TMIN', 100) + b'\n' + dly('TMIN', 100),
                   dly('TMIN', 300) + b'\n' + dly('TMAX', 100)]
        for sample in samples:
            with self.subTest(sample=sample[:21]), self.assertRaises(ValueError):
                self.parse(sample)

    def test_leap_day_and_explicit_absent_date(self):
        rows = self.parse(dly('TMIN', 100, month=2), '2008-02-29', '2008-03-01')
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows['2008-02-29', 'tmin'].value, 10)
        self.assertIsNone(rows['2008-03-01', 'tmin'].value)

    def test_station_identity_and_unrelated_nonascii_metadata(self):
        raw = b'OTHER000001 CAF\xc9\n' + inventory()
        result = h.station_metadata(raw, STATION['source_series'], 'TEST AIRPORT')
        self.assertAlmostEqual(result['longitude'], -87.9317)
        with self.assertRaises(ValueError):
            h.station_metadata(raw, STATION['source_series'], 'WRONG')
        with self.assertRaises(ValueError):
            h.station_metadata(inventory() * 2, STATION['source_series'], 'TEST AIRPORT')

    def test_eia_month_end_comma_values_and_source_missingness(self):
        identifier = 'us_gas_consumption'
        raw = table(identifier, [[('B4', '2008')] + [('B3', v) for v in ['1,234', 'W'] + ['10'] * 10]])
        rows = h.parse_eia_history(raw, eia_spec(identifier), '2008-01-01', '2008-02-29', STAMP)
        self.assertEqual([(r.observation_date, r.value) for r in rows], [('2008-01-31', 1234), ('2008-02-29', None)])
        self.assertIn('source_missing=W', rows[1].quality_flag)

    def test_eia_weekly_dates_and_missing_values(self):
        identifier = 'us_gas_storage'
        row = [('B6', '2008-Jan'), ('B5', '01/04'), ('B3', '3,000'),
               ('B5', '01/11'), ('B3', '-')] + [('B5', ''), ('B3', '')] * 3
        rows = h.parse_eia_history(table(identifier, [row]), eia_spec(identifier), '2008-01-01', '2008-01-31', STAMP)
        self.assertEqual([(r.observation_date, r.value) for r in rows], [('2008-01-04', 3000), ('2008-01-11', None)])
        for changed in ([('B6', '2008-Feb')] + row[1:], row[:-1], row + row):
            with self.assertRaises(ValueError):
                h.parse_eia_history(table(identifier, [changed]), eia_spec(identifier), '2008-01-01', '2008-12-31', STAMP)

    def test_eia_wrong_units_layout_duplicates_or_absent_range_fail(self):
        identifier = 'us_gas_consumption'
        row = [('B4', '2008')] + [('B3', '1')] * 12
        for raw in (table(identifier, [row]).replace(b'Million', b'Billion'),
                    table(identifier, [row, row]), table(identifier, [row[:-1]]),
                    table(identifier, [[('B4', '2009')] + row[1:]])):
            with self.assertRaises(ValueError):
                h.parse_eia_history(raw, eia_spec(identifier), '2008-01-01', '2008-12-31', STAMP)

    def test_coverage_exposes_absent_2008_storage(self):
        spec = eia_spec('us_gas_storage')
        rows = [h.observation(spec, '2014-01-03', 3000, STAMP)]
        result = {r['period']: r for r in h.coverage(rows, [spec], '2000-01-01', '2025-12-31')}
        self.assertEqual(result['2008']['finite_rows'], 0)
        self.assertEqual(result['2008']['first'], '')
        self.assertEqual(result['2014-2016']['finite_rows'], 1)
        self.assertEqual(result['2014-2016']['expected_periods_approx'], 157)


class HistoricalRunTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = self.root / 'history.json'
        self.spec = {'series_id': 'price', 'provider': 'fixture', 'source_series': 'price',
                     'unit': 'USD', 'frequency': 'business_daily', 'timezone': 'UTC',
                     'lag_days': 1, 'available_hour': 18, 'positive': True, 'required': True}
        self.settings = {'start': '2008-01-01', 'end': '2008-01-31', 'market_series': [self.spec],
                         'weather_stations': [STATION], 'eia_history_series': []}
        self.config.write_text(json.dumps(self.settings))
        self.cache = Cache(self.root / 'data/extensions/raw')
        self.url = 'https://fixture.invalid/price'
        for url, payload in [(self.url, b'10'), (h.NOAA + '/ghcnd-stations.txt', inventory()),
                             (h.NOAA + '/all/USW00094846.dly', b'\n'.join([dly('TMIN', 100), dly('TMAX', 300), dly('PRCP', 25)]))]:
            self.cache.get_or_create({'url': url}, lambda payload=payload: payload)
        def fetch(spec, start, end, cache):
            payload, meta = cache.get(self.url)
            return [h.observation(spec, '2008-01-02', float(payload), meta)]
        self.adapter = patch.object(h, 'fetch_series', side_effect=fetch)
        self.adapter.start()
        self.addCleanup(self.adapter.stop)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def run_stage(self):
        return h.run_history(self.config, output_root=self.root, offline=True)

    def test_replay_pins_sources_despite_changed_request_pointers(self):
        original = self.run_stage()
        changed = Cache(self.cache.root, refresh=True)
        changed.get_or_create({'url': self.url}, lambda: b'99')
        replay = h.replay_history(original / 'manifest.json', output_root=self.root)
        self.assertNotEqual(replay, original)
        self.assertEqual((original / 'manifest.json').read_bytes(), (replay / 'manifest.json').read_bytes())
        self.assertEqual(json.loads((original / 'manifest.json').read_text())['status'], 'passed')

    def test_data_rejection_is_saved_published_and_reproducible(self):
        changed = Cache(self.cache.root, refresh=True)
        changed.get_or_create({'url': self.url}, lambda: b'0')
        original = self.run_stage()
        manifest = json.loads((original / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'validation_failed')
        self.assertFalse(json.loads((original / 'validation.json').read_text())['passed'])
        self.assertIn('FAILED', (self.root / 'reports/extensions/history_report.md').read_text())
        replay = h.replay_history(original / 'manifest.json', output_root=self.root)
        self.assertEqual((original / 'manifest.json').read_bytes(), (replay / 'manifest.json').read_bytes())
        with patch.object(h, 'run_history', return_value=original):
            self.assertEqual(h.main([]), 1)

    def test_corrupt_saved_artifact_or_changed_code_prevents_replay(self):
        original = self.run_stage()
        with patch.object(h, 'sha', return_value='changed'), self.assertRaisesRegex(ValueError, 'artifact changed'):
            h.replay_history(original / 'manifest.json', output_root=self.root)
        document = json.loads((original / 'manifest.json').read_text())
        document['code_sha256']['historical.py'] = 'changed'
        (original / 'manifest.json').write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            h.replay_history(original / 'manifest.json', output_root=self.root)

    def test_acquisition_failure_preserves_manifest_and_prior_report(self):
        self.run_stage()
        prior = (self.root / 'reports/extensions/manifest.json').read_bytes()
        with patch.object(h, 'fetch_series', side_effect=ValueError('source fixture failure')):
            with self.assertRaisesRegex(ValueError, 'source fixture failure'):
                self.run_stage()
        failed = sorted((self.root / 'runs').iterdir())[-1]
        manifest = json.loads((failed / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertIn('source fixture failure', manifest['failure'])
        self.assertEqual(prior, (self.root / 'reports/extensions/manifest.json').read_bytes())

    def test_future_range_rejected_before_acquisition(self):
        self.settings['end'] = '2026-01-01'
        self.config.write_text(json.dumps(self.settings))
        with patch.object(h, 'fetch_series', side_effect=AssertionError('source accessed')):
            with self.assertRaisesRegex(ValueError, 'already-exposed history'):
                self.run_stage()


if __name__ == '__main__':
    unittest.main()
