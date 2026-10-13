import json
import os
import unittest
from unittest.mock import patch

from weather_risk.providers import ProviderError, fetch_series


class FixtureCache:
    offline = False

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        value = self.payloads.pop(0)
        if isinstance(value, dict):
            value = json.dumps(value)
        return value.encode(), {"retrieved_at": "2026-01-01T00:00:00+00:00", "sha256": "fixture"}

    def get_or_create(self, key, fetch):
        return self.get("fixture://yahoo", key)


def spec(provider, series="test", field="value", unit="percent", frequency="daily"):
    return {"series_id": "test_series", "provider": provider, "source_series": series,
            "source_field": field, "unit": unit, "timezone": "America/Toronto",
            "lag_days": 1, "available_hour": 8, "frequency": frequency}


class ProviderTests(unittest.TestCase):
    def test_fred_missing_values_and_inclusive_bounds(self):
        cache = FixtureCache("observation_date,DHHNGSP\n2024-01-01,2.5\n2024-01-02,.\n2024-01-03,3.0\n")
        rows = fetch_series(spec("fred", "DHHNGSP", unit="USD/MMBtu"),
                            "2024-01-01", "2024-01-02", cache)
        self.assertEqual([r.value for r in rows], [2.5, None])
        self.assertEqual(rows[0].available_at, "2024-01-02T13:00:00+00:00")
        self.assertEqual(rows[0].availability_basis, "assumed_lag")
        self.assertEqual(rows[0].vintage, "latest")

    def test_monthly_fred_lag_begins_after_reference_month(self):
        cache = FixtureCache("DATE,CPIAUCSL\n2024-01-01,300\n")
        rows = fetch_series(spec("fred", "CPIAUCSL", unit="index_1982_84_100", frequency="monthly"),
                            "2024-01-15", "2024-01-31", cache)
        self.assertEqual(cache.calls[0][1]["cosd"], "2024-01-01")
        self.assertEqual(rows[0].observation_date, "2024-01-31")
        self.assertEqual(rows[0].available_at, "2024-02-01T13:00:00+00:00")

    def test_fred_malformed_values_and_wrong_units_fail(self):
        for value in ("unknown", "inf", "nan"):
            with self.subTest(value=value), self.assertRaises(ProviderError):
                fetch_series(spec("fred", "DHHNGSP", unit="USD/MMBtu"), "2024-01-01", "2024-01-01",
                             FixtureCache(f"DATE,DHHNGSP\n2024-01-01,{value}\n"))
        with self.assertRaisesRegex(ProviderError, "unit"):
            fetch_series(spec("fred", "DHHNGSP", unit="CAD"), "2024-01-01", "2024-01-01",
                         FixtureCache("DATE,DHHNGSP\n2024-01-01,3\n"))

    def test_duplicate_empty_and_unexpected_headers_fail(self):
        payloads = ["DATE,test\n2024-01-01,1\n2024-01-01,2\n", "DATE,test\n",
                    "DATE,test\n2024-01-01,.\n", "<html>Service unavailable</html>"]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ProviderError):
                fetch_series(spec("fred"), "2024-01-01", "2024-01-02", FixtureCache(payload))

    def test_boc_fx_direction_and_source_metadata(self):
        payload = {"seriesDetail": {"FXUSDCAD": {"label": "USD/CAD"}},
                   "observations": [{"d": "2024-01-02", "FXUSDCAD": {"v": "1.33"}}]}
        rows = fetch_series(spec("boc", "FXUSDCAD", unit="CAD/USD"),
                            "2024-01-01", "2024-01-03", FixtureCache(payload))
        self.assertEqual(rows[0].value, 1.33)
        with self.assertRaisesRegex(ProviderError, "unit"):
            fetch_series(spec("boc", "FXUSDCAD", unit="USD/CAD"),
                         "2024-01-01", "2024-01-03", FixtureCache(payload))

    def test_boc_missing_required_value_is_not_silently_dropped(self):
        payload = {"seriesDetail": {"test": {}}, "observations": [{"d": "2024-01-02"}]}
        with self.assertRaisesRegex(ProviderError, "missing requested value"):
            fetch_series(spec("boc"), "2024-01-01", "2024-01-03", FixtureCache(payload))

    def test_eccc_degree_days_preserve_estimated_missing_and_trace_flags(self):
        payload = ('Station Name,Example\n"Date/Time","Mean Temp (°C)","Mean Temp Flag",'
                   '"Total Precip (mm)","Total Precip Flag"\n'
                   '2024-01-01,-5,E,0,T\n2024-01-02,,M,,M\n2024-01-03,20,,1.2,\n')
        rows = fetch_series(spec("eccc", "12345", "HDD18", "degC_day"),
                            "2024-01-01", "2024-01-03", FixtureCache(payload))
        self.assertEqual([r.value for r in rows], [23.0, None, 0.0])
        self.assertEqual([r.quality_flag for r in rows], ["E", "M", ""])
        cool = fetch_series(spec("eccc", "12345", "CDD18", "degC_day"),
                            "2024-01-01", "2024-01-03", FixtureCache(payload))
        self.assertEqual([r.value for r in cool], [0.0, None, 2.0])
        rain = fetch_series(spec("eccc", "12345", "Total Precip (mm)", "mm"),
                            "2024-01-01", "2024-01-03", FixtureCache(payload))
        self.assertEqual(rain[0].value, 0)
        self.assertEqual(rain[0].quality_flag, "T")

    def test_eccc_rejects_wrong_year(self):
        with self.assertRaisesRegex(ProviderError, "unexpected year"):
            fetch_series(spec("eccc", "12345", "Mean Temp (°C)", "°C"),
                         "2024-01-01", "2024-01-03",
                         FixtureCache('Date/Time,Mean Temp (°C)\n2023-01-01,1\n'))

    def test_eccc_station_identity_rejects_wrong_location(self):
        station = {**spec("eccc", "31688", "Mean Temp (°C)", "degC"), "station_name": "TORONTO CITY"}
        with self.assertRaisesRegex(ProviderError, "station name"):
            fetch_series(station, "2024-01-01", "2024-01-01", FixtureCache(
                'Date/Time,Station Name,Mean Temp (°C)\n2024-01-01,OTHER CITY,1\n'))

    @patch.dict(os.environ, {"EIA_API_KEY": "unit-test-key"})
    def test_eia_paginates_and_checks_units(self):
        cache = FixtureCache(
            {"response": {"total": "2", "data": [
                {"period": "2024-01-05", "value": "3000", "units": "Billion Cubic Feet"}]}},
            {"response": {"total": "2", "data": [
                {"period": "2024-01-12", "value": "2900", "units": "Billion Cubic Feet"}]}})
        rows = fetch_series(spec("eia", "NG.NW2_EPG0_SWO_R48_BCF.W", unit="Bcf"),
                            "2024-01-01", "2024-01-12", cache)
        self.assertEqual([r.value for r in rows], [3000, 2900])
        self.assertEqual([p["offset"] for _, p in cache.calls], [0, 1])

    @patch.dict(os.environ, {"EIA_API_KEY": "unit-test-key"})
    def test_eia_rejects_truncation_and_incorrect_units(self):
        with self.assertRaisesRegex(ProviderError, "truncated"):
            fetch_series(spec("eia"), "2024-01-01", "2024-01-12",
                         FixtureCache({"response": {"total": "2", "data": []}}))
        with self.assertRaisesRegex(ProviderError, "unit"):
            fetch_series(spec("eia", unit="USD/barrel"), "2024-01-01", "2024-01-12",
                         FixtureCache({"response": {"total": "1", "data": [
                             {"period": "2024-01-05", "value": "3000", "units": "Billion Cubic Feet"}]}}))

    def test_yahoo_cache_retains_requested_price_and_currency(self):
        payload = {"currency": "CAD", "rows": [
            {"date": "2024-01-02", "Close": 50, "Adj Close": 48}]}
        cache = FixtureCache(payload)
        rows = fetch_series(spec("yahoo", "ENB.TO", "Adj Close", "CAD/share"),
                            "2024-01-01", "2024-01-02", cache)
        self.assertEqual(rows[0].value, 48)
        options = cache.calls[0][1]["options"]
        self.assertEqual(options["end"], "2024-01-03")
        self.assertFalse(options["auto_adjust"])
        self.assertFalse(options["repair"])
        with self.assertRaisesRegex(ProviderError, "currency"):
            fetch_series(spec("yahoo", "ENB.TO", "Adj Close", "USD/share"),
                         "2024-01-01", "2024-01-02", FixtureCache(payload))

    def test_yahoo_never_substitutes_close_for_missing_adjusted_close(self):
        with self.assertRaisesRegex(ProviderError, "price field"):
            fetch_series(spec("yahoo", "ENB.TO", "Adj Close", "CAD/share"),
                         "2024-01-01", "2024-01-02", FixtureCache(
                             {"currency": "CAD", "rows": [{"date": "2024-01-02", "Close": 50}]}))


if __name__ == "__main__":
    unittest.main()
