"""Synthetic contract-level accounting; these fixtures are not historical trades."""

from copy import deepcopy
from dataclasses import replace
import json
import math
import random
import unittest

from weather_risk.execution import (
    FuturesContract, FuturesExecution, SettlementBatch, account_futures,
)


CONTRACT = FuturesContract("SYNTHETIC-NEAR", "USD", 10.)


def trade(quantity, price, *, contract_id=CONTRACT.contract_id, reference=None, commission=0., fees=0.):
    return FuturesExecution(contract_id, quantity, price, price if reference is None else reference,
                            commission, fees)


def batch(day, mark, executions=(), *, fx=1.25):
    return SettlementBatch(f"2026-01-{day:02d}T21:00:00Z", executions,
                           {CONTRACT.contract_id: mark}, {"CAD": 1., "USD": fx})


class FuturesAccountingTests(unittest.TestCase):
    def test_long_sizing_settlement_cash_and_fx_translation_reconcile(self):
        batches = [
            batch(1, 105, [trade(2, 100, commission=3, fees=1)], fx=1.25),
            batch(2, 103, fx=1.5),
            batch(3, 104, [trade(-2, 104, commission=2, fees=1)], fx=1.4),
            SettlementBatch("2026-01-04T21:00:00Z", [], {}, {"CAD": 1., "USD": 2.}),
        ]
        result = account_futures([CONTRACT], batches, initial_cash_cad=1000)
        first, second, third, fourth = result["rows"]
        self.assertEqual(first["futures_pnl_cad"], 125)
        self.assertEqual(first["net_settlement_pnl_cad"], 120)
        self.assertEqual(first["cash_by_currency"], {"CAD": 1000., "USD": 96.})
        self.assertEqual(second["fx_translation_pnl_cad"], 24)
        self.assertEqual(second["net_settlement_pnl_cad"], -60)
        self.assertEqual(second["cash_equity_cad"], 1084)
        self.assertAlmostEqual(third["cash_equity_cad"], 1102.2)
        self.assertEqual(fourth["net_settlement_pnl_cad"], 0)
        self.assertAlmostEqual(fourth["fx_translation_pnl_cad"], 43.8)
        self.assertEqual(result["ending_cash_by_currency"], {"CAD": 1000., "USD": 73.})
        self.assertEqual(result["ending_positions"], {})
        self.assertEqual(result["ending_cash_equity_cad"], 1146)
        self.assertAlmostEqual(result["total_pnl_cad"], 146)
        self.assertAlmostEqual(result["reconciliation_error_cad"], 0, places=10)

    def test_slippage_is_attribution_not_a_second_charge(self):
        result = account_futures([CONTRACT], [
            batch(1, 105, [trade(2, 101, reference=100, commission=3, fees=1)], fx=1),
            batch(2, 105, [trade(-2, 104, reference=105, commission=5, fees=2)], fx=1),
        ], initial_cash_cad=1000)
        first, second = result["rows"]
        self.assertEqual(first["reference_pnl_cad"], 100)
        self.assertEqual(first["slippage_cad"], 20)
        self.assertEqual(first["futures_pnl_cad"], 80)
        self.assertEqual(first["net_settlement_pnl_cad"], 76)
        self.assertEqual(second["slippage_cad"], 20)
        self.assertEqual(second["futures_pnl_cad"], -20)
        self.assertEqual(result["total_pnl_cad"], 49)
        for row in result["rows"]:
            self.assertEqual(row["reference_pnl_cad"] - row["slippage_cad"], row["futures_pnl_cad"])

    def test_favorable_fill_produces_negative_slippage(self):
        row = account_futures([CONTRACT], [batch(1, 100, [trade(1, 99, reference=100)], fx=1)],
                              initial_cash_cad=0)["rows"][0]
        self.assertEqual(row["slippage_cad"], -10)
        self.assertEqual(row["futures_pnl_cad"], 10)

    def test_roll_has_explicit_legs_and_no_cross_contract_price_jump(self):
        far = FuturesContract("SYNTHETIC-FAR", "USD", 10.)
        result = account_futures([CONTRACT, far], [
            batch(1, 100, [trade(2, 100)], fx=1),
            SettlementBatch("2026-01-02T21:00:00Z", [trade(-2, 105), trade(2, 200, contract_id=far.contract_id)],
                            {CONTRACT.contract_id: 105, far.contract_id: 200}, {"CAD": 1., "USD": 1.}),
            SettlementBatch("2026-01-03T21:00:00Z", [], {far.contract_id: 190}, {"CAD": 1., "USD": 1.}),
        ], initial_cash_cad=1000)
        self.assertEqual(result["rows"][1]["futures_pnl_cad"], 100)
        self.assertEqual(result["rows"][2]["futures_pnl_cad"], -200)
        self.assertEqual(result["ending_positions"], {far.contract_id: 2})
        self.assertEqual(result["ending_cash_equity_cad"], 900)

    def test_negative_prices_and_short_position_use_differences(self):
        result = account_futures([CONTRACT], [
            batch(1, -20, [trade(-3, -10)], fx=1),
            batch(2, -15, [trade(3, -15)], fx=1),
        ], initial_cash_cad=0)
        self.assertEqual([row["futures_pnl_cad"] for row in result["rows"]], [300, -150])
        self.assertEqual(result["total_pnl_cad"], 150)

    def test_zero_price_and_sign_flip_preserve_same_contract_basis(self):
        result = account_futures([CONTRACT], [
            batch(1, 0, [trade(2, 0)], fx=1),
            batch(2, 2, [trade(-3, 1)], fx=1),
            batch(3, 3, fx=1),
        ], initial_cash_cad=0)
        self.assertEqual([row["futures_pnl_cad"] for row in result["rows"]], [0, 10, -10])
        self.assertEqual(result["ending_positions"], {CONTRACT.contract_id: -1})
        self.assertEqual(result["total_pnl_cad"], 0)

    def test_multiple_currencies_and_distinct_contract_multipliers(self):
        cad = FuturesContract("SYNTHETIC-CAD", "CAD", 100.)
        event = SettlementBatch("2026-01-01T12:00:00-05:00", [
            trade(2, 100), trade(-1, 50, contract_id=cad.contract_id),
        ], {CONTRACT.contract_id: 101, cad.contract_id: 49}, {"CAD": 1., "USD": 1.4})
        result = account_futures([CONTRACT, cad], [event], initial_cash_cad=1000)
        row = result["rows"][0]
        self.assertEqual(row["timestamp"], "2026-01-01T17:00:00Z")
        self.assertEqual(row["cash_by_currency"], {"CAD": 1100., "USD": 20.})
        self.assertEqual(row["cash_equity_cad"], 1128)

    def test_intraday_checkpoints_are_explicit_and_deterministic(self):
        first = batch(1, 100, [trade(1, 100)])
        first = replace(first, timestamp="2026-01-01T15:00:00Z")
        second = replace(batch(1, 101), timestamp="2026-01-01T15:01:00Z")
        original = deepcopy([first, second])
        left = account_futures([CONTRACT], [first, second], initial_cash_cad=1000)
        right = account_futures([CONTRACT], iter([first, second]), initial_cash_cad=1000)
        self.assertEqual(left, right)
        self.assertEqual([first, second], original)
        json.dumps(left, allow_nan=False)

    def test_losses_may_exceed_cash_without_an_inferred_margin_call(self):
        result = account_futures([CONTRACT], [batch(1, -100, [trade(1, 100)], fx=1)], initial_cash_cad=10)
        self.assertEqual(result["ending_cash_equity_cad"], -1990)

    def test_closed_book_matches_independent_trade_cashflow_identity(self):
        # The closing identity depends only on execution cash flows and costs,
        # not on any intermediate settlement marks or their path.
        rng = random.Random(1729)
        events, signed_purchase_cost, charges, position = [], 0., 0., 0
        for index in range(20):
            quantity = rng.choice([-3, -2, -1, 1, 2, 3])
            fill, mark = rng.uniform(-80, 80), rng.uniform(-80, 80)
            fee = rng.uniform(0, 5)
            position += quantity
            signed_purchase_cost += quantity * CONTRACT.multiplier * fill
            charges += fee
            events.append(replace(batch(1, mark, [trade(quantity, fill, fees=fee)], fx=1.37),
                                  timestamp=f"2026-01-01T15:{index:02d}:00Z"))
        if position:
            signed_purchase_cost -= position * CONTRACT.multiplier * 25
            events.append(replace(batch(1, 10, [trade(-position, 25)], fx=1.37),
                                  timestamp="2026-01-01T16:00:00Z"))
        result = account_futures([CONTRACT], events, initial_cash_cad=1000)
        self.assertEqual(result["ending_positions"], {})
        self.assertAlmostEqual(result["total_pnl_cad"], (-signed_purchase_cost - charges) * 1.37, places=8)
        self.assertAlmostEqual(result["reconciliation_error_cad"], 0, places=8)


class FuturesInputValidationTests(unittest.TestCase):
    def setUp(self):
        self.valid = batch(1, 100, [trade(1, 100)])

    def run_batches(self, batches, contracts=None, initial_cash_cad=1000):
        return account_futures([CONTRACT] if contracts is None else contracts, batches,
                               initial_cash_cad=initial_cash_cad)

    def test_missing_naive_and_duplicate_absolute_timestamps_fail(self):
        for timestamp in (None, "", "2026-01-01", "2026-01-01T21:00:00", "invalid"):
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, "timestamp"):
                self.run_batches([replace(self.valid, timestamp=timestamp)])
        for timestamp in ("2026-01-01T16:00:00-05:00", "2025-12-31T21:00:00Z"):
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, "strictly increasing"):
                self.run_batches([self.valid, replace(batch(2, 100), timestamp=timestamp)])

    def test_missing_fx_including_flat_foreign_cash_fails(self):
        for fx in ({}, {"CAD": 1.}, {"USD": 1.25}):
            with self.subTest(fx=fx), self.assertRaisesRegex(ValueError, "missing FX"):
                self.run_batches([replace(self.valid, fx_cad_per_unit=fx)])
        flat = batch(2, 101, [trade(-1, 101)])
        no_usd = SettlementBatch("2026-01-03T21:00:00Z", [], {}, {"CAD": 1.})
        with self.assertRaisesRegex(ValueError, "missing FX"):
            self.run_batches([self.valid, flat, no_usd])

    def test_invalid_fx_orientation_and_values_fail(self):
        for fx in ({"CAD": .8, "USD": 1.25}, {"CAD": 1., "USD": 0},
                   {"CAD": 1., "USD": -1}, {"CAD": 1., "USD": math.nan},
                   {"CAD": 1., "USD": "1.25"}, {"CAD": 1., "USD": True}):
            with self.subTest(fx=fx), self.assertRaises(ValueError):
                self.run_batches([replace(self.valid, fx_cad_per_unit=fx)])

    def test_missing_or_unknown_settlements_fail_even_on_close(self):
        for marks in ({}, {CONTRACT.contract_id: 100, "UNKNOWN": 100}, {CONTRACT.contract_id: math.nan}):
            with self.subTest(marks=marks), self.assertRaises(ValueError):
                self.run_batches([replace(self.valid, settlements=marks)])
        with self.assertRaisesRegex(ValueError, "including closes"):
            self.run_batches([self.valid, replace(batch(2, 100, [trade(-1, 100)]), settlements={})])

    def test_invalid_contracts_and_duplicate_specifications_fail(self):
        for contracts in ([], [CONTRACT, CONTRACT], [replace(CONTRACT, multiplier=0)],
                          [replace(CONTRACT, multiplier=-1)], [replace(CONTRACT, multiplier=math.inf)],
                          [replace(CONTRACT, contract_id="")], [replace(CONTRACT, currency="usd")],
                          [replace(CONTRACT, currency="US")]):
            with self.subTest(contracts=contracts), self.assertRaises(ValueError):
                self.run_batches([self.valid], contracts=contracts)

    def test_missing_costs_prices_and_invalid_quantities_fail(self):
        valid_trade = trade(1, 100)
        invalids = {"quantity": [0, 1.5, True, "1"], "fill_price": [None, True, math.inf],
                    "reference_price": [None, math.nan], "commission": [None, -1, math.inf],
                    "fees": [None, -1, math.nan]}
        for field, values in invalids.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.run_batches([replace(self.valid, executions=[replace(valid_trade, **{field: value})])])

    def test_duplicate_or_unknown_executions_fail(self):
        for executions in ([trade(1, 100), trade(-1, 100)], [trade(1, 100, contract_id="UNKNOWN")]):
            with self.subTest(executions=executions), self.assertRaises(ValueError):
                self.run_batches([replace(self.valid, executions=executions)])

    def test_empty_batches_invalid_cash_and_numeric_overflow_fail(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.run_batches([])
        for cash in (-1, math.inf, math.nan, None, True):
            with self.subTest(cash=cash), self.assertRaises(ValueError):
                self.run_batches([self.valid], initial_cash_cad=cash)
        with self.assertRaisesRegex(ValueError, "finite numeric range"):
            self.run_batches([batch(1, 1e308, [trade(1, -1e308)])])


if __name__ == "__main__":
    unittest.main()
