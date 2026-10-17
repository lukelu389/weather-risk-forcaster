# Contract-level futures and FX accounting inputs

`weather_risk.execution.account_futures` is a standalone accounting component.
It accepts explicit contracts and cash-settlement batches and returns an audited
CAD ledger. It does not consume or change the sealed Phase 5 forecasts or the
Phase 6 fixed-notional equity sensitivity. The tests contain **synthetic inputs
only**; this is not an empirical futures backtest or a trading system.

## Input contract

The API uses three immutable record types:

- `FuturesContract(contract_id, currency, multiplier)`: use an individual expiry,
  not a continuous ticker. Multiplier is settlement-currency units per quoted
  price unit per contract. The caller must obtain and verify the actual contract
  specification; no size is supplied by the software.
- `FuturesExecution(contract_id, quantity, fill_price, reference_price,
  commission, fees)`: quantity is a nonzero signed integer; positive buys and
  negative sells. Prices may be negative or zero. The actual fill already
  contains slippage. Reference price supplies attribution, not another charge.
  Commission and fees are separate, non-overlapping, nonnegative **total** charges
  for this execution in the contract currency. Supply explicit zeros when known
  to be zero. Charges in another currency require preprocessing, not silent FX
  assumptions.
- `SettlementBatch(timestamp, executions, settlements, fx_cad_per_unit)`:
  timestamp must include a timezone; batches must increase strictly in absolute
  time. Provide a settlement price for exactly every previously held or newly
  executed contract, including fully closed contracts. Provide CAD per one unit
  of every active contract currency and retained cash currency, plus `CAD: 1`.
  For example, `USD: 1.40` means one USD is worth CAD 1.40. Rates must be positive.

Executions in a batch are the supplied fills since the previous settlement and
before this settlement. At most one execution record per contract per batch is
accepted, so duplicate rows cannot silently double an order. Multiple fills can
be consolidated only when the signed quantity, total charges and signed
price-weighted amounts are preserved. A net-zero round trip cannot be represented
by a zero-quantity aggregate; provide distinct verified settlement checkpoints
instead, or extend the input model before using such data.

The caller supplies `initial_cash_cad` (nonnegative) and a nonempty sequence of
batches. No prices, timestamps, FX rates, calendars or expiration/roll events are
inferred or forward-filled. Unknown contracts, duplicate contract definitions,
duplicate batch times, duplicate execution contracts, and missing marks or FX
raise errors. Inputs are not mutated; output dictionaries can be serialized to
JSON without nonfinite values.

## Accounting and reconciliation

At every supplied checkpoint, all marked P&L is **cash settled** into the
contract's native currency. A checkpoint may be daily or intraday only if this
cash-settlement convention is appropriate to the input. Ordinary intraday marks
without cash settlement require a separate unrealized-P&L model; do not label
them settlement events to make this API accept them.

For a contract with previous quantity `q`, previous settlement `s0`, new signed
execution `dq`, fill `p`, new settlement `s`, and multiplier `m`:

```text
actual futures P&L = m × [q × (s − s0) + dq × (s − p)]
slippage attribution = m × dq × (p − reference_price)
net settlement P&L = actual futures P&L − commission − fees
```

Reference-price P&L minus slippage equals actual-fill P&L. Slippage can be negative
when a fill improves on its reference; it is never deducted twice. Previous
quantity is zero for a newly opened contract, so no old settlement is fabricated.
After settlement, the new basis is `s` and quantity is `q + dq`.

A roll is an explicit close of one expiry and open of another. Each leg uses its
own price and multiplier. A price gap between expiries creates no return. The
same difference-based formula handles negative futures prices, short positions,
partial closes and position reversals.

Native-currency variation cash is retained rather than automatically converted.
Each row separately reports settlement P&L translated at current FX and the FX
translation change on cash retained from the prior checkpoint. After settlement,
CAD equity is the sum of native cash balances times current FX. The final output
reports the difference between ending equity and initial cash plus cumulative
P&L as `reconciliation_error_cad`; only floating-point rounding should remain.
Retained foreign cash still requires FX after the related contract is closed.

This ledger excludes margin requirements, collateral transfers, funding charges,
interest, deposits, withdrawals and liquidation rules. Futures notional is not
debited as a purchase price. Negative cash/equity is preserved, not interpreted
as permission to trade. Actual contract, execution, settlement, FX, calendar and
cost data and a separately frozen evaluation design are still required before
making performance or investability claims.

## Synthetic example

```python
from weather_risk.execution import (
    FuturesContract, FuturesExecution, SettlementBatch, account_futures,
)

contract = FuturesContract("SYNTHETIC-EXPIRY", "USD", 10.0)
execution = FuturesExecution("SYNTHETIC-EXPIRY", 2, 101.0, 100.0, 3.0, 1.0)
checkpoint = SettlementBatch(
    "2026-01-01T21:00:00Z", [execution],
    {"SYNTHETIC-EXPIRY": 105.0}, {"CAD": 1.0, "USD": 1.25},
)
ledger = account_futures([contract], [checkpoint], initial_cash_cad=1000.0)
# Reference P&L USD 100; slippage USD 20; actual P&L USD 80;
# commissions/fees USD 4; retained cash USD 76; ending CAD equity 1095.
```

Run the synthetic accounting checks with
`python -m unittest discover -s tests -p 'test_execution.py'` from an environment
where the project's `src` package is installed or `PYTHONPATH=src` is set.
