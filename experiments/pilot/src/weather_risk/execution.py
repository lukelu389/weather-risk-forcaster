"""Explicit futures settlement accounting, separate from research forecasts.

Every batch is a supplied cash-settlement checkpoint. Foreign settlement cash
is retained in its native currency and valued at supplied CAD-per-currency FX.
No exchange calendar, roll schedule, margin rule, execution cost or price is
inferred. See docs/execution-input-contract.md for the input convention.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FuturesContract:
    contract_id: str
    currency: str
    multiplier: float


@dataclass(frozen=True)
class FuturesExecution:
    contract_id: str
    quantity: int
    fill_price: float
    reference_price: float
    commission: float
    fees: float


@dataclass(frozen=True)
class SettlementBatch:
    timestamp: str | datetime
    executions: Sequence[FuturesExecution]
    settlements: Mapping[str, float]
    fx_cad_per_unit: Mapping[str, float]


def _number(value, name: str, *, minimum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("accounting exceeds finite numeric range")
    return value


def _timestamp(value) -> datetime:
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("timestamp must be an explicit timezone-aware datetime or ISO timestamp") from exc


def _identifier(value, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def account_futures(contracts: Sequence[FuturesContract], batches: Iterable[SettlementBatch], *,
                    initial_cash_cad: float) -> dict:
    """Build a deterministic settlement ledger from explicit inputs.

    Quantities are signed integer contracts; multiplier is currency per quoted
    price unit per contract. Executions belong to their batch's settlement
    interval. Each contract can have at most one (possibly pre-aggregated)
    execution in a batch. Timestamps must strictly increase in absolute time.

    Both fill and reference prices are required. Slippage is the signed
    difference between them, for attribution only: actual-fill P&L already
    includes it. Commission and fees are nonnegative total native-currency
    charges for that execution, including explicit zeros.

    Each checkpoint cash-settles all positions. Currency balances are retained,
    not automatically converted. NAV is their CAD value after settlement; FX
    translation of previously retained cash is reported separately. No contract
    purchase principal, margin collateral, financing, deposits or withdrawals
    are modeled. The function does not mutate input objects.
    """
    initial_cash_cad = _number(initial_cash_cad, "initial_cash_cad", minimum=0)
    specifications = {}
    for contract in contracts:
        if not isinstance(contract, FuturesContract):
            raise ValueError("contracts must contain FuturesContract objects")
        contract_id = _identifier(contract.contract_id, "contract_id")
        if contract_id in specifications:
            raise ValueError(f"duplicate contract specification: {contract_id}")
        currency = _identifier(contract.currency, "currency")
        if len(currency) != 3 or not currency.isascii() or not currency.isalpha() or not currency.isupper():
            raise ValueError("currency must be a three-letter uppercase code")
        multiplier = _number(contract.multiplier, "multiplier")
        if multiplier <= 0:
            raise ValueError("multiplier must be strictly positive")
        specifications[contract_id] = (currency, multiplier)
    if not specifications:
        raise ValueError("contracts must be nonempty")

    positions, bases = {}, {}
    cash = {"CAD": initial_cash_cad}
    previous_fx = {"CAD": 1.0}
    previous_timestamp = None
    rows = []
    for batch in batches:
        if not isinstance(batch, SettlementBatch):
            raise ValueError("batches must contain SettlementBatch objects")
        timestamp = _timestamp(batch.timestamp)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("batch timestamps must be unique and strictly increasing")
        trades = {}
        for execution in batch.executions:
            if not isinstance(execution, FuturesExecution):
                raise ValueError("executions must contain FuturesExecution objects")
            contract_id = execution.contract_id
            if contract_id not in specifications:
                raise ValueError(f"unknown execution contract: {contract_id}")
            if contract_id in trades:
                raise ValueError(f"duplicate execution contract in batch: {contract_id}")
            quantity = execution.quantity
            if isinstance(quantity, bool) or not isinstance(quantity, Integral) or quantity == 0:
                raise ValueError("execution quantity must be a nonzero signed integer")
            trades[contract_id] = (
                int(quantity), _number(execution.fill_price, "fill_price"),
                _number(execution.reference_price, "reference_price"),
                _number(execution.commission, "commission", minimum=0),
                _number(execution.fees, "fees", minimum=0),
            )
        required_contracts = set(positions) | set(trades)
        if not isinstance(batch.settlements, Mapping) or set(batch.settlements) != required_contracts:
            raise ValueError("settlements must contain exactly the held and executed contracts, including closes")
        marks = {key: _number(value, "settlement price") for key, value in batch.settlements.items()}
        needed_currencies = {"CAD"} | {specifications[key][0] for key in required_contracts}
        needed_currencies |= {key for key, value in cash.items() if value != 0}
        if not isinstance(batch.fx_cad_per_unit, Mapping) or not needed_currencies.issubset(batch.fx_cad_per_unit):
            raise ValueError("missing FX: provide CAD per unit for every active contract and retained cash currency")
        fx = {}
        for key, value in batch.fx_cad_per_unit.items():
            _identifier(key, "FX currency")
            fx[key] = _number(value, "FX rate")
            if fx[key] <= 0:
                raise ValueError("FX rates must be strictly positive")
        if fx["CAD"] != 1.0:
            raise ValueError("CAD-per-CAD FX must equal 1")

        fx_pnl = _finite(math.fsum(
            value * (fx[currency] - previous_fx[currency])
            for currency, value in cash.items() if value != 0
        ))
        attribution = []
        for contract_id in sorted(required_contracts):
            currency, multiplier = specifications[contract_id]
            old_quantity = positions.get(contract_id, 0)
            quantity, fill, reference, commission, fees = trades.get(contract_id, (0, marks[contract_id], marks[contract_id], 0., 0.))
            mark = marks[contract_id]
            prior_mark = bases.get(contract_id)
            try:
                carried_pnl = old_quantity * multiplier * (mark - prior_mark) if old_quantity else 0.
                actual_pnl = _finite(carried_pnl + quantity * multiplier * (mark - fill))
                reference_pnl = _finite(carried_pnl + quantity * multiplier * (mark - reference))
                slippage = _finite(quantity * multiplier * (fill - reference))
                net_pnl = _finite(actual_pnl - commission - fees)
            except OverflowError as exc:
                raise ValueError("accounting exceeds finite numeric range") from exc
            cash[currency] = _finite(cash.get(currency, 0.) + net_pnl)
            new_quantity = old_quantity + quantity
            if new_quantity:
                positions[contract_id], bases[contract_id] = new_quantity, mark
            else:
                positions.pop(contract_id, None)
                bases.pop(contract_id, None)
            attribution.append({
                "contract_id": contract_id, "currency": currency, "multiplier": multiplier,
                "quantity_before": old_quantity, "executed_quantity": quantity,
                "quantity_after": new_quantity, "previous_settlement": prior_mark,
                "settlement": mark, "fill_price": fill if quantity else None,
                "reference_price": reference if quantity else None, "fx_cad_per_unit": fx[currency],
                "reference_pnl": reference_pnl, "slippage": slippage,
                "futures_pnl": actual_pnl, "commission": commission, "fees": fees,
                "net_settlement_pnl": net_pnl,
            })
        totals = {
            key + "_cad": _finite(math.fsum(item[key] * item["fx_cad_per_unit"] for item in attribution))
            for key in ("reference_pnl", "slippage", "futures_pnl", "commission", "fees", "net_settlement_pnl")
        }
        equity = _finite(math.fsum(value * fx[currency] for currency, value in cash.items() if value != 0))
        rows.append({
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"), **totals,
            "fx_translation_pnl_cad": fx_pnl,
            "total_pnl_cad": _finite(totals["net_settlement_pnl_cad"] + fx_pnl),
            "cash_equity_cad": equity, "cash_by_currency": dict(sorted(cash.items())),
            "positions": dict(sorted(positions.items())), "contracts": attribution,
        })
        previous_timestamp, previous_fx = timestamp, fx
    if not rows:
        raise ValueError("batches must be nonempty")
    total_pnl = _finite(math.fsum(row["total_pnl_cad"] for row in rows))
    return {
        "initial_cash_cad": initial_cash_cad, "rows": rows,
        "ending_cash_equity_cad": rows[-1]["cash_equity_cad"],
        "total_pnl_cad": total_pnl,
        "reconciliation_error_cad": _finite(rows[-1]["cash_equity_cad"] - initial_cash_cad - total_pnl),
        "ending_positions": rows[-1]["positions"].copy(),
        "ending_cash_by_currency": rows[-1]["cash_by_currency"].copy(),
    }
