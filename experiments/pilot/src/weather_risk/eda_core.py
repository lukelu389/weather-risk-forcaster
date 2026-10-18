"""Development-only return construction and descriptive statistics for Phase 2.

No model selection or inferential tests live here. Source quote sequences are
preserved: an explicit missing quote makes both adjacent movements unavailable.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    "asset", "label", "date", "interval_start", "origin_at", "target", "unit",
    "gap_days", "invalid_reason",
]
SUMMARY_COLUMNS = [
    "asset", "n", "mean", "std", "skew", "excess_kurtosis", "q01", "q05",
    "median", "q95", "q99", "min", "max", "acf1", "abs_acf1", "squared_acf1",
    "count_abs_z_gt3", "gaussian_3sigma_expected", "first_date", "last_date", "unit",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development_observations(
    run_dir: Path, analysis_start: str, analysis_end: str,
) -> tuple[pd.DataFrame, dict]:
    """Verify a passed Phase 1 snapshot, then parse only development values.

    Hashing the complete input establishes provenance; numeric conversion and all
    analysis happen only after filtering dates. Checksums detect accidental file
    changes, not malicious replacement of both a file and its manifest.
    """
    run_dir = Path(run_dir)
    start, end = pd.Timestamp(analysis_start), pd.Timestamp(analysis_end)
    if start.tzinfo or end.tzinfo or start != start.normalize() or end != end.normalize():
        raise ValueError("Analysis bounds must be timezone-naive calendar dates")
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("analysis_start must be no later than analysis_end")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("phase") != 1 or manifest.get("status") != "passed":
        raise ValueError("Phase 2 requires a passed Phase 1 manifest")
    hashes = {}
    for name in ("config.json", "observations.csv", "validation.json"):
        actual = _sha256(run_dir / name)
        expected = manifest.get("artifacts", {}).get(name)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}")
        hashes[name] = actual
    if hashes["config.json"] != manifest.get("config_sha256"):
        raise ValueError("Config checksum differs from manifest config_sha256")
    validation = json.loads((run_dir / "validation.json").read_text())
    if validation.get("passed") is not True or validation.get("errors"):
        raise ValueError("Phase 1 validation must have passed without errors")
    config = json.loads((run_dir / "config.json").read_text())
    if start < pd.Timestamp(config["start"]) or end > pd.Timestamp(config["end"]):
        raise ValueError("Analysis interval lies outside the Phase 1 configured coverage")
    raw = pd.read_csv(run_dir / "observations.csv", dtype=str, keep_default_na=False)
    required = {
        "series_id", "observation_date", "value", "unit", "source", "available_at",
        "retrieved_at", "availability_basis", "vintage", "quality_flag",
    }
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing observation columns: {sorted(required - set(raw.columns))}")
    dates = pd.to_datetime(raw["observation_date"], format="%Y-%m-%d", errors="raise")
    mask = dates.between(start, end)
    observations = raw.loc[mask].copy()
    if observations.empty:
        raise ValueError("No observations in the development interval")
    observations["observation_date"] = dates.loc[mask]
    # Do not move numeric parsing above the development-date filter.
    observations["value"] = pd.to_numeric(
        observations["value"].replace("", np.nan), errors="raise",
    ).astype(float)
    if np.isinf(observations["value"]).any():
        raise ValueError("Nonfinite development observation value")
    for name in ("available_at", "retrieved_at"):
        observations[name] = pd.to_datetime(observations[name], utc=True, errors="raise")
        if observations[name].isna().any():
            raise ValueError(f"Missing {name} timestamp")
    if observations.duplicated(["series_id", "observation_date"]).any():
        raise ValueError("Duplicate series/date development observation")
    observations = observations.sort_values(["series_id", "observation_date"]).reset_index(drop=True)
    provenance = {
        "run_dir": str(run_dir.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "config_sha256": hashes["config.json"],
        "observations_sha256": hashes["observations.csv"],
        "validation_sha256": hashes["validation.json"],
        "input_mode": manifest.get("mode"),
        "historical_point_in_time_certified": manifest.get("historical_point_in_time_certified", False),
        "analysis_start": start.date().isoformat(),
        "analysis_end": end.date().isoformat(),
        "input_rows": len(raw),
        "development_rows": len(observations),
        "excluded_rows": int((~mask).sum()),
        "numeric_values_read_outside_development": False,
    }
    return observations, provenance


def _origins(interval_start: pd.Series) -> pd.Series:
    """18:00 Toronto on the preceding calendar date, converted to UTC."""
    previous_day = interval_start - pd.Timedelta(days=1)
    return (previous_day + pd.Timedelta(hours=18)).dt.tz_localize(
        "America/Toronto", ambiguous="raise", nonexistent="raise",
    ).dt.tz_convert("UTC")


def make_targets(
    obs: pd.DataFrame, price_specs: list[dict], max_gap_days: int = 4,
) -> pd.DataFrame:
    """Construct quote-to-quote changes without filling or bridging null quotes.

    These are observation sessions, not an official trading calendar. ``origin_at``
    conservatively predates the entire movement interval, including its first price.
    The equal-weight basket assumes rebalancing to 25% at every common interval start.
    """
    if not isinstance(max_gap_days, int) or isinstance(max_gap_days, bool) or max_gap_days < 1:
        raise ValueError("max_gap_days must be a positive integer")
    ids = [s["series_id"] for s in price_specs]
    if len(ids) != len(set(ids)) or "canadian_basket" in ids:
        raise ValueError("Price specifications must have unique, nonreserved series IDs")
    equity_ids = [s["series_id"] for s in price_specs if s.get("equity", False)]
    if equity_ids and len(equity_ids) != 4:
        raise ValueError("The configured Canadian basket requires exactly four equities")
    pieces = []
    for spec in price_specs:
        if spec["transform"] not in {"log_pct", "difference"}:
            raise ValueError(f"Unsupported target transform: {spec['transform']}")
        if spec.get("equity", False) and spec["transform"] != "log_pct":
            raise ValueError("Basket equity specifications require log_pct targets")
        prices = obs.loc[obs["series_id"].eq(spec["series_id"])].sort_values("observation_date")
        if prices.empty:
            raise ValueError(f"No development observations for {spec['series_id']}")
        if prices["observation_date"].duplicated().any():
            raise ValueError(f"Duplicate price dates for {spec['series_id']}")
        dates = prices["observation_date"].reset_index(drop=True)
        current = prices["value"].reset_index(drop=True).astype(float)
        previous, interval_start = current.shift(), dates.shift()
        gaps = (dates - interval_start).dt.days
        reasons = pd.Series("", index=dates.index)
        reasons.loc[interval_start.isna()] = "no_previous_quote"
        reasons.loc[reasons.eq("") & current.isna()] = "missing_current_price"
        reasons.loc[reasons.eq("") & previous.isna()] = "missing_previous_price"
        reasons.loc[reasons.eq("") & (~np.isfinite(current) | ~np.isfinite(previous))] = "nonfinite_endpoint"
        reasons.loc[reasons.eq("") & gaps.gt(max_gap_days)] = "excessive_gap"
        if spec["transform"] == "log_pct":
            reasons.loc[reasons.eq("") & (current.le(0) | previous.le(0))] = "nonpositive_endpoint"
        valid = reasons.eq("")
        target = pd.Series(np.nan, index=dates.index)
        if spec["transform"] == "log_pct":
            target.loc[valid] = 100 * (np.log(current.loc[valid]) - np.log(previous.loc[valid]))
        else:
            target.loc[valid] = current.loc[valid] - previous.loc[valid]
        pieces.append(pd.DataFrame({
            "asset": spec["series_id"], "label": spec["label"], "date": dates,
            "interval_start": interval_start, "origin_at": _origins(interval_start),
            "target": target, "unit": spec["unit"], "gap_days": gaps,
            "invalid_reason": reasons,
        }))
    if not pieces:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    targets = pd.concat(pieces, ignore_index=True)
    if equity_ids:
        equity = targets.loc[targets["asset"].isin(equity_ids)]
        basket_rows = []
        for day, group in equity.groupby("date", sort=True):
            common_interval = len(group) == 4 and group["interval_start"].notna().all() and group["interval_start"].nunique() == 1
            start = group["interval_start"].iloc[0] if common_interval else pd.NaT
            if len(group) != 4:
                reason = "missing_constituent"
            elif not np.isfinite(group["target"]).all():
                reason = "invalid_constituent_return"
            elif not common_interval:
                reason = "constituent_interval_mismatch"
            else:
                reason = ""
            value = np.nan
            if not reason:
                value = 100 * np.log1p(np.expm1(group["target"].to_numpy() / 100).mean())
            basket_rows.append({
                "asset": "canadian_basket", "label": "Canadian equal-weight basket",
                "date": day, "interval_start": start,
                "origin_at": group["origin_at"].iloc[0] if common_interval else pd.NaT,
                "target": value, "unit": "log return (%)",
                "gap_days": (day - start).days if common_interval else np.nan,
                "invalid_reason": reason,
            })
        targets = pd.concat([targets, pd.DataFrame(basket_rows)], ignore_index=True)
    return targets.sort_values(["asset", "date"]).reset_index(drop=True)[TARGET_COLUMNS]


def _lagged_correlation(values: pd.Series, lag: int) -> tuple[float, int]:
    """Pearson correlation over finite lagged pairs, preserving missing positions."""
    values = values.reset_index(drop=True)
    previous = values.shift(lag)
    valid = np.isfinite(values) & np.isfinite(previous)
    left, right = values.loc[valid], previous.loc[valid]
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan"), int(valid.sum())
    return float(left.corr(right)), int(valid.sum())


def acf_table(targets: pd.DataFrame, max_lag: int = 20) -> pd.DataFrame:
    """Descriptive lag correlations; no iid confidence bands or p-values."""
    if not isinstance(max_lag, int) or isinstance(max_lag, bool) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer")
    rows = []
    for asset, group in targets.groupby("asset", sort=True):
        values = group.sort_values("date")["target"].astype(float)
        for name, transformed in (("return", values), ("absolute", values.abs()), ("squared", values**2)):
            for lag in range(1, max_lag + 1):
                correlation, count = _lagged_correlation(transformed, lag)
                rows.append({"asset": asset, "transform": name, "lag": lag, "acf": correlation, "n_pairs": count})
    return pd.DataFrame(rows, columns=["asset", "transform", "lag", "acf", "n_pairs"])


def describe_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """Sample moments and quantiles on finite targets, with gap-aware lag pairs."""
    rows = []
    for asset, group in targets.groupby("asset", sort=True):
        group = group.sort_values("date")
        if group["unit"].nunique() != 1:
            raise ValueError(f"Mixed target units for {asset}")
        all_values = group["target"].astype(float)
        finite = np.isfinite(all_values)
        values = all_values.loc[finite]
        std, mean = values.std(ddof=1), values.mean()
        quantiles = values.quantile([.01, .05, .5, .95, .99])
        rows.append({
            "asset": asset, "n": len(values), "mean": mean, "std": std,
            "skew": values.skew(), "excess_kurtosis": values.kurt(),
            "q01": quantiles.loc[.01], "q05": quantiles.loc[.05], "median": quantiles.loc[.5],
            "q95": quantiles.loc[.95], "q99": quantiles.loc[.99],
            "min": values.min(), "max": values.max(),
            "acf1": _lagged_correlation(all_values, 1)[0],
            "abs_acf1": _lagged_correlation(all_values.abs(), 1)[0],
            "squared_acf1": _lagged_correlation(all_values**2, 1)[0],
            "count_abs_z_gt3": int(((values - mean).abs() > 3 * std).sum()) if std > 0 else 0,
            "gaussian_3sigma_expected": len(values) * math.erfc(3 / math.sqrt(2)),
            "first_date": group.loc[finite, "date"].min(), "last_date": group.loc[finite, "date"].max(),
            "unit": group["unit"].iloc[0],
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
