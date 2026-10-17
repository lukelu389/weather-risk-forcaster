"""Fixed-notional CAD basket sensitivities from verified Phase 5 forecasts."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import pandas as pd

from .eda import ROOT, sha, table, write_json
from .ml import ALL_MODELS
from .portfolio_risk import (empirical_cad_var_es, gaussian_cad_var_es,
                             log_returns_to_cad_losses)
from .risk import empirical_var_es, gaussian_var_es


def validate_config(config):
    expected = {"asset": "canadian_basket", "notional_cad": 1_000_000,
                "cost_bps": [0, 5, 10], "calendar_subperiods": [2024, 2025]}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected fixed portfolio sensitivity setting: {key}")
    if not isinstance(config.get("input_run"), str) or not config["input_run"]:
        raise ValueError("A preserved Phase 5 input_run is required")
    if not isinstance(config.get("input_manifest_sha256"), str) or len(config["input_manifest_sha256"]) != 64:
        raise ValueError("A pinned Phase 5 manifest SHA256 is required")


def load_final_run(config):
    """Verify all consumed files before parsing either numeric CSV."""
    validate_config(config)
    source = ROOT / config["input_run"]
    manifest_path = source / "manifest.json"
    if sha(manifest_path) != config["input_manifest_sha256"]:
        raise ValueError("Pinned Phase 5 manifest differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("phase") != 5 or manifest.get("status") != "passed":
        raise ValueError("Portfolio sensitivity requires a passed Phase 5 run")
    consumed = ("forecasts.csv", "design_panel.csv", "model_config.json")
    hashes = {}
    for name in consumed:
        hashes[name] = sha(source / name)
        if hashes[name] != manifest.get("artifacts", {}).get(name):
            raise ValueError(f"Phase 5 checksum mismatch: {name}")
    model = json.loads((source / "model_config.json").read_text())
    if (model.get("test_start"), model.get("test_end"), model.get("confidences")) != (
            "2024-01-01", "2025-12-31", [.95, .99]):
        raise ValueError("Unexpected Phase 5 interval or confidence settings")
    if (isinstance(model.get("train_window"), bool) or not isinstance(model.get("train_window"), int)
            or model["train_window"] < 1):
        raise ValueError("Invalid preserved training window")
    equities = [spec for spec in model["prices"] if spec.get("equity")]
    if len(equities) != 4 or any(spec["transform"] != "log_pct" for spec in equities):
        raise ValueError("The CAD basket requires four log-return equity constituents")
    forecasts = pd.read_csv(source / "forecasts.csv")
    panel = pd.read_csv(source / "design_panel.csv")
    provenance = {"run_dir": str(source.resolve()), "manifest_sha256": sha(manifest_path),
                  "artifacts": hashes, "phase": 5, "status": "passed"}
    return forecasts, panel, model, provenance


def basket_inputs(forecasts, panel, config, model):
    """Retain the complete basket grid and verify agreement with its saved design."""
    asset = config["asset"]
    forecasts = forecasts.loc[forecasts.asset.eq(asset)].copy()
    panel = panel.loc[panel.asset.eq(asset)].copy()
    if forecasts.empty or panel.empty or set(forecasts.model) != set(ALL_MODELS):
        raise ValueError("All nine basket models and the saved design are required")
    if forecasts.duplicated(["date", "model", "confidence"]).any() or panel.date.duplicated().any():
        raise ValueError("Duplicate basket forecast or design dates")
    if not forecasts.unit.eq("log return (%)").all() or not panel.unit.eq("log return (%)").all():
        raise ValueError("CAD basket inputs must be log return (%)")
    if not forecasts.status.isin(["passed", "skipped"]).all():
        raise ValueError("A passed Phase 5 run cannot contain failed or unknown basket statuses")
    dates = pd.to_datetime(forecasts.date, errors="raise")
    if not dates.between(model["test_start"], model["test_end"]).all():
        raise ValueError("Basket forecasts lie outside the fixed final interval")
    panel["date"] = pd.to_datetime(panel.date, errors="raise")
    panel = panel.sort_values("date").reset_index(drop=True)
    expected_dates = set(panel.loc[panel.date.between(model["test_start"], model["test_end"]), "date"])
    forecasts["date"] = dates
    expected_groups = {(name, confidence) for name in ALL_MODELS for confidence in model["confidences"]}
    if set(zip(forecasts.model, forecasts.confidence)) != expected_groups:
        raise ValueError("Incomplete basket model/confidence groups")
    for _, group in forecasts.groupby(["model", "confidence"]):
        if set(group.date) != expected_dates:
            raise ValueError("Incomplete basket forecast grid; missing rows must remain explicit")
    for frame in (forecasts, panel):
        for name in ("origin_at", "label_available_at"):
            frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
        frame["interval_start"] = pd.to_datetime(frame.interval_start, errors="raise")
        if np.isinf(frame.target).any():
            raise ValueError("Infinite basket target")
    lookup = panel.set_index("date")
    for _, group in forecasts.groupby(["model", "confidence"]):
        reference = lookup.reindex(group.date)
        if not np.allclose(group.target, reference.target, rtol=0, atol=0, equal_nan=True):
            raise ValueError("Basket forecasts disagree with saved design targets")
        for name in ("origin_at", "interval_start", "label_available_at"):
            left, right = group[name].reset_index(drop=True), reference[name].reset_index(drop=True)
            if not left.equals(right):
                raise ValueError(f"Basket forecasts disagree with saved design {name}")
    return forecasts.sort_values(["date", "model", "confidence"]).reset_index(drop=True), panel


def matured_sample(panel, index, origin, window):
    """Rebuild the original-grid rolling sample, never compressing missing rows."""
    target = panel.target.to_numpy(dtype=float)
    eligible = (panel.label_available_at <= origin).to_numpy() & np.isfinite(target) & (np.arange(len(panel)) < index)
    indices = np.flatnonzero(eligible)
    stop = int(indices[-1] + 1) if len(indices) else 0
    start = max(0, stop - window)
    return target[start:stop][eligible[start:stop]]


def sensitivity_rows(forecasts, panel, config, model):
    forecasts, panel = basket_inputs(forecasts, panel, config, model)
    lookup = {date: index for index, date in enumerate(panel.date)}
    samples, rows = {}, []
    for row in forecasts.itertuples(index=False):
        if row.date not in samples:
            samples[row.date] = matured_sample(panel, lookup[row.date], row.origin_at, model["train_window"])
        sample = samples[row.date]
        if not np.isfinite(row.n_available) or row.n_available != len(sample):
            raise ValueError("Reconstructed matured sample differs from saved n_available")
        passed = row.status == "passed"
        if passed:
            if len(sample) < model["minimum_train"]:
                raise ValueError("Passed forecast has insufficient matured history")
            if row.model == "historical":
                expected = empirical_var_es(sample, row.confidence)
            else:
                if not np.isfinite([row.mean, row.variance]).all() or row.variance < 0:
                    raise ValueError("Invalid saved Gaussian forecast moments")
                expected = gaussian_var_es(row.mean, np.sqrt(row.variance), row.confidence)
            if not np.allclose(expected, [row.var, row.es], rtol=1e-8, atol=1e-9):
                raise ValueError("Reconstructed log-return VaR/ES differs from saved forecast")
        realized = (float(log_returns_to_cad_losses([row.target], notional=config["notional_cad"])[0])
                    if np.isfinite(row.target) else np.nan)
        for cost in config["cost_bps"]:
            budget = config["notional_cad"] * cost / 10000
            risk = {"confidence": row.confidence, "notional_cad": config["notional_cad"],
                    "cost_bps": cost, "cost_budget_cad": budget,
                    "gross_var_cad": np.nan, "gross_es_cad": np.nan, "var_cad": np.nan, "es_cad": np.nan}
            if passed:
                if row.model == "historical":
                    risk = empirical_cad_var_es(sample, row.confidence, notional=config["notional_cad"], cost_bps=cost)
                else:
                    risk = gaussian_cad_var_es(row.mean, np.sqrt(row.variance), row.confidence,
                                               notional=config["notional_cad"], cost_bps=cost)
            loss = realized + budget
            exceed = float(loss > risk["var_cad"]) if passed and np.isfinite(realized) else np.nan
            gross_exceed = float(realized > risk["gross_var_cad"]) if passed and np.isfinite(realized) else np.nan
            if np.isfinite(exceed) and exceed != gross_exceed:
                raise ValueError("Adding the same fixed cost changed an exceedance classification")
            rows.append({"asset": row.asset, "date": row.date.date().isoformat(),
                         "interval_start": row.interval_start.date().isoformat() if pd.notna(row.interval_start) else "",
                         "origin_at": row.origin_at.isoformat() if pd.notna(row.origin_at) else "",
                         "model": row.model, "status": row.status,
                         "error": row.error if isinstance(row.error, str) else "",
                         "target": row.target, "target_invalid_reason": row.target_invalid_reason
                         if isinstance(row.target_invalid_reason, str) else "",
                         "n_available": len(sample), **risk, "gross_realized_loss_cad": realized,
                         "realized_loss_cad": loss, "var_exceedance": exceed,
                         "gross_var_exceedance": gross_exceed})
    return pd.DataFrame(rows)


def summarize(rows, config):
    result = []
    for period in ("all", *map(str, config["calendar_subperiods"])):
        selected = rows if period == "all" else rows.loc[rows.date.str.startswith(period)]
        for (model, confidence, cost), group in rows.groupby(["model", "confidence", "cost_bps"], sort=True):
            subset = selected.loc[selected.model.eq(model) & selected.confidence.eq(confidence) & selected.cost_bps.eq(cost)]
            valid_risk = np.isfinite(subset.var_cad) & np.isfinite(subset.es_cad)
            valid_realized = np.isfinite(subset.realized_loss_cad)
            scored = np.isfinite(subset.var_exceedance)
            result.append({"period": period, "asset": config["asset"], "model": model,
                "confidence": confidence, "cost_bps": cost, "notional_cad": config["notional_cad"],
                "cost_budget_cad": config["notional_cad"] * cost / 10000,
                "forecast_rows": len(subset), "risk_rows": int(valid_risk.sum()),
                "realized_rows": int(valid_realized.sum()), "scored_rows": int(scored.sum()),
                "skipped_rows": int(subset.status.eq("skipped").sum()),
                "missing_realized_rows": int((~valid_realized).sum()),
                "mean_var_cad": float(subset.loc[valid_risk, "var_cad"].mean()) if valid_risk.any() else np.nan,
                "mean_es_cad": float(subset.loc[valid_risk, "es_cad"].mean()) if valid_risk.any() else np.nan,
                "worst_realized_loss_cad": float(subset.loc[valid_realized, "realized_loss_cad"].max()) if valid_realized.any() else np.nan,
                "exceedances": int(subset.loc[scored, "var_exceedance"].sum()),
                "exceedance_rate": float(subset.loc[scored, "var_exceedance"].mean()) if scored.any() else np.nan})
    return pd.DataFrame(result)


def write_report(output, summary, manifest):
    selected = summary.loc[summary.period.eq("all") & summary.cost_bps.eq(0)]
    text = ["# Phase 6 fixed-notional CAD basket sensitivity", "",
        "Each interval starts independently with CAD 1,000,000, allocated 25% to each of the four existing "
        "Canadian equity constituents. This is an adjusted-price basket proxy using latest-vintage data and "
        "assumed availability. It is not an executable strategy, compounded NAV, CAGR, or portfolio optimization.", "",
        "All nine fixed Phase 5 models and both risk levels are retained. Gas and WTI are not aggregated into the CAD basket. "
        "No FX hedge, futures roll, financing, turnover or realized transaction-cost estimate is inferred.", "",
        "A log-percent basket return R becomes gross CAD loss N × (1 − exp(R/100)). Gaussian log-return forecasts "
        "use the exact transformed normal quantile and tail expectation. Historical simulation rebuilds the original "
        "matured rolling sample and transforms each observation before calculating monetary ES; transforming the "
        "previous log-return ES would be incorrect.", "",
        "## Full-period gross sensitivity", "",
        table(selected, ["model", "confidence", "scored_rows", "mean_var_cad", "mean_es_cad",
                         "worst_realized_loss_cad", "exceedance_rate"]), "",
        "## Fixed cost budgets and calendar splits", "",
        "The 0, 5 and 10 basis-point scenarios add CAD 0, 500 and 1,000 per interval to both realized loss and "
        "forecast VaR/ES. Each is the entire fixed interval cost budget, not a per-side or turnover estimate. "
        "Exceedance classifications therefore remain unchanged. summary.csv reports every model, confidence and cost "
        "for the full period and the prespecified calendar years 2024 and 2025. These calendar slices are descriptive, "
        "not market regime classifications or new significance tests.", "",
        f"Saved sensitivity rows: {manifest['sensitivity_rows']}; original basket forecast rows: {manifest['basket_forecast_rows']}.", "",
        "Missing realized intervals and skipped forecasts remain explicit; no interpolation or annualization is performed. "
        "Mean risks use finite risk forecasts, worst realized loss uses finite realizations, and exceedance rates require both. "
        "The fixed survivor basket, adjusted-price conventions, retrospective quote grid and variable interval lengths remain "
        "limitations. No 2008 or 2014–2016 coverage, trading recommendation, capital approval or model selection is claimed.", "",
        "The source Phase 5 manifest and all consumed artifact hashes are verified before numeric parsing. "
        "Saved configuration, source snapshot and hashes support reproduction of this sensitivity calculation."]
    (output / "portfolio_report.md").write_text("\n".join(text) + "\n")


def run_portfolio(config_path=ROOT / "config/portfolio.json", *, publish=True):
    config = json.loads(Path(config_path).read_text())
    forecasts, panel, model, source = load_final_run(config)
    output = ROOT / "runs" / ("portfolio-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).parent
    files = ("portfolio.py", "portfolio_risk.py", "risk.py", "eda.py", "ml.py")
    manifest = {"phase": 6, "status": "started", "source": source,
        "historical_point_in_time_certified": False, "independent_interval_notional": True,
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")},
        "code_sha256": {name: sha(code / name) for name in files}, "artifacts": {}}
    write_json(output / "portfolio_config.json", config)
    write_json(output / "model_config.json", model)
    write_json(output / "source_snapshot.json", {name: (code / name).read_text() for name in files})
    try:
        rows = sensitivity_rows(forecasts, panel, config, model)
        summary = summarize(rows, config)
        rows.to_csv(output / "sensitivity_rows.csv", index=False, float_format="%.12g")
        summary.to_csv(output / "summary.csv", index=False, float_format="%.12g")
        manifest.update(status="passed", sensitivity_rows=len(rows), summary_rows=len(summary),
                        basket_forecast_rows=len(rows) // len(config["cost_bps"]),
                        skipped_sensitivity_rows=int(rows.status.eq("skipped").sum()),
                        missing_realized_sensitivity_rows=int(rows.realized_loss_cad.isna().sum()),
                        cost_exceedance_invariant=True)
        write_report(output, summary, manifest)
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["artifacts"] = {p.name: sha(p) for p in sorted(output.iterdir()) if p.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)
    if publish:
        report = ROOT / "reports/phase6"
        report.mkdir(parents=True, exist_ok=True)
        for name in ("summary.csv", "portfolio_report.md", "manifest.json", "portfolio_config.json"):
            shutil.copy2(output / name, report / name)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/portfolio.json")
    args = parser.parse_args(argv)
    print(run_portfolio(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
