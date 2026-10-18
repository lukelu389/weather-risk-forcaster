"""Reproduce Phase 2 exploratory tables and figures on development data only."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import numpy as np
import pandas as pd

from .eda_core import load_development_observations, make_targets, describe_targets, acf_table
from .eda_weather import weather_features, relationships


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def table(frame, columns, digits=3):
    def cell(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame[columns].itertuples(index=False, name=None))
    return "\n".join(lines)


def validate_protocol(config):
    keys = ["analysis_start", "analysis_end", "tuning_start", "tuning_end", "test_start", "test_end"]
    dates = [pd.Timestamp(config[k]) for k in keys]
    if not (dates[0] <= dates[1] < dates[2] <= dates[3] < dates[4] <= dates[5]):
        raise ValueError("Exploration, tuning and final-test date intervals must be chronological and disjoint")
    if any(pd.isna(d) or d.tzinfo or d != d.normalize() for d in dates):
        raise ValueError("Protocol bounds must be calendar dates")


def run_analysis(config_path=ROOT / "config/analysis.json", *, input_run=None,
                 allow_latest_vintage=False, publish=True):
    config = json.loads(Path(config_path).read_text())
    validate_protocol(config)
    run_dir = Path(input_run) if input_run else ROOT / config["input_run"]
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not (run_dir / "observations.csv").is_file():
        raise FileNotFoundError(
            "Pinned research input observations.csv is absent. The public release omits provider data; "
            "run make verify-results for saved evidence, or restore the exact permitted input bundle "
            "as described in experiments/pilot/README.md before a full refit.")
    actual = sha(run_dir / "observations.csv")
    if input_run is None and actual != config["input_observations_sha256"]:
        raise ValueError("Pinned input differs; use an explicit --input-run to establish a new snapshot")
    config["input_run"] = str(run_dir.resolve())
    config["input_observations_sha256"] = actual
    observations, provenance = load_development_observations(
        run_dir, config["analysis_start"], config["analysis_end"])
    if provenance["input_mode"] != "live":
        raise ValueError("Phase 2 empirical analysis requires a live-data snapshot")
    if not allow_latest_vintage:
        raise ValueError("Use --allow-latest-vintage to acknowledge descriptive analysis of revised historical data")
    source_config = json.loads((run_dir / "config.json").read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / "runs" / f"eda-{stamp}"
    # Resolve dependencies and source provenance before creating a run directory.
    manifest = {"phase": 2, "status": "started", "seed": config["seed"],
                "source": provenance, "allow_latest_vintage": True,
                "historical_point_in_time_certified": False,
                "python": sys.version, "platform": platform.platform(),
                "packages": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "matplotlib"]},
                "code_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "protocol": {k: v for k, v in config.items() if k != "prices"},
                "artifacts": {}}
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_json(out / "analysis_config.json", config)
        write_json(out / "source_snapshot.json", {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})
        lock = ROOT / "requirements-analysis.lock.txt"
        if lock.exists():
            shutil.copy2(lock, out / lock.name)
        targets = make_targets(observations, config["prices"], config["max_gap_days"])
        summary = describe_targets(targets)
        acf = acf_table(targets, config["acf_max_lag"])
        features = weather_features(observations, targets, source_config["series"],
                                    allow_latest_vintage=True,
                                    minimum_reference_observations=config["weather_reference_minimum"])
        weather = relationships(targets, features, seed=config["seed"],
                                block_length=config["bootstrap_block_length"],
                                bootstrap_repetitions=config["bootstrap_repetitions"],
                                minimum_pairs=config["bootstrap_minimum_pairs"])
        subperiods = []
        for label, start, end in [("2018–2019", "2018-01-01", "2019-12-31"),
                                  ("2020–2021", "2020-01-01", "2021-12-31")]:
            section = targets.loc[targets.date.between(start, end)]
            if not section.empty:
                subperiods.append(describe_targets(section).assign(subperiod=label))
        coverage = observations.groupby("series_id").agg(
            rows=("value", "size"), missing=("value", lambda s: s.isna().sum()),
            first_date=("observation_date", "min"), last_date=("observation_date", "max")).reset_index()
        invalid = targets.groupby(["asset", "invalid_reason"], dropna=False).size().rename("rows").reset_index()
        frames = {"targets": targets, "summary": summary, "autocorrelation": acf,
                  "weather_features": features, "weather_relationships": weather,
                  "subperiod_summary": pd.concat(subperiods, ignore_index=True),
                  "development_coverage": coverage, "target_validity": invalid}
        for name, frame in frames.items():
            frame.to_csv(out / f"{name}.csv", index=False, float_format="%.12g")
        from .eda_plots import make_plots
        figures = make_plots(observations, targets, summary, acf, weather, out,
                             f"{config['analysis_start']} to {config['analysis_end']}")
        primary = summary.loc[summary.asset.isin(["henry_hub_spot", "wti_spot", "canadian_basket"])]
        report = ["# Phase 2 exploratory evidence", "",
                  f"Development interval: **{config['analysis_start']} to {config['analysis_end']}**. "
                  f"Tuning: {config['tuning_start']} to {config['tuning_end']}; final test: {config['test_start']} to {config['test_end']}. "
                  "Later-period values are excluded before numeric analysis. Phase 1 already checked their data quality.", "",
                  "Latest-vintage observational analysis with assumed availability times. No forecast model or predictive-accuracy claim.", "",
                  "## Target definitions", "",
                  "Henry Hub and equities use 100 × log(Pₜ/Pₜ₋₁). WTI uses Pₜ − Pₜ₋₁ in USD/barrel so negative prices remain valid. "
                  "The basket converts each equity log return back to a simple return, averages four fixed 25% weights, then applies log1p. "
                  "It is a descriptive rebalanced basket before costs. Missing constituents are never reweighted.", "",
                  "Quote intervals longer than four calendar days and movements adjacent to explicit null prices are withheld. "
                  "This conservative rule also excludes some legitimate holiday intervals; an exchange-calendar treatment is still required before forecasting.", "",
                  table(primary, ["asset", "n", "unit", "std", "excess_kurtosis", "acf1", "abs_acf1", "squared_acf1"]), "",
                  "Kurtosis and autocorrelations are descriptive estimates; a few crisis observations can dominate them. "
                  "Raw dispersion across percentage returns and dollar changes is not directly comparable. See subperiod_summary.csv for the predefined two-half sensitivity.", "",
                  "## Figures", ""]
        report.extend(f"![{name.replace('_', ' ').removesuffix('.png')}]({name})\n" for name in figures if name.endswith(".png"))
        report += ["## Weather timing and interpretation", "",
                   "The weather cutoff is 18:00 Toronto on the calendar day before the earlier price in each target interval. "
                   "Thus both target endpoints occur after the origin. This intentionally conservative information horizon differs from a conventional next-close forecast. "
                   "Weather enters only when its assumed available_at is at or before the origin, subject to staleness checks.", "",
                   "Temperature anomalies subtract the same-month mean from earlier calendar years in the development history, "
                   "using at least 20 available reference observations. This is an expanding historical reference, not an official climate normal.", "",
                   "All 70 asset–feature pairs are reported. Correlations concern absolute target movements; signed correlations are also retained in the table. "
                   "Intervals use paired moving blocks of 20 original observation-grid rows and 500 seeded resamples. "
                   "Missing positions remain in that grid. Intervals are pointwise and exploratory, with no multiple-testing or nonstationarity guarantee. "
                   "Shared seasonality, omitted market variables and US/Canadian geography prevent causal interpretation.", "",
                   "## Coverage and next gate", "",
                   "The pilot does not cover 2008 or 2014–16. Historical source expansion remains required for the originally requested regime analysis. "
                   "No interpolation, winsorization, random train/test split, or test-period tuning is used. "
                   "Before Phase 3, decide the final target horizon, exchange-calendar handling, historical coverage, and whether verified vintages can be obtained. "
                   "Otherwise all later model comparisons must retain a latest-vintage research qualification.", "",
                   "## References", "",
                   "Return tails and nonlinear dependence: [Cont (2001)](https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf). "
                   "Benchmark negative prices: [EIA WTI daily data](https://www.eia.gov/dnav/pet/hist/RWTCD.htm). "
                   "Revision history: [FRED real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)."]
        (out / "analysis_report.md").write_text("\n".join(report) + "\n")
        manifest.update(status="passed", development_observation_rows=len(observations),
                        target_rows=len(targets), finite_target_rows=int(targets.target.notna().sum()),
                        weather_pairs=len(weather), models_fitted=0)
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["artifacts"] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "manifest.json" and p.is_file()}
        write_json(out / "manifest.json", manifest)
    if publish:
        published = ROOT / "reports" / "phase2"
        published.mkdir(parents=True, exist_ok=True)
        names = figures + ["analysis_report.md", "summary.csv", "autocorrelation.csv", "weather_relationships.csv",
                           "subperiod_summary.csv", "development_coverage.csv", "target_validity.csv", "analysis_config.json", "manifest.json"]
        for name in names:
            shutil.copy2(out / name, published / name)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/analysis.json")
    parser.add_argument("--input-run", type=Path, help="Explicitly replace the pinned Phase 1 snapshot")
    parser.add_argument("--allow-latest-vintage", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args(argv)
    out = run_analysis(args.config, input_run=args.input_run,
                       allow_latest_vintage=args.allow_latest_vintage, publish=not args.no_publish)
    print(f"Phase 2 exploration passed: {out}")
    print("Latest-vintage development analysis only. No forecasting models fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
