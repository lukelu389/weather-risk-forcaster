"""Phase 3 baseline forecasts on the tuning period; final-test values stay unread.

These are latest-vintage, retrospective quote-grid research forecasts, not an
executable daily strategy. The quote calendar and availability are approximate.
"""

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

from .eda import ROOT, sha, table, validate_protocol, write_json
from .eda_core import load_development_observations, make_targets
from .eda_weather import weather_features
from .baseline_models import fit_garch, garch_variances, fit_log_variance, predict_log_variance
from .risk import empirical_var_es, gaussian_var_es, quantile_loss

MODELS = ("historical", "garch", "gjr_garch", "seasonal_variance", "weather_variance")
CONTROL_COLUMNS = ["log_recent_second_moment", "annual_sin", "annual_cos"]


def validate_baseline_config(config):
    validate_protocol(config)
    # This phase cannot move the final holdout or use it for tuning.
    if (config["analysis_start"], config["analysis_end"], config["tuning_start"],
        config["tuning_end"], config["test_start"], config["test_end"]) != (
            "2018-01-01", "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01", "2025-12-31"):
        raise ValueError("Phase 3 requires the frozen Phase 2 date protocol")
    for key, minimum in (("train_window", 30), ("minimum_train", 30), ("recent_window", 2)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if config["minimum_train"] > config["train_window"]:
        raise ValueError("minimum_train exceeds train_window")
    if config["refit"] != "monthly" or config["confidences"] != [0.95, 0.99]:
        raise ValueError("Phase 3 uses monthly refits and 95%/99% risk levels")
    if not np.isfinite(config["ridge"]) or config["ridge"] <= 0:
        raise ValueError("ridge must be positive and finite")
    if not config["assets"] or len(set(config["assets"])) != len(config["assets"]):
        raise ValueError("assets must be nonempty and unique")
    if not config["weather_columns"] or len(set(config["weather_columns"])) != len(config["weather_columns"]):
        raise ValueError("weather_columns must be nonempty and unique")


def target_availability(observations, targets, price_specs):
    """A movement matures only when all its endpoint prices are available."""
    available = observations.set_index(["series_id", "observation_date"])["available_at"]
    equity_ids = [s["series_id"] for s in price_specs if s.get("equity")]
    maturity = []
    for row in targets.itertuples():
        ids = equity_ids if row.asset == "canadian_basket" else [row.asset]
        stamps = [available.get((asset, day), pd.NaT)
                  for asset in ids for day in (row.interval_start, row.date)]
        maturity.append(max(stamps) if stamps and all(pd.notna(s) for s in stamps) else pd.NaT)
    result = targets.copy()
    result["label_available_at"] = pd.to_datetime(maturity, utc=True)
    return result


def regression_design(targets, features, weather_columns, recent_window):
    """Construct each row's predictors using that row's original forecast origin."""
    wide = features.pivot(index="origin_at", columns="feature", values="value")
    missing = set(weather_columns) - set(wide.columns)
    if missing:
        raise ValueError(f"Unconfigured weather features: {sorted(missing)}")
    result = targets.merge(wide[weather_columns], left_on="origin_at", right_index=True, how="left")
    recent = pd.Series(np.nan, index=result.index)
    for _, group in result.groupby("asset", sort=True):
        group = group.sort_values("date")
        for index, row in group.iterrows():
            if pd.isna(row.origin_at):
                continue
            eligible = group.loc[(group.label_available_at <= row.origin_at)
                                 & (group.date < row.date) & np.isfinite(group.target), "target"]
            if len(eligible) >= recent_window:
                # Fixed numerical floor, never estimated using later observations.
                recent.loc[index] = np.log(max(float(np.mean(eligible.iloc[-recent_window:] ** 2)), 1e-12))
    result["log_recent_second_moment"] = recent
    local = result.origin_at.dt.tz_convert("America/Toronto")
    angle = 2 * np.pi * (local.dt.dayofyear - 1) / 365.25
    result["annual_sin"], result["annual_cos"] = np.sin(angle), np.cos(angle)
    return result


def forecast_baselines(panel, config):
    """Walk forward with origin-specific label masks and monthly parameter fits.

    Fitting ends at the last matured finite label. Internal missing/unavailable
    rows remain NaN. Filtering then propagates over EVERY grid row to the target,
    without observing unreleased shocks. Historical simulation updates each origin.
    """
    forecasts, fits = [], []
    for asset in config["assets"]:
        group = panel.loc[panel.asset.eq(asset)].sort_values("date").reset_index(drop=True)
        if group.empty:
            raise ValueError(f"No targets for {asset}")
        y = group.target.to_numpy(dtype=float)
        x = group[CONTROL_COLUMNS + config["weather_columns"]].to_numpy(dtype=float)
        cached = {}
        for i, row in enumerate(group.itertuples()):
            if not pd.Timestamp(config["tuning_start"]) <= row.date <= pd.Timestamp(config["tuning_end"]):
                continue
            origin = row.origin_at
            eligible = (group.label_available_at <= origin).to_numpy() & np.isfinite(y) & (np.arange(len(y)) < i)
            indices = np.flatnonzero(eligible)
            stop = int(indices[-1] + 1) if len(indices) else 0
            start = max(0, stop - config["train_window"])
            mask = eligible[start:stop]
            training = np.where(mask, y[start:stop], np.nan)
            n = int(mask.sum())
            month = origin.tz_convert("America/Toronto").strftime("%Y-%m") if pd.notna(origin) else ""
            base = {"asset": asset, "date": row.date.date().isoformat(),
                    "origin_at": origin.isoformat() if pd.notna(origin) else "",
                    "interval_start": row.interval_start.date().isoformat() if pd.notna(row.interval_start) else "",
                    "label_available_at": row.label_available_at.isoformat() if pd.notna(row.label_available_at) else "",
                    "target": row.target, "target_invalid_reason": row.invalid_reason,
                    "unit": row.unit, "n_available": n,
                    "latest_available_label": group.label_available_at.iloc[start:stop].loc[mask].max().isoformat() if len(indices) else "",
                    "unobserved_grid_rows": int(i - stop)}
            for model in MODELS:
                status, error, params, fit_record = "passed", "", None, None
                mean, variance = np.nan, np.nan
                if pd.isna(origin):
                    status, error = "skipped", "missing_origin"
                elif n < config["minimum_train"]:
                    status, error = "skipped", "insufficient_matured_labels"
                elif model == "historical":
                    mean = float(np.nanmean(training))
                    variance = float(np.nanvar(training))
                else:
                    if model not in cached or cached[model]["month"] != month:
                        fit_record = {"fit_id": f"{asset}:{model}:{month}", "asset": asset, "model": model,
                                      "month": month, "fit_origin": origin.isoformat(),
                                      "train_start_index": start, "train_end_index": stop - 1,
                                      "train_first_date": group.date.iloc[start].date().isoformat(),
                                      "train_last_date": group.date.iloc[stop - 1].date().isoformat(),
                                      "max_label_available_at": group.label_available_at.iloc[start:stop].loc[mask].max().isoformat(),
                                      "n_train": n, "grid_rows": stop - start}
                        try:
                            if model in ("garch", "gjr_garch"):
                                params = fit_garch(training, asymmetric=model == "gjr_garch")
                            else:
                                width = len(CONTROL_COLUMNS) if model == "seasonal_variance" else x.shape[1]
                                params = fit_log_variance(x[start:stop, :width][mask], training[mask], ridge=config["ridge"])
                            fit_record.update(status="passed", parameters=params)
                        except (ValueError, RuntimeError, FloatingPointError) as exc:
                            fit_record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        cached[model] = fit_record
                        fits.append(fit_record)
                    fit_record = cached[model]
                    if fit_record["status"] != "passed":
                        status, error = "failed", fit_record["error"]
                    else:
                        try:
                            params = fit_record["parameters"]
                            mean = 0.0
                            if model in ("garch", "gjr_garch"):
                                first = fit_record["train_start_index"]
                                known = np.where(eligible[first:i + 1], y[first:i + 1], np.nan)
                                variance = float(garch_variances(known, params)[-1])
                            else:
                                width = len(CONTROL_COLUMNS) if model == "seasonal_variance" else x.shape[1]
                                variance = float(predict_log_variance(x[i:i + 1, :width], params)[0])
                            if not np.isfinite(variance) or variance <= 0:
                                raise ValueError("Nonpositive/nonfinite forecast variance")
                        except (ValueError, RuntimeError, FloatingPointError) as exc:
                            status, error = "failed", f"{type(exc).__name__}: {exc}"
                for confidence in config["confidences"]:
                    var, es = np.nan, np.nan
                    if status == "passed":
                        if model == "historical":
                            var, es = empirical_var_es(training[np.isfinite(training)].tolist(), confidence)
                        else:
                            var, es = gaussian_var_es(mean, np.sqrt(variance), confidence)
                    forecasts.append({**base, "model": model, "confidence": confidence,
                                      "status": status, "error": error,
                                      "fit_id": fit_record["fit_id"] if fit_record else "",
                                      "mean": mean, "variance": variance,
                                      "second_moment": variance + mean ** 2,
                                      "var": var, "es": es})
    return pd.DataFrame(forecasts), fits


def score_forecasts(forecasts, models=MODELS):
    """Descriptive tuning scores on dates shared by ALL baselines, per asset.

    Formal calibration tests, final holdout evaluation and model selection are
    deliberately deferred. Squared movements are noisy second-moment proxies.
    """
    rows = []
    for (asset, confidence), group in forecasts.groupby(["asset", "confidence"], sort=True):
        valid = group.loc[group.status.eq("passed") & np.isfinite(group.target)
                          & np.isfinite(group["var"]) & group.second_moment.gt(0)].copy()
        counts = valid.groupby("date").model.nunique()
        common = set(counts.index[counts.eq(len(models))])
        for model in models:
            candidate = group.loc[group.model.eq(model)]
            subset = valid.loc[valid.model.eq(model) & valid.date.isin(common)]
            losses = -subset.target.to_numpy()
            vars_ = subset["var"].to_numpy()
            pinball = [quantile_loss(float(loss), float(var), confidence) for loss, var in zip(losses, vars_)]
            qlike = np.log(subset.second_moment) + subset.target ** 2 / subset.second_moment
            rows.append({"asset": asset, "model": model, "confidence": confidence,
                         "forecast_rows": len(candidate), "passed_rows": int(candidate.status.eq("passed").sum()),
                         "failed_rows": int(candidate.status.eq("failed").sum()),
                         "skipped_rows": int(candidate.status.eq("skipped").sum()),
                         "common_scored_rows": len(subset),
                         "mean_quantile_loss": float(np.mean(pinball)) if pinball else np.nan,
                         "mean_qlike": float(qlike.mean()) if len(subset) else np.nan,
                         "unit": candidate.unit.iloc[0]})
    return pd.DataFrame(rows)


def run_baselines(config_path=ROOT / "config/baselines.json", *, input_run=None,
                  allow_latest_vintage=False, publish=True):
    config = json.loads(Path(config_path).read_text())
    validate_baseline_config(config)
    if not allow_latest_vintage:
        raise ValueError("Use --allow-latest-vintage to acknowledge revised history and assumed availability")
    source = Path(input_run) if input_run else ROOT / config["input_run"]
    if not source.is_absolute():
        source = ROOT / source
    if not (source / "observations.csv").is_file():
        raise FileNotFoundError(
            "Pinned research input observations.csv is absent. The public release omits provider data; "
            "run make verify-results for saved evidence, or restore the exact permitted input bundle "
            "as described in experiments/pilot/README.md before a full refit.")
    actual = sha(source / "observations.csv")
    if input_run is None and actual != config["input_observations_sha256"]:
        raise ValueError("Pinned input differs; provide --input-run to establish a new snapshot")
    config.update(input_run=str(source.resolve()), input_observations_sha256=actual)
    # Final-test rows are removed before values or timestamps are numerically parsed.
    observations, provenance = load_development_observations(source, config["analysis_start"], config["tuning_end"])
    provenance["modeling_rows"] = provenance.pop("development_rows")
    provenance["numeric_values_read_outside_modeling_interval"] = provenance.pop("numeric_values_read_outside_development")
    provenance["modeling_start"] = provenance.pop("analysis_start")
    provenance["modeling_end"] = provenance.pop("analysis_end")
    if provenance["input_mode"] != "live":
        raise ValueError("Phase 3 empirical baselines require a live snapshot")
    source_config = json.loads((source / "config.json").read_text())
    out = ROOT / "runs" / ("baselines-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    # Preflight dependency metadata and provenance before creating a run.
    manifest = {"phase": 3, "status": "started", "source": provenance,
                "historical_point_in_time_certified": False, "allow_latest_vintage": True,
                "final_test_numeric_values_read": False, "calendar": "retrospective observed quote grid",
                "python": sys.version, "platform": platform.platform(),
                "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "matplotlib")},
                "code_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "artifacts": {}}
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_json(out / "baseline_config.json", config)
        write_json(out / "source_snapshot.json", {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})
        shutil.copy2(ROOT / "requirements-baselines.lock.txt", out / "requirements-baselines.lock.txt")
        targets = target_availability(observations, make_targets(observations, config["prices"], config["max_gap_days"]), config["prices"])
        targets = targets.loc[targets.asset.isin(config["assets"])].reset_index(drop=True)
        features = weather_features(observations, targets, source_config["series"], allow_latest_vintage=True,
                                    minimum_reference_observations=config["weather_reference_minimum"])
        panel = regression_design(targets, features, config["weather_columns"], config["recent_window"])
        forecasts, fits = forecast_baselines(panel, config)
        scores = score_forecasts(forecasts)
        for name, frame in (("design_panel", panel), ("forecasts", forecasts), ("tuning_scores", scores)):
            frame.to_csv(out / f"{name}.csv", index=False, float_format="%.12g")
        write_json(out / "model_fits.json", fits)
        failures = int(forecasts.status.eq("failed").sum())
        garch_fits = [f for f in fits if f["model"] in ("garch", "gjr_garch") and f["status"] == "passed"]
        boundary_fits = sum(f["parameters"].get("persistence", 0) >= 0.9989 for f in garch_fits)
        manifest.update(status="passed" if not failures else "completed_with_failures",
                        garch_fits_near_persistence_cap=boundary_fits,
                        forecast_rows=len(forecasts), failed_forecast_rows=failures,
                        skipped_forecast_rows=int(forecasts.status.eq("skipped").sum()),
                        model_fits=len(fits), successful_model_fits=sum(f["status"] == "passed" for f in fits))
        report = ["# Phase 3 baseline pilot", "",
                  "**2022–2023 tuning forecasts only. The 2024–2025 final test remains unread.**", "",
                  "Latest-vintage research on a retrospective observed quote grid with assumed availability. "
                  "No executable strategy, official market calendar, or causal/predictive weather claim.", "",
                  "Targets and origins are inherited from Phase 2. Each training label requires both endpoint prices "
                  "(all four constituents for the basket) to have matured. Parameters refit monthly on up to "
                  f"{config['train_window']} original grid rows, requiring {config['minimum_train']} finite labels. "
                  "Historical simulation updates at every origin. All failures and skips remain in forecasts.csv.", "",
                  "GARCH(1,1) and GJR-GARCH(1,1) use zero-mean Gaussian quasi-likelihood. Missing/unreleased shocks "
                  "propagate expected variance; the resulting likelihood is an approximation when shocks are missing. "
                  "Normal VaR/ES at that variance are moment-matched approximations "
                  "when intervening shocks are unobserved, not exact multi-step mixture quantiles.", "",
                  "Both ridge log-variance regressions use past-only recent squared movements and annual sine/cosine. "
                  "The weather variant adds Toronto/Calgary HDD18 and CDD18. Medians, missing indicators, scaling, "
                  "and retransformation are fitted within each training window. Fixed ridge=1; no score-driven selection. "
                  "Recent history means the last 21 finite matured labels, potentially spanning more than 21 grid rows. "
                  "Refit months follow the origin, so the first January target can use a December fit.", "",
                  "Loss = −target. VaR is the upper loss quantile; empirical ES integrates exactly the upper tail mass. "
                  "Gaussian ES uses the analytic normal tail mean. Negative risk numbers are not clamped. "
                  "Units are percentage log returns or USD/barrel changes, not currency portfolio P&L.", "",
                  "## Descriptive tuning scores", "",
                  "Scores use common valid dates across all five models for each asset and confidence. Lower quantile "
                  "loss and QLIKE are better within an asset. QLIKE uses the forecast second moment and squared target "
                  "as a noisy proxy; its absolute values are not comparable across units. No formal calibration tests, "
                  "significance ranking, or final model selection is performed in Phase 3.", "",
                  table(scores, ["asset", "model", "confidence", "common_scored_rows", "mean_quantile_loss", "mean_qlike"]), "",
                  f"Fit attempts: {len(fits)}; failed forecast rows: {failures}; "
                  f"skipped forecast rows: {manifest['skipped_forecast_rows']}. "
                  f"GARCH/GJR fits near the 0.999 persistence cap: {boundary_fits}/{len(garch_fits)}. "
                  "Convergence is not evidence of model adequacy; boundary estimates warrant caution.", "",
                  "## Next gate", "",
                  "Stop before Phase 4. Retain the final holdout. Broader regime coverage, verified release/vintage history, "
                  "a prospective market calendar, investable targets, and formal risk evaluation remain separate work.", "",
                  "## Method references", "",
                  "[GARCH/GJR recurrence](https://arch.readthedocs.io/en/stable/univariate/generated/arch.univariate.GARCH.html); "
                  "[conditional variance forecasting](https://arch.readthedocs.io/en/stable/univariate/forecasting.html); "
                  "[SciPy constrained optimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)."]
        (out / "baseline_report.md").write_text("\n".join(report) + "\n")
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["artifacts"] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", manifest)
    if publish:
        published = ROOT / "reports/phase3"
        published.mkdir(parents=True, exist_ok=True)
        for name in ("baseline_report.md", "tuning_scores.csv", "baseline_config.json", "manifest.json", "model_fits.json"):
            shutil.copy2(out / name, published / name)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/baselines.json")
    parser.add_argument("--input-run", type=Path)
    parser.add_argument("--allow-latest-vintage", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args(argv)
    out = run_baselines(args.config, input_run=args.input_run,
                        allow_latest_vintage=args.allow_latest_vintage, publish=not args.no_publish)
    status = json.loads((out / "manifest.json").read_text())["status"]
    print(f"Phase 3 {status}: {out}")
    print("Tuning-period research only; final test untouched. Stop before Phase 4.")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
