"""Phase 4: matched sequence/state-space variance models on tuning data only."""

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
import torch

from .eda import ROOT, sha, table, write_json
from .eda_core import load_development_observations, make_targets
from .eda_weather import weather_features
from .baselines import (CONTROL_COLUMNS, MODELS as BASELINES, validate_baseline_config,
                        target_availability, regression_design, score_forecasts)
from .risk import gaussian_var_es
from .sequence_models import fit_sequence_variance, predict_sequence_variance

MODELS = ("gru_control", "gru_weather", "ssm_control", "ssm_weather")
ALL_MODELS = BASELINES + MODELS


def validate_ml_config(config):
    validate_baseline_config(config)
    for key in ("sequence_length", "epochs", "gru_hidden_size", "ssm_hidden_size"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not np.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    seeds = config["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")


def sequence_inputs(group, columns, length, *, use_weather=True):
    """Return current + preceding original-grid feature snapshots, not labels.

    A row snapshot is eligible only if its original origin is defined and no
    later than the prediction's origin. Missing steps and leading padding stay
    NaN. Context can precede the rolling label-fit window by length-1 rows.
    """
    if length < 1 or not group.date.is_monotonic_increasing or group.date.duplicated().any():
        raise ValueError("Need positive sequence length and unique chronological grid")
    values = group[columns].to_numpy(dtype=float, copy=True)
    if not use_weather:
        values[:, len(CONTROL_COLUMNS):] = np.nan  # Remove values AND missingness patterns.
    result = np.full((len(group), length, len(columns)), np.nan)
    for i, origin in enumerate(group.origin_at):
        if pd.isna(origin):
            continue
        start = max(0, i - length + 1)
        window = values[start:i + 1].copy()
        allowed = (group.origin_at.iloc[start:i + 1] <= origin).to_numpy()
        window[~allowed] = np.nan
        result[i, -len(window):] = window
    return result


def forecast_sequences(panel, config, output_dir, progress=None):
    """Monthly fits on matured labels, with fixed-seed variance ensembles."""
    output_dir = Path(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    forecasts, fits = [], []
    for asset in config["assets"]:
        group = panel.loc[panel.asset.eq(asset)].sort_values("date").reset_index(drop=True)
        if group.empty:
            raise ValueError(f"No targets for {asset}")
        columns = CONTROL_COLUMNS + config["weather_columns"]
        tensors = {weather: sequence_inputs(group, columns, config["sequence_length"], use_weather=weather)
                   for weather in (False, True)}
        y = group.target.to_numpy(dtype=float)
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
            n = int(mask.sum())
            month = origin.tz_convert("America/Toronto").strftime("%Y-%m") if pd.notna(origin) else ""
            latest = group.label_available_at.iloc[start:stop].loc[mask].max().isoformat() if n else ""
            base = {"asset": asset, "date": row.date.date().isoformat(),
                    "origin_at": origin.isoformat() if pd.notna(origin) else "",
                    "interval_start": row.interval_start.date().isoformat() if pd.notna(row.interval_start) else "",
                    "label_available_at": row.label_available_at.isoformat() if pd.notna(row.label_available_at) else "",
                    "target": row.target, "target_invalid_reason": row.invalid_reason,
                    "unit": row.unit, "n_available": n, "latest_available_label": latest,
                    "unobserved_grid_rows": int(i - stop)}
            for model in MODELS:
                status, error, fit_record, variance = "passed", "", None, np.nan
                architecture, variant = model.split("_")
                X = tensors[variant == "weather"]
                if pd.isna(origin):
                    status, error = "skipped", "missing_origin"
                elif n < config["minimum_train"]:
                    status, error = "skipped", "insufficient_matured_labels"
                else:
                    if model not in cached or cached[model][0]["month"] != month:
                        fit_record = {"fit_id": f"{asset}:{model}:{month}", "asset": asset, "model": model,
                                      "month": month, "fit_origin": origin.isoformat(),
                                      "train_start_index": start, "train_end_index": stop - 1,
                                      "n_train": n, "grid_rows": stop - start,
                                      "max_label_available_at": latest, "members": []}
                        members = []
                        try:
                            for seed in config["seeds"]:
                                metadata, checkpoint = fit_sequence_variance(
                                    X[start:stop][mask], y[start:stop][mask], architecture=architecture,
                                    hidden_size=config[f"{architecture}_hidden_size"], epochs=config["epochs"],
                                    learning_rate=config["learning_rate"], seed=seed)
                                name = f"checkpoints/{asset}__{model}__{month}__{seed}.pt"
                                torch.save(checkpoint, output_dir / name)
                                fit_record["members"].append({"metadata": metadata, "checkpoint": name,
                                                              "checkpoint_sha256": sha(output_dir / name)})
                                members.append((metadata, checkpoint))
                            fit_record["status"] = "passed"
                        except (ValueError, RuntimeError, FloatingPointError) as exc:
                            fit_record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        fits.append(fit_record)
                        cached[model] = (fit_record, members)
                        if progress:
                            progress(f"{fit_record['fit_id']}: {fit_record['status']} ({n} labels)")
                    fit_record, members = cached[model]
                    if fit_record["status"] != "passed":
                        status, error = "failed", fit_record["error"]
                    else:
                        try:
                            variances = [float(predict_sequence_variance(X[i:i + 1], metadata, checkpoint)[0])
                                         for metadata, checkpoint in members]
                            variance = float(np.mean(variances))
                            if not np.isfinite(variance) or variance <= 0:
                                raise ValueError("Nonpositive/nonfinite ensemble variance")
                        except (ValueError, RuntimeError, FloatingPointError) as exc:
                            status, error = "failed", f"{type(exc).__name__}: {exc}"
                for confidence in config["confidences"]:
                    var, es = gaussian_var_es(0.0, np.sqrt(variance), confidence) if status == "passed" else (np.nan, np.nan)
                    forecasts.append({**base, "model": model, "confidence": confidence, "status": status,
                                      "error": error, "fit_id": fit_record["fit_id"] if fit_record else "",
                                      "mean": 0.0, "variance": variance, "second_moment": variance,
                                      "var": var, "es": es})
    return pd.DataFrame(forecasts), fits


def load_baseline_forecasts(config, source_hash):
    """Verify preserved Phase 3 forecasts and matched study conventions."""
    folder = ROOT / config["baseline_run"]
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("phase") != 3 or manifest.get("status") != "passed":
        raise ValueError("Need a passed Phase 3 baseline run")
    if manifest["source"]["observations_sha256"] != source_hash:
        raise ValueError("Baseline and ML input snapshots differ")
    for name in ("forecasts.csv", "baseline_config.json"):
        if sha(folder / name) != manifest["artifacts"].get(name):
            raise ValueError(f"Baseline checksum mismatch: {name}")
    if sha(folder / "forecasts.csv") != config["baseline_forecasts_sha256"]:
        raise ValueError("Pinned baseline forecasts differ")
    baseline_config = json.loads((folder / "baseline_config.json").read_text())
    for key in ("prices", "assets", "analysis_start", "analysis_end", "tuning_start", "tuning_end",
                "test_start", "test_end", "max_gap_days", "train_window", "minimum_train",
                "recent_window", "refit", "confidences", "weather_columns", "weather_reference_minimum"):
        if config[key] != baseline_config[key]:
            raise ValueError(f"Unmatched baseline protocol: {key}")
    forecasts = pd.read_csv(folder / "forecasts.csv")
    if set(forecasts.model) != set(BASELINES):
        raise ValueError("Incomplete baseline model set")
    return forecasts, {"run_dir": str(folder.resolve()), "manifest_sha256": sha(folder / "manifest.json"),
                       "forecasts_sha256": sha(folder / "forecasts.csv")}


def run_ml(config_path=ROOT / "config/ml.json", *, allow_latest_vintage=False, publish=True, progress=None):
    config = json.loads(Path(config_path).read_text())
    validate_ml_config(config)
    if not allow_latest_vintage:
        raise ValueError("Use --allow-latest-vintage to acknowledge revised history and assumed availability")
    source = ROOT / config["input_run"]
    actual = sha(source / "observations.csv")
    if actual != config["input_observations_sha256"]:
        raise ValueError("Pinned input snapshot differs")
    baseline_forecasts, baseline_provenance = load_baseline_forecasts(config, actual)
    observations, provenance = load_development_observations(source, config["analysis_start"], config["tuning_end"])
    if provenance["input_mode"] != "live":
        raise ValueError("Phase 4 requires a live-data snapshot")
    provenance["modeling_rows"] = provenance.pop("development_rows")
    provenance["numeric_values_read_outside_modeling_interval"] = provenance.pop("numeric_values_read_outside_development")
    provenance["modeling_start"] = provenance.pop("analysis_start")
    provenance["modeling_end"] = provenance.pop("analysis_end")
    source_config = json.loads((source / "config.json").read_text())
    out = ROOT / "runs" / ("ml-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"phase": 4, "status": "started", "source": provenance, "baseline_source": baseline_provenance,
                "allow_latest_vintage": True, "historical_point_in_time_certified": False,
                "final_test_numeric_values_read": False, "calendar": "retrospective observed quote grid",
                "python": sys.version, "platform": platform.platform(),
                "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "torch")},
                "code_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}, "artifacts": {}}
    write_json(out / "ml_config.json", config)
    write_json(out / "source_snapshot.json", {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})
    shutil.copy2(ROOT / "requirements-ml.lock.txt", out / "requirements-ml.lock.txt")
    try:
        targets = target_availability(observations, make_targets(observations, config["prices"], config["max_gap_days"]), config["prices"])
        targets = targets.loc[targets.asset.isin(config["assets"])].reset_index(drop=True)
        weather = weather_features(observations, targets, source_config["series"], allow_latest_vintage=True,
                                   minimum_reference_observations=config["weather_reference_minimum"])
        panel = regression_design(targets, weather, config["weather_columns"], config["recent_window"])
        neural_forecasts, fits = forecast_sequences(panel, config, out, progress)
        forecasts = pd.concat([baseline_forecasts, neural_forecasts], ignore_index=True)
        if forecasts.duplicated(["asset", "date", "model", "confidence"]).any():
            raise ValueError("Duplicate comparison forecast rows")
        scores = score_forecasts(forecasts, models=ALL_MODELS)
        for name, frame in (("design_panel", panel), ("forecasts", forecasts), ("tuning_scores", scores)):
            frame.to_csv(out / f"{name}.csv", index=False, float_format="%.12g")
        write_json(out / "model_fits.json", fits)
        failures = int(neural_forecasts.status.eq("failed").sum())
        manifest.update(status="passed" if not failures else "completed_with_failures",
                        neural_forecast_rows=len(neural_forecasts), combined_forecast_rows=len(forecasts),
                        failed_forecast_rows=failures, skipped_forecast_rows=int(neural_forecasts.status.eq("skipped").sum()),
                        ensemble_fits=len(fits), successful_ensemble_fits=sum(f["status"] == "passed" for f in fits),
                        member_fits=sum(len(f["members"]) for f in fits))
        report = ["# Phase 4 sequence and state-space pilot", "",
                  "**2022–2023 tuning only. Final-test 2024–2025 values remain unread.**", "",
                  "One small GRU and one stable diagonal state-space model each have matched weather/control variants. "
                  "The state-space model is a simple learned diagonal recurrence, not Mamba. "
                  "All use the same historical volatility/seasonality controls. Weather adds the same four HDD/CDD features as Phase 3.", "",
                  f"Input histories contain {config['sequence_length']} original-grid feature snapshots ending at the origin. "
                  "Missing target rows remain sequence steps. Each historical snapshot was constructed at its own origin; "
                  "context can extend before the rolling fit-label window. Weather ablation masks values and missingness before preprocessing.", "",
                  f"Both architectures train for {config['epochs']} full-batch Adam updates at learning rate {config['learning_rate']}, "
                  f"from fixed seeds {config['seeds']}, with monthly refits. "
                  "No early stopping, best-seed selection, or test-period tuning is used. Parameter counts and loss trajectories "
                  "are saved per member; identical epoch budgets do not imply identical compute or convergence.", "",
                  "All fits use matured labels and training-only imputation, scaling and target normalization. "
                  "CPU float64, one thread and deterministic PyTorch operations support same-environment repeatability. "
                  "Checkpoints contain tensor state dictionaries and are hashed in the manifest.", "",
                  "The two seed variances are averaged into a zero-mean Gaussian forecast. VaR/ES use this moment-matched normal, "
                  "not the exact Gaussian-mixture tails. All latest-vintage, calendar, geography, survivor-basket and tail assumptions remain.", "",
                  "## Common-date tuning comparison", "",
                  "All nine models share the same scored dates within each asset/confidence. Lower pinball loss and QLIKE are better "
                  "within an asset; no significance test or final model selection is implied.", "",
                  table(scores, ["asset", "model", "confidence", "common_scored_rows", "mean_quantile_loss", "mean_qlike"]), "",
                  f"Ensemble fits: {len(fits)}; constituent fits: {manifest['member_fits']}; failed forecast rows: {failures}.", "",
                  "## Next phase", "",
                  "Phase 5 must freeze model/evaluation settings before reading the final period and report calibration, uncertainty, "
                  "and forecast-comparison limitations. Keep a separate audit of the first final-test evaluation.", "",
                  "## Implementation references", "",
                  "[PyTorch GRU](https://docs.pytorch.org/docs/2.10/generated/torch.nn.GRU.html), "
                  "[Gaussian likelihood](https://docs.pytorch.org/docs/2.10/generated/torch.nn.GaussianNLLLoss.html), "
                  "[determinism limits](https://docs.pytorch.org/docs/2.10/notes/randomness.html)."]
        (out / "ml_report.md").write_text("\n".join(report) + "\n")
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["artifacts"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", manifest)
    if publish:
        published = ROOT / "reports/phase4"
        published.mkdir(parents=True, exist_ok=True)
        for name in ("ml_report.md", "tuning_scores.csv", "ml_config.json", "manifest.json", "model_fits.json"):
            shutil.copy2(out / name, published / name)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/ml.json")
    parser.add_argument("--allow-latest-vintage", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args(argv)
    out = run_ml(args.config, allow_latest_vintage=args.allow_latest_vintage,
                 publish=not args.no_publish, progress=lambda message: print(message, flush=True))
    status = json.loads((out / "manifest.json").read_text())["status"]
    print(f"Phase 4 {status}: {out}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
