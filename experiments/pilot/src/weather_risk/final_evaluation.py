"""Phase 5: sealed, chronological final evaluation of the nine fixed models."""

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
from .eda_core import load_development_observations, make_targets
from .eda_weather import weather_features
from .baselines import forecast_baselines, target_availability, regression_design
from .ml import ALL_MODELS, forecast_sequences, validate_ml_config
from .evaluation import (coverage_tests, fz0_score, es_identification_residual,
                         moving_block_bootstrap, holm_adjust)

SEALED_MODULES = ("__init__.py", "final_evaluation.py", "evaluation.py", "eda.py", "eda_core.py",
                  "eda_weather.py", "eda_plots.py", "baselines.py", "baseline_models.py", "ml.py",
                  "sequence_models.py", "risk.py", "timing.py", "schema.py", "validation.py")


def sealed_paths(config_path, config, model):
    paths = [Path(config_path).resolve(), ROOT / config["model_config"], ROOT / config["baseline_config"],
             ROOT / config["protocol_document"], ROOT / "requirements-ml.lock.txt"]
    paths += [ROOT / "src/weather_risk" / name for name in SEALED_MODULES]
    paths += [ROOT / model["input_run"] / name for name in
              ("observations.csv", "config.json", "manifest.json", "validation.json")]
    return paths


def read_configs(config_path):
    config = json.loads(Path(config_path).read_text())
    model = json.loads((ROOT / config["model_config"]).read_text())
    baseline = json.loads((ROOT / config["baseline_config"]).read_text())
    validate_ml_config(model)
    for key, value in baseline.items():
        if model.get(key) != value:
            raise ValueError(f"Baseline/ML settings differ: {key}")
    if config["models"] != list(ALL_MODELS):
        raise ValueError("Final evaluation requires all nine models in fixed order")
    if config["assets"] != model["assets"]:
        raise ValueError("Final evaluation must preserve all configured assets")
    if (config["test_start"], config["test_end"]) != (model["test_start"], model["test_end"]):
        raise ValueError("Final interval differs from frozen model protocol")
    if config["comparison_metrics"] != ["quantile_loss", "qlike", "fz0"]:
        raise ValueError("Unexpected comparison metrics")
    expected_pairs = [[m, "historical"] for m in ALL_MODELS if m != "historical"] + [
        ["weather_variance", "seasonal_variance"], ["gru_weather", "gru_control"], ["ssm_weather", "ssm_control"]]
    if config["comparison_pairs"] != expected_pairs:
        raise ValueError("Unexpected comparison family")
    for key, expected in (("bootstrap_seed", 1729), ("bootstrap_block_length", 20),
                          ("bootstrap_replications", 1000), ("bootstrap_minimum_observations", 100),
                          ("significance_level", 0.05)):
        if config[key] != expected:
            raise ValueError(f"Unexpected frozen evaluation setting: {key}")
    return config, model


def freeze_protocol(config_path=ROOT / "config/evaluation.json"):
    """Seal local code/settings before first access; never overwrite a seal."""
    config, model = read_configs(config_path)
    if (ROOT / config["first_access_audit"]).exists():
        raise ValueError("Final access already recorded; cannot create a new untouched-test seal")
    source = ROOT / model["input_run"] / "observations.csv"
    if sha(source) != model["input_observations_sha256"]:
        raise ValueError("Pinned input snapshot differs")
    paths = sealed_paths(config_path, config, model)
    seal = {"phase": 5, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "final_test_numeric_values_read_at_freeze": False,
            "input_observations_sha256": sha(source),
            "files": {str(p.relative_to(ROOT)): sha(p) for p in paths}}
    path = ROOT / config["seal"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    return path


def verify_seal(config_path):
    config, model = read_configs(config_path)
    path = ROOT / config["seal"]
    seal = json.loads(path.read_text())
    required = {str(p.relative_to(ROOT)) for p in sealed_paths(config_path, config, model)}
    if not required.issubset(seal["files"]) or seal.get("final_test_numeric_values_read_at_freeze") is not False:
        raise ValueError("Incomplete final-evaluation seal")
    for name, digest in seal["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Sealed file changed: {name}; disclose any post-exposure correction")
    if sha(ROOT / model["input_run"] / "observations.csv") != seal["input_observations_sha256"]:
        raise ValueError("Sealed source snapshot differs")
    return config, model, seal


def record_access(config):
    """Record the attempt before parsing holdout values, including failed runs."""
    path = ROOT / config["first_access_audit"]
    digest = sha(ROOT / config["seal"])
    audit = {"phase": 5, "first_access_attempt_at_utc": datetime.now(timezone.utc).isoformat(),
             "seal_sha256": digest, "test_start": config["test_start"], "test_end": config["test_end"],
             "meaning": "Written before first final-period numeric parse; a failed attempt still consumes the gate."}
    try:
        with path.open("x") as stream:
            stream.write(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        # Parallel reproductions should be launched after one explicit access record.
        audit = json.loads(path.read_text())
        if audit["seal_sha256"] != digest:
            raise ValueError("First-access audit belongs to a different protocol")
    return audit


def evaluate_forecasts(forecasts, config):
    """Common-date scores and tests, retaining holes on each original asset grid."""
    keys = ["asset", "date", "model", "confidence"]
    if forecasts.duplicated(keys).any() or set(forecasts.model) != set(ALL_MODELS):
        raise ValueError("Duplicate forecasts or incomplete model set")
    expected_groups = {(asset, confidence) for asset in config["assets"] for confidence in (0.95, 0.99)}
    if set(zip(forecasts.asset, forecasts.confidence)) != expected_groups:
        raise ValueError("Incomplete planned asset/confidence groups")
    date_values = pd.to_datetime(forecasts.date, errors="raise")
    if not date_values.between(config["test_start"], config["test_end"]).all():
        raise ValueError("Forecast dates outside final evaluation interval")
    grids = {asset: sorted(group.date.unique()) for asset, group in forecasts.groupby("asset")}
    rows, comparisons = [], []
    for (asset, confidence), group in forecasts.groupby(["asset", "confidence"], sort=True):
        dates = grids[asset]
        if any(set(group.loc[group.model.eq(m), "date"]) != set(dates) for m in ALL_MODELS):
            raise ValueError("Incomplete model/confidence grid; retain missing forecasts as explicit status rows")
        frames = {m: group.loc[group.model.eq(m)].set_index("date").reindex(dates) for m in ALL_MODELS}
        reference = frames["historical"]
        for frame in frames.values():
            if not np.allclose(frame.target, reference.target, rtol=0, atol=0, equal_nan=True):
                raise ValueError("Models disagree on realized targets")
            if not frame.interval_start.fillna("").equals(reference.interval_start.fillna("")):
                raise ValueError("Models disagree on target intervals")
        common = np.ones(len(dates), dtype=bool)
        for frame in frames.values():
            common &= (frame.status.eq("passed") & np.isfinite(frame.target) & np.isfinite(frame["var"])
                       & np.isfinite(frame.es) & frame.es.ge(frame["var"])
                       & np.isfinite(frame.second_moment) & frame.second_moment.gt(0)).to_numpy()
        metric_arrays = {}
        for model, frame in frames.items():
            loss = np.where(common, -frame.target.to_numpy(), np.nan)
            q = np.where(common, frame["var"].to_numpy(), np.nan)
            e = np.where(common & frame.es.gt(0).to_numpy(), frame.es.to_numpy(), np.nan)
            scores = {"quantile_loss": (confidence - (loss < q)) * (loss - q),
                      "qlike": np.where(common, np.log(frame.second_moment) + frame.target ** 2 / frame.second_moment, np.nan),
                      "fz0": fz0_score(loss, q, e, confidence)}
            metric_arrays[model] = scores
            coverage = coverage_tests(loss, q, confidence)
            es_result = moving_block_bootstrap(es_identification_residual(loss, q, e, confidence),
                                              **bootstrap_settings(config), key=f"es:{asset}:{model}:{confidence}")
            row = {"asset": asset, "model": model, "confidence": confidence,
                   "grid_rows": len(dates), "common_scored_rows": int(common.sum()),
                   "excluded_rows": int((~common).sum()),
                   "missing_realized_rows": int((~np.isfinite(reference.target)).sum()),
                   "nonpositive_es_rows": int((common & frame.es.le(0).to_numpy()).sum()),
                   "exceedances": coverage["exceedances"], "exceedance_rate": coverage["exceedance_rate"],
                   "nominal_exceedances": int(common.sum()) * (1 - confidence),
                   "coverage_ci_low": coverage["exact"]["ci_low"], "coverage_ci_high": coverage["exact"]["ci_high"],
                   "adjacent_pairs": coverage["adjacent_pairs"], **coverage["transitions"],
                   "es_residual_mean": es_result["mean"], "es_residual_ci_low": es_result["ci_low"],
                   "es_residual_ci_high": es_result["ci_high"], "es_pvalue": es_result["pvalue"],
                   "es_reason": es_result["reason"], "es_score_rows": es_result["observations"],
                   "es_bootstrap_valid": es_result["valid_replications"]}
            for diagnostic in ("unconditional", "exact", "independence", "conditional"):
                row[f"{diagnostic}_pvalue"] = coverage[diagnostic]["pvalue"]
                row[f"{diagnostic}_reason"] = coverage[diagnostic]["reason"]
                if "statistic" in coverage[diagnostic]:
                    row[f"{diagnostic}_lr"] = coverage[diagnostic]["statistic"]
            for name, values in scores.items():
                valid = np.isfinite(values)
                row[f"mean_{name}"] = float(np.mean(np.asarray(values)[valid])) if valid.any() else None
                row[f"{name}_rows"] = int(valid.sum())
            rows.append(row)
        for candidate, reference_model in config["comparison_pairs"]:
            for metric in config["comparison_metrics"]:
                if metric == "qlike" and confidence != 0.95:
                    continue
                diff = metric_arrays[candidate][metric] - metric_arrays[reference_model][metric]
                result = moving_block_bootstrap(diff, **bootstrap_settings(config),
                                               key=f"score:{asset}:{candidate}:{reference_model}:{confidence}:{metric}")
                comparisons.append({"asset": asset, "candidate": candidate, "reference": reference_model,
                                    "confidence": confidence, "metric": metric, **result})
    scores, paired = pd.DataFrame(rows), pd.DataFrame(comparisons)
    for family in ("exact", "independence", "conditional", "es"):
        scores[f"{family}_pvalue_holm"] = holm_adjust(scores[f"{family}_pvalue"].tolist())
    paired["pvalue_holm"] = holm_adjust(paired.pvalue.tolist())
    return scores, paired


def bootstrap_settings(config):
    return {"block_length": config["bootstrap_block_length"], "replications": config["bootstrap_replications"],
            "seed": config["bootstrap_seed"], "min_observations": config["bootstrap_minimum_observations"]}


def plot_results(scores, paired, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .eda_plots import ASSET_LABELS

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    for row, asset in enumerate(sorted(scores.asset.unique())):
        for col, confidence in enumerate((0.95, 0.99)):
            ax = axes[row, col]
            data = scores.loc[scores.asset.eq(asset) & scores.confidence.eq(confidence)].set_index("model").reindex(ALL_MODELS)
            rate = data.exceedance_rate.to_numpy() * 100
            low, high = data.coverage_ci_low.to_numpy() * 100, data.coverage_ci_high.to_numpy() * 100
            y = np.arange(len(data))
            ax.errorbar(rate, y, xerr=[rate - low, high - rate], fmt="o", color="#367DA3", capsize=3)
            ax.axvline(100 * (1 - confidence), color="#C87832", linestyle="--", label="Nominal")
            ax.set_yticks(y, ALL_MODELS, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{ASSET_LABELS[asset]} · {confidence:.0%} VaR", loc="left", fontsize=10)
            ax.set_xlabel("Exceedance rate (%) · exact 95% binomial interval", fontsize=8)
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("Final period 2024–2025 · VaR coverage\nLatest-vintage pilot; binomial intervals assume independent exceedances", fontsize=13)
    save_plot(fig, output, "01_var_coverage")
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), layout="constrained")
    for row, asset in enumerate(sorted(scores.asset.unique())):
        for col, confidence in enumerate((0.95, 0.99)):
            ax = axes[row, col]
            data = paired.loc[paired.asset.eq(asset) & paired.confidence.eq(confidence)
                              & paired.metric.eq("quantile_loss") & ~paired.reference.eq("historical")]
            y = np.arange(len(data))
            means = data["mean"].to_numpy()
            ax.hlines(y, data.ci_low, data.ci_high, color="#367DA3", linewidth=2)
            ax.plot(means, y, "o", color="#18354A")
            ax.axvline(0, color="#C87832", linestyle="--")
            ax.set_yticks(y, ["Regression", "GRU", "Diagonal SSM"], fontsize=9)
            ax.invert_yaxis()
            ax.set_title(f"{ASSET_LABELS[asset]} · {confidence:.0%}", loc="left", fontsize=10)
            ax.set_xlabel("Weather minus control quantile loss (asset units)", fontsize=8)
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("Final period 2024–2025 · Incremental weather scores\nNegative favors weather; pointwise 95% block intervals, not multiplicity-adjusted", fontsize=13)
    save_plot(fig, output, "02_weather_comparisons")


def save_plot(fig, output, stem):
    import matplotlib.pyplot as plt
    for ext in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(output / f"{stem}.{ext}", dpi=150, facecolor="white", bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def write_report(out, scores, paired, manifest):
    weather = paired.loc[paired.metric.eq("quantile_loss") & ~paired.reference.eq("historical")]
    sections = ["# Phase 5 final evaluation", "",
        "**2024–2025 final period, under the sealed protocol. Latest-vintage research pilot; no point-in-time trading claim.**", "",
        "All nine models and training settings were retained before first final-period access. Rolling refits can use earlier "
        "final-period labels only after maturity. These results must not be used to retune and then re-label the same period as untouched.", "",
        f"Forecast rows: {manifest['forecast_rows']}; failed: {manifest['failed_forecast_rows']}; skipped: {manifest['skipped_forecast_rows']}.", "",
        "## Common-date scores and coverage", "",
        table(scores, ["asset", "model", "confidence", "common_scored_rows", "exceedances", "exceedance_rate",
                       "mean_quantile_loss", "mean_qlike", "mean_fz0"]), "",
        "![VaR coverage](01_var_coverage.png)", "",
        "## Paired weather comparisons", "",
        "Candidate-minus-control quantile loss: negative favors weather. Intervals are pointwise, from 20-row blocks; "
        "Holm p-values use all 165 prespecified score comparisons. See paired_comparisons.csv for all metrics and pairs.", "",
        table(weather, ["asset", "candidate", "confidence", "mean", "ci_low", "ci_high", "pvalue_holm"]), "",
        "![Weather comparisons](02_weather_comparisons.png)", "",
        "## Calibration and uncertainty", "",
        "calibration_scores.csv contains exact binomial, Kupiec, transition independence, conditional coverage and relative ES "
        "residual diagnostics, including raw/adjusted p-values, intervals and unavailable reasons. Tests use shared valid dates; "
        "missing grid positions break transition pairs and remain missing within bootstrap blocks. ES/FZ0 require positive ES. "
        "Each diagnostic has its own 54-test Holm family. Binomial tests assume independent hits; LR references and the block "
        "bootstrap are approximate. Sparse 99% tails, missing quotes, nonstationarity and repeated model fitting limit power. "
        "Failure to reject is not model approval; score improvement does not establish calibration.", "",
        "## Provenance and limitations", "",
        "protocol_seal.json and first_access_audit.json preserve the local pre-access records. Source snapshot, dependency lock, "
        "all forecasts, fitted parameters and 600 neural checkpoints are in the full run. Runs use the same preserved data; "
        "replays are reproduction, not additional holdout experiments. The observed quote grid, variable intervals, assumed "
        "availability, revised history, selected-survivor basket and Gaussian tails remain. No 2008/2014–2016 evaluation, "
        "currency portfolio P&L, futures roll, transaction costs or execution claim is supported here.", "",
        "See protocol.md for the prespecified formulas, families and primary sources."]
    (out / "evaluation_report.md").write_text("\n".join(sections) + "\n")


def run_evaluation(config_path=ROOT / "config/evaluation.json", *, allow_latest_vintage=False,
                   publish=True, progress=None):
    if not allow_latest_vintage:
        raise ValueError("Use --allow-latest-vintage to acknowledge revised history and assumed availability")
    config, model, seal = verify_seal(config_path)
    audit = record_access(config)
    out = ROOT / "runs" / ("evaluation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"phase": 5, "status": "started", "allow_latest_vintage": True,
                "historical_point_in_time_certified": False, "final_test_numeric_values_read": False,
                "protocol_seal_sha256": sha(ROOT / config["seal"]), "python": sys.version,
                "platform": platform.platform(),
                "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "torch", "matplotlib")},
                "code_sha256": {Path(n).name: v for n, v in seal["files"].items() if n.startswith("src/")},
                "artifacts": {}}
    write_json(out / "evaluation_config.json", config)
    write_json(out / "model_config.json", model)
    write_json(out / "protocol_seal.json", seal)
    write_json(out / "first_access_audit.json", audit)
    write_json(out / "source_snapshot.json", {n: (ROOT / n).read_text() for n in seal["files"] if n.startswith("src/")})
    shutil.copy2(ROOT / config["protocol_document"], out / "protocol.md")
    shutil.copy2(ROOT / "requirements-ml.lock.txt", out / "requirements-ml.lock.txt")
    try:
        source = ROOT / model["input_run"]
        manifest["final_test_numeric_values_read"] = True  # Conservative even if parsing fails.
        observations, provenance = load_development_observations(source, model["analysis_start"], model["test_end"])
        provenance["modeling_rows"] = provenance.pop("development_rows")
        provenance["numeric_values_read_outside_modeling_interval"] = provenance.pop("numeric_values_read_outside_development")
        provenance["modeling_start"] = provenance.pop("analysis_start")
        provenance["modeling_end"] = provenance.pop("analysis_end")
        manifest["source"] = provenance
        if provenance["input_mode"] != "live":
            raise ValueError("Final evaluation requires the pinned live snapshot")
        source_config = json.loads((source / "config.json").read_text())
        targets = target_availability(observations, make_targets(observations, model["prices"], model["max_gap_days"]), model["prices"])
        targets = targets.loc[targets.asset.isin(model["assets"])].reset_index(drop=True)
        weather = weather_features(observations, targets, source_config["series"], allow_latest_vintage=True,
                                   minimum_reference_observations=model["weather_reference_minimum"])
        panel = regression_design(targets, weather, model["weather_columns"], model["recent_window"])
        final_window = {**model, "tuning_start": model["test_start"], "tuning_end": model["test_end"]}
        if progress:
            progress("Fitting fixed baselines on final-period origins")
        baseline, baseline_fits = forecast_baselines(panel, final_window)
        neural, neural_fits = forecast_sequences(panel, final_window, out, progress)
        forecasts = pd.concat([baseline, neural], ignore_index=True)
        scores, paired = evaluate_forecasts(forecasts, config)
        for name, frame in (("design_panel", panel), ("forecasts", forecasts),
                            ("calibration_scores", scores), ("paired_comparisons", paired)):
            frame.to_csv(out / f"{name}.csv", index=False, float_format="%.12g")
        write_json(out / "model_fits.json", {"baselines": baseline_fits, "neural": neural_fits})
        failures = int(forecasts.status.eq("failed").sum())
        manifest.update(status="passed" if not failures else "completed_with_failures", forecast_rows=len(forecasts),
                        failed_forecast_rows=failures, skipped_forecast_rows=int(forecasts.status.eq("skipped").sum()),
                        baseline_fits=len(baseline_fits), ensemble_fits=len(neural_fits),
                        member_fits=sum(len(f["members"]) for f in neural_fits), paired_comparisons=len(paired))
        plot_results(scores, paired, out)
        write_report(out, scores, paired, manifest)
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["artifacts"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", manifest)
    if publish and manifest["status"] == "passed":
        report = ROOT / "reports/phase5"
        report.mkdir(parents=True, exist_ok=True)
        for name in ("evaluation_report.md", "calibration_scores.csv", "paired_comparisons.csv", "manifest.json",
                     "evaluation_config.json", "protocol.md", "protocol_seal.json", "first_access_audit.json",
                     "01_var_coverage.png", "01_var_coverage.pdf", "02_weather_comparisons.png", "02_weather_comparisons.pdf"):
            shutil.copy2(out / name, report / name)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/evaluation.json")
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--allow-latest-vintage", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze_protocol:
        print(freeze_protocol(args.config))
        return 0
    out = run_evaluation(args.config, allow_latest_vintage=args.allow_latest_vintage, progress=lambda x: print(x, flush=True))
    print(out)
    return 0 if json.loads((out / "manifest.json").read_text())["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
