"""Disclosed post-exposure bootstrap resolution correction of saved Phase 5 forecasts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .eda import ROOT, sha, write_json
from .final_evaluation import SEALED_MODULES, evaluate_forecasts


_PRIMARY_ARTIFACTS = (
    "forecasts.csv", "evaluation_config.json", "calibration_scores.csv",
    "paired_comparisons.csv", "protocol_seal.json", "source_snapshot.json",
    "requirements-ml.lock.txt",
)
_SCORE_KEYS = ["asset", "model", "confidence"]
_PAIRED_KEYS = ["asset", "candidate", "reference", "confidence", "metric"]
_CHANGED_SCORE_COLUMNS = {
    "es_residual_ci_low", "es_residual_ci_high", "es_pvalue", "es_pvalue_holm",
    "es_bootstrap_valid", "es_reason",
}
_CHANGED_PAIRED_COLUMNS = {
    "ci_low", "ci_high", "pvalue", "pvalue_holm", "replications",
    "valid_replications", "available", "reason",
}


def _read_primary(config_path):
    """Verify pinned provenance and sealed evaluator code before any CSV parse."""
    config = json.loads(Path(config_path).read_text())
    if config.get("replications") != 19999 or isinstance(config["replications"], bool):
        raise ValueError("The disclosed correction requires exactly 19999 replications")
    if not isinstance(config.get("reason"), str) or not config["reason"].strip():
        raise ValueError("A post-exposure correction reason is required")
    protocol = ROOT / config["protocol_document"]
    if sha(protocol) != config["protocol_document_sha256"]:
        raise ValueError("Sensitivity protocol document hash differs")
    source = ROOT / config["input_run"]
    if sha(source / "manifest.json") != config["input_manifest_sha256"]:
        raise ValueError("Pinned Phase 5 manifest hash differs")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("phase") != 5 or manifest.get("status") != "passed":
        raise ValueError("Sensitivity requires a passed Phase 5 evaluation")
    for name in _PRIMARY_ARTIFACTS:
        if manifest.get("artifacts", {}).get(name) != sha(source / name):
            raise ValueError(f"Primary artifact hash differs: {name}")
    if manifest.get("protocol_seal_sha256") != sha(source / "protocol_seal.json"):
        raise ValueError("Primary protocol seal does not match the manifest")
    seal = json.loads((source / "protocol_seal.json").read_text())
    snapshot = json.loads((source / "source_snapshot.json").read_text())
    for name in SEALED_MODULES:
        relative = f"src/weather_risk/{name}"
        expected = manifest.get("code_sha256", {}).get(name)
        archived = hashlib.sha256(snapshot.get(relative, "").encode()).hexdigest()
        if (not expected or seal.get("files", {}).get(relative) != expected
                or archived != expected or sha(ROOT / relative) != expected):
            raise ValueError(f"Primary sealed evaluator code differs: {name}")
    if sha(ROOT / "requirements-ml.lock.txt") != sha(source / "requirements-ml.lock.txt"):
        raise ValueError("Primary dependency lock differs")
    evaluation_config = json.loads((source / "evaluation_config.json").read_text())
    if (evaluation_config.get("bootstrap_replications") != 1000
            or evaluation_config.get("bootstrap_seed") != 1729
            or evaluation_config.get("bootstrap_block_length") != 20
            or evaluation_config.get("bootstrap_minimum_observations") != 100
            or evaluation_config.get("significance_level") != .05):
        raise ValueError("Saved evaluation settings differ from the disclosed primary protocol")
    return config, source, manifest, evaluation_config


def _check_invariants(primary, updated, *, keys, changed_columns, expected_rows):
    """Reject changed families/counts; tolerate only 1e-8 CSV arithmetic drift."""
    if len(primary) != expected_rows or len(updated) != expected_rows:
        raise ValueError(f"Expected the complete {expected_rows}-row hypothesis family")
    if (set(primary.columns) != set(updated.columns)
            or primary.duplicated(keys).any() or updated.duplicated(keys).any()):
        raise ValueError("Diagnostic columns or unique hypotheses differ")
    first = primary.sort_values(keys).reset_index(drop=True)
    second = updated.sort_values(keys).reset_index(drop=True)
    if not first[keys].equals(second[keys]):
        raise ValueError("Prespecified hypothesis identities differ")
    maximum_error = 0.
    checked_columns = []
    for name in primary.columns:
        if name in changed_columns or name in keys:
            continue
        left, right = first[name], second[name]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            # Integers include all sample, tail, transition and score counts.
            if pd.api.types.is_integer_dtype(left) and pd.api.types.is_integer_dtype(right):
                valid = np.array_equal(left.to_numpy(), right.to_numpy())
            else:
                a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
                valid = np.allclose(a, b, rtol=0, atol=1e-8, equal_nan=True)
                finite = np.isfinite(a) & np.isfinite(b)
                if finite.any():
                    maximum_error = max(maximum_error, float(np.max(np.abs(a[finite] - b[finite]))))
        else:
            valid = left.fillna("").equals(right.fillna(""))
        if not valid:
            raise ValueError(f"Primary counts or nonbootstrap diagnostic changed: {name}")
        checked_columns.append(name)
    return {"rows": expected_rows, "checked_columns": checked_columns,
            "maximum_absolute_numeric_difference": maximum_error,
            "numeric_tolerance": 1e-8, "integer_counts_match_exactly": True}


def _rejection_summary(primary_scores, primary_paired, scores, paired):
    rows = []
    for label, old, new, column, size in (
        ("ES residual", primary_scores, scores, "es_pvalue_holm", 54),
        ("Paired scores", primary_paired, paired, "pvalue_holm", 165),
    ):
        rows.append({
            "family": label, "hypotheses": size,
            "primary_adjusted_rejections_5pct": int(old[column].lt(.05).sum()),
            "sensitivity_adjusted_rejections_5pct": int(new[column].lt(.05).sum()),
            "sensitivity_unavailable_tests": int(new[column].isna().sum()),
            "primary_minimum_raw_pvalue": 1 / 1001,
            "sensitivity_minimum_raw_pvalue": 1 / 20000,
            "first_holm_threshold_5pct": .05 / size,
        })
    return rows


def run_sensitivity(config_path=ROOT / "config/bootstrap_sensitivity.json", *, publish=True, progress=None):
    """Recompute all 219 bootstrap tests from pinned forecasts without refits.

    This is a post-exposure computational resolution correction. It preserves
    the primary experiment and writes a separate run/report. The original
    saved forecasts have 12 significant digits; invariance checks explicitly
    allow small arithmetic differences from primary in-memory calculations.
    """
    config, source, primary_manifest, primary_config = _read_primary(config_path)
    effective = {**primary_config, "bootstrap_replications": config["replications"]}
    out = ROOT / "runs" / ("bootstrap-sensitivity-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "phase": "5-bootstrap-sensitivity", "status": "started",
        "analysis_type": "post_exposure_computational_resolution_correction",
        "untouched_confirmatory_analysis": False, "models_refitted": False,
        "primary_manifest_sha256": config["input_manifest_sha256"],
        "primary_protocol_seal_sha256": primary_manifest["protocol_seal_sha256"],
        "replications": config["replications"],
        "code_sha256": {"bootstrap_sensitivity.py": sha(Path(__file__))},
        "primary_evaluator_code_sha256": primary_manifest["code_sha256"], "artifacts": {},
    }
    write_json(out / "sensitivity_config.json", config)
    write_json(out / "effective_evaluation_config.json", effective)
    for original, copied in (("manifest.json", "primary_manifest.json"),
                             ("protocol_seal.json", "primary_protocol_seal.json"),
                             ("source_snapshot.json", "primary_source_snapshot.json"),
                             ("evaluation_config.json", "primary_evaluation_config.json"),
                             ("requirements-ml.lock.txt", "requirements-ml.lock.txt")):
        shutil.copyfile(source / original, out / copied)
    shutil.copyfile(ROOT / config["protocol_document"], out / "protocol_addendum.md")
    shutil.copyfile(Path(__file__), out / "bootstrap_sensitivity.py")
    try:
        forecasts = pd.read_csv(source / "forecasts.csv")
        primary_scores = pd.read_csv(source / "calibration_scores.csv")
        primary_paired = pd.read_csv(source / "paired_comparisons.csv")
        if progress:
            progress(f"Recomputing all 54 ES tests and 165 score comparisons with {config['replications']} resamples")
        scores, paired = evaluate_forecasts(forecasts, effective)
        checks = {
            "calibration": _check_invariants(primary_scores, scores, keys=_SCORE_KEYS,
                changed_columns=_CHANGED_SCORE_COLUMNS, expected_rows=54),
            "paired": _check_invariants(primary_paired, paired, keys=_PAIRED_KEYS,
                changed_columns=_CHANGED_PAIRED_COLUMNS, expected_rows=165),
        }
        for name, frame in (("calibration_scores", scores), ("paired_comparisons", paired)):
            frame.to_csv(out / f"{name}.csv", index=False, float_format="%.12g")
        summary = _rejection_summary(primary_scores, primary_paired, scores, paired)
        write_json(out / "rejection_summary.json", summary)
        provenance = {
            "analysis_type": manifest["analysis_type"], "reason": config["reason"],
            "primary_run": config["input_run"],
            "primary_manifest_sha256": config["input_manifest_sha256"],
            "primary_protocol_seal_sha256": primary_manifest["protocol_seal_sha256"],
            "primary_artifact_sha256": {name: primary_manifest["artifacts"][name] for name in _PRIMARY_ARTIFACTS},
            "protocol_addendum_sha256": config["protocol_document_sha256"],
            "changed_evaluation_setting": {"bootstrap_replications": {"before": 1000, "after": 19999}},
            "source_csv_significant_digits": 12, "invariance_checks": checks,
            "models_refitted": False, "forecasts_retuned": False,
            "interpretation": "Post-exposure sensitivity, not a new untouched confirmatory experiment. "
                "CSV rounding can change arithmetic within the documented tolerance; counts must match exactly.",
        }
        write_json(out / "provenance.json", provenance)
        lines = ["# Phase 5 bootstrap resolution sensitivity", "",
            "**Disclosed post-exposure computational correction; not an untouched confirmatory analysis.**", "",
            config["reason"], "",
            "All 54 ES-residual tests and all 165 paired-score comparisons are recomputed using 19,999 resamples. "
            "Models, forecasts, seeds, block length, missing positions and hypothesis families are preserved. "
            "No models were refitted. The primary output remains unchanged.", "",
            "At B=1,000, the smallest attainable p-value was 1/1001, so neither Holm family could reject at 5%. "
            "The new minimum is 1/20000. Original zero-rejection counts therefore cannot establish adequate "
            "ES calibration or equivalent skill. Both sets of intervals are pointwise, not familywise.", "",
            "| Family | Primary 5% Holm rejections | Sensitivity 5% Holm rejections | Unavailable |",
            "|---|---:|---:|---:|"]
        lines.extend(f"| {r['family']} | {r['primary_adjusted_rejections_5pct']} | "
                     f"{r['sensitivity_adjusted_rejections_5pct']} | {r['sensitivity_unavailable_tests']} |"
                     for r in summary)
        lines += ["", "The input forecasts were saved at 12 significant digits. Sample and tail counts match the "
            "primary results exactly, and all checked means and nonbootstrap diagnostics agree within 1e-8. "
            "See provenance.json for measured differences and hashes. Bootstrap uncertainty remains approximate "
            "under dependence, nonstationarity, sparse tails and latest-vintage data. Failure to reject does not "
            "establish equivalence or model approval.", ""]
        (out / "sensitivity_report.md").write_text("\n".join(lines))
        manifest.update(status="passed", forecast_rows=len(forecasts), calibration_tests=len(scores),
                        paired_comparisons=len(paired))
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["artifacts"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", manifest)
    if publish:
        report = ROOT / "reports/phase5-sensitivity"
        report.mkdir(parents=True, exist_ok=True)
        for name in ("sensitivity_report.md", "calibration_scores.csv", "paired_comparisons.csv",
                     "sensitivity_config.json", "effective_evaluation_config.json", "provenance.json",
                     "rejection_summary.json", "protocol_addendum.md", "manifest.json"):
            shutil.copyfile(out / name, report / name)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/bootstrap_sensitivity.json")
    args = parser.parse_args(argv)
    print(run_sensitivity(args.config, progress=lambda message: print(message, flush=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
