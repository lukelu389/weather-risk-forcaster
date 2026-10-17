"""Verify published evidence without provider data or third-party packages.

This verifies artifact integrity, archived source, event-based VaR diagnostics,
and multiplicity adjustments. It does not refit models or rerun bootstraps.
"""

import argparse
import csv
from datetime import date
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"historical", "garch", "gjr_garch", "seasonal_variance", "weather_variance",
          "gru_control", "gru_weather", "ssm_control", "ssm_weather"}
ASSETS = {"canadian_basket", "henry_hub_spot", "wti_spot"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes root: {name}")
    return path


def check_manifest(root, manifest):
    for name, expected in manifest["artifacts"].items():
        path = checked_path(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Artifact missing or changed: {name}")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label):
    if expected in ("", None):
        if actual is not None:
            raise ValueError(f"{label}: expected unavailable, got {actual}")
    elif actual is None or not math.isclose(actual, float(expected), rel_tol=2e-9, abs_tol=1e-10):
        raise ValueError(f"{label}: recomputed {actual}, published {expected}")


def log_likelihood(k, n, probability):
    return ((k * math.log(probability) if k else 0.)
            + ((n - k) * math.log1p(-probability) if n != k else 0.))


def binomial_pvalue(k, n, probability):
    """Two-sided probability-ordering test, including tied probability masses."""
    def log_mass(j):
        return (math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                + j * math.log(probability) + (n - j) * math.log1p(-probability))
    observed = log_mass(k)
    return min(1., math.fsum(math.exp(log_mass(j)) for j in range(n + 1)
                           if log_mass(j) <= observed + math.log1p(1e-7)))


def coverage(events, confidence):
    """None retains a hole; transitions never bridge an unavailable grid row."""
    values = [value for value in events if value is not None]
    n, k = len(values), sum(values)
    if not n:
        raise ValueError("No scored events")
    counts = [0, 0, 0, 0]
    for previous, current in zip(events, events[1:]):
        if previous is not None and current is not None:
            counts[2 * previous + current] += 1
    n00, n01, n10, n11 = counts
    lr = max(0., 2 * (log_likelihood(k, n, k / n) - log_likelihood(k, n, 1 - confidence)))
    result = {"grid_rows": len(events), "common_scored_rows": n, "excluded_rows": len(events) - n,
              "exceedances": k, "exceedance_rate": k / n, "nominal_exceedances": n * (1 - confidence),
              "adjacent_pairs": sum(counts), "n00": n00, "n01": n01, "n10": n10, "n11": n11,
              "unconditional_lr": lr, "unconditional_pvalue": math.erfc(math.sqrt(lr / 2)),
              "exact_pvalue": binomial_pvalue(k, n, 1 - confidence),
              "independence_lr": None, "independence_pvalue": None,
              "conditional_lr": None, "conditional_pvalue": None}
    row0, row1 = n00 + n01, n10 + n11
    if row0 and row1:
        null = log_likelihood(n01 + n11, row0 + row1, (n01 + n11) / (row0 + row1))
        fitted = log_likelihood(n01, row0, n01 / row0) + log_likelihood(n11, row1, n11 / row1)
        independent = max(0., 2 * (fitted - null))
        result.update(independence_lr=independent, independence_pvalue=math.erfc(math.sqrt(independent / 2)),
                      conditional_lr=lr + independent, conditional_pvalue=math.exp(-(lr + independent) / 2))
    return result


def holm(values):
    """Unavailable tests retain their place in the predeclared family size."""
    result = [None] * len(values)
    finite = sorted((float(value), i) for i, value in enumerate(values) if value not in ("", None))
    running = 0.
    for rank, (value, i) in enumerate(finite):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid p-value")
        running = max(running, min(1., (len(values) - rank) * value))
        result[i] = running
    return result


def load_events(path):
    groups = {}
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["asset", "date", "grid_index", "model", "confidence", "valid", "exception"]:
            raise ValueError("Unexpected coverage evidence schema")
        for row in reader:
            key = (row["asset"], row["model"], row["confidence"])
            if row["valid"] not in {"0", "1"} or row["exception"] not in {"", "0", "1"}:
                raise ValueError("Invalid event flag")
            if (row["valid"] == "0") != (row["exception"] == ""):
                raise ValueError("Unavailable events must retain empty exception flags")
            stamp = date.fromisoformat(row["date"])
            if not date(2024, 1, 1) <= stamp <= date(2025, 12, 31):
                raise ValueError("Event outside preserved final period")
            group = groups.setdefault(key, [])
            if int(row["grid_index"]) != len(group) or (group and row["date"] <= group[-1][0]):
                raise ValueError("Duplicate, missing, or unordered original grid row")
            group.append((row["date"], None if row["valid"] == "0" else int(row["exception"])))
    expected = {(a, m, c) for a in ASSETS for m in MODELS for c in ("0.95", "0.99")}
    if set(groups) != expected:
        raise ValueError("Incomplete planned coverage family")
    for asset in ASSETS:
        reference = [(d, e is not None) for d, e in groups[(asset, "historical", "0.95")]]
        for (a, _, _), group in groups.items():
            if a == asset and [(d, e is not None) for d, e in group] != reference:
                raise ValueError("Models/confidences disagree on common scoring grid")
    return groups


def verify(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "evidence/manifest.json").read_text())
    check_manifest(root, manifest)
    archive = root / "experiments/pilot"
    seal = json.loads((archive / "docs/phase-5-seal.json").read_text())
    available, omitted = 0, []
    for name, expected in seal["files"].items():
        path = checked_path(archive, name)
        if name.startswith("runs/"):
            omitted.append(name)
        elif not path.is_file() or digest(path) != expected:
            raise ValueError(f"Archived sealed source changed: {name}")
        else:
            available += 1
    if available != 20 or len(omitted) != 4:
        raise ValueError("Unexpected archived seal scope")
    for folder in sorted((root / "reports").iterdir()):
        if not folder.is_dir():
            continue
        public = json.loads((folder / "manifest.json").read_text())
        if public.get("manifest_kind") != "published_subset":
            raise ValueError(f"Not a published-subset manifest: {folder.name}")
        check_manifest(root if public.get("artifact_root") == "repository" else folder, public)
    groups = load_events(root / "evidence/coverage-events.csv.gz")
    primary = read_csv(root / "reports/phase5/calibration_scores.csv")
    if len(primary) != 54 or len({(r["asset"], r["model"], r["confidence"]) for r in primary}) != 54:
        raise ValueError("Expected 54 distinct calibration rows")
    computed = []
    for row in primary:
        key = (row["asset"], row["model"], row["confidence"])
        result = coverage([event for _, event in groups[key]], float(row["confidence"]))
        for field, value in result.items():
            close(value, row[field], f"{key} {field}")
        computed.append(result)
    for family in ("exact", "independence", "conditional"):
        adjusted = holm([r[family + "_pvalue"] for r in computed])
        for value, row in zip(adjusted, primary):
            close(value, row[family + "_pvalue_holm"], family + " Holm")
    counts = {}
    for phase in ("phase5", "phase5-sensitivity"):
        scores = read_csv(root / "reports" / phase / "calibration_scores.csv")
        paired = read_csv(root / "reports" / phase / "paired_comparisons.csv")
        if len(scores) != 54 or len(paired) != 165:
            raise ValueError("Changed hypothesis family size")
        for rows, raw, adjusted in ((scores, "es_pvalue", "es_pvalue_holm"), (paired, "pvalue", "pvalue_holm")):
            for value, row in zip(holm([r[raw] for r in rows]), rows):
                close(value, row[adjusted], phase + " " + adjusted)
        counts[phase] = {"es_rejections": sum(float(r["es_pvalue_holm"]) < .05 for r in scores),
                         "paired_rejections": sum(float(r["pvalue_holm"]) < .05 for r in paired),
                         "matched_weather_rejections": sum(float(r["pvalue_holm"]) < .05 for r in paired
                                                            if r["reference"] != "historical")}
    expected_counts = {"phase5": {"es_rejections": 0, "paired_rejections": 0, "matched_weather_rejections": 0},
                       "phase5-sensitivity": {"es_rejections": 3, "paired_rejections": 1, "matched_weather_rejections": 0}}
    if counts != expected_counts:
        raise ValueError("Published headline differs from verified tables")
    return {"status": "passed", "public_artifacts": len(manifest["artifacts"]),
            "sealed_source_files": available, "private_sealed_inputs_not_bundled": len(omitted),
            "coverage_event_rows": sum(map(len, groups.values())), "coverage_rows_recomputed": len(primary),
            "rejections_at_5pct": counts,
            "scope": "Integrity, event-based VaR statistics, and saved-p-value Holm adjustments; no model refits, bootstrap reruns, or raw-data validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root), indent=2, sort_keys=True))
    except (ValueError, KeyError, OSError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
