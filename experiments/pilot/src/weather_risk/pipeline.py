"""Single-command, auditable Phase 1 ingestion and alignment pipeline."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys

from .cache import Cache
from .demo import synthetic_observations
from .providers import fetch_series
from .schema import Observation, assumed_availability
from .timing import build_feature_panel
from .validation import validate_observations

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    first, last = date.fromisoformat(config["start"]), date.fromisoformat(config["end"])
    if first > last:
        raise ValueError("start must be no later than end")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    if not config["series"]:
        raise ValueError("at least one series is required")
    ids = [s["series_id"] for s in config["series"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate series_id in config")
    for spec in config["series"]:
        if spec["provider"] not in {"fred", "boc", "eccc", "eia", "yahoo"}:
            raise ValueError(f"Unsupported provider: {spec['provider']}")
        if spec["frequency"] not in {"daily", "business_daily", "weekly", "monthly"}:
            raise ValueError("Unsupported frequency")
        if spec["stale_days"] < 0:
            raise ValueError("stale_days must be nonnegative")
        assumed_availability(config["start"], spec["timezone"], spec["lag_days"], spec["available_hour"])
    return config


def forecast_grid(config: dict) -> list[str]:
    """Weekday diagnostic grid only; not an exchange session calendar."""
    current, last = date.fromisoformat(config["start"]), date.fromisoformat(config["end"])
    times = []
    while current <= last:
        if current.weekday() < 5:
            times.append(assumed_availability(current.isoformat(), config["forecast_timezone"],
                                             0, config["forecast_hour"]))
        current += timedelta(days=1)
    return times


def run(config: dict, mode: str, output_root: Path, *, offline=False,
        refresh=False, allow_latest_vintage=False, pinned_records=None,
        replay_skips: dict[str, str] | None = None) -> Path:
    if mode not in {"demo", "live"}:
        raise ValueError("mode must be demo or live")
    if offline and refresh:
        raise ValueError("offline and refresh cannot be combined")
    random.seed(config["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_root / "runs" / f"{mode}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "source_snapshot.json", {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})
    manifest = {
        "phase": 1, "mode": mode, "seed": config["seed"], "started_at": stamp,
        "status": "started", "python": sys.version, "platform": platform.platform(),
        "allow_latest_vintage": allow_latest_vintage, "offline": offline, "refresh": refresh,
        "config_sha256": digest(run_dir / "config.json"),
        "code_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "packages": {}, "sources": [], "skipped_sources": [], "skipped_series": {}, "artifacts": {},
        "historical_point_in_time_certified": False,
        "scope": "SYNTHETIC INTEGRATION DEMO" if mode == "demo" else "LATEST-VINTAGE DATA STAGING",
    }
    for package in ("yfinance", "pandas", "numpy"):
        try:
            manifest["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    cache = Cache(output_root / "data" / "raw", offline=offline, refresh=refresh,
                  pinned_records=pinned_records)
    try:
        rows = []
        source_errors = []
        if mode == "demo":
            rows = synthetic_observations(config)
        else:
            for spec in config["series"]:
                if replay_skips and spec["series_id"] in replay_skips:
                    message = replay_skips[spec["series_id"]]
                    manifest["skipped_sources"].append(message)
                    manifest["skipped_series"][spec["series_id"]] = message
                    continue
                print(f"Fetching {spec['series_id']} ({spec['provider']})", flush=True)
                try:
                    rows.extend(fetch_series(spec, config["start"], config["end"], cache))
                except Exception as exc:
                    # Providers redact secrets before raising; never log credentials here.
                    message = f"{spec['series_id']}: {type(exc).__name__}: {exc}"
                    if spec.get("required", True):
                        source_errors.append(message)
                    else:
                        manifest["skipped_sources"].append(message)
                        manifest["skipped_series"][spec["series_id"]] = message
        manifest["sources"] = cache.records
        write_csv(run_dir / "observations.csv", [asdict(r) for r in rows], list(Observation.__dataclass_fields__))
        present = {r.series_id for r in rows}
        active_specs = [s for s in config["series"] if s.get("required", True) or s["series_id"] in present]
        report = validate_observations(rows, active_specs, config["start"], config["end"])
        report["errors"].extend(source_errors)
        if not rows:
            report["errors"].append("No observations retrieved")
        report["warnings"].extend(manifest["skipped_sources"])
        report["passed"] = not report["errors"]
        if mode == "live":
            report["warnings"].append("Latest-vintage downloads do not establish historical availability; assumed lags are research hypotheses.")
        write_json(run_dir / "validation.json", report)
        if not report["passed"]:
            raise ValueError(f"Data validation failed: {len(report['errors'])} error(s); inspect {run_dir / 'validation.json'}")
        panel = build_feature_panel(rows, forecast_grid(config), active_specs,
                                    allow_latest_vintage=allow_latest_vintage)
        fields = ["forecast_at", "series_id", "value", "observation_date", "available_at", "age_days", "vintage", "quality_flag", "missing_reason"]
        write_csv(run_dir / "feature_panel.csv", panel, fields)
        manifest["alignment"] = {
            "rows": len(panel), "missing_reasons": dict(Counter(p["missing_reason"] for p in panel)),
            "grid": "Weekdays at configured forecast time; exchange holidays NOT modeled",
            "strict_vintage_gate": not allow_latest_vintage,
        }
        manifest["status"] = "passed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["sources"] = cache.records
        manifest["artifacts"] = {p.name: digest(p) for p in sorted(run_dir.iterdir()) if p.name != "manifest.json"}
        write_json(run_dir / "manifest.json", manifest)
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["demo", "live"], default="demo")
    parser.add_argument("--config", type=Path, default=ROOT / "config/demo.json")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--offline", action="store_true", help="Read cached requests only")
    parser.add_argument("--refresh", action="store_true", help="Fetch a new snapshot; retain old blobs")
    parser.add_argument("--allow-latest-vintage", action="store_true", help="Explicit research-only opt-in for revised historical data")
    parser.add_argument("--series", action="append", help="Select one series ID; repeat for multiple")
    parser.add_argument("--replay", type=Path, help="Replay an earlier manifest with its exact config and cached snapshots")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        pinned_records = None
        replay_skips = None
        if args.replay:
            if args.refresh or args.series or args.allow_latest_vintage:
                raise ValueError("Replay uses saved settings; do not combine with refresh, series selection or vintage override")
            prior = json.loads(args.replay.read_text())
            if prior["status"] != "passed":
                raise ValueError("Only a successful run can be replayed")
            saved_config = args.replay.parent / "config.json"
            if digest(saved_config) != prior["config_sha256"]:
                raise ValueError("Saved config checksum mismatch")
            code = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
            if code != prior["code_sha256"]:
                raise ValueError("Code changed since saved run; restore its source snapshot before exact replay")
            config = load_config(saved_config)
            args.mode, args.offline = prior["mode"], True
            args.allow_latest_vintage = prior["allow_latest_vintage"]
            pinned_records = prior["sources"]
            replay_skips = prior["skipped_series"]
        if args.series:
            known = {s["series_id"] for s in config["series"]}
            if set(args.series) - known:
                raise ValueError(f"Unknown series: {sorted(set(args.series) - known)}")
            config["series"] = [s for s in config["series"] if s["series_id"] in args.series]
        result = run(config, args.mode, args.output_root, offline=args.offline,
                     refresh=args.refresh, allow_latest_vintage=args.allow_latest_vintage,
                     pinned_records=pinned_records, replay_skips=replay_skips)
        if args.replay:
            manifest = json.loads((result / "manifest.json").read_text())
            mismatched = [name for name in ("observations.csv", "feature_panel.csv", "validation.json", "config.json")
                          if digest(result / name) != prior["artifacts"][name]]
            manifest["replay_of"] = str(args.replay.resolve())
            manifest["replay_verified"] = not mismatched
            if mismatched:
                manifest["status"] = "failed"
                manifest["failure"] = f"Replay artifact mismatch: {mismatched}"
            write_json(result / "manifest.json", manifest)
            if mismatched:
                raise ValueError(manifest["failure"])
        print(f"Phase 1 passed ({args.mode}). Artifacts: {result}")
        print("No models fitted. Demo is synthetic; live data are not historical point-in-time certified.")
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"Phase 1 failed: {exc}", file=sys.stderr)
        return 1
