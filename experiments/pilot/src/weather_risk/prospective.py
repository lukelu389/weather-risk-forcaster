"""An explicit, future-only evaluation gate; independent of the sealed pilot.

This module freezes a protocol and its nominated local dependencies. It does not
fit models, acquire data, certify point-in-time data, or choose an experiment.
"""

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CONSUMED_START = date(2024, 1, 1)
CONSUMED_END = date(2025, 12, 31)
GATE_MODULE = "src/weather_risk/prospective.py"


def _utc_now():
    return datetime.now(timezone.utc)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            return invalid_constant(value)
        return parsed
    return json.loads(path.read_text(), object_pairs_hook=_json_object,
                      parse_constant=invalid_constant, parse_float=finite_float)


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError(f"{label} requires exactly: {expected}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _integer(value, minimum, label):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _day(value):
    if not isinstance(value, str):
        raise ValueError("Dates must use YYYY-MM-DD")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return parsed


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Timestamps must be ISO text with an explicit timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamps require an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _path(root, name, *, existing=False):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Use a canonical relative path: {name!r}")
    candidate = root
    for part in name.split("/"):
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Symlink paths are not allowed: {name}")
    if existing and not candidate.is_file():
        raise ValueError(f"Missing regular file: {name}")
    if candidate.exists() and not candidate.is_file():
        raise ValueError(f"Expected a regular file: {name}")
    return candidate


def _config_name(config_path, root):
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    try:
        name = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Protocol must be inside the project root") from exc
    _path(root, name, existing=True)
    return name


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path, value):
    """Publish a complete JSON file atomically, without replacing any file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate(protocol, frozen_day, root, config_name):
    _keys(protocol, "schema_version status protocol_id evaluation hypotheses models refit maturity "
          "inference files holdout outputs", "Protocol")
    if type(protocol["schema_version"]) is not int or protocol["schema_version"] != 1:
        raise ValueError("Unsupported protocol schema_version")
    if protocol["status"] != "ready":
        raise ValueError("Only an explicitly ready protocol can be frozen or verified")
    _text(protocol["protocol_id"], "protocol_id")
    period = protocol["evaluation"]
    _keys(period, "start end access_not_before", "evaluation")
    start, end, access = (_day(period[key]) for key in ("start", "end", "access_not_before"))
    if end < start:
        raise ValueError("Evaluation end precedes start")
    if start <= CONSUMED_END and end >= CONSUMED_START:
        raise ValueError("The consumed 2024–2025 holdout cannot be reused as untouched")
    if start <= frozen_day:
        raise ValueError("A prospective evaluation must start after the UTC freeze date")
    if access <= end:
        raise ValueError("access_not_before must be after the complete evaluation period")

    models = protocol["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("Explicit model names and settings are required")
    for name, settings in models.items():
        _text(name, "model name")
        if not isinstance(settings, dict) or not settings:
            raise ValueError(f"Explicit settings required for model {name}")
    refit = protocol["refit"]
    _keys(refit, "schedule window_rows minimum_training_rows", "refit")
    _text(refit["schedule"], "refit schedule")
    for key in ("window_rows", "minimum_training_rows"):
        _integer(refit[key], 1, key)
    if refit["minimum_training_rows"] > refit["window_rows"]:
        raise ValueError("minimum_training_rows exceeds the refit window")
    maturity = protocol["maturity"]
    _keys(maturity, "target_delay_days forecast_origin availability_policy "
          "prior_holdout_labels_in_later_fits", "maturity")
    _integer(maturity["target_delay_days"], 0, "target_delay_days")
    for key in ("forecast_origin", "availability_policy"):
        _text(maturity[key], key)
    if type(maturity["prior_holdout_labels_in_later_fits"]) is not bool:
        raise ValueError("prior_holdout_labels_in_later_fits must be boolean")
    if (access - end).days < max(1, maturity["target_delay_days"]):
        raise ValueError("access_not_before does not allow the declared target maturity delay")

    inference = protocol["inference"]
    _keys(inference, "alpha adjustment family_sizes bootstrap_families bootstrap missing_policy", "inference")
    alpha = inference["alpha"]
    if type(alpha) not in (float, int) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if inference["adjustment"] != "holm":
        raise ValueError("This gate supports declared Holm families")
    _text(inference["missing_policy"], "missing_policy")
    families = inference["family_sizes"]
    if not isinstance(families, dict) or not families:
        raise ValueError("Declare every Holm family size")
    for name, count in families.items():
        _text(name, "family name")
        _integer(count, 1, "family size")
    hypotheses = protocol["hypotheses"]
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("Declare the individual hypotheses before freezing")
    counts, ids = {name: 0 for name in families}, set()
    for hypothesis in hypotheses:
        _keys(hypothesis, "id family statement", "hypothesis")
        for key in ("id", "family", "statement"):
            _text(hypothesis[key], key)
        if hypothesis["id"] in ids or hypothesis["family"] not in counts:
            raise ValueError("Duplicate hypothesis id or undeclared family")
        ids.add(hypothesis["id"])
        counts[hypothesis["family"]] += 1
    if counts != families:
        raise ValueError("Declared family sizes must equal their enumerated hypothesis counts")
    bootstrap_families = inference["bootstrap_families"]
    if (not isinstance(bootstrap_families, list) or not bootstrap_families
            or any(not isinstance(name, str) for name in bootstrap_families)
            or len(set(bootstrap_families)) != len(bootstrap_families)
            or not set(bootstrap_families).issubset(families)):
        raise ValueError("Declare the distinct families using bootstrap p-values")
    bootstrap = inference["bootstrap"]
    _keys(bootstrap, "replications minimum_valid_fraction block_length minimum_observations seed", "bootstrap")
    for key in ("replications", "block_length", "minimum_observations", "seed"):
        _integer(bootstrap[key], 0 if key == "seed" else 1, key)
    fraction = bootstrap["minimum_valid_fraction"]
    if type(fraction) not in (float, int) or not 0 < fraction <= 1:
        raise ValueError("minimum_valid_fraction must be in (0, 1]")
    if bootstrap["minimum_observations"] < bootstrap["block_length"]:
        raise ValueError("minimum_observations must cover one bootstrap block")
    valid = math.ceil(bootstrap["replications"] * fraction)
    if any(1 / (valid + 1) > alpha / families[name] for name in bootstrap_families):
        raise ValueError("Bootstrap resolution cannot reach the first Holm threshold")

    _keys(protocol["files"], "code config input", "files")
    nominated = []
    for group, names in protocol["files"].items():
        if not isinstance(names, list) or not names:
            raise ValueError(f"Nominate explicit {group} files")
        for name in names:
            _path(root, name, existing=True)
            nominated.append(name)
    if len(nominated) != len(set(nominated)) or config_name in nominated:
        raise ValueError("Nominated files must be distinct; the protocol itself is sealed automatically")
    if GATE_MODULE not in protocol["files"]["code"]:
        raise ValueError("Nominate the prospective gate implementation as code")
    _keys(protocol["outputs"], "seal first_access", "outputs")
    _keys(protocol["holdout"], "path date_column availability_column series_column series_ids numeric_columns", "holdout")
    holdout = protocol["holdout"]
    for key in ("date_column", "availability_column", "series_column"):
        _text(holdout[key], key)
    for key in ("series_ids", "numeric_columns"):
        values = holdout[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Explicit {key} are required")
        for value in values:
            _text(value, key)
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {key}")
    columns = [holdout[key] for key in ("date_column", "availability_column", "series_column")] + holdout["numeric_columns"]
    if len(columns) != len(set(columns)):
        raise ValueError("Holdout metadata and numeric columns must be distinct")
    generated = [holdout["path"], *protocol["outputs"].values()]
    for name in generated:
        _path(root, name)
    all_names = nominated + [config_name] + generated
    if len(set(all_names)) != len(all_names):
        raise ValueError("Holdout, outputs, protocol and nominated inputs must have distinct paths")
    return nominated + [config_name]


def freeze_protocol(config_path, *, root=ROOT):
    """Seal ready settings before a future period; never overwrite prior records."""
    root = Path(root).resolve()
    name = _config_name(config_path, root)
    protocol = _read_json(root / name)
    now = _utc_now()
    names = _validate(protocol, now.date(), root, name)
    if _path(root, protocol["outputs"]["first_access"]).exists():
        raise ValueError("Access already recorded; cannot relabel this protocol untouched")
    if _path(root, protocol["holdout"]["path"]).exists():
        raise ValueError("The future holdout snapshot must be acquired after the protocol is frozen")
    seal = {"schema_version": 1, "protocol_id": protocol["protocol_id"], "protocol_path": name,
            "frozen_at_utc": now.isoformat(), "holdout_numeric_values_parsed": False,
            "files": {item: _sha(root / item) for item in names}}
    path = _path(root, protocol["outputs"]["seal"])
    _exclusive_json(path, seal)
    return path


def verify_protocol(config_path, *, root=ROOT):
    """Check the exact nominated dependency set and its bytes, including the gate."""
    root = Path(root).resolve()
    name = _config_name(config_path, root)
    protocol = _read_json(root / name)
    # Validate output syntax before using any protocol-controlled path.
    _keys(protocol.get("outputs"), "seal first_access", "outputs")
    seal = _read_json(_path(root, protocol["outputs"]["seal"], existing=True))
    _keys(seal, "schema_version protocol_id protocol_path frozen_at_utc holdout_numeric_values_parsed files", "seal")
    frozen = _timestamp(seal["frozen_at_utc"])
    if frozen > _utc_now():
        raise ValueError("Seal timestamp is in the future")
    names = _validate(protocol, frozen.date(), root, name)
    if (seal["schema_version"] != 1 or seal["protocol_id"] != protocol["protocol_id"]
            or seal["protocol_path"] != name or seal["holdout_numeric_values_parsed"] is not False
            or not isinstance(seal["files"], dict) or set(seal["files"]) != set(names)):
        raise ValueError("Seal does not match the exact protocol dependency set")
    for item in names:
        if _sha(_path(root, item, existing=True)) != seal["files"][item]:
            raise ValueError(f"Sealed file changed: {item}")
    return protocol, seal


def _authorized_payload(config_path, root):
    protocol, _ = verify_protocol(config_path, root=root)
    now = _utc_now()
    if now.date() < _day(protocol["evaluation"]["access_not_before"]):
        raise ValueError("Holdout access is not yet permitted by the frozen calendar gate")
    source = _path(root, protocol["holdout"]["path"], existing=True)
    # Hash bytes before any CSV/numeric parse, and parse these same bytes later.
    payload = source.read_bytes()
    seal_hash = _sha(_path(root, protocol["outputs"]["seal"], existing=True))
    source_hash = hashlib.sha256(payload).hexdigest()
    audit = {"protocol_id": protocol["protocol_id"], "seal_sha256": seal_hash,
             "holdout_path": protocol["holdout"]["path"], "holdout_sha256": source_hash,
             "first_access_attempt_at_utc": now.isoformat(),
             "meaning": "Recorded before CSV/numeric parsing; even a failed parse consumes this gate."}
    audit_path = _path(root, protocol["outputs"]["first_access"])
    try:
        _exclusive_json(audit_path, audit)
    except FileExistsError:
        saved = _read_json(audit_path)
        if not isinstance(saved, dict) or set(saved) != set(audit):
            raise ValueError("Invalid first-access audit")
        timestamp = _timestamp(saved["first_access_attempt_at_utc"])
        if (timestamp > now or timestamp.date() < _day(protocol["evaluation"]["access_not_before"])
                or any(saved[key] != audit[key] for key in audit if key != "first_access_attempt_at_utc")):
            raise ValueError("First-access audit does not match the sealed protocol and holdout snapshot")
        audit = saved
    return protocol, audit, payload, now


def record_first_access(config_path, *, root=ROOT):
    """Bind one acquired snapshot for all reproductions, before parsing its values."""
    _, audit, _, _ = _authorized_payload(config_path, Path(root).resolve())
    return audit


def _numeric(value):
    if value == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite holdout numeric value")
    return parsed


def load_holdout_csv(config_path, *, root=ROOT):
    """Authorize first, filter metadata second, parse only eligible numeric cells.

    Blank numeric cells remain None. This gate does not apply forecasting-time
    feature/label eligibility: the nominated research runner must enforce that.
    """
    protocol, audit, payload, _ = _authorized_payload(config_path, Path(root).resolve())
    # Reproductions keep the first access cutoff even if a later run occurs
    # after another row in this same snapshot becomes available.
    as_of = _timestamp(audit["first_access_attempt_at_utc"])
    spec, period = protocol["holdout"], protocol["evaluation"]
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = [spec[key] for key in ("date_column", "availability_column", "series_column")] + spec["numeric_columns"]
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)) or not set(required).issubset(fields):
        raise ValueError("Holdout CSV has missing or duplicate columns")
    start, end = _day(period["start"]), _day(period["end"])
    rows, excluded = [], {"outside_evaluation": 0, "unselected_series": 0, "not_yet_available": 0}
    for row in reader:
        day = _day(row[spec["date_column"]])
        if not start <= day <= end:
            excluded["outside_evaluation"] += 1
            continue
        if row[spec["series_column"]] not in spec["series_ids"]:
            excluded["unselected_series"] += 1
            continue
        if _timestamp(row[spec["availability_column"]]) > as_of:
            excluded["not_yet_available"] += 1
            continue
        if None in row or any(row[name] is None for name in required):
            raise ValueError("Malformed eligible holdout row")
        for column in spec["numeric_columns"]:
            row[column] = _numeric(row[column])
        rows.append(row)
    return {"rows": rows, "excluded": excluded, "audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "read"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze_protocol(args.config, root=args.root))
    elif args.command == "verify":
        protocol, _ = verify_protocol(args.config, root=args.root)
        print(f"Verified prospective protocol: {protocol['protocol_id']}")
    else:
        result = load_holdout_csv(args.config, root=args.root)
        print(json.dumps({"loaded_rows": len(result["rows"]), "excluded": result["excluded"]}, sort_keys=True))


if __name__ == "__main__":
    main()
