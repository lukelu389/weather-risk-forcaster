"""Export categorical coverage evidence from the preserved private forecast CSV.

No prices, returns, forecast magnitudes, or fitted weights are exported. This
is a post-exposure transparency artifact, not a new evaluation. The pinned CSV
hash prevents silently exporting a different run. Public verification does not
require this private input or re-execution of this exporter.
"""
import argparse
import csv
import gzip
import hashlib
import io
import math
from pathlib import Path

FORECAST_SHA256 = "5231450b7b938ea455e7bf37c88b4f353bdbba2957ffb00ba10a74b9bd55ef75"


def number(value):
    return float(value) if value else float("nan")


def export(source, destination):
    if hashlib.sha256(source.read_bytes()).hexdigest() != FORECAST_SHA256:
        raise ValueError("Source is not the preserved Phase 5 forecast artifact")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        key = (row["asset"], row["confidence"], row["date"])
        groups.setdefault(key, []).append(row)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["asset", "date", "grid_index", "model", "confidence", "valid", "exception"])
    assets = sorted({r["asset"] for r in rows})
    for asset in assets:
        dates = sorted({r["date"] for r in rows if r["asset"] == asset})
        for confidence in ("0.95", "0.99"):
            for index, stamp in enumerate(dates):
                group = groups[(asset, confidence, stamp)]
                if len(group) != 9 or len({r["model"] for r in group}) != 9:
                    raise ValueError("Incomplete or duplicate model grid")
                valid = all(r["status"] == "passed" and all(math.isfinite(number(r[k])) for k in
                            ("target", "var", "es", "second_moment"))
                            and number(r["es"]) >= number(r["var"]) and number(r["second_moment"]) > 0
                            for r in group)
                if len({r["target"] for r in group}) != 1 or len({r["interval_start"] for r in group}) != 1:
                    raise ValueError("Models disagree on targets or intervals")
                for row in sorted(group, key=lambda r: r["model"]):
                    hit = int(-number(row["target"]) > number(row["var"])) if valid else ""
                    writer.writerow([asset, stamp, index, row["model"], confidence, int(valid), hit])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(output.getvalue().encode(), mtime=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
