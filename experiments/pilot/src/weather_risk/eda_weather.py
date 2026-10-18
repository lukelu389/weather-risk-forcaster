"""Availability-aware weather diagnostics; no fitted forecasting model."""

import hashlib

import numpy as np
import pandas as pd

from .schema import Observation
from .timing import build_feature_panel


def weather_features(observations, targets, specs, *, allow_latest_vintage=False,
                     minimum_reference_observations=20):
    weather_specs = [s for s in specs if s["provider"] == "eccc"]
    ids = {s["series_id"] for s in weather_specs}
    weather = observations.loc[observations.series_id.isin(ids)].copy()
    records = [Observation(
        r.series_id, r.observation_date.date().isoformat(),
        None if pd.isna(r.value) else float(r.value), r.unit, r.source,
        r.available_at.isoformat(), r.retrieved_at.isoformat(),
        r.availability_basis, r.vintage, r.quality_flag,
    ) for r in weather.itertuples(index=False)]
    origins = sorted(targets.origin_at.dropna().unique())
    panel = pd.DataFrame(build_feature_panel(
        records, [pd.Timestamp(t).isoformat() for t in origins], weather_specs,
        allow_latest_vintage=allow_latest_vintage))
    if panel.empty:
        raise ValueError("No weather forecast origins")
    panel = panel.rename(columns={"forecast_at": "origin_at", "series_id": "feature",
                                  "observation_date": "source_observation_date"})
    panel["origin_at"] = pd.to_datetime(panel.origin_at, utc=True)
    panel["value"] = pd.to_numeric(panel.value, errors="raise")
    panel["reference_n"] = 0
    panel["reference_mean"] = np.nan
    anomaly_rows = []
    for station in ("toronto", "calgary"):
        history = weather.loc[weather.series_id.eq(f"{station}_temp")]
        if not allow_latest_vintage:
            trustworthy = (
                history.vintage.eq("point_in_time") & history.availability_basis.eq("release_timestamp")
            ) | (history.vintage.eq("synthetic") & history.availability_basis.eq("synthetic"))
            history = history.loc[trustworthy]
        for row in panel.loc[panel.feature.eq(f"{station}_temp")].to_dict("records"):
            derived = {**row, "feature": f"{station}_temp_anomaly", "value": np.nan}
            if not row["missing_reason"]:
                observed = pd.Timestamp(row["source_observation_date"])
                prior = history.loc[
                    (history.observation_date.dt.year < observed.year)
                    & (history.observation_date.dt.month == observed.month)
                    & (history.available_at <= row["origin_at"])
                    & history.value.notna(), "value"]
                derived["reference_n"] = len(prior)
                if len(prior) >= minimum_reference_observations:
                    derived["reference_mean"] = prior.mean()
                    derived["value"] = row["value"] - prior.mean()
                else:
                    derived["missing_reason"] = "insufficient_prior_year_reference"
            anomaly_rows.append(derived)
    return pd.concat([panel, pd.DataFrame(anomaly_rows)], ignore_index=True).sort_values(
        ["origin_at", "feature"]).reset_index(drop=True)


def rank_correlation(x, y):
    """Spearman's rho with tied average ranks and pairwise missingness."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    a = pd.Series(np.asarray(x)[valid]).rank(method="average").to_numpy()
    b = pd.Series(np.asarray(y)[valid]).rank(method="average").to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def relationships(targets, features, *, seed=1729, block_length=20,
                  bootstrap_repetitions=500, minimum_pairs=80):
    """Pointwise paired moving-block intervals over the original target grid.

    Bootstrap resampling is uncertainty estimation, not a train/test split.
    All predeclared feature/asset pairs are retained, including unavailable ones.
    """
    if block_length < 2 or bootstrap_repetitions < 20 or minimum_pairs < 3:
        raise ValueError("Require block_length>=2, bootstrap_repetitions>=20 and minimum_pairs>=3")
    wide = features.pivot(index="origin_at", columns="feature", values="value")
    results = []
    for asset, group in targets.groupby("asset", sort=True):
        merged = group.sort_values("date").merge(wide, left_on="origin_at", right_index=True, how="left")
        signed = merged.target.to_numpy(dtype=float)
        magnitude = np.abs(signed)
        for feature in wide.columns:
            x = merged[feature].to_numpy(dtype=float)
            n = int((np.isfinite(x) & np.isfinite(magnitude)).sum())
            rho = rank_correlation(x, magnitude)
            samples = []
            if n >= minimum_pairs and len(x) >= 2 * block_length:
                stable = hashlib.sha256(f"{seed}:{asset}:{feature}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(stable[:8], "big"))
                blocks = int(np.ceil(len(x) / block_length))
                offsets = np.arange(block_length)
                for _ in range(bootstrap_repetitions):
                    starts = rng.integers(0, len(x) - block_length + 1, size=blocks)
                    indices = (starts[:, None] + offsets).ravel()[:len(x)]
                    samples.append(rank_correlation(x[indices], magnitude[indices]))
            finite = np.asarray(samples)[np.isfinite(samples)]
            ci = np.quantile(finite, [.025, .975]) if len(finite) >= 20 else [np.nan, np.nan]
            results.append({"asset": asset, "feature": feature, "rho": rho,
                            "signed_rho": rank_correlation(x, signed), "ci_low": ci[0],
                            "ci_high": ci[1], "n_pairs": n, "n_grid_rows": len(x),
                            "valid_bootstrap_repetitions": len(finite),
                            "block_length": block_length,
                            "interval_type": "pointwise exploratory percentile interval"})
    return pd.DataFrame(results)
