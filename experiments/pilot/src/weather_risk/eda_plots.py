"""Export the Phase 2 development-sample figures without fitting models."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_ASSETS = ("henry_hub_spot", "wti_spot", "canadian_basket")
ASSET_LABELS = {
    "henry_hub_spot": "Henry Hub spot",
    "wti_spot": "WTI spot",
    "canadian_basket": "Canadian equity basket",
}
NAVY, BLUE, TEAL, ORANGE = "#18354A", "#367DA3", "#278A83", "#C87832"
MUTED = "#647481"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _target_series(targets: pd.DataFrame, asset: str) -> pd.Series:
    """Keep invalid rows on the original date grid, as gaps rather than zeros."""
    rows = targets.loc[targets["asset"].eq(asset)].sort_values("date")
    values = pd.to_numeric(rows["target"], errors="coerce").copy()
    if "invalid_reason" in rows:
        valid = rows["invalid_reason"].isna() | rows["invalid_reason"].eq("")
        values = values.where(valid)
    return pd.Series(
        values.replace([np.inf, -np.inf], np.nan).to_numpy(),
        index=pd.DatetimeIndex(rows["date"]), name=asset, dtype=float,
    )


def _unit(targets: pd.DataFrame, asset: str) -> str:
    units = targets.loc[targets["asset"].eq(asset), "unit"].dropna()
    if not len(units):
        return "target units"
    unit = str(units.iloc[0])
    return {"simple_return": "Simple return (fraction)", "log_return": "Log return",
            "percent": "Percent"}.get(unit, unit)


def _empty(ax: plt.Axes, message: str = "Insufficient valid observations") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=MUTED)


def _time_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _finish(fig: plt.Figure, output_dir: Path, stem: str, title: str,
            period_label: str, note: str) -> list[str]:
    fig.suptitle(f"{title}\n{period_label}  |  development sample • latest-vintage exploratory data",
                 fontsize=13, fontweight="semibold", color=NAVY)
    fig.supxlabel(note, fontsize=9, color=MUTED)
    files = []
    for extension in ("png", "pdf"):
        filename = f"{stem}.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        fig.savefig(output_dir / filename, dpi=160, facecolor="white", bbox_inches="tight", metadata=metadata)
        files.append(filename)
    plt.close(fig)
    return files


def _movements(targets: pd.DataFrame, output_dir: Path, period_label: str) -> list[str]:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), layout="constrained")
    for row, asset in enumerate(PRIMARY_ASSETS):
        series = _target_series(targets, asset)
        unit = _unit(targets, asset)
        axes[row, 0].plot(series.index, series, color=BLUE, lw=0.8)
        axes[row, 0].axhline(0, color=MUTED, lw=0.6, alpha=0.6)
        rolling = series.rolling(21, min_periods=21).std(ddof=1)
        axes[row, 1].plot(rolling.index, rolling, color=TEAL, lw=1)
        for col, description in enumerate(("Target movements", "Trailing 21-observation sample SD")):
            ax = axes[row, col]
            ax.set_title(f"{ASSET_LABELS[asset]} · {description}", loc="left", fontsize=10.5)
            ax.set_ylabel(unit)
            _time_axis(ax)
            if not (series if col == 0 else rolling).notna().any():
                _empty(ax)
    return _finish(fig, output_dir, "01_movements_and_dispersion",
                   "Large movements and changing dispersion", period_label,
                   "Trailing windows require 21 valid rows on the original observation grid. SD is not annualized; gaps remain missing.")


def _tails(targets: pd.DataFrame, summary: pd.DataFrame, output_dir: Path,
           period_label: str) -> list[str]:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
    normal = NormalDist()
    for row, asset in enumerate(PRIMARY_ASSETS):
        values = _target_series(targets, asset).dropna().to_numpy()
        for ax in axes[row]:
            ax.set_title(ASSET_LABELS[asset], loc="left", fontsize=11)
        if len(values) < 3 or np.std(values, ddof=1) <= 0:
            for ax in axes[row]:
                _empty(ax)
            continue
        z = (values - values.mean()) / values.std(ddof=1)
        bins = min(90, max(35, int(np.sqrt(len(z)))))
        axes[row, 0].hist(z, bins=bins, density=True, color=BLUE, edgecolor="white",
                          linewidth=0.4, alpha=0.85, label="Empirical distribution")
        x = np.linspace(min(z.min(), -4), max(z.max(), 4), 1000)
        density = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
        # A finite reference segment avoids underflow driving the log axis limits.
        reference = np.abs(x) <= 5
        axes[row, 0].plot(x[reference], density[reference], color=ORANGE, lw=1.5,
                          label="Standard normal")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_xlabel("Standardized target (sample SD units)")
        axes[row, 0].set_ylabel("Density · logarithmic scale")
        axes[row, 0].legend(loc="upper right", fontsize=8.5)
        kurtosis = summary.loc[summary["asset"].eq(asset), "excess_kurtosis"]
        if len(kurtosis) and pd.notna(kurtosis.iloc[0]):
            axes[row, 0].text(0.02, 0.95, f"n = {len(z):,}\nExcess kurtosis = {float(kurtosis.iloc[0]):.1f}",
                               transform=axes[row, 0].transAxes, va="top", color=NAVY,
                               fontsize=9, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        theoretical = np.array([normal.inv_cdf((i + 0.5) / len(z)) for i in range(len(z))])
        axes[row, 1].scatter(theoretical, np.sort(z), color=TEAL, s=9, alpha=0.55, linewidths=0)
        axes[row, 1].plot(theoretical[[0, -1]], theoretical[[0, -1]], color=ORANGE,
                          lw=1.2, label="Normal reference y = x")
        axes[row, 1].set_xlabel("Standard normal quantile")
        axes[row, 1].set_ylabel("Observed standardized quantile")
        axes[row, 1].legend(loc="upper left", fontsize=8.5)
    return _finish(fig, output_dir, "02_distributions_and_qq",
                   "Tail shape relative to a normal distribution", period_label,
                   "All valid observations are shown; tails are not trimmed. Standardization uses the development sample only.")


def _autocorrelations(acf: pd.DataFrame, output_dir: Path, period_label: str) -> list[str]:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), layout="constrained", sharex=True)
    for ax, asset in zip(axes, PRIMARY_ASSETS):
        rows = acf.loc[acf["asset"].eq(asset)]
        for transform, color, label in (("return", BLUE, "Target movement"),
                                         ("absolute", TEAL, "Absolute movement"),
                                         ("squared", ORANGE, "Squared movement")):
            subset = rows.loc[rows["transform"].eq(transform)].sort_values("lag")
            if transform == "return":
                ax.bar(subset["lag"], subset["acf"], color=color, alpha=0.45, width=0.65, label=label)
            else:
                ax.plot(subset["lag"], subset["acf"], color=color, marker="o", ms=3, lw=1.2, label=label)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(ASSET_LABELS[asset], loc="left", fontsize=11)
        ax.set_ylabel("Autocorrelation")
        ax.legend(loc="upper right", ncol=3, fontsize=8.5)
        if rows.empty:
            _empty(ax)
    axes[-1].set_xlabel("Lag (rows of the original observation grid)")
    return _finish(fig, output_dir, "03_autocorrelation",
                   "Dependence in movements and their magnitudes", period_label,
                   "Pairwise valid observations at each grid lag; missing dates are not compressed. No IID significance bands are shown.")


def _seasonality(observations: pd.DataFrame, targets: pd.DataFrame, output_dir: Path,
                 period_label: str) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    ax = axes[0, 0]
    for station, color in (("toronto", BLUE), ("calgary", TEAL)):
        for kind, style in (("hdd18", "-"), ("cdd18", "--")):
            rows = observations.loc[observations["series_id"].eq(f"{station}_{kind}")].copy()
            rows["month"] = pd.to_datetime(rows["observation_date"]).dt.month
            means = rows.groupby("month")["value"].mean().reindex(range(1, 13))
            ax.plot(means.index, means, color=color, linestyle=style, marker="o", ms=3,
                    label=f"{station.title()} {kind[:3].upper()}")
    ax.set_title("Observed heating / cooling demand proxies", loc="left", fontsize=10.5)
    ax.set_ylabel("Monthly mean daily degree days (°C·day)")
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    for ax, asset in zip((axes[0, 1], axes[1, 0], axes[1, 1]), PRIMARY_ASSETS):
        series = _target_series(targets, asset)
        grouped = series.abs().groupby(series.index.month)
        medians = grouped.median().reindex(range(1, 13))
        counts = grouped.count().reindex(range(1, 13), fill_value=0)
        ax.bar(medians.index, medians, color=BLUE, width=0.7, alpha=0.85)
        ax.set_title(f"{ASSET_LABELS[asset]} · median absolute movement", loc="left", fontsize=10.5)
        ax.set_ylabel(_unit(targets, asset))
        for month, value in medians.items():
            if pd.notna(value):
                ax.annotate(str(int(counts.loc[month])), (month, value), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
        ax.margins(y=0.22)
        if not medians.notna().any():
            _empty(ax)
    for ax in axes.flat:
        ax.set_xticks(range(1, 13), MONTHS, rotation=40, ha="right")
        ax.set_xlabel("Calendar month")
    return _finish(fig, output_dir, "04_seasonality", "Seasonal patterns in weather and movement size",
                   period_label,
                   "Weather is grouped by observation month; target magnitudes by interval end month. Bar labels give valid target counts.")


def _feature_label(feature: str) -> str:
    station, _, measure = feature.partition("_")
    names = {"temp": "Temperature", "hdd18": "HDD18", "cdd18": "CDD18",
             "precip": "Precipitation", "temp_anomaly": "Temperature\nanomaly"}
    return f"{station.title()}\n{names.get(measure, measure.replace('_', ' '))}"


def _relationships(relationships: pd.DataFrame, output_dir: Path,
                   period_label: str) -> list[str]:
    desired_features = [f"{station}_{measure}" for station in ("toronto", "calgary")
                        for measure in ("temp", "hdd18", "cdd18", "precip", "temp_anomaly")]
    features = [name for name in desired_features if name in set(relationships["feature"])]
    if not features:
        features = sorted(relationships["feature"].dropna().unique())
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    if not features:
        _empty(ax, "No valid lagged weather–target pairs")
    else:
        rows = relationships.loc[relationships["asset"].isin(PRIMARY_ASSETS)]
        rho = rows.pivot(index="asset", columns="feature", values="rho").reindex(index=PRIMARY_ASSETS, columns=features)
        counts = rows.pivot(index="asset", columns="feature", values="n_pairs").reindex(index=PRIMARY_ASSETS, columns=features)
        values = rho.to_numpy(dtype=float)
        finite = np.abs(values[np.isfinite(values)])
        limit = max(0.25, float(finite.max())) if len(finite) else 0.25
        cmap = matplotlib.colormaps["RdBu_r"].copy()
        cmap.set_bad("#E6EBEE")
        heatmap = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(features)), [_feature_label(name) for name in features], fontsize=9)
        ax.set_yticks(range(len(PRIMARY_ASSETS)), [ASSET_LABELS[asset] for asset in PRIMARY_ASSETS], fontsize=10)
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, len(features), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(PRIMARY_ASSETS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        for i in range(len(PRIMARY_ASSETS)):
            for j in range(len(features)):
                value = values[i, j]
                count = counts.iloc[i, j]
                label = f"{value:+.2f}\nn={int(count):,}" if np.isfinite(value) and pd.notna(count) else "Unavailable"
                color = "white" if np.isfinite(value) and abs(value) > 0.65 * limit else NAVY
                ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)
        fig.colorbar(heatmap, ax=ax, shrink=0.8, label="Spearman rank correlation")
    return _finish(fig, output_dir, "05_lagged_weather_relationships",
                   "Available weather versus subsequent absolute target movements", period_label,
                   "Exploratory pairwise rank correlations; n is the pair count. Block-bootstrap intervals appear in the accompanying table.")


def make_plots(observations: pd.DataFrame, targets: pd.DataFrame, summary: pd.DataFrame,
               acf: pd.DataFrame, relationships: pd.DataFrame, output_dir: Path,
               period_label: str) -> list[str]:
    """Save five development-only figure pairs and return their relative filenames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B6C0C7", "axes.labelcolor": NAVY,
        "axes.titlecolor": NAVY, "axes.grid": True,
        "axes.axisbelow": True, "grid.alpha": 0.3, "grid.color": "#BAC5CD",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "none", "pdf.fonttype": 42,
    }):
        files.extend(_movements(targets, output_dir, period_label))
        files.extend(_tails(targets, summary, output_dir, period_label))
        files.extend(_autocorrelations(acf, output_dir, period_label))
        files.extend(_seasonality(observations, targets, output_dir, period_label))
        files.extend(_relationships(relationships, output_dir, period_label))
    return files
