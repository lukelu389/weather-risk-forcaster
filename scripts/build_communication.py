"""Build the local research dashboard, two-page memo, and audit notebooks.

Run with a Python environment containing reportlab (the bundled document runtime
is sufficient). Research calculations remain in the separately locked environment.
"""

from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(folder, name):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "passed" or digest(folder / name) != manifest["artifacts"][name]:
        raise ValueError(f"Unverified communication source: {folder / name}")
    values = list(csv.DictReader((folder / name).open()))
    for row in values:
        for key, value in row.items():
            if key == "period":
                continue
            try:
                row[key] = float(value) if value else None
            except ValueError:
                pass
    return values


def main():
    out = ROOT / "reports/phase7"
    out.mkdir(parents=True, exist_ok=True)
    primary = records(ROOT / "reports/phase5", "calibration_scores.csv")
    paired = records(ROOT / "reports/phase5", "paired_comparisons.csv")
    sensitivity = records(ROOT / "reports/phase5-sensitivity", "calibration_scores.csv")
    sensitivity_pairs = records(ROOT / "reports/phase5-sensitivity", "paired_comparisons.csv")
    portfolio = records(ROOT / "reports/phase6", "summary.csv")
    payload = {"primary": primary, "paired": paired, "sensitivity": sensitivity,
               "sensitivity_pairs": sensitivity_pairs, "portfolio": portfolio}
    template = (ROOT / "scripts/dashboard.html").read_text()
    dashboard = out / "dashboard"
    dashboard.mkdir(exist_ok=True)
    (dashboard / "index.html").write_text(template.replace("__RESEARCH_DATA__", json.dumps(payload, allow_nan=False)))
    make_memo(out, payload)
    make_notebooks()
    sources = {str(p.relative_to(ROOT)): digest(p) for p in [
        ROOT / "reports/phase5/manifest.json", ROOT / "reports/phase5-sensitivity/manifest.json",
        ROOT / "reports/phase6/manifest.json", ROOT / "docs/phase-5-seal.json",
        ROOT / "docs/phase-5-first-access.json", ROOT / "docs/phase-5-bootstrap-addendum.md"]}
    outputs = [dashboard / "index.html", ROOT / "output/pdf/committee-memo.pdf", out / "committee-memo.md",
               *sorted((ROOT / "notebooks").glob("*.ipynb"))]
    manifest = {"phase": 7, "status": "passed", "sources": sources,
                "artifacts": {str(p.relative_to(ROOT)): digest(p) for p in outputs},
                "builder_sha256": digest(Path(__file__)), "dashboard_template_sha256": digest(ROOT / "scripts/dashboard.html"),
                "scope": "Communication of preserved results; no new model fitting or trading performance."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def make_memo(out, data):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.pdfgen import canvas

    def rejects(rows, key):
        return sum(r.get(key) is not None and r[key] < .05 for r in rows)

    counts = {name: rejects(data["primary"], name + "_pvalue_holm") for name in ("exact", "independence", "conditional")}
    significant = rejects(data["sensitivity_pairs"], "pvalue_holm")
    es_rejections = rejects(data["sensitivity"], "es_pvalue_holm")
    gross = [r for r in data["portfolio"] if r["period"] == "all" and r["cost_bps"] == 0 and r["confidence"] == .99]
    es_range = (min(r["mean_es_cad"] for r in gross), max(r["mean_es_cad"] for r in gross))
    sections = [
        ("Decision requested", "Retain the project as a reproducible research pilot. Do not approve production risk limits or trading deployment. The final sample reveals calibration weaknesses, and the evidence does not establish a general weather-feature advantage."),
        ("Experiment", "The 2018-2025 snapshot supports development in 2018-2021, tuning in 2022-2023 and a locally sealed final evaluation in 2024-2025. Nine fixed models include historical simulation, GARCH/GJR, matched seasonal/weather regressions, and matched GRU/diagonal state-space variants. Monthly rolling fits use matured labels only; earlier final-period labels can enter later fits after their assumed release time."),
        ("Evidence", f"There are 477 shared scored dates for gas, 473 for WTI and 502 for the CAD equity basket. The final run contains 27,882 forecast rows without failed or skipped forecasts. Separate Holm families reject {counts['exact']} exact-coverage, {counts['independence']} independence and {counts['conditional']} conditional-coverage hypotheses. Counts overlap and are not distinct failed-model counts. Sparse tails and asymptotic references limit interpretation."),
        ("Weather result", "Matched weather effects vary by asset and architecture. All three basket weather variants have higher final quantile-loss point estimates at both confidence levels. Gas GRU weather improves the point estimate, with pointwise intervals spanning zero. A richer architecture or an additional feature does not by itself establish calibrated tail risk."),
        ("Disclosed evaluation correction", f"The primary 1,000-draw bootstrap could not attain p-values small enough for its planned Holm families. Its non-rejections are not evidence of equivalence or adequate ES. A separate post-exposure sensitivity repeats every hypothesis with 19,999 draws and the same forecasts, blocks and seeds. It yields {significant} adjusted score-comparison rejections and {es_rejections} adjusted ES-residual rejections at 5%. This correction is supplementary, not a new untouched test."),
        ("Portfolio interpretation", f"A separate sensitivity resets CAD 1 million independently at each interval start, equally split among ENB.TO, SU.TO, FTS.TO and EMA.TO. Across the nine models, full-period mean gross 99% ES ranges from CAD {es_range[0]:,.0f} to CAD {es_range[1]:,.0f}. These are model-dependent risk forecasts, not a capital recommendation or a wealth path."),
        ("Currency, execution and costs", "The basket is CAD-denominated and uses adjusted-price return proxies. Monetary loss is N times (1 - exp(R/100)). Gaussian ES uses the exact transformed tail expectation; empirical ES transforms the entire matured sample before tail averaging. Fixed total interval budgets of 0/5/10 bp add CAD 0/500/1,000 to loss, VaR and ES, preserving exceedance classifications. They are scenarios, not measured spreads or turnover. USD spot benchmarks are not added to this book; no futures roll or FX hedge is inferred."),
        ("Material limitations", "The data are latest-vintage with assumed lags, not verified historical information sets. The quote grid is retrospective, horizons vary, and the fixed equity basket has survivor-selection risk. Missing realizations remain gaps. Normal tails, moment-matched ensembles and sparse 99% events limit model adequacy. Calendar-year slices are descriptive; the pilot does not cover 2008 or 2014-2016."),
        ("Controls and next decision", "Two complete final evaluations matched 617 artifacts, including 600 checkpoints and the plots. The local seal and first-access record preserve the boundary; they are not external preregistration. Preserve these records and the input snapshot. Before any deployment study, obtain verified vintages, prospective calendars and executable instrument/cost data, and reserve a new prospective evaluation period. Any new architecture or feature must not reuse this period as untouched."),
    ]
    md = ["# Research committee memo", "", "Weather-aware energy and commodity risk | 2024-2025 final evaluation", ""]
    for title, text in sections:
        md += ["## " + title, "", text, ""]
    md += ["Sources: reports/phase5, reports/phase5-sensitivity, reports/phase6; docs/phase-5-seal.json and phase-5-first-access.json. All figures are from the preserved local experiment."]
    (out / "committee-memo.md").write_text("\n".join(md) + "\n")
    pdf = ROOT / "output/pdf/committee-memo.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=25, textColor=colors.HexColor("#18354A"), spaceAfter=10),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#647481"), spaceAfter=17),
        "head": ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#18354A"), spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14.2, alignment=TA_LEFT),
        "note": ParagraphStyle("note", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#647481")),
    }
    story = [Paragraph("Weather and tail risk", styles["title"]),
             Paragraph("RESEARCH COMMITTEE MEMO  |  Sealed final period: 2024-2025", styles["subtitle"])]
    for index, (title, text) in enumerate(sections):
        if index == 5:
            story += [PageBreak(), Paragraph("Portfolio meaning and controls", styles["title"]),
                      Paragraph("CAD basket sensitivity  |  Research-only use", styles["subtitle"])]
        story += [Paragraph(title, styles["head"]), Paragraph(text, styles["body"])]
    story += [Spacer(1, 14), Paragraph("Sources: preserved Phase 5 primary and sensitivity reports, Phase 6 summary, local protocol seal and first-access audit. Detailed hashes and limitations accompany the repository.", styles["note"])]
    def footer(c, doc):
        c.setStrokeColor(colors.HexColor("#D8E0E5")); c.line(48, 42, 564, 42)
        c.setFont("Helvetica", 8); c.setFillColor(colors.HexColor("#647481"))
        c.drawString(48, 28, "WEATHER RISK FORECASTER  /  RESEARCH PILOT")
        c.drawRightString(564, 28, str(doc.page))
    class StableCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            kwargs["invariant"] = 1
            super().__init__(*args, **kwargs)
    doc = SimpleDocTemplate(str(pdf), pagesize=(612, 792), leftMargin=48, rightMargin=48, topMargin=42, bottomMargin=58,
                            title="Weather and tail risk - research committee memo", author="Weather Risk Research")
    doc.build(story, onFirstPage=footer, onLaterPages=footer, canvasmaker=StableCanvas)


def make_notebooks():
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    setup = '''from pathlib import Path
import hashlib, json
import pandas as pd
ROOT = Path.cwd().resolve()
if not (ROOT / "reports").is_dir():
    ROOT = ROOT.parent
assert (ROOT / "config/evaluation.json").exists(), "Run from repository root or notebooks/"
def checked_csv(folder, name):
    path = ROOT / "reports" / folder
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "passed"
    assert hashlib.sha256((path / name).read_bytes()).hexdigest() == manifest["artifacts"][name]
    return pd.read_csv(path / name)
'''
    notebooks = {
        "01_final_evaluation.ipynb": [
            ("markdown", "# Final evaluation audit\nThis notebook reads verified saved outputs. It does not refit or reopen an untouched holdout. Install requirements-ml.lock.txt for the research environment. Run all cells in order. The 19,999-draw results are a disclosed post-exposure correction of primary p-value resolution."),
            ("code", setup),
            ("code", 'primary = checked_csv("phase5", "calibration_scores.csv")\npaired = checked_csv("phase5-sensitivity", "paired_comparisons.csv")\nassert len(primary) == 54 and len(paired) == 165\nprint(primary.groupby("asset").common_scored_rows.first().to_string())\nprint("Sensitivity adjusted comparison rejections:", int((paired.pvalue_holm < .05).sum()))'),
            ("code", 'weather = paired.loc[(paired.reference != "historical") & paired.metric.eq("quantile_loss")]\nprint(weather[["asset", "candidate", "confidence", "mean", "ci_low", "ci_high", "pvalue_holm"]].to_string(index=False))'),
            ("code", 'import matplotlib.pyplot as plt\nfig, axes = plt.subplots(1, 3, figsize=(14, 4))\nfor ax, (asset, frame) in zip(axes, primary.loc[primary.confidence.eq(.99)].groupby("asset")):\n    ax.barh(frame.model, 100 * frame.exceedance_rate)\n    ax.axvline(1, color="black", linestyle="--")\n    ax.set_title(asset); ax.set_xlabel("99% VaR exceedance rate (%)")\nfig.tight_layout()\nplt.show()'),
            ("markdown", "To reproduce the sealed forecasts, run `make evaluate` from the root with the preserved input snapshot. Use `make sensitivity` for the separate bootstrap correction. Replays are not independent holdouts. Do not reseal after exposure."),
        ],
        "02_portfolio_sensitivity.ipynb": [
            ("markdown", "# Fixed-notional CAD basket sensitivity\nIndependent CAD1m intervals, four equal starting weights, all nine fixed models, no compounding or annualization. Costs are total interval budgets, not estimated execution costs. Calendar-year splits are descriptive."),
            ("code", setup),
            ("code", 'summary = checked_csv("phase6", "summary.csv")\nbase = summary.loc[summary.period.eq("all") & summary.cost_bps.eq(0)]\nprint(base[["model", "confidence", "scored_rows", "mean_var_cad", "mean_es_cad"]].to_string(index=False))'),
            ("code", 'index = ["period", "model", "confidence"]\ngross = summary.loc[summary.cost_bps.eq(0)].set_index(index)\ncostly = summary.loc[summary.cost_bps.eq(10)].set_index(index)\nimport numpy as np\nnp.testing.assert_allclose(costly.mean_es_cad - gross.mean_es_cad, 1000, atol=1e-6)\nnp.testing.assert_array_equal(costly.exceedances, gross.exceedances)\nprint("CAD1,000 cost shift verified; exceedance counts unchanged.")'),
            ("code", 'import matplotlib.pyplot as plt\nview = base.loc[base.confidence.eq(.99)]\nfig, ax = plt.subplots(figsize=(9, 4))\nax.barh(view.model, view.mean_es_cad)\nax.set_xlabel("Mean gross 99% ES (CAD) on independent CAD1m intervals")\nfig.tight_layout()\nplt.show()'),
            ("markdown", "Run `make portfolio` to rebuild the sensitivity from the verified Phase 5 snapshot. Nonlinear monetary ES is calculated from the entire transformed tail; applying the return-to-loss function to log-return ES is incorrect. USD spot series are excluded from this CAD book."),
        ],
    }
    for name, parts in notebooks.items():
        cells = []
        for kind, source in parts:
            cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
        notebook = {"nbformat": 4, "nbformat_minor": 5,
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python", "version": "3.14.2"}}, "cells": cells}
        for i, cell in enumerate(cells):
            cell["id"] = f"cell-{i + 1}"
        (folder / name).write_text(json.dumps(notebook, indent=2) + "\n")


if __name__ == "__main__":
    main()
