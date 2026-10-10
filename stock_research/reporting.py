"""The stock-research report: one self-contained HTML page plus data files.

Every figure and every verdict carries its data origin. A result from
synthetic data is drawn with a banner across the chart and its verdict pill
reads "synthetic", because a chart cut out of a page keeps its pixels and
loses its caption.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stock_research.config import REPOSITORY_ROOT
from stock_research.data.prices import ORIGIN_REAL
from stock_research.hypotheses import common, registry

OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "stock_research"
_BLUE, _GREEN, _RED, _GREY = "#1565C0", "#2E7D32", "#C62828", "#9E9E9E"

ORIGIN_LABEL = {
    "real_historical": "real historical observations",
    "development_fixture": "development fixture",
    "synthetic": "SYNTHETIC DATA: not a finding",
}
STATUS_CLASS = {
    common.SUPPORTED: "ok", common.EXECUTION_NOT_VERIFIABLE: "warn",
    common.INCONCLUSIVE: "warn", common.NOT_SUPPORTED: "bad",
    common.DATA_INSUFFICIENT: "muted",
}


def _png(fig, origin: str) -> str:
    if origin != ORIGIN_REAL:
        fig.text(0.5, 0.5, ORIGIN_LABEL.get(origin, origin).upper(), fontsize=20, color="#C62828",
                 alpha=0.16, ha="center", va="center", rotation=18, weight="bold")
    fig.text(0.995, 0.005, f"data: {ORIGIN_LABEL.get(origin, origin)}", fontsize=7,
             ha="right", va="bottom", color="#555")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _axes(title: str, xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    return fig, ax


def figures(result: Dict[str, Any]) -> List[str]:
    """Charts for one hypothesis, from the tables its result carries."""

    key, origin, tables = result["hypothesis"], result["data_origin"], result.get("tables", {})
    out: List[str] = []
    if result["status"] == common.DATA_INSUFFICIENT:
        return out

    def means(prefix: str) -> List[Dict[str, Any]]:
        return [e for e in result["exploratory"] if e["test"].startswith(prefix) and e.get("mean") is not None]

    if key == "H1":
        for label, prefix in (("Signed drift CAR(+2,+5) by size tercile", "drift"),
                              ("Signed reaction CAR(0,+1) by size tercile", "reaction")):
            rows = means(prefix)
            if rows:
                fig, ax = _axes(label, ylabel="mean signed CAR")
                names = [r["test"].split(", ")[-1] for r in rows]
                errors = [(r["ci"][1] - r["ci"][0]) / 2 if r.get("ci") else 0 for r in rows]
                ax.bar(names, [r["mean"] for r in rows], yerr=errors, color=_BLUE, alpha=0.85, capsize=3)
                ax.axhline(0, color="black", lw=0.8)
                out.append(_png(fig, origin))
        quintiles = [e for e in result["exploratory"] if "quintile" in e["test"] and e.get("mean") is not None]
        if quintiles:
            fig, ax = _axes("CAR(0,+5) by sentiment quintile (descriptive)", "quintile", "mean CAR")
            ax.bar(range(1, len(quintiles) + 1), [q["mean"] for q in quintiles], color=_BLUE, alpha=0.85)
            ax.axhline(0, color="black", lw=0.8)
            out.append(_png(fig, origin))
    if key == "H2" and tables.get("gap_bins"):
        fig, ax = _axes("Overnight gap versus intraday continuation", "mean gap (decile)", "mean open-to-close return")
        ax.plot([b["gap"] for b in tables["gap_bins"]], [b["intraday"] for b in tables["gap_bins"]],
                marker="o", color=_BLUE)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
        out.append(_png(fig, origin))
    if key in ("H3", "H6"):
        metric = "mse" if key == "H3" else "log_loss"
        rows = [e for e in result["exploratory"] if e["test"].startswith("model:") and metric in e]
        if rows:
            fig, ax = _axes(f"Out-of-sample {metric.replace('_', ' ')} by model", ylabel=metric)
            ax.bar([r["test"].replace("model: ", "").replace("_", "\n") for r in rows],
                   [r[metric] for r in rows], color=_BLUE, alpha=0.85)
            low = min(r[metric] for r in rows)
            ax.set_ylim(low * 0.9, max(r[metric] for r in rows) * 1.03)
            out.append(_png(fig, origin))
    if key == "H4" and tables.get("coverage_bins"):
        fig, ax = _axes("Abnormal coverage versus subsequent return", "abnormal coverage (quintile mean)", "mean CAR(+1,+10)")
        ax.plot([b["abnormal_coverage"] for b in tables["coverage_bins"]],
                [b["car_p1_p10"] for b in tables["coverage_bins"]], marker="o", color=_BLUE)
        ax.axhline(0, color="black", lw=0.8)
        out.append(_png(fig, origin))
    if key == "H5":
        rows = means("mean CAR(+1,+3)")
        if rows:
            fig, ax = _axes("CAR(+1,+3) after a limit-down close", ylabel="mean CAR")
            errors = [(r["ci"][1] - r["ci"][0]) / 2 if r.get("ci") else 0 for r in rows]
            ax.bar([r["test"].split(", ")[-1].replace("_", " ") for r in rows],
                   [r["mean"] for r in rows], yerr=errors, color=_BLUE, alpha=0.85, capsize=3)
            ax.axhline(0, color="black", lw=0.8)
            out.append(_png(fig, origin))
    if key == "H6":
        if tables.get("hazard_by_length"):
            fig, ax = _axes("Limit-up streak hazard", "streak length (6 = six or more)", "share ending next session")
            ax.plot([h["streak_length"] for h in tables["hazard_by_length"]],
                    [h["ended_rate"] for h in tables["hazard_by_length"]], marker="o", color=_BLUE)
            ax.set_ylim(0, 1)
            out.append(_png(fig, origin))
        if tables.get("calibration_baseline_plus_news"):
            fig, ax = _axes("Calibration: model with news features", "predicted probability", "observed rate")
            rows = tables["calibration_baseline_plus_news"]
            ax.plot([r["mean_predicted"] for r in rows], [r["observed_rate"] for r in rows],
                    marker="o", color=_BLUE)
            ax.plot([0, 1], [0, 1], color=_GREY, ls="--", lw=0.8)
            out.append(_png(fig, origin))
    if key == "H7":
        cells = [c for c in result["exploratory"] if c.get("mean") is not None and "category" in c]
        if cells:
            categories = sorted({c["category"] for c in cells})
            windows = sorted({tuple(c["window"]) for c in cells})
            grid = np.full((len(categories), len(windows)), np.nan)
            for cell in cells:
                grid[categories.index(cell["category"]), windows.index(tuple(cell["window"]))] = cell["mean"]
            fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(categories) + 1.6))
            limit = np.nanmax(np.abs(grid)) or 1
            image = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            ax.set_xticks(range(len(windows)))
            ax.set_xticklabels([f"[{a:+d},{b:+d}]" for a, b in windows], fontsize=8)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels([c.replace("_", " ") for c in categories], fontsize=8)
            for i in range(len(categories)):
                for j in range(len(windows)):
                    if np.isfinite(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j] * 100:.1f}", ha="center", va="center", fontsize=7)
            ax.set_title("Mean CAR (%) by disclosure category and window", fontsize=10, weight="bold")
            fig.colorbar(image, ax=ax, fraction=0.03)
            out.append(_png(fig, origin))
    if key in ("H7", "H8") and tables.get("event_time"):
        fig, ax = _axes("Event-time cumulative market-adjusted return",
                        "session relative to day 0", "mean cumulative return")
        for name, path in tables["event_time"].items():
            if path["mean"]:
                ax.plot(path["days"], path["mean"], label=f"{name.replace('_', ' ')} (n={path['events']})", lw=1.4)
        ax.axvline(0, color="black", lw=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.legend(fontsize=7)
        out.append(_png(fig, origin))
    if key == "H9" and tables.get("lag_histogram"):
        histogram = tables["lag_histogram"]
        fig, ax = _axes("Turkish minus English first-report time", "hours (positive: English first)", "matched events")
        edges = np.array(histogram["edges"])
        ax.bar((edges[:-1] + edges[1:]) / 2, histogram["counts"], width=np.diff(edges) * 0.9, color=_BLUE, alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        out.append(_png(fig, origin))
    return out


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_number(v, digits) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if abs(value) >= 1e-4 or value == 0 else f"{value:.2e}"
    return html.escape(str(value))


def _card(result: Dict[str, Any]) -> str:
    spec = registry.SPECS[result["hypothesis"]]
    status = result["status"] or "not decided"
    synthetic = result["data_origin"] != ORIGIN_REAL
    label = status.replace("_", " ")
    if synthetic:
        label += " · synthetic"
    elif not result.get("frame_complete", True):
        label += " · interim"
    parts = [f'<section class="card" id="{spec.id}">',
             f'<h2>{spec.id} · {html.escape(spec.title)} '
             f'<span class="pill {STATUS_CLASS.get(status, "muted")}">{html.escape(label)}</span></h2>',
             f'<p class="claim">{html.escape(spec.claim)}</p>']
    if not result["claim_allowed"]:
        parts.append('<p class="notice">This is not a finding. '
                     + ("It was computed on synthetic data to show the estimator works."
                        if synthetic else "The sample frame was incomplete when it was computed.")
                     + "</p>")
    primary = result.get("primary")
    if primary:
        parts.append(
            "<table><tr><th>Primary estimand</th><th>Estimate</th><th>95% CI</th>"
            "<th>p</th><th>p (Holm, 9)</th><th>n</th><th>Clusters</th><th>Material effect</th></tr>"
            f"<tr><td>{html.escape(spec.estimand)}</td><td>{_number(primary.get('estimate'))}</td>"
            f"<td>{_number(primary.get('ci'))}</td><td>{_number(primary.get('p'))}</td>"
            f"<td>{_number(primary.get('p_holm'))}</td><td>{_number(primary.get('n'))}</td>"
            f"<td>{_number(primary.get('clusters'))}</td>"
            f"<td>{'+' if spec.direction > 0 else '−'}{spec.material_effect}</td></tr></table>")
    if not result["sufficiency"]["met"]:
        parts.append("<p><b>Why there is no estimate:</b></p><ul>"
                     + "".join(f"<li>{html.escape(r)}</li>" for r in result["sufficiency"]["reasons"])
                     + "</ul>")
    counts = result["sufficiency"].get("counts")
    if counts:
        parts.append(f"<details><summary>Sample counts</summary><pre>"
                     f"{html.escape(json.dumps(counts, indent=1, default=str))}</pre></details>")
    for image in figures(result):
        parts.append(f'<img alt="{spec.id} figure" src="data:image/png;base64,{image}">')
    for name, rows in (("Exploratory tests (not decisions; FDR-adjusted where shown)", result["exploratory"]),
                       ("Sensitivity analyses", result["sensitivity"])):
        if rows:
            parts.append(f"<details><summary>{name} ({len(rows)})</summary><pre>"
                         f"{html.escape(json.dumps(rows, indent=1, default=str))}</pre></details>")
    if result.get("execution"):
        parts.append("<details><summary>Execution and costs (scenario assumptions, not measured fees)"
                     f"</summary><pre>{html.escape(json.dumps(result['execution'], indent=1, default=str))}"
                     "</pre></details>")
    if result["limitations"]:
        parts.append("<p><b>Limitations</b></p><ul>"
                     + "".join(f"<li>{html.escape(l)}</li>" for l in result["limitations"]) + "</ul>")
    parts.append("</section>")
    return "\n".join(parts)


_CSS = """
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f4f6f9;color:#1f2937;padding:24px;max-width:1080px;margin:auto}
h1{font-size:22px} h2{font-size:16px;margin:0 0 6px}
.card{background:#fff;border-radius:14px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.07);margin:16px 0;overflow-x:auto}
.pill{font-size:11px;padding:2px 9px;border-radius:99px;font-weight:600;vertical-align:middle}
.pill.ok{background:#dcfce7;color:#166534}.pill.warn{background:#fef3c7;color:#92400e}
.pill.bad{background:#fee2e2;color:#991b1b}.pill.muted{background:#e5e7eb;color:#374151}
.claim{color:#4b5563;margin:4px 0 10px}.notice{background:#fff7ed;border-left:4px solid #f97316;padding:8px 12px}
table{border-collapse:collapse;font-size:12px;margin:8px 0}th,td{border:1px solid #e5e7eb;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f9fafb}img{max-width:100%;margin:8px 0;border:1px solid #e5e7eb;border-radius:8px}
pre{font-size:11px;background:#f9fafb;padding:10px;border-radius:8px;overflow-x:auto}details{margin:6px 0}
"""


def write_report(results: Dict[str, Dict[str, Any]], *, overview: Optional[Dict[str, Any]] = None,
                 manifest: Optional[Dict[str, Any]] = None, output_dir: Optional[Path] = None,
                 name: str = "report") -> Dict[str, str]:
    """Write the HTML report, a summary CSV and the full results as JSON."""

    directory = Path(output_dir or OUTPUT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    origins = sorted({r["data_origin"] for r in results.values()})
    all_real = origins == [ORIGIN_REAL]

    summary_path = directory / f"{name}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hypothesis", "title", "status", "data_origin", "claim_allowed", "estimate",
                         "ci_low", "ci_high", "p", "p_holm", "n", "reasons"])
        for key, result in results.items():
            primary = result.get("primary") or {}
            ci = primary.get("ci") or [None, None]
            writer.writerow([key, result["title"], result["status"], result["data_origin"],
                             result["claim_allowed"], primary.get("estimate"), ci[0], ci[1],
                             primary.get("p"), primary.get("p_holm"), primary.get("n"),
                             " | ".join(result["sufficiency"]["reasons"])])
    json_path = directory / f"{name}_results.json"
    json_path.write_text(json.dumps({"manifest": manifest, "overview": overview, "results": results},
                                    indent=1, default=str, ensure_ascii=False), encoding="utf-8")

    rows = "".join(
        f'<tr><td><a href="#{key}">{key}</a></td><td>{html.escape(r["title"])}</td>'
        f'<td><span class="pill {STATUS_CLASS.get(r["status"], "muted")}">'
        f'{html.escape((r["status"] or "").replace("_", " "))}</span></td>'
        f'<td>{html.escape(ORIGIN_LABEL.get(r["data_origin"], r["data_origin"]))}</td>'
        f'<td>{_number((r.get("primary") or {}).get("n"))}</td>'
        f'<td>{html.escape("; ".join(r["sufficiency"]["reasons"])[:220])}</td></tr>'
        for key, r in results.items())
    banner = "" if all_real else (
        '<p class="notice"><b>This page contains results computed on synthetic data.</b> '
        "They show that each estimator recovers an effect that was planted in the data. "
        "They say nothing about Borsa Istanbul.</p>")
    page = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Stock-level research report</title><style>{_CSS}</style></head><body>",
        "<h1>Stock-level news reaction and KAP event study</h1>", banner,
        '<section class="card"><h2>Research overview</h2>',
        f"<pre>{html.escape(json.dumps(overview or {}, indent=1, default=str, ensure_ascii=False))}</pre>",
        f"<p>Protocol <code>{html.escape(str((manifest or {}).get('protocol_hash', registry.protocol_hash()))[:16])}</code>"
        f" · manifest <code>{html.escape(str((manifest or {}).get('manifest_hash', '–'))[:16])}</code>"
        f" · <a href='{summary_path.name}'>summary CSV</a> · <a href='{json_path.name}'>full results JSON</a></p>",
        "</section>",
        '<section class="card"><h2>Hypotheses</h2><table><tr><th></th><th>Hypothesis</th>'
        "<th>Status</th><th>Data</th><th>n</th><th>Why no estimate</th></tr>" + rows + "</table>"
        "<p>Verdicts use Holm-adjusted p-values over the nine registered hypotheses and require a "
        "material effect. An unadjusted p below 0.05 is not support.</p></section>",
        *[_card(result) for result in results.values()],
        "</body></html>",
    ]
    html_path = directory / f"{name}.html"
    html_path.write_text("\n".join(page), encoding="utf-8")
    return {"html": str(html_path), "summary_csv": str(summary_path), "results_json": str(json_path)}
