"""
Dashboard generator — renders a single self-contained HTML file from the DB.

Designed for NON-TECHNICAL viewers: plain-language cards, one combined chart,
latest headlines with mood chips, and pipeline health dots. No server needed —
the output is one .html file that opens in any browser (Chart.js via CDN).

Usage:
    python dashboard.py                  # writes dashboard.html
    python dashboard.py --output x.html
    python main.py dashboard             # same + opens the browser
"""

import argparse
import json
import logging
from datetime import datetime

import pandas as pd

import database as db
from dashboard_regime import EVENT_CSS, REGIME_CSS
from config import DB_PATH, MINIMUM_HEADLINES_PER_DAY, MINIMUM_OVERLAP_DAYS

logger = logging.getLogger(__name__)

DASHBOARD_OUTPUT = "dashboard.html"
CHART_DAYS = 60          # window shown in the main chart
RECENT_HEADLINES = 12    # rows in the headlines table
RECENT_RUNS = 7          # pipeline run dots


# -- Data collection -------------------------------------------------------------

def _collect(db_path: str = DB_PATH) -> dict:
    # The market-linked chart has one row per reaction session and uses the
    # pre-specified unweighted baseline.  ``avg_score`` remains an internal
    # compatibility name for the HTML template; no publication-date-table
    # fallback is allowed here.
    variants = db.get_signal_variants(db_path=db_path)
    if variants.empty:
        sent = []
        reliable = 0
    else:
        variants = variants.copy().rename(columns={"simple_mean": "avg_score"})
        variants["date"] = pd.to_datetime(variants["date"])
        variants = variants.sort_values("date")
        price_start = variants["date"].min().date().isoformat()
        prices = db.get_prices(start=price_start, db_path=db_path).copy()
        if prices.empty:
            variants["close"] = None
        else:
            prices["date"] = pd.to_datetime(prices["date"])
            variants = variants.merge(
                prices[["date", "close"]], on="date", how="left"
            )
        reliable = int(
            (
                (variants["headline_count"] >= MINIMUM_HEADLINES_PER_DAY)
                & variants["close"].notna()
            ).sum()
        )
        variants["close"] = variants["close"].astype(object).where(
            variants["close"].notna(), None
        )
        variants["date"] = variants["date"].dt.strftime("%Y-%m-%d")
        sent = variants.tail(CHART_DAYS).to_dict("records")

    with db._conn(db_path) as con:
        total = con.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
        sources = con.execute("SELECT COUNT(DISTINCT source) FROM headlines").fetchone()[0]
        first_day, last_day = con.execute(
            "SELECT MIN(published_at), MAX(published_at) FROM headlines"
        ).fetchone()

        cats = con.execute(
            """SELECT category, COUNT(*) AS n FROM headlines
               WHERE category IS NOT NULL
               GROUP BY category ORDER BY n DESC"""
        ).fetchall()

        heads = con.execute(
            """SELECT published_at, source, title, sentiment_label, sentiment_score
               FROM headlines
               WHERE sentiment_score IS NOT NULL AND published_at IS NOT NULL
                 AND COALESCE(relevance, 1.0) >= 0.25
               ORDER BY published_at DESC, id DESC LIMIT ?""",
            (RECENT_HEADLINES,),
        ).fetchall()

        runs = con.execute(
            """SELECT started_at, status, headlines_scraped FROM pipeline_runs
               ORDER BY run_id DESC LIMIT ?""",
            (RECENT_RUNS,),
        ).fetchall()

    return {
        "total": total,
        "sources": sources,
        "first_day": first_day,
        "last_day": last_day,
        "sent": sent,
        "reliable": reliable,
        "cats": [dict(r) for r in cats],
        "heads": [dict(r) for r in heads],
        "runs": [dict(r) for r in runs],
    }


def _regime_html(db_path: str) -> str:
    """Build the News Regime fragment from the stored analytical tables.

    Rendering must never take the dashboard down: if the indicator tables are
    missing or a read fails, the section degrades to a visible notice while the
    rest of the page still renders.
    """
    try:
        from dashboard_regime import render_regime_section
        from indicators.regime import build_regime_report

        family_signals = db.read_table("daily_family_signals", db_path)
        if family_signals.empty:
            return render_regime_section({}, pd.DataFrame())

        drivers = db.get_classified_headlines(db_path=db_path)
        regime = build_regime_report(
            family_signals,
            db.read_table("abnormal_tone_daily", db_path),
            db.read_table("news_disagreement_daily", db_path),
            db.read_table("news_volume_daily", db_path),
            drivers,
        )
        version = ""
        if not family_signals.empty:
            version = str(family_signals["family_version"].iloc[-1])
        experiment = ""
        if not drivers.empty and drivers["experiment_id"].notna().any():
            experiment = str(drivers["experiment_id"].dropna().iloc[-1])
        return render_regime_section(
            regime, drivers, family_version=version, experiment_id=experiment,
        )
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("News Regime section unavailable: %s", exc)
        return (
            '<section class="card"><h2>News Regime</h2>'
            '<p class="null">This section could not be rendered from the stored '
            'indicator tables.</p></section>'
        )


def _events_html(db_path: str) -> str:
    """Build the candidate-event fragment, degrading visibly on failure."""
    try:
        from dashboard_regime import render_event_section

        events = db.read_table("event_groups", db_path)
        if events.empty:
            return render_event_section(events, [])
        version = str(events["algorithm_version"].iloc[-1])
        return render_event_section(events, [], algorithm_version=version)
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("Candidate-event section unavailable: %s", exc)
        return (
            '<section class="card"><h2>Candidate Events</h2>'
            '<p class="null">This section could not be rendered.</p></section>'
        )


def _windows_html(db_path: str) -> str:
    """Build the market-windows fragment, degrading visibly on failure."""
    try:
        from dashboard_regime import render_market_windows_section

        return render_market_windows_section(
            db.read_table("event_return_windows", db_path),
            db.read_table("event_research_dataset", db_path),
        )
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("Market-windows section unavailable: %s", exc)
        return (
            '<section class="card"><h2>Market Windows</h2>'
            '<p class="null">This section could not be rendered.</p></section>'
        )


def _validation_html(db_path: str) -> str:
    """Build the predictive-validation fragment, degrading visibly on failure."""
    try:
        from dashboard_regime import render_validation_section

        return render_validation_section(
            db.read_table("frozen_research_results", db_path),
            later_runs=db.read_table("validation_runs", db_path),
        )
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("Validation section unavailable: %s", exc)
        return (
            '<section class="card"><h2>Predictive Validation</h2>'
            '<p class="null">This section could not be rendered.</p></section>'
        )


def _future_html(db_path: str) -> str:
    """Build the future-validation fragment. Readiness only, never performance."""
    try:
        from scripts.future_readiness import build_report

        # Computed live and never persisted from here: the dashboard is a
        # reader of the sealed boundary, not a participant in it.
        return _render_future(build_report(db_path))
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("Future-validation section unavailable: %s", exc)
        return (
            '<section class="card"><h2>Future Validation Status</h2>'
            '<p class="null">This section could not be rendered.</p></section>'
        )


def _render_future(readiness) -> str:
    from dashboard_regime import render_future_validation_section

    return render_future_validation_section(readiness)


# -- Rendering helpers -----------------------------------------------------------

def _mood(score: float) -> tuple[str, str, str]:
    """(emoji, label, css-class) for a daily average score."""
    if score > 0.05:
        return "&#128578;", "Positive", "pos"
    if score < -0.05:
        return "&#128577;", "Negative", "neg"
    return "&#128528;", "Neutral", "neu"


def _run_status_display(status: str) -> tuple[str, str]:
    """Return a three-state CSS class and human-readable run label."""
    canonical = {
        "ok": "success",
        "recovered": "success",
        "error": "failed",
        "crashed": "failed",
    }.get(status, status)
    if canonical == "success":
        return "ok", "Running normally"
    if canonical in {"degraded", "running"}:
        return "warn", f"Last run: {canonical}"
    return "bad", f"Last run: {canonical}"


_CAT_LABELS = {
    "fx_lira": "Currency / Lira",
    "turkey_macro": "Turkish economy",
    "energy_commodities": "Energy & commodities",
    "rates_tcmb": "Interest rates / Central Bank",
    "bist_company": "Stock market & companies",
    "global_risk": "Global markets",
    "banks": "Banking",
    "political_risk": "Political events",
    "crypto": "Crypto",
    "other": "Other",
}


def generate(db_path: str = DB_PATH, output: str = DASHBOARD_OUTPUT) -> str:
    d = _collect(db_path)
    regime_html = _regime_html(db_path)
    # The seven-section story, in the order a reader should meet it: what the
    # data is, what the news looks like, how it is classified, what events were
    # grouped, which market windows those events can even be measured against,
    # what the frozen evaluation found, and what remains genuinely untested.
    events_html = (
        _events_html(db_path)
        + _windows_html(db_path)
        + _validation_html(db_path)
        + _future_html(db_path)
    )

    # -- Chart data --
    labels   = [r["date"][5:] for r in d["sent"]]              # MM-DD
    scores   = [round(r["avg_score"], 3) for r in d["sent"]]
    closes   = [r["close"] for r in d["sent"]]                 # None on non-trading days
    barcols  = [
        ("rgba(76,175,80,0.85)" if r["avg_score"] >= 0 else "rgba(244,67,54,0.85)")
        if r["headline_count"] >= MINIMUM_HEADLINES_PER_DAY
        else ("rgba(76,175,80,0.30)" if r["avg_score"] >= 0 else "rgba(244,67,54,0.30)")
        for r in d["sent"]
    ]
    cat_labels = [_CAT_LABELS.get(c["category"], c["category"]) for c in d["cats"]]
    cat_counts = [c["n"] for c in d["cats"]]

    # -- Latest day card --
    latest = d["sent"][-1] if d["sent"] else None
    if latest:
        emoji, mood_label, mood_cls = _mood(latest["avg_score"])
        latest_html = (
            f'<div class="big {mood_cls}">{emoji} {mood_label}</div>'
            f'<div class="sub">Reaction session {latest["date"]} &middot; '
            f'{latest["headline_count"]} headlines '
            f'({latest["positive_count"]} good / {latest["neutral_count"]} neutral / '
            f'{latest["negative_count"]} bad)</div>'
        )
    else:
        latest_html = '<div class="big">No data yet</div>'

    # -- Progress card --
    pct = min(100, round(100 * d["reliable"] / MINIMUM_OVERLAP_DAYS))

    # -- Headlines table --
    rows = []
    for h in d["heads"]:
        emoji, lbl, cls = _mood(h["sentiment_score"])
        title = (h["title"] or "").replace("<", "&lt;")
        rows.append(
            f'<tr><td class="dt">{h["published_at"]}</td>'
            f'<td class="ttl">{title}</td>'
            f'<td><span class="chip {cls}">{emoji} {lbl}</span></td></tr>'
        )
    heads_html = "\n".join(rows)

    # -- Run dots --
    dots = []
    for r in reversed(d["runs"]):
        cls, _ = _run_status_display(r["status"])
        tip = f'{r["started_at"][:10]} — {r["status"]}, {r["headlines_scraped"]} new headlines'
        dots.append(f'<span class="dot {cls}" title="{tip}"></span>')
    dots_html = "".join(dots)
    last_status = d["runs"][0]["status"] if d["runs"] else "—"
    run_cls, run_label = _run_status_display(last_status)

    html = _TEMPLATE
    for key, val in {
        "__GENERATED__":   datetime.now().strftime("%d %B %Y, %H:%M"),
        "__TOTAL__":       f'{d["total"]:,}',
        "__SOURCES__":     str(d["sources"]),
        "__FIRST_DAY__":   d["first_day"] or "—",
        "__LAST_DAY__":    d["last_day"] or "—",
        "__LATEST__":      latest_html,
        "__RELIABLE__":    str(d["reliable"]),
        "__NEEDED__":      str(MINIMUM_OVERLAP_DAYS),
        "__PCT__":         str(pct),
        "__HEADS__":       heads_html,
        "__DOTS__":        dots_html,
        "__RUN_STATUS__":  run_label,
        "__RUN_CLS__":     run_cls,
        "__LABELS__":      json.dumps(labels),
        "__SCORES__":      json.dumps(scores),
        "__CLOSES__":      json.dumps(closes),
        "__BARCOLS__":     json.dumps(barcols),
        "__CAT_LABELS__":  json.dumps(cat_labels, ensure_ascii=False),
        "__CAT_COUNTS__":  json.dumps(cat_counts),
        "__REGIME__":      regime_html,
        "__REGIME_CSS__":  REGIME_CSS + EVENT_CSS,
        "__EVENTS__":      events_html,
    }.items():
        html = html.replace(key, val)

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Dashboard written -> %s", output)
    return output


# -- HTML template (placeholders replaced in generate()) --------------------------

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BIST 100 News Sentiment — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { --blue:#2C7BB6; --green:#4CAF50; --red:#F44336; --grey:#9E9E9E; }
  * { box-sizing:border-box; margin:0; }
  body { font-family:'Segoe UI',system-ui,sans-serif; background:#f4f6f9; color:#1f2937; padding:24px; }
  .wrap { max-width:1100px; margin:0 auto; }
  h1 { font-size:1.5rem; margin-bottom:2px; }
  .meta { color:#6b7280; font-size:.85rem; margin-bottom:20px; }
  .pill { display:inline-block; padding:2px 12px; border-radius:999px; font-size:.8rem; font-weight:600; margin-left:8px; }
  .pill.ok  { background:#dcfce7; color:#166534; }
  .pill.warn { background:#fef3c7; color:#92400e; }
  .pill.bad { background:#fee2e2; color:#991b1b; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin-bottom:20px; }
  .card { background:#fff; border-radius:14px; padding:18px 20px; box-shadow:0 1px 4px rgba(0,0,0,.07); }
  .card h3 { font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; color:#6b7280; margin-bottom:8px; }
  .big { font-size:1.7rem; font-weight:700; }
  .big.pos { color:var(--green); } .big.neg { color:var(--red); } .big.neu { color:var(--grey); }
  .sub { color:#6b7280; font-size:.82rem; margin-top:4px; }
  .bar-outer { background:#e5e7eb; border-radius:999px; height:12px; margin-top:10px; overflow:hidden; }
  .bar-inner { background:var(--blue); height:100%; border-radius:999px; }
  .chart-card { margin-bottom:20px; }
  .charts { display:grid; grid-template-columns:2fr 1fr; gap:16px; margin-bottom:20px; }
  @media (max-width:800px){ .charts { grid-template-columns:1fr; } }
  table { width:100%; border-collapse:collapse; font-size:.86rem; }
  th { text-align:left; color:#6b7280; font-size:.75rem; text-transform:uppercase; padding:6px 8px; }
  td { padding:7px 8px; border-top:1px solid #f1f5f9; vertical-align:top; }
  td.dt { white-space:nowrap; color:#6b7280; }
  td.ttl { line-height:1.35; }
  .chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; white-space:nowrap; }
  .chip.pos { background:#dcfce7; color:#166534; }
  .chip.neg { background:#fee2e2; color:#991b1b; }
  .chip.neu { background:#f3f4f6; color:#4b5563; }
  .dot { display:inline-block; width:14px; height:14px; border-radius:50%; margin-right:6px; }
  .dot.ok { background:var(--green); }
  .dot.warn { background:#d97706; }
  .dot.bad { background:var(--red); }
  .note { background:#fffbeb; border:1px solid #fde68a; color:#92400e; border-radius:10px;
          padding:12px 16px; font-size:.84rem; margin-bottom:20px; }
  footer { color:#9ca3af; font-size:.78rem; text-align:center; margin-top:24px; }
__REGIME_CSS__
</style>
</head>
<body>
<div class="wrap">

  <h1>&#128240; BIST 100 News Sentiment <span class="pill __RUN_CLS__">__RUN_STATUS__</span></h1>
  <div class="meta">Reads Turkish financial news every weekday, measures the mood, and compares it
  with the Istanbul stock exchange. Updated: __GENERATED__</div>

  <div class="note"><b>Research project</b> — production prompt p3 reached 83.3% categorical
  agreement with this project's held-out human-label rubric. Market-return analysis remains
  exploratory and is not a validated strategy. Nothing here is investment advice.</div>

  <nav class="toc">
    <a href="#data-health">Data Health</a>
    <a href="#news-regime">News Regime</a>
    <a href="#signal-families">Signal Families</a>
    <a href="#candidate-events">Candidate Events</a>
    <a href="#windows">Market Windows</a>
    <a href="#validation">Predictive Validation</a>
    <a href="#future">Future Validation</a>
  </nav>

  <h2 id="data-health">Data Health</h2>
  <div class="grid">
    <div class="card">
      <h3>Latest session-aligned news mood</h3>
      __LATEST__
    </div>
    <div class="card">
      <h3>Headlines analysed</h3>
      <div class="big">__TOTAL__</div>
      <div class="sub">from __SOURCES__ Turkish news sources<br>__FIRST_DAY__ &rarr; __LAST_DAY__</div>
    </div>
    <div class="card">
      <h3>Exploratory reporting observations</h3>
      <div class="big">__RELIABLE__ / __NEEDED__ observations</div>
      <div class="bar-outer"><div class="bar-inner" style="width:__PCT__%"></div></div>
      <div class="sub">eligible news-and-market observations; __NEEDED__ is a display gate, not validation</div>
    </div>
  </div>

  <div class="card chart-card">
    <h3>Session-aligned unweighted news mood vs. stock market (last 60 sessions)</h3>
    <div class="sub" style="margin-bottom:10px">Bars are the unweighted baseline assigned to
    the first market session able to react. Faded bars are thin observations. Blue line =
    BIST 100 session close. Any subsequent-return comparison is exploratory.</div>
    <canvas id="mainChart" height="95"></canvas>
  </div>

  <div class="charts" id="signal-families">
    <div class="card">
      <h3>Latest headlines</h3>
      <table>
        <tr><th>Date</th><th>Headline</th><th>Mood</th></tr>
        __HEADS__
      </table>
    </div>
    <div class="card">
      <h3>What the news is about</h3>
      <canvas id="catChart"></canvas>
      <h3 style="margin-top:18px">Recent daily runs</h3>
      <div style="margin-top:6px">__DOTS__</div>
      <div class="sub">one dot per pipeline run &mdash; hover for details</div>
    </div>
  </div>

  <footer>Generated automatically by dashboard.py &middot; data: RSS headlines + Yahoo Finance &middot;
  sentiment: gpt-5-mini/p3 (83.3% agreement with the held-out project rubric)</footer>
</div>

<script>
const labels  = __LABELS__;
const scores  = __SCORES__;
const closes  = __CLOSES__;
const barcols = __BARCOLS__;

new Chart(document.getElementById('mainChart'), {
  data: {
    labels: labels,
    datasets: [
      { type:'bar',  label:'Session-aligned unweighted news mood', data:scores, backgroundColor:barcols, yAxisID:'y1', order:2 },
      { type:'line', label:'BIST 100',  data:closes, borderColor:'#2C7BB6', backgroundColor:'#2C7BB6',
        spanGaps:true, pointRadius:0, borderWidth:2, tension:.25, yAxisID:'y', order:1 },
    ]
  },
  options: {
    interaction:{ mode:'index', intersect:false },
    plugins:{ legend:{ labels:{ boxWidth:14 } } },
    scales: {
      y:  { position:'right', title:{ display:true, text:'BIST 100' }, grid:{ display:false } },
      y1: { position:'left',  min:-1, max:1, title:{ display:true, text:'Unweighted news mood (-1 … +1)' } },
      x:  { ticks:{ maxTicksLimit:15 } }
    }
  }
});

new Chart(document.getElementById('catChart'), {
  type:'doughnut',
  data: {
    labels: __CAT_LABELS__,
    datasets:[{ data: __CAT_COUNTS__,
      backgroundColor:['#2C7BB6','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4','#8BC34A','#FFC107','#607D8B','#9E9E9E'] }]
  },
  options:{ plugins:{ legend:{ position:'bottom', labels:{ boxWidth:12, font:{ size:10 } } } } }
});
</script>
__REGIME__
__EVENTS__
</body>
</html>
"""


# -- CLI ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the HTML dashboard")
    parser.add_argument("--db",     default=DB_PATH)
    parser.add_argument("--output", default=DASHBOARD_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = generate(db_path=args.db, output=args.output)
    print(f"Dashboard saved: {path}")


if __name__ == "__main__":
    main()
