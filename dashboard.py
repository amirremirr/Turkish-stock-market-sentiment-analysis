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

import database as db
from config import DB_PATH, MINIMUM_HEADLINES_PER_DAY, MINIMUM_OVERLAP_DAYS

logger = logging.getLogger(__name__)

DASHBOARD_OUTPUT = "dashboard.html"
CHART_DAYS = 60          # window shown in the main chart
RECENT_HEADLINES = 12    # rows in the headlines table
RECENT_RUNS = 7          # pipeline run dots


# -- Data collection -------------------------------------------------------------

def _collect(db_path: str = DB_PATH) -> dict:
    with db._conn(db_path) as con:
        total = con.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
        sources = con.execute("SELECT COUNT(DISTINCT source) FROM headlines").fetchone()[0]
        first_day, last_day = con.execute(
            "SELECT MIN(published_at), MAX(published_at) FROM headlines"
        ).fetchone()

        sent = con.execute(
            """SELECT ds.date, ds.avg_score, ds.headline_count,
                      ds.positive_count, ds.neutral_count, ds.negative_count,
                      bp.close
               FROM daily_sentiment ds
               LEFT JOIN bist100_prices bp ON bp.date = ds.date
               ORDER BY ds.date DESC LIMIT ?""",
            (CHART_DAYS,),
        ).fetchall()
        sent = list(reversed(sent))

        reliable = con.execute(
            """SELECT COUNT(*) FROM daily_sentiment ds
               JOIN bist100_prices bp ON bp.date = ds.date
               WHERE ds.headline_count >= ?""",
            (MINIMUM_HEADLINES_PER_DAY,),
        ).fetchone()[0]

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
        "sent": [dict(r) for r in sent],
        "reliable": reliable,
        "cats": [dict(r) for r in cats],
        "heads": [dict(r) for r in heads],
        "runs": [dict(r) for r in runs],
    }


# -- Rendering helpers -----------------------------------------------------------

def _mood(score: float) -> tuple[str, str, str]:
    """(emoji, label, css-class) for a daily average score."""
    if score > 0.05:
        return "&#128578;", "Positive", "pos"
    if score < -0.05:
        return "&#128577;", "Negative", "neg"
    return "&#128528;", "Neutral", "neu"


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
            f'<div class="sub">{latest["date"]} &middot; {latest["headline_count"]} headlines '
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
        ok = r["status"] in ("ok", "recovered")
        cls = "ok" if ok else "bad"
        tip = f'{r["started_at"][:10]} — {r["status"]}, {r["headlines_scraped"]} new headlines'
        dots.append(f'<span class="dot {cls}" title="{tip}"></span>')
    dots_html = "".join(dots)
    last_status = d["runs"][0]["status"] if d["runs"] else "—"
    status_ok = last_status in ("ok", "recovered")

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
        "__RUN_STATUS__":  "Running normally" if status_ok else f"Last run: {last_status}",
        "__RUN_CLS__":     "ok" if status_ok else "bad",
        "__LABELS__":      json.dumps(labels),
        "__SCORES__":      json.dumps(scores),
        "__CLOSES__":      json.dumps(closes),
        "__BARCOLS__":     json.dumps(barcols),
        "__CAT_LABELS__":  json.dumps(cat_labels, ensure_ascii=False),
        "__CAT_COUNTS__":  json.dumps(cat_counts),
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
  .dot.ok { background:var(--green); } .dot.bad { background:var(--red); }
  .note { background:#fffbeb; border:1px solid #fde68a; color:#92400e; border-radius:10px;
          padding:12px 16px; font-size:.84rem; margin-bottom:20px; }
  footer { color:#9ca3af; font-size:.78rem; text-align:center; margin-top:24px; }
</style>
</head>
<body>
<div class="wrap">

  <h1>&#128240; BIST 100 News Sentiment <span class="pill __RUN_CLS__">__RUN_STATUS__</span></h1>
  <div class="meta">Reads Turkish financial news every weekday, measures the mood, and compares it
  with the Istanbul stock exchange. Updated: __GENERATED__</div>

  <div class="note"><b>Research project</b> — the model agrees with a human reader about 5 times
  out of 6 on checked headlines, and we still need more data before mood&ndash;market statistics
  mean anything. Nothing here is investment advice.</div>

  <div class="grid">
    <div class="card">
      <h3>Today's news mood</h3>
      __LATEST__
    </div>
    <div class="card">
      <h3>Headlines analysed</h3>
      <div class="big">__TOTAL__</div>
      <div class="sub">from __SOURCES__ Turkish news sources<br>__FIRST_DAY__ &rarr; __LAST_DAY__</div>
    </div>
    <div class="card">
      <h3>Progress to reliable statistics</h3>
      <div class="big">__RELIABLE__ / __NEEDED__ days</div>
      <div class="bar-outer"><div class="bar-inner" style="width:__PCT__%"></div></div>
      <div class="sub">days with enough news AND market data &mdash; statistics unlock at __NEEDED__</div>
    </div>
  </div>

  <div class="card chart-card">
    <h3>News mood vs. stock market (last 60 days)</h3>
    <div class="sub" style="margin-bottom:10px">Green bars = positive news days, red = negative.
    Faded bars had too little news to trust. Blue line = BIST 100 index closing value.</div>
    <canvas id="mainChart" height="95"></canvas>
  </div>

  <div class="charts">
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
  sentiment: gpt-5-mini (84.5% on human-checked headlines)</footer>
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
      { type:'bar',  label:'News mood', data:scores, backgroundColor:barcols, yAxisID:'y1', order:2 },
      { type:'line', label:'BIST 100',  data:closes, borderColor:'#2C7BB6', backgroundColor:'#2C7BB6',
        spanGaps:true, pointRadius:0, borderWidth:2, tension:.25, yAxisID:'y', order:1 },
    ]
  },
  options: {
    interaction:{ mode:'index', intersect:false },
    plugins:{ legend:{ labels:{ boxWidth:14 } } },
    scales: {
      y:  { position:'right', title:{ display:true, text:'BIST 100' }, grid:{ display:false } },
      y1: { position:'left',  min:-1, max:1, title:{ display:true, text:'News mood (-1 … +1)' } },
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
