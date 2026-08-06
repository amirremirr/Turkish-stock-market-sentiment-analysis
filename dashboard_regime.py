"""Render the News Regime section of the dashboard.

Reads the stored analytical tables rather than recomputing the corpus on page
load: the dashboard must show what the pipeline computed, and recomputing would
create a second definition that can silently disagree with the first.

Presentation rules that are not cosmetic:

* Every value carries a **label and a number**. Colour is never the only carrier
  of meaning, so the page stays readable for colour-blind viewers and in print.
* NULL and insufficient samples are printed as such. A blank or a zero would
  read as "neutral" when the truth is "not enough observations".
* Level, change, abnormal position, disagreement and attention are shown in
  separate columns because they answer different questions.
* No buy/sell language, no recommendation, and no claim that any variant is
  validated or superior.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

import pandas as pd

# Families are shown under readable names; the raw key stays available in the
# driver table so the page never hides what it is actually reporting.
FAMILY_LABELS = {
    "monetary_policy": "Monetary policy",
    "inflation_macro": "Inflation & macro",
    "political_regulatory_risk": "Political / regulatory risk",
    "fx_lira": "Currency (lira)",
    "banking_financial_sector": "Banking sector",
    "company_kap": "Listed companies (KAP)",
    "global_risk": "Global risk",
    "market_recap": "Market recap",
    "media_narrative": "Media narrative",
    "other": "Other",
    "__domestic__": "Domestic only (composite)",
}

SUFFICIENCY_LABELS = {
    "sufficient": "Sufficient",
    "thin_sample": "Thin sample",
    "single_source": "Single source",
    "insufficient": "Insufficient",
}


def _num(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    """Format a number, or say plainly that there isn't one."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '<span class="null">n/a</span>'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '<span class="null">n/a</span>'
    if pd.isna(number):
        return '<span class="null">n/a</span>'
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '<span class="null">n/a</span>'
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return '<span class="null">n/a</span>'


def _tone_label(value: Any) -> str:
    """A word for the tone, so meaning never rests on colour alone."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "no reading"
    number = float(value)
    if number > 0.05:
        return "positive"
    if number < -0.05:
        return "negative"
    return "neutral"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_regime_section(
    regime: Dict[str, Any],
    drivers: pd.DataFrame,
    *,
    family_version: str = "",
    experiment_id: str = "",
) -> str:
    """Return the News Regime HTML fragment."""

    if not regime or regime.get("status") != "ok":
        return (
            '<section class="card"><h2>News Regime</h2>'
            '<p class="null">No family indicators are stored yet. Run the '
            'pipeline to populate them.</p></section>'
        )

    families = regime.get("families", [])
    composite = regime.get("domestic_only")

    header = f"""
<section class="card" id="news-regime">
  <h2>News Regime</h2>
  <p class="sub">
    Descriptive indicators for {_esc(regime['as_of'])} across
    {len(families)} signal families &middot; {regime.get('sessions_available', 0)}
    sessions available.
    <strong>These are descriptive measures, not trading signals.</strong>
    No result here is a validated predictive relationship.
  </p>
  <p class="sub versions">
    taxonomy <code>{_esc(family_version)}</code> &middot;
    experiment <code>{_esc(experiment_id)}</code> &middot;
    report <code>{_esc(regime.get('version', ''))}</code>
  </p>
"""

    summary_items = [
        ("Most positive family", FAMILY_LABELS.get(
            regime.get("most_positive"), regime.get("most_positive")) or "n/a"),
        ("Most negative family", FAMILY_LABELS.get(
            regime.get("most_negative"), regime.get("most_negative")) or "n/a"),
        ("Largest 5-session move", FAMILY_LABELS.get(
            regime.get("largest_5_session_move"),
            regime.get("largest_5_session_move")) or "n/a"),
        ("Unusual vs own history", ", ".join(
            FAMILY_LABELS.get(f, f) for f in regime.get("unusual_percentiles", [])
        ) or "none"),
        ("Elevated disagreement", ", ".join(
            FAMILY_LABELS.get(f, f) for f in regime.get("elevated_disagreement", [])
        ) or "none"),
        ("Elevated volume", ", ".join(
            FAMILY_LABELS.get(f, f) for f in regime.get("elevated_volume", [])
        ) or "none"),
    ]
    summary = '<div class="regime-summary">' + "".join(
        f'<div class="regime-kv"><span class="k">{_esc(k)}</span>'
        f'<span class="v">{_esc(v)}</span></div>'
        for k, v in summary_items
    ) + "</div>"

    if composite:
        level = composite["level"]["simple_mean"]
        composite_html = f"""
  <div class="composite">
    <strong>Domestic-only composite</strong>
    &middot; tone {_num(level, signed=True)} ({_tone_label(level)})
    &middot; {composite['level']['headline_count']} headlines
    from {composite['level']['source_count']} sources
    &middot; sample {SUFFICIENCY_LABELS.get(composite['sample_sufficiency'],
                                            composite['sample_sufficiency'])}
    <span class="note">Excludes global risk and market recap. The overall
    signal-variant series is published separately and is unchanged.</span>
  </div>
"""
    else:
        composite_html = ""

    rows = []
    for family in families:
        key = family["signal_family"]
        if key == "__domestic__":
            continue
        level = family["level"]["simple_mean"]
        sufficiency = family["sample_sufficiency"]
        insufficient = sufficiency != "sufficient"
        rows.append(f"""
    <tr class="{'insufficient' if insufficient else ''}"
        data-family="{_esc(key)}">
      <td class="fam">{_esc(FAMILY_LABELS.get(key, key))}</td>
      <td>{_num(level, signed=True)} <span class="tag">{_tone_label(level)}</span></td>
      <td>{_num(family['change']['vs_5_sessions'], signed=True)}</td>
      <td>{_num(family['change']['vs_20_sessions'], signed=True)}</td>
      <td>{_num(family['abnormal']['abnormal_tone'], signed=True)}</td>
      <td>{_pct(family['abnormal']['rolling_percentile'])}
          {'<span class="tag">unusual</span>' if family['abnormal']['is_unusual'] else ''}</td>
      <td>{_num(family['disagreement']['cross_outlet_std'])}
          {'' if family['disagreement']['min_sources_met']
             else '<span class="tag">too few sources</span>'}</td>
      <td>{_num(family['attention']['volume_z'], 2, signed=True)}
          {'<span class="tag">elevated</span>' if family['attention']['is_elevated'] else ''}</td>
      <td>{family['attention']['source_breadth']}</td>
      <td>{family['level']['headline_count']}</td>
      <td>{_pct(family['quality']['market_recap_share'])}</td>
      <td>{family['quality']['unknown_timing_count']}</td>
      <td><span class="suff suff-{_esc(sufficiency)}">
          {_esc(SUFFICIENCY_LABELS.get(sufficiency, sufficiency))}</span></td>
    </tr>""")

    table = f"""
  <div class="table-scroll">
  <table class="regime">
    <thead><tr>
      <th>Family</th><th>Level</th><th>Chg 5</th><th>Chg 20</th>
      <th>Abnormal</th><th>Percentile</th><th>Outlet spread</th>
      <th>Volume z</th><th>Sources</th><th>Headlines</th>
      <th>Recap share</th><th>Unknown timing</th><th>Sample</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  </div>
  <p class="legend">
    <strong>Level</strong> current mean tone &middot;
    <strong>Chg</strong> change vs 5 and 20 sessions ago &middot;
    <strong>Abnormal</strong> tone minus its own prior-window mean &middot;
    <strong>Percentile</strong> position within prior history &middot;
    <strong>Outlet spread</strong> dispersion across outlets (news disagreement,
    not market uncertainty) &middot;
    <strong>Volume z</strong> headline count vs prior sessions.
    <em>n/a</em> means the value could not be defensibly calculated, most often
    because too little history or too few independent sources were available.
  </p>
"""

    return header + summary + composite_html + table + _render_drivers(
        regime, drivers
    ) + "</section>"


def _render_drivers(regime: Dict[str, Any], drivers: pd.DataFrame) -> str:
    """Show the individual headlines carrying the most weight."""

    if drivers is None or drivers.empty:
        return '<p class="null">No driver headlines available for this session.</p>'

    session = regime.get("as_of")
    subset = drivers[drivers["signal_date"] == session] if session else drivers
    if subset.empty:
        return '<p class="null">No driver headlines for this session.</p>'

    ordered = subset.sort_values("sentiment_score", ascending=False)
    top = pd.concat([ordered.head(5), ordered.tail(5)]).drop_duplicates(subset=["id"])

    rows = []
    for _, row in top.iterrows():
        score = row.get("sentiment_score")
        recap = "yes" if row.get("is_market_recap") else "no"
        excluded = "excluded" if row.get("is_excluded") else "included"
        rows.append(f"""
    <tr>
      <td class="title">{_esc(row.get('title'))}</td>
      <td>{_esc(row.get('source'))}</td>
      <td>{_esc(row.get('published_timestamp') or row.get('published_at'))}</td>
      <td>{_esc(row.get('timing_bucket'))}</td>
      <td>{_esc(row.get('category'))}</td>
      <td>{_esc(FAMILY_LABELS.get(row.get('signal_family'), row.get('signal_family')))}</td>
      <td>{_num(score, signed=True)} <span class="tag">{_tone_label(score)}</span></td>
      <td>{_num(row.get('relevance'), 2)}</td>
      <td>{_num(row.get('abnormal_contribution'), signed=True)}</td>
      <td>{recap}</td>
      <td>{excluded}</td>
      <td><code>{_esc(row.get('experiment_id'))}</code></td>
    </tr>""")

    return f"""
  <h3>Drivers for {_esc(session)}</h3>
  <p class="sub">Highest and lowest scored headlines in this session. Listing a
  headline here describes its weight in the average; it does not assert that it
  caused anything.</p>
  <div class="table-scroll">
  <table class="drivers">
    <thead><tr>
      <th>Headline</th><th>Source</th><th>Published</th><th>Timing</th>
      <th>Category</th><th>Family</th><th>Sentiment</th><th>Relevance</th>
      <th>Abnormal contrib.</th><th>Recap</th><th>Exclusion</th><th>Experiment</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  </div>
"""


REGIME_CSS = """
#news-regime .versions code { background:#eef2f7; padding:1px 5px; border-radius:3px; }
.regime-summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:8px; margin:12px 0; }
.regime-kv { background:#f7f9fc; border:1px solid #e3e9f0; border-radius:6px; padding:8px 10px; }
.regime-kv .k { display:block; font-size:11px; text-transform:uppercase;
  letter-spacing:.4px; color:#5a6b7d; }
.regime-kv .v { display:block; font-size:14px; font-weight:600; color:#1b2b3a; }
.composite { background:#f2f7ff; border:1px solid #d6e4f7; border-radius:6px;
  padding:10px 12px; margin:10px 0; font-size:13px; }
.composite .note { display:block; color:#5a6b7d; font-size:11px; margin-top:4px; }
.table-scroll { overflow-x:auto; }
table.regime, table.drivers { width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }
table.regime th, table.drivers th { text-align:left; padding:6px 8px;
  border-bottom:2px solid #dfe6ee; color:#3b4c5e; white-space:nowrap; }
table.regime td, table.drivers td { padding:6px 8px; border-bottom:1px solid #eef2f6; }
table.regime tr.insufficient { background:#fcfcf5; }
table.drivers td.title { max-width:340px; }
.fam { font-weight:600; }
.tag { display:inline-block; font-size:10px; padding:1px 5px; border-radius:8px;
  background:#e8edf3; color:#3b4c5e; margin-left:4px; }
.suff { font-size:11px; padding:2px 6px; border-radius:8px; }
.suff-sufficient { background:#e6f4ea; color:#1e4620; }
.suff-thin_sample { background:#fdf0d5; color:#6b4c00; }
.suff-single_source { background:#fdf0d5; color:#6b4c00; }
.suff-insufficient { background:#f3e6e6; color:#6b1e1e; }
.null { color:#8a97a3; font-style:italic; }
.legend { font-size:11px; color:#5a6b7d; margin-top:8px; line-height:1.6; }
"""


# -- Candidate events ----------------------------------------------------------

def render_event_section(
    events: "pd.DataFrame",
    briefs: list,
    *,
    algorithm_version: str = "",
) -> str:
    """Render candidate-event exploration and briefs.

    The wording never upgrades an algorithmic grouping into a verified event,
    every group shows its review state and data-quality warnings, and no
    recommendation of any kind is produced.
    """

    if events is None or events.empty:
        return (
            '<section class="card"><h2>Candidate Events</h2>'
            '<p class="null">No candidate event groups are stored yet.</p>'
            '</section>'
        )

    ranked = events.sort_values(
        ["source_count", "headline_count"], ascending=False
    ).head(25)

    rows = []
    for _, row in ranked.iterrows():
        warnings = []
        if int(row.get("is_single_source") or 0):
            warnings.append("single source")
        if int(row.get("is_singleton") or 0):
            warnings.append("single headline")
        if int(row.get("signal_date_span") or 1) > 1:
            warnings.append("spans sessions")
        if int(row.get("unknown_timestamp_count") or 0):
            warnings.append("unknown time")
        rows.append(f"""
    <tr>
      <td><code>{_esc(str(row.get("group_key"))[-12:])}</code></td>
      <td>{_esc(row.get("event_type") or "unclassified")}</td>
      <td>{_esc(FAMILY_LABELS.get(row.get("signal_family"), row.get("signal_family")))}</td>
      <td>{_esc(row.get("primary_entity") or "none")}</td>
      <td>{_esc(row.get("first_seen_at"))}</td>
      <td>{_esc(row.get("signal_date"))}</td>
      <td>{int(row.get("source_count") or 0)}</td>
      <td>{int(row.get("headline_count") or 0)}</td>
      <td>{_num(row.get("mean_sentiment"), signed=True)}
          <span class="tag">{_tone_label(row.get("mean_sentiment"))}</span></td>
      <td>{_num(row.get("cross_source_dispersion"))}</td>
      <td>{_num(row.get("novelty"), 2)}</td>
      <td><span class="suff suff-{_esc(row.get("review_state"))}">
          {_esc(row.get("review_state"))}</span></td>
      <td>{_esc(", ".join(warnings) or "-")}</td>
    </tr>""")

    return f"""
<section class="card" id="candidate-events">
  <h2>Candidate Events</h2>
  <p class="sub">
    <strong>These are algorithmic groupings, not verified real-world events.</strong>
    Headlines are grouped by shared entity, signal family, time proximity and
    normalized-title similarity. Every grouping keeps its similarity score and
    rule, and none has been human-reviewed unless marked confirmed.
  </p>
  <p class="sub versions">algorithm <code>{_esc(algorithm_version)}</code></p>
  <div class="table-scroll">
  <table class="regime">
    <thead><tr>
      <th>Group</th><th>Type</th><th>Family</th><th>Entity</th>
      <th>First seen</th><th>First reactable</th><th>Sources</th>
      <th>Headlines</th><th>Tone</th><th>Outlet spread</th><th>Novelty</th>
      <th>Review</th><th>Quality</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
  <p class="legend">
    <strong>Outlet spread</strong> is disagreement among the outlets that
    covered the group, not market uncertainty. <strong>Novelty</strong> reflects
    how often the entity has already produced a candidate group, not whether the
    news is new. <em>n/a</em> means a value could not be defensibly computed.
    No trading recommendation is shown.
  </p>
</section>
"""


EVENT_CSS = """
.suff-unreviewed { background:#eef2f7; color:#3b4c5e; }
.suff-confirmed { background:#e6f4ea; color:#1e4620; }
.suff-rejected { background:#f3e6e6; color:#6b1e1e; }
"""
