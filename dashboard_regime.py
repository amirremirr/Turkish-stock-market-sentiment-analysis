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
import json
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

# Database identifiers are unchanged everywhere; these are display strings only.
# A reader should not have to know that `prior_close_to_reactable_close` means
# "yesterday's close to today's close" in order to read a table.
WINDOW_LABELS = {
    "reactable_open_to_close": "Open to close, first session that could react",
    "prior_close_to_reactable_open": "Overnight gap (previous close to open)",
    "prior_close_to_reactable_close": "Previous close to close",
}

CONTROL_LABELS = {
    "raw_return": "Raw return (no adjustment)",
    "residual_none": "Raw return (no adjustment)",
    "residual_em_lagged": "Adjusted for emerging markets",
    "residual_em_oil_fx_lagged": "Adjusted for emerging markets, oil and FX",
    "residual_em_contemporaneous": "Adjusted for same-day emerging markets",
}

BLOCKED_REASON_LABELS = {
    "intraday_prices_unavailable": "Published during trading hours (no intraday prices)",
    "publication_time_unknown": "Publication time unknown",
    "market_recap_excluded_by_default": "Market recap (tone follows the return)",
    "no_complete_price_bar": "No settled price bar",
    "no_prior_session_price_bar": "No settled bar for the previous session",
    "no_following_complete_session": "No following settled session",
    "event_members_span_incompatible_sessions": "Headlines react on different sessions",
}

FEATURE_SET_LABELS = {
    "none": "Unconditional mean",
    "previous_direction": "Yesterday's direction",
    "ar1": "Yesterday's return",
    "headline_count_only": "Headline count only",
    "net_tone_share": "Positive minus negative share",
    "market_controls_only": "Market factors only",
    "family_signals": "Topic tone",
    "abnormal_tone": "Unusual tone vs history",
    "disagreement": "Outlet disagreement",
    "attention_shock": "Attention and breadth",
    "event_tone_novelty": "Event tone and novelty",
    "controls_plus_news": "Market factors plus news",
}

MODEL_LABELS = {
    "mean": "Training average",
    "majority": "Majority direction",
    "ridge": "Ridge regression",
    "logistic": "Logistic regression",
}


def label(mapping: Dict[str, str], key: Any) -> str:
    """Human-readable name, falling back to the identifier itself."""

    text = "" if key is None else str(key)
    return mapping.get(text, text)


def tone_state(value: Any) -> tuple:
    """(word, css class) for a tone level. Never colour alone."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "no reading", "state-null"
    number = float(value)
    if number > 0.15:
        return "clearly positive", "state-pos"
    if number > 0.05:
        return "mildly positive", "state-pos"
    if number < -0.15:
        return "clearly negative", "state-neg"
    if number < -0.05:
        return "mildly negative", "state-neg"
    return "broadly neutral", "state-flat"


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
    """A word for the tone, so meaning never rests on colour alone.

    Delegates to :func:`tone_state` so the thresholds exist in one place; two
    copies would eventually disagree about what counts as positive.
    """

    return tone_state(value)[0]


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_overview_section(
    regime: Dict[str, Any],
    *,
    overall_tone: Any = None,
    frozen=None,
    run_status: str = "",
) -> str:
    """The 20-second read: mood, what is driving it, and whether anything is proven.

    Everything here already exists elsewhere on the page. The point of this
    section is ordering and restraint: six numbers a person can actually hold in
    their head, then the headlines behind them, then the one sentence about
    predictive status. Detail lives further down, not here.
    """

    if not regime or regime.get("status") != "ok":
        return (
            '<section class="card" id="overview"><h2>Overview</h2>'
            '<p class="null">No indicators are stored yet. Run the pipeline to '
            'populate them.</p></section>'
        )

    families = {f["signal_family"]: f for f in regime.get("families", [])}
    composite = regime.get("domestic_only")

    # The canonical overall aggregate, passed in from daily_signal_variants --
    # the same number the chart and the "latest mood" card use. Deriving a
    # second overall here from the family means would create a competing
    # definition that can silently disagree with the published one, which is
    # exactly what this module exists not to do.
    overall = overall_tone
    domestic = composite["level"]["simple_mean"] if composite else None

    def _tone_tile(title: str, value, note: str) -> str:
        word, css = tone_state(value)
        return f"""
    <div class="tile">
      <div class="tile-k">{_esc(title)}</div>
      <div class="tile-v {css}">{_num(value, 2, signed=True)}</div>
      <div class="tile-state {css}">{_esc(word)}</div>
      <div class="tile-note">{note}</div>
    </div>"""

    def _family_tile(title: str, key, note: str) -> str:
        family = families.get(key)
        if family is None:
            # The ranking is deliberately empty rather than filled from a
            # one-headline topic. Say which, instead of printing a bare n/a:
            # "not enough news yet" is a fact a reader can act on, and it is
            # the common state early in a session.
            thin = sorted(
                (f for k, f in families.items()
                 if k not in ("__domestic__", "market_recap")
                 and f["level"]["simple_mean"] is not None),
                key=lambda f: -f["level"]["headline_count"],
            )
            detail = (
                f"best covered so far: "
                f"{FAMILY_LABELS.get(thin[0]['signal_family'], thin[0]['signal_family'])} "
                f"({thin[0]['level']['headline_count']} headlines)"
                if thin else "no economic topic has headlines yet"
            )
            return f"""
    <div class="tile">
      <div class="tile-k">{_esc(title)}</div>
      <div class="tile-v state-null">Not enough news yet</div>
      <div class="tile-note">no economic topic reached a sufficient sample this
      session &mdash; {_esc(detail)}</div>
    </div>"""
        level = family["level"]["simple_mean"]
        word, css = tone_state(level)
        return f"""
    <div class="tile">
      <div class="tile-k">{_esc(title)}</div>
      <div class="tile-v">{_esc(FAMILY_LABELS.get(key, key))}</div>
      <div class="tile-state {css}">{_num(level, 2, signed=True)} &middot; {_esc(word)}</div>
      <div class="tile-note">{note}</div>
    </div>"""

    elevated_volume = regime.get("elevated_volume") or []
    elevated_disagreement = regime.get("elevated_disagreement") or []

    volume_state = (
        f"Busier than usual in {len(elevated_volume)} topic"
        f"{'s' if len(elevated_volume) != 1 else ''}"
        if elevated_volume else "Normal news volume"
    )
    volume_detail = (
        ", ".join(FAMILY_LABELS.get(f, f) for f in elevated_volume[:3])
        if elevated_volume else "no topic is unusually busy today"
    )
    disagreement_state = (
        f"Outlets disagree in {len(elevated_disagreement)} topic"
        f"{'s' if len(elevated_disagreement) != 1 else ''}"
        if elevated_disagreement else "Outlets broadly agree"
    )
    disagreement_detail = (
        ", ".join(FAMILY_LABELS.get(f, f) for f in elevated_disagreement[:3])
        if elevated_disagreement else "no unusual spread between outlets"
    )

    recap_share = regime.get("market_recap_share")
    recap_note = (
        f"{_pct(recap_share)} of today's headlines were market recap &mdash; "
        f"reporting that restates the day's index move. Recap is excluded from "
        f"the topic ranking above because its tone follows the return rather "
        f"than describing news."
        if recap_share is not None else
        "Market-recap share could not be computed for this session."
    )

    drivers_html = _driver_lists(
        regime.get("top_positive_drivers") or [],
        regime.get("top_negative_drivers") or [],
    )

    verdict_line = (
        "No validated predictive signal. The one completed study found none, "
        "and an untouched future test is still accumulating data."
    )
    if frozen is not None and not getattr(frozen, "empty", True):
        record = frozen.sort_values("frozen_at").iloc[0]
        verdict_line = (
            f"<strong>No validated predictive signal.</strong> The completed "
            f"study tested "
            f"{int(record.get('specifications_run') or 0)} specifications over "
            f"{int(record.get('independent_sessions') or 0)} independent "
            f"sessions and found "
            f"{int(record.get('successes') or 0)} that met its criteria."
        )

    health = _health_line(run_status)

    return f"""
<section class="card" id="overview">
  <h2>Overview</h2>
  <p class="sub lede">Turkish financial-news mood for
  <strong>{_esc(regime['as_of'])}</strong>, the first market session able to
  react to it.</p>

  <div class="tiles">
    {_tone_tile("Overall news tone", overall,
                "the published session aggregate, all topics included")}
    {_tone_tile("Domestic-only tone", domestic,
                "excludes global risk and market recap")}
    {_family_tile("Most positive topic", regime.get("most_positive"),
                  "economic topics only")}
    {_family_tile("Most negative topic", regime.get("most_negative"),
                  "economic topics only")}
    <div class="tile">
      <div class="tile-k">News volume</div>
      <div class="tile-v">{_esc(volume_state)}</div>
      <div class="tile-note">{_esc(volume_detail)}</div>
    </div>
    <div class="tile">
      <div class="tile-k">Outlet disagreement</div>
      <div class="tile-v">{_esc(disagreement_state)}</div>
      <div class="tile-note">{_esc(disagreement_detail)}</div>
    </div>
  </div>

  <p class="sub recap-note">{recap_note}</p>

  <h3 class="sub-head">What's driving today?</h3>
  {drivers_html}

  <div class="status-strip">
    <div><span class="strip-k">Predictive status</span> {verdict_line}</div>
    <div><span class="strip-k">Data status</span> {health}</div>
  </div>
</section>
"""


def _driver_lists(positive, negative) -> str:
    """Top positive and negative headlines, from the stored driver rows."""

    if not positive and not negative:
        return '<p class="null">No scored headlines for this session yet.</p>'

    def _column(title: str, rows, css: str) -> str:
        if not rows:
            return (f'<div class="driver-col"><h4 class="{css}">{title}</h4>'
                    f'<p class="null">none</p></div>')
        items = "".join(f"""
      <li>
        <span class="driver-score {css}">{_num(r.get("sentiment_score"), 2, signed=True)}</span>
        <span class="driver-title">{_esc(r.get("title"))}</span>
        <span class="driver-meta">{_esc(r.get("source"))} &middot;
        {_esc(FAMILY_LABELS.get(r.get("signal_family"), r.get("signal_family")))}</span>
      </li>""" for r in rows[:4])
        return (f'<div class="driver-col"><h4 class="{css}">{title}</h4>'
                f'<ul class="drivers">{items}</ul></div>')

    return f"""
  <div class="driver-grid">
    {_column("Most positive", positive, "state-pos")}
    {_column("Most negative", negative, "state-neg")}
  </div>"""


def _health_line(run_status: str) -> str:
    """Plain-language pipeline state.

    "degraded" is accurate and tells a non-technical reader nothing. What it
    almost always means here is that today's price bar has not settled yet,
    which is a normal morning condition rather than a fault.
    """

    status = (run_status or "").strip().lower()
    if status == "success":
        return "All sources collected and settled."
    if status == "degraded":
        return ("<strong>Data partially complete.</strong> Collection and "
                "scoring succeeded; at least one market input is still "
                "provisional or unavailable. Details under Data Health.")
    if status == "failed":
        return ("<strong>Last run failed.</strong> Figures may be stale. "
                "Details under Data Health.")
    return "No completed run recorded yet."


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
  <h2>Signal Families</h2>
  <p class="sub">Tone by topic for {_esc(regime['as_of'])} &mdash;
  {len(families)} topics over {regime.get('sessions_available', 0)} sessions.
  Descriptive measures, not trading signals.</p>
"""

    # The compact view: one row per topic, four columns a person can read.
    # Everything else moved under Advanced metrics rather than being deleted.
    compact_rows = []
    for family in families:
        key = family["signal_family"]
        if key == "__domestic__":
            continue
        level = family["level"]["simple_mean"]
        word, css = tone_state(level)
        flags = []
        if family["abnormal"]["is_unusual"]:
            flags.append("unusual vs history")
        if family["attention"]["is_elevated"]:
            flags.append("busier than usual")
        if (family["disagreement"]["cross_outlet_std"] is not None
                and family["disagreement"]["cross_outlet_std"] >= 0.30):
            flags.append("outlets disagree")
        sufficiency = family["sample_sufficiency"]
        if sufficiency != "sufficient":
            flags.append(SUFFICIENCY_LABELS.get(sufficiency, sufficiency).lower())
        compact_rows.append(f"""
    <tr class="{'insufficient' if sufficiency != 'sufficient' else ''}">
      <td class="fam">{_esc(FAMILY_LABELS.get(key, key))}</td>
      <td class="{css}">{_num(level, 2, signed=True)}
          <span class="tag">{_esc(word)}</span></td>
      <td>{family['level']['headline_count']}</td>
      <td class="flags">{_esc(", ".join(flags)) or "&mdash;"}</td>
    </tr>""")

    compact = f"""
  <div class="table-scroll">
  <table class="regime compact">
    <thead><tr>
      <th>Topic</th><th>Tone</th><th>Headlines</th><th>Notes</th>
    </tr></thead>
    <tbody>{''.join(compact_rows)}</tbody>
  </table>
  </div>
"""

    if composite:
        level = composite["level"]["simple_mean"]
        word, css = tone_state(level)
        composite_html = f"""
  <div class="composite">
    <strong>Domestic-only composite</strong>
    &middot; tone <span class="{css}">{_num(level, 2, signed=True)} ({_esc(word)})</span>
    &middot; {composite['level']['headline_count']} headlines
    from {composite['level']['source_count']} sources
    &middot; sample {SUFFICIENCY_LABELS.get(composite['sample_sufficiency'],
                                            composite['sample_sufficiency'])}
    <span class="note">Excludes global risk and market recap.</span>
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
  <details class="expander">
    <summary>Advanced metrics &mdash; full family table</summary>
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
  </details>

  <details class="expander">
    <summary>Methodology and technical details</summary>
    <p class="sub">Tone is the unweighted mean sentiment of the headlines
    assigned to each topic on the first market session able to react to them.
    Abnormal tone is measured against a <em>prior-only</em> rolling window, so
    no reading uses its own future. Disagreement requires at least three
    independent outlets; below that it is reported as <em>n/a</em> rather than
    as a fabricated zero.</p>
    <p class="sub">These are descriptive measures. No result here is a
    validated predictive relationship, and none is a trading signal.</p>
    <p class="sub versions">
      taxonomy <code>{_esc(family_version)}</code> &middot;
      experiment <code>{_esc(experiment_id)}</code> &middot;
      report <code>{_esc(regime.get('version', ''))}</code>
    </p>
  </details>
"""

    drivers_detail = f"""
  <details class="expander">
    <summary>Headline-level detail for this session</summary>
    {_render_drivers(regime, drivers)}
  </details>
"""

    return header + compact + composite_html + table + drivers_detail + "</section>"


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
        if int(row.get("timing_conflict") or 0):
            # Members react on different sessions, so no single window is
            # unambiguously this group's. Shown, and excluded from evaluation.
            warnings.append("timing conflict")
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
/* Navigation: four groups, not fourteen sections. */
.toc { margin:18px 0 30px; display:flex; flex-wrap:wrap; gap:10px; }
.toc a { font-size:13px; font-weight:600; padding:8px 16px; border-radius:18px;
         text-decoration:none; background:#eef2f7; color:#31465c;
         border:1px solid #dbe3ec; }
.toc a:hover { background:#dbe3ec; }

/* Typography hierarchy: group > section > card > label. Each step is a clear
   size and weight change, so scanning does not depend on reading. */
h2.group { font-size:13px; letter-spacing:.10em; text-transform:uppercase;
           color:#7a8899; font-weight:700; margin:52px 0 14px;
           padding-bottom:9px; border-bottom:2px solid #e6ebf1; }
.card h2 { font-size:21px; margin:0 0 6px; letter-spacing:-.01em; }
.sub-head { font-size:15px; font-weight:700; color:#31465c;
            margin:26px 0 12px; }
.lede { font-size:15px; color:#4a5a6b; margin-bottom:22px; }
.status-line { font-size:12.5px; color:#7a8899; margin:2px 0 22px; }
.status-line a { color:#5a7391; }

/* Tiles: the six numbers a reader should get in ten seconds. */
.tiles { display:grid; gap:14px; margin:6px 0 4px;
         grid-template-columns:repeat(auto-fit, minmax(215px, 1fr)); }
.tile { background:#fbfcfd; border:1px solid #e6ebf1; border-radius:10px;
        padding:16px 18px; }
.tile-k { font-size:11.5px; text-transform:uppercase; letter-spacing:.06em;
          color:#8494a5; font-weight:700; margin-bottom:9px; }
.tile-v { font-size:23px; font-weight:700; color:#22303f; line-height:1.25;
          letter-spacing:-.01em; }
.tile-state { font-size:13px; font-weight:600; margin-top:5px; }
.tile-note { font-size:11.5px; color:#8494a5; margin-top:8px; line-height:1.5; }

/* State is always a word plus a colour, never a colour alone. */
.state-pos  { color:#1e7a3c; }
.state-neg  { color:#a3282b; }
.state-flat { color:#5a6c7e; }
.state-null { color:#98a4b3; }

.recap-note { font-size:12.5px; color:#7a8899; margin:16px 0 4px;
              padding-left:12px; border-left:3px solid #e6ebf1; }

/* What's driving today */
.driver-grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fit, minmax(290px, 1fr)); }
.driver-col h4 { font-size:12px; text-transform:uppercase; letter-spacing:.05em;
                 margin:0 0 10px; }
ul.drivers { list-style:none; margin:0; padding:0; }
ul.drivers li { padding:9px 0; border-bottom:1px solid #f0f3f7;
                display:grid; grid-template-columns:52px 1fr; gap:4px 12px; }
ul.drivers li:last-child { border-bottom:none; }
.driver-score { font-weight:700; font-size:13.5px; grid-row:span 2; }
.driver-title { font-size:13.5px; color:#2b3a48; line-height:1.45; }
.driver-meta { font-size:11.5px; color:#8494a5; }

.status-strip { margin-top:26px; padding-top:18px; border-top:1px solid #e6ebf1;
                display:grid; gap:9px; font-size:13px; color:#4a5a6b;
                line-height:1.55; }
.strip-k { display:inline-block; min-width:128px; font-weight:700;
           color:#8494a5; font-size:11.5px; text-transform:uppercase;
           letter-spacing:.05em; }

/* The result, stated once and plainly. */
.verdict-banner { background:#f6f8fa; border:1px solid #e0e6ed;
                  border-left:4px solid #8494a5; border-radius:8px;
                  padding:18px 20px; margin:4px 0 22px; }
.verdict-headline { font-size:18px; font-weight:700; color:#31465c; }
.verdict-sub { font-size:13px; color:#5a6c7e; margin-top:7px; line-height:1.6; }
.interpretation { font-size:13.5px; color:#4a5a6b; line-height:1.65;
                  margin:20px 0 4px; }

/* Expanders keep detail available without giving it equal weight. */
details.expander { margin:20px 0 0; border-top:1px solid #e6ebf1;
                   padding-top:14px; }
details.expander > summary { cursor:pointer; font-size:12.5px; font-weight:600;
                             color:#5a7391; list-style:none; padding:5px 0; }
details.expander > summary::-webkit-details-marker { display:none; }
details.expander > summary::before { content:"\\25B8  "; color:#98a4b3; }
details.expander[open] > summary::before { content:"\\25BE  "; }
details.expander > summary:hover { color:#31465c; }
details.page-note { margin:34px 0 0; }
details.page-note p { font-size:12.5px; color:#6b7a8b; line-height:1.65; }

table.regime.compact td { padding:9px 12px; }
table.regime.compact .flags { color:#8494a5; font-size:12px; }

@media (max-width: 720px) {
  .tiles { grid-template-columns:1fr; }
  .strip-k { display:block; min-width:0; }
}
.suff-unreviewed { background:#eef2f7; color:#3b4c5e; }
.suff-confirmed { background:#e6f4ea; color:#1e4620; }
.suff-rejected { background:#f3e6e6; color:#6b1e1e; }
.verdict-failure { background:#f3e6e6; color:#6b1e1e; }
.verdict-inconclusive { background:#fdf3e3; color:#6b4a1e; }
.verdict-success { background:#e6f4ea; color:#1e4620; }
"""


def render_market_windows_section(windows, dataset) -> str:
    """How many candidate events produce a tradable target, and why the rest do not.

    The blocked counts are the honest headline here: most of the corpus cannot
    be used for directional research, and the reasons are stated rather than
    quietly filtered away.
    """

    if dataset is None or getattr(dataset, "empty", True):
        return (
            '<section class="card" id="windows"><h2>Market Windows</h2>'
            '<p class="null">No market windows have been built yet.</p></section>'
        )

    primary = dataset[dataset["window_name"] == "reactable_open_to_close"]
    eligible = primary[
        (primary["eligibility_status"] == "eligible")
        & (primary["is_tradable_window"] == 1)
        & primary["raw_return"].notna()
    ]
    sessions = eligible["first_reactable_session"].dropna().nunique()

    blocked = (
        dataset[dataset["eligibility_status"] != "eligible"]
        .groupby("eligibility_reason").size().sort_values(ascending=False)
    )
    blocked_rows = "".join(
        f"<tr><td>{_esc(label(BLOCKED_REASON_LABELS, reason))}</td>"
        f"<td>{int(count)}</td></tr>"
        for reason, count in blocked.items()
    )

    window_rows = ""
    if windows is not None and not getattr(windows, "empty", True):
        summary = windows[windows["is_available"] == 1].groupby(
            ["window_name", "is_tradable"]
        ).size()
        window_rows = "".join(
            f"<tr><td>{_esc(label(WINDOW_LABELS, name))}</td>"
            f"<td>{'yes' if tradable else 'no'}</td><td>{int(count)}</td></tr>"
            for (name, tradable), count in summary.items()
        )

    residuals = {
        column: int(eligible[column].notna().sum())
        for column in ("residual_none", "residual_em_lagged",
                       "residual_em_oil_fx_lagged")
        if column in eligible
    }
    residual_rows = "".join(
        f"<tr><td>{_esc(label(CONTROL_LABELS, name))}</td><td>{count}</td></tr>"
        for name, count in residuals.items()
    )

    return f"""
<section class="card" id="windows">
  <h2>Market Windows</h2>
  <p class="sub">
    A return is only a valid target if someone could have earned it.
    <code>signal_date</code> is the first session able to <em>react</em>, never
    the session that published, so the tradable window opens at that session's
    open. Windows anchored to the prior close measure reaction and could not
    have been traded: entering there means holding a position before the news
    was public.
  </p>
  <div class="grid">
    <div class="card">
      <h3>Tradable event rows</h3>
      <div class="big">{len(eligible):,}</div>
      <div class="sub">across {sessions} independent reaction sessions</div>
    </div>
    <div class="card">
      <h3>Blocked rows</h3>
      <div class="big">{int(blocked.sum()):,}</div>
      <div class="sub">reason stated per row, never silently dropped</div>
    </div>
  </div>
  <div class="table-scroll">
  <table class="regime">
    <thead><tr><th>Window</th><th>Tradable</th><th>Available rows</th></tr></thead>
    <tbody>{window_rows}</tbody>
  </table>
  </div>
  <div class="table-scroll">
  <table class="regime">
    <thead><tr><th>Blocked reason</th><th>Rows</th></tr></thead>
    <tbody>{blocked_rows}</tbody>
  </table>
  </div>
  <div class="table-scroll">
  <table class="regime">
    <thead><tr><th>Control set</th><th>Rows with a residual</th></tr></thead>
    <tbody>{residual_rows}</tbody>
  </table>
  </div>
  <p class="legend">
    Only <code>complete</code> and <code>corrected</code> price bars are visible
    to the window builder; a provisional bar is an intraday snapshot. Rolling
    control betas use a 60-session prior window with a 30-observation minimum,
    and are NULL below it rather than estimated from too little history.
  </p>
</section>
"""


def render_validation_section(frozen, *, later_runs=None) -> str:
    """The *frozen* retrospective result, stated as a result not a leaderboard.

    Rendered from the sealed artifact rather than from whatever ran most
    recently. That distinction is the whole safeguard: if a later protocol
    version were allowed to fill this section, "we tried again and it worked"
    would appear here as "it worked". Runs under a different protocol hash are
    counted and named as separate studies, never merged into these numbers.

    What this section deliberately does not contain: a prediction, a ranking of
    which model to "use", or any figure a reader could act on. The null is the
    headline, shown first and largest, because the usual way a null gets
    misread is by being buried under a table of numbers that look like scores.
    """

    if frozen is None or getattr(frozen, "empty", True):
        return (
            '<section class="card" id="validation"><h2>Predictive Validation</h2>'
            '<p class="null">No study has been frozen yet.</p></section>'
        )

    record = frozen.sort_values("frozen_at").iloc[0]
    artifact = json.loads(record["artifact_json"])
    results = artifact.get("specification_results") or []
    sample = artifact.get("sample") or {}

    fitted = [r for r in results if r["status"] == "fitted" and r.get("mae") is not None]
    blocked_count = sum(1 for r in results if r["status"] != "fitted")
    news = sorted((r for r in fitted if r["kind"] == "news"),
                  key=lambda r: r["mae"])
    baselines = sorted((r for r in fitted if r["kind"] == "baseline"),
                       key=lambda r: r["mae"])
    best_news = news[0] if news else None
    best_baseline = baselines[0] if baselines else None

    rows = "".join(f"""
    <tr>
      <td>{_esc(label(FEATURE_SET_LABELS, r["feature_set"]))}</td>
      <td>{_esc(label(MODEL_LABELS, r["model"]))}</td>
      <td>{_esc(label(CONTROL_LABELS, r["target"]))}</td>
      <td>{_esc(r["kind"])}</td>
      <td>{int(r.get("fitted_folds") or 0)}</td>
      <td>{_num(r.get("mae"), 3)}</td>
      <td>{_num(r.get("directional_accuracy"), 3)}</td>
      <td>{_num(r.get("balanced_accuracy"), 3)}</td>
      <td>{(f'[{_num(r.get("hit_rate_ci_lower"), 2)}, '
            f'{_num(r.get("hit_rate_ci_upper"), 2)}]')
           if r.get("hit_rate_ci_lower") is not None else "n/a"}</td>
    </tr>""" for r in sorted(fitted, key=lambda r: r["mae"])[:20])

    def _tile(title: str, row, features, models) -> str:
        if row is None:
            return (f'<div class="tile"><div class="tile-k">{title}</div>'
                    f'<div class="tile-v null">n/a</div></div>')
        return f"""
    <div class="tile">
      <div class="tile-k">{title}</div>
      <div class="tile-v">{_esc(label(features, row["feature_set"]))}</div>
      <div class="tile-note">{_esc(label(models, row["model"]))} &middot;
      average error {_num(row.get("mae"), 2)} &middot; direction
      {_pct(row.get("directional_accuracy"))}</div>
    </div>"""

    interval_note = "indistinguishable from chance"
    if best_news is not None and best_news.get("hit_rate_ci_lower") is not None:
        lower = best_news.get("hit_rate_ci_lower")
        upper = best_news.get("hit_rate_ci_upper")
        spans_chance = lower is not None and upper is not None and lower <= 0.5 <= upper
        interval_note = (
            f"right {_pct(best_news.get('directional_accuracy'))} of the time, "
            f"with a 95% range of {_pct(lower)}&ndash;{_pct(upper)} &mdash; "
            + ("wide enough to include a coin flip"
               if spans_chance else "narrow enough to exclude a coin flip")
        )

    later = ""
    if later_runs is not None and not getattr(later_runs, "empty", True):
        others = later_runs[
            later_runs["protocol_hash"] != record["protocol_hash"]
        ]
        if len(others):
            later = (
                f'<p class="sub">{len(others)} run(s) under a different protocol '
                f'hash exist in the database. They are separate studies and are '
                f'deliberately not merged into the frozen numbers above.</p>'
            )

    return f"""
<section class="card" id="validation">
  <h2>Predictive Validation</h2>

  <div class="verdict-banner">
    <div class="verdict-headline">No validated predictive signal</div>
    <div class="verdict-sub">{_esc(artifact.get("conclusion", ""))}</div>
  </div>

  <div class="tiles">
    <div class="tile">
      <div class="tile-k">Independent sessions tested</div>
      <div class="tile-v">{int(sample.get("distinct_sessions") or 0)}</div>
      <div class="tile-note">a small sample &mdash; many news items share one
      trading session, and one session is one outcome</div>
    </div>
    <div class="tile">
      <div class="tile-k">News approaches that worked</div>
      <div class="tile-v">{int(artifact.get("successes") or 0)}</div>
      <div class="tile-note">of {int(artifact.get("specifications_run") or 0)}
      tested; {blocked_count} more had too little data to try</div>
    </div>
    {_tile("Best news approach", best_news, FEATURE_SET_LABELS, MODEL_LABELS)}
    {_tile("Best comparison baseline", best_baseline, FEATURE_SET_LABELS,
           MODEL_LABELS)}
  </div>

  <p class="interpretation">
    <strong>How to read this.</strong> The best news approach did not beat a
    baseline that uses no news at all by enough to be believed, and its
    direction was {interval_note}. That is not proof that news is
    uninformative &mdash; with this few sessions the study had little chance of
    detecting a modest effect. It does mean nothing here has earned the word
    <em>validated</em>.
  </p>

  <details class="expander">
    <summary>Full results, folds and protocol</summary>
    <p class="sub">
      <strong>Retrospective, on already-collected data.</strong> Folds run
      forward in time and no training fold postdates its test fold, but this
      corpus was collected and inspected before the protocol was written. An
      untouched test needs data that did not exist yet &mdash; see Future
      Validation.
    </p>
    <div class="table-scroll">
    <table class="regime">
      <thead><tr>
        <th>Approach</th><th>Model</th><th>Target</th><th>Kind</th>
        <th>Folds</th><th>Avg error</th><th>Direction</th><th>Balanced</th>
        <th>Hit-rate 95% range</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    <p class="sub">{blocked_count} specification(s) were refused by the
    sample-size gate and never fitted. A refused specification is reported with
    its binding requirement, not hidden.</p>
    {later}
    <p class="sub versions">
      protocol <code>{_esc(str(record["protocol_hash"])[:16])}</code> &middot;
      frozen artifact <code>{_esc(str(record["artifact_hash"])[:16])}</code>
      &middot; commit <code>{_esc(str(record["code_commit"] or "")[:12])}</code>
      &middot; verdict
      <span class="suff verdict-{_esc(record["verdict"])}">{_esc(record["verdict"])}</span>
      &middot; frozen {_esc(str(record["frozen_at"])[:10])}
    </p>
    <p class="legend">
      Ranges are session-cluster bootstraps. Baselines are re-scored on exactly
      the sessions each news approach predicted, so a coverage difference is not
      read as a model difference. <em>n/a</em> means a value could not be
      defensibly computed. This artifact is immutable; a later version
      performing differently does not revise it.
      <strong>No result here is described as significant, and nothing on this
      page is a trading signal, a recommendation or investment advice.</strong>
    </p>
  </details>
</section>
"""


def render_future_validation_section(readiness) -> str:
    """Readiness of the untouched-future sample. Never its performance.

    The outcome side of the boundary is deliberately absent. Watching accuracy
    accumulate and running the evaluation when it looks good is optional
    stopping, and it corrupts the result invisibly -- so the page cannot show
    it, rather than merely declining to.
    """

    if not readiness:
        return (
            '<section class="card" id="future"><h2>Future Validation Status</h2>'
            '<p class="null">The untouched-future contract is not registered '
            'yet.</p></section>'
        )

    state = str(readiness.get("state", "unknown"))
    sessions = int(readiness.get("untouched_sessions") or 0)
    required = int(readiness.get("required_sessions") or 1)
    pct = min(100, round(100 * sessions / required)) if required else 0

    blocking = "".join(
        f"<li>{_esc(reason)}</li>"
        for reason in readiness.get("blocking_reasons") or []
    )
    families = readiness.get("family_coverage") or {}
    family_rows = "".join(
        f"<tr><td>{_esc(FAMILY_LABELS.get(name, name))}</td><td>{int(count)}</td></tr>"
        for name, count in sorted(families.items(), key=lambda kv: -kv[1])
    ) or '<tr><td colspan="2" class="null">no eligible events yet</td></tr>'

    controls = readiness.get("control_availability") or {}
    control_rows = "".join(
        f"<tr><td>{_esc(label(CONTROL_LABELS, name) if name != 'sessions_total' else 'Sessions in total')}</td>"
        f"<td>{int(count)}</td></tr>"
        for name, count in controls.items()
    )

    return f"""
<section class="card" id="future">
  <h2>Future Validation Status</h2>
  <p class="sub">
    <strong>{_esc(readiness.get("definition_version", ""))}</strong> &mdash; a
    test on data that did not exist when the rules were written. Every
    observation with a first reactable session on or after
    <code>{_esc(readiness.get("first_eligible_session"))}</code> belongs to the
    untouched sample. Feature design, model choice, hyperparameters, target,
    thresholds and success criteria were all sealed before that date.
  </p>

  <div class="grid">
    <div class="card">
      <h3>Untouched sessions accumulated</h3>
      <div class="big">{sessions} / {required}</div>
      <div class="bar-outer"><div class="bar-inner" style="width:{pct}%"></div></div>
      <div class="sub">state: <span class="suff">{_esc(state)}</span></div>
    </div>
    <div class="card">
      <h3>Eligible events</h3>
      <div class="big">{int(readiness.get("eligible_events") or 0)}</div>
      <div class="sub">{int(readiness.get("distinct_outcomes") or 0)} distinct
      outcomes &middot; {int(readiness.get("elapsed_days") or 0)} of
      {int(readiness.get("required_days") or 0)} days elapsed</div>
    </div>
  </div>

  <p class="sub"><b>Eligible to run:</b>
  {"yes" if readiness.get("eligible_to_run") else "no"}</p>
  {f"<ul class='sub'>{blocking}</ul>" if blocking else ""}

  <div class="table-scroll">
  <table class="regime">
    <thead><tr><th>Family</th><th>Untouched eligible events</th></tr></thead>
    <tbody>{family_rows}</tbody>
  </table>
  </div>
  <div class="table-scroll">
  <table class="regime">
    <thead><tr><th>Control availability</th><th>Sessions</th></tr></thead>
    <tbody>{control_rows}</tbody>
  </table>
  </div>

  <p class="legend">
    <strong>The outcome side of this boundary is sealed.</strong> No accuracy,
    error or correlation is computed or displayed for the untouched sample
    until the sample-size and horizon requirements are both met. Inspecting
    performance while a sample accumulates and stopping when it looks
    favourable is optional-stopping: it inflates the false-positive rate and
    leaves no trace in the resulting interval. A failed future validation will
    be reported as a failed future validation.
  </p>
</section>
"""
