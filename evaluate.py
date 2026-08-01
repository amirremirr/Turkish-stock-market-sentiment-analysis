"""
evaluate.py - Quality audit for every layer of the pipeline.

Runs independently; reads from the SQLite DB and prints a structured report.
Nothing is modified.

Sections
--------
  L0  System     - pipeline runs, derived-table freshness, schema completeness
  L1  Scraper    - coverage, date parsing, encoding, source diversity, categories
  L2  Model      - score distribution, reported intensity, score-component completeness,
                   spot-check sample
  L3  Aggregate  - session baseline thickness and weighting sensitivity
  L4  Prices     - gaps, outlier returns, staleness
  L5  Signal     - exploratory session-baseline association and sensitivity
                   (display-gated at MINIMUM_OVERLAP_DAYS observations)

Usage
-----
  python evaluate.py               # full report (L0 through L5)
  python evaluate.py --layer 0     # system health only
  python evaluate.py --layer 2     # model audit only
  python evaluate.py --sample 20 --layer 2  # show 20 spot-check headlines
"""

import argparse
import json
import sys
import textwrap
from datetime import date, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

import database as db
from config import (
    BIST_HOLIDAYS,
    DB_PATH,
    MINIMUM_HEADLINES_PER_DAY,
    MINIMUM_OVERLAP_DAYS,
    RELEVANCE_BLOCKLIST,
    RELEVANCE_STRONG,
)

# -- Formatting helpers --------------------------------------------------------

W = 62  # report width

SIGNAL_VARIANTS = (
    "simple_mean",
    "relevance_weighted",
    "intensity_relevance_weighted",
    "full_weighted",
)


def _signal_variants_frame(db_path: str) -> pd.DataFrame:
    """Load the session-assigned signal table through the database adapter.

    The adapter currently returns a DataFrame; accepting row mappings as well
    keeps the read-only helper simple to isolate in tests.  Historical
    ``signal_date`` and ``session_date`` spellings are normalized to ``date``.
    """
    loaded = db.get_signal_variants(db_path=db_path)
    if isinstance(loaded, pd.DataFrame):
        frame = loaded.copy()
    else:
        frame = pd.DataFrame([dict(row) for row in loaded])
    if frame.empty:
        return pd.DataFrame(columns=("date", *SIGNAL_VARIANTS, "headline_count"))
    date_column = next(
        (name for name in ("date", "session_date", "signal_date") if name in frame.columns),
        None,
    )
    if date_column is None:
        raise ValueError("daily_signal_variants rows do not contain a session date")
    if date_column != "date":
        frame = frame.rename(columns={date_column: "date"})

    required = {"date", "simple_mean", "headline_count"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "daily_signal_variants rows are missing required columns: "
            + ", ".join(missing)
        )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)

def _hdr(title: str) -> None:
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")

def _sub(title: str) -> None:
    print(f"\n  -- {title} --")

def _row(label: str, value, note: str = "", warn: bool = False) -> None:
    tag = " [!]" if warn else ""
    print(f"  {label:<36} {str(value):<14} {note}{tag}")

def _ok(msg: str)   -> None: print(f"  [OK]  {msg}")
def _warn(msg: str) -> None: print(f"  [!!]  {msg}")
def _info(msg: str) -> None: print(f"  [ ]   {msg}")

TURKISH_CHARS = set("sçgioüSÇGIOÜ")   # ASCII-folded forms that signal Turkish content


# -----------------------------------------------------------------------------
# L0 - System health (pipeline runs + derived-table freshness)
# -----------------------------------------------------------------------------

def audit_system(db_path: str) -> dict:
    _hdr("L0 - SYSTEM HEALTH")

    with db._conn(db_path) as con:
        run_count = con.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        last_run  = con.execute(
            """SELECT run_id, status, started_at, finished_at, error_msg,
                      headlines_scraped, headlines_scored, model_name
               FROM pipeline_runs ORDER BY run_id DESC LIMIT 1"""
        ).fetchone()

        # The calendar aggregate is retained only as a legacy descriptive
        # freshness reference.  Session-assigned variants are loaded below and
        # are the primary aggregate used by every market-linked audit.
        hl_max   = con.execute(
            "SELECT MAX(published_at) FROM headlines WHERE sentiment_score IS NOT NULL"
        ).fetchone()[0]
        ds_max   = con.execute("SELECT MAX(date) FROM daily_sentiment").fetchone()[0]

        # Schema completeness: scored rows missing backend-specific components.
        missing_probs = con.execute(
            """SELECT COUNT(*) FROM headlines
               WHERE sentiment_score IS NOT NULL AND p_positive IS NULL"""
        ).fetchone()[0]
        total_scored  = con.execute(
            "SELECT COUNT(*) FROM headlines WHERE sentiment_score IS NOT NULL"
        ).fetchone()[0]

        # Category backfill status
        cat_null = con.execute(
            "SELECT COUNT(*) FROM headlines WHERE category IS NULL"
        ).fetchone()[0]

    session_signals = _signal_variants_frame(db_path)
    session_max = (
        session_signals["date"].max().date().isoformat()
        if not session_signals.empty else None
    )

    # -- Pipeline run log -------------------------------------------------------
    _sub("Pipeline run log")
    _row("Total pipeline runs logged", run_count,
         warn=(run_count == 0))
    if run_count == 0:
        _warn("No runs logged - use 'python main.py run' instead of individual steps")
    elif last_run:
        status = last_run["status"]
        canonical_status = {
            "ok": "success",
            "recovered": "success",
            "error": "failed",
            "crashed": "failed",
        }.get(status, status)
        _row(
            "Last run status", canonical_status,
            warn=(canonical_status != "success"),
        )
        _row("Last run started",  (last_run["started_at"] or "")[:19])
        _row("Last run finished", (last_run["finished_at"] or "-")[:19])
        _row("Last run scraped",  last_run["headlines_scraped"] or 0)
        _row("Last run scored",   last_run["headlines_scored"]  or 0)
        if last_run["model_name"]:
            _row("Model used",    last_run["model_name"])
        if last_run["error_msg"]:
            _warn(f"Last run error: {last_run['error_msg']}")
        elif canonical_status == "success":
            _ok("Last run completed without errors")
        elif canonical_status == "degraded":
            _warn("Last run completed with degraded components; inspect its warnings")
        elif canonical_status == "running":
            _warn("Last run has not reached a terminal state")

    # -- Derived-table freshness -----------------------------------------------
    _sub("Derived-table freshness (session-assigned primary)")
    if hl_max and session_max:
        headline_date = pd.to_datetime(hl_max).date()
        aggregate_date = pd.to_datetime(session_max).date()
        if headline_date <= aggregate_date:
            _ok(f"daily_signal_variants is current  (latest session: {session_max})")
        else:
            _warn(f"daily_signal_variants may be STALE  "
                  f"(scored headline date: {headline_date}  >  "
                  f"latest session: {session_max})")
            _info("Run 'python main.py aggregate' to refresh")
    elif hl_max and not session_max:
        _warn("Scored headlines exist but daily_signal_variants is empty")
        _info("Run 'python main.py aggregate'")
    else:
        _info("No scored headlines yet")

    if ds_max:
        _info(f"Legacy calendar daily_sentiment latest date: {ds_max} (descriptive only)")

    # -- Schema completeness ---------------------------------------------------
    _sub("Schema completeness (score components)")
    if total_scored == 0:
        _info("No scored headlines yet")
    elif missing_probs == 0:
        _ok(f"All {total_scored} scored headlines have score-component fields")
    else:
        pct = missing_probs / total_scored * 100
        _warn(f"{missing_probs} / {total_scored} ({pct:.0f}%) scored headlines "
              f"are missing p_positive/p_neutral/p_negative")
        _info("These were scored before score-component storage was added.")
        _info("Re-score: DELETE sentiment cols, then 'python main.py score'")

    if cat_null > 0:
        _warn(f"{cat_null} headline(s) missing category "
              f"- run 'python main.py aggregate' to backfill")
    else:
        _ok("All headlines have a category assigned")

    return {"run_count": run_count, "missing_probs": missing_probs}


# -----------------------------------------------------------------------------
# L1 - Scraper quality
# -----------------------------------------------------------------------------

def audit_scraper(db_path: str) -> dict:
    _hdr("L1 - SCRAPER QUALITY")

    with db._conn(db_path) as con:
        df = pd.read_sql_query(
            "SELECT id, source, title, url, published_at, scraped_at, category FROM headlines",
            con,
        )

    if df.empty:
        _warn("No headlines in DB - run 'python main.py scrape' first.")
        return {}

    total = len(df)
    _row("Total headlines", total)

    # -- Date coverage ---------------------------------------------------------
    _sub("Date coverage")
    has_date = df["published_at"].notna().sum()
    no_date  = total - has_date
    _row("Headlines with parsed date", has_date)
    if no_date:
        _warn(f"{no_date} headlines have no date - date parser missing a format")
    else:
        _ok("All headlines have dates")

    df_dated = df.dropna(subset=["published_at"])
    if not df_dated.empty:
        df_dated = df_dated.copy()
        df_dated["published_at"] = pd.to_datetime(df_dated["published_at"])
        oldest = df_dated["published_at"].min().date()
        newest = df_dated["published_at"].max().date()
        span   = (newest - oldest).days + 1
        unique_days = df_dated["published_at"].dt.date.nunique()
        _row("Date range", f"{oldest} .. {newest}")
        _row("Calendar days spanned", span)
        _row("Days with at least 1 headline", unique_days)

        per_day = df_dated.groupby(df_dated["published_at"].dt.date).size()
        _row("Headlines/day  min / mean / max",
             f"{per_day.min()} / {per_day.mean():.1f} / {per_day.max()}")
        thin = (per_day < MINIMUM_HEADLINES_PER_DAY).sum()
        if thin:
            _warn(f"{thin} day(s) have < {MINIMUM_HEADLINES_PER_DAY} headlines "
                  f"- signal will be noisy on those days")

    # -- Source diversity ------------------------------------------------------
    _sub("Source breakdown")
    for src, cnt in df["source"].value_counts().items():
        _row(f"  {src}", cnt, f"({cnt/total*100:.0f}%)")

    # -- URL deduplication + dedup audit ---------------------------------------
    _sub("Deduplication")
    has_url  = df["url"].notna().sum()
    no_url   = total - has_url
    dup_urls = df["url"].dropna().duplicated().sum()
    _row("Headlines with URL", has_url)
    _row("Headlines without URL", no_url,
         note="(title-based dedup used for these)" if no_url else "")
    if dup_urls:
        _warn(f"{dup_urls} duplicate URLs exist in DB (INSERT OR IGNORE should prevent this)")
    else:
        _ok("No duplicate URLs in DB")

    # Title-hash audit: catch near-duplicate titles that slipped through dedup.
    # Identical normalised title[:80] = same story; multiple URLs = multi-source
    # duplicate; NULL URL = scraper returned no link.
    _sub("Title-dedup audit (title[:80] hash collisions — false merges?)")
    from scraper import _normalise as _sc_norm
    df["_title_hash"] = df["title"].apply(lambda t: _sc_norm(str(t))[:80])
    hash_groups = df.groupby("_title_hash")
    multi = hash_groups.filter(lambda g: len(g) > 1)
    if multi.empty:
        _ok("No title[:80] hash collisions — dedup working correctly")
    else:
        collision_groups = multi.groupby("_title_hash")
        n_groups  = len(collision_groups)
        n_rows    = len(multi)
        _warn(f"{n_groups} title-hash collision group(s) ({n_rows} rows total) "
              f"— verify these are genuine duplicates, not false merges:")
        for title_hash, grp in list(collision_groups)[:5]:
            distinct_urls = grp["url"].dropna().nunique()
            _info(f"  n={len(grp)}  URLs={distinct_urls}  "
                  f"hash={title_hash[:50]!r}")
        if n_groups > 5:
            _info(f"  ... and {n_groups - 5} more groups")

    # -- published_hour coverage (time-of-day weighting effectiveness) ---------
    _sub("published_hour coverage (time-of-day weighting)")
    with db._conn(db_path) as con:
        hour_df = pd.read_sql_query(
            "SELECT published_hour FROM headlines", con
        )
    has_hour  = hour_df["published_hour"].notna().sum()
    null_hour = len(hour_df) - has_hour
    hour_pct  = has_hour / len(hour_df) * 100 if len(hour_df) else 0.0
    _row("Headlines with published_hour", has_hour, f"({hour_pct:.0f}%)",
         warn=(hour_pct < 50))
    _row("Headlines without hour (NULL)", null_hour,
         note="(treated as market-hours weight 1.0x)")
    if null_hour > 0 and hour_pct < 50:
        _warn("More than half of headlines have no hour — time weighting has limited effect")
        _info("NULL hour is treated as 1.0x (neutral market-hours weight)")
    elif hour_pct >= 80:
        _ok(f"published_hour populated for {hour_pct:.0f}% of headlines — time weighting is active")
    if has_hour > 0:
        hour_vals = hour_df["published_hour"].dropna().astype(int)
        pre_mkt  = (hour_vals < 10).sum()
        mkt      = ((hour_vals >= 10) & (hour_vals <= 18)).sum()
        post_mkt = (hour_vals > 18).sum()
        _row("Pre-market  (h<10,  weight 1.5×)", pre_mkt,  f"({pre_mkt/has_hour*100:.0f}%)")
        _row("Market hrs  (10≤h≤18, weight 1.0×)", mkt,   f"({mkt/has_hour*100:.0f}%)")
        _row("Post-market (h>18,  weight 0.8×)", post_mkt, f"({post_mkt/has_hour*100:.0f}%)")

    # -- Turkish character encoding --------------------------------------------
    _sub("Turkish character encoding")
    def has_turkish(t: str) -> bool:
        return any(c in "şçğıöüŞÇĞİÖÜ" for c in str(t))

    tr_ok  = df["title"].apply(has_turkish).sum()
    tr_pct = tr_ok / total * 100
    _row("Headlines with Turkish chars", tr_ok, f"({tr_pct:.0f}%)")
    if tr_pct < 30:
        _warn("Very few Turkish characters detected - possible encoding corruption")
    else:
        _ok("Turkish character encoding looks healthy")

    # -- Category breakdown ----------------------------------------------------
    _sub("Category breakdown (keyword classifier)")
    cat_counts = df["category"].fillna("NULL").value_counts()
    for cat, cnt in cat_counts.items():
        pct = cnt / total * 100
        _row(f"  {cat}", cnt, f"({pct:.0f}%)",
             warn=(cat == "other" and pct > 20))

    other_cnt = cat_counts.get("other", 0)
    other_pct = other_cnt / total * 100
    if other_pct > 20:
        _warn(f"{other_pct:.0f}% classified as 'other' - taxonomy likely incomplete")
        _info("Inspect: SELECT title FROM headlines WHERE category='other'")
    else:
        _ok(f"Category coverage: {100 - other_pct:.0f}% classified into named buckets")

    # -- Blocklist-override edge cases -----------------------------------------
    # A headline that contains a blocklist term AND a strong Turkey marker is the
    # highest-risk category: it was kept only because the strong marker overrode
    # the blocklist.  Some of these will be genuinely relevant (e.g. "Bitcoin
    # rallied as Türkiye tightened crypto rules") and some will be false-positives
    # (e.g. a Nasdaq story that happens to mention Turkey in passing).
    # Displaying them explicitly lets you catch systematic misclassifications
    # without reading all 40+ headlines manually.
    _sub("Blocklist-override headlines (strong marker beat blocklist - verify these)")

    from scraper import _normalise as _sc_norm

    edge_cases = []
    for _, r in df.iterrows():
        t = _sc_norm(r["title"])
        bl_match = next((b for b in RELEVANCE_BLOCKLIST if b in t), None)
        st_match = next((s for s in RELEVANCE_STRONG   if s in t), None)
        if bl_match and st_match:
            edge_cases.append({
                "id":            int(r["id"]),
                "title":         r["title"],
                "category":      r.get("category") or "?",
                "blocklist_hit": bl_match,
                "strong_hit":    st_match,
            })

    if not edge_cases:
        _ok("No blocklist-override headlines - all passes were clean keyword matches")
    else:
        _warn(f"{len(edge_cases)} headline(s) passed because a strong marker "
              f"overrode a blocklist term:")
        _info(
            "Preserve reviewed false positives with "
            "database.exclude_headline(ID, 'manual_false_positive', "
            "rule='blocklist_override_review'), then rerun aggregation."
        )
        for ec in edge_cases[:12]:   # cap at 12 to keep output readable
            title_w = textwrap.shorten(ec["title"], width=52, placeholder="...")
            print(f"    [id={ec['id']:<5} {ec['category']:18}] "
                  f"bl={ec['blocklist_hit']!r:12}  st={ec['strong_hit']!r:10}  "
                  f"{title_w}")
        if len(edge_cases) > 12:
            _info(f"  ... and {len(edge_cases) - 12} more (query: "
                  "python evaluate.py --layer 1)")

    # -- Source quality dashboard ----------------------------------------------
    _sub("Source quality dashboard")
    with db._conn(db_path) as con:
        full_df = pd.read_sql_query(
            """SELECT source, url, published_at, sentiment_score, category
               FROM headlines""",
            con,
        )

    # Compute per-day totals to identify thin days
    full_df["_date"] = pd.to_datetime(full_df["published_at"]).dt.date
    day_totals = full_df.groupby("_date")["source"].count().rename("day_total")
    full_df    = full_df.join(day_totals, on="_date")
    full_df["_thin"] = full_df["day_total"] < MINIMUM_HEADLINES_PER_DAY

    print()
    hdr_fmt = f"  {'Source':<22}  {'N':>4}  {'NullURL':>7}  {'AvgConf':>7}  {'Thin%':>5}  {'TopCategory'}"
    print(hdr_fmt)
    print(f"  {'-'*22}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*18}")

    for src, g in full_df.groupby("source"):
        n_src      = len(g)
        null_url   = g["url"].isna().mean() * 100
        scored     = g["sentiment_score"].dropna()
        avg_conf   = float(scored.abs().mean()) if len(scored) > 0 else float("nan")
        thin_pct   = g["_thin"].mean() * 100
        top_cat    = g["category"].dropna().mode()
        top_cat    = top_cat.iloc[0] if not top_cat.empty else "—"
        conf_str   = f"{avg_conf:.2f}" if not np.isnan(avg_conf) else "  —"
        warn_thin  = thin_pct > 30
        thin_tag   = " [!]" if warn_thin else ""
        print(f"  {src:<22}  {n_src:>4}  {null_url:>6.0f}%  {conf_str:>7}  "
              f"{thin_pct:>4.0f}%{thin_tag}  {top_cat}")

    _info("AvgConf = mean |sentiment_score|; higher = model is more decisive on this source")
    _info("Thin%   = % of this source's headlines on days with < "
          f"{MINIMUM_HEADLINES_PER_DAY} total headlines")

    return {"total": total, "has_date_pct": has_date / total * 100,
            "other_pct": other_pct}


# -----------------------------------------------------------------------------
# L2 - Sentiment model quality
# -----------------------------------------------------------------------------

def audit_model(db_path: str, spot_n: int = 10) -> dict:
    _hdr("L2 - SENTIMENT MODEL QUALITY")

    with db._conn(db_path) as con:
        df = pd.read_sql_query(
            """SELECT title, sentiment_score, sentiment_label,
                      p_positive, p_neutral, p_negative, model_name
               FROM headlines
               WHERE sentiment_score IS NOT NULL""",
            con,
        )

    if df.empty:
        _warn("No scored headlines - run 'python main.py score' first.")
        return {}

    scores = df["sentiment_score"].values
    labels = df["sentiment_label"].values

    # -- Score-component completeness ------------------------------------------
    _sub("Score-component completeness")
    has_probs = df["p_positive"].notna().sum()
    pct_probs = has_probs / len(df) * 100
    _row("Headlines with score components", has_probs, f"({pct_probs:.0f}%)",
         warn=(pct_probs < 95))
    if pct_probs < 95:
        _warn("Some headlines lack p_positive/p_neutral/p_negative - scored before schema upgrade")
    else:
        _ok("Score-component fields are fully populated")
    _info("LLM p_* fields are synthetic compatibility values, not calibrated probabilities.")

    # Model version
    models_used = df["model_name"].dropna().unique()
    if len(models_used) == 1:
        _row("Model", models_used[0])
    elif len(models_used) > 1:
        _warn(f"Multiple model versions in DB: {list(models_used)}")
        _info("Consider rescoring all headlines with one model for consistency")

    # -- Score distribution ----------------------------------------------------
    _sub("Stored sentiment-score distribution  (range [-1,+1])")
    _row("Count scored",    len(scores))
    _row("Mean",            f"{scores.mean():+.4f}")
    _row("Median",          f"{np.median(scores):+.4f}")
    _row("Std dev",         f"{scores.std():.4f}")
    _row("Min / Max",       f"{scores.min():+.4f} / {scores.max():+.4f}")

    pcts = np.percentile(scores, [10, 25, 50, 75, 90])
    _row("10th / 25th / 50th / 75th / 90th",
         "  ".join(f"{p:+.2f}" for p in pcts))

    # -- Model-reported sentiment intensity -----------------------------------
    _sub("Model-reported sentiment intensity")
    near_zero = (np.abs(scores) < 0.05).sum()
    decisive  = (np.abs(scores) > 0.40).sum()
    _row("Scores near zero  |s| < 0.05", near_zero,
         warn=(near_zero / len(scores) > 0.5),
         note="(weak directional intensity)")
    _row("High-intensity scores |s| > 0.40", decisive,
         f"({decisive/len(scores)*100:.0f}%)")

    if scores.std() < 0.05:
        _warn("Extremely low variance - all headlines scoring the same; check model output")
    elif scores.std() < 0.15:
        _warn("Low variance - model may not be discriminating well on financial Turkish text")
    else:
        _ok("Score variance looks healthy")

    # -- Label breakdown -------------------------------------------------------
    _sub("Label breakdown")
    for lbl in ["positive", "neutral", "negative"]:
        cnt = (labels == lbl).sum()
        _row(f"  {lbl}", cnt, f"({cnt/len(labels)*100:.0f}%)")

    if (labels == "neutral").mean() > 0.70:
        _warn("Over 70% neutral - model may not be capturing financial sentiment well")

    # -- Spot-check sample -----------------------------------------------------
    _sub(f"Spot-check: {spot_n} sampled headlines (review manually)")
    sample = df.sample(min(spot_n, len(df)), random_state=42)
    print()
    for _, row in sample.iterrows():
        bar_len   = int(abs(row["sentiment_score"]) * 20)
        direction = "+" if row["sentiment_score"] >= 0 else "-"
        bar       = direction * bar_len
        score_str = f"{row['sentiment_score']:+.3f}"
        label_str = f"[{row['sentiment_label'][:3].upper()}]"
        title_w   = textwrap.shorten(row["title"], width=55, placeholder="...")
        print(f"  {score_str} {label_str:6}  {'|' + bar:<22}  {title_w}")

    print()
    _info("Manually check: does the label match your intuition for each headline?")
    _info("Key failure modes: negation ('dusmedi'), sarcasm, mixed-signal headlines")
    _info("Track categorical agreement against the written human-label rubric.")

    return {"mean_score": float(scores.mean()), "std_score": float(scores.std())}


# -----------------------------------------------------------------------------
# L3 - Aggregation quality
# -----------------------------------------------------------------------------

def audit_aggregate(db_path: str) -> dict:
    _hdr("L3 - SESSION-ASSIGNED SIGNAL AGGREGATE QUALITY")

    df = _signal_variants_frame(db_path)

    if df.empty:
        _warn("No session-assigned signal variants - run 'python main.py aggregate' first.")
        return {}

    # -- Signal thickness ------------------------------------------------------
    _sub(f"Signal thickness (headlines per session, gate: {MINIMUM_HEADLINES_PER_DAY})")
    counts = df["headline_count"]
    _row("Sessions with signal", len(df))
    _row("Articles/session min/mean/max",
         f"{counts.min()} / {counts.mean():.1f} / {counts.max()}")

    thin_days = df[df["headline_count"] < MINIMUM_HEADLINES_PER_DAY]
    if not thin_days.empty:
        _warn(f"{len(thin_days)} session(s) with < {MINIMUM_HEADLINES_PER_DAY} headlines "
              f"(thin observations; excluded from exploratory statistics):")
        for _, r in thin_days.iterrows():
            print(f"    {r['date'].date()}  n={int(r['headline_count'])}  "
                  f"simple_mean={r['simple_mean']:+.3f}")
    else:
        _ok(f"All sessions meet the minimum of {MINIMUM_HEADLINES_PER_DAY} headlines")

    # -- Primary baseline distribution ----------------------------------------
    baseline = df["simple_mean"].dropna()
    _sub("Primary baseline distribution (simple unweighted mean)")
    _row("Simple mean  mean", f"{baseline.mean():+.4f}")
    _row("Simple mean  std",  f"{baseline.std():.4f}")
    _row("Simple mean  min",  f"{baseline.min():+.4f}")
    _row("Simple mean  max",  f"{baseline.max():+.4f}")

    if baseline.std() < 0.03:
        _warn("Session baseline has very low variance - exploratory tests may be unstable")
    else:
        _ok("Session baseline has non-trivial variation")

    # -- Weighting sensitivity -------------------------------------------------
    _sub("Weighting sensitivity (alternatives are diagnostics, not selected models)")
    for column in SIGNAL_VARIANTS[1:]:
        if column not in df.columns:
            _warn(f"Missing sensitivity variant: {column}")
            continue
        paired = df[["simple_mean", column]].dropna()
        if paired.empty:
            _info(f"{column}: no finite paired sessions")
            continue
        sign_agreement = (
            np.sign(paired["simple_mean"]) == np.sign(paired[column])
        ).mean()
        mean_abs_delta = (paired[column] - paired["simple_mean"]).abs().mean()
        _row(
            column,
            f"mean={paired[column].mean():+.4f}",
            f"sign agreement={sign_agreement:.1%}; mean |delta|={mean_abs_delta:.4f}; "
            f"n={len(paired)}",
        )

    # -- Label-share consistency ----------------------------------------------
    if {"positive_share", "negative_share"}.issubset(df.columns):
        _sub("Simple mean vs observed positive/negative shares")
        directional = df.dropna(
            subset=["simple_mean", "positive_share", "negative_share"]
        ).copy()
        directional["share_direction"] = (
            directional["positive_share"] - directional["negative_share"]
        )
        match_pct = (
            np.sign(directional["simple_mean"])
            == np.sign(directional["share_direction"])
        ).mean() * 100 if not directional.empty else float("nan")
        _row("Sessions with matching direction", f"{match_pct:.0f}%",
             warn=(not np.isnan(match_pct) and match_pct < 55))
        _info("This is a descriptive consistency check, not a predictive result.")

    # -- Per-category signal ---------------------------------------------------
    _sub("Legacy calendar-date category aggregate (descriptive only)")
    with db._conn(db_path) as con:
        cat_df = pd.read_sql_query(
            """SELECT category,
                      AVG(avg_score)      AS mean_score,
                      SUM(headline_count) AS total_headlines,
                      COUNT(*)            AS days
               FROM category_daily_sentiment
               GROUP BY category
               ORDER BY mean_score DESC""",
            con,
        )

    if cat_df.empty:
        _info("No category sentiment rows yet - run 'python main.py aggregate'")
    else:
        _row(f"  {'Category':<22} {'Mean':>8}  {'Days':>5}  {'Headlines':>10}", "")
        for _, row in cat_df.iterrows():
            bar_len   = int(abs(row["mean_score"]) * 15)
            direction = "+" if row["mean_score"] >= 0 else "-"
            bar       = direction * bar_len
            print(f"    {row['category']:<22} {row['mean_score']:+.4f}  "
                  f"{int(row['days']):>5}  {int(row['total_headlines']):>10}  "
                  f"{'|' + bar}")

    return {"days": len(df), "mean_articles_per_day": float(counts.mean())}


# -----------------------------------------------------------------------------
# L4 - Price data quality
# -----------------------------------------------------------------------------

def audit_prices(db_path: str) -> dict:
    _hdr("L4 - PRICE DATA QUALITY")

    with db._conn(db_path) as con:
        df = pd.read_sql_query(
            "SELECT * FROM bist100_prices ORDER BY date", con
        )

    if df.empty:
        _warn("No price data - run 'python main.py prices' first.")
        return {}

    df["date"] = pd.to_datetime(df["date"])

    # -- Coverage --------------------------------------------------------------
    _sub("Coverage")
    _row("Trading days stored", len(df))
    _row("Date range", f"{df['date'].min().date()} .. {df['date'].max().date()}")
    days_stale = (date.today() - df["date"].max().date()).days
    _row("Days since last price", days_stale, warn=(days_stale > 3))
    if days_stale > 3:
        _warn(f"Price data is {days_stale} days stale - run 'python main.py prices'")
    else:
        _ok("Price data is current")

    # -- Missing trading days --------------------------------------------------
    _sub("Gap detection (missing trading days)")
    full_range   = pd.bdate_range(df["date"].min(), df["date"].max(), freq="B")
    stored_dates = set(df["date"].dt.date)
    known_holidays = {date.fromisoformat(d) for d in BIST_HOLIDAYS}

    missing = [d.date() for d in full_range if d.date() not in stored_dates]
    known_gaps    = [d for d in missing if d in known_holidays]
    unknown_gaps  = [d for d in missing if d not in known_holidays]

    if not missing:
        _ok("No missing weekdays detected")
    else:
        if known_gaps:
            _ok(f"{len(known_gaps)} missing weekday(s) are known BIST holidays:")
            for d in known_gaps:
                # Look up holiday name from config list
                print(f"    {d}  (official holiday)")
        if unknown_gaps:
            _warn(f"{len(unknown_gaps)} missing weekday(s) are NOT in the holiday calendar "
                  f"— possible data gap or yfinance outage:")
            for d in unknown_gaps:
                print(f"    {d}  [!] unexpected — verify manually")

    # -- Return distribution ---------------------------------------------------
    _sub("Daily return distribution (sanity check)")
    returns = df["daily_return"].dropna()
    _row("Mean daily return %",   f"{returns.mean():+.4f}")
    _row("Std daily return %",    f"{returns.std():.4f}")
    _row("Min / Max return %",    f"{returns.min():+.2f} / {returns.max():+.2f}")

    extreme = (returns.abs() > 5).sum()
    if extreme:
        _warn(f"{extreme} day(s) with |return| > 5% - verify these aren't data errors:")
        for _, row in df[df["daily_return"].abs() > 5].iterrows():
            print(f"    {row['date'].date()}  return={row['daily_return']:+.2f}%  "
                  f"close={row['close']:,.0f}")
    else:
        _ok("No extreme return outliers")

    # -- Price level sanity ----------------------------------------------------
    _sub("Price level sanity")
    _row("Latest close", f"{df['close'].iloc[-1]:,.0f}")
    _row("52-wk high",   f"{df['close'].max():,.0f}")
    _row("52-wk low",    f"{df['close'].min():,.0f}")

    # -- Cross-check: re-download latest 5 days from yfinance -----------------
    # This catches silent data corruption, stale rows, and ticker mis-config.
    # It is NOT a second independent source — it re-queries the same provider.
    # For a true second source, see BIST_HOLIDAYS note in config.py.
    _sub("Self-consistency check (re-download last 5 days from yfinance)")
    try:
        import yfinance as yf
        from config import BIST100_TICKER
        raw = yf.download(BIST100_TICKER, period="10d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, __import__("pandas").MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=["Close"])
        raw.index = __import__("pandas").to_datetime(raw.index)

        mismatches = 0
        checked = 0
        for idx, row in raw.tail(5).iterrows():
            day_str = idx.strftime("%Y-%m-%d")
            stored = df[df["date"] == __import__("pandas").to_datetime(day_str)]
            if stored.empty:
                continue
            stored_close = float(stored.iloc[0]["close"])
            live_close   = float(row["Close"])
            diff_pct     = abs(stored_close - live_close) / live_close * 100
            checked += 1
            ok = diff_pct < 0.5   # allow <0.5% for rounding/adjustment differences
            status = "[OK]" if ok else "[!!]"
            print(f"    {day_str}  stored={stored_close:>10,.2f}  "
                  f"live={live_close:>10,.2f}  diff={diff_pct:.3f}%  {status}")
            if not ok:
                mismatches += 1

        if checked == 0:
            _info("No overlapping dates to cross-check (holiday period?)")
        elif mismatches == 0:
            _ok(f"All {checked} checked days match yfinance live data (diff <0.5%)")
        else:
            _warn(f"{mismatches}/{checked} days differ by >0.5% from live yfinance data")
            _info("Possible cause: adjusted-price recalculation after corporate action")
            _info("Re-run 'python main.py prices' to refresh")

    except Exception as exc:
        _info(f"Cross-check skipped: {exc}")

    # -- USD/TRY FX rates (Alpha Vantage) -------------------------------------
    _sub("USD/TRY FX rates (Alpha Vantage — independent source)")
    fx = db.get_fx_rates(db_path=db_path)
    if fx.empty:
        _warn("No USD/TRY data yet — run 'python main.py fx-rates' to fetch")
        _info("Alpha Vantage key is configured in config.py (ALPHA_VANTAGE_KEY)")
    else:
        fx["date"] = pd.to_datetime(fx["date"])
        _row("USD/TRY days stored", len(fx))
        _row("Date range", f"{fx['date'].min().date()} .. {fx['date'].max().date()}")
        fx_stale = (date.today() - fx["date"].max().date()).days
        _row("Days since last FX rate", fx_stale, warn=(fx_stale > 3))
        _row("Latest USD/TRY close", f"{fx['close'].iloc[-1]:.4f}")
        _row("Range (close)", f"{fx['close'].min():.4f} .. {fx['close'].max():.4f}")

        # Quick correlation with BIST 100 returns (same-day)
        if not df.empty:
            merged_fx = pd.merge(
                df[["date", "daily_return"]],
                fx[["date", "close"]].rename(columns={"close": "usdtry"}),
                on="date", how="inner",
            ).dropna()
            merged_fx["usdtry_chg"] = merged_fx["usdtry"].pct_change() * 100

            valid = merged_fx.dropna()
            if len(valid) >= 5:
                from scipy.stats import pearsonr
                r, p_val = pearsonr(valid["usdtry_chg"], valid["daily_return"])
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
                _row("Pearson r: USD/TRY daily chg% vs BIST return%",
                     f"{r:+.4f}", f"p={p_val:.3f} {sig}  (n={len(valid)})")
                if r < -0.2:
                    _ok("Negative correlation: TRY weakening → BIST falls (expected)")
                elif r > 0.2:
                    _info("Positive correlation: unusual — worth investigating")
                else:
                    _info("Weak correlation at current data volume")
            else:
                _info(f"Only {len(valid)} overlapping days — correlation needs more data")

    return {"days": len(df), "missing_weekdays": len(missing),
            "unexpected_gaps": len(unknown_gaps)}


# -----------------------------------------------------------------------------
# L5 - Exploratory signal/return association
# -----------------------------------------------------------------------------

def audit_signal(db_path: str) -> dict:
    _hdr("L5 - EXPLORATORY SESSION SIGNAL/RETURN ASSOCIATION")

    with db._conn(db_path) as con:
        prices = pd.read_sql_query(
            "SELECT date, close, daily_return FROM bist100_prices", con
        )
    signals = _signal_variants_frame(db_path)

    if prices.empty or signals.empty:
        _warn("Need both prices and session-assigned signal variants. "
              "Run 'python main.py run' first.")
        return {}

    prices["date"] = pd.to_datetime(prices["date"])

    # next_return MUST be computed on the consecutive exchange-session price series
    # BEFORE merging/filtering. shift(-1) after an inner join pairs a day with
    # the next SURVIVING ROW, which can be many trading days later when the
    # overlap has gaps — silently corrupting every D→D+1-session statistic below.
    prices = prices.sort_values("date")
    prices["daily_return"] = prices["close"].pct_change(fill_method=None) * 100.0
    prices["next_return"] = prices["daily_return"].shift(-1)

    # Apply the coverage gate only after subsequent-session returns have been
    # formed on the complete ordered exchange-session price series above.
    reliable_signals = signals[
        signals["headline_count"] >= MINIMUM_HEADLINES_PER_DAY
    ]
    excluded = len(signals) - len(reliable_signals)
    if excluded:
        _info(f"Excluding {excluded} session(s) with < {MINIMUM_HEADLINES_PER_DAY} "
              "headlines from exploratory signal statistics")

    merged = pd.merge(
        prices, reliable_signals, on="date", how="inner"
    ).sort_values("date")
    merged["same_session_return"] = merged["daily_return"]

    # The primary predictive sample requires the subsequent-session target.
    # Contemporaneous returns are a separate diagnostic and must not shrink it
    # merely because the price history starts on a signal date.
    valid = merged.dropna(subset=["simple_mean", "next_return"]).copy()
    same_valid = merged.dropna(
        subset=["simple_mean", "same_session_return"]
    ).copy()
    n     = len(valid)

    _sub(f"Eligible session observations (simple_mean + subsequent return): {n}")
    _row("Gate threshold (MINIMUM_OVERLAP_DAYS)", MINIMUM_OVERLAP_DAYS)

    if n < MINIMUM_OVERLAP_DAYS:
        _warn(f"Only {n} eligible observations - the display gate is "
              f"{MINIMUM_OVERLAP_DAYS}.")
        _info("Crossing this gate permits exploratory reporting; it is not validation.")
        _info("Keep running 'python main.py run'; the session sample will fill in.")
        _info(f"Observations short of gate: "
              f"{max(0, MINIMUM_OVERLAP_DAYS - n)} exchange sessions")
        return {"overlapping_days": n}

    x      = valid["simple_mean"].values
    y_next = valid["next_return"].values

    # -- Pearson correlation ---------------------------------------------------
    _sub("Primary baseline: simple_mean (unweighted)")

    r_next, p_next = stats.pearsonr(x, y_next)
    if (
        len(same_valid) >= 2
        and same_valid["simple_mean"].nunique() > 1
        and same_valid["same_session_return"].nunique() > 1
    ):
        r_same, p_same = stats.pearsonr(
            same_valid["simple_mean"], same_valid["same_session_return"]
        )
    else:
        r_same, p_same = float("nan"), float("nan")

    def _sig(p: float) -> str:
        if p < 0.01: return "** (p<0.01)"
        if p < 0.05: return "*  (p<0.05)"
        if p < 0.10: return ".  (p<0.10)"
        return "   (not significant)"

    _row("Pearson r: simple_mean(D) vs return(D+1)", f"{r_next:+.4f}", _sig(p_next))
    if np.isfinite(r_same):
        _row("Pearson r: simple_mean(D) vs return(D)", f"{r_same:+.4f}", _sig(p_same))
    else:
        _row(
            "Pearson r: simple_mean(D) vs return(D)", "not estimable",
            note=f"(n={len(same_valid)} complete contemporaneous pairs)",
        )

    # Spearman: robust to outliers / non-linearity (review item, 2026-06-12)
    rho_next, rho_p = stats.spearmanr(x, y_next)
    _row("Spearman rho: simple_mean(D) vs return(D+1)",
         f"{rho_next:+.4f}", _sig(rho_p))

    # -- Weighting sensitivity -------------------------------------------------
    # These alternatives share the same session assignment and return target.
    # They are reported side-by-side without selecting the strongest in-sample
    # result, which would turn a diagnostic into model fitting.
    _sub("Weighting sensitivity vs subsequent-session return")
    for column in SIGNAL_VARIANTS[1:]:
        if column not in merged.columns:
            _warn(f"Missing sensitivity variant: {column}")
            continue
        sensitivity = merged.dropna(subset=[column, "next_return"]).copy()
        if len(sensitivity) < 5:
            _info(f"{column}: n={len(sensitivity)} (fewer than 5 observations)")
            continue
        r_variant, p_variant = stats.pearsonr(
            sensitivity[column], sensitivity["next_return"]
        )
        directional_variant = (
            (sensitivity[column] > 0) == (sensitivity["next_return"] > 0)
        ).mean()
        _row(
            column,
            f"r={r_variant:+.4f}",
            f"p={p_variant:.3f}; direction={directional_variant:.1%}; "
            f"n={len(sensitivity)} (sensitivity only)",
        )

    if np.isfinite(r_same) and abs(r_next) < abs(r_same):
        _info("The contemporaneous association is larger in magnitude than the "
              "subsequent-session association.")
        _info("That timing pattern is consistent with reaction, but does not identify causality.")
    elif abs(r_next) > 0.15 and p_next < 0.10:
        _info("An in-sample subsequent-session association is visible; treat it as exploratory.")
    else:
        _info("No clear subsequent-session relationship is visible in this sample.")

    # -- Hit rate -------------------------------------------------------------
    _sub("Primary-baseline directional agreement")
    directional = valid.loc[
        (valid["simple_mean"] != 0) & (valid["next_return"] != 0)
    ].copy()
    directional["pos_sent"] = directional["simple_mean"] > 0
    directional["up_next"] = directional["next_return"] > 0
    directional_n = len(directional)
    hit_rate = (
        float((directional["pos_sent"] == directional["up_next"]).mean())
        if directional_n else None
    )
    n_pos = int(directional["pos_sent"].sum())
    n_neg = directional_n - n_pos

    if hit_rate is None:
        _info("No non-zero signal/return pairs are eligible for directional agreement")
    else:
        _row(
            "Overall hit rate", f"{hit_rate:.1%}",
            note=f"(n={directional_n}; zero directions excluded; 50% null)",
        )
    if n_pos > 0:
        pos_hit = directional.loc[directional["pos_sent"], "up_next"].mean()
        _row("  When pos sentiment -> market up?", f"{pos_hit:.1%}", f"(n={n_pos})")
    if n_neg > 0:
        neg_hit = directional.loc[~directional["pos_sent"], "up_next"].mean()
        _row("  When neg sentiment -> market down?", f"{1-neg_hit:.1%}", f"(n={n_neg})")

    if directional_n:
        n_hits = int((directional["pos_sent"] == directional["up_next"]).sum())
        binom = stats.binomtest(n_hits, directional_n, p=0.5)
        _row("Binomial test p-value", f"{binom.pvalue:.4f}",
             note="(two-sided test against a 50% null; exploratory)")

    # Fixed-band hit rates: NOT tuned (tuning session-score bands on this sample
    # would be in-sample fitting). Bands chosen a priori to mirror the headline
    # thresholds.
    for band in (0.05, 0.10):
        sel = valid[
            (valid["simple_mean"].abs() >= band) & (valid["next_return"] != 0)
        ]
        if len(sel) >= 5:
            hr = ((sel["simple_mean"] > 0) == (sel["next_return"] > 0)).mean()
            _row(f"  Hit rate where |sentiment| >= {band:.2f}", f"{hr:.1%}",
                 f"(n={len(sel)})")

    # -- Hypothetical exposure diagnostic ------------------------------------
    _sub("Diagnostic exposure: long when simple_mean > 0, flat otherwise")
    valid["strategy_return"] = valid.apply(
        lambda r: r["next_return"] if r["simple_mean"] > 0 else 0.0, axis=1
    )
    strat_total = valid["strategy_return"].sum()
    bnh_total   = valid["next_return"].sum()
    _row("Hypothetical cumulative return %",  f"{strat_total:+.2f}")
    _row("Buy-and-hold cumulative return %",  f"{bnh_total:+.2f}")

    if strat_total > bnh_total:
        _info("Hypothetical exposure has a higher in-sample sum over this window")
    else:
        _info("Hypothetical exposure has a lower in-sample sum over this window")

    _info("Not a validated strategy or alpha estimate; no costs or out-of-sample test.")

    # -- Granger temporal-predictive diagnostic (if enough data) --------------
    if n >= 20:
        _sub("Granger temporal-predictive diagnostic (exploratory)")
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            import io, contextlib

            # Use the same-session return: grangercausalitytests applies its own
            # lags, so lag 1 evaluates simple_mean(D-1) against return(D).
            # Feeding next_return here would double-shift the outcome.
            data = valid[["same_session_return", "simple_mean"]].dropna()
            buf  = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = grangercausalitytests(data, maxlag=2, verbose=True)

            for lag, res in result.items():
                f_stat = res[0]["ssr_ftest"][0]
                p_val  = res[0]["ssr_ftest"][1]
                sig    = "(*)" if p_val < 0.10 else ""
                _row(f"  Granger lag={lag}  F={f_stat:.3f}", f"p={p_val:.4f}", sig)

            _info("These p-values test incremental temporal association at each lag;")
            _info("they do not establish economic or causal effects.")
            _info("Caveat: thin-session filtering creates row gaps, so lags are approximate.")
        except ImportError:
            _info("Install statsmodels for Granger causality: pip install statsmodels")
    else:
        _info(f"Granger diagnostic is hidden below 20 eligible observations (have {n})")

    return {
        "overlapping_days":  n,
        "signal_variant":    "simple_mean",
        "pearson_r_next":    float(r_next),
        "pearson_p_next":    float(p_next),
        "hit_rate":          hit_rate,
        "directional_observations": directional_n,
    }


# -----------------------------------------------------------------------------
# Report saving + trend comparison
# -----------------------------------------------------------------------------

def _save_evaluate_report(results: dict) -> None:
    """Save key metrics to reports/evaluate_YYYY-MM-DD.json and print trend vs previous."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    today     = date.today().isoformat()
    json_path = reports_dir / f"evaluate_{today}.json"

    l0 = results.get("l0", {})
    l1 = results.get("l1", {})
    l2 = results.get("l2", {})
    l3 = results.get("l3", {})
    l4 = results.get("l4", {})
    l5 = results.get("l5", {})

    snapshot = {
        "date":                today,
        "total_headlines":     l1.get("total"),
        "other_pct":           l1.get("other_pct"),
        "mean_score":          l2.get("mean_score"),
        "std_score":           l2.get("std_score"),
        "signal_days":         l3.get("days"),
        "mean_articles_per_day": l3.get("mean_articles_per_day"),
        "price_days":          l4.get("days"),
        "unexpected_gaps":     l4.get("unexpected_gaps"),
        "overlap_days":        l5.get("overlapping_days"),
        "signal_variant":      l5.get("signal_variant"),
        "pearson_r_next":      l5.get("pearson_r_next"),
        "pearson_p_next":      l5.get("pearson_p_next"),
        "hit_rate":            l5.get("hit_rate"),
        "missing_probs":       l0.get("missing_probs"),
    }

    # Load previous report for trend comparison
    prev_files = sorted(reports_dir.glob("evaluate_*.json"))
    prev_files = [p for p in prev_files if p.name != json_path.name]

    if prev_files:
        try:
            with open(prev_files[-1], encoding="utf-8") as f:
                prev = json.load(f)
            _hdr(f"TREND vs {prev.get('date', prev_files[-1].stem)}")
            trend_keys = [
                ("total_headlines",     "Total headlines",    ""),
                ("signal_days",         "Signal sessions",        ""),
                ("overlap_days",        "Overlap sessions (L5)",  f"/{MINIMUM_OVERLAP_DAYS} gate"),
                ("mean_score",          "Mean score",         ""),
                ("other_pct",           "'other' category %", "%"),
                ("pearson_r_next",      "Pearson r (session baseline D→D+1)",  ""),
                ("hit_rate",            "Hit rate",           ""),
            ]
            for key, label, unit in trend_keys:
                curr_val = snapshot.get(key)
                prev_val = prev.get(key)
                if curr_val is None or prev_val is None:
                    continue
                try:
                    delta = curr_val - prev_val
                    sign  = "+" if delta >= 0 else ""
                    fmt   = ".4f" if isinstance(curr_val, float) and abs(curr_val) < 10 else ".1f"
                    print(f"  {label:<34}  {curr_val:{fmt}}{unit}  "
                          f"(Δ {sign}{delta:{fmt}})")
                except Exception:
                    pass
        except Exception:
            pass

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print()
    _ok(f"Report saved → {json_path}")


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    _hdr("SUMMARY")

    l0 = results.get("l0", {})
    l1 = results.get("l1", {})
    l2 = results.get("l2", {})
    l3 = results.get("l3", {})
    l4 = results.get("l4", {})
    l5 = results.get("l5", {})

    rows = [
        ("L0 Pipeline runs logged",        l0.get("run_count",          "-")),
        ("L0 Scored rows missing components", l0.get("missing_probs",  "-")),
        ("L1 Total headlines",             l1.get("total",              "-")),
        ("L1 Has-date rate",               f"{l1.get('has_date_pct', 0):.0f}%"   if l1 else "-"),
        ("L1 'other' category %",          f"{l1.get('other_pct', 0):.1f}%"      if l1 else "-"),
        ("L2 Mean score",                  f"{l2.get('mean_score', 0):+.4f}"     if l2 else "-"),
        ("L2 Score std dev",               f"{l2.get('std_score', 0):.4f}"       if l2 else "-"),
        ("L3 Sessions with signal",         l3.get("days",               "-")),
        ("L3 Avg articles/session",         f"{l3.get('mean_articles_per_day', 0):.1f}" if l3 else "-"),
        ("L4 Price days",                  l4.get("days",               "-")),
        ("L4 Unexpected price gaps",        l4.get("unexpected_gaps",    "-")),
        ("L5 Eligible overlap sessions",     l5.get("overlapping_days", "-")),
        ("L5 Gate threshold",              MINIMUM_OVERLAP_DAYS),
        ("L5 Pearson r (simple_mean D -> D+1 session)",
                                           f"{l5.get('pearson_r_next', 0):+.4f}  "
                                           f"p={l5.get('pearson_p_next', 1):.3f}" if l5 and l5.get("pearson_r_next") else "-"),
        ("L5 Hit rate",                    f"{l5.get('hit_rate', 0):.1%}"         if l5 and l5.get("hit_rate") else "-"),
    ]

    print()
    for label, val in rows:
        print(f"  {label:<42} {val}")
    print()
    _info(f"Exploratory display gate: overlap >= {MINIMUM_OVERLAP_DAYS} observations")
    _info("The gate is not evidence of reliability, validated alpha, or a strategy.")
    _info("To grow signal: run 'python main.py run' (or 'run.bat run') daily")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pipeline quality layer by layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db",     default=DB_PATH, help="SQLite DB path")
    parser.add_argument("--layer",  type=int, choices=[0, 1, 2, 3, 4, 5],
                        help="Run only one layer (0=system, 1-5=pipeline layers)")
    parser.add_argument("--sample", type=int, default=10,
                        help="Number of spot-check headlines in L2 (default: 10)")
    parser.add_argument("--save",   action="store_true",
                        help="Save metrics snapshot to reports/evaluate_YYYY-MM-DD.json")
    args = parser.parse_args()

    db.init_db(args.db)    # ensures migrations are applied before reading
    results = {}

    layers = [args.layer] if args.layer is not None else [0, 1, 2, 3, 4, 5]

    for layer in layers:
        if layer == 0: results["l0"] = audit_system(args.db)
        if layer == 1: results["l1"] = audit_scraper(args.db)
        if layer == 2: results["l2"] = audit_model(args.db, spot_n=args.sample)
        if layer == 3: results["l3"] = audit_aggregate(args.db)
        if layer == 4: results["l4"] = audit_prices(args.db)
        if layer == 5: results["l5"] = audit_signal(args.db)

    if args.layer is None:
        print_summary(results)

    if args.save:
        _save_evaluate_report(results)


if __name__ == "__main__":
    main()
