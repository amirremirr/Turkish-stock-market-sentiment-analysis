"""
Pipeline orchestrator - runs each step individually or as a complete pipeline.

Steps (can be run independently or chained via run_all)
------------------------------------------------------
  1. scrape      - pull latest headlines into the DB
  2. score       - run XLM-RoBERTa on unscored headlines
  3. aggregate   - compute daily sentiment averages from scored headlines
  4. prices      - fetch BIST 100 OHLCV via yfinance
  5. plot        - generate and save the visualisation
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import database as db
import scraper as sc
import visualize as viz
from config import (
    ALPHA_VANTAGE_KEY,
    BIST100_TICKER,
    DB_PATH,
    DEFAULT_LOOKBACK_DAYS,
    LLM_SENTIMENT_MODEL,
    MINIMUM_HEADLINES_PER_DAY,
    PLOT_OUTPUT,
    SENTIMENT_BACKEND,
    SENTIMENT_CONFIDENCE_FLOOR,
    SENTIMENT_MODEL,
)


def _get_scorer():
    """Return the active sentiment scorer per SENTIMENT_BACKEND."""
    if SENTIMENT_BACKEND == "llm":
        from sentiment_llm import get_scorer
    else:
        from sentiment import get_scorer
    return get_scorer()


ACTIVE_SENTIMENT_MODEL = (
    LLM_SENTIMENT_MODEL if SENTIMENT_BACKEND == "llm" else SENTIMENT_MODEL
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Step 1 - Scrape
# -----------------------------------------------------------------------------

def scrape_step(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    db_path: str = DB_PATH,
) -> int:
    """
    Scrape headlines and persist to DB.  Returns number of new headlines.

    Logs per-source status (ok / failed) after each run.  If ALL RSS sources
    fail, logs a CRITICAL warning but does NOT raise so the caller can decide
    whether to abort or fall back to the HTML scraper.
    """
    logger.info("=== STEP 1: Scrape ===")
    since = date.today() - timedelta(days=lookback_days)
    session = sc._make_session()

    # -- RSS --
    rss = sc.RSSFeedScraper(session)
    headlines = rss.scrape_all(since=since)

    # Per-source status report
    for src, status in rss.source_status.items():
        level = logging.WARNING if status.startswith("failed") else logging.INFO
        logger.log(level, "  [source] %-30s %s", src, status)

    failed_count = sum(1 for v in rss.source_status.values() if v.startswith("failed"))
    total_sources = len(rss.source_status)
    if total_sources > 0 and failed_count == total_sources:
        logger.critical("ALL %d RSS sources failed; no headlines collected this run", total_sources)

    # -- HTML fallback --
    if not headlines:
        logger.info("RSS returned nothing - falling back to HTML scraper")
        html = sc.InvestingTRScraper(session)
        raw = html.scrape(max_pages=5)
        headlines = [
            h for h in raw
            if h["published_at"] is None or h["published_at"] >= since
        ]

    if not headlines:
        logger.warning("No headlines returned by any scraper.")
        return 0

    return db.insert_headlines(headlines, db_path=db_path)


# -----------------------------------------------------------------------------
# Step 2 - Score
# -----------------------------------------------------------------------------

def score_step(db_path: str = DB_PATH) -> int:
    """
    Score all unscored headlines.  Returns number of headlines scored.

    Stores raw model probabilities (p_positive, p_neutral, p_negative) and
    model_name alongside the continuous score, so results are fully auditable.
    If the model raises, the exception propagates - stale NULL scores are never
    silently left behind from a partial run.
    """
    logger.info("=== STEP 2: Sentiment scoring ===")
    unscored = db.get_unscored_headlines(db_path=db_path)

    if unscored.empty:
        logger.info("No unscored headlines - nothing to do.")
        return 0

    logger.info("Scoring %d headlines with backend '%s' ...", len(unscored), SENTIMENT_BACKEND)
    scorer = _get_scorer()

    # LLM backend: one combined call per batch returns sentiment + category +
    # relevance grade (0-1). NOTHING is deleted — low-relevance headlines are
    # weighted toward zero in the aggregation instead, so the judgment stays
    # auditable and reversible. Categories replace the provisional keyword ones.
    if hasattr(scorer, "analyze"):
        analyses = scorer.analyze(unscored["title"].tolist())
        rows = list(unscored.itertuples(index=False))

        for a, r in zip(analyses, rows):
            if a["relevance"] < 0.25:
                logger.info("[LLM relevance %.1f] downweighted: %s",
                            a["relevance"], str(r.title)[:80])

        db.batch_update_sentiment(
            [(a["score"], a["label"], a["p_pos"], a["p_neu"], a["p_neg"],
              scorer.model_name, int(r.id))
             for a, r in zip(analyses, rows)],
            db_path=db_path,
        )
        db.update_categories(
            [(a["category"], int(r.id)) for a, r in zip(analyses, rows)],
            db_path=db_path,
        )
        db.update_relevance(
            [(a["relevance"], int(r.id)) for a, r in zip(analyses, rows)],
            db_path=db_path,
        )
        _sync_events(db_path)
        return len(analyses)

    # XLM-R fallback: sentiment only (keyword categories stay).
    # May raise (e.g. torch import failure, model download error).
    # Let it propagate so the caller marks the run as failed.
    results = scorer.score(unscored["title"].tolist())
    updates = [
        (score, label, p_pos, p_neu, p_neg, scorer.model_name, int(row.id))
        for (score, label, p_pos, p_neu, p_neg), row
        in zip(results, unscored.itertuples(index=False))
    ]
    db.batch_update_sentiment(updates, db_path=db_path)
    _sync_events(db_path)
    return len(updates)


def _sync_events(db_path: str) -> None:
    """Dual-write scored headlines into the events table (migration Phase 2)."""
    from config import EVENTS_DUAL_WRITE
    if not EVENTS_DUAL_WRITE:
        return
    try:
        import events_bridge
        events_bridge.sync(db_path=db_path)
    except Exception as exc:
        # The legacy path must never fail because of the new path.
        logger.warning("events bridge failed (legacy path unaffected): %s", exc)


# -----------------------------------------------------------------------------
# Step 3 - Aggregate daily sentiment
# -----------------------------------------------------------------------------

def recategorize_step(db_path: str = DB_PATH, force: bool = False) -> int:
    """
    (Re)assign category to every headline using the current NEWS_CATEGORIES rules.

    When force=False (default): only rows with category IS NULL are updated.
    When force=True: ALL rows are re-classified, picking up any rule changes.

    Returns the number of rows updated.
    """
    from scraper import classify_headline

    with db._conn(db_path) as con:
        if force:
            rows = con.execute("SELECT id, title FROM headlines").fetchall()
        else:
            rows = con.execute(
                "SELECT id, title FROM headlines WHERE category IS NULL"
            ).fetchall()

        if not rows:
            logger.info("recategorize: nothing to update (force=%s)", force)
            return 0

        updates = [(classify_headline(r["title"]), r["id"]) for r in rows]
        con.executemany("UPDATE headlines SET category=? WHERE id=?", updates)
        logger.info(
            "recategorize: updated %d headlines (force=%s)", len(updates), force
        )
    return len(updates)


def recategorize_llm_step(db_path: str = DB_PATH) -> dict:
    """
    One-pass LLM refresh of category + relevance grade for ALL headlines.

    Nothing is deleted: relevance (0-1) is stored and the aggregation weights
    low-relevance rows toward zero. Sentiment is left untouched. Returns
    {'recategorized': n_changed, 'low_relevance': [(grade, title)...]} and
    re-aggregates so the new grades take effect.
    """
    from sentiment_llm import get_scorer as get_llm_scorer

    with db._conn(db_path) as con:
        rows = con.execute("SELECT id, title, category FROM headlines ORDER BY id").fetchall()
    if not rows:
        return {"recategorized": 0, "low_relevance": []}

    scorer = get_llm_scorer()
    analyses = scorer.analyze([r["title"] for r in rows])

    cat_updates, rel_updates, changed, low_rel = [], [], 0, []
    for a, r in zip(analyses, rows):
        cat_updates.append((a["category"], r["id"]))
        rel_updates.append((a["relevance"], r["id"]))
        if a["category"] != r["category"]:
            changed += 1
        if a["relevance"] < 0.25:
            low_rel.append((a["relevance"], r["title"]))

    db.update_categories(cat_updates, db_path=db_path)
    db.update_relevance(rel_updates, db_path=db_path)
    aggregate_step(db_path=db_path)

    logger.info("recategorize-llm: %d categories changed of %d headlines; "
                "%d graded below the aggregation threshold",
                changed, len(rows), len(low_rel))
    return {"recategorized": changed, "low_relevance": low_rel}


def aggregate_step(db_path: str = DB_PATH) -> int:
    """
    Recompute daily (and per-category) sentiment from the scored headlines table.

    CORRECTNESS CONTRACT
    --------------------
    All rows in daily_sentiment and category_daily_sentiment are DELETED before
    recomputing.  This guarantees the derived tables can never contain stale rows
    left over from headlines that were since cleaned or retroactively filtered.

    Also backfills NULL category values for any headlines that were inserted
    before the category column existed.  To force-reclassify ALL categories
    (e.g. after adding new category rules), call recategorize_step(force=True)
    before aggregate_step.

    Returns the number of distinct days processed.
    """
    logger.info("=== STEP 3: Aggregate ===")

    # -- Backfill NULL categories only (fast path) ----------------------------
    recategorize_step(db_path=db_path, force=False)

    # -- Load all scored headlines -------------------------------------------
    with db._conn(db_path) as con:
        df = pd.read_sql_query(
            """SELECT published_at AS date, signal_date, sentiment_score,
                      sentiment_label, category, published_hour, relevance
               FROM headlines
               WHERE sentiment_score IS NOT NULL
                 AND published_at   IS NOT NULL""",
            con,
        )

    # Relevance gate: ungraded rows (NULL) count fully; graded rows below the
    # threshold are excluded from aggregates (kept in the DB for audit).
    if not df.empty:
        from config import RELEVANCE_MIN_FOR_AGGREGATION
        df["relevance"] = df["relevance"].fillna(1.0)
        n_excluded = int((df["relevance"] < RELEVANCE_MIN_FOR_AGGREGATION).sum())
        if n_excluded:
            logger.info("Aggregate: excluding %d low-relevance headline(s) "
                        "(relevance < %.2f)", n_excluded, RELEVANCE_MIN_FOR_AGGREGATION)
        df = df[df["relevance"] >= RELEVANCE_MIN_FOR_AGGREGATION]

    # -- Delete stale derived rows BEFORE recomputing ------------------------
    with db._conn(db_path) as con:
        con.execute("DELETE FROM daily_sentiment")
        con.execute("DELETE FROM daily_sentiment_by_signal")
        con.execute("DELETE FROM category_daily_sentiment")
    logger.info("Cleared stale aggregate rows; recomputing from %d scored headlines", len(df))

    if df.empty:
        logger.warning("No scored headlines with dates found - aggregate tables left empty.")
        return 0

    # -- Per-day overall aggregation -----------------------------------------
    def _time_weight(hour) -> float:
        """
        Istanbul market hours: open ~10:00, close ~18:00.
        Pre-market news (<=9) sets overnight mood → highest weight.
        Post-market (>18) has least immediate impact → discounted.
        """
        if hour is None or (isinstance(hour, float) and np.isnan(hour)):
            return 1.0
        h = int(hour)
        if h < 10:
            return 1.5
        if h <= 18:
            return 1.0
        return 0.8

    def _agg_group(g: pd.DataFrame) -> dict:
        scores = g["sentiment_score"].values
        labels = g["sentiment_label"].values
        hours  = g["published_hour"].values if "published_hour" in g.columns else [None] * len(scores)
        # LLM relevance grade (0-1); ungraded rows count fully
        rel_w  = (g["relevance"].fillna(1.0).values
                  if "relevance" in g.columns else np.ones(len(scores)))

        # Confidence weight: |score| — high-conviction signals outweigh near-zero
        # ones. Floored so neutral headlines still pull the average toward 0
        # instead of being weighted out of existence (a 10-neutral + one +0.6
        # day should NOT aggregate to +0.6).
        conf_w = np.maximum(np.abs(scores), SENTIMENT_CONFIDENCE_FLOOR)
        # Time-of-day weight: pre-market 1.5×, market hours 1.0×, post-market 0.8×
        time_w = np.array([_time_weight(h) for h in hours])
        combined = conf_w * time_w * rel_w

        total_weight = combined.sum()
        avg_score = (
            float(np.average(scores, weights=combined))
            if total_weight > 0
            else float(np.mean(scores))
        )

        pos = int((labels == "positive").sum())
        neg = int((labels == "negative").sum())
        neu = int((labels == "neutral").sum())
        return {
            "avg_score":       avg_score,
            "std_score":       float(np.std(scores)) if len(scores) > 1 else 0.0,
            "headline_count":  len(scores),
            "positive_count":  pos,
            "negative_count":  neg,
            "neutral_count":   neu,
            "bull_bear_ratio": pos / (pos + neg) if (pos + neg) > 0 else None,
        }

    overall_rows = []
    for day, group in df.groupby("date"):
        agg = _agg_group(group)
        agg["date"] = day
        overall_rows.append(agg)

    db.upsert_daily_sentiment(overall_rows, db_path=db_path)

    # -- Signal-aligned aggregation (session the news can first affect) -------
    sig_df = df.dropna(subset=["signal_date"])
    signal_rows = []
    for day, group in sig_df.groupby("signal_date"):
        agg = _agg_group(group)
        agg["date"] = day
        signal_rows.append(agg)
    if signal_rows:
        db.upsert_daily_sentiment(signal_rows, db_path=db_path,
                                  table="daily_sentiment_by_signal")

    # -- Per-category aggregation --------------------------------------------
    # Uses the SAME confidence + time weighting as the overall daily score so
    # category-level signals are directly comparable to daily_sentiment.
    cat_rows = []
    for (day, cat), group in df.groupby(["date", "category"]):
        agg = _agg_group(group)
        cat_rows.append({
            "date":           day,
            "category":       cat,
            "avg_score":      agg["avg_score"],
            "headline_count": agg["headline_count"],
        })

    if cat_rows:
        db.upsert_category_sentiment(cat_rows, db_path=db_path)

    logger.info(
        "Aggregate complete: %d days overall | %d category-day rows",
        len(overall_rows), len(cat_rows),
    )
    return len(overall_rows)


# -----------------------------------------------------------------------------
# Step 4 - Fetch BIST 100 prices
# -----------------------------------------------------------------------------

def prices_step(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ticker: str = BIST100_TICKER,
    db_path: str = DB_PATH,
) -> int:
    """Download BIST100 OHLCV and store daily returns. Returns row count."""
    logger.info("=== STEP 4: Fetch prices (%s) ===", ticker)
    start_date = (date.today() - timedelta(days=lookback_days + 5)).isoformat()

    try:
        raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return 0

    if raw.empty:
        logger.warning("yfinance returned empty data for %s", ticker)
        return 0

    # Flatten possible MultiIndex columns (yfinance >= 0.2.38)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    df = pd.DataFrame(
        {
            "date":         raw.index.strftime("%Y-%m-%d"),
            "open":         raw["Open"].values,
            "high":         raw["High"].values,
            "low":          raw["Low"].values,
            "close":        raw["Close"].values,
            "volume":       raw.get("Volume", pd.Series(dtype=float)).values
                            if "Volume" in raw.columns else [None] * len(raw),
            "daily_return": raw["Close"].pct_change().mul(100).values,
        }
    )
    df = df.dropna(subset=["close"])

    db.upsert_prices(df, db_path=db_path)
    logger.info("Stored %d price rows for %s", len(df), ticker)
    return len(df)


# -----------------------------------------------------------------------------
# Step 4b - Fetch USD/TRY FX rates (Alpha Vantage)
# -----------------------------------------------------------------------------

def fx_rates_step(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    api_key: str = ALPHA_VANTAGE_KEY,
    db_path: str = DB_PATH,
) -> int:
    """
    Download daily USD/TRY FX rates from Alpha Vantage and store them.

    Alpha Vantage FX_DAILY returns up to 100 days of OHLC.
    Free tier: 25 requests/day — this function uses exactly 1 request.
    Returns number of rows upserted (0 on error or if key not configured).
    """
    logger.info("=== STEP 4b: USD/TRY FX rates (Alpha Vantage) ===")

    if not api_key:
        logger.warning("ALPHA_VANTAGE_KEY not set in config.py — skipping FX rates.")
        return 0

    import requests as _req
    url = (
        "https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol=USD&to_symbol=TRY"
        f"&outputsize=compact&apikey={api_key}"
    )
    try:
        resp = _req.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.error("Alpha Vantage request failed: %s", exc)
        return 0

    series = payload.get("Time Series FX (Daily)")
    if not series:
        # Rate limit or error message
        msg = payload.get("Information") or payload.get("Note") or str(payload)[:120]
        logger.warning("Alpha Vantage returned no FX data: %s", msg)
        return 0

    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

    rows = []
    for day, ohlc in series.items():
        if day < cutoff:
            continue
        rows.append({
            "date":  day,
            "open":  float(ohlc["1. open"]),
            "high":  float(ohlc["2. high"]),
            "low":   float(ohlc["3. low"]),
            "close": float(ohlc["4. close"]),
        })

    if not rows:
        logger.warning("Alpha Vantage returned data but nothing within lookback window")
        return 0

    count = db.upsert_fx_rates(rows, db_path=db_path)
    logger.info("Stored %d USD/TRY FX rows (latest: %s  close: %.4f)",
                count, rows[0]["date"], rows[0]["close"])
    return count


# -----------------------------------------------------------------------------
# Step 4d - Clean off-topic headlines
# -----------------------------------------------------------------------------

def clean_step(db_path: str = DB_PATH, dry_run: bool = False) -> int:
    """
    Remove headlines that fail the current relevance filter.

    Use ``dry_run=True`` to see the count without deleting anything.
    Returns the number of headlines deleted (or that would be deleted).
    """
    logger.info("=== STEP: Clean off-topic headlines (dry_run=%s) ===", dry_run)
    if dry_run:
        n = db.count_off_topic_headlines(db_path=db_path)
        logger.info("dry-run: %d headlines would be removed", n)
        return n
    n = db.clean_off_topic_headlines(db_path=db_path)
    if n > 0:
        logger.info("Re-running aggregate step to refresh daily_sentiment ...")
        aggregate_step(db_path=db_path)
    return n


# -----------------------------------------------------------------------------
# Step 5 - Plot
# -----------------------------------------------------------------------------

def plot_step(
    days: int = DEFAULT_LOOKBACK_DAYS,
    output_path: str = PLOT_OUTPUT,
    db_path: str = DB_PATH,
    show: bool = True,
) -> Optional[str]:
    """Generate and save the visualisation. Returns output path or None."""
    logger.info("=== STEP 5: Plot ===")
    return viz.plot_sentiment_vs_price(
        db_path=db_path,
        days=days,
        output_path=output_path,
        show=show,
    )


# -----------------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------------

def run_all(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    db_path: str = DB_PATH,
    output_path: str = PLOT_OUTPUT,
    show_plot: bool = True,
    skip_scrape: bool = False,
    skip_score: bool = False,
    skip_aggregate: bool = False,
    skip_prices: bool = False,
    skip_plot: bool = False,
) -> None:
    """Run every pipeline step in sequence and log the run to pipeline_runs."""
    db.init_db(db_path=db_path)

    run_id = db.log_run_start(model_name=ACTIVE_SENTIMENT_MODEL, db_path=db_path)
    stats = dict(headlines_scraped=0, headlines_scored=0, prices_added=0, sentiment_days=0)

    try:
        if not skip_scrape:
            n = scrape_step(lookback_days=lookback_days, db_path=db_path)
            stats["headlines_scraped"] = n
            print(f"  [OK] Scrape    - {n} new headlines added")

        if not skip_score:
            n = score_step(db_path=db_path)
            stats["headlines_scored"] = n
            print(f"  [OK] Score     - {n} headlines scored")

        if not skip_aggregate:
            n = aggregate_step(db_path=db_path)
            stats["sentiment_days"] = n
            print(f"  [OK] Aggregate - {n} days of sentiment computed")

        if not skip_prices:
            n = prices_step(lookback_days=lookback_days, db_path=db_path)
            stats["prices_added"] = n
            print(f"  [OK] Prices    - {n} trading days fetched")

        n = fx_rates_step(lookback_days=lookback_days, db_path=db_path)
        if n:
            print(f"  [OK] FX rates  - {n} USD/TRY days stored")
        else:
            print("  [ ] FX rates  - skipped or rate-limited")

        if not skip_plot:
            path = plot_step(
                days=lookback_days,
                output_path=output_path,
                db_path=db_path,
                show=show_plot,
            )
            if path:
                print(f"  [OK] Plot      - saved to {path}")
            else:
                print("  [!!] Plot      - not enough data to render")

        db.log_run_end(run_id, status="ok", **stats, db_path=db_path)

    except Exception as exc:
        db.log_run_end(
            run_id, status="error",
            error_msg=f"{type(exc).__name__}: {exc}",
            **stats, db_path=db_path,
        )
        raise
