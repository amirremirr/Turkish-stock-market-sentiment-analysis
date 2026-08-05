"""
Pipeline orchestrator - runs each step individually or as a complete pipeline.

Steps (can be run independently or chained via run_all)
------------------------------------------------------
  1. scrape      - pull latest headlines into the DB
  2. score       - run the configured scorer on unscored headlines
  3. aggregate   - compute session baselines and descriptive sensitivities
  4. prices      - fetch BIST 100 OHLCV via yfinance
  5. plot        - generate and save the visualisation
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

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
    EXPERIMENT_ID,
    LLM_SENTIMENT_MODEL,
    LLM_SCORING_MAX_ATTEMPTS,
    MARKET_DATA_STALE_AFTER_DAYS,
    MINIMUM_HEADLINES_PER_DAY,
    PLOT_OUTPUT,
    SENTIMENT_BACKEND,
    SENTIMENT_INTENSITY_FLOOR,
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


@dataclass
class StepOutcome:
    """Machine-readable result for one pipeline component.

    Public step functions still return integers by default for compatibility.
    ``run_all`` requests these richer outcomes so a zero-row update is not
    confused with success, degraded operation, or failure.
    """

    count: int = 0
    status: str = "success"
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class MixedExperimentAggregationError(RuntimeError):
    """Raised when eligible scores span more than one experiment identity."""

    def __init__(self, experiment_ids: List[str]):
        self.experiment_ids = list(experiment_ids)
        joined = ", ".join(self.experiment_ids)
        super().__init__(
            "aggregation blocked because eligible scores span multiple "
            f"experiment identities: {joined}"
        )


def _issue(component: str, code: str, message: str, **details: Any) -> Dict[str, Any]:
    issue: Dict[str, Any] = {
        "component": component,
        "code": code,
        "message": message,
    }
    if details:
        issue["details"] = details
    return issue


# -----------------------------------------------------------------------------
# Step 1 - Scrape
# -----------------------------------------------------------------------------

def scrape_step(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    db_path: str = DB_PATH,
    return_outcome: bool = False,
):
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
    all_rss_failed = total_sources > 0 and failed_count == total_sources
    if all_rss_failed:
        logger.critical("ALL %d RSS sources failed; no headlines collected this run", total_sources)

    # -- HTML fallback --
    html_attempted = False
    if not headlines:
        html_attempted = True
        logger.info("RSS returned nothing - falling back to HTML scraper")
        html = sc.InvestingTRScraper(session)
        raw = html.scrape(max_pages=5)
        headlines = [
            h for h in raw
            if h["published_at"] is None or h["published_at"] >= since
        ]

    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    if failed_count:
        warnings.append(_issue(
            "scrape", "source_failure",
            f"{failed_count} of {total_sources} RSS sources failed",
            failed_sources=[
                key for key, value in rss.source_status.items()
                if value.startswith("failed")
            ],
        ))

    if not headlines:
        logger.warning("No headlines returned by any scraper.")
        # An empty but otherwise healthy feed can legitimately have no items.
        # It is a hard failure only when every configured RSS source failed and
        # the fallback also produced no observations.
        if all_rss_failed and html_attempted:
            errors.append(_issue(
                "scrape", "all_sources_failed",
                "All RSS sources failed and the HTML fallback returned no observations",
            ))
            outcome = StepOutcome(
                status="failed", errors=errors, warnings=warnings,
                details={"source_status": dict(rss.source_status), "observations": 0},
            )
        else:
            outcome = StepOutcome(
                status="degraded" if failed_count else "success",
                warnings=warnings,
                details={"source_status": dict(rss.source_status), "observations": 0},
            )
        return outcome if return_outcome else 0

    inserted = db.insert_headlines(headlines, db_path=db_path)
    outcome = StepOutcome(
        count=inserted,
        status="degraded" if failed_count else "success",
        warnings=warnings,
        details={
            "source_status": dict(rss.source_status),
            "observations": len(headlines),
            "canonical_rows_inserted": inserted,
        },
    )
    return outcome if return_outcome else inserted


# -----------------------------------------------------------------------------
# Step 2 - Score
# -----------------------------------------------------------------------------

def score_step(db_path: str = DB_PATH, return_outcome: bool = False):
    """
    Score all unscored headlines.  Returns number of headlines scored.

    Stores backend-specific score components (p_positive, p_neutral,
    p_negative) and model_name alongside the continuous score. For the LLM
    backend these are synthetic compatibility fields, not probabilities.
    If the model raises, the exception propagates - stale NULL scores are never
    silently left behind from a partial run.
    """
    logger.info("=== STEP 2: Sentiment scoring ===")
    unscored = db.get_unscored_headlines(db_path=db_path)

    if unscored.empty:
        logger.info("No unscored headlines - nothing to do.")
        outcome = StepOutcome(count=0, status="success", details={"candidates": 0})
        return outcome if return_outcome else 0

    logger.info("Scoring %d headlines with backend '%s' ...", len(unscored), SENTIMENT_BACKEND)
    scorer = _get_scorer()
    outcome = _score_candidates(unscored, scorer, db_path)
    return outcome if return_outcome else outcome.count

def _score_candidates(unscored: pd.DataFrame, scorer, db_path: str) -> StepOutcome:
    """Score candidates with omission-aware, missing-only retries."""

    max_attempts = max(
        1,
        int(getattr(scorer, "max_scoring_attempts", LLM_SCORING_MAX_ATTEMPTS)),
    )
    component_kind = getattr(scorer, "score_components_kind", None)
    current = [
        {
            "id": int(row.id),
            "title": str(row.title),
            "scoring_attempts": int(row.scoring_attempts),
        }
        for row in unscored.itertuples(index=False)
        if int(row.scoring_attempts) < max_attempts
    ]
    scored_count = 0
    failed_ids: List[int] = []

    while current:
        titles = [row["title"] for row in current]
        try:
            if hasattr(scorer, "analyze_partial"):
                partial = scorer.analyze_partial(titles)
                mode = "analysis"
            elif hasattr(scorer, "analyze"):
                aligned = scorer.analyze(titles)
                partial = {
                    idx: value for idx, value in enumerate(aligned)
                    if value is not None
                }
                mode = "analysis"
            elif hasattr(scorer, "score_partial"):
                partial = scorer.score_partial(titles)
                mode = "score"
            else:
                aligned = scorer.score(titles)
                partial = {
                    idx: value for idx, value in enumerate(aligned)
                    if value is not None
                }
                mode = "score"
        except Exception as exc:
            db.mark_scoring_attempts_failed(
                [row["id"] for row in current],
                f"{type(exc).__name__}: {exc}",
                max_attempts,
                db_path=db_path,
            )
            raise

        if not isinstance(partial, dict):
            exc = RuntimeError("scorer partial result must be keyed by input index")
            db.mark_scoring_attempts_failed(
                [row["id"] for row in current], str(exc), max_attempts,
                db_path=db_path,
            )
            raise exc

        valid: Dict[int, Any] = {
            idx: value for idx, value in partial.items()
            if isinstance(idx, int) and not isinstance(idx, bool)
            and 0 <= idx < len(current) and value is not None
        }
        success_rows = [(current[idx], value) for idx, value in sorted(valid.items())]
        missing_rows = [row for idx, row in enumerate(current) if idx not in valid]

        if success_rows:
            if mode == "analysis":
                db.batch_update_sentiment(
                    [
                        (
                            result["score"], result["label"], result["p_pos"],
                            result["p_neu"], result["p_neg"], scorer.model_name,
                            result.get("score_components_kind") or component_kind,
                            row["id"],
                        )
                        for row, result in success_rows
                    ],
                    db_path=db_path,
                    experiment_id=EXPERIMENT_ID,
                )
                db.update_categories(
                    [(result["category"], row["id"]) for row, result in success_rows],
                    db_path=db_path,
                )
                db.update_relevance(
                    [(result["relevance"], row["id"]) for row, result in success_rows],
                    db_path=db_path,
                )
                db.reconcile_relevance_exclusions(
                    [row["id"] for row, _ in success_rows], db_path=db_path,
                )
            else:
                db.batch_update_sentiment(
                    [
                        (*result, scorer.model_name, component_kind, row["id"])
                        for row, result in success_rows
                    ],
                    db_path=db_path,
                    experiment_id=EXPERIMENT_ID,
                )
            scored_count += len(success_rows)

        if not missing_rows:
            break

        statuses = db.mark_scoring_attempts_failed(
            [row["id"] for row in missing_rows],
            "scorer response omitted or invalidated this item",
            max_attempts,
            db_path=db_path,
        )
        failed_ids.extend(
            row["id"] for row in missing_rows
            if statuses[row["id"]] == "failed"
        )
        current = [
            row for row in missing_rows
            if statuses[row["id"]] == "retry_pending"
        ]

    _sync_events(db_path)
    failed_ids = sorted(set(failed_ids))
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    status = "success"
    if failed_ids:
        issue = _issue(
            "scoring",
            "scoring_unavailable" if scored_count == 0 else "items_failed_after_retries",
            f"{len(failed_ids)} headline(s) exhausted the scoring retry limit",
            headline_ids=failed_ids,
            retry_limit=max_attempts,
        )
        if scored_count == 0:
            status = "failed"
            errors.append(issue)
        else:
            status = "degraded"
            warnings.append(issue)
    return StepOutcome(
        count=scored_count,
        status=status,
        warnings=warnings,
        errors=errors,
        details={
            "candidates": len(unscored),
            "scored": scored_count,
            "failed_after_retries": len(failed_ids),
            "retry_limit": max_attempts,
        },
    )


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

    Nothing is deleted: relevance (0-1) is stored and low-relevance decisions
    become reversible, versioned exclusions. Sentiment is left untouched. Returns
    {'recategorized': n_changed, 'low_relevance': [(grade, title)...]} and
    re-aggregates so the new grades take effect.
    """
    from sentiment_llm import get_scorer as get_llm_scorer

    with db._conn(db_path) as con:
        rows = con.execute("SELECT id, title, category FROM headlines ORDER BY id").fetchall()
    if not rows:
        return {"recategorized": 0, "low_relevance": []}

    scorer = get_llm_scorer()
    remaining = list(enumerate(rows))
    analyses_by_index: Dict[int, dict] = {}
    max_attempts = max(
        1, int(getattr(scorer, "max_scoring_attempts", LLM_SCORING_MAX_ATTEMPTS))
    )
    for _attempt in range(max_attempts):
        if not remaining:
            break
        titles = [row["title"] for _, row in remaining]
        if hasattr(scorer, "analyze_partial"):
            partial = scorer.analyze_partial(titles)
        else:
            aligned = scorer.analyze(titles)
            partial = {
                index: result for index, result in enumerate(aligned)
                if result is not None
            }
        next_remaining = []
        for local_index, (original_index, row) in enumerate(remaining):
            result = partial.get(local_index)
            if result is None:
                next_remaining.append((original_index, row))
            else:
                analyses_by_index[original_index] = result
        remaining = next_remaining

    cat_updates, rel_updates, changed, low_rel = [], [], 0, []
    for index, r in enumerate(rows):
        a = analyses_by_index.get(index)
        if a is None:
            continue
        cat_updates.append((a["category"], r["id"]))
        rel_updates.append((a["relevance"], r["id"]))
        if a["category"] != r["category"]:
            changed += 1
        if a["relevance"] < 0.25:
            low_rel.append((a["relevance"], r["title"]))

    db.update_categories(cat_updates, db_path=db_path)
    db.update_relevance(rel_updates, db_path=db_path)
    db.reconcile_relevance_exclusions(
        [headline_id for _, headline_id in rel_updates], db_path=db_path,
    )
    aggregate_step(db_path=db_path)

    logger.info("recategorize-llm: %d categories changed of %d headlines; "
                "%d graded below the aggregation threshold",
                changed, len(rows), len(low_rel))
    return {
        "recategorized": changed,
        "low_relevance": low_rel,
        "missing": len(remaining),
    }


def aggregate_step(
    db_path: str = DB_PATH,
    *,
    allow_mixed_experiments: bool = False,
    return_outcome: bool = False,
):
    """
    Recompute descriptive and session-aligned sentiment derived tables.

    CORRECTNESS CONTRACT
    --------------------
    Derived aggregate rows are rebuilt from eligible scored headlines so stale
    summaries cannot survive a changed exclusion or scoring state. Raw headline
    observations and canonical headlines are never deleted by this step.

    Also backfills NULL category values for any headlines that were inserted
    before the category column existed.  To force-reclassify ALL categories
    (e.g. after adding new category rules), call recategorize_step(force=True)
    before aggregate_step.

    ``daily_signal_variants.simple_mean`` is the primary session baseline. The
    legacy daily tables retain ``full_weighted`` for descriptive compatibility.

    Aggregation is blocked before any mutation when eligible scores span more
    than one experiment identity. ``allow_mixed_experiments=True`` is an
    explicit override; the structured outcome is then degraded and includes a
    persisted-ready warning. Returns the number of distinct signal sessions by
    default, or a :class:`StepOutcome` when ``return_outcome=True``.
    """
    logger.info("=== STEP 3: Aggregate ===")

    experiment_ids = db.get_eligible_experiment_ids(db_path=db_path)
    mixed_experiments = len(experiment_ids) > 1
    if mixed_experiments and not allow_mixed_experiments:
        raise MixedExperimentAggregationError(experiment_ids)

    aggregation_warnings: List[Dict[str, Any]] = []
    aggregation_status = "success"
    if mixed_experiments:
        aggregation_status = "degraded"
        aggregation_warnings.append(_issue(
            "aggregation",
            "mixed_experiments_allowed",
            "Explicit override allowed aggregation across multiple experiment identities",
            experiment_ids=experiment_ids,
        ))

    def _result(count: int):
        outcome = StepOutcome(
            count=count,
            status=aggregation_status,
            warnings=aggregation_warnings,
            details={
                "eligible_experiment_ids": experiment_ids,
                "mixed_experiments": mixed_experiments,
                "mixed_experiments_override": bool(allow_mixed_experiments),
            },
        )
        return outcome if return_outcome else count

    # -- Backfill NULL categories only (fast path) ----------------------------
    recategorize_step(db_path=db_path, force=False)
    db.backfill_session_assignments(db_path=db_path)
    db.reconcile_relevance_exclusions(db_path=db_path)

    # -- Load all scored headlines -------------------------------------------
    with db._conn(db_path) as con:
        df = pd.read_sql_query(
            """SELECT h.id, h.source, h.published_at AS date, h.signal_date,
                      h.sentiment_score, h.sentiment_label, h.category,
                      h.published_hour, h.timing_bucket, h.relevance,
                      e.event_id
               FROM headlines AS h
               LEFT JOIN events AS e ON e.headline_id = h.id
               WHERE h.processing_status = 'scored'
                 AND h.sentiment_score IS NOT NULL
                 AND h.sentiment_label IS NOT NULL
                 AND h.model_name IS NOT NULL
                 AND h.scored_at IS NOT NULL
                 AND h.p_positive IS NOT NULL
                 AND h.p_neutral IS NOT NULL
                 AND h.p_negative IS NOT NULL
                 AND h.published_at IS NOT NULL
                 AND NOT EXISTS (
                     SELECT 1 FROM headline_exclusions AS x
                     WHERE x.headline_id = h.id AND x.restored_at IS NULL
                 )""",
            con,
        )
        observed_source_rows = con.execute(
            """SELECT headline_id, source FROM raw_headline_observations
               WHERE headline_id IS NOT NULL"""
        ).fetchall()

    observed_sources: Dict[int, set] = {}
    for row in observed_source_rows:
        observed_sources.setdefault(int(row["headline_id"]), set()).add(row["source"])

    # Relevance eligibility is represented by active exclusion history above.
    # A manually restored rule decision must therefore remain eligible on this
    # rebuild. NULL relevance keeps neutral weight for legacy rows.
    if not df.empty:
        df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce").fillna(1.0)

    # -- Delete stale derived rows BEFORE recomputing ------------------------
    with db._conn(db_path) as con:
        con.execute("DELETE FROM daily_sentiment")
        con.execute("DELETE FROM daily_sentiment_by_signal")
        con.execute("DELETE FROM category_daily_sentiment")
        con.execute("DELETE FROM daily_signal_variants")
        con.execute("DELETE FROM category_sentiment_by_signal")
    logger.info("Cleared stale aggregate rows; recomputing from %d scored headlines", len(df))

    if df.empty:
        logger.warning("No scored headlines with dates found - aggregate tables left empty.")
        return _result(0)

    from aggregation.signals import compute_signal_variants

    def _variants(g: pd.DataFrame) -> dict:
        result = compute_signal_variants(
            g.to_dict("records"), intensity_floor=SENTIMENT_INTENSITY_FLOOR,
        )
        # The raw audit table preserves sources that share one canonical URL.
        # Count their union so source breadth is not lost to canonical dedup.
        sources = set(str(source) for source in g["source"].dropna())
        for headline_id in g["id"].astype(int):
            sources.update(observed_sources.get(headline_id, set()))
        result["source_count"] = len(sources)
        return result

    def _legacy_row(variants: dict) -> dict:
        pos = int(variants["positive_count"])
        neg = int(variants["negative_count"])
        return {
            "avg_score": variants["full_weighted"],
            "std_score": variants["dispersion"],
            "headline_count": variants["headline_count"],
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": variants["neutral_count"],
            "bull_bear_ratio": pos / (pos + neg) if pos + neg else None,
        }

    overall_rows = []
    for day, group in df.groupby("date"):
        agg = _legacy_row(_variants(group))
        agg["date"] = day
        overall_rows.append(agg)

    db.upsert_daily_sentiment(overall_rows, db_path=db_path)

    # -- Signal-aligned aggregation (session the news can first affect) -------
    sig_df = df.dropna(subset=["signal_date"])
    signal_rows = []
    variant_rows = []
    for day, group in sig_df.groupby("signal_date"):
        variants = _variants(group)
        variants["signal_date"] = day
        variant_rows.append(variants)
        agg = _legacy_row(variants)
        agg["date"] = day
        signal_rows.append(agg)
    if signal_rows:
        db.upsert_daily_sentiment(signal_rows, db_path=db_path,
                                  table="daily_sentiment_by_signal")
        db.upsert_signal_variants(variant_rows, db_path=db_path)

    # -- Per-category aggregation --------------------------------------------
    # The legacy calendar-date category table retains full_weighted for backward
    # compatibility. The session category table below stores the simple baseline.
    cat_rows = []
    for (day, cat), group in df.groupby(["date", "category"]):
        agg = _legacy_row(_variants(group))
        cat_rows.append({
            "date":           day,
            "category":       cat,
            "avg_score":      agg["avg_score"],
            "headline_count": agg["headline_count"],
        })

    if cat_rows:
        db.upsert_category_sentiment(cat_rows, db_path=db_path)

    category_signal_rows = []
    for (day, category), group in sig_df.groupby(["signal_date", "category"]):
        variants = _variants(group)
        category_signal_rows.append({
            "signal_date": day,
            "category": category,
            "simple_mean": variants["simple_mean"],
            "headline_count": variants["headline_count"],
        })
    if category_signal_rows:
        db.upsert_category_signal_sentiment(category_signal_rows, db_path=db_path)

    logger.info(
        "Aggregate complete: %d signal sessions | %d calendar days | %d category-session rows",
        len(variant_rows), len(overall_rows), len(category_signal_rows),
    )
    return _result(len(variant_rows))


# -----------------------------------------------------------------------------
# Step 4 - Fetch BIST 100 prices
# -----------------------------------------------------------------------------

def prices_step(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ticker: str = BIST100_TICKER,
    db_path: str = DB_PATH,
    return_outcome: bool = False,
):
    """Download BIST100 OHLCV and store daily returns. Returns row count."""
    logger.info("=== STEP 4: Fetch prices (%s) ===", ticker)
    start_date = (date.today() - timedelta(days=lookback_days + 5)).isoformat()

    try:
        raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        outcome = _market_data_fallback_outcome(
            db_path, f"yfinance download failed: {type(exc).__name__}: {exc}"
        )
        return outcome if return_outcome else 0

    if raw.empty:
        logger.warning("yfinance returned empty data for %s", ticker)
        outcome = _market_data_fallback_outcome(
            db_path, f"yfinance returned no rows for {ticker}"
        )
        return outcome if return_outcome else 0

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

    # Bars are classified against the fetch time, so a run started before the
    # Istanbul close stores today's row as provisional instead of passing an
    # intraday snapshot off as that session's daily bar.
    counts = db.upsert_prices(df, db_path=db_path)
    db.backfill_price_bar_status(db_path=db_path)
    flagged = db.list_price_bars_for_review(db_path=db_path)

    price_warnings: List[Dict[str, Any]] = []
    price_status = "success"
    if counts.get("provisional"):
        price_warnings.append(_issue(
            "market_data", "provisional_price_bar",
            "Session had not settled at fetch time; bar stored as provisional "
            "and withheld from analysis until a later run confirms it",
            provisional_rows=counts["provisional"],
        ))
    if flagged:
        price_status = "degraded"
        price_warnings.append(_issue(
            "market_data", "price_bars_need_review",
            "Stored bars carry a completeness or volume flag",
            rows=[
                {"date": row["date"], "status": row["bar_status"],
                 "reason": row["bar_review_reason"]}
                for row in flagged[:10]
            ],
            flagged_total=len(flagged),
        ))

    logger.info(
        "Stored %d price rows for %s (%d provisional, %d flagged for review)",
        counts["written"], ticker, counts.get("provisional", 0), len(flagged),
    )
    outcome = StepOutcome(
        count=counts["written"], status=price_status, warnings=price_warnings,
        details={
            "ticker": ticker, "downloaded_rows": len(df),
            "bar_counts": counts, "flagged_for_review": len(flagged),
        },
    )
    return outcome if return_outcome else counts["written"]


def _market_data_fallback_outcome(db_path: str, reason: str) -> StepOutcome:
    """Classify a price-fetch failure using the age of the local cache."""
    with db._conn(db_path) as con:
        latest = con.execute("SELECT MAX(date) FROM bist100_prices").fetchone()[0]
    age_days: Optional[int] = None
    if latest:
        try:
            age_days = (date.today() - date.fromisoformat(str(latest)[:10])).days
        except ValueError:
            age_days = None

    details = {
        "latest_cached_market_date": latest,
        "cache_age_days": age_days,
        "stale_after_days": MARKET_DATA_STALE_AFTER_DAYS,
    }
    if age_days is not None and age_days <= MARKET_DATA_STALE_AFTER_DAYS:
        return StepOutcome(
            status="degraded",
            warnings=[_issue(
                "market_data", "fresh_cache_used", reason, **details,
            )],
            details=details,
        )
    return StepOutcome(
        status="failed",
        errors=[_issue(
            "market_data", "market_data_stale", reason, **details,
        )],
        details=details,
    )


# -----------------------------------------------------------------------------
# Step 4b - Fetch USD/TRY FX rates (Alpha Vantage)
# -----------------------------------------------------------------------------

def fx_rates_step(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    api_key: str = ALPHA_VANTAGE_KEY,
    db_path: str = DB_PATH,
    return_outcome: bool = False,
):
    """
    Download daily USD/TRY FX rates from Alpha Vantage and store them.

    Alpha Vantage FX_DAILY returns up to 100 days of OHLC.
    Free tier: 25 requests/day — this function uses exactly 1 request.
    Returns the row count for compatibility, or a structured outcome when
    ``return_outcome=True``. An absent key is skipped; a configured provider
    failure is degraded and never mislabeled as an intentional skip.
    """
    logger.info("=== STEP 4b: USD/TRY FX rates (Alpha Vantage) ===")

    if not api_key:
        logger.warning("ALPHA_VANTAGE_KEY not set in config.py — skipping FX rates.")
        outcome = StepOutcome(status="skipped", details={"configured": False})
        return outcome if return_outcome else 0

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
        outcome = StepOutcome(
            status="degraded",
            warnings=[_issue(
                "market_data", "fx_provider_failure",
                f"Configured Alpha Vantage request failed: {type(exc).__name__}: {exc}",
            )],
            details={"configured": True},
        )
        return outcome if return_outcome else 0

    series = payload.get("Time Series FX (Daily)")
    if not series:
        # Rate limit or error message
        msg = payload.get("Information") or payload.get("Note") or str(payload)[:120]
        logger.warning("Alpha Vantage returned no FX data: %s", msg)
        outcome = StepOutcome(
            status="degraded",
            warnings=[_issue(
                "market_data", "fx_empty_payload",
                "Configured Alpha Vantage response contained no daily FX series",
                provider_message=msg,
            )],
            details={"configured": True},
        )
        return outcome if return_outcome else 0

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
        outcome = StepOutcome(
            status="degraded",
            warnings=[_issue(
                "market_data", "fx_no_rows_in_window",
                "Alpha Vantage returned no FX rows inside the requested lookback",
                lookback_days=lookback_days,
            )],
            details={"configured": True},
        )
        return outcome if return_outcome else 0

    count = db.upsert_fx_rates(rows, db_path=db_path)
    logger.info("Stored %d USD/TRY FX rows (latest: %s  close: %.4f)",
                count, rows[0]["date"], rows[0]["close"])
    outcome = StepOutcome(
        count=count, status="success", details={"configured": True},
    )
    return outcome if return_outcome else count


# -----------------------------------------------------------------------------
# Step 4c - Market factors (EM index, oil) — context/control series
# -----------------------------------------------------------------------------

def factors_step(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    db_path: str = DB_PATH,
    return_outcome: bool = False,
):
    """Fetch broad market factors (EM, oil) via yfinance. Non-fatal on failure."""
    from config import FACTOR_TICKERS
    logger.info("=== STEP 4c: Market factors %s ===", list(FACTOR_TICKERS))
    start = (date.today() - timedelta(days=lookback_days + 5)).isoformat()
    total = 0
    failures: List[Dict[str, Any]] = []
    for symbol, label in FACTOR_TICKERS.items():
        try:
            raw = yf.download(symbol, start=start, progress=False, auto_adjust=True)
        except Exception as exc:
            logger.warning("factors: %s download failed: %s", symbol, exc)
            failures.append({"symbol": symbol, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if raw.empty:
            logger.warning("factors: %s returned no data", symbol)
            failures.append({"symbol": symbol, "reason": "no rows returned"})
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.sort_index()
        ret = raw["Close"].pct_change().mul(100)
        rows = [
            {"date": idx.strftime("%Y-%m-%d"), "symbol": symbol, "label": label,
             "close": float(close), "daily_return": (None if pd.isna(r) else float(r))}
            for idx, close, r in zip(raw.index, raw["Close"].values, ret.values)
            if not pd.isna(close)
        ]
        total += db.upsert_market_factors(rows, db_path=db_path)
    logger.info("Market factors: %d rows across %d symbols", total, len(FACTOR_TICKERS))
    warnings = []
    status = "success"
    if failures:
        status = "degraded"
        warnings.append(_issue(
            "market_data", "external_factor_failure",
            f"{len(failures)} external factor source(s) failed",
            failures=failures,
        ))
    outcome = StepOutcome(
        count=total, status=status, warnings=warnings,
        details={"factor_rows": total, "factor_failures": failures},
    )
    return outcome if return_outcome else total


# -----------------------------------------------------------------------------
# Step 4d - Reversibly exclude off-topic headlines
# -----------------------------------------------------------------------------

def clean_step(db_path: str = DB_PATH, dry_run: bool = False) -> int:
    """
    Reversibly exclude headlines that fail the current relevance filter.

    Raw and canonical headline rows are never deleted. Use ``dry_run=True`` to
    preview the decision. Returns the number newly excluded (or eligible).
    """
    logger.info("=== STEP: Clean off-topic headlines (dry_run=%s) ===", dry_run)
    if dry_run:
        n = db.count_off_topic_headlines(db_path=db_path)
        logger.info("dry-run: %d headlines would be excluded", n)
        return n
    n = db.clean_off_topic_headlines(db_path=db_path)
    if n > 0:
        logger.info("Re-running aggregate step to refresh derived sentiment tables ...")
        aggregate_step(db_path=db_path)
    return n


def restore_exclusion_step(headline_id: int, db_path: str = DB_PATH) -> bool:
    """Restore one active exclusion and refresh derived aggregates."""
    restored = db.restore_headline_exclusion(headline_id, db_path=db_path)
    if restored:
        aggregate_step(db_path=db_path)
    return restored


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
    allow_mixed_experiments: bool = False,
) -> Dict[str, Any]:
    """Run every component and persist explicit run/component outcomes."""
    db.init_db(db_path=db_path)

    run_id = db.log_run_start(model_name=ACTIVE_SENTIMENT_MODEL, db_path=db_path)
    stats = dict(headlines_scraped=0, headlines_scored=0, prices_added=0, sentiment_days=0)
    component_status = {
        "scrape": "skipped" if skip_scrape else "pending",
        "scoring": "skipped" if skip_score else "pending",
        "aggregation": "skipped" if skip_aggregate else "pending",
        "market_data": "skipped" if skip_prices else "pending",
        "audit": "pending",
    }
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    active_component: Optional[str] = None

    try:
        if not skip_scrape:
            active_component = "scrape"
            component_status[active_component] = "running"
            scrape = scrape_step(
                lookback_days=lookback_days, db_path=db_path, return_outcome=True,
            )
            component_status[active_component] = scrape.status
            warnings.extend(scrape.warnings)
            errors.extend(scrape.errors)
            stats["headlines_scraped"] = scrape.count
            print(f"  [{scrape.status.upper()}] Scrape - {scrape.count} new canonical rows")
            if scrape.status == "failed":
                raise RuntimeError("headline ingestion failed across all configured paths")

        if not skip_score:
            active_component = "scoring"
            component_status[active_component] = "running"
            scoring = score_step(db_path=db_path, return_outcome=True)
            component_status[active_component] = scoring.status
            warnings.extend(scoring.warnings)
            errors.extend(scoring.errors)
            stats["headlines_scored"] = scoring.count
            print(f"  [{scoring.status.upper()}] Score - {scoring.count} headlines scored")
            if scoring.status == "failed":
                raise RuntimeError("sentiment scoring was unavailable for every candidate")

        if not skip_aggregate:
            active_component = "aggregation"
            component_status[active_component] = "running"
            aggregation = aggregate_step(
                db_path=db_path,
                allow_mixed_experiments=allow_mixed_experiments,
                return_outcome=True,
            )
            stats["sentiment_days"] = aggregation.count
            component_status[active_component] = aggregation.status
            warnings.extend(aggregation.warnings)
            errors.extend(aggregation.errors)
            print(
                f"  [{aggregation.status.upper()}] Aggregate - "
                f"{aggregation.count} signal sessions computed"
            )

        price_outcome: Optional[StepOutcome] = None
        if not skip_prices:
            active_component = "market_data"
            component_status[active_component] = "running"
            price_outcome = prices_step(
                lookback_days=lookback_days, db_path=db_path, return_outcome=True,
            )
            component_status[active_component] = price_outcome.status
            warnings.extend(price_outcome.warnings)
            errors.extend(price_outcome.errors)
            stats["prices_added"] = price_outcome.count
            print(
                f"  [{price_outcome.status.upper()}] Prices - "
                f"{price_outcome.count} trading-day rows fetched"
            )
            if price_outcome.status == "failed":
                raise RuntimeError("market price data are unavailable or stale")

        # USD/TRY is optional context. An absent API key is an intentional skip.
        fx_outcome = fx_rates_step(
            lookback_days=lookback_days, db_path=db_path, return_outcome=True,
        )
        warnings.extend(fx_outcome.warnings)
        errors.extend(fx_outcome.errors)
        if fx_outcome.status == "degraded":
            component_status["market_data"] = "degraded"
        if fx_outcome.status == "success":
            print(f"  [SUCCESS] FX rates - {fx_outcome.count} USD/TRY days stored")
        elif fx_outcome.status == "skipped":
            print("  [SKIPPED] FX rates - API key not configured")
        else:
            print(f"  [{fx_outcome.status.upper()}] FX rates - configured provider unavailable")

        try:
            active_component = "market_data"
            factors = factors_step(
                lookback_days=lookback_days, db_path=db_path, return_outcome=True,
            )
            warnings.extend(factors.warnings)
            errors.extend(factors.errors)
            if component_status["market_data"] == "skipped":
                component_status["market_data"] = factors.status
            elif factors.status == "degraded":
                component_status["market_data"] = "degraded"
            print(f"  [{factors.status.upper()}] Factors - {factors.count} rows stored")
        except Exception as exc:
            logger.warning("factors step failed (degraded): %s", exc)
            component_status["market_data"] = "degraded"
            warnings.append(_issue(
                "market_data", "external_factor_failure",
                f"Market-factor step failed: {type(exc).__name__}: {exc}",
            ))
            print("  [DEGRADED] Factors - fetch or persistence error")

        active_component = "audit"
        audit = _processing_audit(db_path)
        component_status[active_component] = audit.status
        warnings.extend(audit.warnings)
        errors.extend(audit.errors)

        if not skip_plot:
            path = plot_step(
                days=lookback_days,
                output_path=output_path,
                db_path=db_path,
                show=show_plot,
            )
            if path:
                print(f"  [SUCCESS] Plot - saved to {path}")
            else:
                print("  [DEGRADED] Plot - insufficient overlapping data")

        final_status = _final_run_status(component_status)
        db.log_run_end(
            run_id,
            status=final_status,
            **stats,
            db_path=db_path,
            scrape_status=component_status["scrape"],
            scoring_status=component_status["scoring"],
            aggregation_status=component_status["aggregation"],
            market_data_status=component_status["market_data"],
            audit_status=component_status["audit"],
            warnings=warnings,
            errors=errors,
        )
        return {
            "run_id": run_id,
            "status": final_status,
            "components": component_status,
            "warnings": warnings,
            "errors": errors,
            **stats,
        }

    except Exception as exc:
        if active_component and component_status.get(active_component) in {"pending", "running"}:
            component_status[active_component] = "failed"
        for key, value in list(component_status.items()):
            if value == "pending":
                component_status[key] = "skipped"
        if isinstance(exc, MixedExperimentAggregationError):
            errors.append(_issue(
                "aggregation",
                "mixed_experiments_blocked",
                str(exc),
                experiment_ids=exc.experiment_ids,
            ))
        error = _issue(
            active_component or "pipeline",
            "component_exception",
            f"{type(exc).__name__}: {exc}",
        )
        errors.append(error)
        db.log_run_end(
            run_id,
            status="failed",
            error_msg=error["message"],
            **stats,
            db_path=db_path,
            scrape_status=component_status["scrape"],
            scoring_status=component_status["scoring"],
            aggregation_status=component_status["aggregation"],
            market_data_status=component_status["market_data"],
            audit_status=component_status["audit"],
            warnings=warnings,
            errors=errors,
        )
        raise


def _processing_audit(db_path: str) -> StepOutcome:
    """Check processing-state integrity without conflating missing and neutral."""
    with db._conn(db_path) as con:
        counts = {
            str(row["processing_status"]): int(row["n"])
            for row in con.execute(
                "SELECT processing_status, COUNT(*) AS n FROM headlines GROUP BY processing_status"
            )
        }
        invalid_scored = int(con.execute(
            """SELECT COUNT(*) FROM headlines
               WHERE processing_status='scored'
                 AND (sentiment_score IS NULL OR sentiment_label IS NULL
                      OR p_positive IS NULL OR p_neutral IS NULL OR p_negative IS NULL
                      OR model_name IS NULL OR scored_at IS NULL)"""
        ).fetchone()[0])

    if invalid_scored:
        return StepOutcome(
            status="failed",
            errors=[_issue(
                "audit", "invalid_scored_state",
                f"{invalid_scored} scored row(s) have incomplete output fields",
            )],
            details={"processing_status_counts": counts},
        )
    unresolved = counts.get("pending", 0) + counts.get("retry_pending", 0)
    failed = counts.get("failed", 0)
    if unresolved or failed:
        return StepOutcome(
            status="degraded",
            warnings=[_issue(
                "audit", "unresolved_processing_items",
                f"{unresolved} pending/retry item(s) and {failed} failed item(s) remain",
                processing_status_counts=counts,
            )],
            details={"processing_status_counts": counts},
        )
    return StepOutcome(status="success", details={"processing_status_counts": counts})


def _final_run_status(component_status: Dict[str, str]) -> str:
    statuses = set(component_status.values())
    if "failed" in statuses:
        return "failed"
    if "degraded" in statuses:
        return "degraded"
    return "success"
