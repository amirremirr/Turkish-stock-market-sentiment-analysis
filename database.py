"""
SQLite layer for the sentiment pipeline.

Three tables
------------
  headlines       raw articles + per-headline sentiment scores
  bist100_prices  daily OHLCV + computed daily return
  daily_sentiment aggregated sentiment signal per day
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from config import DB_PATH

logger = logging.getLogger(__name__)

# -- Schema -------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS headlines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    url             TEXT    UNIQUE,
    published_at    TEXT,
    scraped_at      TEXT    NOT NULL,
    -- category assigned by classify_headline() at scrape time
    category        TEXT,
    -- raw model output -------------------------------------------------------
    sentiment_score REAL,
    sentiment_label TEXT,
    p_positive      REAL,
    p_neutral       REAL,
    p_negative      REAL,
    model_name      TEXT,
    scored_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_headlines_published ON headlines(published_at);
-- Note: idx_headlines_category is created in _apply_migrations() after the
-- 'category' column is added, so it is NOT listed here.

CREATE TABLE IF NOT EXISTS bist100_prices (
    date         TEXT PRIMARY KEY,
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       REAL,
    daily_return REAL
);

CREATE TABLE IF NOT EXISTS daily_sentiment (
    date            TEXT PRIMARY KEY,
    avg_score       REAL    NOT NULL,
    std_score       REAL,
    headline_count  INTEGER NOT NULL,
    positive_count  INTEGER NOT NULL,
    negative_count  INTEGER NOT NULL,
    neutral_count   INTEGER NOT NULL,
    bull_bear_ratio REAL,
    updated_at      TEXT    NOT NULL
);

-- Per-category daily sentiment (separate signal per news bucket)
CREATE TABLE IF NOT EXISTS category_daily_sentiment (
    date           TEXT    NOT NULL,
    category       TEXT    NOT NULL,
    avg_score      REAL    NOT NULL,
    headline_count INTEGER NOT NULL,
    PRIMARY KEY (date, category)
);

-- Signal-aligned daily sentiment: keyed by the trading session the news can
-- first affect (trading_calendar.signal_date), not the calendar publish date.
CREATE TABLE IF NOT EXISTS daily_sentiment_by_signal (
    date            TEXT PRIMARY KEY,
    avg_score       REAL    NOT NULL,
    std_score       REAL,
    headline_count  INTEGER NOT NULL,
    positive_count  INTEGER NOT NULL,
    negative_count  INTEGER NOT NULL,
    neutral_count   INTEGER NOT NULL,
    bull_bear_ratio REAL,
    updated_at      TEXT    NOT NULL
);

-- USD/TRY daily FX rates from Alpha Vantage (second independent data source)
CREATE TABLE IF NOT EXISTS usdtry_rates (
    date   TEXT PRIMARY KEY,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL
);

-- Broad market factors (EM index, oil) for controlling BIST moves: lets us
-- later test BIST returns NET of global/EM moves (abnormal return) so a
-- "signal" is not just "all of emerging markets went up that day".
CREATE TABLE IF NOT EXISTS market_factors (
    date         TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    label        TEXT,
    close        REAL,
    daily_return REAL,
    PRIMARY KEY (date, symbol)
);

-- Experiment registry: one row per named research configuration; walk-forward
-- results are appended into metrics_json (migration Phase 0)
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    git_commit      TEXT,
    schema_version  INTEGER,
    started_at      TEXT,
    metrics_json    TEXT
);

-- Event-centric research store (migration Phase 2). Headlines remain the raw
-- input; events are the unit of analysis. Tier A sources (KAP/TCMB) will
-- create events with no headline_id.
CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_id     INTEGER REFERENCES headlines(id),
    source_tier     TEXT NOT NULL,
    source          TEXT NOT NULL,
    published_at    TEXT NOT NULL,
    signal_date     TEXT NOT NULL,
    session_window  TEXT,
    title           TEXT NOT NULL,
    raw_text        TEXT,
    event_type      TEXT,
    direction       REAL,
    magnitude       REAL,
    novelty         REAL,
    credibility     REAL,
    sentiment_score REAL,
    sentiment_label TEXT,
    model_version   TEXT,
    created_at      TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_headline ON events(headline_id)
    WHERE headline_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_signal ON events(signal_date);

CREATE TABLE IF NOT EXISTS event_entities (
    event_id    INTEGER NOT NULL REFERENCES events(event_id),
    entity_type TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_type, entity_id)
);

-- Generic key/value state (e.g. KAP ingestion cursor)
CREATE TABLE IF NOT EXISTS kv_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Audit trail: one row per full pipeline run
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        TEXT    NOT NULL,
    finished_at       TEXT,
    headlines_scraped INTEGER DEFAULT 0,
    headlines_scored  INTEGER DEFAULT 0,
    prices_added      INTEGER DEFAULT 0,
    sentiment_days    INTEGER DEFAULT 0,
    model_name        TEXT,
    status            TEXT    NOT NULL DEFAULT 'running',
    error_msg         TEXT
);
"""

# Columns added after the initial schema (applied via ALTER TABLE at runtime).
# Tuple: (table_name, column_name, column_definition)
_MIGRATIONS: List[Tuple[str, str, str]] = [
    ("pipeline_runs", "experiment_id", "TEXT"),     # provenance (migration Phase 0)
    ("headlines", "category",       "TEXT"),
    ("headlines", "p_positive",     "REAL"),
    ("headlines", "p_neutral",      "REAL"),
    ("headlines", "p_negative",     "REAL"),
    ("headlines", "model_name",     "TEXT"),
    ("headlines", "published_hour", "INTEGER"),  # Istanbul local hour (0-23), UTC+3
    ("headlines", "relevance",      "REAL"),     # LLM relevance grade 0.0-1.0 (NULL = ungraded -> 1.0)
    ("headlines", "signal_date",    "TEXT"),     # first trading session that can react (trading_calendar.signal_date)
    ("events",    "external_id",    "TEXT"),     # e.g. 'kap:1230800' — dedup for non-headline events
]


def _apply_migrations(con: sqlite3.Connection) -> None:
    """
    Add columns (and dependent indexes) introduced after the initial schema.
    Safe to run on a fresh DB (columns already exist) or an old one.
    """
    for table, col, col_def in _MIGRATIONS:
        existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        if col not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
            logger.info("Migration: added column %s.%s", table, col)

    # Indexes that depend on migrated columns (created here, not in _DDL).
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_headlines_category ON headlines(category)"
    )
    con.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_external ON events(external_id) "
        "WHERE external_id IS NOT NULL"
    )

# -- Connection helper ---------------------------------------------------------

@contextmanager
def _conn(db_path: str = DB_PATH):
    """Yield a connection that auto-commits on clean exit and rolls back on error."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- Initialisation ------------------------------------------------------------

def init_db(db_path: str = DB_PATH) -> None:
    """Create tables (if missing) and apply any pending schema migrations."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _conn(db_path) as con:
        con.executescript(_DDL)
        _apply_migrations(con)
    logger.info("Database ready: %s", db_path)


# -- Headlines -----------------------------------------------------------------

def insert_headlines(
    headlines: Iterable[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> int:
    """
    Bulk-insert headlines, silently skipping rows with duplicate URLs.
    Returns the number of *new* rows inserted.
    """
    from trading_calendar import signal_date as _sig  # local import: avoids cycles

    rows: List[Tuple] = []
    now = _now_iso()
    for h in headlines:
        pub = h.get("published_at")
        if isinstance(pub, date):
            pub = pub.isoformat()
        sig = _sig(pub, h.get("published_hour")) if pub else None
        rows.append((
            h.get("source", "unknown"),
            h["title"],
            h.get("url") or None,
            pub,
            now,
            h.get("category") or None,
            h.get("published_hour"),
            sig,
        ))

    if not rows:
        return 0

    # Cross-run dedup the URL UNIQUE constraint cannot provide:
    # NULL-url headlines never collide in SQLite (NULL != NULL), so feeds
    # without URLs (e.g. ntv_ekonomi) would re-insert the same items every
    # daily run. Two headlines are duplicates when their normalised title[:80]
    # AND published date match — recurring titles on later dates are allowed.
    from scraper import _normalise  # local import to avoid circular import
    with _conn(db_path) as con:
        existing = {
            (_normalise(r[0])[:80], r[1])
            for r in con.execute("SELECT title, published_at FROM headlines")
        }
    fresh = [r for r in rows if (_normalise(r[1])[:80], r[3]) not in existing]
    if len(fresh) < len(rows):
        logger.info("Skipped %d cross-run duplicate title(s)", len(rows) - len(fresh))
    if not fresh:
        return 0

    with _conn(db_path) as con:
        before = con.execute("SELECT total_changes()").fetchone()[0]
        con.executemany(
            """INSERT OR IGNORE INTO headlines
               (source, title, url, published_at, scraped_at, category,
                published_hour, signal_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            fresh,
        )
        inserted = con.execute("SELECT total_changes()").fetchone()[0] - before

    logger.info("Inserted %d new headlines (skipped %d duplicates)", inserted, len(fresh) - inserted)
    return inserted


def get_unscored_headlines(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return all headlines that have not yet been sentiment-scored."""
    with _conn(db_path) as con:
        return pd.read_sql_query(
            "SELECT id, title FROM headlines WHERE sentiment_score IS NULL",
            con,
        )


def batch_update_sentiment(
    scores: Iterable[Tuple],
    db_path: str = DB_PATH,
) -> None:
    """
    Update sentiment for multiple headlines in one transaction.

    Each element of ``scores`` must be a 7-tuple:
        (sentiment_score, sentiment_label,
         p_positive, p_neutral, p_negative,
         model_name,
         headline_id)
    """
    now = _now_iso()
    rows = [
        (score, label, p_pos, p_neu, p_neg, model, now, hid)
        for score, label, p_pos, p_neu, p_neg, model, hid in scores
    ]
    with _conn(db_path) as con:
        con.executemany(
            """UPDATE headlines
               SET sentiment_score=?, sentiment_label=?,
                   p_positive=?, p_neutral=?, p_negative=?,
                   model_name=?, scored_at=?
               WHERE id=?""",
            rows,
        )
    logger.info("Updated sentiment for %d headlines", len(rows))


def relabel_from_probs(
    pos_threshold: float,
    neg_threshold: float,
    db_path: str = DB_PATH,
) -> int:
    """
    Recompute sentiment_label for every scored headline from the STORED
    probabilities, using the given thresholds. No model inference needed.

    Use after changing SENTIMENT_POSITIVE_THRESHOLD / SENTIMENT_NEGATIVE_THRESHOLD
    in config.py — otherwise rows scored under the old thresholds keep stale
    labels and positive_count / bull_bear_ratio mix two labelling regimes.

    Returns the number of rows whose label actually changed.
    """
    case_expr = """CASE
            WHEN (p_positive - p_negative) > :pos THEN 'positive'
            WHEN (p_positive - p_negative) < :neg THEN 'negative'
            ELSE 'neutral' END"""
    with _conn(db_path) as con:
        unlabelable = con.execute(
            "SELECT COUNT(*) FROM headlines "
            "WHERE sentiment_score IS NOT NULL AND p_positive IS NULL"
        ).fetchone()[0]
        if unlabelable:
            logger.warning(
                "%d scored rows have no stored probabilities — "
                "cannot relabel them without re-scoring", unlabelable,
            )
        cur = con.execute(
            f"""UPDATE headlines
                SET sentiment_label = {case_expr}
                WHERE p_positive IS NOT NULL
                  AND sentiment_label <> {case_expr}""",
            {"pos": pos_threshold, "neg": neg_threshold},
        )
        changed = cur.rowcount
    logger.info("relabel: %d label(s) changed", changed)
    return changed


def update_categories(pairs: Iterable[Tuple[str, int]], db_path: str = DB_PATH) -> None:
    """Bulk-update headline categories. Each element: (category, headline_id)."""
    pairs = list(pairs)
    with _conn(db_path) as con:
        con.executemany("UPDATE headlines SET category=? WHERE id=?", pairs)
    logger.info("Updated category on %d headlines", len(pairs))


def update_relevance(pairs: Iterable[Tuple[float, int]], db_path: str = DB_PATH) -> None:
    """Bulk-update relevance grades. Each element: (relevance, headline_id)."""
    pairs = list(pairs)
    with _conn(db_path) as con:
        con.executemany("UPDATE headlines SET relevance=? WHERE id=?", pairs)
    logger.info("Updated relevance on %d headlines", len(pairs))


def delete_headlines(ids: Sequence[int], db_path: str = DB_PATH) -> int:
    """Delete headlines by id (chunked under SQLite's parameter limit)."""
    ids = list(ids)
    if not ids:
        return 0
    with _conn(db_path) as con:
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            con.execute(
                f"DELETE FROM headlines WHERE id IN ({','.join('?' * len(chunk))})", chunk
            )
    logger.info("Deleted %d headlines", len(ids))
    return len(ids)


# -- BIST 100 prices -----------------------------------------------------------

def upsert_prices(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """
    Upsert a price DataFrame into bist100_prices.
    ``df`` must have columns: date, open, high, low, close, volume, daily_return
    (all strings / floats - no DatetimeIndex).
    """
    rows = [
        (row.date, row.open, row.high, row.low, row.close, row.volume, row.daily_return)
        for row in df.itertuples(index=False)
    ]
    with _conn(db_path) as con:
        con.executemany(
            """INSERT OR REPLACE INTO bist100_prices
               (date, open, high, low, close, volume, daily_return)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    logger.info("Upserted %d price rows", len(rows))


def get_prices(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    where, params = [], []
    if start:
        where.append("date >= ?")
        params.append(start)
    if end:
        where.append("date <= ?")
        params.append(end)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with _conn(db_path) as con:
        return pd.read_sql_query(
            f"SELECT * FROM bist100_prices {clause} ORDER BY date",
            con,
            params=params,
        )


# -- Daily sentiment aggregates ------------------------------------------------

def upsert_daily_sentiment(rows: Iterable[Dict[str, Any]], db_path: str = DB_PATH,
                           table: str = "daily_sentiment") -> None:
    """Upsert pre-computed daily sentiment rows (dicts with the table columns).

    `table` may be "daily_sentiment" (calendar-aligned, legacy) or
    "daily_sentiment_by_signal" (session-aligned) — identical schemas.
    """
    assert table in ("daily_sentiment", "daily_sentiment_by_signal")
    now = _now_iso()
    data = [
        (
            r["date"],
            r["avg_score"],
            r.get("std_score"),
            r["headline_count"],
            r["positive_count"],
            r["negative_count"],
            r["neutral_count"],
            r.get("bull_bear_ratio"),
            now,
        )
        for r in rows
    ]
    with _conn(db_path) as con:
        con.executemany(
            f"""INSERT OR REPLACE INTO {table}
               (date, avg_score, std_score, headline_count,
                positive_count, negative_count, neutral_count,
                bull_bear_ratio, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data,
        )
    logger.info("Upserted %d rows into %s", len(data), table)


def get_daily_sentiment(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    where, params = [], []
    if start:
        where.append("date >= ?")
        params.append(start)
    if end:
        where.append("date <= ?")
        params.append(end)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with _conn(db_path) as con:
        return pd.read_sql_query(
            f"SELECT * FROM daily_sentiment {clause} ORDER BY date",
            con,
            params=params,
        )


# -- Category daily sentiment --------------------------------------------------

def upsert_category_sentiment(
    rows: Iterable[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> None:
    """
    Upsert per-category daily sentiment rows.
    Each dict must have: date, category, avg_score, headline_count.
    """
    data = [
        (r["date"], r["category"], r["avg_score"], r["headline_count"])
        for r in rows
    ]
    with _conn(db_path) as con:
        con.executemany(
            """INSERT OR REPLACE INTO category_daily_sentiment
               (date, category, avg_score, headline_count)
               VALUES (?, ?, ?, ?)""",
            data,
        )
    logger.info("Upserted %d category-sentiment rows", len(data))


def get_category_daily_sentiment(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Return per-category daily sentiment, optionally filtered by date range."""
    where, params = [], []
    if start:
        where.append("date >= ?"); params.append(start)
    if end:
        where.append("date <= ?"); params.append(end)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with _conn(db_path) as con:
        return pd.read_sql_query(
            f"SELECT * FROM category_daily_sentiment {clause} ORDER BY date, category",
            con, params=params,
        )


# -- Pipeline run audit --------------------------------------------------------

def log_run_start(model_name: Optional[str] = None, db_path: str = DB_PATH,
                  experiment_id: Optional[str] = None) -> int:
    """Insert a 'running' run record. Returns the new run_id."""
    if experiment_id is None:
        from config import EXPERIMENT_ID
        experiment_id = EXPERIMENT_ID
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO pipeline_runs (started_at, model_name, status, experiment_id) "
            "VALUES (?, ?, 'running', ?)",
            (_now_iso(), model_name, experiment_id),
        )
        return cur.lastrowid  # type: ignore[return-value]


def log_run_end(
    run_id: int,
    status: str = "ok",
    headlines_scraped: int = 0,
    headlines_scored: int = 0,
    prices_added: int = 0,
    sentiment_days: int = 0,
    error_msg: Optional[str] = None,
    db_path: str = DB_PATH,
) -> None:
    """Update an existing run record with completion stats."""
    with _conn(db_path) as con:
        con.execute(
            """UPDATE pipeline_runs
               SET finished_at=?, headlines_scraped=?, headlines_scored=?,
                   prices_added=?, sentiment_days=?, status=?, error_msg=?
               WHERE run_id=?""",
            (
                _now_iso(), headlines_scraped, headlines_scored,
                prices_added, sentiment_days, status, error_msg, run_id,
            ),
        )


# -- Headline cleanup (for `main.py clean`) -----------------------------------

def count_off_topic_headlines(db_path: str = DB_PATH) -> int:
    """Return how many headlines would be removed by clean_off_topic_headlines()."""
    from scraper import _is_relevant   # local import to avoid circular at module load
    with _conn(db_path) as con:
        rows = con.execute("SELECT id, title FROM headlines").fetchall()
    return sum(1 for r in rows if not _is_relevant(r["title"]))


def clean_off_topic_headlines(db_path: str = DB_PATH) -> int:
    """
    Delete headlines that fail the current relevance filter.

    Useful after tightening the filter to purge stale off-topic rows that
    were inserted before the filter existed.  Also invalidates any
    daily_sentiment rows whose date no longer has enough remaining headlines
    (the caller should re-run aggregate_step afterwards).

    Returns the number of rows deleted.
    """
    from scraper import _is_relevant   # local import
    with _conn(db_path) as con:
        rows = con.execute("SELECT id, title FROM headlines").fetchall()
        bad_ids = [r["id"] for r in rows if not _is_relevant(r["title"])]
        if not bad_ids:
            logger.info("clean: no off-topic headlines found")
            return 0
        # SQLite parameter limit is 999; chunk to be safe
        deleted = 0
        chunk = 500
        for i in range(0, len(bad_ids), chunk):
            ids = bad_ids[i : i + chunk]
            placeholders = ",".join("?" * len(ids))
            con.execute(f"DELETE FROM headlines WHERE id IN ({placeholders})", ids)
            deleted += len(ids)
        logger.info("clean: deleted %d off-topic headlines", deleted)
    return deleted


# -- USD/TRY FX rates (Alpha Vantage) -----------------------------------------

def upsert_fx_rates(rows: Iterable[Dict[str, Any]], db_path: str = DB_PATH) -> int:
    """
    Upsert USD/TRY daily FX rows.
    Each dict must have: date, open, high, low, close.
    Returns the number of rows upserted.
    """
    data = [(r["date"], r["open"], r["high"], r["low"], r["close"]) for r in rows]
    with _conn(db_path) as con:
        con.executemany(
            """INSERT OR REPLACE INTO usdtry_rates (date, open, high, low, close)
               VALUES (?, ?, ?, ?, ?)""",
            data,
        )
    logger.info("Upserted %d USD/TRY FX rows", len(data))
    return len(data)


def upsert_market_factors(rows: Iterable[Dict[str, Any]], db_path: str = DB_PATH) -> int:
    """Upsert market-factor rows. Each dict: date, symbol, label, close, daily_return."""
    data = [(r["date"], r["symbol"], r.get("label"), r.get("close"), r.get("daily_return"))
            for r in rows]
    with _conn(db_path) as con:
        con.executemany(
            """INSERT OR REPLACE INTO market_factors
               (date, symbol, label, close, daily_return) VALUES (?, ?, ?, ?, ?)""",
            data,
        )
    logger.info("Upserted %d market-factor rows", len(data))
    return len(data)


def get_market_factors(symbol: Optional[str] = None, start: Optional[str] = None,
                       db_path: str = DB_PATH) -> pd.DataFrame:
    """Return market factors, optionally for one symbol / from a start date."""
    where, params = [], []
    if symbol:
        where.append("symbol = ?"); params.append(symbol)
    if start:
        where.append("date >= ?"); params.append(start)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with _conn(db_path) as con:
        return pd.read_sql_query(
            f"SELECT * FROM market_factors {clause} ORDER BY date", con, params=params)


def get_fx_rates(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Return USD/TRY FX rates, optionally filtered by date range."""
    where, params = [], []
    if start:
        where.append("date >= ?"); params.append(start)
    if end:
        where.append("date <= ?"); params.append(end)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with _conn(db_path) as con:
        return pd.read_sql_query(
            f"SELECT * FROM usdtry_rates {clause} ORDER BY date",
            con, params=params,
        )


# -- Quick stats (for `main.py status`) ---------------------------------------

def db_stats(db_path: str = DB_PATH) -> Dict[str, Any]:
    with _conn(db_path) as con:
        total   = con.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
        scored  = con.execute(
            "SELECT COUNT(*) FROM headlines WHERE sentiment_score IS NOT NULL"
        ).fetchone()[0]
        prices  = con.execute("SELECT COUNT(*) FROM bist100_prices").fetchone()[0]
        sent_d  = con.execute("SELECT COUNT(*) FROM daily_sentiment").fetchone()[0]
        fx_days = con.execute("SELECT COUNT(*) FROM usdtry_rates").fetchone()[0]
        min_pub = con.execute("SELECT MIN(published_at) FROM headlines").fetchone()[0]
        max_pub = con.execute("SELECT MAX(published_at) FROM headlines").fetchone()[0]
        # Category breakdown (top categories by headline count)
        cat_rows = con.execute(
            """SELECT category, COUNT(*) AS n FROM headlines
               WHERE category IS NOT NULL
               GROUP BY category ORDER BY n DESC LIMIT 8"""
        ).fetchall()
        # Last run info
        last_run = con.execute(
            "SELECT status, started_at, error_msg FROM pipeline_runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()

    cat_summary = ", ".join(f"{r[0]}:{r[1]}" for r in cat_rows) if cat_rows else "none"
    stats: Dict[str, Any] = {
        "total_headlines":    total,
        "scored_headlines":   scored,
        "unscored_headlines": total - scored,
        "price_days":         prices,
        "fx_rate_days":       fx_days,
        "sentiment_days":     sent_d,
        "oldest_headline":    min_pub,
        "newest_headline":    max_pub,
        "categories":         cat_summary,
    }
    if last_run:
        stats["last_run_status"] = last_run[0]
        stats["last_run_at"]     = last_run[1]
        if last_run[2]:
            stats["last_run_error"] = last_run[2]
    return stats
