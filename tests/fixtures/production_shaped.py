"""Build a production-shaped legacy database for migration testing.

``production_legacy.sql`` pins the exact legacy schema with a handful of
hand-written rows.  It is precise but tiny, so it cannot exercise what a real
migration meets: thousands of rows, a realistic publication-hour distribution,
populated historical derived tables, and stale ``signal_date`` values left
behind by a superseded trading-calendar rule.

This builder generates that shape deterministically.  Given the same seed it
produces byte-identical content, so a test can assert exact counts rather than
ranges.  Nothing here reads the private project database.

Shape targets are taken from the 2026-07-31 canonical production snapshot:

    3465 headlines, 11 sources, 10 categories, one scorer identity
    publication hour: ~14% NULL, ~21% pre-open, ~62% intraday, ~3% post-close
    ~8% of rows graded below the relevance floor
    signal dates around the Kurban Bayrami closure written by the OLD rule

The stale-calendar rows matter most.  The superseded rule treated the whole
2026-05-25..2026-06-01 stretch as closed and pushed every affected headline to
2026-06-02, although the exchange did trade on 2026-05-26 (a half day) and on
2026-06-01.  Re-deriving those assignments is the migration's one intended
change to stored history, so the fixture must contain the fault to prove the
correction happens.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

# The legacy schema, identical in shape to ``production_legacy.sql``.  A test
# asserts the two stay in step, so a future schema edit cannot silently apply
# to only one fixture.
LEGACY_SCHEMA = """
PRAGMA foreign_keys = OFF;

CREATE TABLE headlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT UNIQUE,
    published_at TEXT,
    scraped_at TEXT NOT NULL,
    sentiment_score REAL,
    sentiment_label TEXT,
    scored_at TEXT,
    category TEXT,
    p_positive REAL,
    p_neutral REAL,
    p_negative REAL,
    model_name TEXT,
    published_hour INTEGER,
    relevance REAL,
    signal_date TEXT
);

CREATE TABLE bist100_prices (
    date TEXT PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, daily_return REAL
);

CREATE TABLE daily_sentiment (
    date TEXT PRIMARY KEY,
    avg_score REAL NOT NULL, std_score REAL,
    headline_count INTEGER NOT NULL, positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL, neutral_count INTEGER NOT NULL,
    bull_bear_ratio REAL, updated_at TEXT NOT NULL
);

CREATE TABLE category_daily_sentiment (
    date TEXT NOT NULL, category TEXT NOT NULL,
    avg_score REAL NOT NULL, headline_count INTEGER NOT NULL,
    PRIMARY KEY (date, category)
);

CREATE TABLE daily_sentiment_by_signal (
    date TEXT PRIMARY KEY,
    avg_score REAL NOT NULL, std_score REAL,
    headline_count INTEGER NOT NULL, positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL, neutral_count INTEGER NOT NULL,
    bull_bear_ratio REAL, updated_at TEXT NOT NULL
);

CREATE TABLE usdtry_rates (
    date TEXT PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL
);

CREATE TABLE market_factors (
    date TEXT NOT NULL, symbol TEXT NOT NULL, label TEXT,
    close REAL, daily_return REAL,
    PRIMARY KEY (date, symbol)
);

CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY, git_commit TEXT,
    schema_version INTEGER, started_at TEXT, metrics_json TEXT
);

CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_id INTEGER REFERENCES headlines(id),
    source_tier TEXT NOT NULL, source TEXT NOT NULL,
    published_at TEXT NOT NULL, signal_date TEXT NOT NULL,
    session_window TEXT, title TEXT NOT NULL, raw_text TEXT,
    event_type TEXT, direction REAL, magnitude REAL, novelty REAL,
    credibility REAL, sentiment_score REAL, sentiment_label TEXT,
    model_version TEXT, created_at TEXT NOT NULL, external_id TEXT
);

CREATE TABLE event_entities (
    event_id INTEGER NOT NULL REFERENCES events(event_id),
    entity_type TEXT NOT NULL, entity_id TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_type, entity_id)
);

CREATE TABLE kv_state (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE external_series (
    date TEXT NOT NULL, series TEXT NOT NULL, value REAL,
    PRIMARY KEY (date, series)
);

CREATE TABLE pipeline_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL, finished_at TEXT,
    headlines_scraped INTEGER DEFAULT 0, headlines_scored INTEGER DEFAULT 0,
    prices_added INTEGER DEFAULT 0, sentiment_days INTEGER DEFAULT 0,
    model_name TEXT, status TEXT NOT NULL DEFAULT 'running',
    error_msg TEXT, experiment_id TEXT
);

PRAGMA foreign_keys = ON;
"""

CORPUS_START = date(2026, 3, 12)
CORPUS_END = date(2026, 7, 31)
LEGACY_MODEL = "gpt-5-mini-2025-08-07/p3"
RELEVANCE_FLOOR = 0.25

# Source weights approximating the canonical snapshot's mix.
SOURCES = [
    ("aa_ekonomi", 769), ("sozcu_gundem", 631), ("dunya", 492),
    ("haberturk_ekonomi", 436), ("bloomberght", 335), ("sabah_ekonomi", 260),
    ("cumhuriyet_ekonomi", 239), ("investing_tr_economy", 211),
    ("hurriyet_ekonomi", 61), ("ntv_ekonomi", 20), ("aa_politika", 11),
]
CATEGORIES = [
    "other", "global_risk", "turkey_macro", "energy_commodities",
    "bist_company", "political_risk", "rates_tcmb", "banks", "fx_lira",
    "crypto",
]

# Full-day closures inside the corpus window, as the exchange actually observed
# them.  2026-05-26 and 2026-06-01 are deliberately absent: both traded.
REAL_CLOSURES = {
    "2026-03-20", "2026-03-21", "2026-03-22", "2026-04-23", "2026-05-01",
    "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30",
    "2026-07-15",
}
# What the superseded rule wrongly treated as closed, producing stale dates.
LEGACY_EXTRA_CLOSURES = {"2026-05-26", "2026-06-01"}


def _is_open(day: date, *, legacy: bool = False) -> bool:
    if day.weekday() >= 5:
        return False
    stamp = day.isoformat()
    if stamp in REAL_CLOSURES:
        return False
    return not (legacy and stamp in LEGACY_EXTRA_CLOSURES)


def _next_open(day: date, *, legacy: bool = False, inclusive: bool = False) -> date:
    candidate = day if inclusive else day + timedelta(days=1)
    while not _is_open(candidate, legacy=legacy):
        candidate += timedelta(days=1)
    return candidate


def _legacy_signal_date(published: date, hour: int | None) -> str:
    """Reproduce the superseded rule that wrote the stored signal dates."""

    if not _is_open(published, legacy=True):
        return _next_open(published, legacy=True, inclusive=True).isoformat()
    if hour is None:
        return _next_open(published, legacy=True).isoformat()
    if hour < 10:
        return published.isoformat()
    if hour <= 18:
        return published.isoformat()
    return _next_open(published, legacy=True).isoformat()


def _label_for(score: float) -> str:
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


def _weighted_choice(rng: random.Random, weighted: List[tuple]) -> Any:
    total = sum(weight for _, weight in weighted)
    cut = rng.uniform(0, total)
    running = 0.0
    for value, weight in weighted:
        running += weight
        if cut <= running:
            return value
    return weighted[-1][0]


def build_production_shaped_legacy_db(
    path: str | Path,
    *,
    headline_count: int = 3465,
    seed: int = 20260731,
) -> Dict[str, Any]:
    """Create a legacy-schema database at *path* and return its expectations.

    The returned dictionary states exactly what the fixture contains so tests
    can assert on precise numbers instead of re-deriving them.
    """

    path = Path(path)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)

    calendar_days = []
    cursor = CORPUS_START
    while cursor <= CORPUS_END:
        calendar_days.append(cursor)
        cursor += timedelta(days=1)
    open_days = [day for day in calendar_days if _is_open(day)]

    headlines: List[tuple] = []
    null_hour_rows = 0
    stale_signal_rows = 0
    below_floor_rows = 0

    for index in range(headline_count):
        published = calendar_days[index % len(calendar_days)]
        bucket = _weighted_choice(
            rng, [("null", 14), ("pre_open", 21), ("intraday", 62), ("post", 3)]
        )
        if bucket == "null":
            hour = None
            null_hour_rows += 1
        elif bucket == "pre_open":
            hour = rng.randint(0, 9)
        elif bucket == "intraday":
            hour = rng.randint(10, 18)
        else:
            hour = rng.randint(19, 23)

        score = round(rng.uniform(-1.0, 1.0), 6)
        label = _label_for(score)
        positive = round(rng.uniform(0.0, 1.0), 6)
        neutral = round(rng.uniform(0.0, 1.0 - positive), 6)
        negative = round(max(0.0, 1.0 - positive - neutral), 6)

        relevance = (
            round(rng.uniform(0.0, RELEVANCE_FLOOR - 0.01), 4)
            if rng.random() < 0.08
            else round(rng.uniform(RELEVANCE_FLOOR, 1.0), 4)
        )
        if relevance < RELEVANCE_FLOOR:
            below_floor_rows += 1

        signal = _legacy_signal_date(published, hour)
        if signal != _next_open(
            published, inclusive=not (hour is not None and hour <= 18)
        ).isoformat():
            stale_signal_rows += 1

        headlines.append((
            index + 1,
            _weighted_choice(rng, SOURCES),
            f"Fixture headline {index + 1}",
            f"https://fixture.test/headline/{index + 1}",
            published.isoformat(),
            f"{published.isoformat()}T23:30:00Z",
            score, label, f"{published.isoformat()}T23:45:00Z",
            CATEGORIES[index % len(CATEGORIES)],
            positive, neutral, negative, LEGACY_MODEL,
            hour, relevance, signal,
        ))

    con = sqlite3.connect(str(path))
    try:
        con.executescript(LEGACY_SCHEMA)
        con.executemany(
            "INSERT INTO headlines (id, source, title, url, published_at,"
            " scraped_at, sentiment_score, sentiment_label, scored_at,"
            " category, p_positive, p_neutral, p_negative, model_name,"
            " published_hour, relevance, signal_date)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            headlines,
        )

        close = 10000.0
        prices, previous = [], None
        for day in open_days:
            close = round(close * (1.0 + rng.uniform(-0.02, 0.02)), 4)
            daily_return = (
                None if previous is None
                else round((close / previous - 1.0) * 100.0, 8)
            )
            prices.append((
                day.isoformat(), round(close * 0.999, 4), round(close * 1.01, 4),
                round(close * 0.99, 4), close,
                float(rng.randint(3_000_000, 12_000_000) * 1000), daily_return,
            ))
            previous = close
        con.executemany("INSERT INTO bist100_prices VALUES (?,?,?,?,?,?,?)", prices)

        con.executemany(
            "INSERT INTO usdtry_rates VALUES (?,?,?,?,?)",
            [
                (day.isoformat(), 39.0, 39.4, 38.9, round(39.0 + rng.uniform(-0.3, 0.3), 4))
                for day in open_days
            ],
        )
        con.executemany(
            "INSERT INTO market_factors VALUES (?,?,?,?,?)",
            [
                (day.isoformat(), symbol, label,
                 round(rng.uniform(40.0, 80.0), 4), round(rng.uniform(-2.0, 2.0), 6))
                for day in open_days
                for symbol, label in (
                    ("EEM", "emerging_markets"),
                    ("BZ=F", "brent_oil"),
                    ("USDTRY=X", "usd_try"),
                )
            ],
        )
        con.executemany(
            "INSERT INTO external_series VALUES (?,?,?)",
            [
                (day.isoformat(), series, round(rng.uniform(0.0, 100.0), 4))
                for day in open_days
                for series in ("gdelt_tone", "gt_dolar")
            ],
        )

        # Historical derived tables, internally consistent with the headlines
        # above so a test can prove the migration leaves them untouched.
        by_publish: Dict[str, List[tuple]] = {}
        by_signal: Dict[str, List[tuple]] = {}
        by_category: Dict[tuple, List[float]] = {}
        for row in headlines:
            published_at, score, label = row[4], row[6], row[7]
            by_publish.setdefault(published_at, []).append((score, label))
            by_signal.setdefault(row[16], []).append((score, label))
            by_category.setdefault((published_at, row[9]), []).append(score)

        def _aggregate(grouped: Dict[str, List[tuple]]) -> List[tuple]:
            rows = []
            for key in sorted(grouped):
                entries = grouped[key]
                scores = [score for score, _ in entries]
                mean = sum(scores) / len(scores)
                positive = sum(1 for _, lab in entries if lab == "positive")
                negative = sum(1 for _, lab in entries if lab == "negative")
                neutral = sum(1 for _, lab in entries if lab == "neutral")
                rows.append((
                    key, round(mean, 8), 0.25, len(entries), positive, negative,
                    neutral,
                    round(positive / (positive + negative), 8) if positive + negative else None,
                    "2026-07-31T09:23:00Z",
                ))
            return rows

        con.executemany(
            "INSERT INTO daily_sentiment VALUES (?,?,?,?,?,?,?,?,?)",
            _aggregate(by_publish),
        )
        con.executemany(
            "INSERT INTO daily_sentiment_by_signal VALUES (?,?,?,?,?,?,?,?,?)",
            _aggregate(by_signal),
        )
        con.executemany(
            "INSERT INTO category_daily_sentiment VALUES (?,?,?,?)",
            [
                (day, category, round(sum(scores) / len(scores), 8), len(scores))
                for (day, category), scores in sorted(by_category.items())
            ],
        )

        con.executemany(
            "INSERT INTO events (event_id, headline_id, source_tier, source,"
            " published_at, signal_date, title, direction, magnitude,"
            " credibility, sentiment_score, sentiment_label, model_version,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (row[0], row[0], "C", row[1], row[4], row[16], row[2],
                 row[6], abs(row[6]), 0.5, row[6], row[7], LEGACY_MODEL,
                 "2026-07-31T09:23:00Z")
                for row in headlines
            ],
        )
        con.execute("INSERT INTO event_entities VALUES (1, 'ticker', 'XU100')")
        con.execute(
            "INSERT INTO experiments VALUES"
            " ('legacy-production-p3', 'fixture', 3, '2026-03-12T00:00:00Z',"
            " '{\"status\":\"historical\"}')"
        )
        con.execute("INSERT INTO kv_state VALUES ('legacy_cursor', '2026-07-31')")
        con.executemany(
            "INSERT INTO pipeline_runs (run_id, started_at, finished_at,"
            " headlines_scraped, headlines_scored, prices_added,"
            " sentiment_days, model_name, status, error_msg, experiment_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (index + 1, f"2026-0{3 + index % 5}-1{index % 9}T09:00:00Z",
                 f"2026-0{3 + index % 5}-1{index % 9}T09:05:00Z",
                 50, 50, 1, 1, LEGACY_MODEL, "ok", None, None)
                for index in range(52)
            ],
        )
        con.commit()
    finally:
        con.close()

    return {
        "path": str(path),
        "headline_count": headline_count,
        "null_published_hour": null_hour_rows,
        "below_relevance_floor": below_floor_rows,
        "stale_signal_date_rows": stale_signal_rows,
        "trading_days": len(open_days),
        "model_name": LEGACY_MODEL,
        "legacy_tables": [
            "bist100_prices", "category_daily_sentiment", "daily_sentiment",
            "daily_sentiment_by_signal", "event_entities", "events",
            "experiments", "external_series", "headlines", "kv_state",
            "market_factors", "pipeline_runs", "usdtry_rates",
        ],
    }
