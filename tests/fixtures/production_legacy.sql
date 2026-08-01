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
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    daily_return REAL
);

CREATE TABLE daily_sentiment (
    date TEXT PRIMARY KEY,
    avg_score REAL NOT NULL,
    std_score REAL,
    headline_count INTEGER NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    neutral_count INTEGER NOT NULL,
    bull_bear_ratio REAL,
    updated_at TEXT NOT NULL
);

CREATE TABLE category_daily_sentiment (
    date TEXT NOT NULL,
    category TEXT NOT NULL,
    avg_score REAL NOT NULL,
    headline_count INTEGER NOT NULL,
    PRIMARY KEY (date, category)
);

CREATE TABLE daily_sentiment_by_signal (
    date TEXT PRIMARY KEY,
    avg_score REAL NOT NULL,
    std_score REAL,
    headline_count INTEGER NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    neutral_count INTEGER NOT NULL,
    bull_bear_ratio REAL,
    updated_at TEXT NOT NULL
);

CREATE TABLE usdtry_rates (
    date TEXT PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL
);

CREATE TABLE market_factors (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    label TEXT,
    close REAL,
    daily_return REAL,
    PRIMARY KEY (date, symbol)
);

CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    git_commit TEXT,
    schema_version INTEGER,
    started_at TEXT,
    metrics_json TEXT
);

CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_id INTEGER REFERENCES headlines(id),
    source_tier TEXT NOT NULL,
    source TEXT NOT NULL,
    published_at TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    session_window TEXT,
    title TEXT NOT NULL,
    raw_text TEXT,
    event_type TEXT,
    direction REAL,
    magnitude REAL,
    novelty REAL,
    credibility REAL,
    sentiment_score REAL,
    sentiment_label TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL,
    external_id TEXT
);

CREATE TABLE event_entities (
    event_id INTEGER NOT NULL REFERENCES events(event_id),
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_type, entity_id)
);

CREATE TABLE kv_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE external_series (
    date TEXT NOT NULL,
    series TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (date, series)
);

CREATE TABLE pipeline_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    headlines_scraped INTEGER DEFAULT 0,
    headlines_scored INTEGER DEFAULT 0,
    prices_added INTEGER DEFAULT 0,
    sentiment_days INTEGER DEFAULT 0,
    model_name TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    error_msg TEXT,
    experiment_id TEXT
);

INSERT INTO headlines (
    id, source, title, url, published_at, scraped_at, sentiment_score,
    sentiment_label, scored_at, category, p_positive, p_neutral, p_negative,
    model_name, published_hour, relevance, signal_date
) VALUES
    (101, 'legacy-feed-a', 'Legacy positive headline',
     'https://example.test/legacy/positive', '2026-06-10',
     '2026-06-10T08:00:00Z', 0.625, 'positive', '2026-06-10T08:01:00Z',
     'bist_company', 0.70, 0.20, 0.10, 'gpt-5-mini-2025-08-07/p3', 11, 0.90,
     '2026-06-10'),
    (102, 'legacy-feed-b', 'Legacy explicit neutral headline',
     'https://example.test/legacy/neutral', '2026-06-10',
     '2026-06-10T09:00:00Z', 0.0, 'neutral', '2026-06-10T09:01:00Z',
     'macro', 0.15, 0.70, 0.15, 'gpt-5-mini-2025-08-07/p3', 12, 0.75,
     '2026-06-10'),
    (103, 'legacy-feed-c', 'Legacy unscored headline',
     'https://example.test/legacy/pending', '2026-06-11',
     '2026-06-11T10:00:00Z', NULL, NULL, NULL, 'market', NULL, NULL, NULL,
     NULL, 13, NULL, '2026-06-11');

INSERT INTO bist100_prices VALUES
    ('2026-06-10', 100.0, 103.0, 99.0, 102.0, 1000000.0, NULL),
    ('2026-06-11', 102.0, 104.0, 101.0, 103.0, 1100000.0, 0.9803921569);

INSERT INTO daily_sentiment VALUES
    ('2026-06-10', 0.314159, 0.2718, 77, 31, 19, 27, 0.62,
     '2026-06-12T00:00:00Z');

INSERT INTO category_daily_sentiment VALUES
    ('2026-06-10', 'legacy-sentinel', -0.123456, 44);

INSERT INTO daily_sentiment_by_signal VALUES
    ('2026-06-10', 0.2468, 0.1357, 66, 20, 18, 28, 0.5263157895,
     '2026-06-12T00:00:00Z');

INSERT INTO usdtry_rates VALUES
    ('2026-06-10', 39.10, 39.30, 39.00, 39.25);

INSERT INTO market_factors VALUES
    ('2026-06-10', 'EEM', 'Emerging markets', 42.50, 0.40);

INSERT INTO experiments VALUES
    ('legacy-production-p3', 'deadbeef', 3, '2026-06-01T00:00:00Z',
     '{"status":"historical"}');

INSERT INTO events (
    event_id, headline_id, source_tier, source, published_at, signal_date,
    session_window, title, event_type, direction, sentiment_score,
    sentiment_label, model_version, created_at, external_id
) VALUES
    (501, 101, 'B', 'legacy-feed-a', '2026-06-10', '2026-06-10',
     'during_session', 'Legacy positive headline', 'bridge_headline', 0.625,
     0.625, 'positive', 'gpt-5-mini-2025-08-07/p3',
     '2026-06-10T08:01:00Z', 'legacy:event:501');

INSERT INTO event_entities VALUES (501, 'ticker', 'XU100');
INSERT INTO kv_state VALUES ('legacy_cursor', '2026-06-10T00:00:00Z');
INSERT INTO external_series VALUES ('2026-06-10', 'legacy_attention', 17.0);

INSERT INTO pipeline_runs (
    run_id, started_at, finished_at, headlines_scraped, headlines_scored,
    prices_added, sentiment_days, model_name, status, error_msg, experiment_id
) VALUES
    (901, '2026-06-10T06:00:00Z', '2026-06-10T06:05:00Z', 3, 2, 2, 1,
     'gpt-5-mini-2025-08-07/p3', 'ok', NULL, 'legacy-production-p3');

PRAGMA foreign_keys = ON;
