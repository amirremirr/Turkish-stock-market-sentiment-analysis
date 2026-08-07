"""
SQLite layer for the sentiment pipeline.

Core tables (plus audit, provenance, and compatibility tables)
--------------------------------------------------------------
  headlines       raw articles + per-headline sentiment scores
  bist100_prices  daily OHLCV + computed daily return
  daily_signal_variants session-aligned baseline and weighting sensitivities
  daily_sentiment legacy calendar-date descriptive aggregate
"""

import hashlib
import json
import logging
import sqlite3
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
    published_timestamp TEXT,
    timing_bucket   TEXT,
    session_rule_version TEXT,
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
    experiment_id   TEXT,
    scored_at       TEXT,
    processing_status       TEXT NOT NULL DEFAULT 'pending'
        CHECK (processing_status IN ('pending', 'scored', 'retry_pending', 'failed')),
    scoring_attempts        INTEGER NOT NULL DEFAULT 0 CHECK (scoring_attempts >= 0),
    last_scoring_attempt_at TEXT,
    scoring_last_error      TEXT,
    score_components_kind   TEXT
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

-- External "zoom-out" series: Google Trends search interest, GDELT global media
-- tone, etc. Generic (date, series, value) so new sources drop straight in.
CREATE TABLE IF NOT EXISTS external_series (
    date   TEXT NOT NULL,
    series TEXT NOT NULL,
    value  REAL,
    PRIMARY KEY (date, series)
);

-- Canonical session-aligned research signals. The unweighted mean is the
-- primary baseline; weighted columns are retained as sensitivity variants.
CREATE TABLE IF NOT EXISTS daily_signal_variants (
    signal_date                    TEXT PRIMARY KEY,
    simple_mean                    REAL NOT NULL,
    relevance_weighted             REAL,
    intensity_relevance_weighted   REAL,
    full_weighted                  REAL,
    headline_count                 INTEGER NOT NULL,
    positive_count                 INTEGER NOT NULL,
    negative_count                 INTEGER NOT NULL,
    neutral_count                  INTEGER NOT NULL,
    unclassified_count             INTEGER NOT NULL DEFAULT 0,
    positive_share                 REAL,
    negative_share                 REAL,
    neutral_share                  REAL,
    sentiment_dispersion           REAL,
    source_count                   INTEGER NOT NULL DEFAULT 0,
    event_count                    INTEGER NOT NULL DEFAULT 0,
    relevance_weight_sum           REAL,
    intensity_relevance_weight_sum REAL,
    full_weight_sum                REAL,
    updated_at                     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS category_sentiment_by_signal (
    signal_date    TEXT NOT NULL,
    category       TEXT NOT NULL,
    simple_mean    REAL NOT NULL,
    headline_count INTEGER NOT NULL,
    PRIMARY KEY (signal_date, category)
);

-- Source-level ingestion audit. This deliberately does not share the
-- headlines.url uniqueness constraint: the same URL observed in two feeds is
-- two source-distinct observations, even if only one canonical headline row
-- is retained. observation_key makes replaying a fetch idempotent.
CREATE TABLE IF NOT EXISTS raw_headline_observations (
    observation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_key  TEXT NOT NULL UNIQUE,
    headline_id      INTEGER REFERENCES headlines(id) ON DELETE SET NULL,
    source           TEXT NOT NULL,
    title            TEXT NOT NULL,
    url              TEXT,
    published_at     TEXT,
    published_timestamp TEXT,
    published_hour   INTEGER,
    timing_bucket    TEXT,
    observed_at      TEXT NOT NULL,
    raw_payload_json TEXT,
    is_excluded      INTEGER NOT NULL DEFAULT 0,
    exclusion_reason TEXT,
    exclusion_rule   TEXT,
    exclusion_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_observations_source_published
    ON raw_headline_observations(source, published_at);
CREATE INDEX IF NOT EXISTS idx_raw_observations_headline
    ON raw_headline_observations(headline_id);

-- Reversible filtering history. Restoring an exclusion timestamps it; a later
-- exclusion appends a new row instead of overwriting the old decision.
CREATE TABLE IF NOT EXISTS headline_exclusions (
    exclusion_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_id       INTEGER NOT NULL REFERENCES headlines(id) ON DELETE CASCADE,
    exclusion_reason  TEXT NOT NULL,
    exclusion_rule    TEXT,
    exclusion_version TEXT,
    excluded_at       TEXT NOT NULL,
    restored_at       TEXT,
    restored_by_user  INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_headline_exclusions_one_active
    ON headline_exclusions(headline_id) WHERE restored_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_headline_exclusions_history
    ON headline_exclusions(headline_id, excluded_at);

-- Phase A descriptive indicators. All keyed by the versions that produced them,
-- so a rule change creates new rows instead of silently rewriting history.
CREATE TABLE IF NOT EXISTS daily_family_signals (
    signal_date            TEXT NOT NULL,
    signal_family          TEXT NOT NULL,
    experiment_id          TEXT NOT NULL,
    family_version         TEXT NOT NULL,
    simple_mean            REAL,
    relevance_weighted     REAL,
    median_sentiment       REAL,
    min_sentiment          REAL,
    max_sentiment          REAL,
    sentiment_std          REAL,
    headline_count         INTEGER NOT NULL,
    source_count           INTEGER NOT NULL,
    positive_share         REAL,
    neutral_share          REAL,
    negative_share         REAL,
    avg_relevance          REAL,
    market_recap_count     INTEGER NOT NULL DEFAULT 0,
    unknown_timing_count   INTEGER NOT NULL DEFAULT 0,
    excluded_count         INTEGER NOT NULL DEFAULT 0,
    unresolved_count       INTEGER NOT NULL DEFAULT 0,
    ambiguous_count        INTEGER NOT NULL DEFAULT 0,
    sample_sufficiency     TEXT NOT NULL,
    updated_at             TEXT NOT NULL,
    PRIMARY KEY (signal_date, signal_family, experiment_id, family_version)
);
CREATE INDEX IF NOT EXISTS idx_daily_family_signals_family
    ON daily_family_signals(signal_family, signal_date);

-- Prior-only normalization. scope is outlet | outlet_family | family.
CREATE TABLE IF NOT EXISTS abnormal_tone_daily (
    signal_date       TEXT NOT NULL,
    scope             TEXT NOT NULL,
    scope_key         TEXT NOT NULL,
    experiment_id     TEXT NOT NULL,
    window_sessions   INTEGER NOT NULL,
    min_history       INTEGER NOT NULL,
    observed_mean     REAL,
    prior_mean        REAL,
    prior_std         REAL,
    prior_count       INTEGER NOT NULL,
    abnormal_tone     REAL,
    rolling_z         REAL,
    rolling_percentile REAL,
    method_version    TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (signal_date, scope, scope_key, experiment_id, window_sessions, method_version)
);

CREATE TABLE IF NOT EXISTS news_disagreement_daily (
    signal_date          TEXT NOT NULL,
    signal_family        TEXT NOT NULL,
    experiment_id        TEXT NOT NULL,
    headline_count       INTEGER NOT NULL,
    source_count         INTEGER NOT NULL,
    within_day_std       REAL,
    cross_outlet_std     REAL,
    max_minus_min        REAL,
    strong_positive_share REAL,
    strong_negative_share REAL,
    sentiment_entropy    REAL,
    camp_gap             REAL,
    camp_gap_sources     INTEGER,
    official_vs_media_gap REAL,
    min_sources_met      INTEGER NOT NULL,
    method_version       TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    PRIMARY KEY (signal_date, signal_family, experiment_id, method_version)
);

CREATE TABLE IF NOT EXISTS news_volume_daily (
    signal_date       TEXT NOT NULL,
    signal_family     TEXT NOT NULL,
    experiment_id     TEXT NOT NULL,
    headline_count    INTEGER NOT NULL,
    observation_count INTEGER NOT NULL,
    source_breadth    INTEGER NOT NULL,
    window_sessions   INTEGER NOT NULL,
    min_history       INTEGER NOT NULL,
    prior_mean        REAL,
    prior_std         REAL,
    prior_count       INTEGER NOT NULL,
    volume_z          REAL,
    volume_percentile REAL,
    change_1          REAL,
    change_5          REAL,
    change_20         REAL,
    method_version    TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (signal_date, signal_family, experiment_id, window_sessions, method_version)
);

-- Candidate event groups. These are ALGORITHMIC groupings, not verified
-- real-world events; review_state records how far that has been checked.
CREATE TABLE IF NOT EXISTS event_groups (
    group_key               TEXT NOT NULL,
    algorithm_version       TEXT NOT NULL,
    signal_family           TEXT,
    event_type              TEXT,
    primary_entity          TEXT,
    entity_ids              TEXT,
    headline_count          INTEGER NOT NULL,
    source_count            INTEGER NOT NULL,
    sources                 TEXT,
    first_seen_at           TEXT,
    last_seen_at            TEXT,
    unknown_timestamp_count INTEGER NOT NULL DEFAULT 0,
    signal_date             TEXT,
    signal_date_span        INTEGER,
    mean_sentiment          REAL,
    median_sentiment        REAL,
    sentiment_dispersion    REAL,
    cross_source_dispersion REAL,
    strong_positive_count   INTEGER,
    strong_negative_count   INTEGER,
    market_recap_count      INTEGER NOT NULL DEFAULT 0,
    is_singleton            INTEGER NOT NULL DEFAULT 0,
    is_single_source        INTEGER NOT NULL DEFAULT 0,
    novelty                 REAL,
    prior_entity_events     INTEGER,
    review_state            TEXT NOT NULL DEFAULT 'unreviewed',
    updated_at              TEXT NOT NULL,
    PRIMARY KEY (group_key, algorithm_version)
);
CREATE INDEX IF NOT EXISTS idx_event_groups_signal
    ON event_groups(signal_date, signal_family);

-- Every headline-to-event assignment with the evidence that produced it.
CREATE TABLE IF NOT EXISTS event_headline_map (
    group_key         TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    headline_id       INTEGER NOT NULL REFERENCES headlines(id),
    similarity        REAL,
    match_rule        TEXT,
    entity_overlap    TEXT,
    assigned_at       TEXT NOT NULL,
    PRIMARY KEY (group_key, algorithm_version, headline_id)
);
CREATE INDEX IF NOT EXISTS idx_event_headline_map_headline
    ON event_headline_map(headline_id);

CREATE TABLE IF NOT EXISTS event_group_entities (
    group_key         TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    entity_type       TEXT NOT NULL,
    entity_id         TEXT NOT NULL,
    rule_version      TEXT NOT NULL,
    PRIMARY KEY (group_key, algorithm_version, entity_type, entity_id)
);

-- Append-only record of manual split/merge review. A correction is a new row,
-- never an edit, so the grouping history stays readable after the fact.
CREATE TABLE IF NOT EXISTS event_group_audit (
    audit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    group_key         TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    action            TEXT NOT NULL,
    actor             TEXT NOT NULL,
    headline_ids      TEXT,
    target_group_key  TEXT,
    rationale         TEXT,
    performed_at      TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trg_event_group_audit_no_update
BEFORE UPDATE ON event_group_audit
BEGIN
    SELECT RAISE(ABORT, 'event_group_audit is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_event_group_audit_no_delete
BEFORE DELETE ON event_group_audit
BEGIN
    SELECT RAISE(ABORT, 'event_group_audit is append-only');
END;

-- Timing-safe market windows per candidate event.
CREATE TABLE IF NOT EXISTS event_return_windows (
    group_key           TEXT NOT NULL,
    algorithm_version   TEXT NOT NULL,
    window_name         TEXT NOT NULL,
    information_cutoff  TEXT,
    assumed_execution   TEXT,
    entry_price_field   TEXT,
    exit_price_field    TEXT,
    entry_date          TEXT,
    exit_date           TEXT,
    entry_price         REAL,
    exit_price          REAL,
    raw_return          REAL,
    is_available        INTEGER NOT NULL DEFAULT 0,
    unavailable_reason  TEXT,
    rule_version        TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    PRIMARY KEY (group_key, algorithm_version, window_name, rule_version)
);

-- Residual returns per (session, window, control set). kind separates tradable
-- controls from contemporaneous descriptive ones.
CREATE TABLE IF NOT EXISTS control_residual_returns (
    exit_date               TEXT NOT NULL,
    window_name             TEXT NOT NULL,
    control_set_id          TEXT NOT NULL,
    kind                    TEXT NOT NULL,
    raw_return              REAL,
    fitted                  REAL,
    residual                REAL,
    coefficients            TEXT,
    estimation_observations INTEGER,
    estimation_window_end   TEXT,
    version                 TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    PRIMARY KEY (exit_date, window_name, control_set_id, version)
);

-- The event-level research dataset. Ready for walk-forward evaluation; no
-- model, strategy or protocol is applied to it here.
CREATE TABLE IF NOT EXISTS event_research_dataset (
    group_key            TEXT NOT NULL,
    algorithm_version    TEXT NOT NULL,
    experiment_id        TEXT NOT NULL,
    window_name          TEXT NOT NULL,
    signal_date          TEXT,
    signal_family        TEXT,
    event_type           TEXT,
    primary_entity       TEXT,
    timing_bucket        TEXT,
    eligibility_status   TEXT NOT NULL,
    eligibility_reason   TEXT,
    headline_count       INTEGER,
    source_count         INTEGER,
    mean_sentiment       REAL,
    median_sentiment     REAL,
    sentiment_dispersion REAL,
    cross_source_dispersion REAL,
    novelty              REAL,
    market_recap_count   INTEGER,
    information_cutoff   TEXT,
    assumed_execution    TEXT,
    entry_date           TEXT,
    exit_date            TEXT,
    raw_return           REAL,
    residual_none        REAL,
    residual_em_lagged   REAL,
    residual_em_oil_fx_lagged REAL,
    residual_em_contemporaneous REAL,
    blocked_features     TEXT,
    dataset_version      TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    PRIMARY KEY (group_key, algorithm_version, window_name, experiment_id, dataset_version)
);
CREATE INDEX IF NOT EXISTS idx_event_research_signal
    ON event_research_dataset(signal_date, eligibility_status);

-- Reviewed reconstruction of legacy score provenance. Append-only by trigger:
-- an assignment and a later rollback are two rows, never an edit of one, so the
-- reconstruction history of any headline stays readable after the fact.
-- assigned_experiment_id is NULL on a rollback row.
CREATE TABLE IF NOT EXISTS experiment_assignment_audit (
    assignment_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_id            INTEGER NOT NULL REFERENCES headlines(id),
    assigned_experiment_id TEXT,
    assignment_method      TEXT NOT NULL,
    evidence               TEXT NOT NULL,
    reviewed_at            TEXT NOT NULL,
    migration_version      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiment_assignment_audit_headline
    ON experiment_assignment_audit(headline_id, assignment_id);

CREATE TRIGGER IF NOT EXISTS trg_experiment_assignment_audit_no_update
BEFORE UPDATE ON experiment_assignment_audit
BEGIN
    SELECT RAISE(ABORT, 'experiment_assignment_audit is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_experiment_assignment_audit_no_delete
BEFORE DELETE ON experiment_assignment_audit
BEGIN
    SELECT RAISE(ABORT, 'experiment_assignment_audit is append-only');
END;

-- The frozen modelling view: one independent row per (reactable session,
-- target window). Events sharing a session share one index return, so a
-- session-level row is the unit inference is entitled to count.
CREATE TABLE IF NOT EXISTS session_modelling_units (
    first_reactable_session TEXT NOT NULL,
    window_name             TEXT NOT NULL,
    modelling_unit          TEXT NOT NULL,
    modelling_unit_version  TEXT NOT NULL,
    event_count_raw         INTEGER NOT NULL,
    group_keys              TEXT,
    entry_date              TEXT,
    exit_date               TEXT,
    dominant_family         TEXT,
    dominant_timing_bucket  TEXT,
    raw_return              REAL,
    residual_none           REAL,
    residual_em_lagged      REAL,
    residual_em_oil_fx_lagged REAL,
    residual_em_contemporaneous REAL,
    features_json           TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    PRIMARY KEY (first_reactable_session, window_name, modelling_unit_version)
);

-- One row per frozen protocol. The hash is the freeze: a changed hash means a
-- different study, not a revision of this one.
CREATE TABLE IF NOT EXISTS validation_protocols (
    protocol_hash     TEXT PRIMARY KEY,
    protocol_version  TEXT NOT NULL,
    status            TEXT NOT NULL,
    specification_json TEXT NOT NULL,
    dataset_version   TEXT NOT NULL,
    feature_version   TEXT NOT NULL,
    target_version    TEXT NOT NULL,
    frozen_at         TEXT NOT NULL
);

-- One row per execution of a protocol, with the provenance needed to repeat it.
CREATE TABLE IF NOT EXISTS validation_runs (
    validation_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    protocol_hash     TEXT NOT NULL REFERENCES validation_protocols(protocol_hash),
    code_commit       TEXT,
    database_snapshot TEXT,
    experiment_id     TEXT,
    session_count     INTEGER,
    fold_count        INTEGER,
    specifications_run     INTEGER,
    specifications_blocked INTEGER,
    verdict           TEXT,
    counts_json       TEXT,
    started_at        TEXT NOT NULL,
    finished_at       TEXT
);

-- One row per (feature set, model, target). A blocked specification is recorded
-- with its binding requirement rather than omitted, so the reader sees what was
-- not attempted and why.
CREATE TABLE IF NOT EXISTS validation_results (
    validation_run_id INTEGER NOT NULL REFERENCES validation_runs(validation_run_id),
    feature_set       TEXT NOT NULL,
    model             TEXT NOT NULL,
    target            TEXT NOT NULL,
    kind              TEXT NOT NULL,
    status            TEXT NOT NULL,
    binding_requirement TEXT,
    rows_complete     INTEGER,
    usable_sessions   INTEGER,
    fitted_folds      INTEGER,
    mae               REAL,
    rmse              REAL,
    pearson_r         REAL,
    directional_accuracy REAL,
    balanced_accuracy REAL,
    brier_score       REAL,
    hit_lower         REAL,
    hit_upper         REAL,
    stability_json    TEXT,
    subgroups_json    TEXT,
    PRIMARY KEY (validation_run_id, feature_set, model, target)
);

-- Every out-of-sample prediction, so a reported metric can be recomputed.
CREATE TABLE IF NOT EXISTS validation_predictions (
    validation_run_id TEXT NOT NULL,
    feature_set       TEXT NOT NULL,
    model             TEXT NOT NULL,
    target            TEXT NOT NULL,
    fold              INTEGER NOT NULL,
    first_reactable_session TEXT NOT NULL,
    exit_date         TEXT,
    actual            REAL,
    predicted         REAL,
    probability       REAL,
    PRIMARY KEY (validation_run_id, feature_set, model, target,
                 first_reactable_session)
);

-- A completed study, sealed. Append-only by trigger: a frozen result that can
-- be updated is not frozen, and "we re-ran it and it worked" must never be
-- presentable as "it worked".
CREATE TABLE IF NOT EXISTS frozen_research_results (
    artifact_hash     TEXT PRIMARY KEY,
    artifact_version  TEXT NOT NULL,
    protocol_version  TEXT NOT NULL,
    protocol_hash     TEXT NOT NULL,
    code_commit       TEXT,
    database_snapshot TEXT,
    validation_run_id INTEGER,
    verdict           TEXT NOT NULL,
    conclusion        TEXT NOT NULL,
    independent_sessions INTEGER,
    specifications_run   INTEGER,
    specifications_blocked INTEGER,
    successes         INTEGER,
    artifact_json     TEXT NOT NULL,
    frozen_at         TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trg_frozen_research_results_no_update
BEFORE UPDATE ON frozen_research_results
BEGIN
    SELECT RAISE(ABORT, 'frozen_research_results is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_frozen_research_results_no_delete
BEFORE DELETE ON frozen_research_results
BEGIN
    SELECT RAISE(ABORT, 'frozen_research_results is append-only');
END;

-- The untouched future-validation contract. Append-only for the same reason:
-- a boundary that can move is not a boundary.
CREATE TABLE IF NOT EXISTS future_validation_definitions (
    definition_hash   TEXT PRIMARY KEY,
    version           TEXT NOT NULL,
    validation_start  TEXT NOT NULL,
    first_eligible_session TEXT NOT NULL,
    protocol_hash     TEXT NOT NULL,
    frozen_artifact_hash TEXT,
    minimum_sessions  INTEGER NOT NULL,
    minimum_horizon_days INTEGER NOT NULL,
    definition_json   TEXT NOT NULL,
    registered_at     TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trg_future_validation_definitions_no_update
BEFORE UPDATE ON future_validation_definitions
BEGIN
    SELECT RAISE(ABORT, 'future_validation_definitions is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_future_validation_definitions_no_delete
BEFORE DELETE ON future_validation_definitions
BEGIN
    SELECT RAISE(ABORT, 'future_validation_definitions is append-only');
END;

-- Readiness snapshots. Data-quality statistics only: no accuracy, no error, no
-- correlation. Peeking at performance while a sample accumulates and stopping
-- when it looks good is optional-stopping, and it leaves no trace in the
-- number it corrupts.
CREATE TABLE IF NOT EXISTS future_validation_readiness (
    observed_at       TEXT NOT NULL,
    definition_hash   TEXT NOT NULL,
    state             TEXT NOT NULL,
    untouched_sessions INTEGER NOT NULL,
    required_sessions  INTEGER NOT NULL,
    eligible_events    INTEGER NOT NULL,
    distinct_outcomes  INTEGER NOT NULL,
    elapsed_days       INTEGER NOT NULL,
    required_days      INTEGER NOT NULL,
    family_coverage_json TEXT,
    missingness_json     TEXT,
    control_availability_json TEXT,
    eligible_to_run    INTEGER NOT NULL DEFAULT 0,
    blocking_reasons   TEXT,
    PRIMARY KEY (observed_at, definition_hash)
);

-- Deterministic stratified sample for manual grouping review. Drawn without
-- any reference to market returns, so a reviewer cannot be nudged by outcome.
CREATE TABLE IF NOT EXISTS event_review_sample (
    sample_version    TEXT NOT NULL,
    stratum           TEXT NOT NULL,
    group_key         TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    comparison_group_key TEXT,
    similarity        REAL,
    rationale         TEXT,
    drawn_at          TEXT NOT NULL,
    PRIMARY KEY (sample_version, stratum, group_key, comparison_group_key)
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
    error_msg         TEXT,
    scrape_status      TEXT,
    scoring_status     TEXT,
    aggregation_status TEXT,
    market_data_status TEXT,
    audit_status       TEXT,
    warnings_json      TEXT NOT NULL DEFAULT '[]',
    errors_json        TEXT NOT NULL DEFAULT '[]'
);
"""

# Columns added after the initial schema (applied via ALTER TABLE at runtime).
# Tuple: (table_name, column_name, column_definition)
_MIGRATIONS: List[Tuple[str, str, str]] = [
    (
        "headlines", "processing_status",
        "TEXT NOT NULL DEFAULT 'pending' CHECK (processing_status IN "
        "('pending', 'scored', 'retry_pending', 'failed'))",
    ),
    (
        "headlines", "scoring_attempts",
        "INTEGER NOT NULL DEFAULT 0 CHECK (scoring_attempts >= 0)",
    ),
    ("headlines", "last_scoring_attempt_at", "TEXT"),
    ("headlines", "scoring_last_error",      "TEXT"),
    ("headlines", "score_components_kind",  "TEXT"),
    ("pipeline_runs", "scrape_status",      "TEXT"),
    ("pipeline_runs", "scoring_status",     "TEXT"),
    ("pipeline_runs", "aggregation_status", "TEXT"),
    ("pipeline_runs", "market_data_status", "TEXT"),
    ("pipeline_runs", "audit_status",       "TEXT"),
    ("pipeline_runs", "warnings_json",      "TEXT NOT NULL DEFAULT '[]'"),
    ("pipeline_runs", "errors_json",        "TEXT NOT NULL DEFAULT '[]'"),
    ("raw_headline_observations", "is_excluded",       "INTEGER NOT NULL DEFAULT 0"),
    ("raw_headline_observations", "published_timestamp", "TEXT"),
    ("raw_headline_observations", "timing_bucket",       "TEXT"),
    ("raw_headline_observations", "exclusion_reason",  "TEXT"),
    ("raw_headline_observations", "exclusion_rule",    "TEXT"),
    ("raw_headline_observations", "exclusion_version", "TEXT"),
    ("headline_exclusions", "restored_by_user", "INTEGER NOT NULL DEFAULT 0"),
    ("pipeline_runs", "experiment_id", "TEXT"),     # provenance (migration Phase 0)
    ("headlines", "category",       "TEXT"),
    ("headlines", "p_positive",     "REAL"),
    ("headlines", "p_neutral",      "REAL"),
    ("headlines", "p_negative",     "REAL"),
    ("headlines", "model_name",     "TEXT"),
    ("headlines", "experiment_id",  "TEXT"),
    ("headlines", "published_hour", "INTEGER"),  # Istanbul local hour (0-23), UTC+3
    ("headlines", "published_timestamp", "TEXT"),
    ("headlines", "timing_bucket",       "TEXT"),
    ("headlines", "session_rule_version", "TEXT"),
    ("headlines", "relevance",      "REAL"),     # LLM relevance grade 0.0-1.0 (NULL = ungraded -> 1.0)
    ("headlines", "signal_date",    "TEXT"),     # first trading session that can react (trading_calendar.signal_date)
    ("events",    "external_id",    "TEXT"),     # e.g. 'kap:1230800' — dedup for non-headline events
    # Daily-bar completeness (see price_bars.py). NULL means unclassified; the
    # backfill resolves historical rows from recorded run times.
    ("bist100_prices", "bar_status",       "TEXT"),
    ("bist100_prices", "bar_observed_at",  "TEXT"),
    ("bist100_prices", "bar_review_reason", "TEXT"),
    ("bist100_prices", "bar_rule_version", "TEXT"),
    # Phase A derived classification. The detailed `category` column is frozen;
    # these are derived beside it and carry their own versions.
    ("headlines", "signal_family",         "TEXT"),
    ("headlines", "signal_family_version", "TEXT"),
    ("headlines", "signal_family_rule",    "TEXT"),
    ("headlines", "signal_family_evidence", "TEXT"),
    ("headlines", "signal_family_ambiguous", "INTEGER NOT NULL DEFAULT 0"),
    ("headlines", "signal_family_review",  "TEXT"),
    ("headlines", "is_market_recap",       "INTEGER"),
    ("headlines", "market_recap_version",  "TEXT"),
    ("headlines", "market_recap_rule",     "TEXT"),
    ("headlines", "market_recap_evidence", "TEXT"),
    ("headlines", "market_recap_confidence", "REAL"),
    # Event timing (research/timing.py). first_reactable_session duplicates
    # signal_date deliberately: signal_date's meaning was implied by three call
    # sites that disagreed, and a field whose name states the semantics cannot
    # be misread the way the old one was.
    ("event_groups", "timing_bucket",             "TEXT"),
    ("event_groups", "first_reactable_session",   "TEXT"),
    ("event_groups", "first_reactable_at",        "TEXT"),
    ("event_groups", "event_information_cutoff",  "TEXT"),
    ("event_groups", "event_timing_rule_version", "TEXT"),
    ("event_groups", "governing_headline_id",     "INTEGER"),
    ("event_groups", "timing_conflict",           "INTEGER NOT NULL DEFAULT 0"),
    ("event_groups", "timing_conflict_reason",    "TEXT"),
    ("event_groups", "member_session_count",      "INTEGER"),
    # A window whose entry predates publication measures reaction but could not
    # have been traded. The flag keeps the two uses separable.
    ("event_return_windows", "is_tradable",         "INTEGER NOT NULL DEFAULT 1"),
    ("event_return_windows", "not_tradable_reason", "TEXT"),
    ("event_research_dataset", "first_reactable_session",   "TEXT"),
    ("event_research_dataset", "first_reactable_at",        "TEXT"),
    ("event_research_dataset", "event_information_cutoff",  "TEXT"),
    ("event_research_dataset", "event_timing_rule_version", "TEXT"),
    ("event_research_dataset", "timing_conflict",           "INTEGER NOT NULL DEFAULT 0"),
    ("event_research_dataset", "timing_conflict_reason",    "TEXT"),
    ("event_research_dataset", "is_tradable_window",        "INTEGER NOT NULL DEFAULT 1"),
    # Which side of the untouched-future boundary an observation falls on. The
    # two epochs are never pooled silently.
    ("event_research_dataset", "corpus_epoch", "TEXT"),
    ("session_modelling_units", "corpus_epoch", "TEXT"),
    # Market-factor provenance: where a number came from, when it was retrieved
    # and under which transformation. A backfilled series and a live-collected
    # one are the same data only if both can say that.
    ("market_factors", "source",            "TEXT"),
    ("market_factors", "retrieved_at",      "TEXT"),
    ("market_factors", "transform_version", "TEXT"),
    ("bist100_prices", "source",            "TEXT"),
    ("bist100_prices", "retrieved_at",      "TEXT"),
]


def _apply_migrations(con: sqlite3.Connection) -> None:
    """
    Add columns (and dependent indexes) introduced after the initial schema.
    Safe to run on a fresh DB (columns already exist) or an old one.
    """
    headline_columns_before = {
        row[1] for row in con.execute("PRAGMA table_info(headlines)")
    }
    processing_status_added = "processing_status" not in headline_columns_before

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
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_headlines_processing_status "
        "ON headlines(processing_status, scoring_attempts)"
    )

    # ALTER TABLE fills every legacy row with the new column's default. Do the
    # content-aware classification exactly once; later init_db() calls must not
    # overwrite explicit retry/failed/scored transitions.
    if processing_status_added:
        con.execute(
            """UPDATE headlines
               SET processing_status = CASE
                   WHEN sentiment_score IS NULL
                    AND sentiment_label IS NULL
                    AND p_positive IS NULL
                    AND p_neutral IS NULL
                    AND p_negative IS NULL
                    AND model_name IS NULL
                    AND scored_at IS NULL
                       THEN 'pending'
                   WHEN sentiment_score IS NOT NULL
                    AND sentiment_label IS NOT NULL
                    AND p_positive IS NOT NULL
                    AND p_neutral IS NOT NULL
                    AND p_negative IS NOT NULL
                    AND model_name IS NOT NULL
                    AND scored_at IS NOT NULL
                       THEN 'scored'
                   ELSE 'retry_pending'
               END,
               last_scoring_attempt_at = COALESCE(last_scoring_attempt_at, scored_at),
               score_components_kind = CASE
                   WHEN score_components_kind IS NOT NULL THEN score_components_kind
                   WHEN p_positive IS NULL AND p_neutral IS NULL AND p_negative IS NULL
                       THEN NULL
                   WHEN lower(COALESCE(model_name, '')) LIKE 'gpt-%'
                     OR lower(COALESCE(model_name, '')) LIKE 'openai:%'
                       THEN 'synthetic_compatibility'
                   WHEN lower(COALESCE(model_name, '')) LIKE '%xlm%'
                     OR lower(COALESCE(model_name, '')) LIKE '%roberta%'
                       THEN 'softmax_probability'
                   ELSE 'legacy_unknown'
               END"""
        )

# -- Connection helper ---------------------------------------------------------

@contextmanager
def _conn(db_path: str = DB_PATH):
    """Yield a connection that auto-commits on clean exit and rolls back on error."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    # Foreign-key enforcement is a per-connection SQLite setting and must be
    # enabled before a transaction begins.
    con.execute("PRAGMA foreign_keys=ON")
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

def _normalise_observation_value(value: Any) -> str:
    """Return a deterministic text representation used by observation keys."""
    if value is None:
        return ""
    if isinstance(value, date):
        value = value.isoformat()
    return " ".join(str(value).strip().split()).casefold()


def make_raw_observation_key(observation: Dict[str, Any]) -> str:
    """Build a stable, source-distinct key for a fetched headline item.

    A feed-native ``source_item_id`` is preferred, then a source-scoped URL,
    then normalized title plus publication date for URL-less feeds. Fetch time
    is deliberately excluded so replaying the same response does not append
    duplicate audit rows.
    """
    source = _normalise_observation_value(observation.get("source", "unknown"))
    source_item_id = _normalise_observation_value(observation.get("source_item_id"))
    if source_item_id:
        identity: List[Any] = [source, "source_item_id", source_item_id]
    elif observation.get("url"):
        identity = [
            source,
            "url",
            _normalise_observation_value(observation.get("url")),
        ]
    else:
        identity = [
            source,
            "title_date",
            _normalise_observation_value(observation.get("title")),
            _normalise_observation_value(observation.get("published_at")),
            _normalise_observation_value(observation.get("published_timestamp")),
        ]
    encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_raw_headline_observations(
    observations: Iterable[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> int:
    """Append source-level fetch observations without replay bloat.

    The audit table intentionally precedes canonical URL/title de-duplication.
    Re-recording the same source item is a no-op because its observation key is
    stable. Returns the number of newly appended audit rows.
    """
    now = _now_iso()
    rows: List[Tuple[Any, ...]] = []
    for observation in observations:
        title = str(observation.get("title") or "").strip()
        if not title:
            raise ValueError("raw headline observation requires a non-empty title")
        source = str(observation.get("source") or "unknown").strip() or "unknown"
        published_at = observation.get("published_at")
        if isinstance(published_at, date):
            published_at = published_at.isoformat()
        raw_payload = observation.get("raw_payload")
        if raw_payload is None:
            raw_payload = observation.get("raw_payload_json")
        if isinstance(raw_payload, str):
            raw_payload_json = raw_payload
        elif raw_payload is None:
            raw_payload_json = None
        else:
            raw_payload_json = json.dumps(
                raw_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        rows.append((
            make_raw_observation_key(observation),
            observation.get("headline_id"),
            source,
            title,
            observation.get("url") or None,
            published_at,
            observation.get("published_timestamp"),
            observation.get("published_hour"),
            observation.get("timing_bucket"),
            observation.get("observed_at") or now,
            raw_payload_json,
            1 if observation.get("is_excluded") else 0,
            observation.get("exclusion_reason") or None,
            observation.get("exclusion_rule") or None,
            observation.get("exclusion_version") or None,
        ))

    if not rows:
        return 0
    with _conn(db_path) as con:
        before = con.total_changes
        con.executemany(
            """INSERT OR IGNORE INTO raw_headline_observations
               (observation_key, headline_id, source, title, url, published_at,
                published_timestamp, published_hour, timing_bucket, observed_at,
                raw_payload_json, is_excluded,
                exclusion_reason, exclusion_rule, exclusion_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        inserted = con.total_changes - before
    logger.info("Recorded %d new raw headline observations", inserted)
    return inserted


def list_raw_headline_observations(
    db_path: str = DB_PATH,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return raw observation audit rows, optionally for one source."""
    query = "SELECT * FROM raw_headline_observations"
    params: List[Any] = []
    if source is not None:
        query += " WHERE source = ?"
        params.append(source)
    query += " ORDER BY observation_id"
    with _conn(db_path) as con:
        return [dict(row) for row in con.execute(query, params).fetchall()]


def _link_observations_and_apply_exclusions(
    con: sqlite3.Connection,
    observations: Sequence[Dict[str, Any]],
) -> None:
    """Link audit rows to canonical headlines and persist filter decisions."""
    from scraper import _normalise  # local import to avoid circular import

    canonical_rows = con.execute(
        "SELECT id, source, title, url, published_at FROM headlines"
    ).fetchall()
    by_url = {row["url"]: int(row["id"]) for row in canonical_rows if row["url"]}
    by_title_date = {
        (
            row["source"], _normalise(row["title"])[:80], row["published_at"]
        ): int(row["id"])
        for row in canonical_rows
    }
    now = _now_iso()
    for observation in observations:
        published_at = observation.get("published_at")
        if isinstance(published_at, date):
            published_at = published_at.isoformat()
        url = observation.get("url") or None
        headline_id = by_url.get(url) if url else None
        if headline_id is None:
            headline_id = by_title_date.get(
                (
                    str(observation.get("source") or "unknown"),
                    _normalise(str(observation.get("title") or ""))[:80],
                    published_at,
                )
            )
        if headline_id is None:
            continue

        con.execute(
            """UPDATE raw_headline_observations
               SET headline_id = COALESCE(headline_id, ?)
               WHERE observation_key = ?""",
            (headline_id, make_raw_observation_key(observation)),
        )
        if observation.get("is_excluded"):
            con.execute(
                """INSERT OR IGNORE INTO headline_exclusions
                   (headline_id, exclusion_reason, exclusion_rule,
                    exclusion_version, excluded_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    headline_id,
                    observation.get("exclusion_reason") or "source_filter",
                    observation.get("exclusion_rule") or None,
                    observation.get("exclusion_version") or None,
                    now,
                ),
            )

def insert_headlines(
    headlines: Iterable[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> int:
    """
    Bulk-insert headlines, silently skipping rows with duplicate URLs.
    Returns the number of *new* rows inserted.
    """
    from config import TRADING_CALENDAR_RULE_VERSION
    from trading_calendar import assign_trading_session  # local import: avoids cycles

    items = list(headlines)
    prepared_items: List[Dict[str, Any]] = []
    rows: List[Tuple] = []
    now = _now_iso()
    for h in items:
        pub = h.get("published_at")
        if isinstance(pub, date):
            pub = pub.isoformat()
        timestamp = h.get("published_timestamp")
        published_hour = h.get("published_hour")
        if timestamp is None and pub and published_hour is not None:
            timestamp = f"{pub}T{int(published_hour):02d}:00:00+03:00"

        assignment = None
        if timestamp is not None or pub:
            assignment = assign_trading_session(timestamp, pub)
        if assignment is not None:
            sig = assignment.signal_date
            timing_bucket = assignment.timing_bucket
            if assignment.published_at_istanbul is not None:
                normalized_timestamp = assignment.published_at_istanbul.isoformat()
                pub = assignment.published_at_istanbul.date().isoformat()
                published_hour = assignment.published_at_istanbul.hour
            else:
                normalized_timestamp = None
        else:
            sig = None
            timing_bucket = "unknown"
            normalized_timestamp = None

        prepared = dict(h)
        prepared.update({
            "published_at": pub,
            "published_timestamp": normalized_timestamp,
            "published_hour": published_hour,
            "timing_bucket": timing_bucket,
            "signal_date": sig,
        })
        prepared_items.append(prepared)
        rows.append((
            h.get("source", "unknown"),
            h["title"],
            h.get("url") or None,
            pub,
            normalized_timestamp,
            timing_bucket,
            TRADING_CALENDAR_RULE_VERSION,
            now,
            h.get("category") or None,
            published_hour,
            sig,
        ))

    if not rows:
        return 0

    # Preserve every source-distinct fetched item before applying the legacy
    # canonical table's global URL/title de-duplication rules.
    record_raw_headline_observations(prepared_items, db_path=db_path)

    # Cross-run dedup the URL UNIQUE constraint cannot provide:
    # NULL-url headlines never collide in SQLite (NULL != NULL), so feeds
    # without URLs (e.g. ntv_ekonomi) would re-insert the same items every
    # daily run. Two headlines are duplicates when their normalised title[:80]
    # AND published date match — recurring titles on later dates are allowed.
    from scraper import _normalise  # local import to avoid circular import
    with _conn(db_path) as con:
        existing = {
            (r[0], _normalise(r[1])[:80], r[2])
            for r in con.execute("SELECT source, title, published_at FROM headlines")
        }
    fresh = []
    seen = set(existing)
    for row in rows:
        identity = (row[0], _normalise(row[1])[:80], row[3])
        if identity in seen:
            continue
        seen.add(identity)
        fresh.append(row)
    if len(fresh) < len(rows):
        logger.info("Skipped %d cross-run duplicate title(s)", len(rows) - len(fresh))
    if not fresh:
        # A duplicate canonical headline may still be a new source observation
        # or carry a newly active exclusion decision.
        with _conn(db_path) as con:
            _link_observations_and_apply_exclusions(con, prepared_items)
        return 0

    with _conn(db_path) as con:
        before = con.execute("SELECT total_changes()").fetchone()[0]
        con.executemany(
            """INSERT OR IGNORE INTO headlines
               (source, title, url, published_at, published_timestamp,
                timing_bucket, session_rule_version, scraped_at, category,
                published_hour, signal_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            fresh,
        )
        inserted = con.execute("SELECT total_changes()").fetchone()[0] - before
        _link_observations_and_apply_exclusions(con, prepared_items)

    logger.info("Inserted %d new headlines (skipped %d duplicates)", inserted, len(fresh) - inserted)
    return inserted


def backfill_session_assignments(db_path: str = DB_PATH) -> int:
    """Version and refresh derived session metadata without changing raw fields."""
    from config import TRADING_CALENDAR_RULE_VERSION
    from trading_calendar import assign_trading_session

    with _conn(db_path) as con:
        rows = con.execute(
            """SELECT id, published_at, published_timestamp, published_hour
               FROM headlines
               WHERE published_at IS NOT NULL
                 AND COALESCE(session_rule_version, '') <> ?""",
            (TRADING_CALENDAR_RULE_VERSION,),
        ).fetchall()
        updates = []
        for row in rows:
            timestamp = row["published_timestamp"]
            if timestamp is None and row["published_hour"] is not None:
                timestamp = (
                    f"{row['published_at']}T{int(row['published_hour']):02d}:00:00+03:00"
                )
            assignment = assign_trading_session(timestamp, row["published_at"])
            updates.append((
                assignment.signal_date,
                assignment.timing_bucket,
                TRADING_CALENDAR_RULE_VERSION,
                row["id"],
            ))
        if updates:
            con.executemany(
                """UPDATE headlines
                   SET signal_date=?, timing_bucket=?, session_rule_version=?
                   WHERE id=?""",
                updates,
            )
    if rows:
        logger.info(
            "Backfilled %d session assignments with rule %s",
            len(rows), TRADING_CALENDAR_RULE_VERSION,
        )
    return len(rows)


def get_unscored_headlines(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return eligible pending/retry headlines, excluding active exclusions."""
    with _conn(db_path) as con:
        return pd.read_sql_query(
            """SELECT h.id, h.title, h.scoring_attempts, h.processing_status
               FROM headlines AS h
               WHERE h.processing_status IN ('pending', 'retry_pending')
                 AND NOT EXISTS (
                     SELECT 1 FROM headline_exclusions AS x
                     WHERE x.headline_id = h.id AND x.restored_at IS NULL
                 )
               ORDER BY h.id""",
            con,
        )


def _infer_score_components_kind(model_name: Optional[str]) -> str:
    """Return conservative metadata for legacy callers that omit component kind."""
    model = (model_name or "").casefold()
    if model.startswith("gpt-") or model.startswith("openai:"):
        return "synthetic_compatibility"
    if "xlm" in model or "roberta" in model:
        return "softmax_probability"
    return "legacy_unknown"


def _resolve_experiment_id(experiment_id: Optional[str]) -> str:
    """Return a non-empty experiment ID for a newly persisted score."""
    if experiment_id is None:
        from config import EXPERIMENT_ID
        experiment_id = EXPERIMENT_ID
    resolved = str(experiment_id).strip()
    if not resolved:
        raise ValueError("experiment_id must be a non-empty string")
    return resolved


def mark_scoring_success(
    headline_id: int,
    score: float,
    label: str,
    p_positive: Optional[float],
    p_neutral: Optional[float],
    p_negative: Optional[float],
    model_name: str,
    score_components_kind: Optional[str] = None,
    db_path: str = DB_PATH,
    *,
    experiment_id: Optional[str] = None,
) -> bool:
    """Atomically persist one successful scoring attempt.

    Returns ``False`` when the row was already marked scored, making a replayed
    completion idempotent. A missing headline raises ``KeyError``.
    """
    if label not in {"positive", "neutral", "negative"}:
        raise ValueError(f"unsupported sentiment label: {label!r}")
    kind = score_components_kind or _infer_score_components_kind(model_name)
    resolved_experiment_id = _resolve_experiment_id(experiment_id)
    now = _now_iso()
    with _conn(db_path) as con:
        row = con.execute(
            "SELECT processing_status FROM headlines WHERE id = ?",
            (headline_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"headline {headline_id} does not exist")
        if row["processing_status"] == "scored":
            return False
        con.execute(
            """UPDATE headlines
               SET sentiment_score=?, sentiment_label=?,
                   p_positive=?, p_neutral=?, p_negative=?,
                   model_name=?, experiment_id=?, scored_at=?, score_components_kind=?,
                   processing_status='scored',
                   scoring_attempts=scoring_attempts + 1,
                   last_scoring_attempt_at=?, scoring_last_error=NULL
               WHERE id=?""",
            (
                score, label, p_positive, p_neutral, p_negative,
                model_name, resolved_experiment_id, now, kind, now, headline_id,
            ),
        )
    return True


def mark_scoring_attempts_failed(
    headline_ids: Iterable[int],
    error: str,
    max_attempts: int,
    db_path: str = DB_PATH,
) -> Dict[int, str]:
    """Record one failed attempt for each candidate and return its new status.

    Sentiment output fields are deliberately absent from the UPDATE, so NULLs
    (and any diagnostically useful partial legacy values) are preserved. Rows
    already exhausted remain ``failed`` without inflating their attempt count.
    Scored rows are rejected because downgrading a committed result would hide
    valid data.
    """
    ids = list(dict.fromkeys(int(headline_id) for headline_id in headline_ids))
    if not ids:
        return {}
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    error_text = str(error).strip()
    if not error_text:
        raise ValueError("a non-empty scoring error is required")

    now = _now_iso()
    statuses: Dict[int, str] = {}
    with _conn(db_path) as con:
        placeholders = ",".join("?" for _ in ids)
        found = {
            int(row["id"]): row["processing_status"]
            for row in con.execute(
                f"SELECT id, processing_status FROM headlines WHERE id IN ({placeholders})",
                ids,
            )
        }
        missing = [headline_id for headline_id in ids if headline_id not in found]
        if missing:
            raise KeyError(f"headline(s) do not exist: {missing}")
        scored = [headline_id for headline_id, state in found.items() if state == "scored"]
        if scored:
            raise ValueError(f"cannot mark scored headline(s) failed: {scored}")

        active_ids = [
            headline_id for headline_id in ids if found[headline_id] != "failed"
        ]
        if active_ids:
            con.executemany(
                """UPDATE headlines
                   SET scoring_attempts = scoring_attempts + 1,
                       last_scoring_attempt_at = ?,
                       scoring_last_error = ?,
                       processing_status = CASE
                           WHEN scoring_attempts + 1 >= ? THEN 'failed'
                           ELSE 'retry_pending'
                       END
                   WHERE id = ?""",
                [(now, error_text, max_attempts, headline_id) for headline_id in active_ids],
            )
        for row in con.execute(
            f"SELECT id, processing_status FROM headlines WHERE id IN ({placeholders})",
            ids,
        ):
            statuses[int(row["id"])] = str(row["processing_status"])
    return statuses


def mark_scoring_attempt_failed(
    headline_id: int,
    error: str,
    max_attempts: int,
    db_path: str = DB_PATH,
) -> str:
    """Singular convenience wrapper for :func:`mark_scoring_attempts_failed`."""
    return mark_scoring_attempts_failed(
        [headline_id], error, max_attempts, db_path=db_path,
    )[headline_id]


def batch_update_sentiment(
    scores: Iterable[Tuple],
    db_path: str = DB_PATH,
    *,
    experiment_id: Optional[str] = None,
) -> None:
    """
    Update sentiment for multiple headlines in one transaction.

    Each element of ``scores`` may be the legacy 7-tuple:
        (sentiment_score, sentiment_label,
         p_positive, p_neutral, p_negative,
         model_name,
         headline_id)

    or the canonical 8-tuple, which inserts ``score_components_kind`` before
    ``headline_id``. The legacy shape remains supported for existing callers.
    """
    now = _now_iso()
    rows: List[Tuple[Any, ...]] = []
    for result in scores:
        if len(result) == 7:
            score, label, p_pos, p_neu, p_neg, model, hid = result
            kind = _infer_score_components_kind(model)
        elif len(result) == 8:
            score, label, p_pos, p_neu, p_neg, model, kind, hid = result
        else:
            raise ValueError("sentiment score rows must contain 7 or 8 values")
        if label not in {"positive", "neutral", "negative"}:
            raise ValueError(f"unsupported sentiment label: {label!r}")
        rows.append((
            score, label, p_pos, p_neu, p_neg, model, now,
            kind or _infer_score_components_kind(model), now, hid,
        ))
    if not rows:
        return
    resolved_experiment_id = _resolve_experiment_id(experiment_id)
    rows = [
        (*row[:6], resolved_experiment_id, *row[6:])
        for row in rows
    ]
    with _conn(db_path) as con:
        con.executemany(
            """UPDATE headlines
               SET sentiment_score=?, sentiment_label=?,
                   p_positive=?, p_neutral=?, p_negative=?,
                   model_name=?, experiment_id=?, scored_at=?, score_components_kind=?,
                   processing_status='scored',
                   scoring_attempts=scoring_attempts + 1,
                   last_scoring_attempt_at=?, scoring_last_error=NULL
               WHERE id=?""",
            rows,
        )
    logger.info("Updated sentiment for %d headlines", len(rows))


def get_eligible_experiment_ids(db_path: str = DB_PATH) -> List[str]:
    """Return distinct score-experiment identities eligible for aggregation.

    Experiment identity is never guessed. Legacy identity may be reconstructed
    only when stored evidence uniquely establishes it, and the reconstruction is
    recorded auditably -- see :func:`backfill_reviewed_legacy_experiment_id`.
    A row that has not been through that reviewed migration keeps NULL and is
    represented here by a clearly marked model-scoped legacy identity.
    """
    with _conn(db_path) as con:
        rows = con.execute(
            """SELECT DISTINCT
                      CASE
                          WHEN TRIM(COALESCE(h.experiment_id, '')) <> ''
                              THEN h.experiment_id
                          ELSE '[legacy-unassigned] model=' || h.model_name
                      END AS eligible_experiment_id
               FROM headlines AS h
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
                 )
               ORDER BY eligible_experiment_id"""
        ).fetchall()
    return [str(row["eligible_experiment_id"]) for row in rows]


# -- Reviewed legacy experiment provenance -------------------------------------

# A row qualifies only when every clause holds. Anything else is left NULL and
# keeps blocking aggregation, which is the safe direction: an unassigned row is
# visible, a wrongly assigned one is not.
_REVIEWED_LEGACY_ELIGIBLE_SQL = """
    sentiment_score IS NOT NULL
AND processing_status = 'scored'
AND model_name = :model_name
AND sentiment_label IS NOT NULL
AND scored_at IS NOT NULL
AND p_positive IS NOT NULL
AND p_neutral IS NOT NULL
AND p_negative IS NOT NULL
AND (score_components_kind IS NULL OR score_components_kind IN ({kinds}))
AND TRIM(COALESCE(experiment_id, '')) = ''
"""


def _reviewed_legacy_params() -> Tuple[str, Dict[str, Any]]:
    """Build the eligibility clause and its bound parameters from config."""

    from config import REVIEWED_LEGACY_COMPONENT_KINDS, REVIEWED_LEGACY_MODEL_NAME

    kinds = list(REVIEWED_LEGACY_COMPONENT_KINDS)
    placeholders = ", ".join(f":kind_{index}" for index in range(len(kinds)))
    params: Dict[str, Any] = {"model_name": REVIEWED_LEGACY_MODEL_NAME}
    for index, kind in enumerate(kinds):
        params[f"kind_{index}"] = kind
    return _REVIEWED_LEGACY_ELIGIBLE_SQL.format(kinds=placeholders), params


def survey_reviewed_legacy_candidates(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Classify unassigned scored rows without changing anything.

    ``blocked`` counts rows that still lack provenance after this migration
    would run. They are reported by reason so a conflicting scorer identity is
    an explicit finding rather than a silent omission.
    """

    from config import REVIEWED_LEGACY_EXPERIMENT_ID, REVIEWED_LEGACY_MODEL_NAME

    clause, params = _reviewed_legacy_params()
    with _conn(db_path) as con:
        eligible = int(
            con.execute(
                f"SELECT COUNT(*) FROM headlines WHERE {clause}", params
            ).fetchone()[0]
        )
        already_assigned = int(
            con.execute(
                "SELECT COUNT(*) FROM headlines "
                "WHERE TRIM(COALESCE(experiment_id, '')) <> ''"
            ).fetchone()[0]
        )
        blocked_rows = con.execute(
            f"""SELECT COALESCE(model_name, '<null>') AS model_name,
                       COUNT(*) AS n
                FROM headlines
                WHERE sentiment_score IS NOT NULL
                  AND TRIM(COALESCE(experiment_id, '')) = ''
                  AND NOT ({clause})
                GROUP BY model_name
                ORDER BY model_name""",
            params,
        ).fetchall()
    return {
        "reviewed_model_name": REVIEWED_LEGACY_MODEL_NAME,
        "reviewed_experiment_id": REVIEWED_LEGACY_EXPERIMENT_ID,
        "eligible": eligible,
        "already_assigned": already_assigned,
        "blocked": {row["model_name"]: int(row["n"]) for row in blocked_rows},
        "blocked_total": sum(int(row["n"]) for row in blocked_rows),
    }


def backfill_reviewed_legacy_experiment_id(
    db_path: str = DB_PATH,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Reconstruct experiment identity for reviewed legacy scores.

    Only rows whose stored evidence uniquely establishes the reviewed identity
    are touched: an exact ``model_name`` match, complete and consistent score
    components, and no existing ``experiment_id``. A non-NULL identity is never
    overwritten, and no score, label, timestamp, or model name is modified.

    Every assignment appends a row to ``experiment_assignment_audit`` recording
    the evidence relied on, so a reconstructed identity is always separable from
    one written at scoring time. Re-running assigns nothing further.
    """

    from config import (
        LEGACY_PROVENANCE_MIGRATION_VERSION,
        REVIEWED_LEGACY_ASSIGNMENT_METHOD,
        REVIEWED_LEGACY_EXPERIMENT_ID,
    )

    survey = survey_reviewed_legacy_candidates(db_path=db_path)
    clause, params = _reviewed_legacy_params()
    now = _now_iso()

    if dry_run:
        return {**survey, "assigned": 0, "dry_run": True, "reviewed_at": None}

    with _conn(db_path) as con:
        candidates = con.execute(
            f"""SELECT id, model_name, score_components_kind, scored_at
                FROM headlines WHERE {clause} ORDER BY id""",
            params,
        ).fetchall()
        if not candidates:
            return {**survey, "assigned": 0, "dry_run": False, "reviewed_at": now}

        audit_rows = []
        for row in candidates:
            evidence = json.dumps(
                {
                    "model_name": row["model_name"],
                    "score_components_kind": row["score_components_kind"],
                    "scored_at": row["scored_at"],
                    "rule": (
                        "exact model/prompt identity with complete score "
                        "components and no prior experiment_id"
                    ),
                },
                sort_keys=True,
            )
            audit_rows.append((
                int(row["id"]),
                REVIEWED_LEGACY_EXPERIMENT_ID,
                REVIEWED_LEGACY_ASSIGNMENT_METHOD,
                evidence,
                now,
                LEGACY_PROVENANCE_MIGRATION_VERSION,
            ))

        # The UPDATE repeats the eligibility clause so a row that stopped
        # qualifying between the SELECT and the write is not assigned anyway.
        con.execute(
            f"""UPDATE headlines
                SET experiment_id = :assigned
                WHERE {clause}""",
            {**params, "assigned": REVIEWED_LEGACY_EXPERIMENT_ID},
        )
        con.executemany(
            """INSERT INTO experiment_assignment_audit
               (headline_id, assigned_experiment_id, assignment_method,
                evidence, reviewed_at, migration_version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            audit_rows,
        )

    logger.info(
        "Reviewed legacy provenance: assigned %s to %d headline(s)",
        REVIEWED_LEGACY_EXPERIMENT_ID, len(audit_rows),
    )
    return {
        **survey,
        "assigned": len(audit_rows),
        "dry_run": False,
        "reviewed_at": now,
        "migration_version": LEGACY_PROVENANCE_MIGRATION_VERSION,
    }


def list_reviewed_legacy_assignments(
    db_path: str = DB_PATH,
    *,
    active_only: bool = True,
) -> List[Dict[str, Any]]:
    """Return the current reconstruction state per headline.

    ``active_only`` keeps headlines whose most recent audit entry is still an
    assignment, which is exactly the set a rollback may revert.
    """

    from config import (
        LEGACY_PROVENANCE_MIGRATION_VERSION,
        REVIEWED_LEGACY_ASSIGNMENT_METHOD,
    )

    with _conn(db_path) as con:
        rows = con.execute(
            """SELECT a.headline_id, a.assigned_experiment_id,
                      a.assignment_method, a.evidence, a.reviewed_at,
                      a.migration_version
               FROM experiment_assignment_audit AS a
               JOIN (SELECT headline_id, MAX(assignment_id) AS last_id
                     FROM experiment_assignment_audit
                     WHERE migration_version = ?
                     GROUP BY headline_id) AS latest
                 ON latest.last_id = a.assignment_id
               ORDER BY a.headline_id""",
            (LEGACY_PROVENANCE_MIGRATION_VERSION,),
        ).fetchall()
    entries = [dict(row) for row in rows]
    if active_only:
        entries = [
            entry for entry in entries
            if entry["assignment_method"] == REVIEWED_LEGACY_ASSIGNMENT_METHOD
        ]
    return entries


def rollback_reviewed_legacy_experiment_id(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Revert only the assignments this migration actually made.

    A headline is reverted when its latest audit entry is an assignment from
    this migration *and* its stored ``experiment_id`` still equals the value
    that migration wrote. A row changed since then is left alone and reported,
    so a rollback can never quietly discard newer provenance.
    """

    from config import (
        LEGACY_PROVENANCE_MIGRATION_VERSION,
        REVIEWED_LEGACY_ROLLBACK_METHOD,
    )

    active = list_reviewed_legacy_assignments(db_path=db_path, active_only=True)
    if not active:
        return {"reverted": 0, "skipped_diverged": 0, "reverted_at": None}

    now = _now_iso()
    reverted: List[int] = []
    diverged: List[int] = []
    with _conn(db_path) as con:
        for entry in active:
            headline_id = int(entry["headline_id"])
            row = con.execute(
                "SELECT experiment_id FROM headlines WHERE id = ?",
                (headline_id,),
            ).fetchone()
            if row is None:
                continue
            if row["experiment_id"] != entry["assigned_experiment_id"]:
                diverged.append(headline_id)
                continue
            reverted.append(headline_id)

        if reverted:
            con.executemany(
                "UPDATE headlines SET experiment_id = NULL WHERE id = ?",
                [(headline_id,) for headline_id in reverted],
            )
            con.executemany(
                """INSERT INTO experiment_assignment_audit
                   (headline_id, assigned_experiment_id, assignment_method,
                    evidence, reviewed_at, migration_version)
                   VALUES (?, NULL, ?, ?, ?, ?)""",
                [
                    (
                        headline_id,
                        REVIEWED_LEGACY_ROLLBACK_METHOD,
                        json.dumps(
                            {"rule": "reverted a reviewed legacy assignment"},
                            sort_keys=True,
                        ),
                        now,
                        LEGACY_PROVENANCE_MIGRATION_VERSION,
                    )
                    for headline_id in reverted
                ],
            )

    logger.info(
        "Reviewed legacy provenance rollback: reverted %d, left %d diverged",
        len(reverted), len(diverged),
    )
    return {
        "reverted": len(reverted),
        "skipped_diverged": len(diverged),
        "diverged_headline_ids": diverged[:20],
        "reverted_at": now if reverted else None,
    }


# -- Phase A derived classification --------------------------------------------

def classify_signal_families(
    db_path: str = DB_PATH,
    *,
    force: bool = False,
) -> Dict[str, int]:
    """Derive market-recap and signal-family labels for headlines.

    Both layers are derived from the frozen ``category`` and ``title``. Nothing
    here writes a sentiment score, label, category, model name or experiment
    identity -- a family is an interpretation laid beside the measurement, never
    a change to it.

    Only rows whose stored versions differ from the current ones are touched, so
    a repeat call is a no-op and a rule revision reclassifies exactly the rows
    that need it.
    """
    from taxonomy.market_recap import MARKET_RECAP_VERSION, classify_market_recap
    from taxonomy.signal_family import SIGNAL_FAMILY_VERSION, assign_signal_family

    with _conn(db_path) as con:
        if force:
            rows = con.execute(
                "SELECT id, title, category FROM headlines ORDER BY id"
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT id, title, category FROM headlines
                   WHERE COALESCE(signal_family_version, '') <> ?
                      OR COALESCE(market_recap_version, '') <> ?
                   ORDER BY id""",
                (SIGNAL_FAMILY_VERSION, MARKET_RECAP_VERSION),
            ).fetchall()

        counts = {"classified": 0, "market_recap": 0, "ambiguous": 0}
        updates = []
        for row in rows:
            recap = classify_market_recap(row["title"])
            family = assign_signal_family(
                row["category"], row["title"], is_market_recap=recap.is_market_recap
            )
            counts["classified"] += 1
            counts["market_recap"] += recap.as_int
            counts["ambiguous"] += 1 if family.ambiguous else 0
            updates.append((
                family.signal_family, family.signal_family_version, family.rule,
                family.evidence, 1 if family.ambiguous else 0, family.review_reason,
                recap.as_int, recap.version, recap.rule, recap.evidence,
                recap.confidence, row["id"],
            ))
        if updates:
            con.executemany(
                """UPDATE headlines
                   SET signal_family=?, signal_family_version=?,
                       signal_family_rule=?, signal_family_evidence=?,
                       signal_family_ambiguous=?, signal_family_review=?,
                       is_market_recap=?, market_recap_version=?,
                       market_recap_rule=?, market_recap_evidence=?,
                       market_recap_confidence=?
                   WHERE id=?""",
                updates,
            )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_headlines_signal_family "
            "ON headlines(signal_family, signal_date)"
        )
    logger.info(
        "Signal families: classified %d headline(s), %d market recap, %d ambiguous",
        counts["classified"], counts["market_recap"], counts["ambiguous"],
    )
    return counts


def get_classified_headlines(
    db_path: str = DB_PATH,
    *,
    eligible_only: bool = True,
) -> pd.DataFrame:
    """Load scored headlines with their derived family and recap fields."""

    exclusion_clause = (
        """AND NOT EXISTS (SELECT 1 FROM headline_exclusions AS x
                           WHERE x.headline_id = h.id AND x.restored_at IS NULL)"""
        if eligible_only else ""
    )
    with _conn(db_path) as con:
        return pd.read_sql_query(
            f"""SELECT h.id, h.source, h.title, h.published_at, h.published_timestamp,
                       h.signal_date, h.timing_bucket, h.category, h.relevance,
                       h.sentiment_score, h.sentiment_label, h.experiment_id,
                       h.signal_family, h.signal_family_version,
                       h.signal_family_rule, h.signal_family_ambiguous,
                       h.signal_family_review, h.is_market_recap,
                       h.market_recap_rule, h.market_recap_confidence,
                       e.event_id
                FROM headlines AS h
                LEFT JOIN events AS e ON e.headline_id = h.id
                WHERE h.processing_status = 'scored'
                  AND h.sentiment_score IS NOT NULL
                  AND h.signal_date IS NOT NULL
                  {exclusion_clause}
                ORDER BY h.signal_date, h.id""",
            con,
        )


def upsert_family_signals(rows: Iterable[Dict[str, Any]], db_path: str = DB_PATH) -> int:
    columns = [
        "signal_date", "signal_family", "experiment_id", "family_version",
        "simple_mean", "relevance_weighted", "median_sentiment", "min_sentiment",
        "max_sentiment", "sentiment_std", "headline_count", "source_count",
        "positive_share", "neutral_share", "negative_share", "avg_relevance",
        "market_recap_count", "unknown_timing_count", "excluded_count",
        "unresolved_count", "ambiguous_count", "sample_sufficiency", "updated_at",
    ]
    now = _now_iso()
    payload = [
        tuple(row.get(column, now if column == "updated_at" else None)
              for column in columns)
        for row in rows
    ]
    if not payload:
        return 0
    with _conn(db_path) as con:
        con.executemany(
            f"""INSERT OR REPLACE INTO daily_family_signals ({','.join(columns)})
                VALUES ({','.join('?' * len(columns))})""",
            payload,
        )
    return len(payload)


def _upsert_indicator(table: str, columns: List[str], rows, db_path: str) -> int:
    now = _now_iso()
    payload = [
        tuple(row.get(column, now if column == "updated_at" else None)
              for column in columns)
        for row in rows
    ]
    if not payload:
        return 0
    with _conn(db_path) as con:
        con.executemany(
            f"""INSERT OR REPLACE INTO {table} ({','.join(columns)})
                VALUES ({','.join('?' * len(columns))})""",
            payload,
        )
    return len(payload)


def upsert_abnormal_tone(rows, db_path: str = DB_PATH) -> int:
    return _upsert_indicator("abnormal_tone_daily", [
        "signal_date", "scope", "scope_key", "experiment_id", "window_sessions",
        "min_history", "observed_mean", "prior_mean", "prior_std", "prior_count",
        "abnormal_tone", "rolling_z", "rolling_percentile", "method_version",
        "updated_at",
    ], rows, db_path)


def upsert_disagreement(rows, db_path: str = DB_PATH) -> int:
    return _upsert_indicator("news_disagreement_daily", [
        "signal_date", "signal_family", "experiment_id", "headline_count",
        "source_count", "within_day_std", "cross_outlet_std", "max_minus_min",
        "strong_positive_share", "strong_negative_share", "sentiment_entropy",
        "camp_gap", "camp_gap_sources", "official_vs_media_gap",
        "min_sources_met", "method_version", "updated_at",
    ], rows, db_path)


def upsert_volume(rows, db_path: str = DB_PATH) -> int:
    return _upsert_indicator("news_volume_daily", [
        "signal_date", "signal_family", "experiment_id", "headline_count",
        "observation_count", "source_breadth", "window_sessions", "min_history",
        "prior_mean", "prior_std", "prior_count", "volume_z", "volume_percentile",
        "change_1", "change_5", "change_20", "method_version", "updated_at",
    ], rows, db_path)


# -- Candidate events ----------------------------------------------------------

def replace_event_groups(
    groups: List[Dict[str, Any]],
    mappings: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    *,
    algorithm_version: str,
    db_path: str = DB_PATH,
) -> Dict[str, int]:
    """Rebuild candidate groups for one algorithm version.

    Rows for *this* algorithm version are replaced so a rerun is idempotent;
    rows written by another version are untouched, and the append-only
    ``event_group_audit`` is never cleared, so manual review history survives a
    regrouping.
    """
    now = _now_iso()
    with _conn(db_path) as con:
        con.execute(
            "DELETE FROM event_groups WHERE algorithm_version = ?",
            (algorithm_version,),
        )
        con.execute(
            "DELETE FROM event_headline_map WHERE algorithm_version = ?",
            (algorithm_version,),
        )
        con.execute(
            "DELETE FROM event_group_entities WHERE algorithm_version = ?",
            (algorithm_version,),
        )
        columns = [
            "group_key", "algorithm_version", "signal_family", "event_type",
            "primary_entity", "entity_ids", "headline_count", "source_count",
            "sources", "first_seen_at", "last_seen_at",
            "unknown_timestamp_count", "signal_date", "signal_date_span",
            "mean_sentiment", "median_sentiment", "sentiment_dispersion",
            "cross_source_dispersion", "strong_positive_count",
            "strong_negative_count", "market_recap_count", "is_singleton",
            "is_single_source", "novelty", "prior_entity_events",
            "review_state", "timing_bucket", "first_reactable_session",
            "first_reactable_at", "event_information_cutoff",
            "event_timing_rule_version", "governing_headline_id",
            "timing_conflict", "timing_conflict_reason", "member_session_count",
            "updated_at",
        ]
        con.executemany(
            f"INSERT OR REPLACE INTO event_groups ({','.join(columns)}) "
            f"VALUES ({','.join('?' * len(columns))})",
            [
                tuple(row.get(column, now if column == "updated_at" else None)
                      for column in columns)
                for row in groups
            ],
        )
        con.executemany(
            """INSERT OR REPLACE INTO event_headline_map
               (group_key, algorithm_version, headline_id, similarity,
                match_rule, entity_overlap, assigned_at)
               VALUES (?,?,?,?,?,?,?)""",
            [
                (row["group_key"], algorithm_version, row["headline_id"],
                 row.get("similarity"), row.get("match_rule"),
                 row.get("entity_overlap"), now)
                for row in mappings
            ],
        )
        con.executemany(
            """INSERT OR REPLACE INTO event_group_entities
               (group_key, algorithm_version, entity_type, entity_id, rule_version)
               VALUES (?,?,?,?,?)""",
            [
                (row["group_key"], algorithm_version, row["entity_type"],
                 row["entity_id"], row["rule_version"])
                for row in entities
            ],
        )
    return {
        "groups": len(groups), "mappings": len(mappings), "entities": len(entities),
    }


def record_event_group_action(
    group_key: str,
    action: str,
    actor: str,
    *,
    algorithm_version: str,
    headline_ids: Optional[Sequence[int]] = None,
    target_group_key: Optional[str] = None,
    rationale: Optional[str] = None,
    db_path: str = DB_PATH,
) -> int:
    """Append one manual review action (split, merge, confirm, reject).

    Appending never rewrites the automatic grouping. The grouping stays as the
    algorithm produced it and the human judgement sits beside it, so the two can
    always be told apart.
    """
    if action not in ("split", "merge", "confirm", "reject", "annotate"):
        raise ValueError(f"unsupported review action: {action!r}")
    with _conn(db_path) as con:
        cursor = con.execute(
            """INSERT INTO event_group_audit
               (group_key, algorithm_version, action, actor, headline_ids,
                target_group_key, rationale, performed_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (group_key, algorithm_version, action, actor,
             ",".join(str(i) for i in headline_ids) if headline_ids else None,
             target_group_key, rationale, _now_iso()),
        )
        if action in ("confirm", "reject"):
            con.execute(
                "UPDATE event_groups SET review_state=? "
                "WHERE group_key=? AND algorithm_version=?",
                ("confirmed" if action == "confirm" else "rejected",
                 group_key, algorithm_version),
            )
    return int(cursor.lastrowid)


def list_event_group_audit(
    group_key: Optional[str] = None, db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    clause = " WHERE group_key = ?" if group_key else ""
    params = (group_key,) if group_key else ()
    with _conn(db_path) as con:
        return [
            dict(row) for row in con.execute(
                f"SELECT * FROM event_group_audit{clause} ORDER BY audit_id", params
            )
        ]


def upsert_event_return_windows(rows, db_path: str = DB_PATH) -> int:
    return _upsert_indicator("event_return_windows", [
        "group_key", "algorithm_version", "window_name", "information_cutoff",
        "assumed_execution", "entry_price_field", "exit_price_field",
        "entry_date", "exit_date", "entry_price", "exit_price", "raw_return",
        "is_available", "unavailable_reason", "is_tradable",
        "not_tradable_reason", "rule_version", "updated_at",
    ], rows, db_path)


def upsert_control_residuals(rows, db_path: str = DB_PATH) -> int:
    return _upsert_indicator("control_residual_returns", [
        "exit_date", "window_name", "control_set_id", "kind", "raw_return",
        "fitted", "residual", "coefficients", "estimation_observations",
        "estimation_window_end", "version", "updated_at",
    ], rows, db_path)


def upsert_event_dataset(rows, db_path: str = DB_PATH) -> int:
    return _upsert_indicator("event_research_dataset", [
        "group_key", "algorithm_version", "experiment_id", "window_name",
        "signal_date", "signal_family", "event_type", "primary_entity",
        "timing_bucket", "eligibility_status", "eligibility_reason",
        "headline_count", "source_count", "mean_sentiment", "median_sentiment",
        "sentiment_dispersion", "cross_source_dispersion", "novelty",
        "market_recap_count", "information_cutoff", "assumed_execution",
        "entry_date", "exit_date", "raw_return", "residual_none",
        "residual_em_lagged", "residual_em_oil_fx_lagged",
        "residual_em_contemporaneous", "blocked_features", "dataset_version",
        "first_reactable_session", "first_reactable_at",
        "event_information_cutoff", "event_timing_rule_version",
        "timing_conflict", "timing_conflict_reason", "is_tradable_window",
        "corpus_epoch", "updated_at",
    ], rows, db_path)


def freeze_research_result(
    artifact: Dict[str, Any], db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Seal a completed study. Re-freezing the same artifact is a no-op.

    Returns ``{"artifact_hash": ..., "already_frozen": bool}``. An attempt to
    store a *different* artifact under an existing hash cannot happen -- the
    hash is derived from the content -- and an attempt to modify a stored row
    is refused by trigger.
    """
    import json as _json

    from research.frozen_result import artifact_hash as _hash

    digest = artifact.get("artifact_hash") or _hash(artifact)
    if _hash(artifact) != digest:
        raise ValueError("artifact hash does not match its content")

    now = _now_iso()
    version = artifact["study"]["protocol_version"]
    protocol = artifact["study"]["protocol_hash"]
    with _conn(db_path) as con:
        existing = con.execute(
            "SELECT frozen_at FROM frozen_research_results WHERE artifact_hash = ?",
            (digest,),
        ).fetchone()
        if existing:
            return {"artifact_hash": digest, "already_frozen": True,
                    "frozen_at": existing[0]}

        # The hazard this guards: a later run under a revised protocol being
        # sealed under the *same* version name, so the record silently becomes
        # the newer, possibly better-looking study. A revised protocol is a new
        # study and needs a new version string.
        clash = con.execute(
            """SELECT protocol_hash FROM frozen_research_results
                WHERE protocol_version = ? AND protocol_hash <> ?""",
            (version, protocol),
        ).fetchone()
        if clash:
            raise ValueError(
                f"{version!r} is already frozen under protocol hash "
                f"{clash[0][:16]}...; this artifact carries {protocol[:16]}.... "
                "A revised protocol is a different study and needs its own "
                "version string."
            )
        con.execute(
            """INSERT INTO frozen_research_results
               (artifact_hash, artifact_version, protocol_version, protocol_hash,
                code_commit, database_snapshot, validation_run_id, verdict,
                conclusion, independent_sessions, specifications_run,
                specifications_blocked, successes, artifact_json, frozen_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                digest, artifact["artifact_version"],
                artifact["study"]["protocol_version"],
                artifact["study"]["protocol_hash"],
                artifact["provenance"].get("code_commit"),
                artifact["provenance"].get("database_snapshot"),
                artifact["provenance"].get("validation_run_id"),
                artifact["verdict"], artifact["conclusion"],
                (artifact.get("sample") or {}).get("distinct_sessions"),
                artifact.get("specifications_run"),
                artifact.get("specifications_blocked"),
                artifact.get("successes"),
                _json.dumps(artifact, sort_keys=True, ensure_ascii=False,
                            default=str),
                artifact.get("frozen_at") or now,
            ),
        )
    return {"artifact_hash": digest, "already_frozen": False, "frozen_at": now}


def list_frozen_results(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Every sealed study, oldest first."""
    with _conn(db_path) as con:
        return [
            dict(row) for row in con.execute(
                "SELECT * FROM frozen_research_results ORDER BY frozen_at"
            )
        ]


def register_future_validation(
    document: Dict[str, Any], db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Register the untouched-future contract. Idempotent by content hash."""
    import json as _json

    from research.future_validation import definition_hash

    digest = definition_hash(document)
    now = _now_iso()
    with _conn(db_path) as con:
        existing = con.execute(
            "SELECT registered_at FROM future_validation_definitions "
            "WHERE definition_hash = ?", (digest,),
        ).fetchone()
        if existing:
            return {"definition_hash": digest, "already_registered": True,
                    "registered_at": existing[0]}
        con.execute(
            """INSERT INTO future_validation_definitions
               (definition_hash, version, validation_start,
                first_eligible_session, protocol_hash, frozen_artifact_hash,
                minimum_sessions, minimum_horizon_days, definition_json,
                registered_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                digest, document["version"], document["validation_start"],
                document["first_eligible_session"], document["protocol_hash"],
                document.get("frozen_retrospective_artifact_hash"),
                document["sample_size_requirement"]["minimum_sessions"],
                document["minimum_evaluation_horizon_days"],
                _json.dumps(document, sort_keys=True, ensure_ascii=False,
                            default=str),
                now,
            ),
        )
    return {"definition_hash": digest, "already_registered": False,
            "registered_at": now}


def record_future_readiness(
    report: Dict[str, Any], db_path: str = DB_PATH,
) -> int:
    """Store one readiness snapshot.

    The report carries counts and coverage only. Nothing here accepts an
    accuracy, an error or a correlation, so a caller cannot use this table to
    watch performance accumulate.
    """
    import json as _json

    forbidden = {
        "mae", "rmse", "accuracy", "directional_accuracy", "balanced_accuracy",
        "pearson_r", "correlation", "brier_score", "residual_mean", "verdict",
        "predicted", "actual",
    }
    leaked = forbidden & set(report)
    if leaked:
        raise ValueError(
            f"readiness reports must not carry outcome statistics: {sorted(leaked)}"
        )

    with _conn(db_path) as con:
        con.execute(
            """INSERT OR REPLACE INTO future_validation_readiness
               (observed_at, definition_hash, state, untouched_sessions,
                required_sessions, eligible_events, distinct_outcomes,
                elapsed_days, required_days, family_coverage_json,
                missingness_json, control_availability_json, eligible_to_run,
                blocking_reasons)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                report["observed_at"], report["definition_hash"],
                report["state"], report["untouched_sessions"],
                report["required_sessions"], report["eligible_events"],
                report["distinct_outcomes"], report["elapsed_days"],
                report["required_days"],
                _json.dumps(report.get("family_coverage") or {}, sort_keys=True),
                _json.dumps(report.get("missingness") or {}, sort_keys=True),
                _json.dumps(report.get("control_availability") or {}, sort_keys=True),
                1 if report.get("eligible_to_run") else 0,
                ";".join(report.get("blocking_reasons") or []) or None,
            ),
        )
    return 1


def replace_event_review_sample(
    rows: Sequence[Dict[str, Any]],
    *,
    sample_version: str,
    db_path: str = DB_PATH,
) -> int:
    """Rebuild the manual-review draw for one sample version."""
    now = _now_iso()
    with _conn(db_path) as con:
        con.execute(
            "DELETE FROM event_review_sample WHERE sample_version = ?",
            (sample_version,),
        )
        con.executemany(
            """INSERT OR REPLACE INTO event_review_sample
               (sample_version, stratum, group_key, algorithm_version,
                comparison_group_key, similarity, rationale, drawn_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (sample_version, row["stratum"], row["group_key"],
                 row["algorithm_version"], row.get("comparison_group_key"),
                 row.get("similarity"), row.get("rationale"), now)
                for row in rows
            ],
        )
    return len(rows)


def prune_superseded_event_derivations(
    *,
    rule_version: str,
    dataset_version: str,
    db_path: str = DB_PATH,
) -> Dict[str, int]:
    """Delete derived rows written under a superseded rule or dataset version.

    These tables are *derived* — every row can be rebuilt from headlines and
    price bars, neither of which is touched here. Keeping the old rows would be
    worse than deleting them: the v1 windows are not merely stale, they are
    known to be one session late, and a reader filtering by group key rather
    than by version would silently get the wrong number.

    Raw observations, scores, categories and the append-only audit tables are
    never in scope.
    """
    with _conn(db_path) as con:
        windows = con.execute(
            "DELETE FROM event_return_windows WHERE rule_version <> ?",
            (rule_version,),
        ).rowcount
        dataset = con.execute(
            "DELETE FROM event_research_dataset WHERE dataset_version <> ?",
            (dataset_version,),
        ).rowcount
    return {"windows_removed": max(0, windows), "dataset_rows_removed": max(0, dataset)}


def replace_session_modelling_units(
    units: Sequence[Dict[str, Any]],
    *,
    modelling_unit_version: str,
    db_path: str = DB_PATH,
) -> int:
    """Rebuild the frozen modelling view for one unit version.

    Rows for this version are replaced wholesale so a rerun is idempotent;
    another version's rows are untouched, which is what lets a superseded unit
    definition stay readable beside its replacement.
    """
    import json as _json

    now = _now_iso()
    feature_columns = {
        "first_reactable_session", "window_name", "modelling_unit",
        "modelling_unit_version", "event_count_raw", "group_keys",
        "entry_date", "exit_date", "dominant_family", "dominant_timing_bucket",
        "raw_return", "residual_none", "residual_em_lagged",
        "residual_em_oil_fx_lagged", "residual_em_contemporaneous",
        "corpus_epoch",
    }
    columns = [
        "first_reactable_session", "window_name", "modelling_unit",
        "modelling_unit_version", "event_count_raw", "group_keys",
        "entry_date", "exit_date", "dominant_family", "dominant_timing_bucket",
        "raw_return", "residual_none", "residual_em_lagged",
        "residual_em_oil_fx_lagged", "residual_em_contemporaneous",
        "corpus_epoch", "features_json", "updated_at",
    ]
    with _conn(db_path) as con:
        con.execute(
            "DELETE FROM session_modelling_units WHERE modelling_unit_version = ?",
            (modelling_unit_version,),
        )
        con.executemany(
            f"INSERT OR REPLACE INTO session_modelling_units "
            f"({','.join(columns)}) VALUES ({','.join('?' * len(columns))})",
            [
                tuple(
                    now if column == "updated_at"
                    else _json.dumps(
                        {k: v for k, v in unit.items() if k not in feature_columns},
                        sort_keys=True, default=str,
                    ) if column == "features_json"
                    else unit.get(column)
                    for column in columns
                )
                for unit in units
            ],
        )
    return len(units)


def record_validation_protocol(
    document: Dict[str, Any], db_path: str = DB_PATH,
) -> str:
    """Store a frozen protocol, keyed by its hash. Re-freezing is a no-op."""
    import json as _json

    specification = document["specification"]
    with _conn(db_path) as con:
        con.execute(
            """INSERT OR IGNORE INTO validation_protocols
               (protocol_hash, protocol_version, status, specification_json,
                dataset_version, feature_version, target_version, frozen_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                document["protocol_hash"],
                specification["protocol_version"],
                specification["status"],
                _json.dumps(specification, sort_keys=True, ensure_ascii=False),
                document["provenance"]["dataset_version"],
                document["provenance"]["feature_version"],
                document["provenance"]["target_version"],
                document["provenance"].get("generated_at") or _now_iso(),
            ),
        )
    return str(document["protocol_hash"])


def record_validation_run(
    *,
    protocol_hash: str,
    code_commit: Optional[str],
    database_snapshot: Optional[str],
    experiment_id: Optional[str],
    session_count: int,
    fold_count: int,
    specifications: Sequence[Dict[str, Any]],
    comparison: Dict[str, Any],
    counts: Dict[str, Any],
    started_at: str,
    db_path: str = DB_PATH,
) -> int:
    """Persist one protocol execution with all its specifications."""
    import json as _json

    with _conn(db_path) as con:
        cursor = con.execute(
            """INSERT INTO validation_runs
               (protocol_hash, code_commit, database_snapshot, experiment_id,
                session_count, fold_count, specifications_run,
                specifications_blocked, verdict, counts_json, started_at,
                finished_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                protocol_hash, code_commit, database_snapshot, experiment_id,
                session_count, fold_count,
                comparison.get("specifications_run", 0),
                comparison.get("specifications_blocked", 0),
                comparison.get("verdict"),
                _json.dumps(counts, sort_keys=True, default=str),
                started_at, _now_iso(),
            ),
        )
        run_id = int(cursor.lastrowid)

        for specification in specifications:
            pooled = specification.get("pooled") or {}
            interval = pooled.get("directional_hit_interval") or {}
            con.execute(
                """INSERT OR REPLACE INTO validation_results
                   (validation_run_id, feature_set, model, target, kind, status,
                    binding_requirement, rows_complete, usable_sessions,
                    fitted_folds, mae, rmse, pearson_r, directional_accuracy,
                    balanced_accuracy, brier_score, hit_lower, hit_upper,
                    stability_json, subgroups_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, specification["feature_set"], specification["model"],
                    specification["target"], specification["kind"],
                    specification["status"],
                    specification["gate"].get("binding_requirement"),
                    specification["gate"].get("rows_complete"),
                    specification["gate"].get("usable_sessions"),
                    (specification.get("stability") or {}).get("fitted_folds"),
                    pooled.get("mae"), pooled.get("rmse"),
                    pooled.get("pearson_r"), pooled.get("directional_accuracy"),
                    pooled.get("balanced_accuracy"), pooled.get("brier_score"),
                    interval.get("lower"), interval.get("upper"),
                    _json.dumps(specification.get("stability") or {}, default=str),
                    _json.dumps(specification.get("subgroups") or {}, default=str),
                ),
            )
            con.executemany(
                """INSERT OR REPLACE INTO validation_predictions
                   (validation_run_id, feature_set, model, target, fold,
                    first_reactable_session, exit_date, actual, predicted,
                    probability)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [
                    (str(run_id), specification["feature_set"],
                     specification["model"], specification["target"],
                     prediction["fold"], prediction["first_reactable_session"],
                     prediction.get("exit_date"), prediction.get("actual"),
                     prediction.get("predicted"), prediction.get("probability"))
                    for prediction in specification.get("predictions") or []
                ],
            )
    return run_id


def read_table(table: str, db_path: str = DB_PATH, order_by: str = "") -> pd.DataFrame:
    """Read one analytical table; used by the dashboard and regime report."""

    clause = f" ORDER BY {order_by}" if order_by else ""
    with _conn(db_path) as con:
        return pd.read_sql_query(f"SELECT * FROM {table}{clause}", con)


def relabel_from_probs(
    pos_threshold: float,
    neg_threshold: float,
    db_path: str = DB_PATH,
) -> int:
    """
    Recompute sentiment_label from the stored backend-specific score components.
    For the LLM backend, p_positive/p_neutral/p_negative are synthetic
    compatibility values rather than calibrated probabilities. No inference is
    performed here.

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
                "%d scored rows have no stored score components — "
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
        # A new relevance judgment supersedes a prior manual override of the
        # old LLM-rule decision. Merely rebuilding aggregates does not.
        con.executemany(
            """UPDATE headline_exclusions
               SET restored_by_user=0
               WHERE headline_id=? AND exclusion_rule='llm_relevance'
                 AND restored_at IS NOT NULL AND restored_by_user=1""",
            [(headline_id,) for _, headline_id in pairs],
        )
    logger.info("Updated relevance on %d headlines", len(pairs))


def reconcile_relevance_exclusions(
    headline_ids: Optional[Iterable[int]] = None,
    db_path: str = DB_PATH,
) -> Dict[str, int]:
    """Apply the current LLM relevance rule as reversible metadata.

    Rows below the configured threshold receive an active exclusion. If a later
    model/rule version grades the row above the threshold, only the exclusion
    created by this LLM rule is restored; unrelated editorial/keyword
    exclusions are never removed.
    """
    from config import LLM_RELEVANCE_RULE_VERSION, RELEVANCE_MIN_FOR_AGGREGATION

    ids = None
    if headline_ids is not None:
        ids = list(dict.fromkeys(int(headline_id) for headline_id in headline_ids))
        if not ids:
            return {"excluded": 0, "restored": 0}

    where = "WHERE h.relevance IS NOT NULL"
    params: List[Any] = []
    if ids is not None:
        placeholders = ",".join("?" for _ in ids)
        where += f" AND h.id IN ({placeholders})"
        params.extend(ids)

    now = _now_iso()
    excluded = 0
    restored = 0
    with _conn(db_path) as con:
        rows = con.execute(
            f"""SELECT h.id, h.relevance,
                       x.exclusion_id, x.exclusion_rule,
                       EXISTS (
                           SELECT 1 FROM headline_exclusions AS restored
                           WHERE restored.headline_id=h.id
                             AND restored.exclusion_rule='llm_relevance'
                             AND restored.exclusion_version=?
                             AND restored.restored_at IS NOT NULL
                             AND restored.restored_by_user=1
                       ) AS has_manual_restore_override
                FROM headlines AS h
                LEFT JOIN headline_exclusions AS x
                  ON x.headline_id = h.id AND x.restored_at IS NULL
                {where}""",
            [LLM_RELEVANCE_RULE_VERSION, *params],
        ).fetchall()
        for row in rows:
            below = float(row["relevance"]) < RELEVANCE_MIN_FOR_AGGREGATION
            if (
                below
                and row["exclusion_id"] is None
                and not row["has_manual_restore_override"]
            ):
                cur = con.execute(
                    """INSERT OR IGNORE INTO headline_exclusions
                       (headline_id, exclusion_reason, exclusion_rule,
                        exclusion_version, excluded_at)
                       VALUES (?, 'low_relevance', 'llm_relevance', ?, ?)""",
                    (row["id"], LLM_RELEVANCE_RULE_VERSION, now),
                )
                excluded += max(0, cur.rowcount)
            elif (
                not below
                and row["exclusion_id"] is not None
                and row["exclusion_rule"] == "llm_relevance"
            ):
                cur = con.execute(
                    "UPDATE headline_exclusions SET restored_at=? WHERE exclusion_id=?",
                    (now, row["exclusion_id"]),
                )
                restored += max(0, cur.rowcount)
    return {"excluded": excluded, "restored": restored}


def exclude_headline(
    headline_id: int,
    reason: str,
    rule: Optional[str] = None,
    version: Optional[str] = None,
    db_path: str = DB_PATH,
) -> bool:
    """Append an active exclusion unless one already exists for the headline."""
    reason = str(reason).strip()
    if not reason:
        raise ValueError("an exclusion reason is required")
    with _conn(db_path) as con:
        exists = con.execute(
            "SELECT 1 FROM headlines WHERE id = ?", (headline_id,)
        ).fetchone()
        if exists is None:
            raise KeyError(f"headline {headline_id} does not exist")
        cur = con.execute(
            """INSERT OR IGNORE INTO headline_exclusions
               (headline_id, exclusion_reason, exclusion_rule,
                exclusion_version, excluded_at)
               VALUES (?, ?, ?, ?, ?)""",
            (headline_id, reason, rule, version, _now_iso()),
        )
        return cur.rowcount == 1


def exclude_headlines(
    headline_ids: Iterable[int],
    reason: str,
    rule: Optional[str] = None,
    version: Optional[str] = None,
    db_path: str = DB_PATH,
) -> int:
    """Idempotently exclude several headlines and return newly active count."""
    ids = list(dict.fromkeys(int(headline_id) for headline_id in headline_ids))
    if not ids:
        return 0
    reason = str(reason).strip()
    if not reason:
        raise ValueError("an exclusion reason is required")
    with _conn(db_path) as con:
        placeholders = ",".join("?" for _ in ids)
        found = {
            int(row[0]) for row in con.execute(
                f"SELECT id FROM headlines WHERE id IN ({placeholders})", ids
            )
        }
        missing = [headline_id for headline_id in ids if headline_id not in found]
        if missing:
            raise KeyError(f"headline(s) do not exist: {missing}")
        before = con.total_changes
        con.executemany(
            """INSERT OR IGNORE INTO headline_exclusions
               (headline_id, exclusion_reason, exclusion_rule,
                exclusion_version, excluded_at)
               VALUES (?, ?, ?, ?, ?)""",
            [(headline_id, reason, rule, version, _now_iso()) for headline_id in ids],
        )
        return con.total_changes - before


def restore_headline_exclusion(
    headline_id: int,
    db_path: str = DB_PATH,
) -> bool:
    """Restore the active exclusion while retaining its history row."""
    with _conn(db_path) as con:
        cur = con.execute(
            """UPDATE headline_exclusions
               SET restored_at = ?, restored_by_user = 1
               WHERE headline_id = ? AND restored_at IS NULL""",
            (_now_iso(), headline_id),
        )
        return cur.rowcount == 1


def restore_headline(headline_id: int, db_path: str = DB_PATH) -> bool:
    """Backward-friendly alias for :func:`restore_headline_exclusion`."""
    return restore_headline_exclusion(headline_id, db_path=db_path)


def count_active_headline_exclusions(db_path: str = DB_PATH) -> int:
    """Return the number of headlines with an active exclusion."""
    with _conn(db_path) as con:
        return int(con.execute(
            "SELECT COUNT(*) FROM headline_exclusions WHERE restored_at IS NULL"
        ).fetchone()[0])


def count_excluded_headlines(db_path: str = DB_PATH) -> int:
    """Alias for callers that use the shorter exclusion count name."""
    return count_active_headline_exclusions(db_path=db_path)


def list_headline_exclusions(
    db_path: str = DB_PATH,
    active_only: bool = True,
    headline_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """List active exclusions by default, or the complete restoration history."""
    where: List[str] = []
    params: List[Any] = []
    if active_only:
        where.append("x.restored_at IS NULL")
    if headline_id is not None:
        where.append("x.headline_id = ?")
        params.append(headline_id)
    clause = " WHERE " + " AND ".join(where) if where else ""
    with _conn(db_path) as con:
        rows = con.execute(
            """SELECT x.*, h.source, h.title, h.url
               FROM headline_exclusions AS x
               JOIN headlines AS h ON h.id = x.headline_id"""
            + clause
            + " ORDER BY x.exclusion_id",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def delete_headlines(
    ids: Sequence[int],
    db_path: str = DB_PATH,
    *,
    confirm: bool = False,
) -> int:
    """Permanently delete headlines only with explicit confirmation."""
    if confirm is not True:
        raise PermissionError(
            "permanent headline deletion requires confirm=True; "
            "use exclude_headline() for reversible filtering"
        )
    ids = list(ids)
    if not ids:
        return 0
    deleted = 0
    with _conn(db_path) as con:
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            cur = con.execute(
                f"DELETE FROM headlines WHERE id IN ({','.join('?' * len(chunk))})", chunk
            )
            deleted += cur.rowcount
    logger.info("Permanently deleted %d headlines", deleted)
    return deleted


def permanently_delete_headlines(
    ids: Sequence[int],
    db_path: str = DB_PATH,
    *,
    confirm: bool = False,
) -> int:
    """Explicitly named alias for the guarded permanent-purge operation."""
    return delete_headlines(ids, db_path=db_path, confirm=confirm)


# -- BIST 100 prices -----------------------------------------------------------

def upsert_prices(
    df: pd.DataFrame,
    db_path: str = DB_PATH,
    *,
    observed_at: Optional[str] = None,
    mark_corrected: bool = False,
) -> Dict[str, int]:
    """
    Upsert a price DataFrame into bist100_prices.
    ``df`` must have columns: date, open, high, low, close, volume, daily_return
    (all strings / floats - no DatetimeIndex).

    Each bar is classified for completeness before it is written. A bar
    observed while its session was still open is stored as provisional, and a
    provisional refetch never demotes an already-settled bar: completeness only
    moves forward. ``mark_corrected`` records that a settled bar deliberately
    replaced a provisional or invalid one.

    Returns counts by outcome so a caller can report what actually changed.
    """
    from config import PRICE_BAR_RULE_VERSION
    from price_bars import classify_price_bar, resolve_bar_status

    observed = observed_at or _now_iso()
    counts = {"written": 0, "skipped_would_demote": 0, "provisional": 0,
              "complete": 0, "provider_invalid": 0, "corrected": 0}

    with _conn(db_path) as con:
        existing = {
            row["date"]: row["bar_status"]
            for row in con.execute("SELECT date, bar_status FROM bist100_prices")
        }
        payload = []
        for row in df.itertuples(index=False):
            classification = classify_price_bar(
                row.date, volume=row.volume, observed_at=observed
            )
            write, status = resolve_bar_status(
                existing.get(row.date),
                classification.status,
                explicit_correction=mark_corrected,
            )
            if not write:
                counts["skipped_would_demote"] += 1
                continue
            counts[status] = counts.get(status, 0) + 1
            counts["written"] += 1
            payload.append((
                row.date, row.open, row.high, row.low, row.close, row.volume,
                row.daily_return, status, observed,
                classification.review_reason, PRICE_BAR_RULE_VERSION,
            ))

        if payload:
            con.executemany(
                """INSERT OR REPLACE INTO bist100_prices
                   (date, open, high, low, close, volume, daily_return,
                    bar_status, bar_observed_at, bar_review_reason,
                    bar_rule_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                payload,
            )
    logger.info(
        "Upserted %d price rows (%d skipped to avoid demoting a settled bar)",
        counts["written"], counts["skipped_would_demote"],
    )
    # A fetch window's first row has no predecessor inside that window, so the
    # provider-derived return is NULL there and would overwrite a valid stored
    # value. Rebuild the series from the complete stored history instead.
    counts["returns_recomputed"] = recompute_daily_returns(db_path=db_path)
    return counts


def recompute_daily_returns(db_path: str = DB_PATH) -> int:
    """Rebuild ``daily_return`` from the full ordered series of settled bars.

    Returns are chained across the stored history rather than within whatever
    window a fetch happened to download, so a boundary row keeps the return to
    its true preceding session. Only complete and corrected bars form the
    series: a provisional bar is an intraday snapshot, and chaining a return
    through one would corrupt both its neighbours. Provisional and invalid rows
    therefore hold NULL until they settle.

    The earliest stored session keeps NULL because its predecessor is genuinely
    unavailable, not because of a window edge.

    Returns the number of rows whose stored return changed.
    """
    from price_bars import ANALYSABLE_STATUSES

    placeholders = ",".join("?" * len(ANALYSABLE_STATUSES))
    with _conn(db_path) as con:
        rows = con.execute(
            f"""SELECT date, close, daily_return FROM bist100_prices
                WHERE bar_status IN ({placeholders})
                ORDER BY date""",
            ANALYSABLE_STATUSES,
        ).fetchall()

        updates: List[Tuple[Any, str]] = []
        previous_close: Optional[float] = None
        for row in rows:
            close = row["close"]
            if previous_close in (None, 0) or close is None:
                expected = None
            else:
                expected = (float(close) / float(previous_close) - 1.0) * 100.0
            stored = row["daily_return"]
            differs = (
                (stored is None) != (expected is None)
                or (
                    stored is not None
                    and expected is not None
                    and abs(float(stored) - expected) > 1e-9
                )
            )
            if differs:
                updates.append((expected, row["date"]))
            if close is not None:
                previous_close = float(close)

        # A bar outside the settled series has no defensible return to report.
        excluded = con.execute(
            f"""SELECT date FROM bist100_prices
                WHERE (bar_status IS NULL OR bar_status NOT IN ({placeholders}))
                  AND daily_return IS NOT NULL""",
            ANALYSABLE_STATUSES,
        ).fetchall()
        updates.extend((None, row["date"]) for row in excluded)

        if updates:
            con.executemany(
                "UPDATE bist100_prices SET daily_return=? WHERE date=?", updates
            )
    if updates:
        logger.info("Recomputed %d daily return(s) on the stored series", len(updates))
    return len(updates)


def backfill_price_bar_status(db_path: str = DB_PATH) -> Dict[str, int]:
    """Classify stored bars that predate completeness tracking.

    Settlement is established from recorded evidence rather than assumed: a
    price fetch downloads the whole lookback window, so any run that started
    after a session settled has already refreshed that session's bar. The
    latest run start is therefore the observation time for every stored bar.

    With no run history the settlement time cannot be established and rows stay
    provisional, which is the safe direction -- an unverifiable bar is withheld
    from analysis rather than trusted.
    """
    from config import PRICE_BAR_RULE_VERSION
    from price_bars import classify_price_bar

    with _conn(db_path) as con:
        latest_run = con.execute(
            "SELECT MAX(started_at) FROM pipeline_runs"
        ).fetchone()[0]
        rows = con.execute(
            "SELECT date, volume FROM bist100_prices "
            "WHERE COALESCE(bar_rule_version, '') <> ?",
            (PRICE_BAR_RULE_VERSION,),
        ).fetchall()

        counts: Dict[str, int] = {"classified": 0, "flagged_for_review": 0}
        updates = []
        for row in rows:
            classification = classify_price_bar(
                row["date"], volume=row["volume"], observed_at=latest_run
            )
            counts[classification.status] = counts.get(classification.status, 0) + 1
            counts["classified"] += 1
            if classification.needs_review:
                counts["flagged_for_review"] += 1
            updates.append((
                classification.status, latest_run,
                classification.review_reason, PRICE_BAR_RULE_VERSION, row["date"],
            ))
        if updates:
            con.executemany(
                """UPDATE bist100_prices
                   SET bar_status=?, bar_observed_at=?, bar_review_reason=?,
                       bar_rule_version=?
                   WHERE date=?""",
                updates,
            )
    logger.info("Classified %d price bars", counts["classified"])
    if updates:
        # Classification decides which bars form the settled series, so the
        # return chain has to be rebuilt against the new membership.
        counts["returns_recomputed"] = recompute_daily_returns(db_path=db_path)
    return counts


def list_price_bars_for_review(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Return bars carrying a data-quality flag or withheld from analysis."""
    from price_bars import ANALYSABLE_STATUSES

    placeholders = ",".join("?" * len(ANALYSABLE_STATUSES))
    with _conn(db_path) as con:
        rows = con.execute(
            f"""SELECT date, close, volume, bar_status, bar_review_reason,
                       bar_observed_at
                FROM bist100_prices
                WHERE bar_review_reason IS NOT NULL
                   OR COALESCE(bar_status, '') NOT IN ({placeholders})
                ORDER BY date""",
            ANALYSABLE_STATUSES,
        ).fetchall()
    return [dict(row) for row in rows]


def get_prices(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db_path: str = DB_PATH,
    *,
    complete_only: bool = True,
) -> pd.DataFrame:
    """Return stored daily bars, settled bars only by default.

    Return construction must not consume a bar captured mid-session, so
    provisional and provider-invalid rows are withheld unless a caller asks for
    them explicitly. Rows predating completeness tracking have a NULL status and
    are also withheld, since an unclassified bar is not a verified one; run
    :func:`backfill_price_bar_status` to resolve them.
    """
    from price_bars import ANALYSABLE_STATUSES

    where, params = [], []
    if start:
        where.append("date >= ?")
        params.append(start)
    if end:
        where.append("date <= ?")
        params.append(end)
    if complete_only:
        placeholders = ",".join("?" * len(ANALYSABLE_STATUSES))
        where.append(f"bar_status IN ({placeholders})")
        params.extend(ANALYSABLE_STATUSES)
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


def upsert_signal_variants(
    rows: Iterable[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> int:
    """Store all session-aligned signal specifications side by side."""
    now = _now_iso()
    data = [
        (
            row.get("signal_date") or row.get("date"),
            row["simple_mean"],
            row.get("relevance_weighted"),
            row.get("intensity_relevance_weighted"),
            row.get("full_weighted"),
            row["headline_count"],
            row.get("positive_count", 0),
            row.get("negative_count", 0),
            row.get("neutral_count", 0),
            row.get("unclassified_count", 0),
            row.get("positive_share"),
            row.get("negative_share"),
            row.get("neutral_share"),
            row.get("sentiment_dispersion", row.get("dispersion")),
            row.get("source_count", 0),
            row.get("event_count", 0),
            row.get("relevance_weight_sum"),
            row.get("intensity_relevance_weight_sum"),
            row.get("full_weight_sum"),
            now,
        )
        for row in rows
    ]
    if not data:
        return 0
    if any(not row[0] for row in data):
        raise ValueError("every signal-variant row requires a signal_date")
    with _conn(db_path) as con:
        con.executemany(
            """INSERT OR REPLACE INTO daily_signal_variants
               (signal_date, simple_mean, relevance_weighted,
                intensity_relevance_weighted, full_weighted, headline_count,
                positive_count, negative_count, neutral_count,
                unclassified_count, positive_share, negative_share,
                neutral_share, sentiment_dispersion, source_count, event_count,
                relevance_weight_sum, intensity_relevance_weight_sum,
                full_weight_sum, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data,
        )
    logger.info("Upserted %d session signal-variant rows", len(data))
    return len(data)


def get_signal_variants(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Return canonical session-aligned signals; ``simple_mean`` is baseline."""
    where: List[str] = []
    params: List[Any] = []
    if start:
        where.append("signal_date >= ?")
        params.append(start)
    if end:
        where.append("signal_date <= ?")
        params.append(end)
    clause = "WHERE " + " AND ".join(where) if where else ""
    with _conn(db_path) as con:
        return pd.read_sql_query(
            f"""SELECT signal_date AS date, *
                FROM daily_signal_variants {clause}
                ORDER BY signal_date""",
            con,
            params=params,
        )


def upsert_category_signal_sentiment(
    rows: Iterable[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> int:
    """Store unweighted category baselines keyed to tradable session."""
    data = [
        (
            row.get("signal_date") or row.get("date"),
            row["category"],
            row["simple_mean"],
            row["headline_count"],
        )
        for row in rows
    ]
    if not data:
        return 0
    with _conn(db_path) as con:
        con.executemany(
            """INSERT OR REPLACE INTO category_sentiment_by_signal
               (signal_date, category, simple_mean, headline_count)
               VALUES (?, ?, ?, ?)""",
            data,
        )
    return len(data)


def get_category_signal_sentiment(
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    with _conn(db_path) as con:
        return pd.read_sql_query(
            """SELECT signal_date AS date, category, simple_mean, headline_count
               FROM category_sentiment_by_signal
               ORDER BY signal_date, category""",
            con,
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

_CANONICAL_RUN_STATUSES = {"running", "success", "degraded", "failed"}
_LEGACY_RUN_STATUSES = {"ok", "error", "recovered", "crashed"}
_COMPONENT_RUN_STATUSES = {
    "pending", "running", "success", "degraded", "failed", "skipped"
}


def _validate_run_status(status: Optional[str], *, component: bool = False) -> None:
    if status is None:
        return
    allowed = (
        _COMPONENT_RUN_STATUSES
        if component
        else _CANONICAL_RUN_STATUSES | _LEGACY_RUN_STATUSES
    )
    if status not in allowed:
        kind = "component" if component else "pipeline"
        raise ValueError(f"unsupported {kind} status: {status!r}")


def _audit_json(value: Any, field: str) -> str:
    """Serialize structured run diagnostics to deterministic valid JSON."""
    if value is None:
        value = []
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-serializable") from exc


def _canonical_run_status(status: str) -> str:
    return {
        "ok": "success",
        "recovered": "success",
        "error": "failed",
        "crashed": "failed",
    }.get(status, status)

def log_run_start(model_name: Optional[str] = None, db_path: str = DB_PATH,
                  experiment_id: Optional[str] = None, *,
                  scrape_status: Optional[str] = "pending",
                  scoring_status: Optional[str] = "pending",
                  aggregation_status: Optional[str] = "pending",
                  market_data_status: Optional[str] = "pending",
                  audit_status: Optional[str] = "pending",
                  warnings: Any = None,
                  errors: Any = None) -> int:
    """Insert a 'running' run record. Returns the new run_id."""
    for component_status in (
        scrape_status, scoring_status, aggregation_status,
        market_data_status, audit_status,
    ):
        _validate_run_status(component_status, component=True)
    if experiment_id is None:
        from config import EXPERIMENT_ID
        experiment_id = EXPERIMENT_ID
    with _conn(db_path) as con:
        cur = con.execute(
            """INSERT INTO pipeline_runs
               (started_at, model_name, status, experiment_id,
                scrape_status, scoring_status, aggregation_status,
                market_data_status, audit_status, warnings_json, errors_json)
               VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _now_iso(), model_name, experiment_id,
                scrape_status, scoring_status, aggregation_status,
                market_data_status, audit_status,
                _audit_json(warnings, "warnings"), _audit_json(errors, "errors"),
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]


def fail_interrupted_pipeline_runs(db_path: str = DB_PATH) -> int:
    """Close stale ``running`` audits with coherent canonical diagnostics."""

    interruption = _audit_json(
        [{
            "component": "pipeline",
            "code": "scheduler_interrupted_run",
            "message": "scheduler detected an interrupted run",
        }],
        "errors",
    )
    component_columns = (
        "scrape_status", "scoring_status", "aggregation_status",
        "market_data_status", "audit_status",
    )
    assignments = [
        "status='failed'",
        "finished_at=?",
        "error_msg=COALESCE(error_msg, 'scheduler detected an interrupted run')",
        "errors_json=CASE WHEN COALESCE(errors_json, '[]')='[]' THEN ? ELSE errors_json END",
    ]
    assignments.extend(
        f"{column}=CASE WHEN {column}='running' THEN 'failed' "
        f"WHEN {column}='pending' THEN 'skipped' ELSE {column} END"
        for column in component_columns
    )
    with _conn(db_path) as con:
        cur = con.execute(
            "UPDATE pipeline_runs SET " + ", ".join(assignments)
            + " WHERE status='running'",
            (_now_iso(), interruption),
        )
        return max(0, cur.rowcount)


def log_run_end(
    run_id: int,
    status: str = "success",
    headlines_scraped: int = 0,
    headlines_scored: int = 0,
    prices_added: int = 0,
    sentiment_days: int = 0,
    error_msg: Optional[str] = None,
    db_path: str = DB_PATH,
    *,
    scrape_status: Optional[str] = None,
    scoring_status: Optional[str] = None,
    aggregation_status: Optional[str] = None,
    market_data_status: Optional[str] = None,
    audit_status: Optional[str] = None,
    warnings: Any = None,
    errors: Any = None,
) -> None:
    """Update a run with legacy-compatible status and structured diagnostics."""
    _validate_run_status(status)
    for component_status in (
        scrape_status, scoring_status, aggregation_status,
        market_data_status, audit_status,
    ):
        _validate_run_status(component_status, component=True)
    if errors is None and error_msg:
        errors = [{"message": error_msg}]
    warnings_json = None if warnings is None else _audit_json(warnings, "warnings")
    errors_json = None if errors is None else _audit_json(errors, "errors")
    with _conn(db_path) as con:
        cur = con.execute(
            """UPDATE pipeline_runs
               SET finished_at=?, headlines_scraped=?, headlines_scored=?,
                   prices_added=?, sentiment_days=?, status=?, error_msg=?,
                   scrape_status=COALESCE(?, scrape_status),
                   scoring_status=COALESCE(?, scoring_status),
                   aggregation_status=COALESCE(?, aggregation_status),
                   market_data_status=COALESCE(?, market_data_status),
                   audit_status=COALESCE(?, audit_status),
                   warnings_json=COALESCE(?, warnings_json),
                   errors_json=COALESCE(?, errors_json)
               WHERE run_id=?""",
            (
                _now_iso(), headlines_scraped, headlines_scored,
                prices_added, sentiment_days, status, error_msg,
                scrape_status, scoring_status, aggregation_status,
                market_data_status, audit_status,
                warnings_json, errors_json,
                run_id,
            ),
        )
        if cur.rowcount != 1:
            raise KeyError(f"pipeline run {run_id} does not exist")


def get_pipeline_run(run_id: int, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Return one run record with warnings/errors decoded for API consumers."""
    with _conn(db_path) as con:
        row = con.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"pipeline run {run_id} does not exist")
    result = dict(row)
    result["warnings"] = json.loads(result.get("warnings_json") or "[]")
    result["errors"] = json.loads(result.get("errors_json") or "[]")
    result["canonical_status"] = _canonical_run_status(str(result["status"]))
    return result


# -- Headline cleanup (for `main.py clean`) -----------------------------------

def count_off_topic_headlines(db_path: str = DB_PATH) -> int:
    """Return how many eligible headlines would receive a relevance exclusion."""
    from scraper import _is_relevant   # local import to avoid circular at module load
    with _conn(db_path) as con:
        rows = con.execute(
            """SELECT h.id, h.title
               FROM headlines AS h
               WHERE NOT EXISTS (
                   SELECT 1 FROM headline_exclusions AS x
                   WHERE x.headline_id = h.id AND x.restored_at IS NULL
               )"""
        ).fetchall()
    return sum(1 for r in rows if not _is_relevant(r["title"]))


def clean_off_topic_headlines(db_path: str = DB_PATH) -> int:
    """
    Reversibly exclude headlines that fail the current relevance filter.

    Existing raw headlines and scores remain intact. Re-running this function
    is idempotent for rows with an active exclusion; restoration remains an
    explicit operation through ``restore_headline_exclusion``.

    Returns the number of newly excluded rows.
    """
    from scraper import _is_relevant   # local import
    try:
        from config import KEYWORD_RELEVANCE_RULE_VERSION
    except ImportError:  # pragma: no cover - compatibility with old config files
        KEYWORD_RELEVANCE_RULE_VERSION = "keyword-relevance-unversioned"
    with _conn(db_path) as con:
        rows = con.execute(
            """SELECT h.id, h.title
               FROM headlines AS h
               WHERE NOT EXISTS (
                   SELECT 1 FROM headline_exclusions AS x
                   WHERE x.headline_id = h.id AND x.restored_at IS NULL
               )"""
        ).fetchall()
        bad_ids = [r["id"] for r in rows if not _is_relevant(r["title"])]
        if not bad_ids:
            logger.info("clean: no off-topic headlines found")
            return 0
        now = _now_iso()
        before = con.total_changes
        con.executemany(
            """INSERT OR IGNORE INTO headline_exclusions
               (headline_id, exclusion_reason, exclusion_rule,
                exclusion_version, excluded_at)
               VALUES (?, 'off_topic', 'keyword_relevance', ?, ?)""",
            [
                (headline_id, KEYWORD_RELEVANCE_RULE_VERSION, now)
                for headline_id in bad_ids
            ],
        )
        excluded = con.total_changes - before
        logger.info("clean: excluded %d off-topic headlines", excluded)
    return excluded


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


def upsert_external_series(rows: Iterable[Tuple], db_path: str = DB_PATH) -> int:
    """Upsert (date, series, value) rows into external_series."""
    data = list(rows)
    with _conn(db_path) as con:
        con.executemany(
            "INSERT OR REPLACE INTO external_series (date, series, value) VALUES (?, ?, ?)", data)
    logger.info("Upserted %d external-series rows", len(data))
    return len(data)


def get_external_series(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return external_series wide (one column per series, indexed by date)."""
    with _conn(db_path) as con:
        long = pd.read_sql_query("SELECT date, series, value FROM external_series", con)
    if long.empty:
        return long
    return long.pivot(index="date", columns="series", values="value").reset_index()


def upsert_market_factors(rows: Iterable[Dict[str, Any]], db_path: str = DB_PATH) -> int:
    """Upsert market-factor rows.

    Each dict: date, symbol, label, close, daily_return, and optionally the
    provenance triple (source, retrieved_at, transform_version). Provenance is
    optional so the live collection path stays unchanged, but a backfilled row
    that omits it is indistinguishable from a collected one, which is why
    scripts/backfill_market_history.py always supplies it.
    """
    now = _now_iso()
    data = [
        (r["date"], r["symbol"], r.get("label"), r.get("close"),
         r.get("daily_return"), r.get("source", "yfinance"),
         r.get("retrieved_at", now), r.get("transform_version", "live-collection"))
        for r in rows
    ]
    with _conn(db_path) as con:
        con.executemany(
            """INSERT OR REPLACE INTO market_factors
               (date, symbol, label, close, daily_return, source, retrieved_at,
                transform_version)
               VALUES (?,?,?,?,?,?,?,?)""",
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
        sent_d  = con.execute("SELECT COUNT(*) FROM daily_signal_variants").fetchone()[0]
        legacy_sent_d = con.execute("SELECT COUNT(*) FROM daily_sentiment").fetchone()[0]
        fx_days = con.execute("SELECT COUNT(*) FROM usdtry_rates").fetchone()[0]
        min_pub = con.execute("SELECT MIN(published_at) FROM headlines").fetchone()[0]
        max_pub = con.execute("SELECT MAX(published_at) FROM headlines").fetchone()[0]
        processing_rows = con.execute(
            """SELECT processing_status, COUNT(*) AS n
               FROM headlines GROUP BY processing_status"""
        ).fetchall()
        active_exclusions = con.execute(
            "SELECT COUNT(*) FROM headline_exclusions WHERE restored_at IS NULL"
        ).fetchone()[0]
        # Category breakdown (top categories by headline count)
        cat_rows = con.execute(
            """SELECT category, COUNT(*) AS n FROM headlines
               WHERE category IS NOT NULL
               GROUP BY category ORDER BY n DESC LIMIT 8"""
        ).fetchall()
        # Last run info
        last_run = con.execute(
            """SELECT status, started_at, error_msg, scrape_status,
                      scoring_status, aggregation_status, market_data_status,
                      audit_status, warnings_json, errors_json
               FROM pipeline_runs ORDER BY run_id DESC LIMIT 1"""
        ).fetchone()

    cat_summary = ", ".join(f"{r[0]}:{r[1]}" for r in cat_rows) if cat_rows else "none"
    stats: Dict[str, Any] = {
        "total_headlines":    total,
        "scored_headlines":   scored,
        "unscored_headlines": total - scored,
        "price_days":         prices,
        "fx_rate_days":       fx_days,
        "signal_session_days": sent_d,
        "sentiment_days": sent_d,
        "legacy_calendar_sentiment_days": legacy_sent_d,
        "oldest_headline":    min_pub,
        "newest_headline":    max_pub,
        "categories":         cat_summary,
        "processing_status_counts": {
            row["processing_status"]: int(row["n"]) for row in processing_rows
        },
        "active_exclusions": int(active_exclusions),
    }
    if last_run:
        stats["last_run_status"] = last_run["status"]
        stats["last_run_final_status"] = _canonical_run_status(last_run["status"])
        stats["last_run_at"] = last_run["started_at"]
        stats["last_run_components"] = {
            "scrape": last_run["scrape_status"],
            "scoring": last_run["scoring_status"],
            "aggregation": last_run["aggregation_status"],
            "market_data": last_run["market_data_status"],
            "audit": last_run["audit_status"],
        }
        try:
            stats["last_run_warnings"] = json.loads(last_run["warnings_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            stats["last_run_warnings"] = []
        try:
            stats["last_run_errors"] = json.loads(last_run["errors_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            stats["last_run_errors"] = []
        if last_run["error_msg"]:
            stats["last_run_error"] = last_run["error_msg"]
    return stats
