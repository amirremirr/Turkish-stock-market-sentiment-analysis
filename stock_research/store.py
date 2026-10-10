"""The stock-research database: a separate SQLite file with its own schema.

Raw tables are append-only by trigger. A fetched disclosure or price bar is a
record of what a provider said at a moment; it is never edited, only superseded
by a later snapshot that sits beside it. Derived tables (events, links) are
rebuildable from the raw ones and carry the version of the rule that built them.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from stock_research.config import SCHEMA_VERSION, STOCK_DB_PATH

_APPEND_ONLY = (
    "sr_raw_kap_listing", "sr_raw_kap_detail", "sr_raw_price_bars",
    "sr_raw_members", "sr_protocols", "sr_manifests", "sr_results",
)

_DDL = """
CREATE TABLE IF NOT EXISTS sr_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- One row per disclosure seen in a listing call. No timestamp: the listing
-- endpoint does not return one.
CREATE TABLE IF NOT EXISTS sr_raw_kap_listing (
    disclosure_index INTEGER PRIMARY KEY,
    frame_version    TEXT NOT NULL,
    block_seq        INTEGER NOT NULL,
    disclosure_type  TEXT,
    disclosure_class TEXT,
    company_id       TEXT,
    title            TEXT,
    payload_json     TEXT NOT NULL,
    source           TEXT NOT NULL,
    retrieved_at     TEXT NOT NULL
);

-- One row per fetched detail. The HTML body is stored as extracted text plus
-- the SHA-256 of the original payload, not the payload itself (~100 KB each).
CREATE TABLE IF NOT EXISTS sr_raw_kap_detail (
    disclosure_index INTEGER PRIMARY KEY,
    sender_id        TEXT,
    sender_title     TEXT,
    sender_codes     TEXT,          -- JSON list, as given
    related_stocks   TEXT,          -- JSON list, as given
    disclosure_type  TEXT,
    disclosure_class TEXT,
    disclosure_reason TEXT,
    subject_tr       TEXT,
    subject_en       TEXT,
    summary_tr       TEXT,
    published_raw    TEXT,          -- 'dd.MM.yyyy HH:mm:ss', Istanbul local
    event_type_code  TEXT,
    body_text        TEXT,
    metadata_json    TEXT NOT NULL, -- every field except the HTML body
    payload_sha256   TEXT NOT NULL,
    source           TEXT NOT NULL,
    retrieved_at     TEXT NOT NULL
);

-- Progress of the block sample. Mutable on purpose: it is bookkeeping.
CREATE TABLE IF NOT EXISTS sr_kap_frame (
    frame_version TEXT NOT NULL,
    block_seq     INTEGER NOT NULL,
    block_start   INTEGER NOT NULL,
    process_order INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending|listed|complete
    listed        INTEGER,
    targets       INTEGER,
    details_done  INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT,
    PRIMARY KEY (frame_version, block_seq)
);

-- Member snapshot. NOT point-in-time: it is the roster the provider served on
-- retrieved_at. Using it for a historical date is a stated limitation.
CREATE TABLE IF NOT EXISTS sr_raw_members (
    snapshot_id  TEXT NOT NULL,
    member_id    TEXT NOT NULL,
    title        TEXT,
    stock_codes  TEXT,
    member_type  TEXT,
    source       TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, member_id)
);

-- Daily bars exactly as a provider returned them, one snapshot at a time.
CREATE TABLE IF NOT EXISTS sr_raw_price_bars (
    snapshot_id  TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    date         TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, adj_close REAL,
    volume       REAL,
    dividend     REAL,
    split_ratio  REAL,
    source       TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, ticker, date)
);
CREATE INDEX IF NOT EXISTS ix_sr_bars_ticker ON sr_raw_price_bars (ticker, date);

-- A ticker a provider could not serve. Missing is a state, not an absence.
CREATE TABLE IF NOT EXISTS sr_price_availability (
    snapshot_id  TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    status       TEXT NOT NULL,     -- ok|no_data|error
    rows         INTEGER NOT NULL DEFAULT 0,
    first_date   TEXT, last_date TEXT,
    detail       TEXT,
    source       TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, ticker)
);

CREATE TABLE IF NOT EXISTS sr_protocols (
    protocol_hash TEXT PRIMARY KEY,
    version       TEXT NOT NULL,
    protocol_json TEXT NOT NULL,
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sr_manifests (
    manifest_hash TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sr_results (
    manifest_hash TEXT NOT NULL,
    hypothesis    TEXT NOT NULL,
    status        TEXT NOT NULL,
    result_json   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (manifest_hash, hypothesis)
);
"""


def _trigger_ddl() -> str:
    parts = []
    for table in _APPEND_ONLY:
        for action in ("UPDATE", "DELETE"):
            parts.append(
                f"CREATE TRIGGER IF NOT EXISTS trg_{table}_no_{action.lower()} "
                f"BEFORE {action} ON {table} BEGIN "
                f"SELECT RAISE(ABORT, '{table} is append-only'); END;"
            )
    return "\n".join(parts)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Optional[str | Path] = None) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(str(db_path or STOCK_DB_PATH), timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(db_path: Optional[str | Path] = None) -> None:
    with connect(db_path) as con:
        con.executescript(_DDL)
        con.executescript(_trigger_ddl())
        con.execute(
            "INSERT OR IGNORE INTO sr_meta (key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def table_counts(db_path: Optional[str | Path] = None) -> Dict[str, int]:
    with connect(db_path) as con:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sr_%'"
        )]
        return {n: con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in sorted(names)}
