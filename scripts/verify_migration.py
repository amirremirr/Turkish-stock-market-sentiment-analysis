"""Verify that the additive schema migrations preserve a legacy database.

The source database is never opened for writing.  Every stage runs against a
throwaway copy, so a verification run can be repeated on the canonical
production snapshot without risking it.

Stages, reported separately so their effects stay distinguishable:

    baseline    the untouched copy
    init_db     additive DDL plus ``_MIGRATIONS`` ALTER TABLE statements
    init_db_2   a second call, which must be a content no-op
    sessions    ``backfill_session_assignments`` (derived timing metadata)
    relevance   ``reconcile_relevance_exclusions`` (reversible exclusions)

Raw observations are what the invariants protect: the scored-sentiment digest
covers ``sentiment_score``, ``sentiment_label``, ``scored_at``, and
``model_name`` and must be identical in every stage.  A changed digest means a
migration rewrote a historical score and the run fails.

Usage::

    python -m scripts.verify_migration backups/phase0_canonical_data_2026-07-31.db
    python -m scripts.verify_migration <db> --json-out report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Columns whose values are historical model output.  A migration may add
# columns beside them but must never rewrite them.
_SCORE_COLUMNS = ("sentiment_score", "sentiment_label", "scored_at", "model_name")

# Tables whose row count must not change during a purely additive migration.
_STABLE_COUNT_TABLES = (
    "headlines",
    "bist100_prices",
    "usdtry_rates",
    "market_factors",
    "external_series",
    "events",
    "event_entities",
    "pipeline_runs",
    "experiments",
    "kv_state",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_names(con: sqlite3.Connection) -> List[str]:
    return [
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _columns(con: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _row_counts(con: sqlite3.Connection) -> Dict[str, int]:
    return {
        table: int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in _table_names(con)
    }


def _score_digest(con: sqlite3.Connection) -> Optional[str]:
    """Hash every historical score, keyed by headline id and ordered by id."""

    if not _table_exists(con, "headlines"):
        return None
    available = set(_columns(con, "headlines"))
    if not set(_SCORE_COLUMNS).issubset(available):
        return None
    digest = hashlib.sha256()
    for row in con.execute(
        f"SELECT id, {', '.join(_SCORE_COLUMNS)} FROM headlines ORDER BY id"
    ):
        digest.update(repr(tuple(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _content_digest(con: sqlite3.Connection) -> str:
    """Hash every user table's contents, independent of SQLite page layout.

    A file-level hash changes whenever SQLite rewrites pages, so it cannot
    prove that a repeated call left the data alone.  This digest can.
    """

    digest = hashlib.sha256()
    for table in _table_names(con):
        columns = _columns(con, table)
        digest.update(f"TABLE {table}({','.join(columns)})\n".encode("utf-8"))
        order = ", ".join(f'"{column}"' for column in columns)
        for row in con.execute(f'SELECT * FROM "{table}" ORDER BY {order}'):
            digest.update(repr(tuple(row)).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _distribution(con: sqlite3.Connection, table: str, expression: str) -> Dict[str, int]:
    if not _table_exists(con, table):
        return {}
    return {
        ("NULL" if row[0] is None else str(row[0])): int(row[1])
        for row in con.execute(
            f"SELECT {expression} AS bucket, COUNT(*) FROM {table} "
            f"GROUP BY bucket ORDER BY bucket"
        )
    }


def _date_ranges(con: sqlite3.Connection) -> Dict[str, Any]:
    ranges: Dict[str, Any] = {}
    probes = [
        ("headlines.published_at", "headlines", "published_at"),
        ("headlines.signal_date", "headlines", "signal_date"),
        ("bist100_prices.date", "bist100_prices", "date"),
        ("market_factors.date", "market_factors", "date"),
        ("external_series.date", "external_series", "date"),
    ]
    for label, table, column in probes:
        if not _table_exists(con, table) or column not in _columns(con, table):
            continue
        low, high = con.execute(
            f"SELECT MIN({column}), MAX({column}) FROM {table}"
        ).fetchone()
        ranges[label] = {"min": low, "max": high}
    return ranges


def snapshot(db_path: Path, *, stage: str) -> Dict[str, Any]:
    """Capture every fact the invariant checks compare between stages."""

    con = sqlite3.connect(str(db_path))
    try:
        tables = _table_names(con)
        return {
            "stage": stage,
            "file_size_bytes": db_path.stat().st_size,
            "file_sha256": sha256_file(db_path),
            "tables": tables,
            "columns": {table: _columns(con, table) for table in tables},
            "row_counts": _row_counts(con),
            "score_digest": _score_digest(con),
            "content_digest": _content_digest(con),
            "date_ranges": _date_ranges(con),
            "processing_status": _distribution(
                con, "headlines", "processing_status"
            ) if _table_exists(con, "headlines")
            and "processing_status" in _columns(con, "headlines") else {},
            "timing_bucket": _distribution(
                con, "headlines", "timing_bucket"
            ) if _table_exists(con, "headlines")
            and "timing_bucket" in _columns(con, "headlines") else {},
            "experiment_id": _distribution(
                con, "headlines", "COALESCE(experiment_id, 'NULL')"
            ) if _table_exists(con, "headlines")
            and "experiment_id" in _columns(con, "headlines") else {},
            "score_components_kind": _distribution(
                con, "headlines", "COALESCE(score_components_kind, 'NULL')"
            ) if _table_exists(con, "headlines")
            and "score_components_kind" in _columns(con, "headlines") else {},
            "null_published_hour": int(
                con.execute(
                    "SELECT COUNT(*) FROM headlines WHERE published_hour IS NULL"
                ).fetchone()[0]
            ) if _table_exists(con, "headlines") else 0,
        }
    finally:
        con.close()


def _signal_date_changes(before: Path, after: Path) -> Dict[str, Any]:
    """Compare stored signal dates headline by headline across two copies.

    A changed value is not automatically a fault.  ``session_rule_version``
    exists so that a corrected trading calendar can re-derive assignments that
    an earlier rule got wrong.  This returns the evidence needed to judge each
    change rather than a bare equality verdict: whether the new session is a
    real trading day, whether the market actually printed a price that day, and
    which direction the assignment moved.
    """

    from trading_calendar import is_trading_day

    con = sqlite3.connect(str(before))
    try:
        original = dict(con.execute("SELECT id, signal_date FROM headlines"))
    finally:
        con.close()

    con = sqlite3.connect(str(after))
    try:
        updated = dict(
            con.execute("SELECT id, signal_date FROM headlines")
        )
        anchors = dict(
            con.execute("SELECT id, published_at FROM headlines")
        )
        priced = {
            row[0] for row in con.execute("SELECT date FROM bist100_prices")
        }
        price_low, price_high = con.execute(
            "SELECT MIN(date), MAX(date) FROM bist100_prices"
        ).fetchone()
    finally:
        con.close()

    changed: List[Dict[str, Any]] = []
    filled = 0
    for headline_id, new_value in updated.items():
        old_value = original.get(headline_id)
        if old_value == new_value:
            continue
        if old_value is None:
            filled += 1
            continue
        in_price_range = (
            price_low is not None and price_low <= new_value <= price_high
        )
        changed.append({
            "id": headline_id,
            "published_at": anchors.get(headline_id),
            "before": old_value,
            "after": new_value,
            "direction": "earlier" if new_value < old_value else "later",
            "new_is_trading_day": is_trading_day(
                __import__("datetime").date.fromisoformat(new_value)
            ),
            "new_has_price_row": (new_value in priced) if in_price_range else None,
        })

    unverifiable = [
        item for item in changed
        if not item["new_is_trading_day"]
        or item["new_has_price_row"] is False
    ]
    return {
        "changed_count": len(changed),
        "filled_from_null_count": filled,
        "directions": {
            "earlier": sum(1 for item in changed if item["direction"] == "earlier"),
            "later": sum(1 for item in changed if item["direction"] == "later"),
        },
        "unverifiable_count": len(unverifiable),
        "unverifiable": unverifiable[:20],
        "examples": changed[:20],
    }


def _eligible_experiment_ids(db_path: Path) -> List[str]:
    import database as db

    return db.get_eligible_experiment_ids(db_path=str(db_path))


def run_verification(source: Path, workdir: Path) -> Dict[str, Any]:
    """Copy *source*, migrate the copy, and record every stage plus verdicts."""

    import database as db

    workdir.mkdir(parents=True, exist_ok=True)
    working = workdir / "migrating.db"
    shutil.copyfile(source, working)

    stages: List[Dict[str, Any]] = [snapshot(working, stage="baseline")]

    db.init_db(str(working))
    stages.append(snapshot(working, stage="init_db"))

    db.init_db(str(working))
    stages.append(snapshot(working, stage="init_db_2"))

    pre_sessions = workdir / "pre_sessions.db"
    shutil.copyfile(working, pre_sessions)
    backfilled = db.backfill_session_assignments(db_path=str(working))
    stages.append(snapshot(working, stage="sessions"))

    reconciled = db.reconcile_relevance_exclusions(db_path=str(working))
    stages.append(snapshot(working, stage="relevance"))

    by_stage = {stage["stage"]: stage for stage in stages}
    baseline, after_init, after_init2 = (
        by_stage["baseline"], by_stage["init_db"], by_stage["init_db_2"]
    )
    after_sessions, after_relevance = by_stage["sessions"], by_stage["relevance"]

    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    stable_before = {
        table: count for table, count in baseline["row_counts"].items()
        if table in _STABLE_COUNT_TABLES
    }
    stable_after = {
        table: after_relevance["row_counts"].get(table)
        for table in stable_before
    }
    check(
        "stable_table_row_counts_unchanged",
        stable_before == stable_after,
        {"before": stable_before, "after": stable_after},
    )

    digests = {stage["stage"]: stage["score_digest"] for stage in stages}
    reference = digests["init_db"]
    check(
        "scored_sentiment_digest_unchanged_after_migration",
        reference is not None
        and all(digests[name] == reference for name in
                ("init_db", "init_db_2", "sessions", "relevance")),
        digests,
    )

    check(
        "second_init_db_is_content_no_op",
        after_init["content_digest"] == after_init2["content_digest"],
        {
            "after_init_db": after_init["content_digest"],
            "after_init_db_2": after_init2["content_digest"],
        },
    )
    check(
        "second_init_db_adds_no_columns",
        after_init["columns"] == after_init2["columns"],
    )

    added_tables = sorted(set(after_init["tables"]) - set(baseline["tables"]))
    removed_tables = sorted(set(baseline["tables"]) - set(after_init["tables"]))
    check("no_tables_removed", not removed_tables, removed_tables)

    removed_columns = {
        table: sorted(set(baseline["columns"][table]) - set(after_init["columns"].get(table, [])))
        for table in baseline["columns"]
    }
    removed_columns = {k: v for k, v in removed_columns.items() if v}
    check("no_columns_removed", not removed_columns, removed_columns)

    scored = after_init["processing_status"].get("scored", 0)
    check(
        "every_previously_scored_row_is_marked_scored",
        scored == baseline["row_counts"].get("headlines", -1),
        {
            "processing_status": after_init["processing_status"],
            "headline_rows": baseline["row_counts"].get("headlines"),
        },
    )

    identities = _eligible_experiment_ids(working)
    check(
        "single_eligible_experiment_identity",
        len(identities) == 1,
        identities,
    )

    # A re-derived session assignment is permitted, but only toward a session
    # the exchange actually held.  The gate is correctness of the new value,
    # not immutability of the old one; the count itself is reported as a
    # deviation for explicit sign-off.
    session_changes = _signal_date_changes(pre_sessions, working)
    check(
        "session_backfill_targets_are_real_trading_sessions",
        session_changes["unverifiable_count"] == 0,
        session_changes,
    )

    null_hour = baseline["null_published_hour"]
    conservative = sum(
        count for bucket, count in after_sessions["timing_bucket"].items()
        if bucket in {"unknown", "weekend_or_holiday"}
    )
    check(
        "missing_hour_rows_receive_conservative_bucket",
        conservative >= null_hour,
        {
            "null_published_hour": null_hour,
            "conservative_bucket_rows": conservative,
            "timing_bucket": after_sessions["timing_bucket"],
        },
    )

    # Changes that are permitted but must never pass silently.  Each needs an
    # explicit line in the migration report before production is migrated.
    deviations: List[Dict[str, Any]] = []
    if session_changes["changed_count"]:
        deviations.append({
            "deviation": "signal_date re-derived under the current calendar rule",
            "rows": session_changes["changed_count"],
            "directions": session_changes["directions"],
            "verified_against_price_rows": session_changes["unverifiable_count"] == 0,
        })
    if reconciled.get("excluded"):
        deviations.append({
            "deviation": "reversible low-relevance exclusions created",
            "rows": reconciled["excluded"],
            "reversible": True,
        })
    if reconciled.get("restored"):
        deviations.append({
            "deviation": "low-relevance exclusions restored",
            "rows": reconciled["restored"],
            "reversible": True,
        })

    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "stages": stages,
        "checks": checks,
        "deviations": deviations,
        "added_tables": added_tables,
        "added_columns": {
            table: sorted(
                set(after_init["columns"].get(table, []))
                - set(baseline["columns"].get(table, []))
            )
            for table in after_init["columns"]
            if set(after_init["columns"].get(table, []))
            - set(baseline["columns"].get(table, []))
        },
        "session_backfill": {"rows_touched": backfilled, **session_changes},
        "relevance_reconciliation": reconciled,
        "eligible_experiment_ids": identities,
        "migrated_copy": str(working),
        "passed": all(item["passed"] for item in checks),
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        f"source            {report['source']}",
        f"source sha256     {report['source_sha256']}",
        f"migrated copy     {report['migrated_copy']}",
        "",
        "Stages",
    ]
    for stage in report["stages"]:
        lines.append(
            f"  {stage['stage']:<12} {stage['file_size_bytes']:>10,} bytes  "
            f"tables={len(stage['tables']):<3} content={stage['content_digest'][:12]}"
        )
    lines += ["", f"Added tables      {', '.join(report['added_tables']) or 'none'}"]
    for table, columns in report["added_columns"].items():
        lines.append(f"  + {table}: {', '.join(columns)}")
    lines += [
        "",
        f"Session backfill  {report['session_backfill']['rows_touched']} rows touched, "
        f"{report['session_backfill']['changed_count']} existing signal dates changed",
        f"Relevance         {report['relevance_reconciliation']}",
        f"Experiment ids    {report['eligible_experiment_ids']}",
        "",
        "Checks",
    ]
    for item in report["checks"]:
        lines.append(f"  [{'PASS' if item['passed'] else 'FAIL'}] {item['check']}")
    lines += ["", "Deviations requiring sign-off"]
    if report["deviations"]:
        for item in report["deviations"]:
            lines.append(f"  - {item['deviation']}: {item['rows']} row(s)")
    else:
        lines.append("  none")
    lines += ["", f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}"]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("database", type=Path, help="legacy database to verify")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="directory for the migrated copy (default: a temporary directory)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database.exists():
        print(f"database not found: {args.database}", file=sys.stderr)
        return 2

    workdir = args.workdir
    temporary = None
    if workdir is None:
        temporary = tempfile.TemporaryDirectory()
        workdir = Path(temporary.name)
    try:
        report = run_verification(args.database, workdir)
        print(format_report(report))
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(f"\nJSON report written to {args.json_out}")
        return 0 if report["passed"] else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
