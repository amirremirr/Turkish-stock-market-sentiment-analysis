"""Extend exogenous market-factor history so rolling controls can be estimated.

The rolling control model needs 60 prior sessions and refuses below 30. Factor
collection only ever fetched a short lookback, so the panel began in late March
2026 while the news corpus begins in mid-March — which left most event sessions
without enough prior factor history to estimate a beta, and their residual
returns NULL.

This is a **data-coverage** problem, not a modelling one, and it has a
data-coverage fix: fetch the same series further back. Nothing about the
estimator changes, and in particular the 30-observation minimum is not lowered.
Lowering it would raise coverage by making each residual worse, which is the
opposite of the point.

Three properties this script is built around:

**Nothing is fitted against outcomes.** It downloads price series and stores
returns. It never reads an event, a headline, a sentiment score or a target.

**Existing rows are never silently altered.** By default a date already present
is left alone, so a backfill cannot rewrite history a live run collected --
including a `corrected` BIST bar. ``--overwrite`` exists for a deliberate
repair and says so in its provenance.

**Every row can say where it came from.** ``source``, ``retrieved_at`` and
``transform_version`` are stored per row, so a backfilled value and a
live-collected one are distinguishable forever.

Usage::

    python -m scripts.backfill_market_history --db finance_sentiment.db --start 2025-01-01
    python -m scripts.backfill_market_history --db finance_sentiment.db --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

TRANSFORM_VERSION = "market-history-backfill-v1"
SOURCE = "yfinance"

#: Two years is comfortably more than the 60-session window needs and keeps the
#: request small enough to stay inside the free provider's limits.
DEFAULT_START = "2025-01-01"


def _download(symbol: str, start: str):
    import pandas as pd
    import yfinance as yf

    raw = yf.download(symbol, start=start, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.sort_index()


def _existing_dates(db_path: str, table: str, symbol: Optional[str]) -> set:
    connection = sqlite3.connect(db_path)
    try:
        if symbol:
            rows = connection.execute(
                f"SELECT date FROM {table} WHERE symbol = ?", (symbol,)
            ).fetchall()
        else:
            rows = connection.execute(f"SELECT date FROM {table}").fetchall()
    finally:
        connection.close()
    return {str(r[0]) for r in rows}


def backfill_factors(
    db_path: str, *, start: str, overwrite: bool = False, dry_run: bool = False,
) -> Dict[str, Any]:
    """Extend ``market_factors`` for every configured factor ticker."""

    import pandas as pd

    import database as db
    from config import FACTOR_TICKERS

    retrieved_at = datetime.now(timezone.utc).isoformat()
    report: Dict[str, Any] = {"symbols": {}, "rows_added": 0}

    for symbol, label in FACTOR_TICKERS.items():
        existing = _existing_dates(db_path, "market_factors", symbol)
        raw = _download(symbol, start)
        if raw is None:
            report["symbols"][symbol] = {"status": "no_data", "added": 0}
            continue

        returns = raw["Close"].pct_change().mul(100)
        candidates = [
            {
                "date": index.strftime("%Y-%m-%d"), "symbol": symbol,
                "label": label, "close": float(close),
                "daily_return": (None if pd.isna(r) else float(r)),
                "source": SOURCE, "retrieved_at": retrieved_at,
                "transform_version": TRANSFORM_VERSION,
            }
            for index, close, r in zip(raw.index, raw["Close"].values, returns.values)
            if not pd.isna(close)
        ]
        fresh = [
            row for row in candidates
            if overwrite or row["date"] not in existing
        ]
        if not dry_run and fresh:
            db.upsert_market_factors(fresh, db_path=db_path)

        report["symbols"][symbol] = {
            "status": "ok",
            "fetched": len(candidates),
            "already_present": len(candidates) - len(fresh),
            "added": len(fresh),
            "earliest_fetched": candidates[0]["date"] if candidates else None,
            "earliest_existing": min(existing) if existing else None,
        }
        report["rows_added"] += len(fresh)

    return report


def backfill_bist(
    db_path: str, *, start: str, overwrite: bool = False, dry_run: bool = False,
) -> Dict[str, Any]:
    """Extend ``bist100_prices`` backwards only.

    Historic sessions settled long ago, so a newly fetched old bar is
    ``complete`` by construction. Dates already stored are left untouched
    without ``--overwrite`` -- a backfill must not be able to overwrite the
    corrected 2026-07-31 bar with a provider value.
    """

    import pandas as pd

    import database as db
    from config import BIST100_TICKER as BIST_TICKER

    retrieved_at = datetime.now(timezone.utc).isoformat()
    existing = _existing_dates(db_path, "bist100_prices", None)
    raw = _download(BIST_TICKER, start)
    if raw is None:
        return {"status": "no_data", "added": 0}

    rows: List[Dict[str, Any]] = []
    previous_close: Optional[float] = None
    for index, row in raw.iterrows():
        day = index.strftime("%Y-%m-%d")
        close = row.get("Close")
        if pd.isna(close):
            continue
        daily_return = (
            None if previous_close in (None, 0)
            else (float(close) / previous_close - 1.0) * 100.0
        )
        previous_close = float(close)
        if not overwrite and day in existing:
            continue
        rows.append({
            "date": day,
            "open": None if pd.isna(row.get("Open")) else float(row["Open"]),
            "high": None if pd.isna(row.get("High")) else float(row["High"]),
            "low": None if pd.isna(row.get("Low")) else float(row["Low"]),
            "close": float(close),
            "volume": None if pd.isna(row.get("Volume")) else float(row["Volume"]),
            "daily_return": daily_return,
            "bar_status": "complete",
            "bar_observed_at": retrieved_at,
            "bar_review_reason": "historical_backfill_settled_long_ago",
            "source": SOURCE,
            "retrieved_at": retrieved_at,
        })

    if not dry_run and rows:
        _insert_bars(db_path, rows)

    return {
        "status": "ok",
        "fetched": int(len(raw)),
        "already_present": len(existing),
        "added": len(rows),
        "earliest_fetched": raw.index[0].strftime("%Y-%m-%d") if len(raw) else None,
        "earliest_existing": min(existing) if existing else None,
    }


def _insert_bars(db_path: str, rows: Sequence[Dict[str, Any]]) -> int:
    """Insert bars, never replacing an existing date."""

    from config import PRICE_BAR_RULE_VERSION

    connection = sqlite3.connect(db_path)
    try:
        connection.executemany(
            """INSERT OR IGNORE INTO bist100_prices
               (date, open, high, low, close, volume, daily_return, bar_status,
                bar_observed_at, bar_review_reason, bar_rule_version, source,
                retrieved_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (r["date"], r["open"], r["high"], r["low"], r["close"],
                 r["volume"], r["daily_return"], r["bar_status"],
                 r["bar_observed_at"], r["bar_review_reason"], PRICE_BAR_RULE_VERSION,
                 r["source"], r["retrieved_at"])
                for r in rows
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return len(rows)


def residual_coverage(db_path: str) -> Dict[str, int]:
    """Distinct primary-window sessions with a residual, per control set."""

    from research.return_windows import PRIMARY_WINDOW

    connection = sqlite3.connect(db_path)
    try:
        coverage = {}
        for column in ("residual_none", "residual_em_lagged",
                       "residual_em_oil_fx_lagged"):
            coverage[column] = connection.execute(
                f"""SELECT COUNT(DISTINCT first_reactable_session)
                      FROM event_research_dataset
                     WHERE window_name = ? AND eligibility_status = 'eligible'
                       AND is_tradable_window = 1 AND {column} IS NOT NULL""",
                (PRIMARY_WINDOW,),
            ).fetchone()[0]
        coverage["eligible_sessions"] = connection.execute(
            """SELECT COUNT(DISTINCT first_reactable_session)
                 FROM event_research_dataset
                WHERE window_name = ? AND eligibility_status = 'eligible'
                  AND is_tradable_window = 1 AND raw_return IS NOT NULL""",
            (PRIMARY_WINDOW,),
        ).fetchone()[0]
    finally:
        connection.close()
    return coverage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default="finance_sentiment.db")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace dates already stored (deliberate repair)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-bist", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    before = residual_coverage(args.db)
    factors = backfill_factors(
        args.db, start=args.start, overwrite=args.overwrite, dry_run=args.dry_run,
    )
    bist = (
        {"status": "skipped"} if args.skip_bist
        else backfill_bist(args.db, start=args.start, overwrite=args.overwrite,
                           dry_run=args.dry_run)
    )

    print(f"backfill from {args.start}  (dry_run={args.dry_run})")
    for symbol, stats in factors["symbols"].items():
        print(f"  {symbol:<12} {stats.get('added', 0):>5} added   "
              f"(earliest now {stats.get('earliest_fetched')}, "
              f"was {stats.get('earliest_existing')})")
    print(f"  {'XU100':<12} {bist.get('added', 0):>5} added   "
          f"(earliest now {bist.get('earliest_fetched')}, "
          f"was {bist.get('earliest_existing')})")

    after = before if args.dry_run else None
    print("\nresidual coverage is recomputed by the next events run; "
          "rerun scripts/run_validation or the pipeline to observe the gain.")
    print(f"  before: {before}")

    report = {
        "start": args.start, "dry_run": args.dry_run,
        "factors": factors, "bist": bist,
        "residual_coverage_before": before,
        "residual_coverage_after_rebuild": after,
        "transform_version": TRANSFORM_VERSION,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
