"""Refetch specific daily price bars and replace incomplete ones.

Used to repair a bar that was captured mid-session and stored as if it were
that day's close. Only price fields are touched: headline scores, labels,
experiment identities, exclusions and session assignments are never read or
written here.

The refetched bar is classified like any other, so a repair attempted while the
market is open will store a provisional bar rather than silently repeating the
original fault. A settled replacement is recorded as ``corrected`` so the
repair stays visible afterwards.

Usage::

    python -m scripts.refresh_price_bar --db copy.db --date 2026-07-31
    python -m scripts.refresh_price_bar --db copy.db --date 2026-07-31 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
from config import BIST100_TICKER, DB_PATH

_PRICE_FIELDS = ("open", "high", "low", "close", "volume", "daily_return")


def _stored_bar(db_path: str, day: str) -> Optional[Dict[str, Any]]:
    with db._conn(db_path) as con:
        row = con.execute(
            "SELECT * FROM bist100_prices WHERE date = ?", (day,)
        ).fetchone()
    return dict(row) if row else None


def fetch_bars(
    dates: List[str], *, ticker: str = BIST100_TICKER
) -> pd.DataFrame:
    """Download a window covering *dates* and return the matching daily bars.

    Returns are recomputed on the complete downloaded series before filtering,
    so a repaired bar's daily return reflects the true preceding close rather
    than whichever neighbour happens to be stored.
    """
    import yfinance as yf

    start = (pd.Timestamp(min(dates)) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(dates)) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=True
    )
    if raw.empty:
        raise RuntimeError(f"provider returned no rows for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    frame = pd.DataFrame({
        "date": raw.index.strftime("%Y-%m-%d"),
        "open": raw["Open"].values,
        "high": raw["High"].values,
        "low": raw["Low"].values,
        "close": raw["Close"].values,
        "volume": raw["Volume"].values if "Volume" in raw.columns else None,
        "daily_return": raw["Close"].pct_change().mul(100).values,
    })
    return frame[frame["date"].isin(dates)].reset_index(drop=True)


def refresh(
    dates: List[str],
    db_path: str = DB_PATH,
    *,
    ticker: str = BIST100_TICKER,
    dry_run: bool = False,
) -> Dict[str, Any]:
    before = {day: _stored_bar(db_path, day) for day in dates}
    fetched = fetch_bars(dates, ticker=ticker)
    if fetched.empty:
        raise RuntimeError(f"provider returned no bar for {dates}")

    counts = None
    if not dry_run:
        counts = db.upsert_prices(fetched, db_path=db_path, mark_corrected=True)
    after = {day: _stored_bar(db_path, day) for day in dates}

    comparisons = []
    for day in dates:
        old, new = before.get(day), after.get(day)
        provider = fetched[fetched["date"] == day]
        provider_row = provider.iloc[0].to_dict() if not provider.empty else None
        changes = {}
        for field in _PRICE_FIELDS:
            old_value = None if old is None else old.get(field)
            new_value = (
                provider_row.get(field) if dry_run and provider_row
                else (None if new is None else new.get(field))
            )
            if old_value != new_value:
                changes[field] = {"before": old_value, "after": new_value}
        comparisons.append({
            "date": day,
            "status_before": None if old is None else old.get("bar_status"),
            "status_after": None if new is None else new.get("bar_status"),
            "review_before": None if old is None else old.get("bar_review_reason"),
            "review_after": None if new is None else new.get("bar_review_reason"),
            "changes": changes,
        })
    return {"dry_run": dry_run, "bars": comparisons, "upsert_counts": counts}


def format_report(result: Dict[str, Any]) -> str:
    lines = ["Price-bar refresh" + (" (dry run)" if result["dry_run"] else "")]
    for bar in result["bars"]:
        lines.append(f"\n  {bar['date']}")
        lines.append(
            f"    status  {bar['status_before']} -> {bar['status_after']}"
        )
        lines.append(
            f"    review  {bar['review_before']} -> {bar['review_after']}"
        )
        if not bar["changes"]:
            lines.append("    values  unchanged")
        for field, change in bar["changes"].items():
            lines.append(
                f"    {field:<13} {change['before']} -> {change['after']}"
            )
    if result["upsert_counts"]:
        lines.append(f"\n  upsert: {result['upsert_counts']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--date", action="append", required=True,
                        help="ISO date to refresh; repeatable")
    parser.add_argument("--ticker", default=BIST100_TICKER)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.db).exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 2
    result = refresh(
        args.date, db_path=args.db, ticker=args.ticker, dry_run=args.dry_run
    )
    print(format_report(result))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
