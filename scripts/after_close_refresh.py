"""Refresh market prices after the Istanbul close. Prices only.

The morning ingestion run fires before the open, so the bar it stores for the
current session is provisional by construction. This job exists to settle it:
it runs after the close plus the safety delay, refetches, and promotes the
provisional bar to complete.

It touches market data only. No feed is scraped, no scorer is constructed, and
no headline, score, label, experiment identity, exclusion or event row is read
for modification. That separation is deliberate -- an after-close job should not
be able to change the research record, so a failure here can only ever cost
price freshness.

The schedule is a coarse trigger; this module makes the real decision. A UTC
cron drifts an hour against Istanbul across daylight-saving changes and GitHub
fires schedules late, so the runtime guard re-checks the actual session close --
including official half-day early closes -- before touching anything.

Usage::

    python -m scripts.after_close_refresh
    python -m scripts.after_close_refresh --force      # ignore the time guard
    python -m scripts.after_close_refresh --check-only # report the decision
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
from config import DB_PATH
from price_bars import after_close_refresh_allowed

logger = logging.getLogger(__name__)

# Exit codes. A skipped run is a success: on a holiday there is nothing to do.
EXIT_REFRESHED = 0
EXIT_SKIPPED = 0
EXIT_FAILED = 1


def _price_snapshot(db_path: str) -> Dict[str, Any]:
    with db._conn(db_path) as con:
        rows = con.execute(
            "SELECT bar_status, COUNT(*) AS n FROM bist100_prices GROUP BY bar_status"
        ).fetchall()
        latest = con.execute("SELECT MAX(date) FROM bist100_prices").fetchone()[0]
    return {
        "status_counts": {str(r["bar_status"]): int(r["n"]) for r in rows},
        "latest_date": latest,
    }


def run(
    db_path: str = DB_PATH,
    *,
    force: bool = False,
    check_only: bool = False,
    now: Optional[str] = None,
) -> Dict[str, Any]:
    """Refresh prices and factors when the session has settled."""

    window = after_close_refresh_allowed(now)
    decision = {
        "allowed": window.allowed,
        "reason": window.reason,
        "session_date": window.session_date,
        "settles_at": window.settles_at.isoformat() if window.settles_at else None,
        "forced": force,
    }
    logger.info("After-close guard: %s (%s)", window.allowed, window.reason)

    if check_only or (not window.allowed and not force):
        return {"action": "skipped", "decision": decision}

    before = _price_snapshot(db_path)

    # Imported here so --check-only never pulls in the network stack.
    import pipeline

    prices = pipeline.prices_step(db_path=db_path, return_outcome=True)
    factors = pipeline.factors_step(db_path=db_path, return_outcome=True)

    after = _price_snapshot(db_path)
    return {
        "action": "refreshed",
        "decision": decision,
        "prices": {
            "status": prices.status,
            "rows": prices.count,
            "details": prices.details,
            "warnings": prices.warnings,
        },
        "factors": {
            "status": getattr(factors, "status", "unknown"),
            "rows": getattr(factors, "count", factors if isinstance(factors, int) else 0),
        },
        "before": before,
        "after": after,
        "bars_needing_review": db.list_price_bars_for_review(db_path=db_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--force", action="store_true",
                        help="refresh even if the session has not settled")
    parser.add_argument("--check-only", action="store_true",
                        help="report the guard decision and exit")
    parser.add_argument("--now", default=None,
                        help="override the current time (testing)")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    )
    args = build_parser().parse_args(argv)
    if not Path(args.db).exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return EXIT_FAILED

    try:
        result = run(
            args.db, force=args.force, check_only=args.check_only, now=args.now
        )
    except Exception as exc:                              # noqa: BLE001
        logger.error("After-close refresh failed: %s", exc)
        return EXIT_FAILED

    decision = result["decision"]
    if result["action"] == "skipped":
        print(
            f"SKIPPED: {decision['reason']} "
            f"(session {decision['session_date']}, settles {decision['settles_at']})"
        )
    else:
        prices = result["prices"]
        print(
            f"REFRESHED: {prices['rows']} price row(s) [{prices['status']}], "
            f"{result['factors']['rows']} factor row(s)"
        )
        print(f"  bar statuses: {result['after']['status_counts']}")
        print(f"  latest date : {result['after']['latest_date']}")
        if result["bars_needing_review"]:
            print(f"  needs review: {len(result['bars_needing_review'])} bar(s)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
    return EXIT_REFRESHED if result["action"] == "refreshed" else EXIT_SKIPPED


if __name__ == "__main__":
    raise SystemExit(main())
