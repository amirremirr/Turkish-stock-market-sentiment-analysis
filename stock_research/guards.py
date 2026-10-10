"""Guards that refuse instead of warning.

The sealed-window guard exists because of the index study. Its untouched test
(``untouched_future_v1``) forbids reading index outcomes from 2026-08-10 until
its single result is recorded. A stock's return around a headline is close
enough to that target that computing it early would be a look at the sealed
sample from the side.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from stock_research.config import REPOSITORY_ROOT, SEALED_INDEX_BOUNDARY

INDEX_DB = REPOSITORY_ROOT / "finance_sentiment.db"


class SealedWindowError(RuntimeError):
    """An outcome inside the index study's sealed window was requested."""


def index_result_recorded(index_db: Optional[str | Path] = None) -> bool:
    """Whether the sealed index test has produced its one result."""

    path = Path(index_db or INDEX_DB)
    if not path.exists():
        return False
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM future_validation_results"
        ).fetchone()
        return bool(row and row[0])
    except sqlite3.OperationalError:
        return False                     # table absent: nothing recorded
    finally:
        connection.close()


def sealed_from(index_db: Optional[str | Path] = None) -> Optional[str]:
    """First session whose outcomes are off limits, or None once lifted."""

    return None if index_result_recorded(index_db) else SEALED_INDEX_BOUNDARY


def assert_outcome_allowed(last_outcome_date: str, *, sealed: Optional[str]) -> None:
    if sealed is not None and str(last_outcome_date)[:10] >= sealed:
        raise SealedWindowError(
            f"outcome on {last_outcome_date} lies inside the sealed index window "
            f"(from {sealed}); it stays unread until untouched_future_v1 is recorded"
        )
