"""Compatibility entry point for the transparent polarization inference module.

New code should import :mod:`analysis.polarization.inference`.  The historical
module name and camp constants remain available, but executing this file now
runs the non-causal selection-versus-framing analysis and does not overwrite a
research figure.
"""

from __future__ import annotations

import sqlite3
import sys
from typing import Sequence

import pandas as pd

from analysis.polarization.inference import (
    DEFAULT_OPPOSITION_SOURCES,
    DEFAULT_PRO_GOVERNMENT_SOURCES,
    load_headlines,
    main as inference_main,
)
from config import DB_PATH


PRO_GOV = list(DEFAULT_PRO_GOVERNMENT_SOURCES)
OPPOSITION = list(DEFAULT_OPPOSITION_SOURCES)
MARKET = ["bloomberght", "investing_tr_economy"]


def load(db_path=DB_PATH):
    """Retain the historical ``(headlines, USD/TRY factors)`` loader shape."""

    headlines = load_headlines(db_path).rename(columns={"sentiment": "s"})
    headlines["date"] = pd.to_datetime(headlines["date"])
    with sqlite3.connect(str(db_path)) as connection:
        try:
            fx = pd.read_sql_query(
                "SELECT date, close FROM market_factors "
                "WHERE symbol='USDTRY=X' ORDER BY date",
                connection,
            )
        except (sqlite3.DatabaseError, pd.errors.DatabaseError):
            fx = pd.DataFrame(columns=["date", "close"])
    fx["date"] = pd.to_datetime(fx["date"])
    return headlines, fx


def camp(headlines, sources):
    """Historical convenience filter retained for downstream notebooks."""

    return headlines[headlines["source"].isin(sources)]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--db" not in args:
        args[0:0] = ["--db", str(DB_PATH)]
    return inference_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
