"""Corporate-action and adjustment checks on the stored bars.

    python -m readiness_audit.price_checks

Works on single-session price ratios at corporate-action dates and at dates
the cleaning rules flagged. No event of the disclosure sample is looked up, no
market model is fitted, and no event-window quantity is computed.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from readiness_audit import snapshot as snap
from stock_research import pipeline
from stock_research.config import BENCHMARK_TICKER
from stock_research.data import prices
from stock_research.limits import limit_on


def raw_bars(db=None) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db or snap.SNAPSHOT_DB}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            "SELECT ticker, date, open, high, low, close, adj_close, volume, dividend, "
            "split_ratio FROM sr_raw_price_bars ORDER BY ticker, date", con)
    finally:
        con.close()


def action_table(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per recorded dividend or split, with the price ratios around it."""

    rows = []
    for ticker, frame in bars.groupby("ticker"):
        if ticker == BENCHMARK_TICKER:
            continue
        frame = frame.reset_index(drop=True)
        traded = frame[(frame["volume"].fillna(0) > 0)].reset_index(drop=True)
        hit = traded.index[(traded["dividend"].fillna(0) != 0) | (traded["split_ratio"].fillna(0) != 0)]
        for i in hit:
            if i == 0:
                continue
            now, before = traded.loc[i], traded.loc[i - 1]
            close_ratio = now["close"] / before["close"]
            adj_ratio = now["adj_close"] / before["adj_close"]
            dividend = float(now["dividend"] or 0.0)
            # What the adjusted ratio should be if the dividend is taken off
            # the prior close (the split is already in ``close``).
            expected = now["close"] / (before["close"] - dividend) if before["close"] > dividend else np.nan
            rows.append({
                "ticker": ticker, "date": now["date"], "dividend": dividend,
                "split_ratio": float(now["split_ratio"] or 0.0),
                "close_ratio": close_ratio, "adj_ratio": adj_ratio,
                "expected_adj_ratio": expected,
                "dividend_yield": dividend / before["close"] if before["close"] else np.nan,
                "limit": limit_on(now["date"]),
            })
    out = pd.DataFrame(rows)
    out["adj_return"] = out["adj_ratio"] - 1
    out["adjustment_error"] = out["adj_ratio"] / out["expected_adj_ratio"] - 1
    return out


def summarise(db=None) -> Dict[str, Any]:
    bars = raw_bars(db)
    actions = action_table(bars)
    panel = prices.load_panel(f"yahoo-{pipeline.PRICE_START}-{pipeline.PRICE_END}",
                              BENCHMARK_TICKER, db or snap.SNAPSHOT_DB)

    flagged = pd.DataFrame([
        {"ticker": t, **issue} for t, issues in panel.issues.items() for issue in issues
        if issue["issue"] in ("unexplained_jump", "move_beyond_price_limit")])
    flagged = flagged.drop_duplicates(["ticker", "date", "issue"]) if len(flagged) else flagged
    by_date = flagged.groupby("date")["ticker"].nunique().sort_values(ascending=False) \
        if len(flagged) else pd.Series(dtype=int)

    splits = actions[actions["split_ratio"] != 0]
    dividends = actions[(actions["dividend"] != 0) & (actions["split_ratio"] == 0)]
    limit = actions["limit"].fillna(0.10)
    out = {
        "tickers_with_bars": int(bars["ticker"].nunique() - 1),
        "split_records": int(len(splits)),
        "split_tickers": int(splits["ticker"].nunique()),
        "split_days_adjusted_move_beyond_limit": int(
            (splits["adj_return"].abs() > limit.loc[splits.index] + 0.005).sum()),
        "split_days_adjusted_move_beyond_25pct": int((splits["adj_return"].abs() > 0.25).sum()),
        "dividend_records": int(len(dividends)),
        "dividend_tickers": int(dividends["ticker"].nunique()),
        "dividend_adjustment_error_abs_quantiles": {
            q: float(dividends["adjustment_error"].abs().quantile(float(q)))
            for q in ("0.5", "0.9", "0.99", "1.0")},
        "dividend_days_adjustment_error_over_1pct": int(
            (dividends["adjustment_error"].abs() > 0.01).sum()),
        "dividend_days_adjusted_move_beyond_limit": int(
            (dividends["adj_return"].abs() > limit.loc[dividends.index] + 0.005).sum()),
        "flagged_returns_total": int(len(flagged)),
        "flagged_by_issue": flagged["issue"].value_counts().to_dict() if len(flagged) else {},
        "flagged_dates_with_10_or_more_tickers": {d: int(n) for d, n in by_date.items() if n >= 10},
        "flagged_isolated": int((by_date < 10).sum()) if len(by_date) else 0,
    }
    actions.to_csv(snap.AUDIT_OUT / "corporate_actions.csv", index=False)
    flagged.to_csv(snap.AUDIT_OUT / "flagged_returns.csv", index=False)
    (snap.AUDIT_OUT / "price_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(summarise(), indent=1))
