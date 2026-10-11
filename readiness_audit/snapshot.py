"""A consistent, read-only copy of the research database, and the sample's
structure read from it.

"Structure" means: which issuer, which first actionable session, which
category, which anchor, and whether the returns an event would need *exist*.
No event-window return value is read. ``availability`` looks at ``notna()``
only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from stock_research import events as sr_events
from stock_research.config import (
    BENCHMARK_TICKER, ESTIMATION_MIN_OBSERVATIONS, ESTIMATION_WINDOW, EVENT_WINDOWS,
    REPOSITORY_ROOT, STOCK_DB_PATH,
)
from stock_research.data import prices
from stock_research.hypotheses import h7
from stock_research.hypotheses.common import Context

AUDIT_OUT = REPOSITORY_ROOT / "outputs" / "stock_research" / "readiness"
SNAPSHOT_DB = AUDIT_OUT / "audit_snapshot.db"
#: Last session before the first sampled event. Everything up to here is
#: "pre-sample": no sampled disclosure had been published yet.
PRE_SAMPLE_END = "2022-12-30"


def take_snapshot(source: Path = STOCK_DB_PATH, target: Path = SNAPSHOT_DB) -> Path:
    """Copy the database with SQLite's backup API (consistent while a writer
    is active). The source is opened read-only."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    src = sqlite3.connect(f"file:{Path(source).as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return target


def load(db: Path = SNAPSHOT_DB):
    """Panel and KAP event table from the snapshot."""

    with sqlite3.connect(str(db)) as con:
        snapshot_id = con.execute(
            "SELECT snapshot_id FROM sr_price_availability GROUP BY snapshot_id "
            "ORDER BY MAX(retrieved_at) DESC LIMIT 1").fetchone()[0]
    panel = prices.load_panel(snapshot_id, BENCHMARK_TICKER, str(db))
    details = sr_events.load_details(str(db))
    table = sr_events.build_kap_events(details, panel)
    table["anchor"] = details["block_seq"].to_numpy()
    return panel, table, details


def availability(panel, ticker: str, day0: str) -> Dict[str, bool]:
    """Whether the registered study *could* produce CAR(0,+5) for an event.

    Mirrors ``eventstudy.study_event`` using only the presence of returns:
    the horizon rule over all registered windows, the 60-observation minimum
    in the estimation window, and a complete [0,+5] window.
    """

    frame = panel.frame(ticker)
    position = panel.calendar.index(day0) if isinstance(day0, str) else None
    out = {"in_panel": frame is not None, "day0_in_calendar": position is not None,
           "horizon": False, "estimation": False, "window_0_5": False}
    if frame is None or position is None:
        return out
    first = position + min(a for a, _ in EVENT_WINDOWS)
    last = position + max(b for _, b in EVENT_WINDOWS)
    out["horizon"] = first >= 1 and last < len(panel.calendar.sessions)
    if not out["horizon"]:
        return out
    have = frame["ret"].notna().to_numpy() & panel.market["ret"].notna().to_numpy()
    est = have[max(0, position + ESTIMATION_WINDOW[0]):max(0, position + ESTIMATION_WINDOW[1] + 1)]
    out["estimation"] = int(est.sum()) >= ESTIMATION_MIN_OBSERVATIONS
    out["window_0_5"] = bool(have[position:position + 6].all())
    return out


def h7_structure(panel, table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The events H7's primary test would use, by the registered eligibility
    rules, with no return read. Returns the event rows and the counts along
    the way."""

    eligible, counts = h7.prepare(Context(panel=panel, kap_events=table))
    anchors = table.set_index("event_id")["anchor"]
    eligible = eligible.copy()
    eligible["anchor"] = eligible["event_id"].map(anchors)
    flags = pd.DataFrame([availability(panel, r.ticker, r.day0) for r in eligible.itertuples()],
                         index=eligible.index)
    eligible = pd.concat([eligible, flags], axis=1)
    eligible["usable"] = flags.all(axis=1)
    counts["price_and_estimation_available"] = int(eligible["usable"].sum())
    counts["lost_to_availability"] = {c: int((~flags[c]).sum()) for c in flags.columns}
    usable = eligible[eligible["usable"]].copy()
    usable["position"] = usable["day0"].map(panel.calendar.index).astype(int)
    return usable[["event_id", "anchor", "ticker", "day0", "position", "category", "bucket",
                   "n_events"]], counts


def pre_sample_calibration(panel) -> Dict[str, Any]:
    """Volatility and dependence of market-model residuals in the pre-sample
    period only (through 2022-12-30, before any sampled disclosure).

    Used to choose simulation parameters. It conditions on no event and reads
    nothing from 2023.
    """

    sessions = [s for s in panel.calendar.sessions if s <= PRE_SAMPLE_END]
    market = panel.market.loc[sessions, "ret"]
    residuals = {}
    for ticker, frame in panel.bars.items():
        if ticker == panel.benchmark:
            continue
        both = pd.concat([frame.loc[sessions, "ret"], market], axis=1, keys=["r", "m"]).dropna()
        if len(both) < 60:
            continue
        beta = np.cov(both["m"], both["r"], bias=True)[0, 1] / np.var(both["m"])
        alpha = both["r"].mean() - beta * both["m"].mean()
        residuals[ticker] = (both["r"] - alpha - beta * both["m"]).reindex(sessions)
    resid = pd.DataFrame(residuals)
    sd = resid.std()
    standardised = resid / sd
    # Share of residual variance common to all stocks on a day: variance of the
    # cross-sectional mean of standardised residuals, less its sampling part.
    daily_mean = standardised.mean(axis=1)
    n_bar = standardised.notna().sum(axis=1).mean()
    common_share = max(0.0, float(daily_mean.var() - 1.0 / n_bar))
    ar1 = float(np.nanmean([s.dropna().autocorr(1) for _, s in standardised.items()]))
    kurt = float(np.nanmedian(standardised.kurt()))
    betas = {}
    for ticker in resid.columns:
        both = pd.concat([panel.bars[ticker].loc[sessions, "ret"], market], axis=1,
                         keys=["r", "m"]).dropna()
        betas[ticker] = np.cov(both["m"], both["r"], bias=True)[0, 1] / np.var(both["m"])
        betas[ticker] = (betas[ticker], 1 - resid[ticker].var() / both["r"].var())
    beta_values = np.array([b for b, _ in betas.values()])
    r2_values = np.array([r for _, r in betas.values()])
    return {
        "period": [sessions[0], sessions[-1]], "sessions": len(sessions),
        "tickers": int(resid.shape[1]),
        "residual_sd_quantiles": {q: float(sd.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "residual_sd_by_ticker": sd.to_dict(),
        "common_daily_share": common_share,
        "mean_ar1": ar1, "median_excess_kurtosis": kurt,
        "beta_quantiles": {q: float(np.quantile(beta_values, q)) for q in (0.1, 0.5, 0.9)},
        "r2_quantiles": {q: float(np.quantile(r2_values, q)) for q in (0.1, 0.5, 0.9)},
        "market_sd": float(market.std()),
    }
