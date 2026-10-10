"""The shared event-study engine.

Definitions
-----------
* **Day 0** is the first session able to act on the event
  (:mod:`stock_research.calendar`), never the publication date as such.
* **Return** on session *t* is the adjusted close-to-close simple return.
* **Expected return** comes from a market model fitted on the estimation
  window, which ends before the earliest event window begins. Nothing at or
  after day -10 enters the fit.
* **Abnormal return** ``AR_t = R_t - (alpha + beta * Rm_t)``.
* **CAR(a, b)** is the sum of ``AR_t`` for ``t`` in ``[a, b]``.
* **BHAR(a, b)** is ``prod(1 + R_t) - prod(1 + Rm_t)`` over the same sessions.

Missing data
------------
A window with any missing return has no CAR. It is reported as missing with
the reason, and the event stays in the table so the loss is countable. An
event whose estimation window has too few returns has no abnormal returns at
all under the market model; the market-adjusted variant (``alpha = 0``,
``beta = 1``) needs no estimation and is kept alongside as a sensitivity.

What this module does not do
----------------------------
It does not decide which events are independent. One row per event comes out;
inference over those rows must cluster (:mod:`stock_research.stats`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stock_research.config import (
    ESTIMATION_MIN_OBSERVATIONS, ESTIMATION_WINDOW, EVENT_WINDOWS,
)
from stock_research.data.prices import Panel
from stock_research.guards import SealedWindowError, assert_outcome_allowed

ENGINE_VERSION = "event-study-engine-v1"

STATUS_OK = "ok"
STATUS_NO_TICKER = "ticker_not_in_panel"
STATUS_NO_DAY0 = "day0_outside_calendar"
STATUS_HORIZON = "window_beyond_data"
STATUS_SEALED = "sealed_window"
STATUS_NO_ESTIMATION = "estimation_insufficient"

MODEL_MARKET = "market_model"
MODEL_MARKET_ADJUSTED = "market_adjusted"


def window_label(window: Tuple[int, int], prefix: str = "car") -> str:
    def part(v: int) -> str:
        return f"m{abs(v)}" if v < 0 else f"p{v}"
    return f"{prefix}_{part(window[0])}_{part(window[1])}"


def _check_windows(windows: Sequence[Tuple[int, int]],
                   estimation: Tuple[int, int]) -> None:
    earliest = min(a for a, _ in windows)
    if estimation[1] >= min(earliest, 0):
        raise ValueError(
            f"estimation window {estimation} overlaps the event windows "
            f"(earliest day {earliest}); expected returns would see the event"
        )
    if estimation[0] > estimation[1] or any(a > b for a, b in windows):
        raise ValueError("a window must run from its first day to its last")


@dataclass(frozen=True)
class MarketModel:
    alpha: float
    beta: float
    observations: int
    residual_sd: float


def fit_market_model(stock: np.ndarray, market: np.ndarray) -> Optional[MarketModel]:
    """OLS of stock on market returns over the rows where both exist."""

    keep = np.isfinite(stock) & np.isfinite(market)
    n = int(keep.sum())
    if n < ESTIMATION_MIN_OBSERVATIONS:
        return None
    x, y = market[keep], stock[keep]
    variance = float(np.var(x))
    if variance <= 0:
        return None
    beta = float(np.cov(x, y, bias=True)[0, 1] / variance)
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)
    return MarketModel(alpha, beta, n, float(resid.std(ddof=2)) if n > 2 else float("nan"))


def study_event(panel: Panel, ticker: str, day0: Optional[str], *,
                windows: Sequence[Tuple[int, int]] = EVENT_WINDOWS,
                estimation: Tuple[int, int] = ESTIMATION_WINDOW,
                sealed: Optional[str] = None) -> Dict[str, Any]:
    """Abnormal returns and CARs for one event. Never raises on bad data."""

    _check_windows(windows, estimation)
    row: Dict[str, Any] = {"status": STATUS_OK}
    frame = panel.frame(ticker)
    if frame is None:
        return {"status": STATUS_NO_TICKER}
    position = panel.calendar.index(day0) if day0 else None
    if position is None:
        return {"status": STATUS_NO_DAY0}

    # Index 0 has no return (no prior close), so a window must start after it.
    first = position + min(a for a, _ in windows)
    last = position + max(b for _, b in windows)
    if first < 1 or last >= len(panel.calendar.sessions):
        return {"status": STATUS_HORIZON}
    try:
        assert_outcome_allowed(panel.calendar.sessions[last], sealed=sealed)
    except SealedWindowError:
        return {"status": STATUS_SEALED}

    stock = frame["ret"].to_numpy()
    market = panel.market["ret"].to_numpy()

    # A short history shrinks the estimation sample; it never moves the window
    # toward the event. Too few returns means no model, not a shorter rule.
    est = slice(max(0, position + estimation[0]), max(0, position + estimation[1] + 1))
    model = fit_market_model(stock[est], market[est])
    row.update({
        "alpha": model.alpha if model else None,
        "beta": model.beta if model else None,
        "estimation_observations": model.observations if model else int(
            (np.isfinite(stock[est]) & np.isfinite(market[est])).sum()),
        "residual_sd": model.residual_sd if model else None,
    })
    if model is None:
        row["status"] = STATUS_NO_ESTIMATION

    for a, b in windows:
        span = slice(position + a, position + b + 1)
        r, m = stock[span], market[span]
        complete = bool(np.isfinite(r).all() and np.isfinite(m).all())
        adjusted = float((r - m).sum()) if complete else None
        row[window_label((a, b), "mar")] = adjusted           # market-adjusted
        row[window_label((a, b), "car")] = (
            float((r - (model.alpha + model.beta * m)).sum())
            if complete and model else None
        )
        row[window_label((a, b), "bhar")] = (
            float(np.prod(1 + r) - np.prod(1 + m)) if complete else None
        )
        row[window_label((a, b), "raw")] = float(np.prod(1 + r) - 1) if complete else None
    return row


def event_table(panel: Panel, events: Iterable[Dict[str, Any]], *,
                windows: Sequence[Tuple[int, int]] = EVENT_WINDOWS,
                estimation: Tuple[int, int] = ESTIMATION_WINDOW,
                sealed: Optional[str] = None) -> pd.DataFrame:
    """One row per event: its identifying fields plus the study outputs.

    Each event needs ``ticker`` and ``day0``. Everything else it carries is
    passed through untouched.
    """

    rows: List[Dict[str, Any]] = []
    for event in events:
        study = study_event(panel, event["ticker"], event.get("day0"),
                            windows=windows, estimation=estimation, sealed=sealed)
        rows.append({**event, **study})
    return pd.DataFrame(rows)


def collapse_same_day(events: Sequence[Dict[str, Any]], *,
                      keys: Sequence[str] = ("ticker", "day0")) -> List[Dict[str, Any]]:
    """Merge events that share one outcome into one observation.

    Two disclosures from one issuer that first act on the same session have a
    single return between them. Kept as two rows they would be two
    observations of one number. The merged row records how many events it
    stands for and every distinct category among them, so a test that needs an
    unconfounded category can require ``n_categories == 1``.
    """

    merged: Dict[tuple, Dict[str, Any]] = {}
    for event in events:
        key = tuple(event.get(k) for k in keys)
        if key not in merged:
            merged[key] = {**event, "n_events": 0, "categories": [], "event_ids": []}
        row = merged[key]
        row["n_events"] += 1
        row["event_ids"].append(event.get("event_id"))
        category = event.get("category")
        if category is not None and category not in row["categories"]:
            row["categories"].append(category)
    for row in merged.values():
        row["n_categories"] = len(row["categories"])
        row["category"] = row["categories"][0] if row["n_categories"] == 1 else (
            "mixed" if row["categories"] else None)
    return list(merged.values())


def flag_overlaps(table: pd.DataFrame, horizon: int, calendar) -> pd.Series:
    """True where an earlier event of the same ticker lies within *horizon*
    sessions before this one, so their post-event windows overlap."""

    position = table["day0"].map(lambda d: calendar.index(d) if isinstance(d, str) else None)
    overlap = pd.Series(False, index=table.index)
    for _, group in table.assign(_pos=position).dropna(subset=["_pos"]).groupby("ticker"):
        ordered = group.sort_values("_pos")
        gaps = ordered["_pos"].diff()
        overlap.loc[ordered.index[(gaps <= horizon).fillna(False).to_numpy()]] = True
    return overlap
