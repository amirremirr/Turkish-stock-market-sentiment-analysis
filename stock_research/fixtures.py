"""Synthetic panels with known answers.

Used for two things only: testing that an estimator recovers an effect that
was put there (and finds nothing where nothing was put), and letting the
report render when real data are absent. Every panel built here carries
``origin = "synthetic"`` and nothing derived from it may be reported as a
finding.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stock_research.data.prices import ORIGIN_SYNTHETIC, Panel, build_panel

BENCHMARK = "MKT"


def weekdays(start: str, count: int, *, skip: Sequence[str] = ()) -> List[str]:
    day, out, skipped = date.fromisoformat(start), [], set(skip)
    while len(out) < count:
        if day.weekday() < 5 and day.isoformat() not in skipped:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def bars_from_returns(sessions: Sequence[str], returns: np.ndarray, *, start_price: float = 10.0,
                      volume: Optional[np.ndarray] = None, intraday_share: float = 0.5,
                      rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """OHLCV consistent with close-to-close *returns*. The overnight gap takes
    ``1 - intraday_share`` of each return; highs and lows bracket both."""

    close = start_price * np.cumprod(1 + returns)
    prior = np.concatenate([[start_price], close[:-1]])
    open_ = prior * (1 + returns * (1 - intraday_share))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    if volume is None:
        generator = rng or np.random.default_rng(0)
        volume = generator.integers(200_000, 2_000_000, size=len(sessions)).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "adj_close": close,
        "volume": volume, "dividend": 0.0, "split_ratio": 0.0,
    }, index=list(sessions))


def synthetic_panel(*, tickers: int = 40, sessions: int = 420, seed: int = 1,
                    start: str = "2022-01-03",
                    effects: Optional[Dict[Tuple[str, str], Dict[int, float]]] = None,
                    market_sd: float = 0.012, idio_sd: float = 0.02,
                    sector_sd: float = 0.0, sectors: int = 4) -> Panel:
    """A factor-model panel. ``effects`` maps ``(ticker, day0)`` to
    ``{relative_day: abnormal_return}``, added on top of the model return."""

    rng = np.random.default_rng(seed)
    days = weekdays(start, sessions)
    market = rng.normal(0.0003, market_sd, sessions)
    sector_shocks = rng.normal(0, sector_sd, (sectors, sessions)) if sector_sd else None
    frames = {BENCHMARK: bars_from_returns(days, market, start_price=1000.0, rng=rng)}
    index = {d: i for i, d in enumerate(days)}
    for n in range(tickers):
        name = f"S{n:03d}"
        beta = rng.uniform(0.6, 1.4)
        returns = beta * market + rng.normal(0, idio_sd, sessions)
        if sector_shocks is not None:
            returns = returns + sector_shocks[n % sectors]
        for (ticker, day0), path in (effects or {}).items():
            if ticker == name and day0 in index:
                for relative, value in path.items():
                    target = index[day0] + relative
                    if 0 <= target < sessions:
                        returns[target] += value
        frames[name] = bars_from_returns(days, returns, rng=rng)
    return build_panel(frames, BENCHMARK, origin=ORIGIN_SYNTHETIC,
                       snapshot_id=f"synthetic-seed{seed}")


def random_events(panel: Panel, count: int, *, seed: int = 2, first: int = 150,
                  last_margin: int = 25, per_day: int = 1) -> List[Dict[str, str]]:
    """``count`` events on random tickers. ``per_day`` > 1 stacks several
    events on each chosen session, which is what makes clustering matter."""

    rng = np.random.default_rng(seed)
    names = [t for t in panel.bars if t != panel.benchmark]
    sessions = panel.calendar.sessions
    events = []
    while len(events) < count:
        day = sessions[int(rng.integers(first, len(sessions) - last_margin))]
        for ticker in rng.choice(names, size=min(per_day, len(names)), replace=False):
            if len(events) < count:
                events.append({"event_id": f"E{len(events):05d}", "ticker": str(ticker),
                               "day0": day})
    return events
