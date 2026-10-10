"""Transaction costs and execution feasibility.

An abnormal return is a statistic. A trading profit needs a price someone
could actually have traded at, and a cost. This module keeps the two apart.

The cost numbers are **scenarios**. Nothing here was measured from a broker
statement or an order book, and the report says so wherever a net figure
appears.

Feasibility with daily bars
---------------------------
Daily OHLC cannot show the order book, so feasibility is decided by rule and
every rule errs toward "not filled":

* no bar, no volume, or no opening print on the entry session -> not filled;
* an entry session that opened and stayed at one price all day
  (``open == high == low == close``) and moved by about a full price limit
  from the prior close -> **locked**, not filled. One side of the book was
  empty; which side an order would have needed cannot be seen.
* entry at a session's high or low is never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from stock_research.config import ROUND_TRIP_COST_SCENARIOS

COST_MODEL_VERSION = "cost-scenarios-v1"

FILL_OK = "filled"
FILL_NO_BAR = "no_bar"
FILL_NO_VOLUME = "no_volume"
FILL_NO_OPEN = "no_opening_print"
FILL_LOCKED = "locked_at_one_price"


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost as a fraction of traded value, by component.

    The components exist so an assumption can be changed in one place and
    named in the output. Their defaults sum to the middle scenario.
    """

    commission: float = 0.0010      # both sides
    exchange_fees: float = 0.0002
    spread: float = 0.0015          # one full spread crossed over the round trip
    slippage: float = 0.0008
    price_impact: float = 0.0005
    label: str = "component_default"

    @property
    def round_trip(self) -> float:
        return (self.commission + self.exchange_fees + self.spread
                + self.slippage + self.price_impact)


def scenarios() -> Dict[str, float]:
    """The registered sensitivity scenarios. Assumptions, not measurements."""

    return {f"round_trip_{int(round(c * 10000))}bp": c for c in ROUND_TRIP_COST_SCENARIOS}


def entry_fill(frame: Optional[pd.DataFrame], session: Optional[str], *,
               limit: Optional[float] = None, tolerance: float = 0.004) -> str:
    """Whether an order at *session*'s open could plausibly have traded."""

    if frame is None or session is None or session not in frame.index:
        return FILL_NO_BAR
    bar = frame.loc[session]
    if not np.isfinite(bar.get("close", np.nan)):
        return FILL_NO_BAR
    if not np.isfinite(bar.get("open", np.nan)):
        return FILL_NO_OPEN
    if not np.isfinite(bar.get("volume", np.nan)) or bar["volume"] <= 0:
        return FILL_NO_VOLUME
    one_price = bar["open"] == bar["high"] == bar["low"] == bar["close"]
    move = bar.get("raw_ret", np.nan)
    if one_price and limit is not None and np.isfinite(move) and abs(move) >= limit - tolerance:
        return FILL_LOCKED
    return FILL_OK


def net_summary(gross: Sequence[float], *, trades_per_observation: int = 1) -> Dict[str, Any]:
    """Gross mean and the same mean after each cost scenario.

    ``gross`` holds one signed return per executed trade. A value that could
    not be executed must not be in it; the caller reports coverage separately.
    """

    values = np.asarray([g for g in gross if g is not None and np.isfinite(g)], dtype=float)
    out: Dict[str, Any] = {
        "cost_model_version": COST_MODEL_VERSION,
        "trades": int(len(values)),
        "gross_mean": float(values.mean()) if len(values) else None,
        "note": "cost scenarios are assumptions, not measured fees",
        "net_mean": {},
    }
    for name, cost in scenarios().items():
        out["net_mean"][name] = (
            float(values.mean() - cost * trades_per_observation) if len(values) else None
        )
    return out


def max_drawdown(returns: Sequence[float]) -> Optional[float]:
    """Largest peak-to-trough loss of a compounded sequence of trade returns."""

    values = np.asarray(list(returns), dtype=float)
    if not len(values):
        return None
    curve = np.cumprod(1 + values)
    peaks = np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
    return float((curve / peaks - 1).min())
