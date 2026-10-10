"""Daily price limits: the rule by date, detection from daily bars, and a check
that the data agree with the rule.

Source for the rule
-------------------
Borsa Istanbul announcement 2020/20, "Daily Price Limits": effective
2020-03-13 the equity price margin became 10% for all market groups (it had
been 20% or 15% depending on the group) "until a further announcement".
Checked 2026-10-11; no later official change was found.

What is *not* known here
------------------------
* Before 2020-03-13 the limit depended on a stock's market group, which this
  project does not record per stock per date. Detection is refused there.
* Instruments with their own margins (rights coupons, real-estate
  certificates, some watchlist segments) are not distinguished.
* A change after the announcement that the search did not surface.

So the rule is never trusted alone. :func:`empirical_check` measures, on the
bars actually in use, how many sessions moved past the limit. If the rule is
right that count is close to zero outside corporate-action dates.

Touching and locking
--------------------
``touched``: the day's high (low) reached the limit price at some point.
``locked_close``: the close itself is at the limit and equals the day's
extreme. ``one_price``: open, high, low and close are the same price -- the
stock sat at the limit all day. Only ``locked_close`` defines a limit day for
the hypotheses; a touch that reversed is a different event.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

LIMIT_RULE_VERSION = "bist-equity-price-limit-v1"

LIMIT_RULES: List[Dict[str, Any]] = [
    {
        "from": "2020-03-13", "to": None, "limit": 0.10,
        "source": "Borsa Istanbul announcement 2020/20 'Daily Price Limits' "
                  "(https://borsaistanbul.com/files/2020-20-Daily-Price-Limits.pdf)",
        "verified": True,
        "note": "all equity market groups; in force until a further announcement",
    },
]
#: Limit prices are rounded to the tick, so an observed limit move is a little
#: under the nominal percentage. Measured against the close-to-close return.
TOLERANCE = 0.005


def limit_on(day: str, rules: Optional[List[Dict[str, Any]]] = None) -> Optional[float]:
    """The price limit in force on *day*, or None where no verified rule applies."""

    for rule in rules if rules is not None else LIMIT_RULES:
        if day >= rule["from"] and (rule["to"] is None or day <= rule["to"]):
            return rule["limit"] if rule.get("verified") else None
    return None


def detect(frame: pd.DataFrame, rules: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
    """Limit flags for each session of one aligned ticker frame.

    Sessions with a corporate action, a missing prior close, or no verified
    rule get no flags at all (False), and ``limit`` is NaN there.
    """

    limit = pd.Series([limit_on(d, rules) for d in frame.index], index=frame.index, dtype=float)
    move = frame["raw_ret"]                 # NaN on corporate-action dates
    usable = move.notna() & limit.notna()
    prior = frame["close"].shift(1)
    at_high = np.isclose(frame["close"], frame["high"], rtol=1e-6, atol=0)
    at_low = np.isclose(frame["close"], frame["low"], rtol=1e-6, atol=0)

    out = pd.DataFrame(index=frame.index)
    out["limit"] = limit.where(usable)
    out["up_touched"] = usable & (frame["high"] / prior - 1 >= limit - TOLERANCE)
    out["down_touched"] = usable & (frame["low"] / prior - 1 <= -(limit - TOLERANCE))
    out["up_locked_close"] = usable & (move >= limit - TOLERANCE) & at_high
    out["down_locked_close"] = usable & (move <= -(limit - TOLERANCE)) & at_low
    out["one_price"] = usable & at_high & at_low & np.isclose(
        frame["open"], frame["close"], rtol=1e-6, atol=0)
    flags = [c for c in out.columns if c != "limit"]
    out[flags] = out[flags].fillna(False).astype(bool)
    return out


def empirical_check(panel, rules: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """How well the bars agree with the rule: moves beyond the limit should be
    rare, and limit-sized moves should pile up just under it."""

    beyond = at_limit = sessions = 0
    worst: List[Dict[str, Any]] = []
    for ticker, frame in panel.bars.items():
        if ticker == panel.benchmark:
            continue
        limit = pd.Series([limit_on(d, rules) for d in frame.index], index=frame.index, dtype=float)
        move = frame["raw_ret"].abs()
        usable = move.notna() & limit.notna()
        sessions += int(usable.sum())
        over = usable & (move > limit + TOLERANCE)
        beyond += int(over.sum())
        at_limit += int((usable & (move >= limit - TOLERANCE) & ~over).sum())
        for day in frame.index[over.to_numpy()][:3]:
            worst.append({"ticker": ticker, "date": day, "move": float(frame.loc[day, "raw_ret"])})
    return {
        "rule_version": LIMIT_RULE_VERSION, "sessions_checked": sessions,
        "at_limit": at_limit, "beyond_limit": beyond,
        "beyond_share": (beyond / sessions) if sessions else None,
        "examples_beyond": worst[:15],
        "reading": "beyond_limit should be near zero if the rule and the "
                   "adjustment of the bars are both right",
    }


def streaks(flags: pd.Series) -> pd.Series:
    """Length of the current run of True at each position (0 where False).

    A missing session breaks nothing by itself here -- the caller passes flags
    on the benchmark's session grid, where a session the ticker did not trade
    is False and therefore ends the run.
    """

    values = flags.to_numpy(dtype=bool)
    run = np.zeros(len(values), dtype=int)
    for i, value in enumerate(values):
        run[i] = (run[i - 1] + 1 if i else 1) if value else 0
    return pd.Series(run, index=flags.index)
