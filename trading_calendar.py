"""
BIST trading-calendar helpers — weekend + holiday aware.

signal_date answers: "which trading session is the FIRST that can react to
this headline?" That is the day the headline's sentiment should be aligned to
when testing predictive power (METHODOLOGY: temporal alignment).

Rules (Istanbul time, BIST session ~10:00-18:00):
  - published on a non-trading day        -> next trading day
  - published_hour is NULL (unknown time) -> next trading day (conservative:
        the headline might be post-close; assigning the same session would
        inject lookahead into signal-aligned statistics)
  - hour <= 18 (pre-open or intraday)     -> same day
  - hour  > 18 (post-close)               -> next trading day
"""

from datetime import date, timedelta
from typing import Optional

from config import BIST_HOLIDAYS

_HOLIDAYS = set(BIST_HOLIDAYS)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in _HOLIDAYS


def next_trading_day(d: date) -> date:
    n = d + timedelta(days=1)
    while not is_trading_day(n):
        n += timedelta(days=1)
    return n


def signal_date(published_at: str, published_hour: Optional[int]) -> str:
    """Map (publish date, Istanbul hour) -> ISO date of the first session that
    can price the news."""
    d = date.fromisoformat(str(published_at)[:10])
    if not is_trading_day(d):
        return next_trading_day(d).isoformat()
    if published_hour is None:
        return next_trading_day(d).isoformat()
    if int(published_hour) <= 18:
        return d.isoformat()
    return next_trading_day(d).isoformat()
