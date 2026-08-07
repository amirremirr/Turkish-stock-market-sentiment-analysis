"""Borsa Istanbul trading-session assignment helpers.

The predictive signal date is the first session able to react to a publication.
Pre-open publications on a trading day can affect that day's opening session.
Post-close, non-trading-day, and unknown-time publications are assigned
conservatively to the next available session.

All timestamps are normalized to Europe/Istanbul before classification.  A
naive timestamp is treated as already being in Istanbul local time, matching
the convention used by the Turkish source feeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal, Optional, Union
from zoneinfo import ZoneInfo

from config import BIST_HALF_DAYS, BIST_HOLIDAYS


ISTANBUL = ZoneInfo("Europe/Istanbul")
REGULAR_SESSION_OPEN = time(10, 0)
REGULAR_SESSION_CLOSE = time(18, 10)

TimingBucket = Literal[
    "pre_open",
    "during_session",
    "post_close",
    "weekend_or_holiday",
    "unknown",
]


_HALF_DAY_CLOSES = {
    date.fromisoformat(day): time.fromisoformat(close)
    for day, close in BIST_HALF_DAYS.items()
}
_FULL_DAY_HOLIDAYS = set(BIST_HOLIDAYS)


@dataclass(frozen=True)
class SessionAssignment:
    """Result of assigning one publication to a tradable session.

    ``published_at_istanbul`` is ``None`` when the timestamp's time component
    is absent or invalid.  ``signal_date`` remains an ISO string so callers can
    store it directly in the existing SQLite schema.
    """

    signal_date: str
    timing_bucket: TimingBucket
    published_at_istanbul: Optional[datetime]


DateLike = Union[str, date, datetime]
TimestampLike = Union[str, datetime]


def is_trading_day(day: date) -> bool:
    """Return whether *day* has a Borsa Istanbul trading session."""

    return day.weekday() < 5 and day.isoformat() not in _FULL_DAY_HOLIDAYS


def session_close(day: date) -> time:
    """Return the scheduled local close for a trading day."""

    return _HALF_DAY_CLOSES.get(day, REGULAR_SESSION_CLOSE)


def next_trading_day(day: date) -> date:
    """Return the first trading day strictly after *day*."""

    candidate = day + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def previous_trading_day(day: date) -> date:
    """Return the last trading day strictly before *day*.

    The counterpart to :func:`next_trading_day`. A return window that opens at a
    session's open needs the close that preceded it, and "yesterday" is the
    wrong answer across weekends and the nine-day religious holidays Borsa
    Istanbul observes.
    """

    candidate = day - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def _next_session_on_or_after(day: date) -> date:
    candidate = day
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def _parse_date(value: Optional[DateLike]) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Optional[TimestampLike]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        # A date without a time deliberately represents an unknown timestamp.
        if not raw or "T" not in raw and " " not in raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ISTANBUL)
    return parsed.astimezone(ISTANBUL)


def assign_trading_session(
    published_timestamp: Optional[TimestampLike],
    published_date: Optional[DateLike] = None,
) -> SessionAssignment:
    """Assign a publication to a trading session and timing bucket.

    Args:
        published_timestamp: ISO timestamp or ``datetime``.  Aware values are
            converted to Europe/Istanbul; naive values are interpreted there.
            ``None``, a date-only string, or an invalid string has unknown time.
        published_date: Optional date fallback for unknown/invalid timestamps.

    Raises:
        ValueError: if neither input supplies a usable publication date.  A
            date anchor is necessary to choose a deterministic next session.
    """

    local_timestamp = _parse_timestamp(published_timestamp)
    anchor_date = (
        local_timestamp.date()
        if local_timestamp is not None
        else _parse_date(published_date) or _parse_date(published_timestamp)
    )
    if anchor_date is None:
        raise ValueError("a publication date is required for session assignment")

    if not is_trading_day(anchor_date):
        return SessionAssignment(
            signal_date=_next_session_on_or_after(anchor_date).isoformat(),
            timing_bucket="weekend_or_holiday",
            published_at_istanbul=local_timestamp,
        )

    if local_timestamp is None:
        return SessionAssignment(
            signal_date=next_trading_day(anchor_date).isoformat(),
            timing_bucket="unknown",
            published_at_istanbul=None,
        )

    local_time = local_timestamp.timetz().replace(tzinfo=None)
    if local_time < REGULAR_SESSION_OPEN:
        bucket: TimingBucket = "pre_open"
        assigned_date = anchor_date
    elif local_time <= session_close(anchor_date):
        bucket = "during_session"
        assigned_date = anchor_date
    else:
        bucket = "post_close"
        assigned_date = next_trading_day(anchor_date)

    return SessionAssignment(
        signal_date=assigned_date.isoformat(),
        timing_bucket=bucket,
        published_at_istanbul=local_timestamp,
    )


def signal_date(published_at: str, published_hour: Optional[int]) -> str:
    """Compatibility wrapper for the legacy ``(date, hour)`` call shape.

    A pre-open hour maps to that day's session because the opening auction can
    react to already-public information. Use :func:`assign_trading_session`
    when the original timestamp and timing bucket also need to be retained.
    """

    publication_date = _parse_date(published_at)
    if publication_date is None:
        raise ValueError(f"invalid publication date: {published_at!r}")

    if published_hour is None:
        timestamp: Optional[datetime] = None
    else:
        timestamp = datetime.combine(
            publication_date,
            time(int(published_hour), 0),
            tzinfo=ISTANBUL,
        )
    return assign_trading_session(timestamp, publication_date).signal_date
