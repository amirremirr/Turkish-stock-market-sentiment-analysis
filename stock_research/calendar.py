"""Session calendar derived from data, and the first session able to act.

The repository's ``trading_calendar`` knows official holidays for 2025-2026
only. The KAP sample is from 2023, so a calendar built from a holiday list
would invent sessions. Here a date is a session if, and only if, the benchmark
has a bar on it.

Day 0 of an event is the **first session whose prices could reflect it**:

=====================  ==========================  =======================
published              day 0                       bucket
=====================  ==========================  =======================
non-session day        next session                ``weekend_or_holiday``
before 10:00           same session                ``pre_open``
10:00 to the close     same session                ``during_session``
after the close        next session                ``post_close``
time unknown           next session                ``unknown``
=====================  ==========================  =======================

For ``during_session`` the day-0 close-to-close return also contains whatever
happened before publication, so day 0 is not a tradable window for those
events. ``post_close``, ``pre_open`` and ``weekend_or_holiday`` are the buckets
with a clean first reaction.

Half days. Official early closes are known for 2025-2026 (``config``). For
other years they are not recorded anywhere in this repository, so a
publication after 12:30 on a session immediately followed by a weekday closure
is treated as after the close and flagged ``timing_ambiguous``. That can delay
day 0; it cannot advance it.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

ISTANBUL = ZoneInfo("Europe/Istanbul")
SESSION_OPEN = time(10, 0)
SESSION_CLOSE = time(18, 10)
ASSUMED_HALF_DAY_CLOSE = time(12, 30)

BUCKET_PRE_OPEN = "pre_open"
BUCKET_DURING = "during_session"
BUCKET_POST_CLOSE = "post_close"
BUCKET_NON_SESSION = "weekend_or_holiday"
BUCKET_UNKNOWN = "unknown"
#: Buckets whose day 0 starts from a price formed before the news.
CLEAN_REACTION_BUCKETS = frozenset({BUCKET_PRE_OPEN, BUCKET_POST_CLOSE, BUCKET_NON_SESSION})

CALENDAR_RULE_VERSION = "stock-session-calendar-v1"


@dataclass(frozen=True)
class Actionable:
    day0: Optional[str]            # ISO date of the first session able to act
    bucket: str
    timing_ambiguous: bool
    published_local: Optional[str]  # ISO timestamp, Europe/Istanbul
    published_utc: Optional[str]


def parse_kap_time(raw: Optional[str]) -> Optional[datetime]:
    """'29.12.2023 18:23:08' (Istanbul local) -> aware datetime, or None."""

    try:
        return datetime.strptime(str(raw).strip(), "%d.%m.%Y %H:%M:%S").replace(tzinfo=ISTANBUL)
    except (ValueError, TypeError):
        return None


class SessionCalendar:
    """Ordered trading sessions, taken from the dates a benchmark traded."""

    def __init__(self, sessions: Sequence[str]):
        self.sessions = sorted({str(s)[:10] for s in sessions})
        if not self.sessions:
            raise ValueError("a session calendar needs at least one session")
        self._set = set(self.sessions)

    @property
    def first(self) -> str:
        return self.sessions[0]

    @property
    def last(self) -> str:
        return self.sessions[-1]

    def covers(self, day: str) -> bool:
        """Whether *day* lies inside the range the calendar can speak about."""

        return self.first <= str(day)[:10] <= self.last

    def is_session(self, day: str) -> bool:
        return str(day)[:10] in self._set

    def on_or_after(self, day: str) -> Optional[str]:
        position = bisect_left(self.sessions, str(day)[:10])
        return self.sessions[position] if position < len(self.sessions) else None

    def after(self, day: str) -> Optional[str]:
        position = bisect_right(self.sessions, str(day)[:10])
        return self.sessions[position] if position < len(self.sessions) else None

    def index(self, day: str) -> Optional[int]:
        key = str(day)[:10]
        position = bisect_left(self.sessions, key)
        if position < len(self.sessions) and self.sessions[position] == key:
            return position
        return None

    def offset(self, day: str, steps: int) -> Optional[str]:
        position = self.index(day)
        if position is None:
            return None
        target = position + steps
        return self.sessions[target] if 0 <= target < len(self.sessions) else None

    # -- closes ---------------------------------------------------------------
    def _close_time(self, day: date) -> tuple[time, bool]:
        """(close, assumed) for a session day."""

        from config import BIST_HALF_DAYS, BIST_HOLIDAYS

        covered_years = {entry[:4] for entry in BIST_HOLIDAYS}
        if f"{day.year}" in covered_years:
            official = BIST_HALF_DAYS.get(day.isoformat())
            return (time.fromisoformat(official), False) if official else (SESSION_CLOSE, False)

        following = day + timedelta(days=1)
        while following.weekday() >= 5:
            following += timedelta(days=1)
        if self.covers(following.isoformat()) and not self.is_session(following.isoformat()):
            return ASSUMED_HALF_DAY_CLOSE, True
        return SESSION_CLOSE, False

    def actionable(self, published: Optional[datetime], *,
                   published_date: Optional[str] = None) -> Actionable:
        """First session able to act on something published at *published*."""

        if published is not None and published.tzinfo is None:
            published = published.replace(tzinfo=ISTANBUL)
        local = published.astimezone(ISTANBUL) if published is not None else None
        anchor = local.date().isoformat() if local is not None else (
            str(published_date)[:10] if published_date else None)
        stamps = (
            local.isoformat() if local is not None else None,
            local.astimezone(ZoneInfo("UTC")).isoformat() if local is not None else None,
        )
        if anchor is None or not self.covers(anchor):
            # Outside the calendar there is no basis for choosing a session.
            return Actionable(None, BUCKET_UNKNOWN, True, *stamps)

        if not self.is_session(anchor):
            return Actionable(self.on_or_after(anchor), BUCKET_NON_SESSION, False, *stamps)
        if local is None:
            return Actionable(self.after(anchor), BUCKET_UNKNOWN, True, *stamps)

        moment = local.timetz().replace(tzinfo=None)
        close, assumed = self._close_time(local.date())
        if moment < SESSION_OPEN:
            return Actionable(anchor, BUCKET_PRE_OPEN, False, *stamps)
        if moment <= close:
            return Actionable(anchor, BUCKET_DURING, False, *stamps)
        # After an *assumed* early close the true close may have been 18:10,
        # so the bucket itself is uncertain until the regular close has passed.
        ambiguous = assumed and moment <= SESSION_CLOSE
        return Actionable(self.after(anchor), BUCKET_POST_CLOSE, ambiguous, *stamps)
