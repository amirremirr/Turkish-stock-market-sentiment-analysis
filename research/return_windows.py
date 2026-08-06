"""Market return windows matched to when information actually became public.

A return is only a valid target if it could have been earned by someone acting
on the information. That requires three things to line up: when the news was
published, when a position could first be opened, and which prices were final
by then. Getting any of them wrong produces a number that looks like a result
and is not one.

Windows
-------
``pre_open``   published before the open: buy at that session's open, exit at
               its close. Information cutoff is the open.
``post_close`` published after the close: buy at the next open, exit at the next
               close. Also reported close-to-next-open and close-to-next-close.
``during``     published while the market was trading. **Blocked**: without
               intraday prices there is no honest entry price, so these are
               marked ineligible with a stated reason and kept for descriptive
               use only.

Every window records its information cutoff, its assumed execution timestamp,
and the exact price fields used, so the assumption behind any number is
readable rather than implied.

Only ``complete`` and ``corrected`` price bars are used. A provisional bar is an
intraday snapshot; treating one as a close would reintroduce exactly the fault
that corrupted 2026-07-31.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from trading_calendar import ISTANBUL, REGULAR_SESSION_OPEN, session_close

RETURN_WINDOW_VERSION = "return-windows-v1"

WINDOW_SAME_SESSION_OPEN_CLOSE = "same_session_open_to_close"
WINDOW_CLOSE_TO_NEXT_OPEN = "close_to_next_open"
WINDOW_NEXT_OPEN_TO_NEXT_CLOSE = "next_open_to_next_close"
WINDOW_CLOSE_TO_NEXT_CLOSE = "close_to_next_close"

# Why a window could not be built, or why an event is ineligible.
REASON_INTRADAY_UNAVAILABLE = "intraday_prices_unavailable"
REASON_UNKNOWN_TIMING = "publication_time_unknown"
REASON_NO_PRICE_BAR = "no_complete_price_bar"
REASON_NO_FOLLOWING_SESSION = "no_following_complete_session"
REASON_MARKET_RECAP = "market_recap_excluded_by_default"

ELIGIBLE = "eligible"
BLOCKED = "blocked"


@dataclass(frozen=True)
class ReturnWindow:
    """One market window with its full timing provenance."""

    window_name: str
    information_cutoff: Optional[str]
    assumed_execution: Optional[str]
    entry_price_field: str
    exit_price_field: str
    entry_date: Optional[str]
    exit_date: Optional[str]
    entry_price: Optional[float]
    exit_price: Optional[float]
    raw_return: Optional[float]
    is_available: bool
    unavailable_reason: Optional[str]
    rule_version: str = RETURN_WINDOW_VERSION

    def as_row(self) -> Dict[str, Any]:
        return {
            "window_name": self.window_name,
            "information_cutoff": self.information_cutoff,
            "assumed_execution": self.assumed_execution,
            "entry_price_field": self.entry_price_field,
            "exit_price_field": self.exit_price_field,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "raw_return": self.raw_return,
            "is_available": 1 if self.is_available else 0,
            "unavailable_reason": self.unavailable_reason,
            "rule_version": self.rule_version,
        }


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _timestamp(day: str, moment) -> str:
    return datetime.combine(
        datetime.fromisoformat(day).date(), moment, tzinfo=ISTANBUL
    ).isoformat()


def _unavailable(name: str, reason: str, entry: str, exit_: str) -> ReturnWindow:
    return ReturnWindow(
        window_name=name, information_cutoff=None, assumed_execution=None,
        entry_price_field=entry, exit_price_field=exit_,
        entry_date=None, exit_date=None, entry_price=None, exit_price=None,
        raw_return=None, is_available=False, unavailable_reason=reason,
    )


class PriceSeries:
    """Ordered access to settled price bars only.

    Provisional and provider-invalid bars are absent by construction, so a
    caller cannot accidentally build a return across one.
    """

    def __init__(self, bars: Sequence[Dict[str, Any]]):
        usable = [
            bar for bar in bars
            if str(bar.get("bar_status")) in ("complete", "corrected")
        ]
        self._by_date = {str(bar["date"]): bar for bar in usable}
        self._dates = sorted(self._by_date)

    @property
    def dates(self) -> List[str]:
        return list(self._dates)

    def get(self, day: Optional[str]) -> Optional[Dict[str, Any]]:
        return self._by_date.get(str(day)) if day else None

    def next_after(self, day: str) -> Optional[Dict[str, Any]]:
        """The first settled session strictly after *day*."""
        for candidate in self._dates:
            if candidate > str(day):
                return self._by_date[candidate]
        return None

    def previous_before(self, day: str) -> Optional[Dict[str, Any]]:
        previous = None
        for candidate in self._dates:
            if candidate >= str(day):
                break
            previous = self._by_date[candidate]
        return previous


def build_return_windows(
    signal_date: Optional[str],
    timing_bucket: Optional[str],
    prices: PriceSeries,
) -> List[ReturnWindow]:
    """Build every window applicable to one event's timing.

    The timing bucket decides which windows are even meaningful. A pre-open
    publication can be acted on at that session's open; a post-close one cannot
    be acted on until the next. Asking for the wrong one is how look-ahead
    enters a dataset.
    """

    windows: List[ReturnWindow] = []

    if timing_bucket == "during_session":
        # No intraday prices exist, so there is no defensible entry price.
        return [
            _unavailable(WINDOW_SAME_SESSION_OPEN_CLOSE,
                         REASON_INTRADAY_UNAVAILABLE, "intraday", "close"),
        ]

    if timing_bucket in (None, "", "unknown"):
        return [
            _unavailable(WINDOW_SAME_SESSION_OPEN_CLOSE,
                         REASON_UNKNOWN_TIMING, "open", "close"),
        ]

    session = prices.get(signal_date)

    if timing_bucket == "pre_open":
        if session is None:
            windows.append(_unavailable(
                WINDOW_SAME_SESSION_OPEN_CLOSE, REASON_NO_PRICE_BAR,
                "open", "close",
            ))
        else:
            entry = _finite(session.get("open"))
            exit_price = _finite(session.get("close"))
            day = str(session["date"])
            windows.append(ReturnWindow(
                window_name=WINDOW_SAME_SESSION_OPEN_CLOSE,
                # Everything known before the bell; execution at the open.
                information_cutoff=_timestamp(day, REGULAR_SESSION_OPEN),
                assumed_execution=_timestamp(day, REGULAR_SESSION_OPEN),
                entry_price_field="open", exit_price_field="close",
                entry_date=day, exit_date=day,
                entry_price=entry, exit_price=exit_price,
                raw_return=(
                    (exit_price / entry - 1.0) * 100.0
                    if entry and exit_price else None
                ),
                is_available=bool(entry and exit_price),
                unavailable_reason=None if (entry and exit_price) else REASON_NO_PRICE_BAR,
            ))
        return windows

    # post_close and weekend_or_holiday: the earliest action is the next open.
    reference = prices.get(signal_date) or (
        prices.previous_before(signal_date) if signal_date else None
    )
    following = prices.next_after(str(reference["date"])) if reference else (
        prices.get(signal_date)
    )

    if reference is None or following is None:
        return [
            _unavailable(WINDOW_CLOSE_TO_NEXT_OPEN,
                         REASON_NO_FOLLOWING_SESSION, "close", "open"),
            _unavailable(WINDOW_NEXT_OPEN_TO_NEXT_CLOSE,
                         REASON_NO_FOLLOWING_SESSION, "open", "close"),
            _unavailable(WINDOW_CLOSE_TO_NEXT_CLOSE,
                         REASON_NO_FOLLOWING_SESSION, "close", "close"),
        ]

    reference_day = str(reference["date"])
    next_day = str(following["date"])
    reference_close = _finite(reference.get("close"))
    next_open = _finite(following.get("open"))
    next_close = _finite(following.get("close"))
    cutoff = _timestamp(reference_day, session_close(
        datetime.fromisoformat(reference_day).date()
    ))
    execution = _timestamp(next_day, REGULAR_SESSION_OPEN)

    windows.append(ReturnWindow(
        window_name=WINDOW_CLOSE_TO_NEXT_OPEN,
        information_cutoff=cutoff, assumed_execution=execution,
        entry_price_field="close", exit_price_field="open",
        entry_date=reference_day, exit_date=next_day,
        entry_price=reference_close, exit_price=next_open,
        raw_return=(
            (next_open / reference_close - 1.0) * 100.0
            if reference_close and next_open else None
        ),
        is_available=bool(reference_close and next_open),
        unavailable_reason=None if (reference_close and next_open) else REASON_NO_PRICE_BAR,
    ))
    windows.append(ReturnWindow(
        window_name=WINDOW_NEXT_OPEN_TO_NEXT_CLOSE,
        information_cutoff=cutoff, assumed_execution=execution,
        entry_price_field="open", exit_price_field="close",
        entry_date=next_day, exit_date=next_day,
        entry_price=next_open, exit_price=next_close,
        raw_return=(
            (next_close / next_open - 1.0) * 100.0
            if next_open and next_close else None
        ),
        is_available=bool(next_open and next_close),
        unavailable_reason=None if (next_open and next_close) else REASON_NO_PRICE_BAR,
    ))
    windows.append(ReturnWindow(
        window_name=WINDOW_CLOSE_TO_NEXT_CLOSE,
        information_cutoff=cutoff, assumed_execution=execution,
        entry_price_field="close", exit_price_field="close",
        entry_date=reference_day, exit_date=next_day,
        entry_price=reference_close, exit_price=next_close,
        raw_return=(
            (next_close / reference_close - 1.0) * 100.0
            if reference_close and next_close else None
        ),
        is_available=bool(reference_close and next_close),
        unavailable_reason=None if (reference_close and next_close) else REASON_NO_PRICE_BAR,
    ))
    return windows


def timing_eligibility(
    timing_bucket: Optional[str],
    *,
    is_market_recap: bool = False,
) -> Dict[str, Optional[str]]:
    """Classify whether an event may enter execution-sensitive research.

    Recaps are excluded by default from directional work because their tone
    follows the return by construction; they stay available for descriptive and
    reverse-causality analysis.
    """

    if is_market_recap:
        return {"status": BLOCKED, "reason": REASON_MARKET_RECAP}
    if timing_bucket == "during_session":
        return {"status": BLOCKED, "reason": REASON_INTRADAY_UNAVAILABLE}
    if timing_bucket in (None, "", "unknown"):
        return {"status": BLOCKED, "reason": REASON_UNKNOWN_TIMING}
    return {"status": ELIGIBLE, "reason": None}
