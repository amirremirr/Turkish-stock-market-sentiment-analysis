"""Market return windows matched to when information actually became public.

A return is only a valid target if it could have been earned by someone acting
on the information. That requires three things to line up: when the news was
published, when a position could first be opened, and which prices were final by
then. Getting any of them wrong produces a number that looks like a result and
is not one.

The v1 defect, and why the names changed
----------------------------------------
v1 read ``signal_date`` as the *publication* session and then stepped forward to
"the next session". ``signal_date`` is in fact already the first **reactable**
session (proven in :mod:`research.timing` and ``scripts/timing_audit.py``), so
every ``post_close`` and ``weekend_or_holiday`` window was built one session
late: a headline published Monday 21:00 was scored against Tuesday-close to
Wednesday-open, missing the entire Tuesday reaction it should have measured.

That is not a look-ahead leak — a late window uses *less* information than it
could — but it is worse in a subtler way. It measures the session *after* the
one the news moved, so a genuine relationship would show up as a null and be
reported as "no predictive content found". v2 fixes the alignment and renames
the windows so their names describe the reaction session rather than an implied
step from an ambiguous anchor.

With ``D`` the first reactable session and ``P`` the trading session before it:

============================== =============== =========== ==================
window                         entry -> exit   tradable?   applies to
============================== =============== =========== ==================
``reactable_open_to_close``    open(D)->close(D)  **yes**  pre_open, post_close,
                                                           weekend_or_holiday
``prior_close_to_reactable_open``  close(P)->open(D)  no    same
``prior_close_to_reactable_close`` close(P)->close(D) no    same
============================== =============== =========== ==================

Why the two gap windows are not tradable, in any bucket: entering at close(P)
requires holding a position before the news was public. Pre-open news is
published after close(P); post-close news is published after close(P) by
definition. The gap is where a pre-open story's reaction actually lands, so the
windows are kept and reported — they are just labelled for what they are, a
measurement of reaction rather than an achievable return.

``during_session`` and ``unknown`` remain blocked: without intraday prices there
is no honest entry price for the first, and no known reaction session for the
second.

Only ``complete`` and ``corrected`` price bars are visible here. A provisional
bar is an intraday snapshot; treating one as a close would reintroduce exactly
the fault that corrupted 2026-07-31.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from research.timing import (
    BUCKET_DURING, BUCKET_UNKNOWN, previous_session, session_close_at,
    session_open_at,
)

RETURN_WINDOW_VERSION = "return-windows-v2"

WINDOW_REACTABLE_OPEN_CLOSE = "reactable_open_to_close"
WINDOW_PRIOR_CLOSE_TO_OPEN = "prior_close_to_reactable_open"
WINDOW_PRIOR_CLOSE_TO_CLOSE = "prior_close_to_reactable_close"

ALL_WINDOWS = (
    WINDOW_REACTABLE_OPEN_CLOSE,
    WINDOW_PRIOR_CLOSE_TO_OPEN,
    WINDOW_PRIOR_CLOSE_TO_CLOSE,
)

#: The single window a position could actually have been opened for.
PRIMARY_WINDOW = WINDOW_REACTABLE_OPEN_CLOSE

#: Windows retained for measuring reaction, never for execution-sensitive claims.
DESCRIPTIVE_WINDOWS = frozenset({
    WINDOW_PRIOR_CLOSE_TO_OPEN, WINDOW_PRIOR_CLOSE_TO_CLOSE,
})

NOT_TRADABLE_REASON = (
    "entry at the prior close predates publication; reaction measure only"
)

# Why a window could not be built, or why an event is ineligible.
REASON_INTRADAY_UNAVAILABLE = "intraday_prices_unavailable"
REASON_UNKNOWN_TIMING = "publication_time_unknown"
REASON_NO_PRICE_BAR = "no_complete_price_bar"
REASON_NO_PRIOR_SESSION_BAR = "no_prior_session_price_bar"
REASON_NO_FOLLOWING_SESSION = "no_following_complete_session"
REASON_MARKET_RECAP = "market_recap_excluded_by_default"
REASON_TIMING_CONFLICT = "event_members_span_incompatible_sessions"

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
    is_tradable: bool = True
    not_tradable_reason: Optional[str] = None
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
            "is_tradable": 1 if self.is_tradable else 0,
            "not_tradable_reason": self.not_tradable_reason,
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


def _unavailable(
    name: str, reason: str, entry: str, exit_: str, *, tradable: bool = True,
) -> ReturnWindow:
    return ReturnWindow(
        window_name=name, information_cutoff=None, assumed_execution=None,
        entry_price_field=entry, exit_price_field=exit_,
        entry_date=None, exit_date=None, entry_price=None, exit_price=None,
        raw_return=None, is_available=False, unavailable_reason=reason,
        is_tradable=tradable,
        not_tradable_reason=None if tradable else NOT_TRADABLE_REASON,
    )


def _percent(exit_price: Optional[float], entry: Optional[float]) -> Optional[float]:
    if entry and exit_price:
        return (exit_price / entry - 1.0) * 100.0
    return None


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
    first_reactable_session: Optional[str],
    timing_bucket: Optional[str],
    prices: PriceSeries,
) -> List[ReturnWindow]:
    """Build every window applicable to one event's timing.

    ``first_reactable_session`` is the stored ``signal_date``: the session that
    can react, *not* the session that published. Passing a publication date here
    reintroduces the v1 one-session shift.
    """

    if timing_bucket == BUCKET_DURING:
        # No intraday prices exist, so there is no defensible entry price.
        return [_unavailable(
            WINDOW_REACTABLE_OPEN_CLOSE, REASON_INTRADAY_UNAVAILABLE,
            "intraday", "close",
        )]

    if timing_bucket in (None, "", BUCKET_UNKNOWN):
        return [_unavailable(
            WINDOW_REACTABLE_OPEN_CLOSE, REASON_UNKNOWN_TIMING, "open", "close",
        )]

    session = prices.get(first_reactable_session)
    if session is None:
        return [
            _unavailable(WINDOW_REACTABLE_OPEN_CLOSE, REASON_NO_PRICE_BAR,
                         "open", "close"),
            _unavailable(WINDOW_PRIOR_CLOSE_TO_OPEN, REASON_NO_PRICE_BAR,
                         "close", "open", tradable=False),
            _unavailable(WINDOW_PRIOR_CLOSE_TO_CLOSE, REASON_NO_PRICE_BAR,
                         "close", "close", tradable=False),
        ]

    day = str(session["date"])
    open_price = _finite(session.get("open"))
    close_price = _finite(session.get("close"))
    execution = session_open_at(day)

    windows: List[ReturnWindow] = [ReturnWindow(
        window_name=WINDOW_REACTABLE_OPEN_CLOSE,
        # The news is public before the bell; the position opens at it.
        information_cutoff=execution,
        assumed_execution=execution,
        entry_price_field="open", exit_price_field="close",
        entry_date=day, exit_date=day,
        entry_price=open_price, exit_price=close_price,
        raw_return=_percent(close_price, open_price),
        is_available=bool(open_price and close_price),
        unavailable_reason=(
            None if (open_price and close_price) else REASON_NO_PRICE_BAR
        ),
    )]

    # The gap windows need the close that immediately preceded the reaction.
    # "The last bar we happen to have" is not that: across a missing bar it
    # would silently span several sessions and report the total as one gap.
    prior_day = previous_session(day)
    prior = prices.get(prior_day)
    if prior is None:
        windows.append(_unavailable(
            WINDOW_PRIOR_CLOSE_TO_OPEN, REASON_NO_PRIOR_SESSION_BAR,
            "close", "open", tradable=False,
        ))
        windows.append(_unavailable(
            WINDOW_PRIOR_CLOSE_TO_CLOSE, REASON_NO_PRIOR_SESSION_BAR,
            "close", "close", tradable=False,
        ))
        return windows

    prior_close = _finite(prior.get("close"))
    cutoff = session_close_at(prior_day)

    windows.append(ReturnWindow(
        window_name=WINDOW_PRIOR_CLOSE_TO_OPEN,
        information_cutoff=cutoff, assumed_execution=execution,
        entry_price_field="close", exit_price_field="open",
        entry_date=str(prior_day), exit_date=day,
        entry_price=prior_close, exit_price=open_price,
        raw_return=_percent(open_price, prior_close),
        is_available=bool(prior_close and open_price),
        unavailable_reason=(
            None if (prior_close and open_price) else REASON_NO_PRICE_BAR
        ),
        is_tradable=False, not_tradable_reason=NOT_TRADABLE_REASON,
    ))
    windows.append(ReturnWindow(
        window_name=WINDOW_PRIOR_CLOSE_TO_CLOSE,
        information_cutoff=cutoff, assumed_execution=execution,
        entry_price_field="close", exit_price_field="close",
        entry_date=str(prior_day), exit_date=day,
        entry_price=prior_close, exit_price=close_price,
        raw_return=_percent(close_price, prior_close),
        is_available=bool(prior_close and close_price),
        unavailable_reason=(
            None if (prior_close and close_price) else REASON_NO_PRICE_BAR
        ),
        is_tradable=False, not_tradable_reason=NOT_TRADABLE_REASON,
    ))
    return windows


def timing_eligibility(
    timing_bucket: Optional[str],
    *,
    is_market_recap: bool = False,
    timing_conflict: bool = False,
) -> Dict[str, Optional[str]]:
    """Classify whether an event may enter execution-sensitive research.

    Recaps are excluded by default from directional work because their tone
    follows the return by construction; they stay available for descriptive and
    reverse-causality analysis. A timing conflict blocks for a different reason:
    the event has no single reaction session, so no window is unambiguously its
    own.
    """

    if is_market_recap:
        return {"status": BLOCKED, "reason": REASON_MARKET_RECAP}
    if timing_bucket == BUCKET_DURING:
        return {"status": BLOCKED, "reason": REASON_INTRADAY_UNAVAILABLE}
    if timing_bucket in (None, "", BUCKET_UNKNOWN):
        return {"status": BLOCKED, "reason": REASON_UNKNOWN_TIMING}
    if timing_conflict:
        return {"status": BLOCKED, "reason": REASON_TIMING_CONFLICT}
    return {"status": ELIGIBLE, "reason": None}
