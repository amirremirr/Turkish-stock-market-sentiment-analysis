"""The one place that defines what a timing field means.

Every downstream number depends on a single question: **when could someone
first have acted on this?** Before this module existed the answer was implied by
three different call sites, and they did not agree. The disagreement produced a
real defect: post-close and weekend return windows were built one session late
(see :mod:`research.return_windows` and ``docs/TIMING.md``).

Proven convention
-----------------
``signal_date`` is **the first trading session capable of reacting to the
publication** — not the session the news was published in. This is not an
assumption inherited from tests; it is what
:func:`trading_calendar.assign_trading_session` computes and what every
production row records:

===================  ==========================  ==========================
timing bucket        signal_date                 discriminates A vs B?
===================  ==========================  ==========================
``pre_open``         publication day             no (the two coincide)
``during_session``   publication day             no (the two coincide)
``post_close``       **next** trading day        yes
``weekend_or_holiday`` next session on/after     yes
``unknown``          **next** trading day        yes
===================  ==========================  ==========================

Hypothesis A (``signal_date`` = the publication/associated market session) is
refuted by every ``post_close``, ``weekend_or_holiday`` and ``unknown`` row.
Hypothesis B (first session capable of reacting) holds for all five buckets.
``scripts/timing_audit.py`` re-proves this against production records rather
than asserting it.

The invariant that follows, and that this module exports so nothing has to
re-derive it:

    ``first_reactable_session == signal_date``   for every bucket.

Event-level cutoff
------------------
A candidate event is not fully known until its **last** member is published, so
a group's target is derived from the last-reactable member — never from the
earliest ``signal_date`` combined with some other member's timing bucket. Groups
whose members straddle different reaction sessions are flagged
``timing_conflict`` and kept out of primary evaluation while remaining available
descriptively.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional, Sequence

from trading_calendar import (
    ISTANBUL, REGULAR_SESSION_OPEN, assign_trading_session, is_trading_day,
    next_trading_day, previous_trading_day, session_close,
)

TIMING_RULE_VERSION = "event-timing-v1"

#: What ``signal_date`` means, proven in ``scripts/timing_audit.py``.
SIGNAL_DATE_SEMANTICS = "first_reactable_session"

BUCKET_PRE_OPEN = "pre_open"
BUCKET_DURING = "during_session"
BUCKET_POST_CLOSE = "post_close"
BUCKET_WEEKEND = "weekend_or_holiday"
BUCKET_UNKNOWN = "unknown"

ALL_BUCKETS = (
    BUCKET_PRE_OPEN, BUCKET_DURING, BUCKET_POST_CLOSE,
    BUCKET_WEEKEND, BUCKET_UNKNOWN,
)

#: Buckets whose reaction can be priced from daily bars alone.
TRADABLE_BUCKETS = frozenset({BUCKET_PRE_OPEN, BUCKET_POST_CLOSE, BUCKET_WEEKEND})

# How restrictive a bucket is, lowest first. A during-session publication is the
# most restrictive because no daily bar can price its reaction.
_RESTRICTIVENESS = {
    BUCKET_DURING: 0,
    BUCKET_UNKNOWN: 1,
    BUCKET_WEEKEND: 2,
    BUCKET_POST_CLOSE: 3,
    BUCKET_PRE_OPEN: 4,
}

CONFLICT_MULTIPLE_SESSIONS = "members_span_multiple_reactable_sessions"
CONFLICT_GOVERNING_UNKNOWN = "governing_member_timing_unknown"


def restrictiveness(bucket: Optional[str]) -> int:
    """Rank a bucket by how little freedom it leaves to execute."""

    return _RESTRICTIVENESS.get(str(bucket), _RESTRICTIVENESS[BUCKET_UNKNOWN])


def _as_date(value: Any) -> Optional[date]:
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


def _at(day: Any, moment: time) -> Optional[str]:
    parsed = _as_date(day)
    if parsed is None:
        return None
    return datetime.combine(parsed, moment, tzinfo=ISTANBUL).isoformat()


def session_open_at(day: Any) -> Optional[str]:
    """The opening bell of *day* as an Istanbul timestamp."""

    return _at(day, REGULAR_SESSION_OPEN)


def session_close_at(day: Any) -> Optional[str]:
    """The closing bell of *day*, honouring half-day schedules."""

    parsed = _as_date(day)
    if parsed is None:
        return None
    return _at(parsed, session_close(parsed))


def previous_session(day: Any) -> Optional[str]:
    """The trading session immediately before *day*, as an ISO date."""

    parsed = _as_date(day)
    return previous_trading_day(parsed).isoformat() if parsed else None


def expected_signal_date(
    published_timestamp: Any, published_date: Any = None,
) -> Optional[str]:
    """Independently derive the first reactable session for one publication.

    Deliberately re-implemented from the timing bucket rather than delegating to
    :func:`trading_calendar.assign_trading_session`, so the audit compares two
    derivations instead of comparing a function with itself.
    """

    assignment = _assign(published_timestamp, published_date)
    if assignment is None:
        return None
    bucket, anchor = assignment
    if bucket in (BUCKET_PRE_OPEN, BUCKET_DURING):
        return anchor.isoformat()
    if bucket == BUCKET_WEEKEND:
        # A non-trading anchor reacts at the next session on or after it.
        candidate = anchor
        while not is_trading_day(candidate):
            candidate = date.fromordinal(candidate.toordinal() + 1)
        return candidate.isoformat()
    # post_close and unknown both wait for the next session strictly after.
    return next_trading_day(anchor).isoformat()


def expected_publication_session(
    published_timestamp: Any, published_date: Any = None,
) -> Optional[str]:
    """The market session the publication *belongs to*, or None off-calendar.

    This is hypothesis A. It exists so the audit can refute it explicitly
    instead of assuming the convention.
    """

    assignment = _assign(published_timestamp, published_date)
    if assignment is None:
        return None
    _, anchor = assignment
    return anchor.isoformat() if is_trading_day(anchor) else None


def _assign(published_timestamp: Any, published_date: Any):
    """Return ``(timing_bucket, publication_date)`` or None."""

    anchor = _as_date(published_date) or _as_date(published_timestamp)
    try:
        result = assign_trading_session(published_timestamp, published_date)
    except ValueError:
        return None
    if anchor is None:
        anchor = (
            result.published_at_istanbul.date()
            if result.published_at_istanbul else None
        )
    if anchor is None:
        return None
    if result.published_at_istanbul is not None:
        anchor = result.published_at_istanbul.date()
    return result.timing_bucket, anchor


def first_reactable_at(
    signal_date: Any,
    timing_bucket: Optional[str],
    published_at_istanbul: Any = None,
) -> Optional[str]:
    """The earliest moment a position could be opened in response.

    For every bucket that a daily bar can price this is the opening bell of the
    first reactable session. A during-session publication is reactable at the
    publication moment itself — which is precisely why it is blocked: no daily
    bar contains that price.
    """

    if timing_bucket == BUCKET_DURING and published_at_istanbul:
        return str(published_at_istanbul)
    return session_open_at(signal_date)


@dataclass(frozen=True)
class EventTiming:
    """Timing of a candidate event group, derived from one governing member.

    ``governing_headline_id`` names the member that supplied *both* the bucket
    and the session, so the pair can never be assembled from two different
    headlines.
    """

    first_reactable_session: Optional[str]
    first_reactable_at: Optional[str]
    timing_bucket: str
    event_information_cutoff: Optional[str]
    governing_headline_id: Optional[int]
    timing_conflict: int
    timing_conflict_reason: Optional[str]
    member_session_count: int
    rule_version: str = TIMING_RULE_VERSION

    def as_row(self) -> Dict[str, Any]:
        return {
            "first_reactable_session": self.first_reactable_session,
            "first_reactable_at": self.first_reactable_at,
            "timing_bucket": self.timing_bucket,
            "event_information_cutoff": self.event_information_cutoff,
            "governing_headline_id": self.governing_headline_id,
            "timing_conflict": self.timing_conflict,
            "timing_conflict_reason": self.timing_conflict_reason,
            "member_session_count": self.member_session_count,
            "event_timing_rule_version": self.rule_version,
        }


def derive_event_timing(members: Sequence[Dict[str, Any]]) -> EventTiming:
    """Derive one event's timing from an explicitly documented cutoff.

    **The cutoff rule:** a candidate event is not fully known until its last
    member is published, so the event's reaction session is the *latest*
    ``signal_date`` among its members. Ties are broken toward the most
    restrictive bucket on that session, and the bucket is then read from the
    same member that supplied the session.

    Taking the earliest ``signal_date`` — the behaviour this replaces — would
    claim the event was actionable before part of it existed.
    """

    usable = [m for m in members if m.get("signal_date")]
    sessions = sorted({str(m["signal_date"]) for m in usable})

    if not usable:
        return EventTiming(
            first_reactable_session=None, first_reactable_at=None,
            timing_bucket=BUCKET_UNKNOWN, event_information_cutoff=None,
            governing_headline_id=None, timing_conflict=1,
            timing_conflict_reason=CONFLICT_GOVERNING_UNKNOWN,
            member_session_count=0,
        )

    target_session = sessions[-1]
    on_session = [m for m in usable if str(m["signal_date"]) == target_session]
    governing = min(
        on_session,
        key=lambda m: (
            restrictiveness(m.get("timing_bucket")),
            int(m.get("headline_id") or m.get("id") or 0),
        ),
    )
    bucket = str(governing.get("timing_bucket") or BUCKET_UNKNOWN)

    reasons: List[str] = []
    if len(sessions) > 1:
        reasons.append(CONFLICT_MULTIPLE_SESSIONS)
    if bucket == BUCKET_UNKNOWN and any(
        m.get("timing_bucket") not in (None, "", BUCKET_UNKNOWN) for m in usable
    ):
        reasons.append(CONFLICT_GOVERNING_UNKNOWN)

    # The event is public once its last member is; that moment, not the first
    # member's, is what a downstream model is allowed to condition on.
    moments = [
        str(m.get("published_timestamp")) for m in members
        if m.get("published_timestamp")
    ]
    cutoff = max(moments) if moments else None

    return EventTiming(
        first_reactable_session=target_session,
        first_reactable_at=first_reactable_at(
            target_session, bucket, governing.get("published_timestamp"),
        ),
        timing_bucket=bucket,
        event_information_cutoff=cutoff,
        governing_headline_id=(
            int(governing["headline_id"]) if governing.get("headline_id") is not None
            else None
        ),
        timing_conflict=1 if reasons else 0,
        timing_conflict_reason=";".join(reasons) or None,
        member_session_count=len(sessions),
    )
