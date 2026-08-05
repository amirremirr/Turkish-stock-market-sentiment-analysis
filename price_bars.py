"""Daily price-bar completeness classification.

A daily bar fetched while the exchange is still open is not a daily bar. It is
an intraday snapshot wearing a daily bar's schema, and nothing downstream can
tell the difference once it is stored. That is how the 2026-07-31 BIST row
entered production: the scheduled run fires at 06:30 UTC, hours before the
18:10 Istanbul close, so it recorded a mid-session price as that day's close.

This module decides what a bar is, from the bar's own date and when it was
observed. It has no database, network, or pandas dependency so the rules can be
tested directly.

Statuses
--------
``provisional``       observed before the session settled; may be replaced
``complete``          observed after settlement; usable for analysis
``provider_invalid``  a bar on a date the exchange did not trade
``corrected``         a previously provisional/invalid bar replaced by a
                      verified refetch (set by the correction path, never
                      inferred here)

Settlement is the scheduled close plus a safety delay, because providers
publish the settled bar slightly after the bell. Half-days close early and are
handled from the official calendar rather than assumed.

A zero or missing volume on a session that did trade is reported through
``review_reason`` rather than by changing the status: it is a data-quality
signal, not proof the bar is wrong. Callers decide what to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal, Optional, Union

from trading_calendar import ISTANBUL, is_trading_day, session_close

BarStatus = Literal["provisional", "complete", "provider_invalid", "corrected"]
SessionType = Literal["full", "half", "closed"]

STATUS_PROVISIONAL: BarStatus = "provisional"
STATUS_COMPLETE: BarStatus = "complete"
STATUS_PROVIDER_INVALID: BarStatus = "provider_invalid"
STATUS_CORRECTED: BarStatus = "corrected"

# Statuses whose close is trustworthy for return construction.
ANALYSABLE_STATUSES = (STATUS_COMPLETE, STATUS_CORRECTED)

REVIEW_NON_TRADING_DAY = "bar_on_non_trading_day"
REVIEW_UNVERIFIED = "settlement_time_unverifiable"
REVIEW_BEFORE_SETTLEMENT = "observed_before_settlement"
REVIEW_ZERO_VOLUME_FULL = "zero_volume_on_full_session"
REVIEW_ZERO_VOLUME_HALF = "zero_volume_on_half_day"
REVIEW_MISSING_VOLUME_FULL = "missing_volume_on_full_session"
REVIEW_MISSING_VOLUME_HALF = "missing_volume_on_half_day"

DateLike = Union[str, date, datetime]
TimestampLike = Union[str, datetime]


@dataclass(frozen=True)
class BarClassification:
    """What a stored daily bar actually is."""

    status: BarStatus
    session_type: SessionType
    review_reason: Optional[str]
    settles_at: Optional[datetime]

    @property
    def is_analysable(self) -> bool:
        return self.status in ANALYSABLE_STATUSES

    @property
    def needs_review(self) -> bool:
        return self.review_reason is not None


def _as_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip()[:10])


def _as_istanbul(value: Optional[TimestampLike]) -> Optional[datetime]:
    """Normalize to Europe/Istanbul; a naive value is read as Istanbul local."""

    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if not raw:
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


def session_type(bar_date: DateLike) -> SessionType:
    """Return whether the exchange held a full session, a half day, or none."""

    from config import BIST_HALF_DAYS

    day = _as_date(bar_date)
    if not is_trading_day(day):
        return "closed"
    return "half" if day.isoformat() in BIST_HALF_DAYS else "full"


def settlement_time(
    bar_date: DateLike, *, settlement_minutes: Optional[int] = None
) -> Optional[datetime]:
    """Return when *bar_date*'s daily bar can be trusted as final."""

    day = _as_date(bar_date)
    if not is_trading_day(day):
        return None
    if settlement_minutes is None:
        from config import PRICE_BAR_SETTLEMENT_MINUTES

        settlement_minutes = PRICE_BAR_SETTLEMENT_MINUTES
    close = datetime.combine(day, session_close(day), tzinfo=ISTANBUL)
    return close + timedelta(minutes=settlement_minutes)


def _volume_review_reason(volume: object, kind: SessionType) -> Optional[str]:
    """Flag a session that traded but reports no volume."""

    if volume is None:
        return (
            REVIEW_MISSING_VOLUME_HALF if kind == "half"
            else REVIEW_MISSING_VOLUME_FULL
        )
    try:
        numeric = float(volume)
    except (TypeError, ValueError):
        return (
            REVIEW_MISSING_VOLUME_HALF if kind == "half"
            else REVIEW_MISSING_VOLUME_FULL
        )
    if numeric != numeric:  # NaN
        return (
            REVIEW_MISSING_VOLUME_HALF if kind == "half"
            else REVIEW_MISSING_VOLUME_FULL
        )
    if numeric == 0.0:
        # A half day trades thinly but not to a standstill, so zero volume is
        # still worth surfacing -- under its own reason, so the two cases can
        # be triaged differently rather than lumped together.
        return (
            REVIEW_ZERO_VOLUME_HALF if kind == "half"
            else REVIEW_ZERO_VOLUME_FULL
        )
    return None


def classify_price_bar(
    bar_date: DateLike,
    *,
    volume: object = None,
    observed_at: Optional[TimestampLike] = None,
    settlement_minutes: Optional[int] = None,
) -> BarClassification:
    """Classify one daily bar.

    Args:
        bar_date: the session the bar claims to describe.
        volume: reported volume, used only for the review flag.
        observed_at: when the bar was fetched. ``None`` means the settlement
            time cannot be established, which yields ``provisional`` -- an
            unverifiable bar is treated as unfinished, never as complete.
        settlement_minutes: override for the configured safety delay.
    """

    kind = session_type(bar_date)
    if kind == "closed":
        # The exchange was shut; a bar here is a provider artifact, not a
        # session. It must never reach analysis regardless of its values.
        return BarClassification(
            status=STATUS_PROVIDER_INVALID,
            session_type="closed",
            review_reason=REVIEW_NON_TRADING_DAY,
            settles_at=None,
        )

    settles_at = settlement_time(bar_date, settlement_minutes=settlement_minutes)
    observed = _as_istanbul(observed_at)

    if observed is None:
        return BarClassification(
            status=STATUS_PROVISIONAL,
            session_type=kind,
            review_reason=REVIEW_UNVERIFIED,
            settles_at=settles_at,
        )

    if observed < settles_at:
        return BarClassification(
            status=STATUS_PROVISIONAL,
            session_type=kind,
            review_reason=REVIEW_BEFORE_SETTLEMENT,
            settles_at=settles_at,
        )

    return BarClassification(
        status=STATUS_COMPLETE,
        session_type=kind,
        review_reason=_volume_review_reason(volume, kind),
        settles_at=settles_at,
    )


def may_replace(existing_status: Optional[str], incoming_status: str) -> bool:
    """Return whether an incoming bar may overwrite the stored one.

    A settled bar is the better observation, so completion moves forward only:
    a provisional refetch must never demote a bar that was already verified.
    Re-observing at the same rank is allowed, since a later fetch of an equally
    trustworthy bar is simply fresher data.
    """

    rank = {
        None: 0,
        STATUS_PROVIDER_INVALID: 1,
        STATUS_PROVISIONAL: 2,
        STATUS_COMPLETE: 3,
        STATUS_CORRECTED: 3,
    }
    if existing_status not in rank:
        existing_status = None
    return rank[incoming_status] >= rank[existing_status]
