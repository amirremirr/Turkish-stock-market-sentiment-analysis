"""The frozen statistical unit: what counts as one independent observation.

The target is the BIST 100 index return. Every candidate event whose first
reactable session is 2026-06-09 is scored against **the same number**. Treating
those as independent samples is the single most effective way to manufacture
significance out of nothing: 3 126 event rows over ~75 sessions would report
standard errors roughly six times too small, and a null relationship would
cross any threshold you care to name.

So the primary unit is the **session**, not the event:

    one row per (first_reactable_session, target_window)

Event features are aggregated onto the session by a rule fixed in advance, in
:data:`AGGREGATION_RULES`, rather than chosen after seeing which aggregation
helps. Event-level rows are retained as a declared sensitivity analysis, to be
evaluated with session-cluster-aware inference — never as independent draws.

Fold safety
-----------
Because a session appears exactly once, no event can reach the test fold through
a second row, and no candidate group can straddle a boundary. What a one-row-
per-session design does *not* rule out is a story spanning two adjacent
sessions, so folds are additionally separated by an embargo gap
(:data:`DEFAULT_EMBARGO_SESSIONS`) — the last session before a test fold is
dropped from training rather than trusted.

Counts of rows, distinct events, distinct sessions and distinct outcomes are
reported together by :func:`unit_counts`, because the gap between the first and
the last is the whole point of this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from research.return_windows import (
    ALL_WINDOWS, DESCRIPTIVE_WINDOWS, PRIMARY_WINDOW,
)

MODELLING_UNIT_VERSION = "modelling-unit-v1"

UNIT_SESSION = "session"
UNIT_EVENT = "event"

#: How many sessions between the end of a training fold and the start of a test
#: fold are discarded. One session is enough to break a two-day story, which is
#: the only leakage path a session-level unit leaves open.
DEFAULT_EMBARGO_SESSIONS = 1

#: Minimum events on a session before its aggregate features are trusted. A
#: session carrying one headline is not a measurement of the day's news tone.
DEFAULT_MIN_EVENTS_PER_SESSION = 1


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return math.fsum(values) / len(values) if values else None


def _pstdev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))


def _weighted_mean(
    pairs: Sequence[tuple], default_weight: float = 1.0,
) -> Optional[float]:
    total = math.fsum(w for _, w in pairs)
    if not pairs or total <= 0:
        return _mean([v for v, _ in pairs])
    return math.fsum(v * w for v, w in pairs) / total


@dataclass(frozen=True)
class AggregationRule:
    """One pre-specified way of collapsing events onto their session."""

    name: str
    description: str
    apply: Callable[[List[Dict[str, Any]]], Optional[float]]


def _tones(rows: List[Dict[str, Any]]) -> List[float]:
    return [
        value for value in (_finite(r.get("mean_sentiment")) for r in rows)
        if value is not None
    ]


AGGREGATION_RULES: Dict[str, AggregationRule] = {
    "event_count": AggregationRule(
        "event_count", "number of candidate events reactable on the session",
        lambda rows: float(len(rows)),
    ),
    "headline_count": AggregationRule(
        "headline_count", "total headlines behind those events",
        lambda rows: float(sum(int(r.get("headline_count") or 0) for r in rows)),
    ),
    "source_breadth": AggregationRule(
        "source_breadth", "sum of distinct sources across events",
        lambda rows: float(sum(int(r.get("source_count") or 0) for r in rows)),
    ),
    "mean_tone": AggregationRule(
        "mean_tone", "unweighted mean of event mean sentiment",
        lambda rows: _mean(_tones(rows)),
    ),
    "breadth_weighted_tone": AggregationRule(
        "breadth_weighted_tone",
        "event tone weighted by source count; wider coverage counts for more",
        lambda rows: _weighted_mean([
            (_finite(r.get("mean_sentiment")), float(r.get("source_count") or 1))
            for r in rows if _finite(r.get("mean_sentiment")) is not None
        ]),
    ),
    "tone_dispersion": AggregationRule(
        "tone_dispersion", "spread of event tone across the session",
        lambda rows: _pstdev(_tones(rows)),
    ),
    "positive_share": AggregationRule(
        "positive_share", "share of events with tone above zero",
        lambda rows: (
            sum(1 for v in _tones(rows) if v > 0) / len(_tones(rows))
            if _tones(rows) else None
        ),
    ),
    "net_tone_share": AggregationRule(
        "net_tone_share", "positive minus negative share of events",
        lambda rows: (
            (sum(1 for v in _tones(rows) if v > 0)
             - sum(1 for v in _tones(rows) if v < 0)) / len(_tones(rows))
            if _tones(rows) else None
        ),
    ),
    "max_novelty": AggregationRule(
        "max_novelty", "highest novelty among the session's events",
        lambda rows: max(
            (v for v in (_finite(r.get("novelty")) for r in rows) if v is not None),
            default=None,
        ),
    ),
    "multi_source_events": AggregationRule(
        "multi_source_events", "events carried by more than one outlet",
        lambda rows: float(sum(1 for r in rows if int(r.get("source_count") or 0) > 1)),
    ),
    "mean_cross_source_dispersion": AggregationRule(
        "mean_cross_source_dispersion",
        "mean disagreement between outlets covering the same event",
        lambda rows: _mean([
            v for v in (_finite(r.get("cross_source_dispersion")) for r in rows)
            if v is not None
        ]),
    ),
}

#: Columns produced by the aggregation, in a fixed order.
AGGREGATED_FEATURES = tuple(AGGREGATION_RULES)

#: Target columns carried onto every session row.
TARGET_COLUMNS = (
    "raw_return", "residual_none", "residual_em_lagged",
    "residual_em_oil_fx_lagged", "residual_em_contemporaneous",
)


def eligible_rows(
    dataset: Iterable[Dict[str, Any]],
    *,
    window_name: str = PRIMARY_WINDOW,
    require_tradable: bool = True,
    exclude_conflicted: bool = True,
    exclude_singletons: bool = False,
    multi_source_only: bool = False,
) -> List[Dict[str, Any]]:
    """Filter the event dataset down to rows admissible for modelling.

    Every switch here corresponds to a sensitivity analysis named in the frozen
    protocol. None of them is chosen after seeing a result.
    """

    selected: List[Dict[str, Any]] = []
    for row in dataset:
        if row.get("window_name") != window_name:
            continue
        if row.get("eligibility_status") != "eligible":
            continue
        if row.get("raw_return") is None:
            continue
        if require_tradable and not row.get("is_tradable_window"):
            continue
        if exclude_conflicted and int(row.get("timing_conflict") or 0):
            continue
        if exclude_singletons and int(row.get("headline_count") or 0) <= 1:
            continue
        if multi_source_only and int(row.get("source_count") or 0) <= 1:
            continue
        selected.append(row)
    return selected


def build_session_units(
    dataset: Iterable[Dict[str, Any]],
    *,
    window_name: str = PRIMARY_WINDOW,
    min_events: int = DEFAULT_MIN_EVENTS_PER_SESSION,
    **filters: Any,
) -> List[Dict[str, Any]]:
    """Collapse the event dataset to one row per reactable session.

    The target is taken from the session's rows and asserted to be identical
    across them — it is the index return, so any disagreement means two
    different windows were mixed, which would be a bug rather than a data
    quirk.
    """

    rows = eligible_rows(dataset, window_name=window_name, **filters)

    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        session = row.get("first_reactable_session") or row.get("signal_date")
        if session:
            by_session.setdefault(str(session), []).append(row)

    units: List[Dict[str, Any]] = []
    for session in sorted(by_session):
        members = by_session[session]
        if len(members) < min_events:
            continue

        targets: Dict[str, Optional[float]] = {}
        for column in TARGET_COLUMNS:
            values = {
                round(v, 10) for v in (
                    _finite(member.get(column)) for member in members
                ) if v is not None
            }
            if len(values) > 1:
                raise ValueError(
                    f"session {session} carries {len(values)} distinct values for "
                    f"{column!r}; events on one session must share one outcome"
                )
            targets[column] = next(iter(values)) if values else None

        unit = {
            "first_reactable_session": session,
            "window_name": window_name,
            "modelling_unit": UNIT_SESSION,
            "modelling_unit_version": MODELLING_UNIT_VERSION,
            "event_count_raw": len(members),
            "group_keys": ",".join(sorted(m["group_key"] for m in members)),
            "exit_date": members[0].get("exit_date"),
            "entry_date": members[0].get("entry_date"),
            **targets,
        }
        for name, rule in AGGREGATION_RULES.items():
            unit[name] = rule.apply(members)

        # Descriptive labels for subgroup reporting. Never features: the
        # dominant family is decided by count, which is a property of the same
        # session the target measures.
        families: Dict[str, int] = {}
        buckets: Dict[str, int] = {}
        for member in members:
            if member.get("signal_family"):
                families[str(member["signal_family"])] = (
                    families.get(str(member["signal_family"]), 0) + 1
                )
            if member.get("timing_bucket"):
                buckets[str(member["timing_bucket"])] = (
                    buckets.get(str(member["timing_bucket"]), 0) + 1
                )
        unit["dominant_family"] = (
            max(sorted(families), key=lambda k: families[k]) if families else None
        )
        unit["dominant_timing_bucket"] = (
            max(sorted(buckets), key=lambda k: buckets[k]) if buckets else None
        )
        units.append(unit)
    return units


def attach_lagged_features(
    units: Sequence[Dict[str, Any]],
    *,
    target: str = "raw_return",
    factor_panel: Optional[Dict[str, Dict[str, float]]] = None,
    abnormal_tone: Optional[Dict[str, Dict[str, float]]] = None,
    regimes: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Add strictly backward-looking features to ordered session units.

    ``prev_return`` and ``prev_return_sign`` come from the previous unit in the
    ordered list -- never from the current one, and never from a full-sample
    shift that could wrap. The first unit has no predecessor and gets ``None``
    rather than zero, so the sample gate drops it instead of a fabricated
    neutral value entering the fit.

    ``factor_panel`` is expected to already contain ``*_lag1`` columns built by
    :func:`research.controls.build_control_panel`; only those are read here, so
    a same-session factor value cannot reach a tradable specification through
    this path.
    """

    ordered = sorted(units, key=lambda unit: str(unit["first_reactable_session"]))
    enriched: List[Dict[str, Any]] = []
    previous: Optional[float] = None

    for unit in ordered:
        session = str(unit["first_reactable_session"])
        row = dict(unit)
        row["prev_return"] = previous
        row["prev_return_sign"] = (
            None if previous is None else (1.0 if previous > 0 else -1.0)
        )

        factors = (factor_panel or {}).get(session, {})
        row["eem_lag1"] = factors.get("EEM_lag1")
        row["brent_lag1"] = factors.get("BZ=F_lag1")
        row["usdtry_lag1"] = factors.get("USDTRY=X_lag1")

        tone = (abnormal_tone or {}).get(session, {})
        row["abnormal_tone"] = tone.get("all")
        row["abnormal_tone_domestic"] = tone.get("domestic")

        row["regime"] = (regimes or {}).get(session)
        row["event_count"] = unit.get("event_count")

        enriched.append(row)
        current = _finite(unit.get(target))
        previous = current if current is not None else previous

    return enriched


def unit_counts(
    dataset: Iterable[Dict[str, Any]],
    units: Sequence[Dict[str, Any]],
    *,
    window_name: str = PRIMARY_WINDOW,
    target: str = "raw_return",
) -> Dict[str, Any]:
    """Rows, distinct events, distinct sessions and distinct outcomes.

    The last two are what inference is actually entitled to use. Reporting them
    side by side makes the duplication visible instead of implicit.
    """

    rows = eligible_rows(dataset, window_name=window_name)
    sessions = {
        str(row.get("first_reactable_session") or row.get("signal_date"))
        for row in rows
    }
    outcomes = {
        round(value, 10) for value in (_finite(row.get(target)) for row in rows)
        if value is not None
    }
    return {
        "event_rows": len(rows),
        "distinct_events": len({row["group_key"] for row in rows}),
        "distinct_sessions": len(sessions),
        "distinct_outcomes": len(outcomes),
        "session_units": len(units),
        "events_per_session": (
            round(len(rows) / len(sessions), 3) if sessions else None
        ),
        "duplication_factor": (
            round(len(rows) / len(outcomes), 3) if outcomes else None
        ),
        "window_name": window_name,
        "target": target,
        "modelling_unit_version": MODELLING_UNIT_VERSION,
    }


def sensitivity_windows() -> Dict[str, str]:
    """Windows retained for sensitivity, with why each is not primary."""

    return {
        window: (
            "primary tradable target" if window == PRIMARY_WINDOW
            else "reaction measure; entry predates publication"
            if window in DESCRIPTIVE_WINDOWS else "secondary"
        )
        for window in ALL_WINDOWS
    }
