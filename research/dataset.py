"""Assemble the event-level research dataset.

One row per (candidate event, market window). Each row carries the event's
descriptive features, the exact timing assumptions behind its target, the raw
return, residuals under each pre-specified control set, and an explicit record
of what could not be computed and why.

No model is fitted to it, no split is taken, and no feature is selected. The
dataset is deliberately built before any of that so those decisions can later be
made under a frozen protocol rather than against the data.

Blocked features are named per row rather than dropped. A missing column reads
as an oversight; a stated reason reads as a limitation, which is what it is.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from research.controls import CONTROL_SETS, compute_residual_returns
from research.return_windows import (
    BLOCKED, PRIMARY_WINDOW, PriceSeries, build_return_windows,
    timing_eligibility,
)
from research.timing import TIMING_RULE_VERSION

DATASET_VERSION = "event-research-dataset-v2"

# Data the project does not have. Recorded on every row so downstream work
# cannot mistake absence for irrelevance.
BLOCKED_FEATURES = {
    "intraday_prices": "no intraday BIST data; during-session events are descriptive only",
    "consensus_expectations": "licensed; macro surprise cannot be computed",
    "kap_structured_events": "KAP production access pending; issuer events are headline-derived",
}

RESIDUAL_COLUMNS = {
    "none": "residual_none",
    "em_lagged": "residual_em_lagged",
    "em_oil_fx_lagged": "residual_em_oil_fx_lagged",
    "em_contemporaneous": "residual_em_contemporaneous",
}


def build_event_dataset(
    events: Sequence[Dict[str, Any]],
    price_bars: Sequence[Dict[str, Any]],
    factor_rows: Sequence[Dict[str, Any]],
    *,
    experiment_id: str,
    algorithm_version: str,
    dataset_version: str = DATASET_VERSION,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return ``{"windows": [...], "residuals": [...], "dataset": [...]}``."""

    from research.controls import build_control_panel

    prices = PriceSeries(price_bars)
    panel = build_control_panel(factor_rows)

    window_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []

    # 1. Build every window first, so residuals can be computed once per
    #    (exit_date, window) rather than once per event.
    per_event_windows: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        timing = _dominant_timing(event)
        windows = build_return_windows(
            _reactable_session(event), timing, prices,
        )
        rows = []
        for window in windows:
            row = window.as_row()
            row.update({
                "group_key": event["group_key"],
                "algorithm_version": algorithm_version,
            })
            window_rows.append(row)
            rows.append(row)
        per_event_windows[event["group_key"]] = rows

    # 2. Residuals per exit session and window, from a rolling prior window.
    residual_rows: List[Dict[str, Any]] = []
    residual_lookup: Dict[tuple, Dict[str, Any]] = {}
    by_window: Dict[str, Dict[str, float]] = {}
    for row in window_rows:
        if row["is_available"] and row["exit_date"] and row["raw_return"] is not None:
            by_window.setdefault(row["window_name"], {})[row["exit_date"]] = (
                row["raw_return"]
            )

    for window_name, series in sorted(by_window.items()):
        observations = sorted(series.items())
        for control_set_id in CONTROL_SETS:
            residuals = compute_residual_returns(
                observations, panel, control_set_id,
            )
            for exit_date, result in residuals.items():
                record = {
                    "exit_date": exit_date, "window_name": window_name, **result,
                }
                residual_rows.append(record)
                residual_lookup[(exit_date, window_name, control_set_id)] = record

    # 3. One dataset row per (event, window).
    blocked = ";".join(f"{key}={reason}" for key, reason in sorted(BLOCKED_FEATURES.items()))
    for event in events:
        timing = _dominant_timing(event)
        conflict = bool(int(event.get("timing_conflict") or 0))
        eligibility = timing_eligibility(
            timing,
            is_market_recap=bool(event.get("market_recap_count"))
            and int(event.get("market_recap_count", 0)) == int(event.get("headline_count", 0)),
            timing_conflict=conflict,
        )
        for row in per_event_windows[event["group_key"]]:
            entry = {
                "group_key": event["group_key"],
                "algorithm_version": algorithm_version,
                "experiment_id": experiment_id,
                "window_name": row["window_name"],
                "signal_date": _reactable_session(event),
                "first_reactable_session": _reactable_session(event),
                "first_reactable_at": event.get("first_reactable_at"),
                "event_information_cutoff": event.get("event_information_cutoff"),
                "event_timing_rule_version": event.get(
                    "event_timing_rule_version", TIMING_RULE_VERSION,
                ),
                "timing_conflict": 1 if conflict else 0,
                "timing_conflict_reason": event.get("timing_conflict_reason"),
                "is_tradable_window": row["is_tradable"],
                "signal_family": event.get("signal_family"),
                "event_type": event.get("event_type"),
                "primary_entity": event.get("primary_entity"),
                "timing_bucket": timing,
                "eligibility_status": (
                    eligibility["status"] if row["is_available"] else BLOCKED
                ),
                "eligibility_reason": (
                    eligibility["reason"] or row["unavailable_reason"]
                    if not (row["is_available"] and eligibility["status"] != BLOCKED)
                    else None
                ),
                "headline_count": event.get("headline_count"),
                "source_count": event.get("source_count"),
                "mean_sentiment": event.get("mean_sentiment"),
                "median_sentiment": event.get("median_sentiment"),
                "sentiment_dispersion": event.get("sentiment_dispersion"),
                "cross_source_dispersion": event.get("cross_source_dispersion"),
                "novelty": event.get("novelty"),
                "market_recap_count": event.get("market_recap_count"),
                "information_cutoff": row["information_cutoff"],
                "assumed_execution": row["assumed_execution"],
                "entry_date": row["entry_date"],
                "exit_date": row["exit_date"],
                "raw_return": row["raw_return"],
                "blocked_features": blocked,
                "dataset_version": dataset_version,
            }
            for control_set_id, column in RESIDUAL_COLUMNS.items():
                found = residual_lookup.get(
                    (row["exit_date"], row["window_name"], control_set_id)
                )
                entry[column] = found["residual"] if found else None
            dataset_rows.append(entry)

    return {
        "windows": window_rows,
        "residuals": residual_rows,
        "dataset": dataset_rows,
    }


def _dominant_timing(event: Dict[str, Any]) -> Optional[str]:
    """The timing bucket that governs an event's tradability.

    Supplied by :func:`research.timing.derive_event_timing`, which reads it from
    the same governing member that supplied the reaction session. This function
    only unwraps it; it must never re-derive a bucket independently, because a
    bucket from one member paired with a session from another is exactly the
    combination the timing rule exists to forbid.
    """

    timing = event.get("timing_bucket")
    if timing:
        return timing
    return event.get("dominant_timing")


def _reactable_session(event: Dict[str, Any]) -> Optional[str]:
    """The session that can react, preferring the explicit field."""

    return event.get("first_reactable_session") or event.get("signal_date")


def dataset_coverage(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise how much of the dataset is usable, and why the rest is not."""

    materialized = list(rows)
    total = len(materialized)
    eligible = [row for row in materialized if row["eligibility_status"] == "eligible"]
    with_return = [row for row in eligible if row.get("raw_return") is not None]

    reasons: Dict[str, int] = {}
    for row in materialized:
        if row["eligibility_status"] != "eligible":
            key = row.get("eligibility_reason") or "unspecified"
            reasons[key] = reasons.get(key, 0) + 1

    windows: Dict[str, int] = {}
    for row in with_return:
        windows[row["window_name"]] = windows.get(row["window_name"], 0) + 1

    residual_coverage = {
        column: sum(1 for row in with_return if row.get(column) is not None)
        for column in RESIDUAL_COLUMNS.values()
    }

    tradable = [row for row in with_return if row.get("is_tradable_window")]
    primary = [row for row in tradable if row["window_name"] == PRIMARY_WINDOW]

    return {
        "total_rows": total,
        "eligible_rows": len(eligible),
        "rows_with_return": len(with_return),
        "tradable_rows_with_return": len(tradable),
        "primary_window_rows": len(primary),
        "blocked_reasons": reasons,
        "rows_by_window": windows,
        "residual_coverage": residual_coverage,
        "distinct_events": len({row["group_key"] for row in materialized}),
        "distinct_eligible_events": len({row["group_key"] for row in with_return}),
        # The count that matters for inference: many events share one session,
        # and every event on a session shares its index return exactly.
        "distinct_primary_sessions": len({
            row.get("first_reactable_session") or row.get("signal_date")
            for row in primary
        }),
        "timing_conflict_rows": sum(
            1 for row in materialized if row.get("timing_conflict")
        ),
        "dataset_version": DATASET_VERSION,
    }
