"""News-volume and attention shocks, normalized against prior sessions only.

Volume is the cheapest attention proxy available and often moves before tone
does: a story breaking generates coverage regardless of how outlets frame it.

Two distinctions matter enough to encode in the schema:

*Headline count* is not *coverage breadth*. Wires syndicate, so ten copies of
one agency story is one event covered widely, not ten independent signals.
``headline_count`` counts rows, ``observation_count`` counts distinct events
where an event id is available, and ``source_breadth`` counts the outlets that
actually carried it. Only the last is named as breadth, so the numbers cannot be
quietly swapped for each other.

*Prior only.* Every rolling statistic uses sessions strictly before the date
being described. A window that included today would make an unusual day look
normal by folding it into its own baseline.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional

VOLUME_SHOCK_VERSION = "news-volume-prior-v1"

DEFAULT_WINDOW_SESSIONS = 20
DEFAULT_MIN_HISTORY = 5

ALL_FAMILIES_KEY = "__all__"


def _percentile_of(value: float, history: List[float]) -> Optional[float]:
    if not history:
        return None
    return sum(1 for item in history if item <= value) / len(history)


def _change(current: float, history: List[float], lag: int) -> Optional[float]:
    """Change versus the session *lag* steps back, or None if unavailable."""

    if len(history) < lag:
        return None
    return current - history[-lag]


def compute_volume_shocks(
    records: Iterable[Mapping[str, Any]],
    *,
    window_sessions: int = DEFAULT_WINDOW_SESSIONS,
    min_history: int = DEFAULT_MIN_HISTORY,
    experiment_id: str = "unknown",
    method_version: str = VOLUME_SHOCK_VERSION,
    observed_sources: Optional[Mapping[int, set]] = None,
) -> List[Dict[str, Any]]:
    """Return volume rows per family per session, plus an all-news row.

    The all-news series is keyed ``__all__`` so it lives in the same table as
    the families without pretending to be one of them.
    """

    if window_sessions < 1:
        raise ValueError("window_sessions must be at least 1")

    # family -> date -> aggregates
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    events: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    sources: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for record in records:
        date = record.get("signal_date")
        if not date:
            continue
        date = str(date)
        family = str(record.get("signal_family") or "other")
        identifier = record.get("id")
        source = record.get("source")

        for key in (family, ALL_FAMILIES_KEY):
            counts[key][date] += 1
            event_id = record.get("event_id")
            # Fall back to the headline id so a missing event mapping counts as
            # its own observation rather than collapsing distinct rows.
            events[key][date].add(
                f"e{event_id}" if event_id is not None else f"h{identifier}"
            )
            if source:
                sources[key][date].add(str(source))
            if observed_sources and identifier is not None:
                sources[key][date].update(observed_sources.get(int(identifier), set()))

    rows: List[Dict[str, Any]] = []
    for family, by_date in counts.items():
        dates = sorted(by_date)
        series = [float(by_date[date]) for date in dates]
        for index, date in enumerate(dates):
            prior = series[max(0, index - window_sessions):index]
            current = series[index]

            if len(prior) < min_history:
                prior_mean = prior_std = volume_z = percentile = None
            else:
                prior_mean = math.fsum(prior) / len(prior)
                variance = math.fsum(
                    (item - prior_mean) ** 2 for item in prior
                ) / len(prior)
                prior_std = math.sqrt(variance)
                volume_z = (
                    (current - prior_mean) / prior_std if prior_std > 0 else None
                )
                percentile = _percentile_of(current, prior)

            rows.append({
                "signal_date": date,
                "signal_family": family,
                "experiment_id": experiment_id,
                "headline_count": int(current),
                "observation_count": len(events[family][date]),
                "source_breadth": len(sources[family][date]),
                "window_sessions": window_sessions,
                "min_history": min_history,
                "prior_mean": prior_mean,
                "prior_std": prior_std,
                "prior_count": len(prior),
                "volume_z": volume_z,
                "volume_percentile": percentile,
                "change_1": _change(current, prior, 1),
                "change_5": _change(current, prior, 5),
                "change_20": _change(current, prior, 20),
                "method_version": method_version,
            })

    rows.sort(key=lambda row: (row["signal_date"], row["signal_family"]))
    return rows
