"""Prior-only historical normalization of outlet and family tone.

An outlet's absolute tone says little: some outlets are structurally more
negative than others, so -0.2 from a habitually gloomy paper is unremarkable
while the same value from an upbeat one is news. Normalizing against that
outlet's own history is what turns a level into a signal.

The whole value of this indicator depends on one property: **the value for date
t uses only observations strictly before t**. A full-sample mean would leak the
future into every historical value and make any downstream evaluation
meaningless -- the indicator would already know what it was supposed to predict.
So the window is a strict prior window, the minimum-history requirement is
enforced, and values are NULL rather than approximated when history is short.

This is *time-series* normalization: each key is compared against its own past.
It is not cross-sectional -- no key is ranked against other keys on the same
date. The two answer different questions and are kept separate deliberately.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional

ABNORMAL_TONE_VERSION = "abnormal-tone-prior-v1"

DEFAULT_WINDOW_SESSIONS = 20
DEFAULT_MIN_HISTORY = 5

SCOPE_OUTLET = "outlet"
SCOPE_OUTLET_FAMILY = "outlet_family"
SCOPE_FAMILY = "family"


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile_of(value: float, history: List[float]) -> Optional[float]:
    """Fraction of prior observations at or below *value*."""

    if not history:
        return None
    below = sum(1 for item in history if item <= value)
    return below / len(history)


def compute_abnormal_tone(
    records: Iterable[Mapping[str, Any]],
    *,
    window_sessions: int = DEFAULT_WINDOW_SESSIONS,
    min_history: int = DEFAULT_MIN_HISTORY,
    experiment_id: str = "unknown",
    method_version: str = ABNORMAL_TONE_VERSION,
) -> List[Dict[str, Any]]:
    """Return abnormal-tone rows for every (scope, key, date) with observations.

    Each record needs ``signal_date``, ``sentiment_score``, ``source`` and
    ``signal_family``. Daily means are formed first, then each date is compared
    against the preceding ``window_sessions`` daily means for the same key --
    never including the date itself.
    """

    if window_sessions < 1:
        raise ValueError("window_sessions must be at least 1")
    if min_history < 1:
        raise ValueError("min_history must be at least 1")

    # scope -> key -> date -> [scores]
    buckets: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        SCOPE_OUTLET: defaultdict(lambda: defaultdict(list)),
        SCOPE_OUTLET_FAMILY: defaultdict(lambda: defaultdict(list)),
        SCOPE_FAMILY: defaultdict(lambda: defaultdict(list)),
    }

    for record in records:
        score = _finite(record.get("sentiment_score"))
        date = record.get("signal_date")
        if score is None or not date:
            continue
        source = record.get("source")
        family = record.get("signal_family")
        if source:
            buckets[SCOPE_OUTLET][str(source)][str(date)].append(score)
        if family:
            buckets[SCOPE_FAMILY][str(family)][str(date)].append(score)
        if source and family:
            key = f"{source}|{family}"
            buckets[SCOPE_OUTLET_FAMILY][key][str(date)].append(score)

    rows: List[Dict[str, Any]] = []
    for scope, keyed in buckets.items():
        for scope_key, by_date in keyed.items():
            dates = sorted(by_date)
            daily_means = {
                date: math.fsum(by_date[date]) / len(by_date[date]) for date in dates
            }
            for index, date in enumerate(dates):
                # Strictly prior: the slice ends at index, excluding today.
                prior_dates = dates[max(0, index - window_sessions):index]
                history = [daily_means[prior] for prior in prior_dates]
                observed = daily_means[date]

                if len(history) < min_history:
                    rows.append({
                        "signal_date": date, "scope": scope, "scope_key": scope_key,
                        "experiment_id": experiment_id,
                        "window_sessions": window_sessions, "min_history": min_history,
                        "observed_mean": observed,
                        "prior_mean": None, "prior_std": None,
                        "prior_count": len(history),
                        "abnormal_tone": None, "rolling_z": None,
                        "rolling_percentile": None,
                        "method_version": method_version,
                    })
                    continue

                prior_mean = math.fsum(history) / len(history)
                variance = math.fsum(
                    (item - prior_mean) ** 2 for item in history
                ) / len(history)
                prior_std = math.sqrt(variance)
                abnormal = observed - prior_mean
                # Zero prior variance means the key never varied; a z-score
                # would divide by zero and imply infinite surprise.
                rolling_z = abnormal / prior_std if prior_std > 0 else None

                rows.append({
                    "signal_date": date, "scope": scope, "scope_key": scope_key,
                    "experiment_id": experiment_id,
                    "window_sessions": window_sessions, "min_history": min_history,
                    "observed_mean": observed,
                    "prior_mean": prior_mean, "prior_std": prior_std,
                    "prior_count": len(history),
                    "abnormal_tone": abnormal, "rolling_z": rolling_z,
                    "rolling_percentile": _percentile_of(observed, history),
                    "method_version": method_version,
                })

    rows.sort(key=lambda row: (row["signal_date"], row["scope"], row["scope_key"]))
    return rows
