"""Deterministic, storage-independent sentiment signal variants.

The pipeline historically published one aggregate whose weights combined
sentiment intensity, LLM relevance, and publication time.  This module keeps
that calculation available as a sensitivity variant and makes the unweighted
mean the explicit baseline.  It deliberately has no database, pandas, or
configuration dependency so the formula can be tested and reused by offline
analysis code.

Input records may be ordinary dictionaries (including records produced by
``DataFrame.to_dict("records")``).  ``sentiment_score``/``sentiment_label`` are
the canonical field names; ``score``/``label`` are accepted for lightweight
fixtures.  A score must be a finite real number.  Missing, non-numeric, NaN,
and infinite scores are excluded, while an explicit score of ``0.0`` is valid.

Weight resolution
-----------------
* relevance: ``relevance_weight``, then ``relevance``, otherwise ``1.0``
* intensity: ``intensity_weight``, otherwise
  ``max(abs(sentiment_score), intensity_floor)``
* source: ``source_weight``, then the optional ``source_weights`` mapping,
  otherwise ``1.0``
* time: ``time_weight``, otherwise :func:`legacy_time_weight`
* category: ``category_weight``, then the optional ``category_weights``
  mapping, otherwise ``1.0``

An explicit ``None`` weight is treated as missing and therefore takes the
documented default.  Explicit weights must be finite and non-negative; invalid
weights raise ``ValueError`` instead of silently changing the sample.

Zero-denominator contract
-------------------------
For an empty eligible sample, all four signals and dispersion are ``None``.
For a non-empty sample, ``simple_mean`` is always defined.  A weighted variant
whose weights sum to zero is ``None``; it never silently falls back to the
unweighted mean.  The corresponding ``*_weight_sum`` fields expose each
denominator.  Label shares use the eligible-headline count as their denominator
and are ``None`` only for an empty sample.  Unknown/missing labels are reported
in ``unclassified_count``, so the three named shares may sum to less than one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Real
from statistics import pstdev
from typing import Any


_LABELS = ("positive", "negative", "neutral")


def legacy_time_weight(hour: object) -> float:
    """Return the historical Istanbul publication-time multiplier.

    Missing or invalid hours receive neutral weight ``1.0``.  Hours before
    10:00 receive ``1.5``; hours from 10 through 18 receive ``1.0``; later
    hours receive ``0.8``.  This reproduces the pre-variant pipeline formula.
    Session-aware assignment belongs upstream and may instead provide an
    explicit ``time_weight`` on each record.
    """

    if hour is None or isinstance(hour, bool):
        return 1.0
    try:
        numeric_hour = float(hour)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(numeric_hour):
        return 1.0
    integer_hour = int(numeric_hour)
    if integer_hour < 0 or integer_hour > 23:
        return 1.0
    if integer_hour < 10:
        return 1.5
    if integer_hour <= 18:
        return 1.0
    return 0.8


def _finite_score(record: Mapping[str, Any]) -> float | None:
    value = record.get("sentiment_score", record.get("score"))
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def _validated_weight(value: object, *, name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite, non-negative real number")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{name} must be a finite, non-negative real number")
    return weight


def _mapped_weight(
    record: Mapping[str, Any],
    *,
    field: str,
    key_field: str,
    weights: Mapping[object, float] | None,
) -> float:
    explicit = record.get(field)
    if explicit is not None:
        return _validated_weight(explicit, name=field, default=1.0)
    key = record.get(key_field)
    mapped = weights.get(key) if weights is not None and key in weights else None
    return _validated_weight(mapped, name=field, default=1.0)


def _weighted_mean(scores: list[float], weights: list[float]) -> float | None:
    denominator = math.fsum(weights)
    if denominator == 0.0:
        return None
    numerator = math.fsum(score * weight for score, weight in zip(scores, weights))
    return numerator / denominator


def _distinct_count(values: Iterable[object]) -> int:
    distinct: set[object] = set()
    for value in values:
        if value is None or (isinstance(value, str) and value == ""):
            continue
        # SQL NULL identifiers commonly become float NaN after a pandas read.
        if isinstance(value, Real) and not math.isfinite(float(value)):
            continue
        try:
            distinct.add(value)
        except TypeError as exc:
            raise ValueError("source and event identifiers must be hashable") from exc
    return len(distinct)


def compute_signal_variants(
    records: Iterable[Mapping[str, Any]],
    *,
    intensity_floor: float = 0.10,
    source_weights: Mapping[object, float] | None = None,
    category_weights: Mapping[object, float] | None = None,
) -> dict[str, float | int | None]:
    """Compute four signal variants and descriptive audit statistics.

    ``simple_mean`` is the primary baseline.  ``relevance_weighted`` applies
    only relevance.  ``intensity_relevance_weighted`` applies relevance and
    sentiment intensity.  ``full_weighted`` applies relevance, intensity,
    source, publication-time, and category weights.  When source/category
    multipliers are not supplied, their neutral default of ``1.0`` means the
    full variant exactly matches the historical intensity × relevance × time
    formula.

    The returned flat dictionary can be augmented with a session/date key and
    passed directly to a storage adapter after selecting its desired columns.
    """

    floor = _validated_weight(
        intensity_floor, name="intensity_floor", default=0.10
    )
    materialized = list(records)

    scores: list[float] = []
    labels: list[str | None] = []
    sources: list[object] = []
    events: list[object] = []
    relevance_weights: list[float] = []
    intensity_relevance_weights: list[float] = []
    full_weights: list[float] = []

    for record in materialized:
        if not isinstance(record, Mapping):
            raise TypeError("each signal record must be a mapping")
        score = _finite_score(record)
        if score is None:
            continue

        relevance_value = record.get("relevance_weight")
        if relevance_value is None:
            relevance_value = record.get("relevance")
        relevance = _validated_weight(
            relevance_value, name="relevance_weight", default=1.0
        )
        intensity = _validated_weight(
            record.get("intensity_weight"),
            name="intensity_weight",
            default=max(abs(score), floor),
        )
        source = _mapped_weight(
            record,
            field="source_weight",
            key_field="source",
            weights=source_weights,
        )
        category = _mapped_weight(
            record,
            field="category_weight",
            key_field="category",
            weights=category_weights,
        )
        time = _validated_weight(
            record.get("time_weight"),
            name="time_weight",
            default=legacy_time_weight(record.get("published_hour")),
        )

        raw_label = record.get("sentiment_label", record.get("label"))
        label = str(raw_label).strip().lower() if raw_label is not None else None
        label = label if label in _LABELS else None

        scores.append(score)
        labels.append(label)
        sources.append(record.get("source"))
        events.append(record.get("event_id"))
        relevance_weights.append(relevance)
        intensity_relevance_weights.append(relevance * intensity)
        full_weights.append(relevance * intensity * source * time * category)

    headline_count = len(scores)
    label_counts = {label: labels.count(label) for label in _LABELS}
    unclassified_count = labels.count(None)

    if headline_count:
        simple_mean: float | None = math.fsum(scores) / headline_count
        dispersion: float | None = float(pstdev(scores))
        shares = {
            label: label_counts[label] / headline_count for label in _LABELS
        }
    else:
        simple_mean = None
        dispersion = None
        shares = {label: None for label in _LABELS}

    return {
        "simple_mean": simple_mean,
        "relevance_weighted": _weighted_mean(scores, relevance_weights),
        "intensity_relevance_weighted": _weighted_mean(
            scores, intensity_relevance_weights
        ),
        "full_weighted": _weighted_mean(scores, full_weights),
        "headline_count": headline_count,
        "input_count": len(materialized),
        "excluded_incomplete_score_count": len(materialized) - headline_count,
        "positive_count": label_counts["positive"],
        "negative_count": label_counts["negative"],
        "neutral_count": label_counts["neutral"],
        "unclassified_count": unclassified_count,
        "positive_share": shares["positive"],
        "negative_share": shares["negative"],
        "neutral_share": shares["neutral"],
        "dispersion": dispersion,
        "source_count": _distinct_count(sources),
        "event_count": _distinct_count(events),
        "relevance_weight_sum": math.fsum(relevance_weights),
        "intensity_relevance_weight_sum": math.fsum(
            intensity_relevance_weights
        ),
        "full_weight_sum": math.fsum(full_weights),
    }
