"""Per-family daily descriptive signals.

Extends :func:`aggregation.signals.compute_signal_variants` rather than
reimplementing it: the mean, shares, dispersion, source count and weight
denominators all come from that function, and this module adds only the
order statistics, relevance summary and audit counts it does not cover.
Duplicating the arithmetic would let two definitions of "the signal" drift.

Insufficient samples report NULL, never zero. A zero mean reads as "the news was
neutral"; the truth is often "there were two headlines", and those are different
claims.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from statistics import median, pstdev
from typing import Any, Optional

from aggregation.signals import compute_signal_variants

FAMILY_SIGNAL_VERSION = "family-signals-v1"

# Below this many headlines a family-day is reported but marked insufficient.
MIN_HEADLINES_FOR_SUFFICIENCY = 3
# Dispersion across outlets is meaningless from a single outlet.
MIN_SOURCES_FOR_SUFFICIENCY = 2

SUFFICIENT = "sufficient"
THIN = "thin_sample"
SINGLE_SOURCE = "single_source"
INSUFFICIENT = "insufficient"


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sample_sufficiency(headline_count: int, source_count: int) -> str:
    """Classify how much weight a family-day's numbers can carry."""

    if headline_count == 0:
        return INSUFFICIENT
    if headline_count < MIN_HEADLINES_FOR_SUFFICIENCY:
        return THIN
    if source_count < MIN_SOURCES_FOR_SUFFICIENCY:
        return SINGLE_SOURCE
    return SUFFICIENT


def compute_family_signal(
    records: Iterable[Mapping[str, Any]],
    *,
    signal_date: str,
    signal_family: str,
    experiment_id: str,
    family_version: str,
    intensity_floor: float = 0.10,
    observed_sources: Optional[Mapping[int, set]] = None,
) -> dict:
    """Build one ``daily_family_signals`` row.

    ``observed_sources`` maps headline id to the set of feeds that carried it,
    so syndicated copies collapsed into one canonical headline still count as
    the breadth they actually represent.
    """

    materialized = [dict(record) for record in records]
    variants = compute_signal_variants(materialized, intensity_floor=intensity_floor)

    scores = [
        value for value in (_finite(record.get("sentiment_score"))
                            for record in materialized)
        if value is not None
    ]
    relevances = [
        value for value in (_finite(record.get("relevance"))
                            for record in materialized)
        if value is not None
    ]

    sources = {
        str(record["source"]) for record in materialized
        if record.get("source") not in (None, "")
    }
    if observed_sources:
        for record in materialized:
            identifier = record.get("id")
            if identifier is not None:
                sources.update(observed_sources.get(int(identifier), set()))
    source_count = len(sources)
    headline_count = int(variants["headline_count"])

    row = {
        "signal_date": signal_date,
        "signal_family": signal_family,
        "experiment_id": experiment_id,
        "family_version": family_version,
        "simple_mean": variants["simple_mean"],
        "relevance_weighted": variants["relevance_weighted"],
        "median_sentiment": float(median(scores)) if scores else None,
        "min_sentiment": min(scores) if scores else None,
        "max_sentiment": max(scores) if scores else None,
        "sentiment_std": float(pstdev(scores)) if len(scores) > 1 else None,
        "headline_count": headline_count,
        "source_count": source_count,
        "positive_share": variants["positive_share"],
        "neutral_share": variants["neutral_share"],
        "negative_share": variants["negative_share"],
        "avg_relevance": (
            math.fsum(relevances) / len(relevances) if relevances else None
        ),
        "market_recap_count": sum(
            1 for record in materialized if record.get("is_market_recap")
        ),
        "unknown_timing_count": sum(
            1 for record in materialized
            if record.get("timing_bucket") in (None, "", "unknown")
        ),
        "excluded_count": sum(
            1 for record in materialized if record.get("is_excluded")
        ),
        "unresolved_count": int(variants["excluded_incomplete_score_count"]),
        "ambiguous_count": sum(
            1 for record in materialized if record.get("signal_family_ambiguous")
        ),
        "sample_sufficiency": sample_sufficiency(headline_count, source_count),
    }

    # A single observation has no dispersion to report, and one outlet cannot
    # disagree with itself. NULL says so; 0.0 would claim consensus.
    if headline_count < 2:
        row["sentiment_std"] = None
    return row
