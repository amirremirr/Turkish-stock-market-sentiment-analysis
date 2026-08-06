"""News-disagreement indicators.

These measure how much the *press* disagrees on a given day. That is not market
uncertainty and must not be relabelled as such: outlets can disagree loudly
about a story markets ignore, and agree completely about one that moves prices.
Every field here is named for what was measured -- disagreement among observed
news sources -- and nothing infers a market state from it.

Dispersion needs independent observers. One outlet cannot disagree with itself,
and two syndicated copies of one wire story are one observation wearing two
bylines. Below the minimum source requirement the cross-outlet fields are NULL,
because 0.0 would assert consensus that was never observed.

The government/opposition camp mapping is reused from the existing polarization
work rather than reinvented, so the daily indicator and the inferential analysis
describe the same construct.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import pstdev
from typing import Any, Dict, List, Optional

DISAGREEMENT_VERSION = "news-disagreement-v1"

# Cross-outlet statistics need at least this many independently represented
# sources before they mean anything.
MIN_SOURCES_FOR_DISPERSION = 3
# |score| above this counts as a strong directional read.
STRONG_SENTIMENT_THRESHOLD = 0.5

# Outlets whose copy is an official or wire channel rather than general press.
OFFICIAL_SOURCES = ("aa_ekonomi", "aa_politika")


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _entropy(labels: List[str]) -> Optional[float]:
    """Shannon entropy over positive/neutral/negative shares, base 3.

    Base 3 normalizes the maximum to 1.0, so the value reads directly as "how
    evenly split the day was" across the three available labels.
    """
    if not labels:
        return None
    counts: Dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    total = len(labels)
    entropy = 0.0
    for count in counts.values():
        share = count / total
        if share > 0:
            entropy -= share * math.log(share, 3)
    return entropy


def compute_disagreement(
    records: Iterable[Mapping[str, Any]],
    *,
    signal_date: str,
    signal_family: str,
    experiment_id: str,
    pro_government_sources: Iterable[str] = (),
    opposition_sources: Iterable[str] = (),
    min_sources: int = MIN_SOURCES_FOR_DISPERSION,
    method_version: str = DISAGREEMENT_VERSION,
) -> Dict[str, Any]:
    """Build one ``news_disagreement_daily`` row."""

    materialized = [dict(record) for record in records]
    scored = [
        (record, _finite(record.get("sentiment_score")))
        for record in materialized
    ]
    scored = [(record, score) for record, score in scored if score is not None]
    scores = [score for _, score in scored]
    headline_count = len(scores)

    by_source: Dict[str, List[float]] = defaultdict(list)
    for record, score in scored:
        source = record.get("source")
        if source:
            by_source[str(source)].append(score)
    source_count = len(by_source)
    sources_met = source_count >= min_sources

    outlet_means = {
        source: math.fsum(values) / len(values)
        for source, values in by_source.items()
    }

    labels = [
        str(record.get("sentiment_label")).lower()
        for record, _ in scored
        if record.get("sentiment_label") in ("positive", "neutral", "negative")
    ]

    pro = set(pro_government_sources)
    opposition = set(opposition_sources)
    pro_scores = [
        score for record, score in scored if str(record.get("source")) in pro
    ]
    opposition_scores = [
        score for record, score in scored
        if str(record.get("source")) in opposition
    ]
    camp_sources = len(
        {str(r.get("source")) for r, _ in scored if str(r.get("source")) in pro}
        | {str(r.get("source")) for r, _ in scored
           if str(r.get("source")) in opposition}
    )
    # Both camps must actually be represented; a one-sided day has no gap.
    camp_gap = (
        math.fsum(pro_scores) / len(pro_scores)
        - math.fsum(opposition_scores) / len(opposition_scores)
        if pro_scores and opposition_scores else None
    )

    official_scores = [
        score for record, score in scored
        if str(record.get("source")) in OFFICIAL_SOURCES
    ]
    general_scores = [
        score for record, score in scored
        if str(record.get("source")) not in OFFICIAL_SOURCES
    ]
    official_gap = (
        math.fsum(official_scores) / len(official_scores)
        - math.fsum(general_scores) / len(general_scores)
        if official_scores and general_scores else None
    )

    return {
        "signal_date": signal_date,
        "signal_family": signal_family,
        "experiment_id": experiment_id,
        "headline_count": headline_count,
        "source_count": source_count,
        # Within-day dispersion needs two headlines; outlet dispersion needs
        # enough independent outlets to be a real comparison.
        "within_day_std": float(pstdev(scores)) if headline_count > 1 else None,
        "cross_outlet_std": (
            float(pstdev(list(outlet_means.values())))
            if sources_met and len(outlet_means) > 1 else None
        ),
        "max_minus_min": (
            max(outlet_means.values()) - min(outlet_means.values())
            if sources_met and len(outlet_means) > 1 else None
        ),
        "strong_positive_share": (
            sum(1 for score in scores if score >= STRONG_SENTIMENT_THRESHOLD)
            / headline_count if headline_count else None
        ),
        "strong_negative_share": (
            sum(1 for score in scores if score <= -STRONG_SENTIMENT_THRESHOLD)
            / headline_count if headline_count else None
        ),
        "sentiment_entropy": _entropy(labels),
        "camp_gap": camp_gap,
        "camp_gap_sources": camp_sources if camp_gap is not None else None,
        "official_vs_media_gap": official_gap,
        "min_sources_met": 1 if sources_met else 0,
        "method_version": method_version,
    }
