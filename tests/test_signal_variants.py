"""Exact-number tests for the pure Stage 3 signal variants."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aggregation.signals import compute_signal_variants, legacy_time_weight


def test_exact_signal_variants_and_audit_fields():
    rows = [
        {
            "sentiment_score": 0.6,
            "sentiment_label": "positive",
            "relevance": 0.5,
            "source": "outlet-a",
            "event_id": 10,
            "source_weight": 2.0,
            "time_weight": 1.5,
            "category_weight": 0.5,
        },
        {
            "sentiment_score": -0.2,
            "sentiment_label": "negative",
            "relevance": 1.0,
            "source": "outlet-b",
            "event_id": 11,
            "source_weight": 0.5,
            "time_weight": 0.8,
            "category_weight": 2.0,
        },
        {
            # A zero score is an observed neutral, not a missing score.
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "relevance": 0.25,
            "source": "outlet-a",
            "event_id": 10,
            "source_weight": 3.0,
            "time_weight": 0.5,
            "category_weight": 0.25,
        },
    ]

    result = compute_signal_variants(rows, intensity_floor=0.1)

    assert result["simple_mean"] == pytest.approx(2 / 15)
    assert result["relevance_weighted"] == pytest.approx(2 / 35)
    assert result["intensity_relevance_weighted"] == pytest.approx(4 / 15)
    # Full weights are 0.45, 0.16, and 0.009375.
    assert result["full_weighted"] == pytest.approx(0.238 / 0.619375)
    assert result["relevance_weight_sum"] == pytest.approx(1.75)
    assert result["intensity_relevance_weight_sum"] == pytest.approx(0.525)
    assert result["full_weight_sum"] == pytest.approx(0.619375)

    assert result["headline_count"] == 3
    assert result["input_count"] == 3
    assert result["excluded_incomplete_score_count"] == 0
    assert result["positive_count"] == 1
    assert result["negative_count"] == 1
    assert result["neutral_count"] == 1
    assert result["unclassified_count"] == 0
    assert result["positive_share"] == pytest.approx(1 / 3)
    assert result["negative_share"] == pytest.approx(1 / 3)
    assert result["neutral_share"] == pytest.approx(1 / 3)
    assert result["dispersion"] == pytest.approx(math.sqrt(26 / 225))
    assert result["source_count"] == 2
    assert result["event_count"] == 2


def test_incomplete_scores_are_excluded_but_explicit_neutral_is_included():
    rows = [
        {"score": 0.0, "label": "neutral", "source": "a", "event_id": 1},
        {"score": None, "label": "neutral", "source": "b", "event_id": 2},
        {"label": "positive", "source": "c", "event_id": 3},
        {"score": float("nan"), "label": "negative"},
        {"score": float("inf"), "label": "positive"},
        {"score": "0.3", "label": "positive"},
    ]

    result = compute_signal_variants(rows)

    assert result["input_count"] == 6
    assert result["headline_count"] == 1
    assert result["excluded_incomplete_score_count"] == 5
    assert result["simple_mean"] == 0.0
    assert result["full_weighted"] == 0.0
    assert result["neutral_count"] == 1
    assert result["neutral_share"] == 1.0
    assert result["dispersion"] == 0.0
    # Counts cover only records admitted to the signal sample.
    assert result["source_count"] == 1
    assert result["event_count"] == 1


def test_zero_weight_denominators_are_none_without_baseline_fallback():
    rows = [
        {"score": 0.4, "label": "positive", "relevance": 0.0},
        {"score": -0.2, "label": "negative", "relevance": 0.0},
    ]

    result = compute_signal_variants(rows)

    assert result["simple_mean"] == pytest.approx(0.1)
    assert result["relevance_weighted"] is None
    assert result["intensity_relevance_weighted"] is None
    assert result["full_weighted"] is None
    assert result["relevance_weight_sum"] == 0.0
    assert result["intensity_relevance_weight_sum"] == 0.0
    assert result["full_weight_sum"] == 0.0


def test_empty_input_has_explicit_empty_sample_contract():
    result = compute_signal_variants([])

    for name in (
        "simple_mean",
        "relevance_weighted",
        "intensity_relevance_weighted",
        "full_weighted",
        "dispersion",
        "positive_share",
        "negative_share",
        "neutral_share",
    ):
        assert result[name] is None
    assert result["headline_count"] == 0
    assert result["source_count"] == 0
    assert result["event_count"] == 0


def test_null_explicit_relevance_falls_through_and_nan_ids_are_not_counted():
    result = compute_signal_variants([
        {
            "score": 0.5,
            "label": "positive",
            "relevance_weight": None,
            "relevance": 0.25,
            "source": float("nan"),
            "event_id": float("nan"),
        }
    ])

    assert result["relevance_weight_sum"] == pytest.approx(0.25)
    assert result["source_count"] == 0
    assert result["event_count"] == 0


def test_full_variant_uses_legacy_time_and_optional_weight_maps():
    rows = [
        {
            "score": 0.5,
            "label": "positive",
            "relevance": 1.0,
            "source": "a",
            "category": "macro",
            "published_hour": 9,
        },
        {
            "score": -0.5,
            "label": "negative",
            "relevance": 1.0,
            "source": "b",
            "category": "company",
            "published_hour": 20,
        },
    ]

    result = compute_signal_variants(
        rows,
        source_weights={"a": 2.0, "b": 0.5},
        category_weights={"macro": 0.5, "company": 2.0},
    )

    # Full weights: .5*1*2*1.5*.5=.75 and .5*1*.5*.8*2=.4.
    assert result["full_weight_sum"] == pytest.approx(1.15)
    assert result["full_weighted"] == pytest.approx(0.175 / 1.15)
    # Baseline is invariant to every multiplier above.
    assert result["simple_mean"] == 0.0
    assert legacy_time_weight(None) == 1.0
    assert legacy_time_weight(9) == 1.5
    assert legacy_time_weight(18) == 1.0
    assert legacy_time_weight(19) == 0.8


@pytest.mark.parametrize("field", [
    "relevance_weight",
    "intensity_weight",
    "source_weight",
    "time_weight",
    "category_weight",
])
def test_invalid_explicit_weights_are_rejected(field):
    with pytest.raises(ValueError, match="finite, non-negative"):
        compute_signal_variants([{"score": 0.2, "label": "positive", field: -1}])
