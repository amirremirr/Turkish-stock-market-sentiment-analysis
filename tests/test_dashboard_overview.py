"""Display logic changed by the dashboard simplification.

Scope is deliberately narrow: the economic ranking's exclusion of market recap
(the one analytical change), the Overview section, and the human-readable
labels. Everything else on the page is unchanged and covered elsewhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dashboard_regime import (
    BLOCKED_REASON_LABELS, CONTROL_LABELS, FEATURE_SET_LABELS, WINDOW_LABELS,
    label, render_overview_section, tone_state,
)
from indicators.regime import build_regime_report


def _family_frame(rows):
    return pd.DataFrame([
        {
            "signal_date": "2026-06-09", "signal_family": family,
            "simple_mean": tone, "relevance_weighted": tone,
            "median_sentiment": tone, "headline_count": count,
            "source_count": sources, "market_recap_count": recap,
            "unknown_timing_count": 0, "ambiguous_count": 0,
            "sample_sufficiency": sufficiency, "family_version": "v1",
        }
        for family, tone, count, sources, recap, sufficiency in rows
    ])


def _report(rows, drivers=None):
    empty = pd.DataFrame(columns=["signal_date", "signal_family", "scope",
                                  "scope_key"])
    return build_regime_report(
        _family_frame(rows), empty, empty, empty,
        drivers if drivers is not None else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
class TestRecapExcludedFromEconomicRanking:
    def test_recap_never_wins_the_ranking(self):
        """Recap tone restates the index move; it is not an economic topic."""

        report = _report([
            ("market_recap", 0.9, 12, 4, 12, "sufficient"),
            ("monetary_policy", 0.2, 10, 3, 0, "sufficient"),
            ("fx_lira", -0.4, 9, 3, 0, "sufficient"),
        ])
        assert report["most_positive"] == "monetary_policy"
        assert report["most_negative"] == "fx_lira"

    def test_recap_never_loses_the_ranking_either(self):
        report = _report([
            ("market_recap", -0.9, 12, 4, 12, "sufficient"),
            ("monetary_policy", 0.2, 10, 3, 0, "sufficient"),
            ("fx_lira", -0.4, 9, 3, 0, "sufficient"),
        ])
        assert report["most_negative"] == "fx_lira"

    def test_ranking_is_empty_rather_than_filled_from_a_thin_topic(self):
        report = _report([
            ("market_recap", 0.9, 12, 4, 12, "sufficient"),
            ("monetary_policy", 0.2, 1, 1, 0, "thin_sample"),
        ])
        assert report["most_positive"] is None
        assert report["most_negative"] is None

    def test_recap_share_is_reported_separately(self):
        report = _report([
            ("market_recap", 0.9, 10, 4, 10, "sufficient"),
            ("monetary_policy", 0.2, 10, 3, 0, "sufficient"),
        ])
        assert report["market_recap_share"] == pytest.approx(0.5)
        assert report["market_recap"]["signal_family"] == "market_recap"

    def test_other_families_are_untouched_by_the_exclusion(self):
        """Only the ranking changed; every family still reports its own tone."""

        report = _report([
            ("market_recap", 0.9, 12, 4, 12, "sufficient"),
            ("monetary_policy", 0.2, 10, 3, 0, "sufficient"),
        ])
        tones = {f["signal_family"]: f["level"]["simple_mean"]
                 for f in report["families"]}
        assert tones == {"market_recap": 0.9, "monetary_policy": 0.2}


# ---------------------------------------------------------------------------
class TestOverview:
    def _html(self, rows, drivers=None, run_status="success"):
        return render_overview_section(
            _report(rows, drivers), run_status=run_status,
        )

    def test_shows_the_six_required_items_in_order(self):
        html = self._html([
            ("__domestic__", 0.3, 20, 4, 0, "sufficient"),
            ("monetary_policy", 0.2, 10, 3, 0, "sufficient"),
            ("fx_lira", -0.4, 9, 3, 0, "sufficient"),
        ])
        labels = re.findall(r'tile-k">([^<]+)<', html)
        assert labels == [
            "Overall news tone", "Domestic-only tone", "Most positive topic",
            "Most negative topic", "News volume", "Outlet disagreement",
        ]

    def test_ranked_topics_are_economic_not_recap(self):
        html = self._html([
            ("market_recap", 0.9, 12, 4, 12, "sufficient"),
            ("monetary_policy", 0.2, 10, 3, 0, "sufficient"),
            ("fx_lira", -0.4, 9, 3, 0, "sufficient"),
        ])
        positive = html.split("Most positive topic")[1][:400]
        assert "Monetary policy" in positive
        assert "Market recap" not in positive

    def test_recap_share_appears_outside_the_ranking(self):
        html = self._html([
            ("market_recap", 0.9, 10, 4, 10, "sufficient"),
            ("monetary_policy", 0.2, 10, 3, 0, "sufficient"),
        ])
        assert "recap-note" in html
        assert "50%" in html

    def test_empty_ranking_explains_itself(self):
        """A bare n/a tells a reader nothing they can act on."""

        html = self._html([
            ("market_recap", 0.9, 12, 4, 12, "sufficient"),
            ("monetary_policy", 0.2, 1, 1, 0, "thin_sample"),
        ])
        assert "Not enough news yet" in html
        assert "sufficient sample" in html
        assert "Monetary policy" in html, "names the best-covered topic so far"

    def test_drivers_section_uses_existing_driver_data(self):
        drivers = pd.DataFrame([
            {"signal_date": "2026-06-09", "id": 1, "title": "Rate cut cheers",
             "source": "AA", "sentiment_score": 0.8,
             "signal_family": "monetary_policy", "relevance": 1.0,
             "timing_bucket": "pre_open", "is_market_recap": 0},
            {"signal_date": "2026-06-09", "id": 2, "title": "Lira slides again",
             "source": "Dunya", "sentiment_score": -0.7,
             "signal_family": "fx_lira", "relevance": 1.0,
             "timing_bucket": "pre_open", "is_market_recap": 0},
        ])
        html = self._html(
            [("monetary_policy", 0.2, 10, 3, 0, "sufficient")], drivers,
        )
        assert "What&#x27;s driving today?" in html or "driving today" in html
        assert "Rate cut cheers" in html
        assert "Lira slides again" in html

    def test_tone_carries_a_word_not_only_a_colour(self):
        html = self._html([("monetary_policy", 0.4, 10, 3, 0, "sufficient")])
        assert "positive" in html
        for value, word in ((0.4, "clearly positive"), (-0.4, "clearly negative"),
                            (0.0, "broadly neutral"), (None, "no reading")):
            assert tone_state(value)[0] == word

    def test_degraded_run_reads_as_partially_complete(self):
        html = self._html(
            [("monetary_policy", 0.2, 10, 3, 0, "sufficient")],
            run_status="degraded",
        )
        assert "Data partially complete" in html
        assert ">degraded<" not in html

    def test_no_data_state_is_explicit(self):
        assert "No indicators are stored yet" in render_overview_section({})

    def test_overall_tone_is_the_published_aggregate_not_a_new_one(self):
        """Two definitions of "overall tone" would eventually disagree.

        Overview must display the same session aggregate the chart plots, so it
        is passed in rather than derived from the family means.
        """

        rows = [
            ("__domestic__", 0.30, 20, 4, 0, "sufficient"),
            ("monetary_policy", 0.90, 10, 3, 0, "sufficient"),
            ("fx_lira", -0.90, 10, 3, 0, "sufficient"),
        ]
        html = render_overview_section(_report(rows), overall_tone=-0.12)
        overall = html.split("Overall news tone")[1][:220]
        assert "-0.12" in overall
        assert "mildly negative" in overall

    def test_overall_tone_absent_reads_as_no_reading(self):
        html = render_overview_section(
            _report([("monetary_policy", 0.2, 10, 3, 0, "sufficient")]),
            overall_tone=None,
        )
        overall = html.split("Overall news tone")[1][:220]
        assert "n/a" in overall and "no reading" in overall

    def test_overview_makes_no_recommendation(self):
        html = self._html([("monetary_policy", 0.9, 10, 3, 0, "sufficient")])
        for forbidden in ("buy", "sell", "alpha", "should invest", "profit"):
            assert forbidden not in html.lower()
        assert "No validated predictive signal" in html


# ---------------------------------------------------------------------------
class TestReadableLabels:
    def test_identifiers_map_to_plain_language(self):
        assert label(WINDOW_LABELS, "prior_close_to_reactable_close") == (
            "Previous close to close"
        )
        assert label(CONTROL_LABELS, "residual_em_oil_fx_lagged") == (
            "Adjusted for emerging markets, oil and FX"
        )
        assert label(BLOCKED_REASON_LABELS, "intraday_prices_unavailable") == (
            "Published during trading hours (no intraday prices)"
        )
        assert label(FEATURE_SET_LABELS, "controls_plus_news") == (
            "Market factors plus news"
        )

    def test_unknown_identifier_falls_back_to_itself(self):
        """A new identifier must appear, not vanish behind an empty string."""

        assert label(WINDOW_LABELS, "some_future_window") == "some_future_window"
        assert label(CONTROL_LABELS, None) == ""

    def test_every_window_and_control_in_use_has_a_label(self):
        from research.return_windows import ALL_WINDOWS
        from research.dataset import RESIDUAL_COLUMNS

        for window in ALL_WINDOWS:
            assert window in WINDOW_LABELS, f"{window} would render raw"
        for column in RESIDUAL_COLUMNS.values():
            assert column in CONTROL_LABELS, f"{column} would render raw"

    def test_every_protocol_feature_set_has_a_label(self):
        from research.protocol import FEATURE_SETS

        for name in FEATURE_SETS:
            assert name in FEATURE_SET_LABELS, f"{name} would render raw"
