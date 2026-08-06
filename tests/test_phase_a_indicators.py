"""Descriptive indicator contracts: arithmetic, NULL discipline, and no leakage.

The property that matters most here is temporal: an indicator describing date t
must be computable from information available before t. A leak would make every
downstream evaluation meaningless, and it is invisible in ordinary output -- the
numbers look fine either way. So it is tested directly, by mutating the future
and requiring the past not to move.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
import pipeline
from indicators.abnormal_tone import SCOPE_FAMILY, SCOPE_OUTLET, compute_abnormal_tone
from indicators.disagreement import (
    MIN_SOURCES_FOR_DISPERSION, compute_disagreement,
)
from indicators.family_signals import (
    FAMILY_SIGNAL_VERSION, compute_family_signal, sample_sufficiency,
)
from indicators.regime import build_coverage_report, build_regime_report, write_reports
from indicators.volume_shock import ALL_FAMILIES_KEY, compute_volume_shocks


def _record(date, score, source="a", family="monetary_policy", **extra):
    base = {
        "id": extra.pop("id", abs(hash((date, score, source))) % 10_000),
        "signal_date": date,
        "sentiment_score": score,
        "sentiment_label": (
            "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        ),
        "source": source,
        "signal_family": family,
        "relevance": extra.pop("relevance", 1.0),
        "timing_bucket": extra.pop("timing_bucket", "pre_open"),
    }
    base.update(extra)
    return base


# -- Family signals -------------------------------------------------------------

def test_family_signal_arithmetic():
    records = [
        _record("2026-07-20", 0.6, "a"),
        _record("2026-07-20", -0.2, "b"),
        _record("2026-07-20", 0.1, "c"),
    ]
    row = compute_family_signal(
        records, signal_date="2026-07-20", signal_family="monetary_policy",
        experiment_id="v1-p3", family_version="signal-family-v1",
    )
    assert row["headline_count"] == 3
    assert row["source_count"] == 3
    assert row["simple_mean"] == pytest.approx((0.6 - 0.2 + 0.1) / 3)
    assert row["median_sentiment"] == pytest.approx(0.1)
    assert row["min_sentiment"] == pytest.approx(-0.2)
    assert row["max_sentiment"] == pytest.approx(0.6)
    assert row["avg_relevance"] == pytest.approx(1.0)
    assert row["sample_sufficiency"] == "sufficient"


def test_single_headline_reports_null_dispersion_not_zero():
    row = compute_family_signal(
        [_record("2026-07-20", 0.5)], signal_date="2026-07-20",
        signal_family="fx_lira", experiment_id="v1-p3", family_version="v1",
    )
    assert row["sentiment_std"] is None, "one observation has no dispersion"
    assert row["sample_sufficiency"] == "thin_sample"


def test_sample_sufficiency_classification():
    assert sample_sufficiency(0, 0) == "insufficient"
    assert sample_sufficiency(2, 2) == "thin_sample"
    assert sample_sufficiency(5, 1) == "single_source"
    assert sample_sufficiency(5, 3) == "sufficient"


def test_family_signal_counts_recap_timing_and_ambiguity():
    records = [
        _record("2026-07-20", 0.3, "a", is_market_recap=1, timing_bucket="unknown"),
        _record("2026-07-20", 0.1, "b", signal_family_ambiguous=1),
        _record("2026-07-20", -0.4, "c"),
    ]
    row = compute_family_signal(
        records, signal_date="2026-07-20", signal_family="market_recap",
        experiment_id="v1-p3", family_version="v1",
    )
    assert row["market_recap_count"] == 1
    assert row["unknown_timing_count"] == 1
    assert row["ambiguous_count"] == 1


def test_syndicated_sources_expand_breadth_without_inflating_headlines():
    records = [_record("2026-07-20", 0.2, "a", id=7)]
    row = compute_family_signal(
        records, signal_date="2026-07-20", signal_family="other",
        experiment_id="v1-p3", family_version="v1",
        observed_sources={7: {"a", "b", "c"}},
    )
    assert row["headline_count"] == 1
    assert row["source_count"] == 3


# -- Abnormal tone: prior-only ---------------------------------------------------

def _series(days, score_for):
    return [
        _record(f"2026-07-{day:02d}", score_for(day), "outlet_a", "monetary_policy")
        for day in days
    ]


def test_abnormal_tone_returns_null_below_minimum_history():
    rows = compute_abnormal_tone(
        _series(range(1, 4), lambda d: 0.1), min_history=5,
    )
    family_rows = [r for r in rows if r["scope"] == SCOPE_FAMILY]
    assert family_rows, "the family scope must still be reported"
    assert all(row["abnormal_tone"] is None for row in family_rows)
    assert all(row["rolling_z"] is None for row in family_rows)
    assert all(row["prior_count"] < 5 for row in family_rows)


def test_abnormal_tone_uses_only_prior_observations():
    rows = compute_abnormal_tone(
        _series(range(1, 12), lambda d: 0.1 if d < 11 else 0.9),
        window_sessions=20, min_history=5,
    )
    last = [
        r for r in rows if r["scope"] == SCOPE_FAMILY and r["signal_date"] == "2026-07-11"
    ][0]
    # Prior mean must reflect only the flat 0.1 history, never the 0.9 spike.
    assert last["prior_mean"] == pytest.approx(0.1)
    assert last["observed_mean"] == pytest.approx(0.9)
    assert last["abnormal_tone"] == pytest.approx(0.8)


def test_changing_the_future_cannot_change_a_past_indicator():
    """Property test: date t is invariant to every observation dated >= t."""

    base = _series(range(1, 16), lambda d: 0.1 * (d % 3))
    baseline = {
        (r["scope"], r["scope_key"], r["signal_date"]): (
            r["prior_mean"], r["prior_std"], r["abnormal_tone"],
            r["rolling_z"], r["rolling_percentile"],
        )
        for r in compute_abnormal_tone(base, min_history=3)
    }

    # Mutate every observation from 2026-07-10 onward, and add new later ones.
    mutated = [
        dict(record, sentiment_score=-0.95)
        if record["signal_date"] >= "2026-07-10" else record
        for record in base
    ]
    mutated += _series(range(16, 21), lambda d: 0.99)
    after = compute_abnormal_tone(mutated, min_history=3)

    for row in after:
        key = (row["scope"], row["scope_key"], row["signal_date"])
        if row["signal_date"] >= "2026-07-10" or key not in baseline:
            continue
        assert baseline[key] == (
            row["prior_mean"], row["prior_std"], row["abnormal_tone"],
            row["rolling_z"], row["rolling_percentile"],
        ), f"indicator for {key} moved when only later dates changed"


def test_abnormal_tone_covers_all_three_scopes():
    records = [
        _record("2026-07-01", 0.1, "a", "fx_lira"),
        _record("2026-07-02", 0.2, "a", "fx_lira"),
    ]
    scopes = {row["scope"] for row in compute_abnormal_tone(records, min_history=1)}
    assert scopes == {"outlet", "outlet_family", "family"}


def test_zero_prior_variance_yields_null_z_not_infinity():
    rows = compute_abnormal_tone(
        _series(range(1, 10), lambda d: 0.25), min_history=3,
    )
    last = [r for r in rows if r["scope"] == SCOPE_OUTLET][-1]
    assert last["prior_std"] == pytest.approx(0.0)
    assert last["rolling_z"] is None


# -- Disagreement ----------------------------------------------------------------

def test_disagreement_requires_a_minimum_number_of_sources():
    records = [_record("2026-07-20", 0.5, "a"), _record("2026-07-20", -0.5, "a")]
    row = compute_disagreement(
        records, signal_date="2026-07-20", signal_family="other",
        experiment_id="v1-p3",
    )
    assert row["source_count"] == 1
    assert row["min_sources_met"] == 0
    assert row["cross_outlet_std"] is None, "NULL, never a fabricated zero"
    assert row["max_minus_min"] is None
    # Within-day dispersion across two headlines is still defensible.
    assert row["within_day_std"] is not None


def test_disagreement_reports_cross_outlet_spread_when_sources_suffice():
    records = [
        _record("2026-07-20", 0.8, "a"), _record("2026-07-20", 0.0, "b"),
        _record("2026-07-20", -0.8, "c"),
    ]
    row = compute_disagreement(
        records, signal_date="2026-07-20", signal_family="other",
        experiment_id="v1-p3",
    )
    assert row["source_count"] == MIN_SOURCES_FOR_DISPERSION
    assert row["min_sources_met"] == 1
    assert row["cross_outlet_std"] is not None
    assert row["max_minus_min"] == pytest.approx(1.6)


def test_camp_gap_needs_both_camps_present():
    one_sided = compute_disagreement(
        [_record("2026-07-20", 0.5, "sabah_ekonomi")],
        signal_date="2026-07-20", signal_family="other", experiment_id="v1-p3",
        pro_government_sources=["sabah_ekonomi"], opposition_sources=["sozcu_ekonomi"],
    )
    assert one_sided["camp_gap"] is None

    both = compute_disagreement(
        [_record("2026-07-20", 0.6, "sabah_ekonomi"),
         _record("2026-07-20", -0.2, "sozcu_ekonomi")],
        signal_date="2026-07-20", signal_family="other", experiment_id="v1-p3",
        pro_government_sources=["sabah_ekonomi"], opposition_sources=["sozcu_ekonomi"],
    )
    assert both["camp_gap"] == pytest.approx(0.8)


def test_entropy_is_maximal_on_an_even_three_way_split():
    records = [
        _record("2026-07-20", 0.5, "a"), _record("2026-07-20", 0.0, "b"),
        _record("2026-07-20", -0.5, "c"),
    ]
    row = compute_disagreement(
        records, signal_date="2026-07-20", signal_family="other",
        experiment_id="v1-p3",
    )
    assert row["sentiment_entropy"] == pytest.approx(1.0)


def test_official_versus_general_media_gap():
    row = compute_disagreement(
        [_record("2026-07-20", 0.4, "aa_ekonomi"),
         _record("2026-07-20", -0.2, "dunya")],
        signal_date="2026-07-20", signal_family="other", experiment_id="v1-p3",
    )
    assert row["official_vs_media_gap"] == pytest.approx(0.6)


# -- Volume shocks ---------------------------------------------------------------

def test_volume_shock_uses_prior_sessions_only():
    """A spike must be measured against the days before it, not including itself."""

    records = []
    # Baseline alternates 2/4 so the prior window has real variance; a perfectly
    # flat baseline would make the z-score genuinely undefined.
    for day in range(1, 11):
        count = (2 if day % 2 else 4) if day < 10 else 40
        for index in range(count):
            records.append(
                _record(f"2026-07-{day:02d}", 0.1, f"src{index % 3}", "fx_lira")
            )
    rows = compute_volume_shocks(records, min_history=3, window_sessions=20)
    spike = [
        r for r in rows
        if r["signal_family"] == "fx_lira" and r["signal_date"] == "2026-07-10"
    ][0]
    assert spike["headline_count"] == 40
    assert spike["prior_count"] == 9
    assert spike["prior_mean"] == pytest.approx(
        (2 + 4 + 2 + 4 + 2 + 4 + 2 + 4 + 2) / 9
    ), "today must not be folded into its own baseline"
    assert spike["volume_z"] > 3
    assert spike["volume_percentile"] == pytest.approx(1.0)


def test_zero_variance_baseline_yields_null_z_not_infinity():
    """A flat history cannot say how surprising today is."""

    records = [
        _record(f"2026-07-{day:02d}", 0.1, "a", "fx_lira")
        for day in range(1, 11) for _ in range(2)
    ]
    rows = compute_volume_shocks(records, min_history=3)
    last = [
        r for r in rows
        if r["signal_family"] == "fx_lira" and r["signal_date"] == "2026-07-10"
    ][0]
    assert last["prior_std"] == pytest.approx(0.0)
    assert last["volume_z"] is None


def test_volume_shock_is_null_below_minimum_history():
    records = [_record("2026-07-01", 0.1), _record("2026-07-02", 0.1)]
    rows = compute_volume_shocks(records, min_history=5)
    assert all(row["volume_z"] is None for row in rows)
    assert all(row["prior_mean"] is None for row in rows)


def test_source_breadth_counts_outlets_not_headlines():
    records = [
        _record("2026-07-20", 0.1, "a", id=1),
        _record("2026-07-20", 0.1, "a", id=2),
        _record("2026-07-20", 0.1, "a", id=3),
    ]
    rows = compute_volume_shocks(records, min_history=1)
    row = [r for r in rows if r["signal_family"] == ALL_FAMILIES_KEY][0]
    assert row["headline_count"] == 3
    assert row["source_breadth"] == 1, "three stories from one outlet is one outlet"


def test_syndicated_copies_do_not_inflate_observation_count():
    records = [
        _record("2026-07-20", 0.1, "a", id=1, event_id=99),
        _record("2026-07-20", 0.1, "b", id=2, event_id=99),
    ]
    rows = compute_volume_shocks(records, min_history=1)
    row = [r for r in rows if r["signal_family"] == ALL_FAMILIES_KEY][0]
    assert row["headline_count"] == 2
    assert row["observation_count"] == 1, "one event carried twice is one event"
    assert row["source_breadth"] == 2, "but it is two outlets of coverage breadth"


def test_all_families_series_is_keyed_separately():
    rows = compute_volume_shocks(
        [_record("2026-07-20", 0.1, "a", "fx_lira")], min_history=1
    )
    keys = {row["signal_family"] for row in rows}
    assert keys == {"fx_lira", ALL_FAMILIES_KEY}


# -- Regime report ---------------------------------------------------------------

@pytest.fixture
def indicator_db(tmp_path):
    path = str(tmp_path / "ind.db")
    db.init_db(path)
    with db._conn(path) as con:
        for index in range(1, 31):
            day = f"2026-07-{(index % 10) + 1:02d}"
            family = "monetary_policy" if index % 2 else "global_risk"
            con.execute(
                """INSERT INTO headlines (id, source, title, url, published_at,
                   scraped_at, sentiment_score, sentiment_label, scored_at,
                   p_positive, p_neutral, p_negative, model_name, experiment_id,
                   category, relevance, signal_date, timing_bucket,
                   processing_status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'scored')""",
                (index, f"src{index % 4}", f"Başlık {index}", f"u{index}", day,
                 f"{day}T09:00:00Z", 0.1 * (index % 5 - 2), "neutral",
                 f"{day}T09:05:00Z", 0.4, 0.4, 0.2,
                 "gpt-5-mini-2025-08-07/p3", "v1-p3",
                 "rates_tcmb" if index % 2 else "global_risk", 0.9, day, "pre_open"),
            )
    pipeline.indicators_step(db_path=path)
    return path


def test_regime_report_is_deterministic(indicator_db):
    def _build():
        return build_regime_report(
            db.read_table("daily_family_signals", indicator_db),
            db.read_table("abnormal_tone_daily", indicator_db),
            db.read_table("news_disagreement_daily", indicator_db),
            db.read_table("news_volume_daily", indicator_db),
            db.get_classified_headlines(db_path=indicator_db),
        )
    import json
    first, second = _build(), _build()
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(
        second, sort_keys=True, default=str
    )


def test_regime_report_separates_level_change_abnormal_and_attention(indicator_db):
    report = build_regime_report(
        db.read_table("daily_family_signals", indicator_db),
        db.read_table("abnormal_tone_daily", indicator_db),
        db.read_table("news_disagreement_daily", indicator_db),
        db.read_table("news_volume_daily", indicator_db),
        db.get_classified_headlines(db_path=indicator_db),
    )
    assert report["status"] == "ok"
    for family in report["families"]:
        assert set(family) >= {
            "level", "change", "abnormal", "disagreement", "attention", "quality"
        }
    assert any("not a validated predictive" in note or "Descriptive only" in note
               for note in report["notes"])


def test_regime_report_excludes_the_domestic_composite_from_rankings(indicator_db):
    report = build_regime_report(
        db.read_table("daily_family_signals", indicator_db),
        db.read_table("abnormal_tone_daily", indicator_db),
        db.read_table("news_disagreement_daily", indicator_db),
        db.read_table("news_volume_daily", indicator_db),
        db.get_classified_headlines(db_path=indicator_db),
    )
    assert report["most_positive"] != "__domestic__"
    assert report["most_negative"] != "__domestic__"
    assert report["domestic_only"] is not None


def test_regime_report_handles_an_empty_database():
    report = build_regime_report(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert report["status"] == "no_data"
    assert report["families"] == []


def test_reports_write_json_and_csv(indicator_db, tmp_path):
    report = build_regime_report(
        db.read_table("daily_family_signals", indicator_db),
        db.read_table("abnormal_tone_daily", indicator_db),
        db.read_table("news_disagreement_daily", indicator_db),
        db.read_table("news_volume_daily", indicator_db),
        db.get_classified_headlines(db_path=indicator_db),
    )
    coverage = build_coverage_report(db.get_classified_headlines(db_path=indicator_db))
    written = write_reports(report, coverage, tmp_path / "out")
    assert set(written) == {"regime_json", "coverage_json", "regime_csv", "coverage_csv"}
    for path in written.values():
        assert Path(path).stat().st_size > 0


def test_coverage_report_surfaces_ambiguity_and_recap_share(indicator_db):
    coverage = build_coverage_report(db.get_classified_headlines(db_path=indicator_db))
    assert coverage["total"] > 0
    assert "ambiguous_share" in coverage
    assert "market_recap_share" in coverage
    assert "other_family_share" in coverage
    assert coverage["assignment_rules"], "every assignment records its rule"
    assert coverage["family_by_timing"]
    assert coverage["family_by_experiment"]


# -- Storage identity ------------------------------------------------------------

def test_family_signal_rows_are_keyed_by_version(indicator_db):
    with db._conn(indicator_db) as con:
        duplicates = con.execute(
            """SELECT signal_date, signal_family, experiment_id, family_version,
                      COUNT(*) AS n
               FROM daily_family_signals
               GROUP BY 1,2,3,4 HAVING n > 1"""
        ).fetchall()
    assert duplicates == []


def test_indicator_step_is_idempotent(indicator_db):
    before = db.read_table("daily_family_signals", indicator_db, "signal_date")
    pipeline.indicators_step(db_path=indicator_db)
    after = db.read_table("daily_family_signals", indicator_db, "signal_date")
    assert len(before) == len(after)


def test_domestic_composite_excludes_global_risk(indicator_db):
    frame = db.read_table("daily_family_signals", indicator_db)
    domestic = frame[frame["signal_family"] == "__domestic__"]
    global_rows = frame[frame["signal_family"] == "global_risk"]
    assert not domestic.empty
    assert not global_rows.empty
    for session in domestic["signal_date"]:
        dom = domestic[domestic["signal_date"] == session]["headline_count"].iloc[0]
        glob = global_rows[global_rows["signal_date"] == session]
        if not glob.empty:
            total = frame[
                (frame["signal_date"] == session)
                & (frame["signal_family"] != "__domestic__")
            ]["headline_count"].sum()
            assert dom <= total - glob["headline_count"].iloc[0]


def test_existing_overall_variant_table_is_untouched_by_indicators(indicator_db):
    """Phase A must not modify the pre-existing overall aggregate."""

    with db._conn(indicator_db) as con:
        con.execute(
            """INSERT INTO daily_signal_variants
               (signal_date, simple_mean, headline_count, positive_count,
                negative_count, neutral_count, updated_at)
               VALUES ('2026-07-01', 0.5, 10, 5, 2, 3, '2026-07-01T00:00:00Z')"""
        )
        before = con.execute("SELECT * FROM daily_signal_variants").fetchall()
    pipeline.indicators_step(db_path=indicator_db)
    with db._conn(indicator_db) as con:
        after = con.execute("SELECT * FROM daily_signal_variants").fetchall()
    assert before == after
