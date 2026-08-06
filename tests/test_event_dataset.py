"""Candidate-event grouping, market windows, controls and the research dataset.

The two properties worth most of the effort here are timing safety and
determinism. A return that could not have been earned is worse than no return,
and a grouping that changes between runs cannot be argued with.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
import pipeline
from events.briefs import (
    WARN_SINGLE_SOURCE, WARN_SINGLETON, WARN_UNREVIEWED, build_event_brief,
)
from events.clustering import (
    CLUSTER_ALGORITHM_VERSION, group_candidate_events, jaccard, summarise_event,
    title_tokens,
)
from events.entities import classify_event_type, extract_entities
from research.controls import (
    CONTROL_SETS, KIND_CONTEMPORANEOUS, KIND_TRADABLE, build_control_panel,
    compute_residual_returns, is_tradable,
)
from research.dataset import build_event_dataset, dataset_coverage
from research.return_windows import (
    REASON_INTRADAY_UNAVAILABLE, REASON_MARKET_RECAP, REASON_UNKNOWN_TIMING,
    PriceSeries, build_return_windows, timing_eligibility,
)


def _headline(hid, title, *, source="a", family="monetary_policy",
              ts="2026-07-30T08:00:00+03:00", signal_date="2026-07-30",
              timing="pre_open", score=0.2, recap=0):
    return {
        "id": hid, "title": title, "source": source, "signal_family": family,
        "published_timestamp": ts, "published_at": ts[:10],
        "signal_date": signal_date, "timing_bucket": timing,
        "sentiment_score": score, "relevance": 0.9, "is_market_recap": recap,
    }


# -- Entities and event types ---------------------------------------------------

def test_entities_are_normalized_to_canonical_ids():
    result = extract_entities("Merkez Bankası faiz kararını açıkladı")
    assert "TCMB" in result.entity_ids
    assert result.primary_entity == "TCMB"


def test_issuer_outranks_index_as_primary_entity():
    result = extract_entities("THY hisseleri BIST 100'de yükseldi")
    assert result.primary_entity == "THYAO"


def test_unknown_headline_yields_no_entity_rather_than_a_guess():
    result = extract_entities("Hava durumu raporu yayımlandı")
    assert result.entities == frozenset()
    assert result.primary_entity is None


def test_entity_matching_respects_word_boundaries():
    assert "BORSA_ISTANBUL" in extract_entities("BIST 100 yükseldi").entity_ids
    assert "BORSA_ISTANBUL" not in extract_entities("bistro acildi").entity_ids


@pytest.mark.parametrize(
    "title, expected",
    [
        ("TCMB politika faizi kararı", "rate_decision"),
        ("Enflasyon verisi açıklandı", "data_release"),
        ("Moody's kredi notu güncellemesi", "rating_action"),
        ("Şirket bilanço açıkladı", "earnings"),
        ("Temettü kararı alındı", "corporate_action"),
        ("Genel müdür atandı", "appointment"),
    ],
)
def test_event_type_classification(title, expected):
    assert classify_event_type(title)[0] == expected


def test_unmatched_headline_has_no_event_type():
    assert classify_event_type("Belirsiz bir başlık")[0] is None


# -- Grouping -------------------------------------------------------------------

def test_similar_headlines_about_one_entity_group_together():
    records = [
        _headline(1, "Merkez Bankası faiz kararını açıkladı"),
        _headline(2, "Merkez Bankası faiz kararı belli oldu", source="b"),
    ]
    groups = group_candidate_events(records)
    assert len(groups) == 1
    assert len(groups[0].members) == 2
    assert groups[0].members[1]["match_rule"] == "entity+family+time+title"
    assert groups[0].members[1]["similarity"] > 0


def test_different_families_never_group():
    records = [
        _headline(1, "Merkez Bankası faiz kararını açıkladı"),
        _headline(2, "Merkez Bankası faiz kararını açıkladı",
                  family="global_risk"),
    ]
    assert len(group_candidate_events(records)) == 2


def test_distant_headlines_do_not_group():
    records = [
        _headline(1, "Merkez Bankası faiz kararını açıkladı"),
        _headline(2, "Merkez Bankası faiz kararını açıkladı",
                  ts="2026-08-30T08:00:00+03:00", signal_date="2026-08-31"),
    ]
    assert len(group_candidate_events(records)) == 2


def test_a_daily_recurring_headline_does_not_chain_across_months():
    """The anchor is the group's first member, so a drumbeat cannot chain."""

    records = [
        _headline(
            index,
            "Borsa güne yükselişle başladı",
            family="market_recap",
            ts=f"2026-06-{index:02d}T08:00:00+03:00",
            signal_date=f"2026-06-{index:02d}",
        )
        for index in range(1, 16)
    ]
    groups = group_candidate_events(records)
    # A group may not outlive its window: at 24h spacing inside a 48h window
    # anchored on the first member, three consecutive sessions is the maximum.
    for group in groups:
        sessions = {member["signal_date"] for member in group.members}
        assert len(sessions) <= 3, "a group outlived its time window"
    # Fifteen daily headlines must therefore split into several groups rather
    # than chaining into one long-running pseudo-event.
    assert len(groups) >= 5
    assert max(len(group.members) for group in groups) <= 3


def test_grouping_is_deterministic_regardless_of_input_order():
    records = [
        _headline(1, "Merkez Bankası faiz kararını açıkladı"),
        _headline(2, "Merkez Bankası faiz kararı belli oldu", source="b"),
        _headline(3, "Enflasyon verisi açıklandı", source="c",
                  family="inflation_macro"),
    ]
    forward = [
        sorted(g.headline_ids) for g in group_candidate_events(records)
    ]
    reverse = [
        sorted(g.headline_ids) for g in group_candidate_events(list(reversed(records)))
    ]
    assert sorted(forward) == sorted(reverse)


def test_single_source_group_is_flagged():
    records = [
        _headline(1, "Merkez Bankası faiz kararını açıkladı"),
        _headline(2, "Merkez Bankası faiz kararı belli oldu"),
    ]
    group = group_candidate_events(records)[0]
    assert group.is_single_source
    assert summarise_event(group)["is_single_source"] == 1


def test_summary_reports_cross_source_dispersion_only_with_two_voices():
    single = group_candidate_events([
        _headline(1, "Merkez Bankası faiz kararını açıkladı", score=0.5),
        _headline(2, "Merkez Bankası faiz kararı belli oldu", score=-0.5),
    ])[0]
    assert summarise_event(single)["cross_source_dispersion"] is None

    multi = group_candidate_events([
        _headline(1, "Merkez Bankası faiz kararını açıkladı", score=0.5),
        _headline(2, "Merkez Bankası faiz kararı belli oldu", source="b", score=-0.5),
    ])[0]
    assert summarise_event(multi)["cross_source_dispersion"] is not None


def test_novelty_falls_as_an_entity_repeats():
    group = group_candidate_events([_headline(1, "Merkez Bankası faiz kararı")])[0]
    assert summarise_event(group, prior_entity_events=0)["novelty"] == 1.0
    assert summarise_event(group, prior_entity_events=9)["novelty"] == pytest.approx(0.1)


def test_jaccard_and_tokens_ignore_short_and_common_words():
    assert jaccard(frozenset(), frozenset({"a"})) == 0.0
    tokens = title_tokens("Merkez Bankası bugün açıkladı")
    assert "bugun" not in tokens and "acikladi" not in tokens


# -- Return windows: timing safety ----------------------------------------------

@pytest.fixture
def price_series():
    return PriceSeries([
        {"date": "2026-07-29", "open": 100.0, "close": 102.0, "bar_status": "complete"},
        {"date": "2026-07-30", "open": 103.0, "close": 105.0, "bar_status": "complete"},
        {"date": "2026-07-31", "open": 106.0, "close": 104.0, "bar_status": "corrected"},
        # A provisional bar must be invisible to window construction.
        {"date": "2026-08-03", "open": 99.0, "close": 98.0, "bar_status": "provisional"},
    ])


def test_provisional_bars_are_never_used(price_series):
    assert "2026-08-03" not in price_series.dates
    assert price_series.get("2026-08-03") is None


def test_pre_open_uses_same_session_open_to_close(price_series):
    windows = build_return_windows("2026-07-30", "pre_open", price_series)
    assert len(windows) == 1
    window = windows[0]
    assert window.window_name == "same_session_open_to_close"
    assert window.entry_price == 103.0 and window.exit_price == 105.0
    assert window.raw_return == pytest.approx((105.0 / 103.0 - 1) * 100)
    assert window.information_cutoff.endswith("10:00:00+03:00")
    assert window.assumed_execution == window.information_cutoff


def test_post_close_cannot_act_before_the_next_open(price_series):
    windows = {
        w.window_name: w
        for w in build_return_windows("2026-07-30", "post_close", price_series)
    }
    assert set(windows) == {
        "close_to_next_open", "next_open_to_next_close", "close_to_next_close",
    }
    nxt = windows["close_to_next_close"]
    assert nxt.entry_date == "2026-07-30" and nxt.exit_date == "2026-07-31"
    assert nxt.entry_price == 105.0 and nxt.exit_price == 104.0
    # Execution is the next open, never the close that has already happened.
    assert nxt.assumed_execution.startswith("2026-07-31")
    assert nxt.information_cutoff.startswith("2026-07-30")


def test_during_session_is_blocked_for_want_of_intraday_data(price_series):
    windows = build_return_windows("2026-07-30", "during_session", price_series)
    assert len(windows) == 1
    assert not windows[0].is_available
    assert windows[0].unavailable_reason == REASON_INTRADAY_UNAVAILABLE


def test_unknown_timing_is_blocked(price_series):
    windows = build_return_windows("2026-07-30", "unknown", price_series)
    assert not windows[0].is_available
    assert windows[0].unavailable_reason == REASON_UNKNOWN_TIMING


def test_window_reports_unavailable_when_no_following_session(price_series):
    windows = build_return_windows("2026-07-31", "post_close", price_series)
    assert all(not window.is_available for window in windows)


@pytest.mark.parametrize(
    "timing, recap, status, reason",
    [
        ("pre_open", False, "eligible", None),
        ("post_close", False, "eligible", None),
        ("during_session", False, "blocked", REASON_INTRADAY_UNAVAILABLE),
        ("unknown", False, "blocked", REASON_UNKNOWN_TIMING),
        ("pre_open", True, "blocked", REASON_MARKET_RECAP),
    ],
)
def test_timing_eligibility(timing, recap, status, reason):
    result = timing_eligibility(timing, is_market_recap=recap)
    assert result["status"] == status
    assert result["reason"] == reason


# -- Controls -------------------------------------------------------------------

def test_control_sets_declare_tradability():
    assert is_tradable("em_lagged")
    assert is_tradable("em_oil_fx_lagged")
    assert not is_tradable("em_contemporaneous")
    assert CONTROL_SETS["em_contemporaneous"]["kind"] == KIND_CONTEMPORANEOUS
    assert CONTROL_SETS["em_lagged"]["kind"] == KIND_TRADABLE


def test_control_panel_materialises_lagged_values():
    panel = build_control_panel([
        {"date": "2026-07-29", "symbol": "EEM", "daily_return": 1.0},
        {"date": "2026-07-30", "symbol": "EEM", "daily_return": 2.0},
    ])
    assert panel["2026-07-29"]["EEM"] == 1.0
    # The value observable on 07-30 is the one published on 07-29.
    assert panel["2026-07-30"]["EEM_lag1"] == 1.0


def test_residuals_use_a_rolling_prior_window_only():
    """The date being described must not contribute to its own coefficients."""

    returns = [(f"2026-06-{day:02d}", float(day % 7)) for day in range(1, 29)]
    panel = {
        date: {"EEM_lag1": float(index % 5)}
        for index, (date, _) in enumerate(returns)
    }
    results = compute_residual_returns(
        returns, panel, "em_lagged", estimation_window=10, min_observations=5,
    )
    early = results["2026-06-03"]
    assert early["residual"] is None, "too little history to fit"
    assert early["estimation_observations"] < 5

    later = results["2026-06-28"]
    assert later["residual"] is not None
    # The estimation window must end strictly before the described date.
    assert later["estimation_window_end"] < "2026-06-28"


def test_none_control_set_returns_the_raw_return():
    results = compute_residual_returns(
        [("2026-07-30", 1.5)], {}, "none",
    )
    assert results["2026-07-30"]["residual"] == 1.5


def test_unknown_control_set_is_rejected():
    with pytest.raises(KeyError):
        compute_residual_returns([], {}, "nonexistent")


# -- Dataset --------------------------------------------------------------------

def test_dataset_records_timing_and_blocked_features():
    events = [{
        "group_key": "g1", "signal_date": "2026-07-30", "signal_family": "monetary_policy",
        "event_type": "rate_decision", "primary_entity": "TCMB",
        "timing_bucket": "pre_open", "headline_count": 2, "source_count": 2,
        "mean_sentiment": 0.3, "median_sentiment": 0.3, "sentiment_dispersion": 0.1,
        "cross_source_dispersion": 0.1, "novelty": 1.0, "market_recap_count": 0,
    }]
    bars = [
        {"date": "2026-07-30", "open": 103.0, "close": 105.0, "bar_status": "complete"},
    ]
    built = build_event_dataset(
        events, bars, [], experiment_id="v1-p3",
        algorithm_version=CLUSTER_ALGORITHM_VERSION,
    )
    row = built["dataset"][0]
    assert row["eligibility_status"] == "eligible"
    assert row["raw_return"] == pytest.approx((105.0 / 103.0 - 1) * 100)
    assert row["information_cutoff"]
    assert row["assumed_execution"]
    for blocked in ("intraday_prices", "consensus_expectations", "kap_structured_events"):
        assert blocked in row["blocked_features"]


def test_recap_only_event_is_blocked_from_directional_research():
    events = [{
        "group_key": "g_recap", "signal_date": "2026-07-30",
        "signal_family": "market_recap", "timing_bucket": "pre_open",
        "headline_count": 2, "market_recap_count": 2, "source_count": 1,
    }]
    bars = [
        {"date": "2026-07-30", "open": 103.0, "close": 105.0, "bar_status": "complete"},
    ]
    built = build_event_dataset(
        events, bars, [], experiment_id="v1-p3",
        algorithm_version=CLUSTER_ALGORITHM_VERSION,
    )
    row = built["dataset"][0]
    assert row["eligibility_status"] == "blocked"
    assert row["eligibility_reason"] == REASON_MARKET_RECAP


def test_dataset_coverage_reports_blocked_reasons():
    events = [
        {"group_key": "a", "signal_date": "2026-07-30", "timing_bucket": "pre_open",
         "headline_count": 1, "market_recap_count": 0},
        {"group_key": "b", "signal_date": "2026-07-30",
         "timing_bucket": "during_session", "headline_count": 1,
         "market_recap_count": 0},
    ]
    bars = [
        {"date": "2026-07-30", "open": 103.0, "close": 105.0, "bar_status": "complete"},
    ]
    built = build_event_dataset(
        events, bars, [], experiment_id="v1-p3",
        algorithm_version=CLUSTER_ALGORITHM_VERSION,
    )
    coverage = dataset_coverage(built["dataset"])
    assert coverage["rows_with_return"] == 1
    assert REASON_INTRADAY_UNAVAILABLE in coverage["blocked_reasons"]


# -- Storage, audit and briefs ---------------------------------------------------

@pytest.fixture
def event_db(tmp_path):
    path = str(tmp_path / "ev.db")
    db.init_db(path)
    with db._conn(path) as con:
        for index, (title, source) in enumerate([
            ("Merkez Bankası faiz kararını açıkladı", "aa_ekonomi"),
            ("Merkez Bankası faiz kararı belli oldu", "dunya"),
        ], start=1):
            con.execute(
                """INSERT INTO headlines (id, source, title, url, published_at,
                   scraped_at, sentiment_score, sentiment_label, scored_at,
                   p_positive, p_neutral, p_negative, model_name, experiment_id,
                   category, relevance, signal_date, timing_bucket,
                   published_timestamp, processing_status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'scored')""",
                (index, source, title, f"u{index}", "2026-07-30",
                 "2026-07-30T09:00:00Z", 0.3, "positive", "2026-07-30T09:05:00Z",
                 0.7, 0.2, 0.1, "m", "v1-p3", "rates_tcmb", 0.9, "2026-07-30",
                 "pre_open", "2026-07-30T08:00:00+03:00"),
            )
        con.execute(
            """INSERT INTO bist100_prices (date, open, high, low, close, volume,
               daily_return, bar_status) VALUES
               ('2026-07-30',103.0,106.0,102.0,105.0,1e9,NULL,'complete')"""
        )
    db.classify_signal_families(db_path=path)
    pipeline.events_step(db_path=path)
    return path


def test_event_tables_are_populated(event_db):
    with db._conn(event_db) as con:
        assert con.execute("SELECT COUNT(*) FROM event_groups").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM event_headline_map").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM event_group_entities").fetchone()[0] >= 1
        assert con.execute("SELECT COUNT(*) FROM event_research_dataset").fetchone()[0] >= 1


def test_regrouping_is_idempotent(event_db):
    with db._conn(event_db) as con:
        before = con.execute("SELECT COUNT(*) FROM event_headline_map").fetchone()[0]
    pipeline.events_step(db_path=event_db)
    with db._conn(event_db) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM event_headline_map"
        ).fetchone()[0] == before


def test_similarity_evidence_is_retained(event_db):
    with db._conn(event_db) as con:
        rows = con.execute(
            "SELECT similarity, match_rule, algorithm_version FROM event_headline_map"
        ).fetchall()
    assert all(row["algorithm_version"] == CLUSTER_ALGORITHM_VERSION for row in rows)
    assert any(row["match_rule"] == "entity+family+time+title" for row in rows)
    assert all(row["similarity"] is not None for row in rows)


def test_manual_split_and_merge_are_appended_not_applied(event_db):
    with db._conn(event_db) as con:
        key = con.execute("SELECT group_key FROM event_groups").fetchone()[0]

    db.record_event_group_action(
        key, "split", "analyst", algorithm_version=CLUSTER_ALGORITHM_VERSION,
        headline_ids=[2], rationale="second headline is a different decision",
        db_path=event_db,
    )
    db.record_event_group_action(
        key, "confirm", "analyst", algorithm_version=CLUSTER_ALGORITHM_VERSION,
        db_path=event_db,
    )
    audit = db.list_event_group_audit(key, db_path=event_db)
    assert [entry["action"] for entry in audit] == ["split", "confirm"]

    with db._conn(event_db) as con:
        # The automatic grouping is untouched; only review_state moved.
        assert con.execute(
            "SELECT COUNT(*) FROM event_headline_map"
        ).fetchone()[0] == 2
        assert con.execute(
            "SELECT review_state FROM event_groups"
        ).fetchone()[0] == "confirmed"


def test_event_group_audit_is_append_only(event_db):
    with db._conn(event_db) as con:
        key = con.execute("SELECT group_key FROM event_groups").fetchone()[0]
    db.record_event_group_action(
        key, "annotate", "analyst", algorithm_version=CLUSTER_ALGORITHM_VERSION,
        db_path=event_db,
    )
    with pytest.raises(sqlite3.IntegrityError):
        with db._conn(event_db) as con:
            con.execute("UPDATE event_group_audit SET actor='x'")
    with pytest.raises(sqlite3.IntegrityError):
        with db._conn(event_db) as con:
            con.execute("DELETE FROM event_group_audit")


def test_regrouping_preserves_manual_audit_history(event_db):
    with db._conn(event_db) as con:
        key = con.execute("SELECT group_key FROM event_groups").fetchone()[0]
    db.record_event_group_action(
        key, "annotate", "analyst", algorithm_version=CLUSTER_ALGORITHM_VERSION,
        rationale="checked", db_path=event_db,
    )
    pipeline.events_step(db_path=event_db)
    assert len(db.list_event_group_audit(db_path=event_db)) == 1


def test_brief_never_claims_a_verified_event(event_db):
    with db._conn(event_db) as con:
        event = dict(con.execute("SELECT * FROM event_groups").fetchone())
        headlines = [dict(r) for r in con.execute(
            """SELECT h.title, h.source, h.sentiment_score, m.similarity,
                      m.match_rule, h.is_market_recap, h.published_timestamp
               FROM event_headline_map m JOIN headlines h ON h.id = m.headline_id"""
        )]
        windows = [dict(r) for r in con.execute("SELECT * FROM event_return_windows")]

    brief = build_event_brief(event, headlines, windows)
    assert brief["status"] == "candidate_event_group"
    assert "not a verified real-world event" in brief["status_note"]
    assert WARN_UNREVIEWED in brief["data_quality_warnings"]
    assert brief["market_windows_for_later_evaluation"]
    # The brief may *deny* making a recommendation; it must never make one.
    assert "no trading recommendation is made" in brief["disclaimer"].lower()
    body = str(
        {key: value for key, value in brief.items()
         if key not in ("disclaimer", "status_note")}
    ).lower()
    for forbidden in ("buy", "sell", "recommend", "target price", "outperform"):
        assert forbidden not in body


def test_brief_warns_about_thin_groups():
    event = {
        "group_key": "g", "algorithm_version": "v", "is_single_source": 1,
        "is_singleton": 1, "headline_count": 1, "market_recap_count": 0,
        "unknown_timestamp_count": 0, "signal_date_span": 1,
        "primary_entity": None, "review_state": "unreviewed",
    }
    brief = build_event_brief(event, [], [])
    assert WARN_SINGLE_SOURCE in brief["data_quality_warnings"]
    assert WARN_SINGLETON in brief["data_quality_warnings"]


def test_events_step_fails_soft(event_db, monkeypatch):
    with db._conn(event_db) as con:
        before = con.execute("SELECT COUNT(*) FROM event_groups").fetchone()[0]

    def _boom(*args, **kwargs):
        raise RuntimeError("clustering exploded")

    monkeypatch.setattr(db, "get_classified_headlines", _boom)
    outcome = pipeline.events_step(db_path=event_db, return_outcome=True)
    assert outcome.status == "degraded"
    assert [w["code"] for w in outcome.warnings] == ["event_dataset_failed"]

    with db._conn(event_db) as con:
        assert con.execute("SELECT COUNT(*) FROM event_groups").fetchone()[0] == before
