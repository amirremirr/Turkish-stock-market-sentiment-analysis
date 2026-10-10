"""H1-H9 on synthetic contexts: an effect that was planted is found, an absent
one is not, and thin or missing data is reported as such.

A null check here uses one fixed seed, so it is a smoke test, not a calibration
study; the rejection rate under the null is measured in test_core.
"""

import numpy as np
import pandas as pd
import pytest

from stock_research import fixtures as F
from stock_research.hypotheses import common, h1, h2, h3, h4, h5, h6, h7, h8, h9, prep, registry
from stock_research.hypotheses.common import DATA_INSUFFICIENT, INCONCLUSIVE, SUPPORTED


def found(result, level=0.01):
    """Primary estimate in the hypothesised direction and significant."""

    primary, spec = result["primary"], registry.SPECS[result["hypothesis"]]
    return (result["status"] is None and primary["p"] is not None and primary["p"] < level
            and spec.direction * primary["estimate"] > 0)


@pytest.fixture(scope="module")
def null_news():
    return F.news_context()


@pytest.fixture(scope="module")
def null_kap():
    return F.kap_context()


@pytest.fixture(scope="module")
def effect_kap():
    return F.kap_context(effect=True)


@pytest.fixture(scope="module")
def null_limits():
    return F.limit_context()


@pytest.fixture(scope="module")
def effect_limits():
    return F.limit_context(effect=True)


# -- Planted effects are recovered, absent ones are not ------------------------
@pytest.mark.parametrize("module,name", [(h1, "h1"), (h2, "h2"), (h3, "h3"), (h4, "h4")])
def test_news_hypotheses_find_their_planted_effect(module, name, null_news):
    assert found(module.run(F.news_context(effect=name)))
    assert not found(module.run(null_news))


def test_h5_and_h6_find_planted_effects_and_not_absent_ones(null_limits, effect_limits):
    for module in (h5, h6):
        assert found(module.run(effect_limits))
        assert not found(module.run(null_limits))


def test_h7_and_h8_find_planted_effects_and_not_absent_ones(null_kap, effect_kap):
    for module in (h7, h8):
        assert found(module.run(effect_kap))
        assert not found(module.run(null_kap))
    primary = h7.run(effect_kap)["primary"]
    assert (primary["highest_category"], primary["lowest_category"]) == ("cat_up", "cat_down")


def test_h9_finds_a_planted_lead_and_not_an_absent_one():
    assert found(h9.run(F.language_context(effect=True)))
    assert not found(h9.run(F.language_context()))


def test_h2_null_fixture_has_no_built_in_gap_continuation(null_news):
    """The overnight and intraday parts are drawn independently, so a gap
    coefficient far from zero here would be the fixture's doing."""

    primary = h2.run(null_news)["primary"]
    assert abs(primary["estimate"]) < 0.15 and primary["p"] > 0.05


# -- Missing or thin data is data_insufficient, not a weak result ---------------
def test_blocked_requirement_stops_before_any_estimate(null_news):
    availability = dict(F.ALL_AVAILABLE)
    availability["point_in_time_market_cap"] = {"available": False, "reason": "no source"}
    ctx = F.news_context(availability=availability)
    result = h1.run(ctx)
    assert result["status"] == DATA_INSUFFICIENT and result["primary"] is None
    assert "point_in_time_market_cap" in result["sufficiency"]["reasons"][0]


def test_thin_samples_are_insufficient():
    thin_news = F.news_context(news_tickers=6, events_per_ticker=8)
    for module in (h1, h2, h3, h4):
        result = module.run(thin_news)
        assert result["status"] == DATA_INSUFFICIENT and result["primary"] is None
    thin_kap = F.kap_context(per_category=12, insider_buys=10)
    assert h7.run(thin_kap)["status"] == DATA_INSUFFICIENT
    assert h8.run(thin_kap)["status"] == DATA_INSUFFICIENT
    thin_limits = F.limit_context(limit_downs=20, streak_starts=20)
    assert h5.run(thin_limits)["status"] == DATA_INSUFFICIENT
    assert h6.run(thin_limits)["status"] == DATA_INSUFFICIENT
    assert h9.run(F.language_context(events=30))["status"] == DATA_INSUFFICIENT


def test_h5_and_h6_report_price_only_counts_when_news_is_incomplete(null_limits):
    availability = dict(F.ALL_AVAILABLE)
    availability["complete_company_news"] = {"available": False, "reason": "4% sample"}
    ctx = common.Context(panel=null_limits.panel, news_events=null_limits.news_events,
                         availability=availability)
    down, up = h5.run(ctx), h6.run(ctx)
    assert down["status"] == up["status"] == DATA_INSUFFICIENT
    assert down["sufficiency"]["counts"]["limit_down_closes"] == 240
    assert up["sufficiency"]["counts"]["at_risk_sessions"] > 300


# -- The verdict rule ----------------------------------------------------------
def _primary(estimate, p, ci, p_holm):
    return {"estimate": estimate, "p": p, "ci": ci, "p_holm": p_holm}


def test_unadjusted_significance_is_not_support():
    spec = registry.SPECS["H8"]                       # direction +1, material 0.01
    assert common.decide(spec, _primary(0.03, 0.03, [0.003, 0.057], 0.27)) == INCONCLUSIVE
    assert common.decide(spec, _primary(0.03, 0.001, [0.015, 0.045], 0.009)) == SUPPORTED


def test_a_significant_but_immaterial_effect_is_not_support():
    spec = registry.SPECS["H8"]
    assert common.decide(spec, _primary(0.004, 0.0001, [0.003, 0.005], 0.0009)) == common.NOT_SUPPORTED


def test_wrong_sign_is_never_support():
    spec = registry.SPECS["H4"]                       # hypothesised negative
    assert common.decide(spec, _primary(+0.01, 0.0001, [0.006, 0.014], 0.0009)) == common.NOT_SUPPORTED


def test_execution_dependent_support_is_labelled_unverifiable():
    spec = registry.SPECS["H5"]
    verdict = common.decide(spec, _primary(0.03, 0.0001, [0.02, 0.04], 0.0009))
    assert verdict == common.EXECUTION_NOT_VERIFIABLE


def test_run_all_applies_holm_over_nine_and_marks_synthetic_claims(effect_kap):
    results = registry.run_all(effect_kap)
    assert set(results) == {f"H{i}" for i in range(1, 10)}
    assert results["H7"]["status"] == SUPPORTED and results["H8"]["status"] == SUPPORTED
    assert results["H8"]["primary"]["p_holm"] >= results["H8"]["primary"]["p"]
    for key in ("H1", "H2", "H3", "H4", "H5", "H6", "H9"):
        assert results[key]["status"] == DATA_INSUFFICIENT
    assert not any(r["claim_allowed"] for r in results.values())


def test_holm_family_stays_nine_when_only_one_test_runs(monkeypatch, null_kap):
    def insufficient(ctx, key):
        return common.insufficient(common.base_result(registry.SPECS[key], ctx), ["x"])

    runners = {key: (lambda ctx, key=key: insufficient(ctx, key)) for key in registry.RUNNERS}

    def lone(ctx):
        result = common.base_result(registry.SPECS["H8"], ctx)
        result["primary"] = common.primary_block(
            registry.SPECS["H8"], estimate=0.03, se=0.012, ci=[0.006, 0.054], p=0.012,
            n=100, clusters=[60])
        return result

    runners["H8"] = lone
    monkeypatch.setattr(registry, "RUNNERS", runners)
    result = registry.run_all(null_kap)["H8"]
    assert result["primary"]["p_holm"] == pytest.approx(0.108)
    assert result["status"] == INCONCLUSIVE


# -- Protocol ------------------------------------------------------------------
def test_protocol_hash_is_stable_and_moves_with_any_rule(monkeypatch):
    first = registry.protocol_hash()
    assert first == registry.protocol_hash()
    from stock_research import config
    monkeypatch.setattr(config, "MIN_EVENTS_PER_GROUP", 31)
    assert registry.protocol_hash() != first


def test_registration_is_idempotent_and_results_are_append_only(tmp_path, null_kap):
    import sqlite3

    path = str(tmp_path / "s.db")
    first, second = registry.register(path), registry.register(path)
    assert not first["already_registered"] and second["already_registered"]
    results = registry.run_all(null_kap)
    manifest = registry.manifest(null_kap, snapshot_id="t", code_commit="c", analysis_date="2026-10-11")
    registry.store_results(manifest, results, path)
    registry.store_results(manifest, results, path)       # second call is a no-op
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM sr_results").fetchone()[0] == 9
        for statement in ("UPDATE sr_results SET status='supported'", "DELETE FROM sr_results",
                          "DELETE FROM sr_protocols", "UPDATE sr_manifests SET manifest_json='{}'"):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement)
    finally:
        connection.close()


# -- Leak and admissibility checks ---------------------------------------------
def test_size_tercile_uses_only_capitalisation_known_before_day0(null_news):
    days = null_news.panel.calendar.sessions
    table = pd.DataFrame([{"ticker": "S000", "day0": days[200]}])
    before = prep.size_tercile(null_news, table).iloc[0]
    # Make the stock the largest company from day 0 onwards: must change nothing.
    future = null_news.size.copy()
    future.loc[(future["ticker"] == "S000") & (future["date"] >= days[200]), "market_cap"] = 1e12
    ctx = common.Context(panel=null_news.panel, size=future, availability=F.ALL_AVAILABLE)
    assert prep.size_tercile(ctx, table).iloc[0] == before == "small"


def test_sentiment_surprise_is_strictly_lagged_and_safe():
    calendar = F.news_context(news_tickers=1, events_per_ticker=1).panel.calendar
    days = calendar.sessions
    events = pd.DataFrame({
        "ticker": "A", "day0": [days[100 + 3 * i] for i in range(8)],
        "sentiment": [0.1, 0.3, -0.2, 0.4, 0.0, 0.2, 0.9, -0.5]})
    z = h3.sentiment_surprise(events, calendar)
    assert z.iloc[:5].isna().all()                    # fewer than five earlier events
    history = np.array([0.1, 0.3, -0.2, 0.4, 0.0])
    assert z.iloc[5] == pytest.approx((0.2 - history.mean()) / history.std(ddof=1))
    later = events.copy()
    later.loc[7, "sentiment"] = 5.0                   # a later event changes nothing earlier
    assert h3.sentiment_surprise(later, calendar).iloc[:7].equals(z.iloc[:7])
    flat = events.assign(sentiment=0.25)
    assert h3.sentiment_surprise(flat, calendar).isna().all()      # zero variance


def test_abnormal_coverage_counts_a_story_once_and_looks_backwards():
    calendar = F.news_context(news_tickers=1, events_per_ticker=1).panel.calendar
    days = calendar.sessions
    rows = [{"event_id": f"s{i}", "ticker": "A", "day0": days[100 + 5 * i], "sentiment": 0.0}
            for i in range(4)]
    rows += [{"event_id": "burst-a", "ticker": "A", "day0": days[125], "sentiment": 0.0},
             {"event_id": "burst-a", "ticker": "A", "day0": days[125], "sentiment": 0.0},   # syndicated copy
             {"event_id": "burst-b", "ticker": "A", "day0": days[125], "sentiment": 0.0}]
    daily = h4.abnormal_coverage(pd.DataFrame(rows), calendar).set_index("day0")
    assert daily.loc[days[125], "stories"] == 2
    assert daily.loc[days[125], "abnormal_coverage"] == pytest.approx(np.log(2 / (4 / 60)))
    assert daily["abnormal_coverage"].iloc[:3].isna().all()        # under three prior days


def test_h7_drops_updates_corrections_and_mixed_days(null_kap):
    events = null_kap.kap_events.copy()
    base, _ = h7.prepare(null_kap)
    first = events.iloc[0]
    extra = pd.DataFrame([
        {**first, "event_id": "upd", "is_update": True, "ticker": "S001", "day0": first["day0"]},
        {**first, "event_id": "mix", "category": "cat_down"},     # same ticker-day, other category
    ])
    ctx = common.Context(panel=null_kap.panel, kap_events=pd.concat([events, extra], ignore_index=True),
                         availability=null_kap.availability)
    changed, counts = h7.prepare(ctx)
    assert len(changed) == len(base) - 1                           # the mixed day is gone
    assert "upd" not in set(changed["event_id"])
    assert counts["not_update_or_correction"] == counts["timing_usable"] - 1


def test_h8_keeps_pure_purchase_days_only(null_kap):
    events, counts = h8.prepare(null_kap)
    assert counts["by_side"] == {"buy": 90, "sell": 25}
    assert len(events) == 90
    buy = null_kap.kap_events[null_kap.kap_events["insider_side"] == "buy"].iloc[0]
    clash = pd.DataFrame([{**buy, "event_id": "clash", "insider_side": "sell"}])
    ctx = common.Context(panel=null_kap.panel,
                         kap_events=pd.concat([null_kap.kap_events, clash], ignore_index=True),
                         availability=null_kap.availability)
    assert len(h8.prepare(ctx)[0]) == 89


def test_h5_news_classes_and_h6_censoring(null_limits):
    events = h5.limit_down_events(null_limits)
    labels = h5.classify_news(null_limits, events).value_counts().to_dict()
    assert set(labels) == {"no_news", "negative_news"} and sum(labels.values()) == 240
    table = h6.at_risk_sessions(null_limits)
    assert {"news_today", "streak_length", "ended"} <= set(table.columns)
    assert table["streak_length"].min() == 1
    # Removing the session after a streak day censors that day instead of
    # counting it as an ending.
    row = table[table["ended"] == 1].iloc[0]      # the next session is not a streak day
    frame = null_limits.panel.bars[row["ticker"]]
    following = null_limits.panel.calendar.offset(row["date"], 1)
    saved = frame.loc[following].copy()
    frame.loc[following, ["close", "raw_ret"]] = np.nan
    try:
        again = h6.at_risk_sessions(null_limits)
        assert len(again) == len(table) - 1
        assert again.attrs["censored"] == table.attrs["censored"] + 1
    finally:
        frame.loc[following] = saved


def test_sealed_window_blocks_limit_and_event_outcomes(null_limits, null_kap):
    sessions = null_limits.panel.calendar.sessions
    sealed = common.Context(panel=null_limits.panel, news_events=null_limits.news_events,
                            availability=F.ALL_AVAILABLE, sealed=sessions[0])
    assert h5.limit_down_events(sealed).empty
    assert h6.at_risk_sessions(sealed).empty
    sealed_kap = common.Context(panel=null_kap.panel, kap_events=null_kap.kap_events,
                                availability=null_kap.availability, sealed=sessions[0])
    assert h7.run(sealed_kap)["status"] == DATA_INSUFFICIENT
    assert h8.run(sealed_kap)["sufficiency"]["counts"]["studied"] == {"sealed_window": 90}


def test_h9_matching_is_one_to_one_and_windowed():
    english = pd.DataFrame([
        {"story_id": "e1", "subject": "A", "first_published_utc": "2023-01-02T08:00:00+00:00", "sentiment": 0.1},
        {"story_id": "e2", "subject": "A", "first_published_utc": "2023-01-02T09:00:00+00:00", "sentiment": 0.1},
        {"story_id": "e3", "subject": "B", "first_published_utc": "2023-01-02T08:00:00+00:00", "sentiment": 0.1}])
    turkish = pd.DataFrame([
        {"story_id": "t1", "subject": "A", "first_published_utc": "2023-01-02T10:00:00+00:00", "sentiment": 0.2},
        {"story_id": "t3", "subject": "B", "first_published_utc": "2023-01-09T08:00:00+00:00", "sentiment": 0.2}])
    matched = h9.match_streams(english, turkish)
    assert len(matched) == 1                                   # B is a week apart; A pairs once
    assert matched.iloc[0]["english_story"] == "e2" and matched.iloc[0]["lag_hours"] == 1.0
