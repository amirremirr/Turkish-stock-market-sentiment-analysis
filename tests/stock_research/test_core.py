"""Calendar, prices, the event-study engine and inference, on synthetic data."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stock_research import fixtures, stats
from stock_research.calendar import (
    BUCKET_DURING, BUCKET_NON_SESSION, BUCKET_POST_CLOSE, BUCKET_PRE_OPEN,
    BUCKET_UNKNOWN, ISTANBUL, SessionCalendar, parse_kap_time,
)
from stock_research.data import prices
from stock_research.eventstudy import (
    STATUS_HORIZON, STATUS_NO_DAY0, STATUS_NO_ESTIMATION, STATUS_OK,
    STATUS_SEALED, collapse_same_day, event_table, flag_overlaps, study_event,
)


def _at(text):
    return datetime.fromisoformat(text).replace(tzinfo=ISTANBUL)


# -- Calendar -----------------------------------------------------------------
@pytest.fixture
def calendar_2023():
    # Thursday 20 April 2023 precedes a closure (21 April is absent).
    return SessionCalendar([
        "2023-04-17", "2023-04-18", "2023-04-19", "2023-04-20",
        "2023-04-24", "2023-04-25", "2023-04-26",
    ])


@pytest.mark.parametrize("stamp,day0,bucket", [
    ("2023-04-18T09:59:59", "2023-04-18", BUCKET_PRE_OPEN),
    ("2023-04-18T10:00:00", "2023-04-18", BUCKET_DURING),
    ("2023-04-18T18:10:00", "2023-04-18", BUCKET_DURING),
    ("2023-04-18T18:10:01", "2023-04-19", BUCKET_POST_CLOSE),
    ("2023-04-22T12:00:00", "2023-04-24", BUCKET_NON_SESSION),   # Saturday
    ("2023-04-21T11:00:00", "2023-04-24", BUCKET_NON_SESSION),   # weekday holiday
])
def test_day0_is_the_first_session_able_to_act(calendar_2023, stamp, day0, bucket):
    result = calendar_2023.actionable(_at(stamp))
    assert (result.day0, result.bucket) == (day0, bucket)
    assert not result.timing_ambiguous


def test_afternoon_before_a_closure_is_delayed_and_flagged(calendar_2023):
    # 14:00 on the eve of a holiday: an early close is possible but unrecorded.
    eve = calendar_2023.actionable(_at("2023-04-20T14:00:00"))
    assert (eve.day0, eve.bucket, eve.timing_ambiguous) == (
        "2023-04-24", BUCKET_POST_CLOSE, True)
    # Morning of the same day is unambiguous, and so is after the regular close.
    assert calendar_2023.actionable(_at("2023-04-20T11:00:00")).bucket == BUCKET_DURING
    late = calendar_2023.actionable(_at("2023-04-20T18:30:00"))
    assert (late.bucket, late.timing_ambiguous) == (BUCKET_POST_CLOSE, False)


def test_unknown_time_and_out_of_range_dates_are_not_guessed(calendar_2023):
    no_time = calendar_2023.actionable(None, published_date="2023-04-18")
    assert (no_time.day0, no_time.bucket) == ("2023-04-19", BUCKET_UNKNOWN)
    outside = calendar_2023.actionable(_at("2024-01-05T11:00:00"))
    assert outside.day0 is None and outside.timing_ambiguous


def test_utc_input_is_converted_to_istanbul(calendar_2023):
    # 15:30 UTC is 18:30 in Istanbul: after the close.
    utc = datetime.fromisoformat("2023-04-18T15:30:00+00:00")
    assert calendar_2023.actionable(utc).bucket == BUCKET_POST_CLOSE


def test_official_half_day_is_used_where_known():
    calendar = SessionCalendar(["2026-10-27", "2026-10-28", "2026-10-30"])
    # 2026-10-28 closes at 13:00 officially (config.BIST_HALF_DAYS).
    result = calendar.actionable(_at("2026-10-28T14:00:00"))
    assert (result.day0, result.bucket, result.timing_ambiguous) == (
        "2026-10-30", BUCKET_POST_CLOSE, False)


def test_kap_time_parses_seconds_and_rejects_junk():
    assert parse_kap_time("29.12.2023 18:23:08") == _at("2023-12-29T18:23:08")
    assert parse_kap_time("yesterday") is None and parse_kap_time(None) is None


# -- Prices -------------------------------------------------------------------
def _flat(days, close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "adj_close": close, "volume": 1000.0, "dividend": 0.0, "split_ratio": 0.0,
    }, index=days)


def test_a_missing_session_leaves_two_returns_missing_not_one_long_return():
    days = fixtures.weekdays("2023-01-02", 6)
    market = _flat(days, [100, 101, 102, 103, 104, 105])
    stock = _flat(days, [10, 11, 12, 13, 14, 15]).drop(index=days[2])
    panel = prices.build_panel({"MKT": market, "X": stock}, "MKT", origin="synthetic",
                               snapshot_id="t")
    ret = panel.frame("X")["ret"]
    assert np.isnan(ret[days[2]]) and np.isnan(ret[days[3]])     # gap, then spanning
    assert ret[days[4]] == pytest.approx(14 / 13 - 1)


def test_unexplained_jump_is_flagged_and_removed_but_an_explained_one_is_kept():
    days = fixtures.weekdays("2023-01-02", 5)
    market = _flat(days, [100, 100, 100, 100, 100])
    jumpy = _flat(days, [10, 10, 5, 5, 5])
    explained = jumpy.copy()
    explained.loc[days[2], "split_ratio"] = 2.0
    panel = prices.build_panel({"MKT": market, "J": jumpy, "E": explained}, "MKT",
                               origin="synthetic", snapshot_id="t")
    assert np.isnan(panel.frame("J")["ret"][days[2]])
    assert {"date": days[2], "issue": "unexplained_jump"} in panel.issues["J"]
    assert panel.frame("E")["ret"][days[2]] == pytest.approx(-0.5)
    # Same-day quantities are withheld on the action date.
    assert np.isnan(panel.frame("E")["gap"][days[2]])


def test_structural_defects_are_reported():
    days = fixtures.weekdays("2023-01-02", 3)
    frame = _flat(days, [10, 10, 10])
    frame.loc[days[1], ["high", "low"]] = [9.0, 11.0]
    frame.loc[days[2], "volume"] = 0.0
    found = {(i["date"], i["issue"]) for i in prices.validate_bars(frame)}
    assert (days[1], "high_below_low") in found
    assert (days[2], "zero_or_missing_volume") in found


def test_market_cap_is_declared_unavailable_not_approximated():
    availability = prices.market_cap_availability()
    assert not availability.available and "look-ahead" in availability.reason


# -- Event study --------------------------------------------------------------
@pytest.fixture(scope="module")
def panel():
    return fixtures.synthetic_panel(tickers=30, sessions=420, seed=5)


def test_estimation_window_may_not_touch_the_event_windows(panel):
    with pytest.raises(ValueError, match="overlaps"):
        study_event(panel, "S000", panel.calendar.sessions[300], estimation=(-60, -3))


def test_future_returns_cannot_change_the_expected_return_model():
    sessions = fixtures.weekdays("2022-01-03", 420)
    day0 = sessions[300]
    base = fixtures.synthetic_panel(tickers=3, sessions=420, seed=9)
    shocked = fixtures.synthetic_panel(
        tickers=3, sessions=420, seed=9,
        effects={("S000", day0): {k: 0.03 for k in range(-5, 21)}})
    a, b = study_event(base, "S000", day0), study_event(shocked, "S000", day0)
    assert (a["alpha"], a["beta"]) == (b["alpha"], b["beta"])
    assert b["car_p0_p5"] - a["car_p0_p5"] == pytest.approx(0.18, abs=1e-9)


def test_an_injected_abnormal_return_is_recovered(panel):
    day0 = panel.calendar.sessions[300]
    effects = {("S001", day0): {0: 0.04, 2: 0.01, 3: 0.01}}
    shocked = fixtures.synthetic_panel(tickers=30, sessions=420, seed=5, effects=effects)
    before, after = study_event(panel, "S001", day0), study_event(shocked, "S001", day0)
    assert after["car_p0_p0"] - before["car_p0_p0"] == pytest.approx(0.04, abs=1e-9)
    assert after["car_p2_p5"] - before["car_p2_p5"] == pytest.approx(0.02, abs=1e-9)
    assert after["car_m5_m1"] == pytest.approx(before["car_m5_m1"], abs=1e-12)


def test_statuses_name_why_an_event_has_no_result(panel):
    sessions = panel.calendar.sessions
    assert study_event(panel, "S000", "2019-01-01")["status"] == STATUS_NO_DAY0
    assert study_event(panel, "S000", sessions[-3])["status"] == STATUS_HORIZON
    early = study_event(panel, "S000", sessions[30])
    assert early["status"] == STATUS_NO_ESTIMATION
    assert early["car_p0_p5"] is None and early["mar_p0_p5"] is not None
    sealed = study_event(panel, "S000", sessions[300], sealed=sessions[310])
    assert sealed["status"] == STATUS_SEALED and "car_p0_p5" not in sealed


def test_a_window_with_a_missing_return_has_no_car():
    panel = fixtures.synthetic_panel(tickers=2, sessions=420, seed=3)
    day0 = panel.calendar.sessions[300]
    panel.bars["S000"].loc[panel.calendar.sessions[303], "ret"] = np.nan
    row = study_event(panel, "S000", day0)
    assert row["car_p2_p5"] is None and row["car_p0_p1"] is not None


def test_same_day_events_collapse_to_one_observation():
    events = [
        {"event_id": 1, "ticker": "A", "day0": "2023-01-02", "category": "buyback"},
        {"event_id": 2, "ticker": "A", "day0": "2023-01-02", "category": "buyback"},
        {"event_id": 3, "ticker": "A", "day0": "2023-01-02", "category": "contract"},
        {"event_id": 4, "ticker": "B", "day0": "2023-01-02", "category": "contract"},
    ]
    merged = {r["ticker"]: r for r in collapse_same_day(events)}
    assert merged["A"]["n_events"] == 3 and merged["A"]["category"] == "mixed"
    assert merged["B"]["n_events"] == 1 and merged["B"]["category"] == "contract"


def test_overlapping_windows_are_flagged(panel):
    sessions = panel.calendar.sessions
    table = pd.DataFrame([
        {"ticker": "A", "day0": sessions[200]}, {"ticker": "A", "day0": sessions[203]},
        {"ticker": "A", "day0": sessions[260]}, {"ticker": "B", "day0": sessions[201]},
    ])
    assert flag_overlaps(table, 5, panel.calendar).tolist() == [False, True, False, False]


# -- Inference ----------------------------------------------------------------
def test_holm_uses_the_registered_family_size():
    assert stats.holm([0.01, None, 0.04]) == [0.02, None, 0.04]
    # Nine registered hypotheses, two runnable: the family is still nine.
    assert stats.holm([0.01, None, 0.04], family_size=9) == pytest.approx([0.09, None, 0.32])
    with pytest.raises(ValueError):
        stats.holm([0.01, 0.02], family_size=1)


def test_benjamini_hochberg_matches_a_worked_example():
    adjusted = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert adjusted == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20], rel=1e-5)


def test_no_p_value_below_the_cluster_minimum():
    result = stats.cluster_mean([0.01] * 40, [[i % 4 for i in range(40)]])
    assert result["status"] == "too_few_clusters" and result["p"] is None
    assert result["mean"] == pytest.approx(0.01)


def test_null_events_are_rejected_at_about_the_nominal_rate():
    """No effect anywhere, several events per day, sector shocks shared within
    a day. Date-clustered tests should reject about 5% of the time."""

    rejections, trials = 0, 200
    for trial in range(trials):
        panel = fixtures.synthetic_panel(tickers=24, sessions=260, seed=1000 + trial,
                                         sector_sd=0.01)
        events = fixtures.random_events(panel, 120, seed=trial, first=140, per_day=4)
        table = event_table(panel, events, windows=((0, 1),))
        ok = table[table["status"] == STATUS_OK]
        result = stats.cluster_mean(ok["car_p0_p1"], [ok["day0"]])
        rejections += result["p"] < 0.05
    assert rejections / trials < 0.10


def test_ignoring_clusters_overstates_confidence():
    """One shared shock per day and many events on it: row-level standard
    errors must come out smaller than date-clustered ones."""

    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(30), 20)
    values = rng.normal(0, 1, 30)[days] + rng.normal(0, 0.2, 600)
    clustered = stats.cluster_mean(values, [days])
    by_row = stats.cluster_mean(values, [np.arange(600)])
    assert clustered["se"] > 3 * by_row["se"]


def test_two_way_clustering_runs_and_is_reported_per_dimension():
    rng = np.random.default_rng(1)
    days, firms = rng.integers(0, 25, 500), rng.integers(0, 20, 500)
    fit = stats.ols_cluster(np.column_stack([np.ones(500), rng.normal(size=500)]),
                            rng.normal(size=500), [days, firms], names=["const", "x"])
    assert fit["clusters"] == [25, 20] and fit["status"] in {"ok", "variance_not_positive"}


def test_cluster_bootstrap_is_reproducible():
    values, clusters = np.arange(40.0), np.arange(40) // 4
    first = stats.cluster_bootstrap(values, clusters, resamples=500, seed=7)
    assert first == stats.cluster_bootstrap(values, clusters, resamples=500, seed=7)
    assert first["lower"] < first["mean"] < first["upper"]
