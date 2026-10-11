"""Tests of the pre-run readiness audit of 2026-10-11.

Everything here is synthetic: generated returns, a calendar built from the
official 2023 closure list, and designs made up for the test. Nothing reads
the research database or a real return.
"""

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from readiness_audit import calibration, power
from stock_research import calendar as cal
from stock_research import pipeline, stats
from stock_research.hypotheses import common, h7, registry

ISTANBUL = ZoneInfo("Europe/Istanbul")

#: Equity-market closures on 2023 weekdays. Official holidays from Borsa
#: Istanbul's 2023 holiday tables (announcement E-18454353-100.04.02-13031 of
#: 2022-12-07); 9-14 February is the closure after the 6 February earthquakes,
#: and 8 February is the session whose trades were cancelled.
CLOSED_2023 = {
    "2023-02-08", "2023-02-09", "2023-02-10", "2023-02-13", "2023-02-14",
    "2023-04-21", "2023-05-01", "2023-05-19", "2023-06-28", "2023-06-29",
    "2023-06-30", "2023-08-30",
}
#: 25 December was a session; the benchmark's provider has no bar for it.
DATA_GAPS_2023 = {"2023-12-25"}
HALF_DAYS_2023 = {"2023-04-20", "2023-06-27"}        # arife: closing session 12:30-12:40


def _calendar_2023() -> cal.SessionCalendar:
    day, sessions = date(2023, 1, 2), []
    while day <= date(2023, 12, 29):
        if day.weekday() < 5 and day.isoformat() not in CLOSED_2023 | DATA_GAPS_2023:
            sessions.append(day.isoformat())
        day += timedelta(days=1)
    return cal.SessionCalendar(sessions)


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp).replace(tzinfo=ISTANBUL)


# -- Inference: the implementation is right, the reference distribution is not --
def _random_design(seed: int = 3, n: int = 600):
    rng = np.random.default_rng(seed)
    dates, issuers = rng.integers(0, 40, n), rng.integers(0, 70, n)
    category = rng.integers(0, 4, n)
    X = np.column_stack([np.ones(n)] + [(category == c).astype(float) for c in (1, 2, 3)])
    y = rng.normal(size=n) + 0.5 * rng.normal(size=40)[dates] + 0.5 * rng.normal(size=70)[issuers]
    return X, y, dates, issuers


def test_two_way_covariance_matches_a_brute_force_reference():
    X, y, dates, issuers = _random_design()
    fit = stats.ols_cluster(X, y, [dates, issuers])
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    resid = y - X @ (bread @ X.T @ y)

    def one_way(labels) -> np.ndarray:
        meat = np.zeros((k, k))
        groups = sorted(set(labels))
        for g in groups:                              # the textbook sum over clusters
            score = (X[labels == g] * resid[labels == g][:, None]).sum(axis=0)
            meat += np.outer(score, score)
        scale = len(groups) / (len(groups) - 1) * (n - 1) / (n - k)
        return scale * bread @ meat @ bread

    pair = np.array([f"{a}|{b}" for a, b in zip(dates, issuers)])
    expected = one_way(dates) + one_way(issuers) - one_way(pair)
    assert np.allclose(np.array(fit["cov"]), expected, rtol=1e-10, atol=1e-14)
    assert fit["dof"] == min(len(set(dates)), len(set(issuers))) - 1


def test_two_way_covariance_and_wald_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    X, y, dates, issuers = _random_design()
    fit = stats.ols_cluster(X, y, [dates, issuers])
    reference = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={
        "groups": np.column_stack([dates, issuers]), "use_correction": True,
        "df_correction": True})
    assert np.allclose(fit["coef"], reference.params, atol=1e-12)
    assert np.allclose(np.array(fit["cov"]), reference.cov_params(), rtol=1e-10)
    R = np.zeros((3, 4))
    R[:, 1:] = np.eye(3)
    ours, theirs = stats.wald(fit, R), reference.wald_test(R, use_f=True, scalar=True)
    assert ours["f"] == pytest.approx(float(theirs.statistic), rel=1e-10)
    assert ours["dof"] == theirs.df_denom
    assert ours["p"] == pytest.approx(float(theirs.pvalue), rel=1e-9)


def _design(sizes, dates_per_category, seed=0) -> power.Design:
    """Independent events: a different issuer each, categories of given sizes
    spread over the given number of dates."""

    rng = np.random.default_rng(seed)
    category, position = [], []
    for name, (size, dates) in enumerate(zip(sizes, dates_per_category)):
        category += [f"c{name}"] * size
        position += list(140 + rng.integers(0, dates, size) * 3 + name)
    n = len(category)
    return power.Design(issuer=np.arange(n), position=np.array(position),
                        date=np.array(position), category=np.array(category),
                        kept=[f"c{i}" for i in range(len(sizes))],
                        sigma=np.full(n, 0.03), sessions=440, anchors=0)


def test_audit_joint_test_is_the_registered_h7_test():
    design = _design([200, 120, 60], [40, 30, 20])
    rng = np.random.default_rng(1)
    y = power.simulate_cars(design, rng, common_share=0.05, ar1=0.08)
    frame = pd.DataFrame({"category": design.category, "y": y,
                          "day0": design.date.astype(str), "ticker": design.issuer.astype(str)})
    registered = h7._joint(frame, design.kept, "y", common.TWO_WAY)
    audit = power.JointTest(design)(y)
    assert audit["p"] == pytest.approx(registered["test"]["p"], rel=1e-9)
    assert audit["spread"] == pytest.approx(registered["spread"], rel=1e-12)
    in_calibration = calibration.Procedures(design)(y)["registered_two_way"]
    assert in_calibration == pytest.approx(registered["test"]["p"], rel=1e-9)


def test_simulated_noise_has_no_effect_and_the_stated_volatility():
    design = _design([400, 400], [60, 60])
    rng = np.random.default_rng(5)
    draws = np.concatenate([power.simulate_cars(design, rng, common_share=0.0, ar1=0.0)
                            for _ in range(20)])
    # Six sessions at 3% a day, plus the error in the estimated intercept.
    expected_sd = 0.03 * np.sqrt(6 * (1 + 6 / 120))
    assert abs(draws.mean()) < 4 * expected_sd / np.sqrt(len(draws))
    assert draws.std() == pytest.approx(expected_sd, rel=0.05)


def test_planted_spread_is_exactly_the_stated_effect():
    design = _design([200, 120, 60], [40, 30, 20])
    test = power.JointTest(design)
    for scenario in ("linear", "one:c2"):
        shift = power.shifts(test, scenario, 0.02)
        means = [shift[m].mean() for m in test.masks.values()]
        assert max(means) - min(means) == pytest.approx(0.02)


def _null_rates(design, runs, seed):
    rng = np.random.default_rng(seed)
    procedures = calibration.Procedures(design)
    hits = {"registered_two_way": 0, "date_only_min_category_dof": 0}
    for _ in range(runs):
        y = power.simulate_cars(design, rng, common_share=0.0, ar1=0.0)
        out = procedures(y)
        for name in hits:
            hits[name] += out[name] < 0.05
    return {name: count / runs for name, count in hits.items()}


def test_registered_joint_test_overrejects_when_a_category_sits_on_few_dates():
    """The finding of the readiness audit, in miniature: independent events,
    no effect, and still the registered F(q, G-1) test rejects too often when
    two of the categories are small. Degrees of freedom taken from the
    smallest category bring it back. Balanced categories are fine."""

    unbalanced = _design([450, 320, 190, 110, 40, 35], [80, 75, 70, 50, 22, 18])
    rates = _null_rates(unbalanced, 600, seed=11)
    # 5% nominal; three standard errors of a 600-run estimate is 2.7 points.
    assert rates["registered_two_way"] > 0.05 + 0.027
    assert rates["date_only_min_category_dof"] < 0.05 + 0.027

    balanced = _design([300, 300, 300], [80, 80, 80])
    assert _null_rates(balanced, 600, seed=12)["registered_two_way"] < 0.05 + 0.027


def test_projection_keeps_only_categories_that_meet_the_registered_size_rule():
    rng = np.random.default_rng(2)
    rows = []
    for anchor in range(0, 80, 6):
        for i in range(25):
            rows.append({"anchor": anchor, "ticker": f"T{rng.integers(0, 60)}",
                         "position": 150 + anchor * 3 + int(rng.integers(0, 2)),
                         "category": str(rng.choice(["a", "b", "c", "rare"],
                                                    p=[0.5, 0.3, 0.19, 0.01]))})
    structure = pd.DataFrame(rows).drop_duplicates(["ticker", "position"])
    design = power.project(structure, {"residual_sd_by_ticker": {"T1": 0.03}},
                           anchors_total=80, frame_anchors=80, sessions=440, rng=rng)
    counts = pd.Series(design.category).value_counts()
    assert set(design.kept) == set(counts.index)
    for category in design.kept:
        mask = design.category == category
        assert mask.sum() >= 30 and len(np.unique(design.date[mask])) >= 15
    # One row per issuer and day 0, as the registered study collapses them.
    assert not pd.DataFrame({"i": design.issuer, "p": design.position}).duplicated().any()


def test_anchors_that_yield_nothing_stay_in_the_projection_pool():
    """An anchor whose disclosures were all filed during the session gives the
    primary sample no event. Projecting the unseen anchors from the productive
    ones only would promise a larger sample than the frame can deliver."""

    rows = [{"anchor": anchor, "ticker": f"T{anchor}_{i}", "position": 150 + anchor * 3,
             "category": "a"} for anchor in (0, 20, 40, 60) for i in range(40)]
    structure = pd.DataFrame(rows)
    calibration = {"residual_sd_by_ticker": {"T0_0": 0.03}}

    def events(ingested, seed):
        design = power.project(structure, calibration, anchors_total=80, frame_anchors=80,
                               sessions=440, rng=np.random.default_rng(seed),
                               recurrence="fresh", ingested=ingested)
        return len(design.category)

    productive_only = np.mean([events(None, seed) for seed in range(20)])
    with_empty = np.mean([events([0, 10, 20, 30, 40, 50, 60, 70], seed) for seed in range(20)])
    assert productive_only == 80 * 40                     # every unseen anchor yields 40
    # Half the ingested anchors are empty, so about half the unseen ones are.
    assert 0.4 * productive_only < with_empty < 0.65 * productive_only


# -- Timing: the boundaries of a 2023 session ------------------------------------
@pytest.mark.parametrize("stamp, day0, bucket", [
    ("2023-03-15 09:30:00", "2023-03-15", cal.BUCKET_PRE_OPEN),       # before the opening auction
    ("2023-03-15 09:59:59", "2023-03-15", cal.BUCKET_PRE_OPEN),
    ("2023-03-15 10:00:00", "2023-03-15", cal.BUCKET_DURING),
    ("2023-03-15 14:00:00", "2023-03-15", cal.BUCKET_DURING),
    ("2023-03-15 18:03:00", "2023-03-15", cal.BUCKET_DURING),         # closing auction
    ("2023-03-15 18:10:00", "2023-03-15", cal.BUCKET_DURING),
    ("2023-03-15 18:10:01", "2023-03-16", cal.BUCKET_POST_CLOSE),
    ("2023-03-15 23:59:59", "2023-03-16", cal.BUCKET_POST_CLOSE),
    ("2023-03-17 19:00:00", "2023-03-20", cal.BUCKET_POST_CLOSE),     # Friday evening
    ("2023-03-18 11:00:00", "2023-03-20", cal.BUCKET_NON_SESSION),    # Saturday
    ("2023-05-01 11:00:00", "2023-05-02", cal.BUCKET_NON_SESSION),    # official holiday
    ("2023-06-29 11:00:00", "2023-07-03", cal.BUCKET_NON_SESSION),    # Kurban Bayrami
    ("2023-02-08 11:00:00", "2023-02-15", cal.BUCKET_NON_SESSION),    # cancelled session
    ("2023-02-10 11:00:00", "2023-02-15", cal.BUCKET_NON_SESSION),    # earthquake closure
])
def test_day0_at_the_session_boundaries(stamp, day0, bucket):
    result = _calendar_2023().actionable(_at(stamp))
    assert (result.day0, result.bucket, result.timing_ambiguous) == (day0, bucket, False)


def test_a_disclosure_never_gets_a_day0_before_it_was_published():
    calendar = _calendar_2023()
    moment = _at("2023-01-02 00:00:00")
    while moment < _at("2023-12-22 00:00:00"):
        result = calendar.actionable(moment)
        assert result.day0 >= moment.date().isoformat()
        if result.bucket in (cal.BUCKET_POST_CLOSE, cal.BUCKET_NON_SESSION) \
                and calendar.is_session(moment.date().isoformat()):
            assert result.day0 > moment.date().isoformat()
        moment += timedelta(hours=7, minutes=13)


@pytest.mark.parametrize("half_day, next_session", [("2023-04-20", "2023-04-24"),
                                                    ("2023-06-27", "2023-07-03")])
def test_half_day_afternoon_is_never_given_the_same_session(half_day, next_session):
    calendar = _calendar_2023()
    morning = calendar.actionable(_at(f"{half_day} 11:00:00"))
    assert (morning.day0, morning.bucket) == (half_day, cal.BUCKET_DURING)
    # After the early close the filing waits for the next session. Under v2
    # the early close was inferred and the row flagged ambiguous; v3 records
    # the 2023 half days, so the timing is known.
    afternoon = calendar.actionable(_at(f"{half_day} 14:00:00"))
    assert (afternoon.day0, afternoon.bucket) == (next_session, cal.BUCKET_POST_CLOSE)
    assert not afternoon.timing_ambiguous
    evening = calendar.actionable(_at(f"{half_day} 18:30:00"))
    assert (evening.day0, evening.timing_ambiguous) == (next_session, False)


def test_half_day_closing_session_runs_to_12_40():
    """On a half day continuous trading ended at 12:30 and the closing session
    at 12:40. v2 cut at 12:30 (readiness finding C-4); v3 records 12:40, the
    counterpart of 18:10 on a full day."""

    calendar = _calendar_2023()
    inside = calendar.actionable(_at("2023-04-20 12:35:00"))
    assert (inside.day0, inside.bucket, inside.timing_ambiguous) == (
        "2023-04-20", cal.BUCKET_DURING, False)
    after = calendar.actionable(_at("2023-04-20 12:40:01"))
    assert (after.day0, after.bucket, after.timing_ambiguous) == (
        "2023-04-24", cal.BUCKET_POST_CLOSE, False)


@pytest.mark.parametrize("full_day", ["2023-02-07", "2023-04-28", "2023-05-18",
                                      "2023-08-29", "2023-12-22", "2023-12-29"])
def test_full_sessions_before_a_closure_are_no_longer_taken_for_half_days(full_day):
    """Readiness finding C-2: v2 inferred a half day from a closure on the next
    weekday and delayed afternoon filings on six full sessions of 2023. With
    the schedule recorded, an afternoon filing on those days is what it was:
    published during the session."""

    calendar = _calendar_2023()
    assert full_day not in HALF_DAYS_2023
    result = calendar.actionable(_at(f"{full_day} 15:00:00"))
    assert (result.day0, result.bucket, result.timing_ambiguous) == (
        full_day, cal.BUCKET_DURING, False)


def test_timestamps_are_read_in_istanbul_time():
    calendar = _calendar_2023()
    # 15:30 UTC is 18:30 in Istanbul: after the close.
    utc = datetime(2023, 3, 15, 15, 30, tzinfo=ZoneInfo("UTC"))
    assert calendar.actionable(utc).day0 == "2023-03-16"
    # 06:30 UTC is 09:30 in Istanbul: before the open of the same session.
    early = datetime(2023, 3, 15, 6, 30, tzinfo=ZoneInfo("UTC"))
    assert calendar.actionable(early).bucket == cal.BUCKET_PRE_OPEN
    # A KAP stamp has no zone and is local time.
    parsed = cal.parse_kap_time("15.03.2023 18:23:08")
    assert parsed.utcoffset() == timedelta(hours=3)
    assert calendar.actionable(parsed).day0 == "2023-03-16"


# -- Protocol integrity -----------------------------------------------------------
REGISTERED = {
    "stock-research-protocol-v2":
        "09e9382e77ec44bc06aaaf5c4bde0d64e06a6584d2aeb4b65ed35da0b2e7d99a",
}


def test_the_code_still_is_the_registered_protocol():
    """A change to any fingerprinted file must come with a new protocol
    version; under an old version name the hash has to be the registered one."""

    if registry.PROTOCOL_VERSION in REGISTERED:
        assert registry.protocol_hash() == REGISTERED[registry.PROTOCOL_VERSION]
    assert registry.LINEAGE[0]["protocol_hash"].startswith("92366aa0a6342e9c")
    assert registry.SUPERSEDES["protocol_hash"] == REGISTERED["stock-research-protocol-v2"]


def test_the_audit_code_is_outside_the_fingerprint():
    files = registry.code_fingerprint()
    assert not any("readiness" in name for name in files)
    assert "stats.py" in files and "eventstudy.py" in files and "data/kap.py" in files


def test_synthetic_results_are_identical_from_one_run_to_the_next():
    first = registry.canonical(pipeline.synthetic_demo())
    second = registry.canonical(pipeline.synthetic_demo())
    assert first == second
    assert all(not r.get("claim_allowed") for r in json.loads(first).values())
