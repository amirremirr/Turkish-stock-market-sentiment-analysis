"""Tests of the candidate protocol v3.

Synthetic data throughout. Each block is one of the corrections the readiness
audit called for; where the audit measured a number, that number is the
benchmark here.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from readiness_audit import power
from stock_research import calendar as cal
from stock_research import config, events, fixtures as F, pipeline, stats
from stock_research.data import prices, secondary
from stock_research.hypotheses import common, h7, registry

from tests.stock_research.test_pipeline import _database

ISTANBUL = ZoneInfo("Europe/Istanbul")


# -- 1. Inference ----------------------------------------------------------------
def _clustered(seed=1, n=300):
    rng = np.random.default_rng(seed)
    dates, issuers, group = rng.integers(0, 25, n), rng.integers(0, 40, n), rng.integers(0, 3, n)
    X = np.column_stack([np.ones(n), group == 1, group == 2]).astype(float)
    y = rng.normal(size=n) + 0.5 * rng.normal(size=25)[dates] + 0.3 * (group == 2)
    return X, y, np.array([[0, 1, 0], [0, 0, 1.0]]), dates, issuers


@pytest.mark.parametrize("two_way", [False, True])
def test_wild_bootstrap_equals_refitting_every_bootstrap_sample(two_way):
    """The fast implementation works from cluster sums. Here each bootstrap
    sample is built in full and the registered estimator is run on it."""

    X, y, R, dates, issuers = _clustered()
    other = issuers if two_way else None
    fast = stats.wild_cluster_wald(X, y, R, dates, other_clusters=other, draws=199, seed=5)

    bread, _, restricted = stats._restricted(X, y, R)
    resid = y - X @ restricted
    codes = stats._codes(dates)
    rng = np.random.default_rng(5)
    exceed, done = 0, 0
    while done < 199:
        size = min(64, 199 - done)                         # the implementation's chunking
        weights = rng.integers(0, 2, size=(size, codes.max() + 1)) * 2.0 - 1.0
        for row in weights:
            sample = X @ restricted + resid * row[codes]
            fit = stats.ols_cluster(X, sample, [dates] + ([issuers] if two_way else []))
            test = stats.wald(fit, R)
            exceed += test.get("f") is None or test["f"] >= fast["f"]
        done += size
    assert fast["p"] == pytest.approx((1 + exceed) / 200)
    assert fast["statistic"] == ("two_way" if two_way else "one_way")


def test_wild_bootstrap_is_reproducible_and_seeded():
    X, y, R, dates, issuers = _clustered()
    first = stats.wild_cluster_wald(X, y, R, dates, other_clusters=issuers, draws=499, seed=7)
    again = stats.wild_cluster_wald(X, y, R, dates, other_clusters=issuers, draws=499, seed=7)
    other = stats.wild_cluster_wald(X, y, R, dates, other_clusters=issuers, draws=499, seed=8)
    assert first == again and first["p"] != other["p"]
    assert 1 / 500 <= first["p"] <= 1


def test_cr2_satterthwaite_reduces_to_the_t_test_on_cluster_means():
    """For a mean with equal clusters the degrees of freedom are G - 1 and the
    test is the ordinary t-test on the cluster means."""

    rng = np.random.default_rng(2)
    groups, size = 20, 5
    y = rng.normal(size=groups * size)
    clusters = np.repeat(np.arange(groups), size)
    result = stats.cr2_wald(np.ones((groups * size, 1)), y, np.array([[1.0]]), clusters)
    assert result["eta"] == pytest.approx(groups - 1)
    assert result["p"] == pytest.approx(sps.ttest_1samp(y.reshape(groups, size).mean(axis=1), 0).pvalue)


def test_cr2_degrees_of_freedom_fall_when_a_class_sits_on_few_clusters():
    rng = np.random.default_rng(3)
    dates = np.concatenate([rng.integers(0, 80, 600), rng.integers(0, 12, 30)])
    group = np.concatenate([rng.integers(0, 2, 600), np.full(30, 2)])
    X = np.column_stack([np.ones(630), group == 1, group == 2]).astype(float)
    y = rng.normal(size=630)
    R = np.array([[0, 1, 0], [0, 0, 1.0]])
    test = stats.cr2_wald(X, y, R, dates)
    # 80 date clusters, but the small class's mean rests on a dozen of them.
    assert test["eta"] < 30 and test["status"] == "ok"


def _unbalanced(seed=0):
    """Independent events, a reference group and four classes, two of them at
    the registered minimum."""

    rng = np.random.default_rng(seed)
    category, position = [], []
    for k, (size, dates) in enumerate([(200, 80), (320, 75), (190, 70), (30, 20), (32, 20)]):
        category += [h7.REFERENCE if k == 0 else f"class{k}"] * size
        position += list(140 + rng.integers(0, dates, size) * 3 + k % 3)
    n = len(category)
    return power.Design(issuer=np.arange(n), position=np.array(position),
                        date=np.array(position), category=np.array(category),
                        kept=[h7.REFERENCE, "class1", "class2", "class3", "class4"],
                        sigma=np.full(n, 0.03), sessions=440, anchors=0)


def test_the_bootstrap_holds_its_level_where_the_f_test_does_not():
    """The readiness audit's finding and its remedy, in miniature. No effect in
    the data, classes at the registered minimum: the v2 F reference rejects too
    often, the wild bootstrap of the same statistic does not."""

    design = _unbalanced()
    tested = design.kept[1:]
    X = np.column_stack([np.ones(len(design.category))]
                        + [(design.category == g).astype(float) for g in tested])
    R = np.zeros((4, 5))
    R[:, 1:] = np.eye(4)
    rng = np.random.default_rng(11)
    runs = 400
    hits = {"f": 0, "wild": 0}
    for _ in range(runs):
        y = power.simulate_cars(design, rng, common_share=0.0, ar1=0.0)
        fit = stats.ols_cluster(X, y, [design.date, design.issuer])
        hits["f"] += stats.wald(fit, R)["p"] < 0.05
        hits["wild"] += stats.wild_cluster_wald(
            X, y, R, design.date, other_clusters=design.issuer, draws=199,
            seed=int(rng.integers(1 << 30)))["p"] < 0.05
    three_standard_errors = 3 * np.sqrt(0.05 * 0.95 / runs)
    assert hits["wild"] / runs < 0.05 + three_standard_errors
    assert hits["f"] / runs > hits["wild"] / runs


# -- 2. Two price sources ----------------------------------------------------------
def _two_sources():
    days = F.weekdays("2023-01-02", 60)
    panel = F.make_panel(days, ["AAA", "BBB", "CCC"], seed=4)
    frames = {t: f[prices.BAR_COLUMNS].dropna(subset=["close"]) for t, f in panel.bars.items()}
    second = {t: pd.DataFrame({"close_raw": f["close"], "close_adjusted": f["adj_close"]})
              for t, f in frames.items()}
    return days, frames, second, panel.benchmark


def test_a_return_the_two_sources_disagree_on_is_missing():
    days, frames, second, benchmark = _two_sources()
    # The second source's close for AAA on day 20 is 5% higher: two returns disagree.
    second["AAA"].loc[days[20], "close_adjusted"] *= 1.05
    # It has no bar for BBB on day 30: two returns cannot be checked.
    second["BBB"] = second["BBB"].drop(index=days[30])
    # And no series at all for CCC.
    del second["CCC"]
    plain = prices.build_panel(frames, benchmark, origin=prices.ORIGIN_SYNTHETIC, snapshot_id="t")
    checked = prices.build_panel(frames, benchmark, origin=prices.ORIGIN_SYNTHETIC,
                                 snapshot_id="t", secondary=second)

    a, b, c = (checked.frame(t)["ret"] for t in ("AAA", "BBB", "CCC"))
    assert a[[days[20], days[21]]].isna().all() and a.drop([days[0], days[20], days[21]]).notna().all()
    assert b[[days[30], days[31]]].isna().all() and b.notna().sum() == len(days) - 3
    assert c.isna().all()                                   # nothing unverified is kept
    assert plain.frame("AAA")["ret"].notna().sum() == len(days) - 1 and not plain.reconciliation

    issues = {t: {(i["date"], i["issue"]) for i in found} for t, found in checked.issues.items()}
    assert (days[20], prices.ISSUE_DISAGREEMENT) in issues["AAA"]
    assert (days[30], prices.ISSUE_UNVERIFIED) in issues["BBB"]
    summary = checked.reconciliation
    assert summary["returns_set_missing_for_disagreement"] == 2
    assert summary["returns_set_missing_for_no_second_source"] == 2 + len(days) - 1
    assert summary["tickers_without_second_source"] == ["CCC"]
    assert summary["benchmark_has_second_source"]
    # The second source is evidence, not a substitute: no return was taken from it.
    kept = a.dropna().index
    assert np.allclose(a[kept], plain.frame("AAA")["ret"][kept])


def test_an_event_whose_window_holds_an_unverified_return_has_no_car():
    from stock_research.eventstudy import study_event

    days = F.weekdays("2022-06-01", 200)
    panel = F.make_panel(days, ["AAA"], seed=4)
    frames = {t: f[prices.BAR_COLUMNS].dropna(subset=["close"]) for t, f in panel.bars.items()}
    second = {t: pd.DataFrame({"close_raw": f["close"], "close_adjusted": f["adj_close"]})
              for t, f in frames.items()}
    second["AAA"].loc[days[152], "close_adjusted"] *= 1.04
    checked = prices.build_panel(frames, panel.benchmark, origin=prices.ORIGIN_SYNTHETIC,
                                 snapshot_id="t", secondary=second)
    hit = study_event(checked, "AAA", days[150])            # window 0..+5 spans the bad close
    clear = study_event(checked, "AAA", days[160])
    assert hit["car_p0_p5"] is None and clear["car_p0_p5"] is not None


def test_a_corporate_action_does_not_excuse_a_move_beyond_the_limit():
    """Readiness finding A-2: four capital increases carried a wrong factor and
    produced adjusted returns of +13% to +43%. v2 exempted action dates from
    both tests that would have caught them."""

    days = F.weekdays("2023-03-01", 40)
    panel = F.make_panel(days, ["AAA"], seed=6)
    frame = panel.bars["AAA"][prices.BAR_COLUMNS].dropna(subset=["close"]).copy()
    # A split is recorded on day 10 and the adjusted series jumps 30% across it.
    frame.loc[days[10], "split_ratio"] = 2.0
    frame.loc[days[10]:, "adj_close"] *= 1.30
    bars = {"AAA": frame, panel.benchmark: panel.bars[panel.benchmark][prices.BAR_COLUMNS]}
    built = prices.build_panel(bars, panel.benchmark, origin=prices.ORIGIN_SYNTHETIC,
                               snapshot_id="t")
    assert np.isnan(built.frame("AAA")["ret"][days[10]])
    assert {"date": days[10], "issue": prices.ISSUE_ACTION_LIMIT} in built.issues["AAA"]
    # A dividend-sized adjusted move on an action date is left alone.
    frame2 = panel.bars["AAA"][prices.BAR_COLUMNS].dropna(subset=["close"]).copy()
    frame2.loc[days[10], "dividend"] = 0.5
    ok = prices.build_panel({"AAA": frame2, panel.benchmark: bars[panel.benchmark]},
                            panel.benchmark, origin=prices.ORIGIN_SYNTHETIC, snapshot_id="t")
    assert np.isfinite(ok.frame("AAA")["ret"][days[10]])


def test_corporate_actions_of_the_two_sources_are_compared():
    days, frames, second, benchmark = _two_sources()
    frames["AAA"] = frames["AAA"].copy()
    frames["AAA"].loc[days[15], "dividend"] = 0.2           # both sources record this one
    frames["BBB"] = frames["BBB"].copy()
    frames["BBB"].loc[days[25], "split_ratio"] = 2.0        # only the first records this one
    panel = prices.build_panel(frames, benchmark, origin=prices.ORIGIN_SYNTHETIC,
                               snapshot_id="t", secondary=second)
    actions = pd.DataFrame([
        {"ticker": "AAA", "action_date": days[15], "kind": secondary.KIND_DIVIDEND},
        {"ticker": "CCC", "action_date": days[40], "kind": secondary.KIND_RIGHTS},   # only the second
    ])
    report = prices.action_consistency(panel, actions)
    summary = report["summary"]
    assert summary["cash_dividends"] == {"listed": 1, "recorded_by_first_provider": 1,
                                         "return_by_state": {"verified": 1}}
    assert summary["capital_increases"]["listed"] == 1
    assert summary["capital_increases"]["recorded_by_first_provider"] == 0
    assert summary["actions_only_the_first_provider_records"] == 1


def test_second_source_snapshot_is_stored_append_only_and_kept_apart(tmp_path):
    import sqlite3

    path = str(tmp_path / "p.db")
    days, frames, second, benchmark = _two_sources()
    provider = secondary.FrameSecondary(second, actions={"AAA": [{
        "action_date": days[5], "kind": secondary.KIND_BONUS, "bonus_pct": 100.0,
        "adjustment_ratio": 0.5}]})
    first = secondary.ingest(["AAA", "BBB", "ZZZ"], days[0], "2099-01-01", provider, "sec", path)
    again = secondary.ingest(["AAA", "BBB", "ZZZ"], days[0], "2099-01-01", provider, "sec", path)
    assert first == {"ok": 2, "no_data": 1, "error": 0, "skipped": 0}
    assert again["skipped"] == 3
    loaded, actions = secondary.load("sec", path)
    assert set(loaded) == {"AAA", "BBB"} and len(actions) == 1
    assert np.allclose(loaded["AAA"]["close_adjusted"], second["AAA"]["close_adjusted"])
    connection = sqlite3.connect(path)
    try:
        for table in ("sr_raw_price_bars_secondary", "sr_raw_capital_actions"):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(f"DELETE FROM {table}")
        assert connection.execute("SELECT COUNT(*) FROM sr_raw_price_bars").fetchone()[0] == 0
    finally:
        connection.close()


# -- 3. The 2023 schedule ----------------------------------------------------------
def _calendar():
    closed = cal.EXCHANGE_SCHEDULE["2023"]
    return cal.SessionCalendar(F.weekdays("2023-01-02", 250, skip=(
        *closed["holidays"], *closed["cancelled_sessions"], *closed["unscheduled_closures"])))


def _at(stamp):
    return datetime.fromisoformat(stamp).replace(tzinfo=ISTANBUL)


def test_the_recorded_schedule_matches_a_calendar_built_from_it():
    calendar = _calendar()
    assert calendar.schedule_mismatches("2023") == []
    # A holiday that carries a session, and a weekday that lacks one, are both found.
    wrong = cal.SessionCalendar([*calendar.sessions, "2023-05-01"])
    assert any("2023-05-01" in problem for problem in wrong.schedule_mismatches("2023"))
    gap = cal.SessionCalendar([s for s in calendar.sessions if s != "2023-03-15"])
    assert any("2023-03-15" in problem for problem in gap.schedule_mismatches("2023"))
    assert gap.schedule_mismatches("2023", known_gaps=["2023-03-15"]) == []
    assert calendar.schedule_mismatches("2019") == ["no schedule recorded for 2019"]


def test_half_days_close_at_12_40_and_nothing_is_inferred_for_2023():
    calendar = _calendar()
    assert cal.EXCHANGE_SCHEDULE["2023"]["half_days"] == {"2023-04-20": "12:40",
                                                         "2023-06-27": "12:40"}
    during = calendar.actionable(_at("2023-06-27 12:39:00"))
    after = calendar.actionable(_at("2023-06-27 12:41:00"))
    assert (during.day0, during.bucket) == ("2023-06-27", cal.BUCKET_DURING)
    assert (after.day0, after.bucket, after.timing_ambiguous) == (
        "2023-07-03", cal.BUCKET_POST_CLOSE, False)
    # 29 August precedes a holiday and ran to 18:10.
    late = calendar.actionable(_at("2023-08-29 17:00:00"))
    assert (late.day0, late.bucket, late.timing_ambiguous) == (
        "2023-08-29", cal.BUCKET_DURING, False)


@pytest.mark.parametrize("stamp, delayed", [
    ("2023-02-07 11:00:00", False),      # traded that day
    ("2023-02-07 19:00:00", True),       # after the last close before the closure
    ("2023-02-08 11:00:00", True),       # the cancelled session
    ("2023-02-11 11:00:00", True),       # a weekend inside the closure
    ("2023-02-14 23:00:00", True),
    ("2023-02-15 09:00:00", False),      # the reopening morning: a normal pre-open filing
    ("2023-03-10 19:00:00", False),      # an ordinary Friday evening
    ("2023-04-22 11:00:00", False),      # a scheduled holiday weekend
])
def test_filings_dated_to_the_reopening_session_are_flagged(stamp, delayed):
    result = _calendar().actionable(_at(stamp))
    assert result.closure_delayed is delayed
    if delayed:
        assert result.day0 == "2023-02-15"


# -- 4 and 5. Taxonomy, eligibility, sufficiency -----------------------------------
def _rows(panel, specs, seed=3, issuers=None):
    rng = np.random.default_rng(seed)
    days = panel.calendar.sessions
    names = issuers or [t for t in panel.bars if t != panel.benchmark]
    rows = []
    for category, count in specs:
        for _ in range(count):
            rows.append({"event_id": f"K{len(rows)}", "ticker": str(rng.choice(names)),
                         "day0": days[int(rng.integers(150, len(days) - 25))],
                         "bucket": "post_close", "timing_ambiguous": False, "category": category,
                         "is_update": False, "is_correction": False, "closure_delayed": False})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def flat_panel():
    return F.make_panel(F.weekdays("2022-01-03", 520), [f"S{n:03d}" for n in range(60)], seed=9)


def test_the_registered_taxonomy_names_every_class_exactly_once():
    every = {getattr(events, name) for name in (
        "NEW_CONTRACT", "BUYBACK", "INSIDER_TRADE", "BONUS_ISSUE", "RIGHTS_ISSUE",
        "CAPITAL_CHANGE_OTHER", "DIVIDEND", "ASSET_ACQUISITION", "ASSET_SALE", "DEBT_ISSUANCE",
        "UNUSUAL_PRICE_VOLUME", "ADMINISTRATIVE", "GENERAL_OTHER", "OTHER_FORM")}
    tested, reference, outside = set(h7.INTEREST), set(h7.REFERENCE_CLASSES), set(h7.NOT_TESTED)
    assert tested | reference | outside == every
    assert not (tested & reference) and not (tested & outside) and not (reference & outside)
    assert set(h7.SPEC.parameters["interest"]) == tested
    assert h7.group_of("administrative") == h7.REFERENCE and h7.group_of("general_other") is None
    assert h7.group_of("general_other", h7.POOLED_REFERENCE_CLASSES) == h7.REFERENCE


def test_only_registered_classes_enter_and_closure_delayed_filings_do_not(flat_panel):
    rows = _rows(flat_panel, [("administrative", 40), ("buyback", 40), ("general_other", 40),
                              ("insider_trade", 20), ("unusual_price_volume", 20)])
    rows.loc[rows.index[:5], "closure_delayed"] = True     # five of the reference filings
    ctx = common.Context(panel=flat_panel, kap_events=rows, availability=F.ALL_AVAILABLE)
    eligible, counts = h7.prepare(ctx)
    assert set(eligible["group"]) == {h7.REFERENCE, "buyback"}
    assert counts["not_delayed_by_a_closure"] == counts["timing_usable"] - 5
    assert counts["in_registered_classes"] < counts["single_category_ticker_days"]


def test_an_issuers_overlapping_filings_count_once(flat_panel):
    days = flat_panel.calendar.sessions
    table = pd.DataFrame({"ticker": ["S001"] * 5 + ["S002"],
                          "day0": [days[200], days[201], days[205], days[206], days[212], days[201]]})
    # S001: 200 kept; 201 and 205 overlap it; 206 is six sessions on, kept; 212 kept.
    assert h7.thin(table, flat_panel.calendar).tolist() == [True, False, False, True, True, True]


def test_a_class_carried_by_a_few_issuers_is_not_tested(flat_panel):
    """Readiness finding I-2: v2 counted events and dates only."""

    everyone = [f"S{n:03d}" for n in range(60)]
    rows = pd.concat([
        _rows(flat_panel, [("administrative", 130), ("buyback", 90)], issuers=everyone),
        _rows(flat_panel, [("new_contract", 90)], seed=5, issuers=everyone[:6]),
    ], ignore_index=True)
    rows["event_id"] = [f"K{i}" for i in range(len(rows))]
    ctx = common.Context(panel=flat_panel, kap_events=rows, availability=F.ALL_AVAILABLE)
    result = h7.run(ctx)
    counts = result["sufficiency"]["counts"]
    assert counts["by_group"]["new_contract"]["issuers"] == 6
    assert counts["by_group"]["new_contract"]["events"] >= config.H7_MIN_EVENTS
    assert counts["tested_classes"] == ["buyback"]
    assert "new_contract" in counts["classes_below_the_minimum"]


def test_without_a_sufficient_reference_group_nothing_is_tested(flat_panel):
    rows = _rows(flat_panel, [("administrative", 30), ("buyback", 90), ("new_contract", 90)])
    result = h7.run(common.Context(panel=flat_panel, kap_events=rows, availability=F.ALL_AVAILABLE))
    assert result["status"] == common.DATA_INSUFFICIENT and result["primary"] is None
    assert "reference group" in result["sufficiency"]["reasons"][0]


# -- 7. Effect size ----------------------------------------------------------------
def _frame(effects, n=150, seed=0, sd=0.05):
    rng = np.random.default_rng(seed)
    parts = []
    for group, shift in effects.items():
        parts.append(pd.DataFrame({
            "group": group, "y": shift + sd * rng.normal(size=n),
            "day0": [f"d{int(v)}" for v in rng.integers(0, 60, n)],
            "ticker": [f"{group}{int(v)}" for v in rng.integers(0, 70, n)]}))
    return pd.concat(parts, ignore_index=True)


def test_contrasts_are_differences_from_the_reference_group():
    frame = _frame({h7.REFERENCE: 0.01, "buyback": 0.04, "dividend": 0.01})
    out = h7.contrasts(frame, ["buyback", "dividend"], "y", draws=999, seed=1)
    by_class = {row["class"]: row for row in out["contrasts"]}
    means = frame.groupby("group")["y"].mean()
    assert by_class["buyback"]["contrast"] == pytest.approx(means["buyback"] - means[h7.REFERENCE])
    assert out["reference_mean"] == pytest.approx(means[h7.REFERENCE])
    assert out["significant_classes"] == ["buyback"]
    assert out["largest_significant_contrast"] == pytest.approx(abs(by_class["buyback"]["contrast"]))
    assert by_class["dividend"]["p_holm"] >= by_class["dividend"]["p"]
    assert out["joint"]["p"] < 0.01


def test_under_no_effect_the_reported_size_is_usually_zero_not_a_max_minus_min():
    """v2 reported the highest class mean minus the lowest, which exceeded the
    1% material effect in 85% of samples with no effect at all (readiness
    finding W-3). The v3 size is zero unless a contrast is itself significant."""

    spreads, sizes, corrected = [], [], []
    for seed in range(40):
        frame = _frame({h7.REFERENCE: 0.0, "buyback": 0.0, "dividend": 0.0, "new_contract": 0.0},
                       n=40, seed=seed, sd=0.075)
        out = h7.contrasts(frame, ["buyback", "dividend", "new_contract"], "y", draws=199, seed=seed)
        means = frame.groupby("group")["y"].mean()
        spreads.append(means.max() - means.min())
        sizes.append(out["largest_significant_contrast"])
        corrected.append(out["rms_contrast_bias_corrected"])
    assert np.mean(np.array(spreads) >= 0.01) > 0.8          # the old measure, on noise
    assert np.mean(np.array(sizes) >= 0.01) <= 0.15          # the new one
    assert np.median(corrected) < 0.01


def test_h7_verdict_uses_the_contrast_not_the_spread():
    spec = h7.SPEC
    base = {"joint_test": True, "p_holm": 0.001}
    assert common.decide(spec, {**base, "estimate": 0.0}) == common.INCONCLUSIVE
    assert common.decide(spec, {**base, "estimate": 0.02}) == common.SUPPORTED
    assert common.decide(spec, {"joint_test": True, "p_holm": 0.2, "estimate": 0.02}) == common.INCONCLUSIVE
    assert "contrast" in spec.estimand and "bootstrap" in spec.inference


# -- 4. Gates ----------------------------------------------------------------------
def test_no_hypothesis_computes_an_outcome_on_real_data_before_the_gates_pass(tmp_path):
    """Readiness finding P-1: in v2 the checks sat in ``run_real`` and the
    runners could be called around them."""

    path = _database(str(tmp_path / "g.db"), per_form=5)
    ctx, meta = pipeline.real_context(path, index_db=tmp_path / "none.db")
    assert not ctx.outcomes_allowed
    assert not meta["gates"]["protocol_registered"]["passed"]
    for key, runner in registry.RUNNERS.items():
        with pytest.raises(common.OutcomesNotAllowed):
            runner(ctx)
    with pytest.raises(common.OutcomesNotAllowed):
        registry.run_all(ctx)
    # Counts are still available: they read no return.
    assert pipeline.describe(ctx, meta)["kap_events"] == 30
    # Synthetic contexts are not gated.
    assert F.kap_context().outcomes_allowed


def test_the_run_refuses_without_a_second_source_for_every_issuer(tmp_path):
    path = _database(str(tmp_path / "one.db"), per_form=5, second_source=False)
    registry.register(path)
    with pytest.raises(pipeline.NotReady, match="second price source"):
        pipeline.run_real(path, index_db=tmp_path / "none.db")
    coverage = pipeline.price_coverage(path)
    assert coverage["primary"]["never_asked"] == 0
    assert coverage["secondary"]["snapshot_id"] is None


def test_the_run_refuses_when_an_issuer_was_never_priced(tmp_path):
    """Readiness finding K-2: an issuer first seen after the last price fetch
    dropped out of the sample as 'unresolved', silently."""

    from stock_research import store
    from stock_research.data import kap

    path = _database(str(tmp_path / "late.db"), per_form=5)
    late = kap.detail_row({"disclosureIndex": "2999999", "senderId": "NEW", "senderTitle": "Yeni",
                           "senderExchCodes": ["NEWCO"], "relatedStocks": [],
                           "subject": {"tr": "Yeni İş İlişkisi"}, "summary": {"tr": "x"},
                           "time": "03.04.2023 19:00:00"}, b"late")
    with store.connect(path) as con:
        con.execute("INSERT INTO sr_raw_kap_listing VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (2999999, config.KAP_FRAME_VERSION, 0, "ODA", "ODA", "NEW", "Yeni", "{}",
                     "t", "now"))
        con.execute("INSERT INTO sr_kap_frame_items VALUES (?,?,?,?,?)",
                    (config.KAP_FRAME_VERSION, 2999999, 0, 1, "now"))
        con.execute(
            "INSERT INTO sr_raw_kap_detail VALUES (:disclosure_index,:sender_id,:sender_title,"
            ":sender_codes,:related_stocks,:disclosure_type,:disclosure_class,:disclosure_reason,"
            ":subject_tr,:subject_en,:summary_tr,:published_raw,:event_type_code,:body_text,"
            ":metadata_json,:payload_sha256,:source,:retrieved_at)", late)
    registry.register(path)
    found = pipeline.gates(path)
    assert not found["prices_requested_for_every_issuer"]["passed"]
    assert pipeline.price_coverage(path)["primary"]["never_asked_examples"] == ["NEWCO.IS"]
    with pytest.raises(pipeline.NotReady, match="never requested"):
        pipeline.run_real(path, index_db=tmp_path / "none.db")


def test_the_run_refuses_when_sessions_differ_from_the_recorded_schedule(tmp_path, monkeypatch):
    path = _database(str(tmp_path / "cal.db"), per_form=5)
    registry.register(path)
    schedule = {"2023": {**cal.EXCHANGE_SCHEDULE["2023"],
                         "holidays": cal.EXCHANGE_SCHEDULE["2023"]["holidays"] + ("2023-03-15",)}}
    monkeypatch.setattr(cal, "EXCHANGE_SCHEDULE", schedule)
    registry.register(path)                                # the schedule is part of the protocol
    with pytest.raises(pipeline.NotReady, match="recorded exchange schedule"):
        pipeline.run_real(path, index_db=tmp_path / "none.db")


# -- Protocol ----------------------------------------------------------------------
def test_v3_is_a_new_protocol_that_names_what_it_replaces():
    document = registry.protocol_document()
    assert document["protocol_version"] == "stock-research-protocol-v3"
    assert document["supersedes"]["protocol_hash"].startswith("09e9382e77ec44bc")
    assert document["lineage"][0]["protocol_hash"].startswith("92366aa0a6342e9c")
    assert len(document["changes_from_previous"]) >= 10
    assert document["constants"]["kap_frame"]["kind"] == "census"
    assert document["constants"]["h7"]["min_issuers"] == config.H7_MIN_ISSUERS
    assert document["constants"]["exchange_schedule"]["2023"]["half_days"]
    assert "data/secondary.py" in document["code_fingerprint"]


def test_constants_read_from_outside_the_package_move_the_hash(monkeypatch):
    """Readiness finding P-3."""

    import config as root

    before = registry.protocol_hash()
    monkeypatch.setattr(root, "BIST_HALF_DAYS", {**root.BIST_HALF_DAYS, "2023-04-20": "12:30"})
    assert registry.protocol_hash() != before


def test_every_rule_constant_of_v3_moves_the_hash(monkeypatch):
    before = registry.protocol_hash()
    for name, value in (("RECONCILE_TOLERANCE", 0.02), ("H7_MIN_ISSUERS", 5),
                        ("WILD_BOOTSTRAP_DRAWS", 999), ("KAP_MAX_UNAVAILABLE_SHARE", 0.01)):
        monkeypatch.setattr(config, name, value)
        assert registry.protocol_hash() != before, name
        monkeypatch.undo()


# -- Pacing: how fast the census arrives, never what is in it ----------------------
class _Reply:
    def __init__(self, status, retry_after=None):
        self.status_code, self.headers = status, ({} if retry_after is None
                                                  else {"Retry-After": retry_after})

    def raise_for_status(self):
        pass

    def json(self):
        return {"ok": True}


def _paced(monkeypatch, replies):
    from stock_research.data import kap

    waits, queue = [], list(replies)
    monkeypatch.setattr(kap, "PACE", kap.Pace())
    monkeypatch.setattr(kap, "_base_url", lambda: "http://gateway")
    monkeypatch.setattr(kap, "_auth", lambda: ("k", "s"))
    monkeypatch.setattr(kap.requests, "get", lambda *a, **k: queue.pop(0))
    monkeypatch.setattr(kap.time, "sleep", waits.append)
    return kap, waits


def test_a_refusal_widens_the_spacing_and_accepted_calls_narrow_it_again(monkeypatch):
    kap, waits = _paced(monkeypatch, [_Reply(429), _Reply(200)] + [_Reply(200)] * 70)
    assert kap.api_get("/x") == {"ok": True}
    # The refusal names no wait: the default applies, then the wider spacing.
    assert waits == [config.KAP_RATE_LIMIT_WAIT_SECONDS,
                     config.KAP_THROTTLE_SECONDS + config.KAP_THROTTLE_STEP_SECONDS]
    for _ in range(70):
        kap.api_get("/x")
    assert kap.PACE.spacing == config.KAP_THROTTLE_SECONDS      # back to the plan's spacing
    assert min(waits[1:]) >= config.KAP_THROTTLE_SECONDS         # never faster than the plan


def test_the_gateways_own_wait_is_used_and_refusals_do_not_use_up_attempts(monkeypatch):
    kap, waits = _paced(monkeypatch, [_Reply(429, "20")] * 8 + [_Reply(200)])
    assert kap.api_get("/x", attempts=3) == {"ok": True}        # eight refusals, three attempts
    assert waits[0] == 20 and kap.PACE.refusals == 8
    assert kap.PACE.spacing <= config.KAP_THROTTLE_MAX_SECONDS
    # A wait the gateway names is never shorter than the current spacing.
    assert all(w >= 20 for w in waits[:8])


def test_pacing_cannot_exceed_its_ceiling_and_ignores_a_nonsense_header(monkeypatch):
    from stock_research.data import kap

    pace = kap.Pace()
    for _ in range(50):
        wait = pace.refused("soon")
        assert wait == config.KAP_RATE_LIMIT_WAIT_SECONDS
    assert pace.spacing == config.KAP_THROTTLE_MAX_SECONDS
