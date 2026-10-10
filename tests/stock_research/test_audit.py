"""Regression tests for the findings of the 2026-10-11 independent audit.

Each test reproduces a defect that was confirmed in the first implementation
and fails if it comes back. docs/stock_research/AUDIT_REPORT.md has the
findings these belong to.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from stock_research import fixtures as F
from stock_research import stats
from stock_research.eventstudy import (
    STATUS_OK, collapse_same_day, event_table, non_overlapping,
)
from stock_research.hypotheses import common, h1, h2, h3, h5, h6, h7, h8, registry


# -- C1: overlapping windows of one issuer ------------------------------------
def test_overlapping_windows_need_issuer_clustering():
    """No effect anywhere; twelve issuers each file on ten nearby sessions, so
    their 20-session windows overlap. Clustering by date alone treats those
    windows as independent and rejects far too often."""

    trials, by_date, two_way = 120, 0, 0
    for trial in range(trials):
        panel = F.synthetic_panel(tickers=12, sessions=300, seed=5000 + trial)
        rng = np.random.default_rng(trial)
        sessions = panel.calendar.sessions
        events = []
        for n in range(12):
            start = int(rng.integers(150, 240))
            events += [{"ticker": f"S{n:03d}", "day0": sessions[start + 2 * k]} for k in range(10)]
        table = event_table(panel, events, windows=((1, 20),))
        ok = table[(table["status"] == STATUS_OK) & table["car_p1_p20"].notna()]
        values, dates, tickers = (ok["car_p1_p20"].to_numpy(), ok["day0"].to_numpy(),
                                  ok["ticker"].to_numpy())
        by_date += stats.cluster_mean(values, [dates])["p"] < 0.05
        result = stats.cluster_mean(values, [dates, tickers])
        two_way += bool(result["p"] is not None and result["p"] < 0.05)
    assert by_date / trials > 0.25          # the defect: about half in practice
    assert two_way / trials < 0.12          # the fix holds near the nominal 5%


def test_primary_tests_with_multi_day_windows_cluster_two_ways():
    kap = F.kap_context(effect=True)
    assert len(h8.run(kap)["primary"]["clusters"]) == 2
    assert len(h7.run(kap)["primary"]["clusters"]) == 2
    assert len(h1.run(F.news_context(effect="h1"))["primary"]["clusters"]) == 2
    assert len(h5.run(F.limit_context(effect=True))["primary"]["clusters"]) == 2
    for key in ("H1", "H3", "H4", "H5", "H6", "H7", "H8"):
        assert "issuer" in registry.SPECS[key].inference


def test_non_overlapping_thins_greedily_per_issuer():
    panel = F.synthetic_panel(tickers=2, sessions=200, seed=1)
    days = panel.calendar.sessions
    table = pd.DataFrame({"ticker": ["A"] * 4 + ["B"],
                          "day0": [days[100], days[110], days[121], days[142], days[105]]})
    # A: keep 100; 110 is within 20; 121 is 21 after 100, keep; 142 is 21 after 121, keep.
    assert non_overlapping(table, 20, panel.calendar).tolist() == [True, False, True, True, True]


# -- H1/H2: price-triggered disclosures and during-session day 0 ---------------
def _kap_rows(panel, specs, seed=3):
    rng = np.random.default_rng(seed)
    days, names = panel.calendar.sessions, [t for t in panel.bars if t != panel.benchmark]
    rows = []
    for category, bucket, count in specs:
        for _ in range(count):
            rows.append({"event_id": f"K{len(rows)}", "ticker": str(rng.choice(names)),
                         "day0": days[int(rng.integers(150, len(days) - 25))], "bucket": bucket,
                         "timing_ambiguous": False, "category": category, "is_update": False,
                         "is_correction": False})
    return pd.DataFrame(rows)


def test_h7_ignores_price_triggered_disclosures():
    """A category filed *because* the price moved has a large day-0 return by
    construction. It must not be able to produce a 'categories differ' result."""

    days = F.weekdays("2022-01-03", 520)
    names = [f"S{n:03d}" for n in range(60)]
    flat = F.make_panel(days, names, seed=9)
    rows = _kap_rows(flat, [("cat_a", "post_close", 60), ("cat_b", "post_close", 60),
                            ("cat_c", "post_close", 60), ("unusual_price_volume", "post_close", 60)])
    position = {d: i for i, d in enumerate(days)}
    effects = {(r.ticker, position[r.day0]): 0.08 for r in rows.itertuples()
               if r.category == "unusual_price_volume"}
    panel = F.make_panel(days, names, seed=9, intraday_effects=effects)
    ctx = common.Context(panel=panel, kap_events=rows, availability=F.ALL_AVAILABLE)
    result = h7.run(ctx)
    assert "unusual_price_volume" not in result["sufficiency"]["counts"]["by_category"]
    assert result["primary"]["p"] > 0.01


def test_h7_primary_uses_only_events_whose_day0_starts_before_publication():
    panel = F.make_panel(F.weekdays("2022-01-03", 520), [f"S{n:03d}" for n in range(60)], seed=4)
    rows = _kap_rows(panel, [("cat_a", "post_close", 40), ("cat_a", "during_session", 40),
                             ("cat_b", "pre_open", 40), ("cat_c", "post_close", 40)])
    ctx = common.Context(panel=panel, kap_events=rows, availability=F.ALL_AVAILABLE)
    clean, counts = h7.prepare(ctx)
    assert set(clean["bucket"]) == {"post_close", "pre_open"}
    assert counts["clean_reaction_timing"] <= counts["not_price_triggered"] - 35
    everything, _ = h7.prepare(ctx, clean_timing=False)
    assert "during_session" in set(everything["bucket"])


def test_events_sharing_a_day0_but_not_a_timing_are_marked_mixed():
    events = [{"event_id": 1, "ticker": "A", "day0": "2023-01-02", "category": "x", "bucket": "pre_open"},
              {"event_id": 2, "ticker": "A", "day0": "2023-01-02", "category": "x", "bucket": "during_session"},
              {"event_id": 3, "ticker": "B", "day0": "2023-01-02", "category": "x", "bucket": "pre_open"}]
    merged = {r["ticker"]: r for r in collapse_same_day(events)}
    assert merged["A"]["bucket"] == "mixed" and merged["B"]["bucket"] == "pre_open"


# -- H3: label leakage across the train/test boundary ---------------------------
def test_training_events_with_unresolved_outcomes_are_embargoed():
    panel = F.synthetic_panel(tickers=2, sessions=120, seed=1)
    days = panel.calendar.sessions
    table = pd.DataFrame({"day0": days[10:110], "x": range(100)})
    train, test, info = common.chronological_split(table, "day0", 0.7, 5, panel.calendar)
    first_test = panel.calendar.index(test["day0"].min())
    # A training event's window (day0 .. day0+5) must end before the first test date.
    assert panel.calendar.index(train["day0"].max()) + 5 < first_test
    assert info["embargoed_rows"] == 5 and len(train) == 65 and len(test) == 30
    assert set(train["day0"]).isdisjoint(test["day0"])


def test_h3_and_h6_record_their_embargo():
    assert h3.run(F.news_context(effect="h3"))["sufficiency"]["counts"]["split"]["embargo_sessions"] == 5
    assert h6.run(F.limit_context(effect=True))["sufficiency"]["counts"]["split"]["embargo_sessions"] == 1


# -- M1: the surprise and the raw level must describe the same number ----------
def test_surprise_is_the_events_own_sentiment_against_its_baseline():
    ctx = F.news_context(effect="h3")
    # Add casual mentions with very different sentiment on the same ticker-days.
    extra = ctx.news_events.groupby(["ticker", "day0"]).head(1).copy()
    extra["event_id"] = "X" + extra["event_id"]
    extra["mention_type"], extra["sentiment"] = "casual", -0.9
    noisy = common.Context(panel=ctx.panel, size=ctx.size, availability=ctx.availability,
                           news_events=pd.concat([ctx.news_events, extra], ignore_index=True))
    from stock_research.hypotheses import prep

    events, _ = prep.admissible_news(noisy)
    history = noisy.news_events.groupby(["ticker", "day0"], as_index=False)["sentiment"].mean()
    baseline = h3.sentiment_baseline(history, noisy.panel.calendar)
    merged = events.merge(baseline, on=["ticker", "day0"]).dropna(subset=["baseline_sd"])
    row = merged.iloc[0]
    expected = (row["sentiment"] - row["baseline_mean"]) / row["baseline_sd"]
    # The admissible event's sentiment is not the all-mentions mean for the day.
    day_mean = history.set_index(["ticker", "day0"]).loc[(row["ticker"], row["day0"]), "sentiment"]
    assert abs(row["sentiment"] - day_mean) > 0.01
    from_day_mean = (day_mean - row["baseline_mean"]) / row["baseline_sd"]
    assert np.isfinite(expected) and abs(expected - from_day_mean) > 0.01
    assert h3.run(noisy)["status"] is None            # still runs on the consistent definition


# -- M2: the protocol hash covers code and module constants ---------------------
def test_protocol_hash_covers_source_files_and_hypothesis_parameters():
    document = registry.protocol_document()
    files = document["code_fingerprint"]
    assert {"eventstudy.py", "stats.py", "data/prices.py", "hypotheses/h8.py",
            "hypotheses/common.py", "insider.py", "events.py"} <= set(files)
    assert "fixtures.py" not in files and "reporting.py" not in files
    assert document["hypotheses"]["H5"]["parameters"]["negative_sentiment_threshold"] == h5.NEGATIVE
    assert document["hypotheses"]["H3"]["parameters"]["train_share"] == h3.TRAIN_SHARE
    assert document["hypotheses"]["H6"]["parameters"]["ridge"] == h6.RIDGE
    assert document["hypotheses"]["H2"]["parameters"]["limit_open_threshold"] == h2.LIMIT_OPEN
    assert all(spec["parameters"] for spec in document["hypotheses"].values())


def test_code_fingerprint_ignores_line_endings_but_not_content(monkeypatch, tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "config.py").write_bytes(b"A = 1\nB = 2\n")
    fake = type("M", (), {"__file__": str(package / "config.py")})
    monkeypatch.setattr(registry, "config", fake)
    unix = registry.code_fingerprint()
    (package / "config.py").write_bytes(b"A = 1\r\nB = 2\r\n")
    assert registry.code_fingerprint() == unix
    (package / "config.py").write_bytes(b"A = 1\nB = 3\n")
    assert registry.code_fingerprint() != unix
    assert unix == {"config.py": hashlib.sha256(b"A = 1\nB = 2\n").hexdigest()}


# -- M3: execution assumptions ---------------------------------------------------
def test_hedged_strategies_pay_for_both_legs_and_do_not_assume_short_sales():
    cases = {"H1": h1.run(F.news_context(effect="h1")), "H2": h2.run(F.news_context(effect="h2")),
             "H5": h5.run(F.limit_context(effect=True)), "H8": h8.run(F.kap_context(effect=True))}
    for key, result in cases.items():
        execution = result["execution"]
        assert execution["round_trips_per_trade"] == 2, key
        net, gross = execution["net_mean"], execution["gross_mean"]
        assert net["round_trip_40bp"] == pytest.approx(gross - 2 * 0.004), key
    assert "short" in cases["H1"]["execution"]["not_assumed"]
    assert "short" in cases["H2"]["execution"]["not_assumed"]
    small_events = cases["H1"]["execution"]["coverage"]
    assert small_events["long_signals"] < small_events["small_cap_events"]
