import json

import pandas as pd
import pytest

from analysis.prediction import sensitivity


def _signals(dates, values):
    return pd.DataFrame(
        {
            "signal_date": dates,
            "simple_mean": values,
            "relevance_weighted": [value * 0.8 for value in values],
            "intensity_relevance_weighted": [value * 0.6 for value in values],
            "full_weighted": [value * -0.5 for value in values],
        }
    )


def test_next_session_return_is_computed_before_sparse_signal_join(monkeypatch):
    signal_frame = _signals(["2026-01-02", "2026-01-06"], [1.0, -1.0])
    price_frame = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"],
            "close": [100.0, 110.0, 99.0, 108.9],
        }
    )
    monkeypatch.setattr(
        sensitivity.database, "get_signal_variants", lambda **_: signal_frame, raising=False
    )
    monkeypatch.setattr(sensitivity.database, "get_prices", lambda **_: price_frame)

    report = sensitivity.run_sensitivity_analysis("ignored.db")
    aligned = report["aligned_observations"]

    assert aligned[0]["signal_date"] == "2026-01-02"
    assert aligned[0]["next_session_date"] == "2026-01-05"
    assert aligned[0]["next_session_return"] == pytest.approx(0.10)
    assert aligned[1]["signal_date"] == "2026-01-06"
    assert aligned[1]["next_session_date"] == "2026-01-07"
    assert aligned[1]["next_session_return"] == pytest.approx(0.10)
    # A post-join shift would incorrectly compare Jan 2 directly with Jan 6.
    assert aligned[0]["next_session_return"] != pytest.approx(-0.01)


def test_reports_every_variant_without_preference_and_exact_metrics(monkeypatch):
    dates = ["2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"]
    signal_frame = _signals(dates, [1.0, -1.0, 1.0, -1.0])
    price_frame = pd.DataFrame(
        {
            "date": dates + ["2026-02-06"],
            "close": [100.0, 110.0, 99.0, 108.9, 98.01],
        }
    )
    monkeypatch.setattr(
        sensitivity.database, "get_signal_variants", lambda **_: signal_frame, raising=False
    )
    monkeypatch.setattr(sensitivity.database, "get_prices", lambda **_: price_frame)

    report = sensitivity.run_sensitivity_analysis("ignored.db")

    assert report["metadata"]["preferred_variant"] is None
    assert report["metadata"]["variants"] == list(sensitivity.VARIANTS)
    predictive = {row["variant"]: row for row in report["predictive"]}
    assert set(predictive) == set(sensitivity.VARIANTS)
    assert predictive["simple_mean"]["pearson_r"] == pytest.approx(1.0)
    assert predictive["simple_mean"]["directional_hit_rate"] == 1.0
    assert predictive["full_weighted"]["pearson_r"] == pytest.approx(-1.0)
    assert predictive["full_weighted"]["directional_hit_rate"] == 0.0
    assert all(row["low_sample_size"] for row in report["predictive"])
    assert all("exploratory" in row["analysis_type"] for row in report["predictive"])

    pairs = report["directional_agreement"]
    opposite = next(
        row
        for row in pairs
        if row["variant_a"] == "simple_mean" and row["variant_b"] == "full_weighted"
    )
    assert opposite["agreement_rate_including_zero"] == 0.0
    assert len(report["distributions"]) == 4


def test_output_is_strict_json_and_contains_alignment_audit(monkeypatch, tmp_path):
    signal_frame = _signals(["2026-03-02"], [0.0])
    price_frame = pd.DataFrame({"date": ["2026-03-02"], "close": [100.0]})
    monkeypatch.setattr(
        sensitivity.database, "get_signal_variants", lambda **_: signal_frame, raising=False
    )
    monkeypatch.setattr(sensitivity.database, "get_prices", lambda **_: price_frame)
    output = tmp_path / "sensitivity.json"

    returned = sensitivity.run_sensitivity_analysis("ignored.db", output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted == returned
    assert persisted["aligned_observations"][0]["next_session_return"] is None
    assert persisted["predictive"][0]["pearson_r"] is None
    assert persisted["metadata"]["signals_with_next_session_target"] == 0


def test_duplicate_market_dates_fail_loudly(monkeypatch):
    signal_frame = _signals(["2026-04-01"], [0.2])
    price_frame = pd.DataFrame(
        {"date": ["2026-04-01", "2026-04-01"], "close": [100.0, 101.0]}
    )
    monkeypatch.setattr(
        sensitivity.database, "get_signal_variants", lambda **_: signal_frame, raising=False
    )
    monkeypatch.setattr(sensitivity.database, "get_prices", lambda **_: price_frame)

    with pytest.raises(ValueError, match="duplicate market dates"):
        sensitivity.run_sensitivity_analysis("ignored.db")
