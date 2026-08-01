"""Focused regression tests for session-table consumer alignment."""

import sqlite3

import pandas as pd
import pytest

import analyze_external
import dashboard
import database as database
import explore_signal
import visualize


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "close": [100.0, 110.0, 99.0],
            # Deliberately inconsistent: consumers must derive from close.
            "daily_return": [0.0, 999.0, 999.0],
        }
    )


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-06"],
            "simple_mean": [0.2, -0.1],
            "relevance_weighted": [0.25, -0.15],
            "intensity_relevance_weighted": [0.3, -0.2],
            "full_weighted": [0.35, -0.25],
            "headline_count": [3, 4],
            "positive_count": [2, 1],
            "negative_count": [0, 2],
            "neutral_count": [1, 1],
        }
    )


def test_visualize_uses_variant_baseline_and_complete_price_lead(monkeypatch):
    monkeypatch.setattr(visualize.db, "get_prices", lambda **_: _prices())
    monkeypatch.setattr(
        visualize.db, "get_signal_variants", lambda **_: _signals()
    )

    _, sentiment, merged = visualize._load_data("unused.db", days=90)

    assert sentiment["avg_score"].tolist() == [0.2, -0.1]
    first = merged.loc[merged["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    # The missing 2026-01-05 signal must not turn this into the -10% move to
    # 2026-01-06; the actual subsequent session close-to-close return is +10%.
    assert first["next_return"] == pytest.approx(10.0)


def test_explore_targets_are_formed_before_signal_filtering():
    prices = _prices()
    prices["date"] = pd.to_datetime(prices["date"])
    em = pd.DataFrame(
        {
            "date": pd.to_datetime(prices["date"]),
            "daily_return": [0.0, 1.0, -1.0],
        }
    )
    fx = pd.DataFrame(
        {
            "date": pd.to_datetime(prices["date"]),
            "close": [30.0, 30.3, 30.0],
        }
    )

    targets, _ = explore_signal.build_targets(prices, em, fx)

    first = targets.loc[targets["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    assert first["ret_next"] == pytest.approx(10.0)


def test_dashboard_uses_session_unweighted_baseline(tmp_path, monkeypatch):
    path = tmp_path / "dashboard.sqlite"
    database.init_db(str(path))
    run_id = database.log_run_start(db_path=str(path))
    database.log_run_end(
        run_id,
        status="degraded",
        market_data_status="degraded",
        db_path=str(path),
    )
    monkeypatch.setattr(dashboard.db, "get_prices", lambda **_: _prices())
    monkeypatch.setattr(
        dashboard.db, "get_signal_variants", lambda **_: _signals()
    )

    collected = dashboard._collect(str(path))

    assert [row["avg_score"] for row in collected["sent"]] == [0.2, -0.1]
    assert collected["reliable"] == 2

    output = tmp_path / "dashboard.html"
    dashboard.generate(str(path), str(output))
    html = output.read_text(encoding="utf-8")
    assert "Session-aligned unweighted news mood" in html
    assert "Reaction session 2026-01-06" in html
    assert 'class="pill warn">Last run: degraded' in html
    assert 'class="dot warn"' in html


def test_external_loader_has_no_calendar_fallback_and_preserves_true_lead(
    tmp_path, monkeypatch
):
    path = tmp_path / "external.sqlite"
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE external_series (date TEXT, series TEXT, value REAL)"
        )
        for day, offset in (("2026-01-02", 0.0), ("2026-01-06", 1.0)):
            con.executemany(
                "INSERT INTO external_series VALUES (?, ?, ?)",
                [
                    (day, "gt_dolar", 10.0 + offset),
                    (day, "gt_kriz", 20.0 + offset),
                    (day, "gdelt_tone", -1.0 + offset),
                ],
            )
        con.execute(
            "CREATE TABLE bist100_prices (date TEXT, close REAL)"
        )
        con.executemany(
            "INSERT INTO bist100_prices VALUES (?, ?)",
            list(_prices()[["date", "close"]].itertuples(index=False, name=None)),
        )
        con.execute(
            "CREATE TABLE market_factors (date TEXT, symbol TEXT, close REAL)"
        )
        con.executemany(
            "INSERT INTO market_factors VALUES (?, 'USDTRY=X', ?)",
            [
                ("2026-01-02", 30.0),
                ("2026-01-05", 30.3),
                ("2026-01-06", 30.0),
            ],
        )
        con.execute(
            "CREATE TABLE headlines "
            "(source TEXT, published_at TEXT, sentiment_score REAL)"
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        analyze_external.db, "get_signal_variants", lambda **_: _signals()
    )

    # The fixture deliberately has no daily_sentiment table. A legacy fallback
    # would therefore fail instead of silently changing the time convention.
    frame = analyze_external.load(str(path))
    assert frame.loc[pd.Timestamp("2026-01-02"), "avg_score"] == pytest.approx(0.2)
    assert frame.loc[pd.Timestamp("2026-01-02"), "bist_ret_next"] == pytest.approx(10.0)
