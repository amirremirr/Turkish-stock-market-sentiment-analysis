"""Focused regression tests for session-aligned market evaluation."""

import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import evaluate


def test_aggregate_audit_labels_simple_mean_as_primary(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "aggregate.db"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """CREATE TABLE category_daily_sentiment (
                   date TEXT,
                   category TEXT,
                   avg_score REAL,
                   headline_count INTEGER
               )"""
        )

    rows = [
        {
            "signal_date": "2026-06-01",
            "simple_mean": 0.2,
            "relevance_weighted": 0.3,
            "intensity_relevance_weighted": 0.4,
            "full_weighted": 0.5,
            "headline_count": 4,
            "positive_share": 0.75,
            "negative_share": 0.25,
        },
        {
            "signal_date": "2026-06-02",
            "simple_mean": -0.1,
            "relevance_weighted": -0.2,
            "intensity_relevance_weighted": -0.3,
            "full_weighted": -0.4,
            "headline_count": 6,
            "positive_share": 0.25,
            "negative_share": 0.75,
        },
    ]
    monkeypatch.setattr(
        evaluate.db,
        "get_signal_variants",
        # Production currently returns both the normalized date alias and the
        # stored signal_date column.  The loader must not create duplicate dates.
        lambda start=None, end=None, db_path=None: pd.DataFrame(rows).assign(
            date=lambda frame: frame["signal_date"]
        ),
        raising=False,
    )
    monkeypatch.setattr(evaluate, "MINIMUM_HEADLINES_PER_DAY", 1)

    result = evaluate.audit_aggregate(str(db_path))
    output = capsys.readouterr().out

    assert result == {"days": 2, "mean_articles_per_day": 5.0}
    assert "Primary baseline distribution (simple unweighted mean)" in output
    assert "alternatives are diagnostics, not selected models" in output
    assert "Legacy calendar-date category aggregate (descriptive only)" in output


def test_audit_signal_uses_session_baseline_and_full_price_sequence(
    tmp_path, monkeypatch, capsys
):
    """A missing signal session must not change the subsequent-session target."""
    db_path = tmp_path / "evaluation.db"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """CREATE TABLE bist100_prices (
                   date TEXT PRIMARY KEY,
                   close REAL,
                   daily_return REAL
               )"""
        )
        con.executemany(
            "INSERT INTO bist100_prices(date, close, daily_return) VALUES (?, ?, ?)",
            [
                ("2026-06-01", 100.0, 10.0),
                ("2026-06-02", 98.0, -2.0),
                # There is deliberately no signal for June 2.
                ("2026-06-03", 88.2, -10.0),
                ("2026-06-04", 84.672, -4.0),
            ],
        )

    rows = [
        {
            "session_date": "2026-06-01",
            "simple_mean": 1.0,
            "relevance_weighted": 0.9,
            "intensity_relevance_weighted": 0.8,
            "full_weighted": 0.7,
            "headline_count": 5,
        },
        {
            "session_date": "2026-06-03",
            "simple_mean": -1.0,
            "relevance_weighted": -0.9,
            "intensity_relevance_weighted": -0.8,
            "full_weighted": -0.7,
            "headline_count": 5,
        },
    ]
    monkeypatch.setattr(
        evaluate.db,
        "get_signal_variants",
        lambda start=None, end=None, db_path=None: rows,
        raising=False,
    )
    monkeypatch.setattr(evaluate, "MINIMUM_HEADLINES_PER_DAY", 1)
    monkeypatch.setattr(evaluate, "MINIMUM_OVERLAP_DAYS", 2)

    result = evaluate.audit_signal(str(db_path))
    output = capsys.readouterr().out

    # Targets are June 2 (-2%) and June 4 (-4%). Computing shift after the
    # sparse signal merge would leave only one observation instead.
    assert result["overlapping_days"] == 2
    assert result["signal_variant"] == "simple_mean"
    assert result["pearson_r_next"] == pytest.approx(1.0)
    assert "Primary baseline: simple_mean (unweighted)" in output
    assert "daily_sentiment_by_signal" not in output


def test_directional_agreement_excludes_zero_signal_and_zero_return(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "zero-directions.db"
    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE bist100_prices (date TEXT PRIMARY KEY, close REAL, daily_return REAL)"
        )
        con.executemany(
            "INSERT INTO bist100_prices VALUES (?, ?, ?)",
            [
                ("2026-06-01", 100.0, None),
                ("2026-06-02", 100.0, 0.0),
                ("2026-06-03", 110.0, 10.0),
                ("2026-06-04", 99.0, -10.0),
            ],
        )
    signals = pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-02", "2026-06-03"],
            "simple_mean": [1.0, 0.0, -1.0],
            "headline_count": [3, 3, 3],
        }
    )
    monkeypatch.setattr(
        evaluate.db, "get_signal_variants", lambda **kwargs: signals,
    )
    monkeypatch.setattr(evaluate, "MINIMUM_HEADLINES_PER_DAY", 1)
    monkeypatch.setattr(evaluate, "MINIMUM_OVERLAP_DAYS", 3)

    result = evaluate.audit_signal(str(db_path))

    assert result["overlapping_days"] == 3
    assert result["directional_observations"] == 1
    assert result["hit_rate"] == 1.0
