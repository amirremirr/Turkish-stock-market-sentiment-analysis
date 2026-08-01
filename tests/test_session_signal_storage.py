"""Integration tests for timestamp storage and session-level signal variants."""

import pytest

import database as db
from pipeline import aggregate_step
from scraper import _parse_date, _parse_hour, _parse_timestamp


@pytest.fixture
def signal_db(tmp_path):
    path = str(tmp_path / "signals.db")
    db.init_db(path)
    return path


def test_timezone_normalization_moves_date_and_hour_together():
    raw = "2026-05-26T22:30:00Z"
    assert _parse_date(raw).isoformat() == "2026-05-27"
    assert _parse_hour(raw) == 1
    assert _parse_timestamp(raw) == "2026-05-27T01:30:00+03:00"
    assert _parse_timestamp("2026-05-26") is None


def test_insert_stores_timestamp_bucket_and_first_reactable_session(signal_db):
    rows = [
        {
            "source": "feed", "title": "BIST pre open",
            "url": "https://example.test/pre", "published_at": "2026-06-10",
            "published_timestamp": "2026-06-10T09:15:00+03:00",
        },
        {
            "source": "feed", "title": "BIST after close",
            "url": "https://example.test/post", "published_at": "2026-06-10",
            "published_timestamp": "2026-06-10T18:11:00+03:00",
        },
        {
            "source": "feed", "title": "BIST unknown time",
            "url": "https://example.test/unknown", "published_at": "2026-06-10",
        },
    ]
    assert db.insert_headlines(rows, signal_db) == 3

    with db._conn(signal_db) as con:
        stored = con.execute(
            """SELECT published_timestamp, timing_bucket, signal_date,
                      session_rule_version
               FROM headlines ORDER BY id"""
        ).fetchall()
        raw = con.execute(
            """SELECT published_timestamp, timing_bucket
               FROM raw_headline_observations ORDER BY observation_id"""
        ).fetchall()

    assert [row["timing_bucket"] for row in stored] == [
        "pre_open", "post_close", "unknown"
    ]
    assert [row["signal_date"] for row in stored] == [
        "2026-06-10", "2026-06-11", "2026-06-11"
    ]
    assert stored[0]["published_timestamp"] == "2026-06-10T09:15:00+03:00"
    assert all(row["session_rule_version"] for row in stored)
    assert raw[0]["timing_bucket"] == "pre_open"
    assert raw[1]["published_timestamp"] == "2026-06-10T18:11:00+03:00"


def test_aggregate_stores_simple_baseline_and_weighted_sensitivities(signal_db):
    rows = [
        {
            "source": "feed-a", "title": "BIST positive pre open",
            "url": "https://a.example/positive", "published_at": "2026-06-10",
            "published_timestamp": "2026-06-10T09:00:00+03:00",
            "category": "bist_company",
        },
        {
            "source": "feed-b", "title": "BIST negative intraday",
            "url": "https://b.example/negative", "published_at": "2026-06-10",
            "published_timestamp": "2026-06-10T12:00:00+03:00",
            "category": "bist_company",
        },
        {
            "source": "feed-c", "title": "BIST explicit neutral post close",
            "url": "https://c.example/neutral", "published_at": "2026-06-10",
            "published_timestamp": "2026-06-10T19:00:00+03:00",
            "category": "bist_company",
        },
    ]
    db.insert_headlines(rows, signal_db)
    with db._conn(signal_db) as con:
        ids = [row[0] for row in con.execute("SELECT id FROM headlines ORDER BY id")]
    db.batch_update_sentiment(
        [
            (1.0, "positive", 1.0, 0.0, 0.0, "gpt-test/p3",
             "synthetic_compatibility", ids[0]),
            (-0.5, "negative", 0.0, 0.5, 0.5, "gpt-test/p3",
             "synthetic_compatibility", ids[1]),
            (0.0, "neutral", 0.0, 1.0, 0.0, "gpt-test/p3",
             "synthetic_compatibility", ids[2]),
        ],
        signal_db,
    )
    db.update_relevance([(1.0, ids[0]), (0.5, ids[1]), (1.0, ids[2])], signal_db)

    assert aggregate_step(signal_db) == 2
    signals = db.get_signal_variants(db_path=signal_db)
    first = signals[signals["date"] == "2026-06-10"].iloc[0]
    second = signals[signals["date"] == "2026-06-11"].iloc[0]

    assert first["simple_mean"] == pytest.approx(0.25)
    assert first["relevance_weighted"] == pytest.approx(0.5)
    assert first["intensity_relevance_weighted"] == pytest.approx(0.7)
    assert first["full_weighted"] == pytest.approx(1.375 / 1.75)
    assert first["headline_count"] == 2
    assert first["positive_share"] == pytest.approx(0.5)
    assert first["negative_share"] == pytest.approx(0.5)
    assert first["neutral_share"] == pytest.approx(0.0)
    assert first["source_count"] == 2

    # Explicit neutral is retained and assigned to the next session; it is not
    # confused with a missing scorer response.
    assert second["simple_mean"] == 0.0
    assert second["neutral_share"] == 1.0
    assert second["headline_count"] == 1

    categories = db.get_category_signal_sentiment(signal_db)
    assert set(categories["date"]) == {"2026-06-10", "2026-06-11"}
    assert set(categories["simple_mean"]) == {0.25, 0.0}
