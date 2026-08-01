"""Focused integrity tests for additive database state and audit migrations."""

import json
import sqlite3
from pathlib import Path

import pytest

import database as db


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def integrity_db(tmp_path):
    path = str(tmp_path / "integrity.db")
    db.init_db(path)
    return path


@pytest.fixture
def production_legacy_db(tmp_path):
    path = tmp_path / "production-legacy.db"
    with sqlite3.connect(path) as con:
        con.executescript(
            (FIXTURES / "production_legacy.sql").read_text(encoding="utf-8")
        )
    return str(path)


def _headline_rows(path, columns="*"):
    with db._conn(path) as con:
        return con.execute(
            f"SELECT {columns} FROM headlines ORDER BY id"
        ).fetchall()


def test_legacy_backfill_classifies_rows_once_without_rewriting_scores(tmp_path):
    path = str(tmp_path / "legacy.db")
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE headlines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT UNIQUE,
            published_at TEXT,
            scraped_at TEXT NOT NULL,
            category TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            p_positive REAL,
            p_neutral REAL,
            p_negative REAL,
            model_name TEXT,
            scored_at TEXT
        );
        """
    )
    con.executemany(
        """INSERT INTO headlines
           (source, title, url, published_at, scraped_at, sentiment_score,
            sentiment_label, p_positive, p_neutral, p_negative, model_name,
            scored_at)
           VALUES (?, ?, ?, '2026-07-01', '2026-07-01T10:00:00Z', ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "feed", "complete", "https://example.test/complete",
                0.6, "positive", 0.7, 0.2, 0.1,
                "cardiffnlp/twitter-xlm-roberta", "2026-07-01T10:01:00Z",
            ),
            (
                "feed", "untouched", "https://example.test/untouched",
                None, None, None, None, None, None, None,
            ),
            (
                "feed", "partial", "https://example.test/partial",
                -0.4, None, None, None, None, None, None,
            ),
        ],
    )
    before = con.execute(
        """SELECT sentiment_score, sentiment_label, p_positive, p_neutral,
                  p_negative, model_name, scored_at
           FROM headlines ORDER BY id"""
    ).fetchall()
    con.commit()
    con.close()

    db.init_db(path)
    rows = _headline_rows(
        path,
        "processing_status, scoring_attempts, last_scoring_attempt_at, "
        "score_components_kind, sentiment_score, sentiment_label, p_positive, "
        "p_neutral, p_negative, model_name, scored_at",
    )
    assert [row["processing_status"] for row in rows] == [
        "scored", "pending", "retry_pending"
    ]
    assert [row["scoring_attempts"] for row in rows] == [0, 0, 0]
    assert rows[0]["last_scoring_attempt_at"] == "2026-07-01T10:01:00Z"
    assert rows[0]["score_components_kind"] == "softmax_probability"
    after_scores = [tuple(row)[4:] for row in rows]
    assert after_scores == [tuple(row) for row in before]

    with db._conn(path) as migrated:
        migrated.execute(
            "UPDATE headlines SET processing_status='failed' WHERE title='partial'"
        )
    db.init_db(path)
    assert _headline_rows(path, "processing_status")[2][0] == "failed"


def test_production_legacy_migration_is_additive_idempotent_and_nonregenerating(
    production_legacy_db,
):
    historical_tables = (
        "headlines",
        "bist100_prices",
        "daily_sentiment",
        "category_daily_sentiment",
        "daily_sentiment_by_signal",
        "usdtry_rates",
        "market_factors",
        "experiments",
        "events",
        "event_entities",
        "kv_state",
        "external_series",
        "pipeline_runs",
    )
    derived_tables = (
        "daily_sentiment",
        "category_daily_sentiment",
        "daily_sentiment_by_signal",
    )
    score_columns = (
        "id, sentiment_score, sentiment_label, p_positive, p_neutral, "
        "p_negative, model_name, scored_at"
    )

    with sqlite3.connect(production_legacy_db) as con:
        row_counts_before = {
            table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in historical_tables
        }
        scores_before = con.execute(
            f"SELECT {score_columns} FROM headlines ORDER BY id"
        ).fetchall()
        derived_before = {
            table: con.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in derived_tables
        }

    db.init_db(production_legacy_db)
    db.init_db(production_legacy_db)

    with sqlite3.connect(production_legacy_db) as con:
        row_counts_after = {
            table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in historical_tables
        }
        scores_after = con.execute(
            f"SELECT {score_columns} FROM headlines ORDER BY id"
        ).fetchall()
        derived_after = {
            table: con.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in derived_tables
        }
        newly_created_counts = {
            table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "daily_signal_variants",
                "category_sentiment_by_signal",
                "raw_headline_observations",
                "headline_exclusions",
            )
        }
        statuses = con.execute(
            "SELECT processing_status FROM headlines ORDER BY id"
        ).fetchall()

    assert row_counts_after == row_counts_before
    assert scores_after == scores_before
    assert derived_after == derived_before
    assert newly_created_counts == {
        "daily_signal_variants": 0,
        "category_sentiment_by_signal": 0,
        "raw_headline_observations": 0,
        "headline_exclusions": 0,
    }
    assert [row[0] for row in statuses] == ["scored", "scored", "pending"]


def test_schema_and_foreign_keys_are_additive_and_idempotent(integrity_db):
    db.init_db(integrity_db)
    with db._conn(integrity_db) as con:
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        headline_columns = {
            row[1] for row in con.execute("PRAGMA table_info(headlines)")
        }
        run_columns = {
            row[1] for row in con.execute("PRAGMA table_info(pipeline_runs)")
        }
        indexes = {row[1] for row in con.execute("PRAGMA index_list(headlines)")}
    assert {
        "processing_status", "scoring_attempts", "last_scoring_attempt_at",
        "scoring_last_error", "score_components_kind",
    } <= headline_columns
    assert {
        "scrape_status", "scoring_status", "aggregation_status",
        "market_data_status", "audit_status", "warnings_json", "errors_json",
    } <= run_columns
    assert "idx_headlines_processing_status" in indexes


def test_raw_observations_survive_global_url_dedup_without_rerun_bloat(integrity_db):
    observations = [
        {
            "source": "feed-a",
            "title": "BIST şirket haberi",
            "url": "https://example.test/shared",
            "published_at": "2026-07-02",
        },
        {
            "source": "feed-b",
            "title": "Aynı bağlantının ikinci kaynak gözlemi",
            "url": "https://example.test/shared",
            "published_at": "2026-07-02",
            "is_excluded": True,
            "exclusion_reason": "off_topic",
            "exclusion_rule": "missing_relevance_keyword",
            "exclusion_version": "keyword-v1",
        },
    ]
    assert db.insert_headlines(observations, integrity_db) == 1
    assert db.insert_headlines(observations, integrity_db) == 0

    raw = db.list_raw_headline_observations(integrity_db)
    assert len(raw) == 2
    assert len({row["observation_key"] for row in raw}) == 2
    assert {row["source"] for row in raw} == {"feed-a", "feed-b"}
    assert all(row["headline_id"] is not None for row in raw)
    excluded_raw = next(row for row in raw if row["source"] == "feed-b")
    assert excluded_raw["is_excluded"] == 1
    assert excluded_raw["exclusion_version"] == "keyword-v1"
    assert len(_headline_rows(integrity_db)) == 1
    assert db.count_active_headline_exclusions(integrity_db) == 1
    assert db.get_unscored_headlines(integrity_db).empty


def test_exclusions_are_idempotent_restorable_and_keep_history(integrity_db):
    db.insert_headlines(
        [{
            "source": "feed", "title": "BIST 100 yükseldi",
            "url": "https://example.test/one", "published_at": "2026-07-03",
        }],
        integrity_db,
    )
    headline_id = _headline_rows(integrity_db, "id")[0][0]

    assert db.exclude_headline(
        headline_id, "off_topic", "keyword", "v1", integrity_db
    )
    assert not db.exclude_headline(
        headline_id, "off_topic", "keyword", "v1", integrity_db
    )
    assert db.count_excluded_headlines(integrity_db) == 1
    assert len(db.list_headline_exclusions(integrity_db)) == 1

    assert db.restore_headline_exclusion(headline_id, integrity_db)
    assert not db.restore_headline_exclusion(headline_id, integrity_db)
    assert db.count_active_headline_exclusions(integrity_db) == 0
    assert list(db.get_unscored_headlines(integrity_db)["id"]) == [headline_id]

    assert db.exclude_headline(
        headline_id, "off_topic", "keyword", "v2", integrity_db
    )
    history = db.list_headline_exclusions(
        integrity_db, active_only=False, headline_id=headline_id
    )
    assert len(history) == 2
    assert history[0]["restored_at"] is not None
    assert history[1]["restored_at"] is None


def test_relevance_rule_exclusion_is_reversible_without_restoring_other_rules(
    integrity_db,
):
    db.insert_headlines(
        [{
            "source": "feed", "title": "BIST relevance audit",
            "url": "https://example.test/relevance", "published_at": "2026-07-03",
        }],
        integrity_db,
    )
    headline_id = _headline_rows(integrity_db, "id")[0][0]

    db.update_relevance([(0.1, headline_id)], integrity_db)
    assert db.reconcile_relevance_exclusions([headline_id], integrity_db) == {
        "excluded": 1, "restored": 0,
    }
    assert db.reconcile_relevance_exclusions([headline_id], integrity_db) == {
        "excluded": 0, "restored": 0,
    }

    db.update_relevance([(0.8, headline_id)], integrity_db)
    assert db.reconcile_relevance_exclusions([headline_id], integrity_db) == {
        "excluded": 0, "restored": 1,
    }
    history = db.list_headline_exclusions(
        integrity_db, active_only=False, headline_id=headline_id,
    )
    assert history[0]["exclusion_rule"] == "llm_relevance"
    assert history[0]["restored_at"] is not None

    db.exclude_headline(headline_id, "manual review", "manual", "v1", integrity_db)
    assert db.reconcile_relevance_exclusions([headline_id], integrity_db) == {
        "excluded": 0, "restored": 0,
    }
    assert db.count_active_headline_exclusions(integrity_db) == 1


def test_same_title_from_different_sources_remains_canonical_when_urls_differ(
    integrity_db,
):
    observations = [
        {
            "source": "feed-a", "title": "BIST ortak baslik",
            "url": "https://a.example/story", "published_at": "2026-07-03",
        },
        {
            "source": "feed-b", "title": "BIST ortak baslik",
            "url": "https://b.example/story", "published_at": "2026-07-03",
        },
    ]
    assert db.insert_headlines(observations, integrity_db) == 2
    assert len(_headline_rows(integrity_db)) == 2
    assert len(db.list_raw_headline_observations(integrity_db)) == 2


def test_scoring_failures_exhaust_and_preserve_nulls(integrity_db):
    db.insert_headlines(
        [{
            "source": "feed", "title": "TCMB faiz kararı",
            "url": "https://example.test/score", "published_at": "2026-07-04",
        }],
        integrity_db,
    )
    headline_id = _headline_rows(integrity_db, "id")[0][0]

    assert db.mark_scoring_attempt_failed(
        headline_id, "missing response item", 2, integrity_db
    ) == "retry_pending"
    assert db.mark_scoring_attempt_failed(
        headline_id, "missing response item", 2, integrity_db
    ) == "failed"
    # Exhaustion is idempotent and does not manufacture a neutral score.
    assert db.mark_scoring_attempt_failed(
        headline_id, "late duplicate", 2, integrity_db
    ) == "failed"
    row = _headline_rows(
        integrity_db,
        "processing_status, scoring_attempts, sentiment_score, sentiment_label, "
        "p_positive, p_neutral, p_negative, scoring_last_error",
    )[0]
    assert row["processing_status"] == "failed"
    assert row["scoring_attempts"] == 2
    assert all(row[name] is None for name in (
        "sentiment_score", "sentiment_label", "p_positive", "p_neutral", "p_negative"
    ))
    assert row["scoring_last_error"] == "missing response item"


def test_scoring_success_and_legacy_batch_shapes_set_component_metadata(integrity_db):
    items = [
        {
            "source": "feed", "title": f"BIST scoring {i}",
            "url": f"https://example.test/scoring/{i}",
            "published_at": f"2026-07-0{i + 4}",
        }
        for i in range(1, 4)
    ]
    db.insert_headlines(items, integrity_db)
    ids = [row[0] for row in _headline_rows(integrity_db, "id")]

    assert db.mark_scoring_attempt_failed(ids[0], "omitted", 3, integrity_db) == "retry_pending"
    assert db.mark_scoring_success(
        ids[0], 0.7, "positive", 0.7, 0.3, 0.0,
        "gpt-5-mini/p3", "synthetic_compatibility", integrity_db,
    )
    assert not db.mark_scoring_success(
        ids[0], 0.7, "positive", 0.7, 0.3, 0.0,
        "gpt-5-mini/p3", "synthetic_compatibility", integrity_db,
    )
    db.batch_update_sentiment(
        [
            (0.2, "positive", 0.5, 0.2, 0.3, "xlm-roberta", ids[1]),
            (-0.4, "negative", 0.1, 0.4, 0.5, "custom", "custom_components", ids[2]),
        ],
        integrity_db,
    )

    rows = _headline_rows(
        integrity_db,
        "processing_status, scoring_attempts, score_components_kind, "
        "scoring_last_error",
    )
    assert [row["processing_status"] for row in rows] == ["scored"] * 3
    assert [row["score_components_kind"] for row in rows] == [
        "synthetic_compatibility", "softmax_probability", "custom_components"
    ]
    assert rows[0]["scoring_attempts"] == 2
    assert rows[0]["scoring_last_error"] is None


def test_permanent_delete_requires_explicit_confirmation(integrity_db):
    db.insert_headlines(
        [{
            "source": "feed", "title": "BIST silme koruması",
            "url": "https://example.test/delete", "published_at": "2026-07-08",
        }],
        integrity_db,
    )
    headline_id = _headline_rows(integrity_db, "id")[0][0]
    db.exclude_headline(headline_id, "test", db_path=integrity_db)

    with pytest.raises(PermissionError):
        db.delete_headlines([headline_id], integrity_db)
    assert len(_headline_rows(integrity_db)) == 1
    assert db.permanently_delete_headlines(
        [headline_id], integrity_db, confirm=True
    ) == 1
    assert not _headline_rows(integrity_db)
    assert db.count_active_headline_exclusions(integrity_db) == 0


def test_pipeline_run_component_statuses_json_and_legacy_statuses(integrity_db):
    run_id = db.log_run_start(model_name="test", db_path=integrity_db)
    db.log_run_end(
        run_id,
        status="degraded",
        headlines_scraped=5,
        headlines_scored=4,
        db_path=integrity_db,
        scrape_status="success",
        scoring_status="degraded",
        aggregation_status="success",
        market_data_status="success",
        audit_status="skipped",
        warnings=[{"code": "one_item_omitted", "headline_id": 4}],
        errors=[],
    )
    run = db.get_pipeline_run(run_id, integrity_db)
    assert run["canonical_status"] == "degraded"
    assert run["scoring_status"] == "degraded"
    assert run["warnings"] == [{"code": "one_item_omitted", "headline_id": 4}]
    assert json.loads(run["errors_json"]) == []

    legacy_id = db.log_run_start(model_name="legacy", db_path=integrity_db)
    db.log_run_end(legacy_id, status="ok", db_path=integrity_db)
    legacy = db.get_pipeline_run(legacy_id, integrity_db)
    assert legacy["status"] == "ok"
    assert legacy["canonical_status"] == "success"
    stats = db.db_stats(integrity_db)
    assert stats["last_run_status"] == "ok"
    assert stats["last_run_final_status"] == "success"
    assert set(stats["last_run_components"]) == {
        "scrape", "scoring", "aggregation", "market_data", "audit"
    }

    with pytest.raises(ValueError):
        db.log_run_end(run_id, status="mystery", db_path=integrity_db)


def test_interrupted_run_is_failed_with_structured_component_diagnostics(integrity_db):
    run_id = db.log_run_start(
        model_name="test",
        db_path=integrity_db,
        scrape_status="running",
        scoring_status="pending",
        aggregation_status="success",
        market_data_status="pending",
        audit_status="pending",
    )

    assert db.fail_interrupted_pipeline_runs(integrity_db) == 1
    assert db.fail_interrupted_pipeline_runs(integrity_db) == 0
    run = db.get_pipeline_run(run_id, integrity_db)
    assert run["status"] == "failed"
    assert run["scrape_status"] == "failed"
    assert run["scoring_status"] == "skipped"
    assert run["aggregation_status"] == "success"
    assert run["market_data_status"] == "skipped"
    assert run["audit_status"] == "skipped"
    assert run["errors"][0]["code"] == "scheduler_interrupted_run"
