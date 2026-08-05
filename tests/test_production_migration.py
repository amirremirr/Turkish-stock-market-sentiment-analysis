"""Migration invariants on a production-shaped legacy database.

``test_database_integrity.py`` already locks the small hand-written legacy
fixture.  These tests cover what only volume and realistic distributions can
reach: that thousands of historical scores survive untouched, that a superseded
trading-calendar rule is corrected rather than trusted, that repeated
initialisation is inert, and that a stale local database cannot overwrite a
newer canonical snapshot.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
from scripts.verify_migration import run_verification, snapshot
from tests.fixtures.production_shaped import (
    LEGACY_SCHEMA,
    build_production_shaped_legacy_db,
)


FIXTURE_SQL = REPOSITORY_ROOT / "tests" / "fixtures" / "production_legacy.sql"


def _table_columns(sql_text: str) -> dict[str, list[str]]:
    """Return ``table -> [column, ...]`` by executing the script in memory.

    Letting SQLite parse the DDL is what makes this comparison trustworthy:
    a regex would have to re-implement column parsing and would silently
    disagree with the engine on formatting the two fixtures do differently.
    """

    con = sqlite3.connect(":memory:")
    try:
        con.executescript(sql_text)
        names = [
            row[0] for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            name: [row[1] for row in con.execute(f"PRAGMA table_info({name})")]
            for name in names
        }
    finally:
        con.close()


@pytest.fixture(scope="module")
def legacy_db(tmp_path_factory) -> dict:
    path = tmp_path_factory.mktemp("legacy") / "production_shaped.db"
    return build_production_shaped_legacy_db(path)


@pytest.fixture(scope="module")
def verification(legacy_db, tmp_path_factory) -> dict:
    workdir = tmp_path_factory.mktemp("migrated")
    return run_verification(Path(legacy_db["path"]), workdir)


def test_fixture_schema_matches_the_pinned_legacy_schema():
    """Both legacy fixtures must describe the same pre-migration database."""

    pinned = _table_columns(FIXTURE_SQL.read_text(encoding="utf-8"))
    generated = _table_columns(LEGACY_SCHEMA)
    assert set(pinned) == set(generated)
    for table in sorted(pinned):
        assert pinned[table] == generated[table], f"{table} columns drifted"


def test_fixture_has_production_shape(legacy_db):
    assert legacy_db["headline_count"] == 3465
    assert legacy_db["null_published_hour"] > 300
    assert legacy_db["below_relevance_floor"] > 100
    assert legacy_db["stale_signal_date_rows"] > 0, (
        "the fixture must contain stale calendar assignments to correct"
    )

    con = sqlite3.connect(legacy_db["path"])
    try:
        present = {
            row[0] for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert set(legacy_db["legacy_tables"]).issubset(present)
        for table in ("daily_sentiment", "daily_sentiment_by_signal",
                      "category_daily_sentiment"):
            assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0
    finally:
        con.close()


def test_fixture_generation_is_deterministic(tmp_path):
    first = build_production_shaped_legacy_db(tmp_path / "a.db")
    second = build_production_shaped_legacy_db(tmp_path / "b.db")
    assert first["null_published_hour"] == second["null_published_hour"]
    assert (
        snapshot(Path(first["path"]), stage="a")["content_digest"]
        == snapshot(Path(second["path"]), stage="b")["content_digest"]
    )


def test_migration_passes_every_gate(verification):
    failed = [item["check"] for item in verification["checks"] if not item["passed"]]
    assert not failed, f"failed gates: {failed}"
    assert verification["passed"]


def test_historical_scores_are_never_rewritten(verification):
    digests = {
        stage["stage"]: stage["score_digest"] for stage in verification["stages"]
    }
    baseline = digests["baseline"]
    assert baseline is not None
    for stage in ("init_db", "init_db_2", "sessions", "relevance"):
        assert digests[stage] == baseline, f"{stage} rewrote historical scores"


def test_migration_is_purely_additive(verification):
    stages = {stage["stage"]: stage for stage in verification["stages"]}
    before, after = stages["baseline"], stages["init_db"]
    assert not set(before["tables"]) - set(after["tables"])
    for table, columns in before["columns"].items():
        assert set(columns).issubset(set(after["columns"][table]))


def test_second_init_db_is_a_no_op(verification):
    stages = {stage["stage"]: stage for stage in verification["stages"]}
    assert stages["init_db"]["content_digest"] == stages["init_db_2"]["content_digest"]
    assert stages["init_db"]["columns"] == stages["init_db_2"]["columns"]


def test_every_legacy_scored_row_becomes_scored(verification, legacy_db):
    stages = {stage["stage"]: stage for stage in verification["stages"]}
    assert stages["init_db"]["processing_status"] == {
        "scored": legacy_db["headline_count"]
    }


def test_legacy_rows_keep_one_compatible_experiment_identity(verification, legacy_db):
    assert verification["eligible_experiment_ids"] == [
        f"[legacy-unassigned] model={legacy_db['model_name']}"
    ]


def test_missing_hour_rows_receive_a_conservative_bucket(verification, legacy_db):
    stages = {stage["stage"]: stage for stage in verification["stages"]}
    buckets = stages["sessions"]["timing_bucket"]
    conservative = buckets.get("unknown", 0) + buckets.get("weekend_or_holiday", 0)
    assert conservative >= legacy_db["null_published_hour"]
    assert "NULL" not in buckets, "every row must receive a timing bucket"


def test_corrected_signal_dates_target_real_trading_sessions(verification):
    changes = verification["session_backfill"]
    assert changes["changed_count"] > 0, "the fixture's stale dates must be corrected"
    assert changes["unverifiable_count"] == 0
    for example in changes["examples"]:
        assert example["new_is_trading_day"]
        if example["new_has_price_row"] is not None:
            assert example["new_has_price_row"]


def test_calendar_correction_is_reported_as_a_deviation(verification):
    kinds = {item["deviation"] for item in verification["deviations"]}
    assert "signal_date re-derived under the current calendar rule" in kinds
    assert "reversible low-relevance exclusions created" in kinds


def test_low_relevance_exclusions_are_reversible_not_deletions(verification, legacy_db):
    stages = {stage["stage"]: stage for stage in verification["stages"]}
    assert (
        stages["relevance"]["row_counts"]["headlines"] == legacy_db["headline_count"]
    ), "relevance reconciliation must never delete a headline"
    assert verification["relevance_reconciliation"]["excluded"] > 0

    con = sqlite3.connect(verification["migrated_copy"])
    try:
        active, restorable = con.execute(
            "SELECT COUNT(*), SUM(restored_at IS NULL) FROM headline_exclusions"
        ).fetchone()
        assert active == restorable, "every exclusion must start restorable"
    finally:
        con.close()


def test_historical_derived_tables_survive_initialisation(verification):
    """init_db must not clear the legacy aggregates; only aggregate_step may."""

    stages = {stage["stage"]: stage for stage in verification["stages"]}
    for table in ("daily_sentiment", "daily_sentiment_by_signal",
                  "category_daily_sentiment"):
        assert (
            stages["init_db"]["row_counts"][table]
            == stages["baseline"]["row_counts"][table]
        )


def test_verification_never_writes_to_the_source(legacy_db, tmp_path):
    before = snapshot(Path(legacy_db["path"]), stage="before")
    run_verification(Path(legacy_db["path"]), tmp_path / "isolated")
    after = snapshot(Path(legacy_db["path"]), stage="after")
    assert before["file_sha256"] == after["file_sha256"]
