"""Contracts for the reviewed reconstruction of legacy score provenance.

Experiment identity is never guessed.  It may be reconstructed only where the
stored evidence uniquely establishes it, and the reconstruction must stay
visibly distinct from provenance recorded at scoring time.  These tests hold
that line: the exact eligible set is assigned, nothing else is touched, a
conflicting scorer keeps blocking aggregation, and a rollback reverts only what
this migration actually wrote.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
import pipeline
from config import (
    LEGACY_PROVENANCE_MIGRATION_VERSION,
    REVIEWED_LEGACY_ASSIGNMENT_METHOD,
    REVIEWED_LEGACY_EXPERIMENT_ID,
    REVIEWED_LEGACY_MODEL_NAME,
    REVIEWED_LEGACY_ROLLBACK_METHOD,
)
from tests.fixtures.production_shaped import build_production_shaped_legacy_db


def _score_digest(db_path: str) -> list[tuple]:
    """Every field the migration must leave alone."""

    with db._conn(db_path) as con:
        return con.execute(
            "SELECT id, sentiment_score, sentiment_label, scored_at, model_name"
            " FROM headlines ORDER BY id"
        ).fetchall()


@pytest.fixture
def migrated_legacy(tmp_path):
    """A production-shaped legacy database with schema migrations applied."""

    path = tmp_path / "legacy.db"
    expectations = build_production_shaped_legacy_db(path)
    db.init_db(str(path))
    return str(path), expectations


def _add_headline(
    db_path: str,
    *,
    headline_id: int,
    model_name: str = REVIEWED_LEGACY_MODEL_NAME,
    experiment_id=None,
    processing_status: str = "scored",
    sentiment_score=0.4,
    components=(0.7, 0.2, 0.1),
    score_components_kind="synthetic_compatibility",
) -> int:
    with db._conn(db_path) as con:
        con.execute(
            """INSERT INTO headlines
               (id, source, title, url, published_at, scraped_at,
                sentiment_score, sentiment_label, scored_at, category,
                p_positive, p_neutral, p_negative, model_name, experiment_id,
                published_hour, relevance, signal_date, processing_status,
                score_components_kind)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                headline_id, "feed-x", f"Injected headline {headline_id}",
                f"https://example.test/injected/{headline_id}",
                "2026-07-20", "2026-07-20T09:00:00Z",
                sentiment_score, "positive", "2026-07-20T09:05:00Z",
                "bist_company", *components, model_name, experiment_id,
                11, 0.9, "2026-07-20", processing_status, score_components_kind,
            ),
        )
    return headline_id


# -- Assignment ---------------------------------------------------------------

def test_exactly_the_eligible_rows_receive_the_reviewed_identity(migrated_legacy):
    path, expectations = migrated_legacy
    result = db.backfill_reviewed_legacy_experiment_id(db_path=path)

    assert result["assigned"] == expectations["headline_count"] == 3465
    with db._conn(path) as con:
        assigned = con.execute(
            "SELECT COUNT(*) FROM headlines WHERE experiment_id = ?",
            (REVIEWED_LEGACY_EXPERIMENT_ID,),
        ).fetchone()[0]
        unassigned = con.execute(
            "SELECT COUNT(*) FROM headlines"
            " WHERE TRIM(COALESCE(experiment_id, '')) = ''"
        ).fetchone()[0]
    assert assigned == 3465
    assert unassigned == 0


def test_no_score_label_timestamp_or_model_name_changes(migrated_legacy):
    path, _ = migrated_legacy
    before = _score_digest(path)
    db.backfill_reviewed_legacy_experiment_id(db_path=path)
    assert _score_digest(path) == before


def test_existing_experiment_ids_are_never_overwritten(migrated_legacy):
    path, _ = migrated_legacy
    _add_headline(path, headline_id=900001, experiment_id="v2-preexisting")

    db.backfill_reviewed_legacy_experiment_id(db_path=path)

    with db._conn(path) as con:
        preserved = con.execute(
            "SELECT experiment_id FROM headlines WHERE id = ?", (900001,)
        ).fetchone()[0]
        audited = con.execute(
            "SELECT COUNT(*) FROM experiment_assignment_audit WHERE headline_id = ?",
            (900001,),
        ).fetchone()[0]
    assert preserved == "v2-preexisting"
    assert audited == 0, "an untouched row must not appear in the audit trail"


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"model_name": "gpt-5-mini-2025-08-07/p2"}, "different prompt version"),
        ({"model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment"}, "different scorer"),
        ({"processing_status": "retry_pending"}, "not resolved as scored"),
        ({"sentiment_score": None}, "no score"),
        ({"components": (None, 0.2, 0.1)}, "incomplete score components"),
        ({"score_components_kind": "softmax_probability"}, "conflicting component kind"),
    ],
)
def test_conflicting_evidence_leaves_provenance_unassigned(
    migrated_legacy, kwargs, reason
):
    path, _ = migrated_legacy
    _add_headline(path, headline_id=900002, **kwargs)

    db.backfill_reviewed_legacy_experiment_id(db_path=path)

    with db._conn(path) as con:
        experiment_id = con.execute(
            "SELECT experiment_id FROM headlines WHERE id = ?", (900002,)
        ).fetchone()[0]
    assert experiment_id is None, f"{reason} must not be reconstructed"


def test_conflicting_scorer_keeps_blocking_aggregation(migrated_legacy):
    """An unresolvable row must stay visible, not be quietly assimilated."""

    path, _ = migrated_legacy
    _add_headline(
        path, headline_id=900003,
        model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        score_components_kind="softmax_probability",
    )
    db.backfill_reviewed_legacy_experiment_id(db_path=path)

    identities = db.get_eligible_experiment_ids(db_path=path)
    assert len(identities) == 2
    assert REVIEWED_LEGACY_EXPERIMENT_ID in identities
    assert any("legacy-unassigned" in identity for identity in identities)

    with pytest.raises(pipeline.MixedExperimentAggregationError):
        pipeline.aggregate_step(db_path=path)


def test_survey_reports_blocked_rows_by_scorer(migrated_legacy):
    path, _ = migrated_legacy
    _add_headline(
        path, headline_id=900004,
        model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        score_components_kind="softmax_probability",
    )
    survey = db.survey_reviewed_legacy_candidates(db_path=path)
    assert survey["eligible"] == 3465
    assert survey["blocked_total"] == 1
    assert survey["blocked"] == {
        "cardiffnlp/twitter-xlm-roberta-base-sentiment": 1
    }


def test_dry_run_changes_nothing(migrated_legacy):
    path, _ = migrated_legacy
    before = _score_digest(path)
    result = db.backfill_reviewed_legacy_experiment_id(db_path=path, dry_run=True)

    assert result["assigned"] == 0
    assert result["eligible"] == 3465
    assert _score_digest(path) == before
    with db._conn(path) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM headlines"
            " WHERE TRIM(COALESCE(experiment_id, '')) <> ''"
        ).fetchone()[0] == 0


# -- Additivity and idempotency ----------------------------------------------

def test_migration_is_idempotent(migrated_legacy):
    path, _ = migrated_legacy
    first = db.backfill_reviewed_legacy_experiment_id(db_path=path)
    second = db.backfill_reviewed_legacy_experiment_id(db_path=path)

    assert first["assigned"] == 3465
    assert second["assigned"] == 0
    with db._conn(path) as con:
        audited = con.execute(
            "SELECT COUNT(*) FROM experiment_assignment_audit"
        ).fetchone()[0]
    assert audited == 3465, "a repeat run must not duplicate audit rows"


def test_migration_is_additive_and_row_counts_are_stable(migrated_legacy):
    path, expectations = migrated_legacy
    with db._conn(path) as con:
        before = {
            row[0]: con.execute(f"SELECT COUNT(*) FROM {row[0]}").fetchone()[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
                " AND name NOT LIKE 'sqlite_%'"
            )
        }
    db.backfill_reviewed_legacy_experiment_id(db_path=path)
    with db._conn(path) as con:
        after = {
            table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before
        }
    for table, count in before.items():
        if table == "experiment_assignment_audit":
            continue
        assert after[table] == count, f"{table} row count changed"
    assert after["headlines"] == expectations["headline_count"]


# -- Audit trail --------------------------------------------------------------

def test_audit_record_is_complete(migrated_legacy):
    path, _ = migrated_legacy
    db.backfill_reviewed_legacy_experiment_id(db_path=path)

    with db._conn(path) as con:
        rows = con.execute(
            "SELECT * FROM experiment_assignment_audit ORDER BY headline_id"
        ).fetchall()
    assert len(rows) == 3465
    for row in rows[:50]:
        assert row["headline_id"] is not None
        assert row["assigned_experiment_id"] == REVIEWED_LEGACY_EXPERIMENT_ID
        assert row["assignment_method"] == REVIEWED_LEGACY_ASSIGNMENT_METHOD
        assert row["migration_version"] == LEGACY_PROVENANCE_MIGRATION_VERSION
        assert row["reviewed_at"]
        evidence = json.loads(row["evidence"])
        assert evidence["model_name"] == REVIEWED_LEGACY_MODEL_NAME
        assert evidence["scored_at"]
        assert evidence["rule"]


def test_reconstruction_is_distinguishable_from_original_assignment(migrated_legacy):
    """A reviewed reconstruction and a scoring-time assignment must not blur."""

    path, _ = migrated_legacy
    db.backfill_reviewed_legacy_experiment_id(db_path=path)
    _add_headline(path, headline_id=900005, experiment_id=REVIEWED_LEGACY_EXPERIMENT_ID)

    reconstructed = {
        entry["headline_id"]
        for entry in db.list_reviewed_legacy_assignments(db_path=path)
    }
    assert len(reconstructed) == 3465
    assert 900005 not in reconstructed, (
        "a row assigned at scoring time must not appear as reconstructed"
    )


def test_audit_table_is_append_only(migrated_legacy):
    path, _ = migrated_legacy
    db.backfill_reviewed_legacy_experiment_id(db_path=path)

    with pytest.raises(sqlite3.IntegrityError):
        with db._conn(path) as con:
            con.execute(
                "UPDATE experiment_assignment_audit SET evidence = 'tampered'"
            )
    with pytest.raises(sqlite3.IntegrityError):
        with db._conn(path) as con:
            con.execute("DELETE FROM experiment_assignment_audit")

    with db._conn(path) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM experiment_assignment_audit"
        ).fetchone()[0] == 3465


# -- Aggregation outcome ------------------------------------------------------

def test_single_identity_after_the_reviewed_backfill(migrated_legacy):
    path, _ = migrated_legacy
    assert len(db.get_eligible_experiment_ids(db_path=path)) == 1

    db.backfill_reviewed_legacy_experiment_id(db_path=path)

    assert db.get_eligible_experiment_ids(db_path=path) == [
        REVIEWED_LEGACY_EXPERIMENT_ID
    ]


def test_new_v1p3_headline_aggregates_with_reviewed_legacy_rows(migrated_legacy):
    """The production failure mode, reproduced and then resolved."""

    path, _ = migrated_legacy
    db.backfill_reviewed_legacy_experiment_id(db_path=path)
    _add_headline(path, headline_id=900006, experiment_id=REVIEWED_LEGACY_EXPERIMENT_ID)

    assert db.get_eligible_experiment_ids(db_path=path) == [
        REVIEWED_LEGACY_EXPERIMENT_ID
    ]
    sessions = pipeline.aggregate_step(db_path=path)
    assert sessions > 0

    with db._conn(path) as con:
        variants = con.execute(
            "SELECT COUNT(*) FROM daily_signal_variants"
        ).fetchone()[0]
    assert variants > 0


def test_without_the_backfill_a_new_headline_blocks_aggregation(migrated_legacy):
    """Proves the test above resolves a real failure rather than a hypothetical."""

    path, _ = migrated_legacy
    _add_headline(path, headline_id=900007, experiment_id=REVIEWED_LEGACY_EXPERIMENT_ID)

    assert len(db.get_eligible_experiment_ids(db_path=path)) == 2
    with pytest.raises(pipeline.MixedExperimentAggregationError):
        pipeline.aggregate_step(db_path=path)


# -- Rollback -----------------------------------------------------------------

def test_rollback_restores_null_only_for_rows_this_migration_changed(
    migrated_legacy,
):
    path, _ = migrated_legacy
    _add_headline(path, headline_id=900008, experiment_id="v2-preexisting")
    db.backfill_reviewed_legacy_experiment_id(db_path=path)
    before = _score_digest(path)

    result = db.rollback_reviewed_legacy_experiment_id(db_path=path)

    assert result["reverted"] == 3465
    with db._conn(path) as con:
        nulled = con.execute(
            "SELECT COUNT(*) FROM headlines"
            " WHERE TRIM(COALESCE(experiment_id, '')) = ''"
        ).fetchone()[0]
        untouched = con.execute(
            "SELECT experiment_id FROM headlines WHERE id = ?", (900008,)
        ).fetchone()[0]
    assert nulled == 3465
    assert untouched == "v2-preexisting", "rollback must not clear foreign provenance"
    assert _score_digest(path) == before


def test_rollback_appends_rather_than_erasing_history(migrated_legacy):
    path, _ = migrated_legacy
    db.backfill_reviewed_legacy_experiment_id(db_path=path)
    db.rollback_reviewed_legacy_experiment_id(db_path=path)

    with db._conn(path) as con:
        methods = dict(
            con.execute(
                "SELECT assignment_method, COUNT(*)"
                " FROM experiment_assignment_audit GROUP BY assignment_method"
            ).fetchall()
        )
    assert methods == {
        REVIEWED_LEGACY_ASSIGNMENT_METHOD: 3465,
        REVIEWED_LEGACY_ROLLBACK_METHOD: 3465,
    }
    assert db.list_reviewed_legacy_assignments(db_path=path) == []


def test_rollback_is_idempotent(migrated_legacy):
    path, _ = migrated_legacy
    db.backfill_reviewed_legacy_experiment_id(db_path=path)
    first = db.rollback_reviewed_legacy_experiment_id(db_path=path)
    second = db.rollback_reviewed_legacy_experiment_id(db_path=path)

    assert first["reverted"] == 3465
    assert second["reverted"] == 0


def test_rollback_skips_rows_reassigned_after_the_migration(migrated_legacy):
    path, _ = migrated_legacy
    db.backfill_reviewed_legacy_experiment_id(db_path=path)
    with db._conn(path) as con:
        con.execute("UPDATE headlines SET experiment_id = 'v2-later' WHERE id = 1")

    result = db.rollback_reviewed_legacy_experiment_id(db_path=path)

    assert result["skipped_diverged"] == 1
    assert 1 in result["diverged_headline_ids"]
    with db._conn(path) as con:
        assert con.execute(
            "SELECT experiment_id FROM headlines WHERE id = 1"
        ).fetchone()[0] == "v2-later"


def test_apply_rollback_apply_returns_to_the_assigned_state(migrated_legacy):
    path, _ = migrated_legacy
    db.backfill_reviewed_legacy_experiment_id(db_path=path)
    db.rollback_reviewed_legacy_experiment_id(db_path=path)
    reapplied = db.backfill_reviewed_legacy_experiment_id(db_path=path)

    assert reapplied["assigned"] == 3465
    assert db.get_eligible_experiment_ids(db_path=path) == [
        REVIEWED_LEGACY_EXPERIMENT_ID
    ]
