"""Phase A pipeline integration, migration safety and workflow validity.

The load-bearing guarantee: descriptive indicators are additive analysis layered
on top of the canonical tables. They may fail, and the run must survive with
every pre-existing aggregate intact -- a nice-to-have analytic must never be
able to take down measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
import pipeline
from scripts.cohort_integrity import compare, fingerprint
from scripts.verify_migration import snapshot
from tests.fixtures.production_shaped import build_production_shaped_legacy_db

WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"

PHASE_A_TABLES = (
    "daily_family_signals",
    "abnormal_tone_daily",
    "news_disagreement_daily",
    "news_volume_daily",
)


@pytest.fixture(scope="module")
def legacy_migrated(tmp_path_factory):
    """A production-shaped legacy database carried through every migration."""

    path = tmp_path_factory.mktemp("phasea") / "legacy.db"
    expectations = build_production_shaped_legacy_db(path)
    db.init_db(str(path))
    db.backfill_session_assignments(db_path=str(path))
    db.reconcile_relevance_exclusions(db_path=str(path))
    db.backfill_reviewed_legacy_experiment_id(db_path=str(path))
    return str(path), expectations


# -- Migration additivity and idempotency --------------------------------------

def test_phase_a_migration_is_additive(legacy_migrated):
    path, _ = legacy_migrated
    before = snapshot(Path(path), stage="before")
    db.init_db(path)
    after = snapshot(Path(path), stage="after")

    assert not set(before["tables"]) - set(after["tables"])
    for table, columns in before["columns"].items():
        assert set(columns).issubset(set(after["columns"][table]))
    for table in PHASE_A_TABLES:
        assert table in after["tables"]


def test_repeat_initialisation_is_a_content_no_op(legacy_migrated):
    path, _ = legacy_migrated
    db.init_db(path)
    first = snapshot(Path(path), stage="first")
    db.init_db(path)
    second = snapshot(Path(path), stage="second")
    assert first["content_digest"] == second["content_digest"]


def test_indicators_do_not_touch_scores_categories_or_provenance(legacy_migrated):
    path, _ = legacy_migrated
    baseline = fingerprint(path)
    pipeline.indicators_step(db_path=path)
    result = compare(baseline, fingerprint(path))
    assert result["passed"], result["failed"]


def test_indicators_leave_the_overall_variant_table_unchanged(legacy_migrated):
    path, _ = legacy_migrated
    pipeline.aggregate_step(db_path=path)
    with db._conn(path) as con:
        before = con.execute(
            "SELECT * FROM daily_signal_variants ORDER BY signal_date"
        ).fetchall()
    pipeline.indicators_step(db_path=path)
    with db._conn(path) as con:
        after = con.execute(
            "SELECT * FROM daily_signal_variants ORDER BY signal_date"
        ).fetchall()
    assert before == after


def test_indicator_step_is_idempotent_on_production_shape(legacy_migrated):
    path, _ = legacy_migrated
    first = pipeline.indicators_step(db_path=path, return_outcome=True)
    second = pipeline.indicators_step(db_path=path, return_outcome=True)
    assert first.count == second.count
    assert second.details["classified"]["classified"] == 0


def test_indicators_populate_every_phase_a_table(legacy_migrated):
    path, _ = legacy_migrated
    pipeline.indicators_step(db_path=path)
    with db._conn(path) as con:
        for table in PHASE_A_TABLES:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count > 0, f"{table} is empty"


# -- Degradation behaviour -------------------------------------------------------

def test_indicator_failure_degrades_without_corrupting_aggregates(
    legacy_migrated, monkeypatch
):
    path, _ = legacy_migrated
    pipeline.aggregate_step(db_path=path)
    with db._conn(path) as con:
        variants_before = con.execute(
            "SELECT * FROM daily_signal_variants ORDER BY signal_date"
        ).fetchall()

    def _boom(*args, **kwargs):
        raise RuntimeError("indicator computation exploded")

    monkeypatch.setattr(db, "classify_signal_families", _boom)
    outcome = pipeline.indicators_step(db_path=path, return_outcome=True)

    assert outcome.status == "degraded"
    assert outcome.count == 0
    assert [w["code"] for w in outcome.warnings] == ["indicator_computation_failed"]

    with db._conn(path) as con:
        variants_after = con.execute(
            "SELECT * FROM daily_signal_variants ORDER BY signal_date"
        ).fetchall()
    assert variants_after == variants_before, "aggregates must survive intact"


def test_ambiguous_assignments_do_not_degrade_a_healthy_run(legacy_migrated):
    """Ambiguity is a reported property of the taxonomy, not a fault."""

    path, _ = legacy_migrated
    db.classify_signal_families(db_path=path, force=True)
    outcome = pipeline.indicators_step(db_path=path, return_outcome=True)
    assert outcome.status == "success"
    codes = {w["code"] for w in outcome.warnings}
    assert codes <= {"ambiguous_family_assignments"}


def test_empty_database_reports_success_not_failure(tmp_path):
    path = str(tmp_path / "empty.db")
    db.init_db(path)
    outcome = pipeline.indicators_step(db_path=path, return_outcome=True)
    assert outcome.status == "success"
    assert outcome.count == 0


# -- Workflow validity ------------------------------------------------------------

def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_both_workflows_still_parse_and_share_concurrency():
    daily = _workflow("daily.yml")
    after = _workflow("after_close_prices.yml")
    assert daily["concurrency"]["group"] == after["concurrency"]["group"]


def test_after_close_workflow_does_not_run_phase_a_headline_analytics():
    """Price refresh must not touch the taxonomy or indicator tables."""

    after = _workflow("after_close_prices.yml")
    commands = "\n".join(
        step.get("run", "") for step in after["jobs"]["refresh"]["steps"]
    )
    for forbidden in ("indicators_step", "classify_signal_families",
                      "main.py run", "main.py aggregate"):
        assert forbidden not in commands


def test_after_close_refresh_never_calls_indicator_code(tmp_path, monkeypatch):
    from scripts import after_close_refresh

    def _fail(*args, **kwargs):                       # pragma: no cover
        raise AssertionError("the price job must not run headline analytics")

    monkeypatch.setattr(pipeline, "indicators_step", _fail)
    monkeypatch.setattr(db, "classify_signal_families", _fail)
    path = str(tmp_path / "p.db")
    db.init_db(path)
    result = after_close_refresh.run(path, now="2026-07-30T09:00:00+03:00")
    assert result["action"] == "skipped"


def test_daily_workflow_still_guards_before_publishing():
    daily = _workflow("daily.yml")
    names = [step["name"] for step in daily["jobs"]["run"]["steps"]]
    assert names.index("Guard against publishing a stale snapshot") < names.index(
        "Persist DB + chart to data branch"
    )
