"""Contracts for the three operational defects found in the controlled run.

1. The processing audit counted excluded-at-ingest headlines as unresolved, so
   every healthy run reported degraded and the signal lost its meaning.
2. A repaired price bar lost its ``corrected`` provenance to an ordinary
   refresh that returned the same settled values.
3. The first row of a fetch window had no predecessor inside that window, so
   its provider-derived return was NULL and overwrote a valid stored value.

Plus the after-close refresh guard, which must decide from the real Istanbul
session close rather than from a drifting UTC cron.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
import pipeline
from price_bars import (
    STATUS_COMPLETE,
    STATUS_CORRECTED,
    STATUS_PROVIDER_INVALID,
    STATUS_PROVISIONAL,
    after_close_refresh_allowed,
    resolve_bar_status,
)

WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"
FULL_DAY = "2026-07-30"
HALF_DAY = "2026-05-26"
HOLIDAY = "2026-07-15"
SATURDAY = "2026-08-01"
SETTLED = "2026-08-05T12:00:00+03:00"


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume", "daily_return"],
    )


@pytest.fixture
def audit_db(tmp_path):
    path = str(tmp_path / "audit.db")
    db.init_db(path)
    return path


def _add_headline(path, headline_id, status, *, excluded=False, scored=True):
    with db._conn(path) as con:
        if scored and status == "scored":
            con.execute(
                """INSERT INTO headlines (id, source, title, url, published_at,
                   scraped_at, sentiment_score, sentiment_label, scored_at,
                   p_positive, p_neutral, p_negative, model_name, experiment_id,
                   processing_status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (headline_id, "feed", f"H{headline_id}", f"u{headline_id}",
                 "2026-07-20", "2026-07-20T09:00:00Z", 0.3, "positive",
                 "2026-07-20T09:05:00Z", 0.7, 0.2, 0.1, "m", "v1-p3", status),
            )
        else:
            con.execute(
                """INSERT INTO headlines (id, source, title, url, published_at,
                   scraped_at, processing_status)
                   VALUES (?,?,?,?,?,?,?)""",
                (headline_id, "feed", f"H{headline_id}", f"u{headline_id}",
                 "2026-07-20", "2026-07-20T09:00:00Z", status),
            )
    if excluded:
        db.exclude_headline(headline_id, "off_topic", "keyword", "v1", db_path=path)


# -- 1. Audit semantics --------------------------------------------------------

def test_excluded_pending_rows_do_not_degrade_the_run(audit_db):
    """The production case: 198 filtered headlines must not mark a run degraded."""

    _add_headline(audit_db, 1, "scored")
    for index in range(2, 12):
        _add_headline(audit_db, index, "pending", excluded=True, scored=False)

    outcome = pipeline._processing_audit(audit_db)
    assert outcome.status == "success"
    assert outcome.details["pending_excluded"] == 10
    assert outcome.details["pending_eligible"] == 0
    assert outcome.details["active_exclusions"] == 10
    # Visible, but as information rather than an alarm.
    assert [w["code"] for w in outcome.warnings] == ["excluded_items_not_scored"]


def test_eligible_pending_rows_do_degrade_the_run(audit_db):
    _add_headline(audit_db, 1, "scored")
    _add_headline(audit_db, 2, "pending", scored=False)

    outcome = pipeline._processing_audit(audit_db)
    assert outcome.status == "degraded"
    assert outcome.details["pending_eligible"] == 1
    assert [w["code"] for w in outcome.warnings] == ["unresolved_processing_items"]


def test_eligible_retry_and_failed_rows_remain_visible(audit_db):
    _add_headline(audit_db, 1, "retry_pending", scored=False)
    _add_headline(audit_db, 2, "failed", scored=False)
    _add_headline(audit_db, 3, "failed", excluded=True, scored=False)

    outcome = pipeline._processing_audit(audit_db)
    assert outcome.status == "degraded"
    assert outcome.details["retry_pending_eligible"] == 1
    assert outcome.details["failed_eligible"] == 1
    assert outcome.details["failed_excluded"] == 1


def test_audit_detail_carries_every_required_key(audit_db):
    _add_headline(audit_db, 1, "scored")
    _add_headline(audit_db, 2, "scored", excluded=True)
    _add_headline(audit_db, 3, "pending", excluded=True, scored=False)

    detail = pipeline._processing_audit(audit_db).details
    for key in ("scored", "pending_eligible", "retry_pending_eligible",
                "failed_eligible", "pending_excluded", "scored_excluded",
                "active_exclusions"):
        assert key in detail, key
    assert detail["scored"] == 2
    assert detail["scored_excluded"] == 1


def test_excluded_rows_never_enter_aggregation(audit_db):
    _add_headline(audit_db, 1, "scored")
    _add_headline(audit_db, 2, "scored", excluded=True)
    with db._conn(audit_db) as con:
        con.execute("UPDATE headlines SET signal_date='2026-07-20'")

    pipeline.aggregate_step(db_path=audit_db)
    with db._conn(audit_db) as con:
        counted = con.execute(
            "SELECT headline_count FROM daily_signal_variants"
        ).fetchone()[0]
    assert counted == 1, "the excluded headline must not reach the aggregate"


def test_run53_shaped_state_is_success_not_degraded(audit_db):
    """Reproduces run 53: many excluded pending rows, no eligible backlog."""

    for index in range(1, 51):
        _add_headline(audit_db, index, "scored")
    for index in range(51, 249):
        _add_headline(audit_db, index, "pending", excluded=True, scored=False)

    outcome = pipeline._processing_audit(audit_db)
    assert outcome.status == "success"
    assert outcome.details["pending_excluded"] == 198
    assert outcome.details["pending_eligible"] == 0
    assert outcome.details["failed_eligible"] == 0
    assert outcome.details["retry_pending_eligible"] == 0


# -- 2. Corrected-bar provenance ----------------------------------------------

@pytest.mark.parametrize(
    "existing, incoming, explicit, expected_write, expected_status",
    [
        (None, STATUS_PROVISIONAL, False, True, STATUS_PROVISIONAL),
        (None, STATUS_COMPLETE, False, True, STATUS_COMPLETE),
        (STATUS_PROVISIONAL, STATUS_COMPLETE, False, True, STATUS_COMPLETE),
        (STATUS_PROVISIONAL, STATUS_COMPLETE, True, True, STATUS_CORRECTED),
        (STATUS_PROVIDER_INVALID, STATUS_COMPLETE, True, True, STATUS_CORRECTED),
        (STATUS_PROVIDER_INVALID, STATUS_COMPLETE, False, True, STATUS_COMPLETE),
        # corrected is sticky through ordinary refreshes
        (STATUS_CORRECTED, STATUS_COMPLETE, False, True, STATUS_CORRECTED),
        (STATUS_CORRECTED, STATUS_CORRECTED, False, True, STATUS_CORRECTED),
        # corrected must never regress
        (STATUS_CORRECTED, STATUS_PROVISIONAL, False, False, None),
        (STATUS_CORRECTED, STATUS_PROVIDER_INVALID, False, False, None),
        # complete -> corrected only through the explicit repair path
        (STATUS_COMPLETE, STATUS_COMPLETE, False, True, STATUS_COMPLETE),
        (STATUS_COMPLETE, STATUS_COMPLETE, True, True, STATUS_CORRECTED),
        (STATUS_COMPLETE, STATUS_PROVISIONAL, False, False, None),
    ],
)
def test_bar_status_transitions(
    existing, incoming, explicit, expected_write, expected_status
):
    write, status = resolve_bar_status(
        existing, incoming, explicit_correction=explicit
    )
    assert write is expected_write
    assert status == expected_status


def test_corrected_survives_an_identical_settled_refresh(tmp_path):
    """The run-53 regression: a refresh returning the same values kept 'complete'."""

    path = str(tmp_path / "p.db")
    db.init_db(path)
    db.upsert_prices(
        _frame([(FULL_DAY, 1.0, 1.0, 1.0, 99.0, 0.0, None)]),
        db_path=path, observed_at=f"{FULL_DAY}T06:30:00Z",
    )
    db.upsert_prices(
        _frame([(FULL_DAY, 1.0, 1.0, 1.0, 102.5, 7.9e9, None)]),
        db_path=path, observed_at=SETTLED, mark_corrected=True,
    )
    assert db.get_prices(db_path=path).loc[0, "bar_status"] == STATUS_CORRECTED

    # An ordinary later refresh returning the identical settled bar.
    db.upsert_prices(
        _frame([(FULL_DAY, 1.0, 1.0, 1.0, 102.5, 7.9e9, None)]),
        db_path=path, observed_at="2026-08-06T06:30:00Z",
    )
    stored = db.get_prices(db_path=path)
    assert stored.loc[0, "bar_status"] == STATUS_CORRECTED
    assert stored.loc[0, "close"] == 102.5
    assert stored.loc[0, "bar_observed_at"] == "2026-08-06T06:30:00Z"


def test_corrected_retained_when_a_later_refetch_changes_values(tmp_path):
    path = str(tmp_path / "p.db")
    db.init_db(path)
    db.upsert_prices(
        _frame([(FULL_DAY, 1.0, 1.0, 1.0, 99.0, 0.0, None)]),
        db_path=path, observed_at=f"{FULL_DAY}T06:30:00Z",
    )
    db.upsert_prices(
        _frame([(FULL_DAY, 1.0, 1.0, 1.0, 102.5, 7.9e9, None)]),
        db_path=path, observed_at=SETTLED, mark_corrected=True,
    )
    db.upsert_prices(
        _frame([(FULL_DAY, 1.0, 1.0, 1.0, 103.9, 8.1e9, None)]),
        db_path=path, observed_at="2026-08-06T06:30:00Z",
    )
    stored = db.get_prices(db_path=path)
    assert stored.loc[0, "bar_status"] == STATUS_CORRECTED
    assert stored.loc[0, "close"] == 103.9


# -- 3. Fetch-window boundary returns -----------------------------------------

def test_boundary_row_keeps_its_return_across_a_refresh(tmp_path):
    """The 2026-05-04 regression: a window's first row lost a valid return."""

    path = str(tmp_path / "p.db")
    db.init_db(path)
    db.upsert_prices(
        _frame([
            ("2026-07-28", 1.0, 1.0, 1.0, 100.0, 1e9, None),
            ("2026-07-29", 1.0, 1.0, 1.0, 110.0, 1e9, None),
            (FULL_DAY, 1.0, 1.0, 1.0, 121.0, 1e9, None),
        ]),
        db_path=path, observed_at=SETTLED,
    )
    before = db.get_prices(db_path=path).set_index("date")
    assert before.loc["2026-07-29", "daily_return"] == pytest.approx(10.0)

    # A later fetch whose window starts mid-series: the provider hands back a
    # NULL return for its first row.
    db.upsert_prices(
        _frame([
            ("2026-07-29", 1.0, 1.0, 1.0, 110.0, 1e9, None),
            (FULL_DAY, 1.0, 1.0, 1.0, 121.0, 1e9, 10.0),
        ]),
        db_path=path, observed_at="2026-08-06T06:30:00Z",
    )
    after = db.get_prices(db_path=path).set_index("date")
    assert after.loc["2026-07-29", "daily_return"] == pytest.approx(10.0)
    assert after.loc["2026-07-28", "daily_return"] is None or pd.isna(
        after.loc["2026-07-28", "daily_return"]
    ), "the genuinely first session has no predecessor"


def test_new_row_uses_a_predecessor_outside_the_fetch_window(tmp_path):
    path = str(tmp_path / "p.db")
    db.init_db(path)
    db.upsert_prices(
        _frame([("2026-07-29", 1.0, 1.0, 1.0, 100.0, 1e9, None)]),
        db_path=path, observed_at=SETTLED,
    )
    # A window containing only the new session; its predecessor is already stored.
    db.upsert_prices(
        _frame([(FULL_DAY, 1.0, 1.0, 1.0, 105.0, 1e9, None)]),
        db_path=path, observed_at=SETTLED,
    )
    stored = db.get_prices(db_path=path).set_index("date")
    assert stored.loc[FULL_DAY, "daily_return"] == pytest.approx(5.0)


def test_corrected_close_updates_its_own_and_the_following_return(tmp_path):
    path = str(tmp_path / "p.db")
    db.init_db(path)
    db.upsert_prices(
        _frame([
            ("2026-07-28", 1.0, 1.0, 1.0, 100.0, 1e9, None),
            ("2026-07-29", 1.0, 1.0, 1.0, 200.0, 1e9, None),
            (FULL_DAY, 1.0, 1.0, 1.0, 220.0, 1e9, None),
        ]),
        db_path=path, observed_at=SETTLED,
    )
    db.upsert_prices(
        _frame([("2026-07-29", 1.0, 1.0, 1.0, 110.0, 1e9, None)]),
        db_path=path, observed_at=SETTLED, mark_corrected=True,
    )
    stored = db.get_prices(db_path=path).set_index("date")
    assert stored.loc["2026-07-29", "daily_return"] == pytest.approx(10.0)
    assert stored.loc[FULL_DAY, "daily_return"] == pytest.approx(100.0)


def test_provisional_row_never_enters_the_return_series(tmp_path):
    path = str(tmp_path / "p.db")
    db.init_db(path)
    db.upsert_prices(
        _frame([("2026-07-29", 1.0, 1.0, 1.0, 100.0, 1e9, None)]),
        db_path=path, observed_at=SETTLED,
    )
    db.upsert_prices(
        _frame([(FULL_DAY, 1.0, 1.0, 1.0, 150.0, 0.0, 50.0)]),
        db_path=path, observed_at=f"{FULL_DAY}T06:30:00Z",
    )
    everything = db.get_prices(db_path=path, complete_only=False).set_index("date")
    assert everything.loc[FULL_DAY, "bar_status"] == STATUS_PROVISIONAL
    assert pd.isna(everything.loc[FULL_DAY, "daily_return"]), (
        "an unsettled bar has no defensible return"
    )
    assert FULL_DAY not in set(db.get_prices(db_path=path)["date"])


# -- 4. After-close refresh guard ---------------------------------------------

def test_guard_blocks_before_settlement():
    window = after_close_refresh_allowed(f"{FULL_DAY}T15:00:00+03:00")
    assert not window.allowed
    assert window.reason == "before_settlement"


def test_guard_allows_after_settlement():
    window = after_close_refresh_allowed(f"{FULL_DAY}T18:41:00+03:00")
    assert window.allowed
    assert window.reason == "after_settlement"
    assert window.session_date == FULL_DAY


@pytest.mark.parametrize("day", [SATURDAY, HOLIDAY])
def test_guard_no_ops_on_weekends_and_holidays(day):
    window = after_close_refresh_allowed(f"{day}T20:00:00+03:00")
    assert not window.allowed
    assert window.reason == "not_a_trading_day"


def test_guard_uses_the_half_day_close():
    """13:00 close: 14:00 is after settlement, though it would not be on a full day."""

    assert after_close_refresh_allowed(f"{HALF_DAY}T14:00:00+03:00").allowed
    assert not after_close_refresh_allowed(f"{HALF_DAY}T12:00:00+03:00").allowed


def test_guard_converts_utc_to_istanbul():
    """16:10 UTC is 19:10 Istanbul, past the 18:40 settlement."""

    assert after_close_refresh_allowed(f"{FULL_DAY}T16:10:00Z").allowed
    # 13:00 UTC is 16:00 Istanbul, still mid-session.
    assert not after_close_refresh_allowed(f"{FULL_DAY}T13:00:00Z").allowed


def test_check_only_skips_without_touching_the_database(tmp_path):
    from scripts.after_close_refresh import run

    path = str(tmp_path / "p.db")
    db.init_db(path)
    result = run(path, check_only=True, now=f"{FULL_DAY}T09:00:00+03:00")
    assert result["action"] == "skipped"
    assert result["decision"]["reason"] == "before_settlement"


def test_guard_needs_no_database_file():
    """The settlement decision is a clock question, not a data question.

    The workflow runs the guard before restoring the snapshot. Refusing there
    would make the decision depend on a file the guard never reads -- which is
    one of the two faults that broke the scheduled job on 2026-08-06/07.
    """

    from scripts.after_close_refresh import main

    assert main([
        "--db", "no/such/database.db", "--check-only",
        "--now", f"{FULL_DAY}T09:00:00+03:00",
    ]) == 0


def test_guard_imports_nothing_outside_the_standard_library():
    """The other fault: --check-only pulled in pandas through ``database``.

    The guard runs before dependencies are installed, so its import chain must
    stay stdlib-only. Asserted by walking the chain in a subprocess with the
    data stack blocked, rather than by trusting a comment.
    """

    import subprocess
    import sys as _sys

    probe = (
        "import sys\n"
        "for name in ('pandas', 'numpy', 'yfinance', 'requests', 'torch'):\n"
        "    sys.modules[name] = None\n"
        "import scripts.after_close_refresh as m\n"
        "print(m.main(['--db', 'missing.db', '--check-only',\n"
        f"              '--now', '{FULL_DAY}T09:00:00+03:00']))\n"
    )
    completed = subprocess.run(
        [_sys.executable, "-c", probe], cwd=str(REPOSITORY_ROOT),
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("0")


def test_guard_runs_before_dependencies_are_installed():
    """Step order is part of the fix and must not drift back."""

    steps = _workflow("after_close_prices.yml")["jobs"]["refresh"]["steps"]
    names = [str(step.get("name", "")) for step in steps]
    guard = next(i for i, n in enumerate(names) if "guard" in n.lower())
    setup = next(i for i, n in enumerate(names) if "set up python" in n.lower())
    install = next(i for i, n in enumerate(names) if "install" in n.lower())
    assert setup < guard, "the guard needs a pinned interpreter"
    assert guard < install, "the guard must decide before deps are installed"


def test_before_settlement_is_a_no_op(tmp_path, monkeypatch):
    from scripts import after_close_refresh

    def _fail(*args, **kwargs):                       # pragma: no cover
        raise AssertionError("no price fetch may run before settlement")

    monkeypatch.setattr(pipeline, "prices_step", _fail)
    path = str(tmp_path / "p.db")
    db.init_db(path)
    result = after_close_refresh.run(path, now=f"{FULL_DAY}T09:00:00+03:00")
    assert result["action"] == "skipped"


# -- Workflow configuration ----------------------------------------------------

def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_both_workflows_share_one_database_writer_concurrency_group():
    daily = _workflow("daily.yml")
    after = _workflow("after_close_prices.yml")
    assert daily["concurrency"]["group"] == after["concurrency"]["group"]
    assert daily["concurrency"]["cancel-in-progress"] is False
    assert after["concurrency"]["cancel-in-progress"] is False


def test_after_close_workflow_never_scrapes_or_scores():
    """Assert on what the job can actually do, not on prose that mentions it."""

    after = _workflow("after_close_prices.yml")
    job = after["jobs"]["refresh"]

    exposed = dict(after.get("env") or {})
    exposed.update(job.get("env") or {})
    for step in job["steps"]:
        exposed.update(step.get("env") or {})
    assert "OPENAI_API_KEY" not in exposed, "the scorer must not be reachable here"

    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "secrets." not in commands, "no credential is injected into a command"
    assert "main.py run" not in commands, "that entry point scrapes and scores"
    assert "main.py scrape" not in commands
    assert "main.py score" not in commands
    assert "after_close_refresh" in commands


def test_after_close_workflow_guards_before_publishing():
    after = _workflow("after_close_prices.yml")
    names = [step["name"] for step in after["jobs"]["refresh"]["steps"]]
    assert names.index("Guard against publishing a stale snapshot") < names.index(
        "Persist DB to data branch"
    )


def test_after_close_workflow_never_commits_readme_figures():
    text = (WORKFLOWS / "after_close_prices.yml").read_text(encoding="utf-8")
    assert "docs/sample_output.png" not in text


def test_after_close_workflow_runs_a_runtime_guard_before_any_work():
    """Everything that does work must wait for the guard.

    Checking out the repo and setting up an interpreter are what the guard
    needs in order to run at all, so they precede it unconditionally. They are
    exempt by name rather than by position: an ungated step that fetched data,
    installed the stack or wrote the snapshot would still fail this.
    """

    after = _workflow("after_close_prices.yml")
    steps = after["jobs"]["refresh"]["steps"]
    guard = next(s for s in steps if s["name"].startswith("Check the Istanbul"))
    assert "--check-only" in guard["run"]

    prerequisites = {"Checkout repo (main)", "Set up Python", guard["name"]}
    for index, step in enumerate(steps):
        if step["name"] in prerequisites:
            # A prerequisite must actually come first; one appearing after the
            # guard would be doing work under an exemption it does not deserve.
            assert index <= steps.index(guard), step["name"]
            continue
        if step["name"] == "Report skip":
            continue
        assert "steps.guard.outputs.allowed == 'true'" in step.get("if", ""), step["name"]

    # The prerequisites are also the only steps allowed to be that cheap: none
    # of them may touch the database, the network stack or the data branch.
    for step in steps[:steps.index(guard)]:
        body = str(step.get("run", "")) + str(step.get("uses", ""))
        for forbidden in ("finance_sentiment.db", "pip install", "origin/data",
                          "git push"):
            assert forbidden not in body, f"{step['name']} does work before the guard"


def test_validation_is_opt_in_and_never_runs_on_a_schedule():
    """A frozen protocol re-run every morning invites reading the sequence.

    Re-running is legitimate when the corpus has grown enough to change what
    the sample gate permits; running it daily produces near-identical studies
    whose varying verdicts would look like signal.
    """

    steps = _workflow("daily.yml")["jobs"]["run"]["steps"]
    step = next(s for s in steps if "walk-forward" in str(s.get("name", "")).lower())
    condition = step["if"]
    assert "workflow_dispatch" in condition
    assert "inputs.run_validation" in condition

    inputs = _workflow("daily.yml").get("on", _workflow("daily.yml").get(True))
    assert inputs["workflow_dispatch"]["inputs"]["run_validation"]["default"] is False


def test_readiness_runs_daily_but_cannot_break_collection():
    """Future validation must never cost a day of headlines."""

    steps = _workflow("daily.yml")["jobs"]["run"]["steps"]
    step = next(s for s in steps if "readiness" in str(s.get("name", "")).lower())
    assert step.get("continue-on-error") is True
    assert "if" not in step, "readiness is recorded on every run, not opt-in"
    assert "OPENAI_API_KEY" not in str(step.get("env") or {})


def test_readiness_step_runs_after_collection():
    """A readiness snapshot describes the data the run just collected."""

    steps = _workflow("daily.yml")["jobs"]["run"]["steps"]
    names = [str(s.get("name", "")) for s in steps]
    pipeline_index = next(i for i, n in enumerate(names) if n == "Run pipeline")
    readiness = next(i for i, n in enumerate(names) if "readiness" in n.lower())
    assert pipeline_index < readiness


def test_frozen_artifacts_are_verified_on_every_run():
    """A sealed result changing is the one thing that must never be quiet."""

    steps = _workflow("daily.yml")["jobs"]["run"]["steps"]
    step = next(
        s for s in steps if "frozen artifacts" in str(s.get("name", "")).lower()
    )
    assert "--verify-only" in step["run"]
    assert "if" not in step, "artifact verification is unconditional"
    assert step.get("continue-on-error") is not True, (
        "a changed artifact must fail the run loudly"
    )


def test_artifact_verification_runs_after_publishing():
    """Headlines roll off the feeds; a check is equally informative a minute later.

    Blocking the persist step on this would trade unrecoverable data for a
    verification that loses nothing by running immediately afterwards.
    """

    steps = _workflow("daily.yml")["jobs"]["run"]["steps"]
    names = [str(s.get("name", "")) for s in steps]
    persist = next(i for i, n in enumerate(names) if n.startswith("Persist DB"))
    verify = next(i for i, n in enumerate(names) if "frozen artifacts" in n.lower())
    assert persist < verify


def test_validation_step_cannot_score_or_scrape():
    """It reads stored tables; it must not be handed a scoring credential."""

    steps = _workflow("daily.yml")["jobs"]["run"]["steps"]
    step = next(s for s in steps if "walk-forward" in str(s.get("name", "")).lower())
    assert "OPENAI_API_KEY" not in str(step.get("env") or {})
    assert "scrape" not in step["run"] and "main.py run" not in step["run"]


def test_after_close_schedule_is_weekdays_only():
    after = _workflow("after_close_prices.yml")
    trigger = after.get("on", after.get(True))
    cron = trigger["schedule"][0]["cron"]
    assert cron.endswith("1-5"), "weekend runs would only ever no-op"
