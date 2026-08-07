"""Freezing a result, and sealing a future test.

Every property here protects against the same failure mode: a study quietly
becoming a different, more favourable study. A frozen artifact that can be
edited, a boundary that can move backwards, or a readiness report that leaks an
accuracy would each let "we tried again and it worked" be presented as "it
worked", and none of them would show up as an error.
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
from research.frozen_result import (
    RETROSPECTIVE_CONCLUSION, artifact_hash, build_artifact, summary,
    verify_artifact,
)
from research.future_validation import (
    EPOCH_RETROSPECTIVE, EPOCH_UNTOUCHED, FIRST_ELIGIBLE_SESSION,
    MINIMUM_SESSIONS, corpus_epoch, definition, definition_hash, partition,
)
from research.protocol import protocol_hash


def _document():
    return {
        "specification": {
            "protocol_version": "walk-forward-protocol-v1",
            "status": "retrospective_walk_forward_exploration",
            "success_criteria": {"success": "s", "failure": "f"},
            "decision_thresholds": {"alpha": 0.05},
        },
        "protocol_hash": "a" * 64,
        "provenance": {
            "code_commit": "c" * 40, "database_snapshot": "d" * 64,
            "dataset_version": "ds", "feature_version": "fv",
            "target_version": "tv", "modelling_unit_version": "mu",
            "timing_rule_version": "tr", "return_window_version": "rw",
        },
    }


def _specifications():
    return [
        {
            "feature_set": "none", "model": "mean", "target": "raw_return",
            "kind": "baseline", "status": "fitted",
            "gate": {"binding_requirement": None, "rows_complete": 50,
                     "usable_sessions": 50, "missing_by_column": {}},
            "stability": {"fitted_folds": 3},
            "pooled": {"mae": 1.0, "rmse": 1.2, "pearson_r": 0.0,
                       "directional_accuracy": 0.5, "balanced_accuracy": 0.5,
                       "directional_hit_interval": {"lower": 0.3, "upper": 0.7}},
            "predictions": [
                {"fold": 1, "first_reactable_session": "2026-07-01",
                 "exit_date": "2026-07-01", "actual": 1.0, "predicted": 0.5,
                 "probability": None},
            ],
        },
        {
            "feature_set": "abnormal_tone", "model": "ridge",
            "target": "raw_return", "kind": "news", "status": "insufficient_sample",
            "gate": {"binding_requirement": "11 usable rows < 25",
                     "rows_complete": 11, "usable_sessions": 11,
                     "missing_by_column": {"abnormal_tone": 39}},
            "stability": {},
            "pooled": None,
            "predictions": [],
        },
    ]


def _artifact(**overrides):
    payload = {
        "protocol_document": _document(),
        "specifications": _specifications(),
        "comparison": {"verdict": "failure", "specifications_run": 1,
                       "specifications_blocked": 1, "successes": 0,
                       "comparisons": [], "multiplicity_note": "note"},
        "counts": {"distinct_sessions": 50, "event_rows": 773,
                   "distinct_outcomes": 50, "duplication_factor": 15.46},
        "folds": [{"fold": 1, "train_sessions": 24, "test_sessions": 6}],
        "frozen_at": "2026-08-08T00:00:00+00:00",
    }
    payload.update(overrides)
    return build_artifact(**payload)


# ---------------------------------------------------------------------------
class TestFrozenArtifact:
    def test_conclusion_is_stored_verbatim(self):
        artifact = _artifact()
        assert artifact["conclusion"] == RETROSPECTIVE_CONCLUSION
        assert verify_artifact(artifact)["conclusion_verbatim"] is True

    def test_conclusion_is_not_regenerated_from_the_numbers(self):
        """A formatter bug must not be able to reword a finding."""

        strong = _artifact(comparison={
            "verdict": "success", "specifications_run": 9,
            "specifications_blocked": 0, "successes": 4, "comparisons": [],
            "multiplicity_note": "n",
        })
        assert strong["conclusion"] == RETROSPECTIVE_CONCLUSION

    def test_hash_covers_the_content(self):
        artifact = _artifact()
        assert verify_artifact(artifact)["intact"] is True

        tampered = json.loads(json.dumps(artifact))
        tampered["verdict"] = "success"
        assert verify_artifact(tampered)["intact"] is False

    def test_hash_ignores_the_freeze_timestamp(self):
        """Re-checking a study must prove identity, not mint a new hash."""

        first = _artifact(frozen_at="2026-08-08T00:00:00+00:00")
        second = _artifact(frozen_at="2027-01-01T12:00:00+00:00")
        assert first["artifact_hash"] == second["artifact_hash"]

    def test_every_prediction_and_specification_is_retained(self):
        artifact = _artifact()
        assert len(artifact["specification_results"]) == 2
        assert len(artifact["out_of_sample_predictions"]) == 1
        blocked = next(
            r for r in artifact["specification_results"]
            if r["status"] == "insufficient_sample"
        )
        assert blocked["binding_requirement"] == "11 usable rows < 25"
        assert blocked["missing_by_column"] == {"abnormal_tone": 39}

    def test_summary_reports_sample_and_verdict(self):
        brief = summary(_artifact())
        assert brief["independent_sessions"] == 50
        assert brief["duplication_factor"] == 15.46
        assert brief["successful_news_specifications"] == 0
        assert brief["verdict"] == "failure"
        assert brief["conclusion"] == RETROSPECTIVE_CONCLUSION

    def test_storing_is_idempotent_by_content(self, tmp_path):
        path = str(tmp_path / "frozen.db")
        db.init_db(db_path=path)
        artifact = _artifact()

        first = db.freeze_research_result(artifact, db_path=path)
        second = db.freeze_research_result(artifact, db_path=path)
        assert first["already_frozen"] is False
        assert second["already_frozen"] is True
        assert len(db.list_frozen_results(db_path=path)) == 1

    def test_a_revised_protocol_cannot_reuse_the_frozen_version_name(self, tmp_path):
        """"We re-ran it and it worked" must not be able to become "it worked"."""

        path = str(tmp_path / "frozen.db")
        db.init_db(db_path=path)
        db.freeze_research_result(_artifact(), db_path=path)

        revised = _artifact(protocol_document={
            **_document(), "protocol_hash": "b" * 64,
        })
        with pytest.raises(ValueError, match="already frozen under protocol hash"):
            db.freeze_research_result(revised, db_path=path)
        assert len(db.list_frozen_results(db_path=path)) == 1

    def test_a_mismatched_hash_is_refused(self, tmp_path):
        path = str(tmp_path / "frozen.db")
        db.init_db(db_path=path)
        artifact = {**_artifact(), "artifact_hash": "0" * 64}
        with pytest.raises(ValueError, match="does not match"):
            db.freeze_research_result(artifact, db_path=path)

    def test_stored_artifact_cannot_be_edited_or_removed(self, tmp_path):
        path = str(tmp_path / "frozen.db")
        db.init_db(db_path=path)
        db.freeze_research_result(_artifact(), db_path=path)

        with sqlite3.connect(path) as connection:
            for statement in (
                "UPDATE frozen_research_results SET verdict = 'success'",
                "DELETE FROM frozen_research_results",
            ):
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    connection.execute(statement)


# ---------------------------------------------------------------------------
class TestUntouchedBoundary:
    def test_boundary_labels_both_sides(self):
        assert corpus_epoch("2026-07-01") == EPOCH_RETROSPECTIVE
        assert corpus_epoch(FIRST_ELIGIBLE_SESSION) == EPOCH_UNTOUCHED
        assert corpus_epoch("2027-01-01") == EPOCH_UNTOUCHED

    def test_the_boundary_is_inclusive_of_its_first_session(self):
        assert corpus_epoch(FIRST_ELIGIBLE_SESSION) == EPOCH_UNTOUCHED

    def test_missing_session_is_never_counted_as_untouched(self):
        """An unknown date must not be able to sneak into the sealed sample."""

        assert corpus_epoch(None) == EPOCH_RETROSPECTIVE
        assert corpus_epoch("") == EPOCH_RETROSPECTIVE

    def test_partition_never_pools_the_two_epochs(self):
        rows = [
            {"first_reactable_session": "2026-06-01"},
            {"first_reactable_session": "2026-12-01"},
        ]
        grouped = partition(rows)
        assert len(grouped[EPOCH_RETROSPECTIVE]) == 1
        assert len(grouped[EPOCH_UNTOUCHED]) == 1
        assert not (
            {id(r) for r in grouped[EPOCH_RETROSPECTIVE]}
            & {id(r) for r in grouped[EPOCH_UNTOUCHED]}
        )

    def test_definition_starts_after_the_retrospective_study(self):
        contract = definition(protocol_hash=protocol_hash())
        assert contract["first_eligible_session"] >= contract["validation_start"][:10]

    def test_definition_seals_every_design_decision(self):
        sealed = definition(protocol_hash=protocol_hash())["sealed"]
        for item in ("feature design", "feature selection", "model selection",
                     "hyperparameters", "target choice", "decision thresholds",
                     "success criteria"):
            assert item in sealed

    def test_sample_requirement_is_derived_from_the_fold_geometry(self):
        contract = definition(protocol_hash=protocol_hash())
        requirement = contract["sample_size_requirement"]
        geometry = contract["fold_geometry"]
        assert requirement["minimum_sessions"] == MINIMUM_SESSIONS
        assert requirement["minimum_sessions"] >= (
            geometry["initial_train_sessions"] + geometry["test_sessions"]
        )

    def test_definition_hash_changes_with_the_boundary(self):
        base = definition(protocol_hash=protocol_hash())
        moved = {**base, "first_eligible_session": "2026-01-01"}
        assert definition_hash(base) != definition_hash(moved)

    def test_registration_is_idempotent_and_immutable(self, tmp_path):
        path = str(tmp_path / "future.db")
        db.init_db(db_path=path)
        contract = definition(protocol_hash=protocol_hash())

        first = db.register_future_validation(contract, db_path=path)
        second = db.register_future_validation(contract, db_path=path)
        assert first["already_registered"] is False
        assert second["already_registered"] is True

        with sqlite3.connect(path) as connection:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(
                    "UPDATE future_validation_definitions "
                    "SET first_eligible_session = '2020-01-01'"
                )


# ---------------------------------------------------------------------------
class TestReadinessIsSealed:
    def test_readiness_rejects_outcome_statistics(self, tmp_path):
        """The table must not be usable to watch performance accumulate."""

        path = str(tmp_path / "readiness.db")
        db.init_db(db_path=path)
        base = {
            "observed_at": "2026-09-01T00:00:00+00:00",
            "definition_hash": "h", "state": "accumulating",
            "untouched_sessions": 3, "required_sessions": 51,
            "eligible_events": 9, "distinct_outcomes": 3,
            "elapsed_days": 20, "required_days": 120,
        }
        assert db.record_future_readiness(base, db_path=path) == 1

        for leak in ("mae", "directional_accuracy", "pearson_r", "verdict"):
            with pytest.raises(ValueError, match="outcome statistics"):
                db.record_future_readiness({**base, leak: 0.61}, db_path=path)

    def test_report_carries_no_performance_field(self, tmp_path):
        from scripts.future_readiness import build_report

        path = str(tmp_path / "empty.db")
        db.init_db(db_path=path)
        report = build_report(path, now="2026-09-01T00:00:00+00:00")

        forbidden = {
            "mae", "rmse", "accuracy", "directional_accuracy",
            "balanced_accuracy", "pearson_r", "brier_score", "verdict",
        }
        assert not forbidden & set(report)
        assert report["eligible_to_run"] is False
        assert report["blocking_reasons"]

    def test_a_not_yet_open_window_can_never_satisfy_the_horizon(self, tmp_path):
        from scripts.future_readiness import build_report

        path = str(tmp_path / "early.db")
        db.init_db(db_path=path)
        report = build_report(path, now="2026-01-01T00:00:00+00:00")
        assert report["elapsed_days"] == 0
        assert any("has not started" in r for r in report["blocking_reasons"])

    def test_readiness_counts_only_untouched_rows(self, tmp_path):
        from scripts.future_readiness import build_report

        path = str(tmp_path / "mixed.db")
        db.init_db(db_path=path)
        with sqlite3.connect(path) as connection:
            for session, epoch in (("2026-06-01", "retrospective"),
                                   ("2027-01-04", "untouched_future")):
                connection.execute(
                    """INSERT INTO event_research_dataset
                       (group_key, algorithm_version, experiment_id, window_name,
                        first_reactable_session, signal_date, eligibility_status,
                        is_tradable_window, timing_conflict, raw_return,
                        signal_family, corpus_epoch, dataset_version, updated_at)
                       VALUES (?, 'a', 'e', 'reactable_open_to_close', ?, ?,
                               'eligible', 1, 0, 1.5, 'monetary_policy', ?,
                               'v', 'now')""",
                    (f"g-{session}", session, session, epoch),
                )
        report = build_report(path, now="2027-02-01T00:00:00+00:00")
        assert report["untouched_sessions"] == 1
        assert report["eligible_events"] == 1
        assert report["family_coverage"] == {"monetary_policy": 1}


# ---------------------------------------------------------------------------
class TestControlsUseTheFullHistory:
    def test_estimation_series_is_strictly_prior(self):
        from research.controls import compute_residual_returns

        panel = {
            f"2026-01-{d:02d}": {"EEM_lag1": (d % 5) - 2.0}
            for d in range(1, 29)
        }
        series = [(f"2026-01-{d:02d}", (d % 7) - 3.0) for d in range(1, 29)]
        wanted = [series[-1]]

        results = compute_residual_returns(
            wanted, panel, "em_lagged",
            estimation_window=60, min_observations=5,
            estimation_series=series,
        )
        record = results[series[-1][0]]
        assert record["estimation_observations"] >= 5
        # The described date must never contribute to its own coefficients.
        assert record["estimation_window_end"] < series[-1][0]

    def test_wider_estimation_series_raises_coverage(self):
        from research.controls import compute_residual_returns

        panel = {
            f"2026-01-{d:02d}": {"EEM_lag1": (d % 5) - 2.0}
            for d in range(1, 29)
        }
        full = [(f"2026-01-{d:02d}", (d % 7) - 3.0) for d in range(1, 29)]
        sparse = full[-3:]

        narrow = compute_residual_returns(
            sparse, panel, "em_lagged", min_observations=5,
        )
        wide = compute_residual_returns(
            sparse, panel, "em_lagged", min_observations=5,
            estimation_series=full,
        )
        assert all(r["residual"] is None for r in narrow.values())
        assert any(r["residual"] is not None for r in wide.values())

    def test_minimum_observation_rule_is_not_weakened(self):
        from research.controls import DEFAULT_MIN_OBSERVATIONS

        assert DEFAULT_MIN_OBSERVATIONS == 30

    def test_market_return_series_covers_sessions_without_events(self):
        from research.return_windows import (
            PRIMARY_WINDOW, PriceSeries, market_return_series,
        )

        bars = [
            {"date": "2026-06-08", "open": 100.0, "close": 101.0,
             "bar_status": "complete"},
            {"date": "2026-06-09", "open": 101.0, "close": 102.0,
             "bar_status": "complete"},
            {"date": "2026-06-10", "open": 102.0, "close": 101.0,
             "bar_status": "provisional"},
        ]
        series = market_return_series(PriceSeries(bars))
        dates = [day for day, _ in series[PRIMARY_WINDOW]]
        assert dates == ["2026-06-08", "2026-06-09"]
        assert "2026-06-10" not in dates, "a provisional bar must not enter a beta"


# ---------------------------------------------------------------------------
class TestEventReviewSample:
    def test_draw_is_deterministic(self, tmp_path):
        from scripts.event_review_sample import draw_sample

        path = _review_fixture(tmp_path)
        assert (
            [r["group_key"] for r in draw_sample(path, per_stratum=3)]
            == [r["group_key"] for r in draw_sample(path, per_stratum=3)]
        )

    def test_draw_never_reads_market_data(self, tmp_path, monkeypatch):
        """A reviewer nudged by the outcome is not reviewing the grouping.

        Asserted on the SQL actually executed rather than on the module text:
        the docstring necessarily mentions returns in order to say it never
        reads them, and a text scan cannot tell those two apart.
        """

        from scripts.event_review_sample import draw_sample

        path = _review_fixture(tmp_path)
        executed: list = []

        class _Recording(sqlite3.Connection):
            """sqlite3.Connection.execute is read-only, so subclass instead."""

            def execute(self, sql, *rest):           # noqa: D102
                executed.append(str(sql))
                return super().execute(sql, *rest)

        real_connect = sqlite3.connect

        def _recording_connect(*args, **kwargs):
            kwargs.setdefault("factory", _Recording)
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", _recording_connect)
        draw_sample(path, per_stratum=3)

        assert executed, "the draw ran no queries at all"
        combined = " ".join(executed).lower()
        for forbidden in ("raw_return", "residual", "bist100_prices",
                          "event_return_windows", "event_research_dataset",
                          "market_factors"):
            assert forbidden not in combined, (
                f"the review draw queried {forbidden}"
            )

    def test_sheet_has_the_four_verdicts_and_blank_columns(self, tmp_path):
        from scripts.event_review_sample import (
            REVIEW_VERDICTS, draw_sample, write_sheet,
        )
        import csv

        path = _review_fixture(tmp_path)
        target = write_sheet(draw_sample(path, per_stratum=2),
                             tmp_path / "sheet.csv")
        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for row in rows:
            assert row["verdict"] == ""
            assert row["reviewer"] == ""
        assert set(REVIEW_VERDICTS) == {
            "correct_group", "false_merge", "missed_merge", "uncertain",
        }

    def test_sample_is_stored_and_rebuildable(self, tmp_path):
        from scripts.event_review_sample import SAMPLE_VERSION, draw_sample

        path = _review_fixture(tmp_path)
        rows = draw_sample(path, per_stratum=2)
        first = db.replace_event_review_sample(
            rows, sample_version=SAMPLE_VERSION, db_path=path)
        second = db.replace_event_review_sample(
            rows, sample_version=SAMPLE_VERSION, db_path=path)
        assert first == second == len(rows)
        with sqlite3.connect(path) as connection:
            stored = connection.execute(
                "SELECT COUNT(*) FROM event_review_sample"
            ).fetchone()[0]
        assert stored == len({
            (r["stratum"], r["group_key"], r.get("comparison_group_key"))
            for r in rows
        })


def _review_fixture(tmp_path) -> str:
    """A small corpus with at least one group in most strata."""

    path = str(tmp_path / "review.db")
    db.init_db(db_path=path)
    with sqlite3.connect(path) as connection:
        for index in range(1, 13):
            connection.execute(
                """INSERT INTO headlines (id, source, title, published_at,
                        signal_date, timing_bucket, scraped_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (index, f"outlet{index % 3}", f"TCMB faiz karari {index}",
                 "2026-06-08", "2026-06-09", "pre_open", "now"),
            )
        for index in range(1, 7):
            multi = index % 2 == 0
            connection.execute(
                """INSERT INTO event_groups
                   (group_key, algorithm_version, signal_family, event_type,
                    primary_entity, headline_count, source_count, sources,
                    first_reactable_session, signal_date, signal_date_span,
                    is_singleton, is_single_source, updated_at)
                   VALUES (?, 'v1', 'monetary_policy', 'rate_decision', 'TCMB',
                           ?, ?, 'a,b', '2026-06-09', '2026-06-09', ?, ?, ?,
                           'now')""",
                (f"v1:{index}", 2 if multi else 1, 2 if multi else 1,
                 2 if index == 3 else 1, 0 if multi else 1, 0 if multi else 1),
            )
            connection.execute(
                """INSERT INTO event_headline_map
                   (group_key, algorithm_version, headline_id, similarity,
                    match_rule, assigned_at)
                   VALUES (?, 'v1', ?, 1.0, 'seed', 'now')""",
                (f"v1:{index}", index),
            )
            if multi:
                connection.execute(
                    """INSERT INTO event_headline_map
                       (group_key, algorithm_version, headline_id, similarity,
                        match_rule, assigned_at)
                       VALUES (?, 'v1', ?, 0.85, 'entity+family+time+title',
                               'now')""",
                    (f"v1:{index}", index + 6),
                )
    return path
