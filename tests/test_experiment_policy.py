"""Focused contracts for score provenance and mixed-experiment aggregation."""

import pytest

import database as db
import main
import pipeline


@pytest.fixture
def experiment_db(tmp_path):
    path = str(tmp_path / "experiments.db")
    db.init_db(path)
    return path


def _insert_headlines(path):
    db.insert_headlines(
        [
            {
                "source": "feed-a",
                "title": "BIST experiment A",
                "url": "https://example.test/experiment/a",
                "published_at": "2026-07-06",
                "published_hour": 10,
                "category": "bist_company",
            },
            {
                "source": "feed-b",
                "title": "BIST experiment B",
                "url": "https://example.test/experiment/b",
                "published_at": "2026-07-06",
                "published_hour": 11,
                "category": "bist_company",
            },
        ],
        db_path=path,
    )
    with db._conn(path) as con:
        return [row[0] for row in con.execute("SELECT id FROM headlines ORDER BY id")]


def _score(path, headline_id, score, label, experiment_id):
    if label == "positive":
        components = (0.8, 0.2, 0.0)
    else:
        components = (0.0, 0.2, 0.8)
    db.batch_update_sentiment(
        [(
            score,
            label,
            *components,
            "gpt-experiment/p3",
            "synthetic_compatibility",
            headline_id,
        )],
        db_path=path,
        experiment_id=experiment_id,
    )


def _seed_mixed_experiments(path):
    first_id, second_id = _insert_headlines(path)
    _score(path, first_id, 0.6, "positive", "experiment-a")
    pipeline.aggregate_step(path)
    _score(path, second_id, -0.4, "negative", "experiment-b")
    return first_id, second_id


def _variant_rows(path):
    with db._conn(path) as con:
        return [
            tuple(row)
            for row in con.execute(
                "SELECT * FROM daily_signal_variants ORDER BY signal_date"
            )
        ]


def test_default_policy_blocks_before_replacing_existing_aggregates(experiment_db):
    _seed_mixed_experiments(experiment_db)
    before = _variant_rows(experiment_db)

    assert db.get_eligible_experiment_ids(experiment_db) == [
        "experiment-a",
        "experiment-b",
    ]
    with pytest.raises(
        pipeline.MixedExperimentAggregationError,
        match="experiment-a, experiment-b",
    ):
        pipeline.aggregate_step(experiment_db)

    assert _variant_rows(experiment_db) == before


def test_explicit_override_aggregates_and_returns_degraded_state(experiment_db):
    _seed_mixed_experiments(experiment_db)

    outcome = pipeline.aggregate_step(
        experiment_db,
        allow_mixed_experiments=True,
        return_outcome=True,
    )

    assert outcome.status == "degraded"
    assert outcome.count == 1
    assert outcome.details == {
        "eligible_experiment_ids": ["experiment-a", "experiment-b"],
        "mixed_experiments": True,
        "mixed_experiments_override": True,
    }
    assert outcome.warnings[0]["code"] == "mixed_experiments_allowed"
    assert outcome.warnings[0]["details"]["experiment_ids"] == [
        "experiment-a",
        "experiment-b",
    ]
    assert db.get_signal_variants(db_path=experiment_db).iloc[0]["headline_count"] == 2


def test_excluded_experiment_is_not_eligible_for_aggregation(experiment_db):
    _, second_id = _seed_mixed_experiments(experiment_db)
    assert db.exclude_headline(
        second_id,
        "experiment isolation test",
        "manual",
        "v1",
        experiment_db,
    )

    assert db.get_eligible_experiment_ids(experiment_db) == ["experiment-a"]
    outcome = pipeline.aggregate_step(experiment_db, return_outcome=True)
    assert outcome.status == "success"
    assert outcome.details["mixed_experiments"] is False
    assert db.get_signal_variants(db_path=experiment_db).iloc[0]["headline_count"] == 1


def test_run_all_persists_default_failure_and_override_degradation(
    experiment_db, monkeypatch,
):
    _seed_mixed_experiments(experiment_db)

    with pytest.raises(pipeline.MixedExperimentAggregationError):
        pipeline.run_all(
            db_path=experiment_db,
            show_plot=False,
            skip_scrape=True,
            skip_score=True,
            skip_prices=True,
            skip_plot=True,
        )

    with db._conn(experiment_db) as con:
        failed_run_id = con.execute(
            "SELECT MAX(run_id) FROM pipeline_runs"
        ).fetchone()[0]
    failed_run = db.get_pipeline_run(failed_run_id, experiment_db)
    assert failed_run["status"] == "failed"
    assert failed_run["aggregation_status"] == "failed"
    assert failed_run["market_data_status"] == "skipped"
    assert failed_run["audit_status"] == "skipped"
    blocked = next(
        issue for issue in failed_run["errors"]
        if issue["code"] == "mixed_experiments_blocked"
    )
    assert blocked["details"]["experiment_ids"] == [
        "experiment-a",
        "experiment-b",
    ]

    monkeypatch.setattr(
        pipeline,
        "fx_rates_step",
        lambda **kwargs: pipeline.StepOutcome(status="skipped"),
    )
    monkeypatch.setattr(
        pipeline,
        "factors_step",
        lambda **kwargs: pipeline.StepOutcome(status="skipped"),
    )
    result = pipeline.run_all(
        db_path=experiment_db,
        show_plot=False,
        skip_scrape=True,
        skip_score=True,
        skip_prices=True,
        skip_plot=True,
        allow_mixed_experiments=True,
    )

    assert result["status"] == "degraded"
    assert result["components"]["aggregation"] == "degraded"
    persisted = db.get_pipeline_run(result["run_id"], experiment_db)
    assert persisted["status"] == "degraded"
    assert persisted["aggregation_status"] == "degraded"
    assert persisted["warnings"][0]["code"] == "mixed_experiments_allowed"


def test_cli_requires_explicit_mixed_experiment_flag():
    parser = main._build_parser()

    default_args = parser.parse_args(["aggregate"])
    override_args = parser.parse_args(["run", "--allow-mixed-experiments"])

    assert default_args.allow_mixed_experiments is False
    assert override_args.allow_mixed_experiments is True
