"""End-to-end contracts for omission handling and component run states."""

from datetime import date

import pandas as pd
import pytest

import database as db
import pipeline


@pytest.fixture
def stage2_db(tmp_path):
    path = str(tmp_path / "stage2.db")
    db.init_db(path)
    return path


def _insert_candidates(path, count=3):
    db.insert_headlines(
        [
            {
                "source": "feed",
                "title": f"BIST aday haber {index}",
                "url": f"https://example.test/candidate/{index}",
                "published_at": "2026-07-06",
                "published_hour": 12,
                "category": "bist_company",
            }
            for index in range(count)
        ],
        db_path=path,
    )
    with db._conn(path) as con:
        return [row[0] for row in con.execute("SELECT id FROM headlines ORDER BY id")]


def _analysis(label, score, relevance=1.0):
    if label == "positive":
        components = (score, 1.0 - score, 0.0)
    elif label == "negative":
        components = (0.0, 1.0 + score, -score)
    else:
        components = (0.0, 1.0, 0.0)
    return {
        "score": score,
        "label": label,
        "p_pos": components[0],
        "p_neu": components[1],
        "p_neg": components[2],
        "category": "bist_company",
        "relevance": relevance,
        "score_components_kind": "synthetic_compatibility",
    }


def test_scrape_partial_source_failure_is_degraded(monkeypatch, stage2_db):
    class PartialRSS:
        def __init__(self, session):
            self.source_status = {"feed-a": "ok", "feed-b": "failed: timeout"}

        def scrape_all(self, since):
            return [{
                "source": "feed-a", "title": "BIST source result",
                "url": "https://example.test/source", "published_at": date.today(),
            }]

    monkeypatch.setattr(pipeline.sc, "_make_session", lambda: object())
    monkeypatch.setattr(pipeline.sc, "RSSFeedScraper", PartialRSS)

    outcome = pipeline.scrape_step(
        db_path=stage2_db, return_outcome=True,
    )

    assert outcome.status == "degraded"
    assert outcome.count == 1
    assert outcome.warnings[0]["code"] == "source_failure"


def test_scrape_all_source_failure_is_failed(monkeypatch, stage2_db):
    class FailedRSS:
        def __init__(self, session):
            self.source_status = {
                "feed-a": "failed: timeout", "feed-b": "failed: HTTP 503",
            }

        def scrape_all(self, since):
            return []

    class EmptyHTML:
        def __init__(self, session):
            pass

        def scrape(self, max_pages):
            return []

    monkeypatch.setattr(pipeline.sc, "_make_session", lambda: object())
    monkeypatch.setattr(pipeline.sc, "RSSFeedScraper", FailedRSS)
    monkeypatch.setattr(pipeline.sc, "InvestingTRScraper", EmptyHTML)

    outcome = pipeline.scrape_step(
        db_path=stage2_db, return_outcome=True,
    )

    assert outcome.status == "failed"
    assert outcome.count == 0
    assert outcome.errors[0]["code"] == "all_sources_failed"


def test_incomplete_batch_retries_only_missing_and_preserves_explicit_neutral(
    stage2_db, monkeypatch
):
    ids = _insert_candidates(stage2_db)

    class PartialScorer:
        model_name = "gpt-test/p3"
        score_components_kind = "synthetic_compatibility"
        max_scoring_attempts = 3

        def __init__(self):
            self.calls = []

        def analyze_partial(self, titles):
            self.calls.append(list(titles))
            if len(self.calls) == 1:
                return {
                    0: _analysis("positive", 0.7),
                    2: _analysis("neutral", 0.0),
                }
            assert titles == ["BIST aday haber 1"]
            return {0: _analysis("negative", -0.4)}

    scorer = PartialScorer()
    monkeypatch.setattr(pipeline, "_get_scorer", lambda: scorer)

    outcome = pipeline.score_step(stage2_db, return_outcome=True)

    assert outcome.status == "success"
    assert outcome.count == 3
    assert [len(call) for call in scorer.calls] == [3, 1]
    with db._conn(stage2_db) as con:
        rows = con.execute(
            """SELECT id, processing_status, scoring_attempts,
                      sentiment_score, sentiment_label
               FROM headlines ORDER BY id"""
        ).fetchall()
    assert [row["id"] for row in rows] == ids
    assert [row["processing_status"] for row in rows] == ["scored"] * 3
    assert [row["scoring_attempts"] for row in rows] == [1, 2, 1]
    assert rows[2]["sentiment_label"] == "neutral"
    assert rows[2]["sentiment_score"] == 0.0


def test_omissions_exhaust_retry_limit_without_becoming_neutral(
    stage2_db, monkeypatch
):
    _insert_candidates(stage2_db, count=2)

    class EmptyScorer:
        model_name = "gpt-test/p3"
        score_components_kind = "synthetic_compatibility"
        max_scoring_attempts = 2

        def __init__(self):
            self.call_sizes = []

        def analyze_partial(self, titles):
            self.call_sizes.append(len(titles))
            return {}

    scorer = EmptyScorer()
    monkeypatch.setattr(pipeline, "_get_scorer", lambda: scorer)

    outcome = pipeline.score_step(stage2_db, return_outcome=True)

    assert outcome.status == "failed"
    assert outcome.count == 0
    assert outcome.errors[0]["code"] == "scoring_unavailable"
    assert scorer.call_sizes == [2, 2]
    with db._conn(stage2_db) as con:
        rows = con.execute(
            """SELECT processing_status, scoring_attempts, sentiment_score,
                      sentiment_label, p_positive, p_neutral, p_negative
               FROM headlines ORDER BY id"""
        ).fetchall()
    assert [row["processing_status"] for row in rows] == ["failed", "failed"]
    assert [row["scoring_attempts"] for row in rows] == [2, 2]
    assert all(
        all(row[field] is None for field in (
            "sentiment_score", "sentiment_label", "p_positive", "p_neutral", "p_negative"
        ))
        for row in rows
    )


def test_failed_and_excluded_rows_are_omitted_from_aggregation(stage2_db):
    ids = _insert_candidates(stage2_db)
    db.batch_update_sentiment(
        [
            (0.0, "neutral", 0.0, 1.0, 0.0, "gpt-test/p3",
             "synthetic_compatibility", ids[0]),
            (0.9, "positive", 0.9, 0.1, 0.0, "gpt-test/p3",
             "synthetic_compatibility", ids[2]),
        ],
        db_path=stage2_db,
    )
    db.mark_scoring_attempts_failed([ids[1]], "omitted", 1, db_path=stage2_db)
    db.exclude_headline(ids[2], "test exclusion", "test", "v1", stage2_db)

    assert pipeline.aggregate_step(stage2_db) == 1
    daily = db.get_daily_sentiment(db_path=stage2_db)

    assert len(daily) == 1
    assert daily.iloc[0]["headline_count"] == 1
    assert daily.iloc[0]["neutral_count"] == 1
    assert daily.iloc[0]["avg_score"] == 0.0


def test_restored_llm_relevance_exclusion_survives_aggregate_reconciliation(stage2_db):
    headline_id = _insert_candidates(stage2_db, count=1)[0]
    db.batch_update_sentiment(
        [(
            0.4, "positive", 0.6, 0.3, 0.1, "gpt-test/p3",
            "synthetic_compatibility", headline_id,
        )],
        db_path=stage2_db,
    )
    db.update_relevance([(0.1, headline_id)], stage2_db)
    assert db.reconcile_relevance_exclusions([headline_id], stage2_db)["excluded"] == 1

    assert pipeline.restore_exclusion_step(headline_id, stage2_db)
    assert db.count_active_headline_exclusions(stage2_db) == 0
    variants = db.get_signal_variants(db_path=stage2_db)
    assert len(variants) == 1
    assert variants.iloc[0]["headline_count"] == 1
    history = db.list_headline_exclusions(
        stage2_db, active_only=False, headline_id=headline_id,
    )
    assert history[-1]["restored_by_user"] == 1

    # A newly stored relevance judgment supersedes the explicit override and
    # lets the current rule make a fresh, auditable decision.
    db.update_relevance([(0.1, headline_id)], stage2_db)
    assert db.reconcile_relevance_exclusions([headline_id], stage2_db)["excluded"] == 1


def test_market_download_failure_distinguishes_fresh_cache_from_stale(stage2_db):
    stale = pipeline._market_data_fallback_outcome(stage2_db, "offline")
    assert stale.status == "failed"
    assert stale.errors[0]["code"] == "market_data_stale"

    today = date.today().isoformat()
    prices = pd.DataFrame([{
        "date": today, "open": 1.0, "high": 1.0, "low": 1.0,
        "close": 1.0, "volume": 1.0, "daily_return": None,
    }])
    db.upsert_prices(prices, db_path=stage2_db)
    fresh = pipeline._market_data_fallback_outcome(stage2_db, "offline")
    assert fresh.status == "degraded"
    assert fresh.warnings[0]["code"] == "fresh_cache_used"


def test_fx_absence_is_skipped_but_configured_provider_failure_degrades(
    stage2_db, monkeypatch
):
    skipped = pipeline.fx_rates_step(
        api_key="", db_path=stage2_db, return_outcome=True,
    )
    assert skipped.status == "skipped"

    import requests

    def fail_request(*args, **kwargs):
        raise RuntimeError("provider offline")

    monkeypatch.setattr(requests, "get", fail_request)
    degraded = pipeline.fx_rates_step(
        api_key="configured-test-key", db_path=stage2_db, return_outcome=True,
    )
    assert degraded.status == "degraded"
    assert degraded.warnings[0]["code"] == "fx_provider_failure"


def test_run_all_persists_degraded_component_states(stage2_db, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "fx_rates_step",
        lambda **kwargs: pipeline.StepOutcome(status="skipped"),
    )
    monkeypatch.setattr(
        pipeline,
        "factors_step",
        lambda **kwargs: pipeline.StepOutcome(
            status="degraded",
            warnings=[pipeline._issue(
                "market_data", "external_factor_failure", "one factor unavailable"
            )],
        ),
    )

    result = pipeline.run_all(
        db_path=stage2_db,
        show_plot=False,
        skip_scrape=True,
        skip_score=True,
        skip_aggregate=True,
        skip_prices=True,
        skip_plot=True,
    )

    assert result["status"] == "degraded"
    assert result["components"]["market_data"] == "degraded"
    run = db.get_pipeline_run(result["run_id"], stage2_db)
    assert run["status"] == "degraded"
    assert run["market_data_status"] == "degraded"
    assert run["audit_status"] == "success"
    assert run["warnings"][0]["code"] == "external_factor_failure"


def test_run_all_fails_when_every_scoring_candidate_exhausts(stage2_db, monkeypatch):
    _insert_candidates(stage2_db, count=2)

    class EmptyScorer:
        model_name = "gpt-test/p3"
        score_components_kind = "synthetic_compatibility"
        max_scoring_attempts = 1

        def analyze_partial(self, titles):
            return {}

    monkeypatch.setattr(pipeline, "_get_scorer", lambda: EmptyScorer())

    with pytest.raises(RuntimeError, match="unavailable for every candidate"):
        pipeline.run_all(
            db_path=stage2_db,
            show_plot=False,
            skip_scrape=True,
            skip_aggregate=True,
            skip_prices=True,
            skip_plot=True,
        )

    with db._conn(stage2_db) as con:
        run_id = con.execute("SELECT MAX(run_id) FROM pipeline_runs").fetchone()[0]
    run = db.get_pipeline_run(run_id, stage2_db)
    assert run["status"] == "failed"
    assert run["scoring_status"] == "failed"
    assert any(issue["code"] == "scoring_unavailable" for issue in run["errors"])
