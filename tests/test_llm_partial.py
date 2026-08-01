"""Contract tests for partial, non-fabricating LLM responses."""

import json

import pytest
import requests

import sentiment_llm
from config import LLM_SCORING_MAX_ATTEMPTS
from sentiment_llm import LLMSentimentScorer, SCORE_COMPONENTS_KIND


def _body(field, items, model="gpt-5-mini-test"):
    return {
        "model": model,
        "choices": [{"message": {"content": json.dumps({field: items})}}],
    }


def test_score_partial_keeps_omitted_item_absent(monkeypatch):
    scorer = LLMSentimentScorer(batch_size=10)
    monkeypatch.setattr(
        scorer,
        "_request",
        lambda *args, **kwargs: _body(
            "labels",
            [
                {"id": 0, "label": "positive", "strength": 0.7},
                {"id": 2, "label": "negative", "strength": 0.4},
            ],
        ),
    )

    partial = scorer.score_partial(["one", "missing", "three"])
    aligned = scorer.score(["one", "missing", "three"])

    assert set(partial) == {0, 2}
    assert aligned[0][1] == "positive"
    assert aligned[1] is None
    assert aligned[2][1] == "negative"


def test_duplicate_and_out_of_range_score_ids_are_never_fabricated(monkeypatch):
    scorer = LLMSentimentScorer(batch_size=10)
    monkeypatch.setattr(
        scorer,
        "_request",
        lambda *args, **kwargs: _body(
            "labels",
            [
                {"id": 0, "label": "positive", "strength": 0.8},
                {"id": 0, "label": "negative", "strength": 0.8},
                {"id": 1, "label": "neutral", "strength": 0.0},
                {"id": 99, "label": "neutral", "strength": 0.0},
            ],
        ),
    )

    partial = scorer.score_partial(["duplicate", "explicit neutral", "omitted"])

    assert set(partial) == {1}
    assert partial[1] == (0.0, "neutral", 0.0, 1.0, 0.0)
    assert scorer.score(["duplicate", "explicit neutral", "omitted"]) == [
        None,
        (0.0, "neutral", 0.0, 1.0, 0.0),
        None,
    ]


def test_analyze_partial_preserves_only_explicit_valid_results(monkeypatch):
    scorer = LLMSentimentScorer(batch_size=10)
    monkeypatch.setattr(
        scorer,
        "_request",
        lambda *args, **kwargs: _body(
            "analyses",
            [
                {
                    "id": 0,
                    "relevance": 0.9,
                    "category": "turkey_macro",
                    "label": "neutral",
                    "strength": 0.0,
                },
                {
                    "id": 2,
                    "relevance": 0.5,
                    "category": "banks",
                    "label": "positive",
                    "strength": 0.3,
                },
                {
                    "id": 2,
                    "relevance": 0.5,
                    "category": "banks",
                    "label": "negative",
                    "strength": 0.3,
                },
                {
                    "id": -1,
                    "relevance": 1.0,
                    "category": "other",
                    "label": "neutral",
                    "strength": 0.0,
                },
            ],
        ),
    )

    partial = scorer.analyze_partial(["explicit neutral", "omitted", "duplicate"])
    aligned = scorer.analyze(["explicit neutral", "omitted", "duplicate"])

    assert set(partial) == {0}
    assert partial[0]["label"] == "neutral"
    assert partial[0]["score"] == 0.0
    assert partial[0]["score_components_kind"] == SCORE_COMPONENTS_KIND
    assert aligned == [partial[0], None, None]
    assert scorer.score_components_kind == "synthetic_compatibility"
    assert scorer.max_scoring_attempts == LLM_SCORING_MAX_ATTEMPTS


def test_malformed_structured_response_raises_instead_of_filling(monkeypatch):
    scorer = LLMSentimentScorer(batch_size=10)
    monkeypatch.setattr(
        scorer,
        "_request",
        lambda *args, **kwargs: {
            "choices": [{"message": {"content": "not-json"}}]
        },
    )

    with pytest.raises(RuntimeError, match="malformed structured response"):
        scorer.score(["headline"])


def test_http_retry_limit_and_backoff_are_configurable_without_sleep(monkeypatch):
    scorer = LLMSentimentScorer(batch_size=10)
    scorer._api_key = "test-key"
    calls = []
    sleeps = []

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return {"model": "gpt-5-mini-test", "choices": []}

    responses = iter([Response(503), Response(503), Response(200)])

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return next(responses)

    monkeypatch.setattr(sentiment_llm, "LLM_HTTP_RETRY_LIMIT", 3)
    monkeypatch.setattr(sentiment_llm, "LLM_HTTP_RETRY_BACKOFF_SECONDS", 2)
    monkeypatch.setattr(sentiment_llm.requests, "post", fake_post)
    monkeypatch.setattr(sentiment_llm.time, "sleep", sleeps.append)

    result = scorer._request("0. headline")

    assert result["model"] == "gpt-5-mini-test"
    assert len(calls) == 3
    assert sleeps == [2, 4]
