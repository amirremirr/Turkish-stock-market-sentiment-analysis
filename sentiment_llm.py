"""
LLM-based sentiment scoring (OpenAI gpt-5-mini) — drop-in alternative to the
XLM-RoBERTa scorer in sentiment.py.

Why this exists: held-out scorer comparisons led to the current p3 prompt. On
the canonical 270-row held-out set, p3 reached 83.3% categorical agreement with
the project's human-label rubric. This is rubric agreement, not objective truth,
calibration, or predictive evidence. See benchmark_llm.py and LABELING.md.

Interface contract (same as sentiment.SentimentScorer):
    scorer.score(texts) -> aligned list of score tuples or None for omissions
    scorer.score_partial(texts) -> {input_index: score tuple}
    scorer.analyze_partial(texts) -> {input_index: analysis dict}
    scorer.model_name   -> stored in headlines.model_name for provenance

Score/label consistency: the LLM returns a label plus a strength in [0, 1].
We derive  score = +strength | -strength | 0.0  so that the stored continuous
score ALWAYS agrees with the label under the +-0.05 config thresholds, and the
legacy component/storage contract remains available for sensitivity variants. Synthetic
compatibility components are derived so p_pos - p_neg == score; they are not
calibrated probabilities or estimates of correctness.

Requires OPENAI_API_KEY (env var or .env). Fails loudly if missing — silent
fallback to a different scorer would mix two scoring regimes in the DB.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import requests

from config import (
    LLM_HTTP_RETRY_BACKOFF_SECONDS,
    LLM_HTTP_RETRY_LIMIT,
    LLM_SCORING_MAX_ATTEMPTS,
    LLM_SENTIMENT_MODEL,
    LLM_SENTIMENT_BATCH_SIZE,
    SENTIMENT_POSITIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Stored alongside LLM results so downstream code cannot mistake the derived
# p_* compatibility fields for calibrated class probabilities.
SCORE_COMPONENTS_KIND = "synthetic_compatibility"

_T = TypeVar("_T")

# Bump whenever _SYSTEM_PROMPT_BASE, _ANALYZE_PROMPT_EXTRA, or the few-shot set
# changes — stored with every scored row (model_name column) so results are
# attributable to an exact prompt version. History: p1 = launch prompt
# (2026-06-12 morning), p2 = graded relevance (2026-06-12 evening),
# p3 = recalibrated to the LABELING.md conventions after the 300-label set
# (2026-06-13) — neutral-default + the analyst's documented judgment calls.
PROMPT_VERSION = "p3"

_FEWSHOT_PATH = Path(__file__).parent / "fewshot_examples.json"

_SYSTEM_PROMPT_BASE = """\
You are a financial-news sentiment classifier for the Turkish stock market (BIST 100).

You will receive a numbered list of Turkish financial news headlines. For each one,
classify the sentiment a Turkish equity investor would read into it:

- "positive": good news for the Turkish economy or market mood (growth, exports up,
  rate cuts hoped for, records, deals, upgrades, strong earnings)
- "negative": bad news for the Turkish economy or market mood (inflation up, lira
  weakness, downgrades, crises, bankruptcies, sanctions, political instability)
- "neutral": routine reporting with no clear directional read (announcements of data
  without surprise, schedules, mixed/balanced reports, factual price listings)

Judge market-relevant sentiment, not emotional tone. "Reserves fell slightly as
expected" is neutral routine reporting, not negative. A record harvest is positive
even if phrased dryly.

NEUTRAL IS THE DEFAULT. Assign positive/negative only when a Turkish equity
investor's mood would clearly move. Most routine reporting is neutral.

Conventions our analyst follows — match them exactly:
- Judge through Turkey's lens. Turkey imports nearly all its energy: oil/gas
  prices falling = positive; rising = negative. US-specific inventory or
  production statistics = neutral.
- Gold/silver/copper price moves = neutral unless explicitly tied to the lira
  or to crisis flight.
- Foreign-economy data (German PMI, Eurozone forecasts, other countries'
  currencies) = neutral — UNLESS a clear global risk event that hits all
  emerging markets. A surprise Fed/ECB hike ANNOUNCEMENT = negative; hike
  previews ("bekleniyor") and currency-reaction stories = neutral.
- Rate-HIKE expectations (TCMB or Fed) = negative (easing deferred);
  rate-CUT expectations = positive.
- Rising FX deposits / dollarization = negative for TL sentiment.
- Ministerial PR, ribbon-cuttings, and speeches without new policy = neutral.
- Intra-party political turmoil (congress calls, internal resignations) =
  neutral; arrests or probes of major political figures (mayors, opposition
  leaders) = negative.
- Foreign investment interest in Turkey = positive.
- Company-level news counts: bankruptcies/fines/disclosed problems = negative;
  records/major contracts = positive — regardless of company size.
- Genuinely ambiguous after brief consideration = neutral.

For each headline also give a "strength" between 0.1 and 1.0 expressing how strong
and unambiguous the sentiment is (use 0.0 for neutral): a dramatic crisis headline
is ~0.9, a mildly encouraging data point is ~0.2.

Return a JSON object with a "labels" array containing one entry per headline,
in the same order, each with the headline's "id", your "label", and "strength".
"""

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                    "strength": {"type": "number"},
                },
                "required": ["id", "label", "strength"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

# -- Combined analysis (sentiment + category + relevance) ------------------------

CATEGORIES = ["bist_company", "rates_tcmb", "political_risk", "turkey_macro",
              "crypto", "global_risk", "fx_lira", "banks", "energy_commodities",
              "other"]

_ANALYZE_PROMPT_EXTRA = """

Additionally, for each headline decide:

"relevance" — how relevant is this headline to Turkish financial markets and the
Turkish economy, as a number between 0.0 and 1.0:
- 1.0:  directly about Turkish markets, economy, or policy (BIST, TCMB, lira,
        inflation, Turkish companies, Turkish trade)
- 0.7:  global financial / commodity / geopolitical news with clear implications
        for Turkish markets (Fed, ECB, oil prices, wars, EU economy)
- 0.4:  business or economy news with only an indirect or weak connection
        (foreign company stories, distant markets)
- 0.1:  barely related (tech curiosities, lifestyle stories with a money angle)
- 0.0:  unrelated (celebrity, sports, prayer times, ordinary crime, lottery,
        holiday greetings, tourism listicles)
When in doubt, grade higher rather than lower.

"category" — exactly one of:
- "bist_company":       Borsa Istanbul, listed companies, IPOs, earnings, KAP disclosures
- "rates_tcmb":         central bank (TCMB), interest rates, bonds, treasury, monetary policy
- "political_risk":     market-moving political events (arrests, resignations, elections, crises)
- "turkey_macro":       Turkish economy data — inflation, growth, trade, employment, tourism revenue
- "crypto":             cryptocurrency
- "global_risk":        global markets, Fed/ECB, geopolitics, wars, sanctions, credit ratings
- "fx_lira":            currency / exchange rates / lira
- "banks":              banking sector, loans, deposits
- "energy_commodities": oil, gas, gold, metals, agriculture, electricity
- "other":              relevant to the economy but fits none of the above
"""

_ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "relevance": {"type": "number"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "label": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                    "strength": {"type": "number"},
                },
                "required": ["id", "relevance", "category", "label", "strength"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["analyses"],
    "additionalProperties": False,
}


def _build_system_prompt() -> str:
    """Base prompt + the 30 benchmark-validated few-shot examples."""
    prompt = _SYSTEM_PROMPT_BASE
    try:
        examples = json.loads(_FEWSHOT_PATH.read_text(encoding="utf-8"))
        lines = "\n".join(f'- "{e["title"]}" -> {e["label"]}' for e in examples)
        prompt += ("\nHere are examples labeled by our analyst — match their "
                   "labeling style and judgment:\n\n" + lines + "\n")
    except FileNotFoundError:
        logger.warning(
            "fewshot_examples.json not found — running outside the documented "
            "p3 few-shot configuration; results are not benchmark-comparable"
        )
    return prompt


def _to_tuple(label: str, strength: float) -> Tuple[float, str, float, float, float]:
    """
    Convert (label, strength) to the scorer tuple contract.

    Guarantees: label agrees with score under the config thresholds, and
    p_pos - p_neg == score (so the compatibility relabel path reproduces it).

    The three derived components are synthetic compatibility values. They are
    not calibrated class probabilities or model-confidence estimates.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if label == "positive":
        # Strength must clear the positive threshold or the label would be
        # inconsistent with the stored score.
        score = max(strength, SENTIMENT_POSITIVE_THRESHOLD + 0.01)
        return score, "positive", score, 1.0 - score, 0.0
    if label == "negative":
        score = -max(strength, SENTIMENT_POSITIVE_THRESHOLD + 0.01)
        return score, "negative", 0.0, 1.0 + score, -score
    if label == "neutral":
        return 0.0, "neutral", 0.0, 1.0, 0.0
    raise ValueError(f"unsupported sentiment label: {label!r}")


def _validated_partial(
    items: List[dict],
    expected_ids: set[int],
    converter: Callable[[dict], _T],
    operation: str,
) -> Dict[int, _T]:
    """Return only unambiguous, valid results for the current request batch.

    An omitted ID remains absent. A duplicate invalidates that ID entirely, and
    an ID outside the current batch is ignored. This deliberately makes an
    incomplete response visible to the caller instead of manufacturing a
    neutral observation.
    """
    results: Dict[int, _T] = {}
    invalid_ids: set[int] = set()

    for item in items:
        try:
            raw_id = item["id"]
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise ValueError("IDs must be JSON integers")
            idx = raw_id
        except (KeyError, TypeError, ValueError):
            logger.warning("LLM %s: ignored result with invalid or missing id", operation)
            continue

        if idx not in expected_ids:
            logger.warning("LLM %s: ignored out-of-range id %s", operation, idx)
            continue

        if idx in results or idx in invalid_ids:
            results.pop(idx, None)
            invalid_ids.add(idx)
            logger.warning("LLM %s: duplicate id %s invalidated", operation, idx)
            continue

        try:
            results[idx] = converter(item)
        except (KeyError, TypeError, ValueError) as exc:
            invalid_ids.add(idx)
            logger.warning("LLM %s: invalid result for id %s (%s)", operation, idx, exc)

    return results


class LLMSentimentScorer:
    """Batch sentiment scorer backed by the OpenAI API."""

    score_components_kind = SCORE_COMPONENTS_KIND
    max_scoring_attempts = LLM_SCORING_MAX_ATTEMPTS

    def __init__(self, model: str = LLM_SENTIMENT_MODEL,
                 batch_size: int = LLM_SENTIMENT_BATCH_SIZE):
        self.model = model            # what we send to the API — never mutated
        self.model_name = model       # provenance string stored in the DB;
                                      # locked to "<api-snapshot>/<prompt-ver>"
                                      # after the first successful response
        self.batch_size = batch_size
        self._system_prompt = _build_system_prompt()
        self._api_key: Optional[str] = None

    def _key(self) -> str:
        if self._api_key is None:
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set (env var or .env). The LLM scorer "
                    "fails loudly rather than silently mixing scoring backends — "
                    "set the key, or set SENTIMENT_BACKEND='xlmr' in config.py."
                )
            self._api_key = key
        return self._api_key

    def _request(self, listing: str, system_prompt: Optional[str] = None,
                 schema: Optional[dict] = None, schema_name: str = "sentiment_labels") -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or self._system_prompt},
                {"role": "user", "content": listing},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True,
                                "schema": schema or _JSON_SCHEMA},
            },
            "max_completion_tokens": 8000,
        }
        if self.model.startswith("gpt-5"):
            payload["reasoning_effort"] = "low"

        last_err = None
        attempt_limit = max(1, int(LLM_HTTP_RETRY_LIMIT))
        for attempt in range(attempt_limit):
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self._key()}"},
                    json=payload,
                    timeout=180,
                )
            except requests.RequestException as exc:
                # Dropped connections / timeouts are as transient as a 503.
                last_err = f"{type(exc).__name__}"
                if attempt + 1 < attempt_limit:
                    wait = LLM_HTTP_RETRY_BACKOFF_SECONDS * (attempt + 1)
                    logger.warning("LLM scorer: %s — retrying in %ds", last_err, wait)
                    time.sleep(wait)
                continue
            if resp.status_code in (429, 500, 503):
                last_err = f"HTTP {resp.status_code}"
                if attempt + 1 < attempt_limit:
                    wait = LLM_HTTP_RETRY_BACKOFF_SECONDS * (attempt + 1)
                    logger.warning("LLM scorer: transient %s — retrying in %ds", last_err, wait)
                    time.sleep(wait)
                continue
            resp.raise_for_status()
            body = resp.json()
            # Provenance: lock model_name to the API's dated snapshot + prompt
            # version on first successful call (e.g. "gpt-5-mini-2025-08-07/p2").
            api_model = body.get("model")
            if api_model and "/" not in self.model_name:
                self.model_name = f"{api_model}/{PROMPT_VERSION}"
            return body
        raise RuntimeError(
            f"LLM scorer: still failing after {attempt_limit} attempts (last: {last_err})"
        )

    @staticmethod
    def _response_array(body: dict, field: str, operation: str) -> List[dict]:
        """Extract a structured-output array or fail without inventing rows."""
        try:
            content = body["choices"][0]["message"]["content"]
            items = json.loads(content)[field]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM {operation}: malformed structured response") from exc
        if not isinstance(items, list):
            raise RuntimeError(f"LLM {operation}: response field {field!r} is not an array")
        return items

    def score_partial(
        self, texts: List[str]
    ) -> Dict[int, Tuple[float, str, float, float, float]]:
        """Return valid scores keyed by input index; missing rows stay absent."""
        results: Dict[int, Tuple[float, str, float, float, float]] = {}
        indexed = list(enumerate(texts))
        batches = [indexed[i:i + self.batch_size]
                   for i in range(0, len(indexed), self.batch_size)]

        for n, batch in enumerate(batches, 1):
            listing = "\n".join(f"{idx}. {title}" for idx, title in batch)
            body = self._request(listing)
            items = self._response_array(body, "labels", "score")
            expected = {idx for idx, _ in batch}
            parsed = _validated_partial(
                items,
                expected,
                lambda item: _to_tuple(item["label"], item["strength"]),
                "score",
            )
            results.update(parsed)
            missing = sorted(expected - parsed.keys())
            if missing:
                logger.warning(
                    "LLM scorer: batch %d/%d omitted or invalidated %d item(s): %s",
                    n, len(batches), len(missing), missing[:10],
                )
            logger.info("LLM scorer: batch %d/%d done (%d/%d scored)",
                        n, len(batches), len(results), len(texts))

        return results

    def score(
        self, texts: List[str]
    ) -> List[Optional[Tuple[float, str, float, float, float]]]:
        """Return an aligned list, using ``None`` for every missing result."""
        partial = self.score_partial(texts)
        return [partial.get(i) for i in range(len(texts))]

    def analyze_partial(self, texts: List[str]) -> Dict[int, dict]:
        """
        Combined sentiment + category + relevance analysis in one API call
        per batch. Returns valid dicts keyed by input index:
            {score, label, p_pos, p_neu, p_neg, category, relevance}
        Missing, duplicate, malformed, and out-of-range results are not present.
        """
        system_prompt = self._system_prompt + _ANALYZE_PROMPT_EXTRA
        results: Dict[int, dict] = {}
        indexed = list(enumerate(texts))
        batches = [indexed[i:i + self.batch_size]
                   for i in range(0, len(indexed), self.batch_size)]

        def convert(item: dict) -> dict:
            if item["category"] not in CATEGORIES:
                raise ValueError(f"unsupported category: {item['category']!r}")
            score, label, p_pos, p_neu, p_neg = _to_tuple(
                item["label"], item["strength"])
            return {
                "score": score, "label": label,
                "p_pos": p_pos, "p_neu": p_neu, "p_neg": p_neg,
                "category": item["category"],
                "relevance": max(0.0, min(1.0, float(item["relevance"]))),
                "score_components_kind": SCORE_COMPONENTS_KIND,
            }

        for n, batch in enumerate(batches, 1):
            listing = "\n".join(f"{idx}. {title}" for idx, title in batch)
            body = self._request(listing, system_prompt=system_prompt,
                                 schema=_ANALYZE_SCHEMA, schema_name="headline_analyses")
            items = self._response_array(body, "analyses", "analyze")
            expected = {idx for idx, _ in batch}
            parsed = _validated_partial(items, expected, convert, "analyze")
            results.update(parsed)
            missing = sorted(expected - parsed.keys())
            if missing:
                logger.warning(
                    "LLM analyze: batch %d/%d omitted or invalidated %d item(s): %s",
                    n, len(batches), len(missing), missing[:10],
                )
            logger.info("LLM analyze: batch %d/%d done (%d/%d)",
                        n, len(batches), len(results), len(texts))

        return results

    def analyze(self, texts: List[str]) -> List[Optional[dict]]:
        """Return aligned analyses, using ``None`` for every missing result."""
        partial = self.analyze_partial(texts)
        return [partial.get(i) for i in range(len(texts))]


# -- Module-level singleton (mirrors sentiment.get_scorer) -----------------------

_scorer: Optional[LLMSentimentScorer] = None


def get_scorer() -> LLMSentimentScorer:
    global _scorer
    if _scorer is None:
        _scorer = LLMSentimentScorer()
    return _scorer
