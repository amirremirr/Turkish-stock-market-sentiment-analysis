"""
LLM-based sentiment scoring (OpenAI gpt-5-mini) — drop-in alternative to the
XLM-RoBERTa scorer in sentiment.py.

Why this exists: benchmarked on 2026-06-12 against 198 human-labeled headlines,
gpt-5-mini with 30 few-shot examples scored 84.5% on held-out data vs 76.8% for
XLM-R with thresholds tuned in-sample. See benchmark_llm.py for the methodology.

Interface contract (same as sentiment.SentimentScorer):
    scorer.score(texts) -> list of (score, label, p_pos, p_neu, p_neg) tuples
    scorer.model_name   -> stored in headlines.model_name for provenance

Score/label consistency: the LLM returns a label plus a strength in [0, 1].
We derive  score = +strength | -strength | 0.0  so that the stored continuous
score ALWAYS agrees with the label under the +-0.05 config thresholds, and the
confidence-weighted daily aggregation keeps working unchanged. Pseudo-probs are
derived so that p_pos - p_neg == score, which keeps `main.py relabel` coherent.

Requires OPENAI_API_KEY (env var or .env). Fails loudly if missing — silent
fallback to a different scorer would mix two scoring regimes in the DB.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from config import (
    LLM_SENTIMENT_MODEL,
    LLM_SENTIMENT_BATCH_SIZE,
    SENTIMENT_POSITIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)

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
        logger.warning("fewshot_examples.json not found — running zero-shot "
                       "(benchmarked 82.3%% vs 84.5%% with examples)")
    return prompt


def _to_tuple(label: str, strength: float) -> Tuple[float, str, float, float, float]:
    """
    Convert (label, strength) to the scorer tuple contract.

    Guarantees: label agrees with score under the config thresholds, and
    p_pos - p_neg == score (so relabel_from_probs reproduces the label).
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
    return 0.0, "neutral", 0.0, 1.0, 0.0


class LLMSentimentScorer:
    """Batch sentiment scorer backed by the OpenAI API."""

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
        for attempt in range(4):
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
                wait = 15 * (attempt + 1)
                logger.warning("LLM scorer: %s — retrying in %ds", last_err, wait)
                time.sleep(wait)
                continue
            if resp.status_code in (429, 500, 503):
                last_err = f"HTTP {resp.status_code}"
                wait = 15 * (attempt + 1)
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
        raise RuntimeError(f"LLM scorer: still failing after 4 attempts (last: {last_err})")

    def score(self, texts: List[str]) -> List[Tuple[float, str, float, float, float]]:
        """Score a list of headlines. Returns tuples aligned with `texts`."""
        if not texts:
            return []

        results: dict[int, Tuple[float, str, float, float, float]] = {}
        batches = [list(enumerate(texts))[i:i + self.batch_size]
                   for i in range(0, len(texts), self.batch_size)]

        for n, batch in enumerate(batches, 1):
            listing = "\n".join(f"{idx}. {title}" for idx, title in batch)
            body = self._request(listing)
            text = body["choices"][0]["message"]["content"]
            for item in json.loads(text)["labels"]:
                idx = int(item["id"])
                if 0 <= idx < len(texts):
                    results[idx] = _to_tuple(item["label"], item.get("strength", 0.5))
            logger.info("LLM scorer: batch %d/%d done (%d/%d scored)",
                        n, len(batches), len(results), len(texts))

        # Any headline the model skipped gets a neutral default (rare; logged).
        missing = [i for i in range(len(texts)) if i not in results]
        if missing:
            logger.warning("LLM scorer: %d headline(s) missing from response — "
                           "defaulting to neutral 0.0: %s", len(missing), missing[:10])
            for i in missing:
                results[i] = (0.0, "neutral", 0.0, 1.0, 0.0)

        return [results[i] for i in range(len(texts))]


    def analyze(self, texts: List[str]) -> List[dict]:
        """
        Combined sentiment + category + relevance analysis in one API call
        per batch. Returns dicts aligned with `texts`:
            {score, label, p_pos, p_neu, p_neg, category, relevance}
        relevance is a 0.0-1.0 grade — NOTHING is deleted on its basis; the
        aggregation weights low-relevance headlines toward zero instead.
        Used by the scoring step (full analysis of new headlines) and by
        `main.py recategorize --llm` (category/relevance refresh).
        """
        if not texts:
            return []

        system_prompt = self._system_prompt + _ANALYZE_PROMPT_EXTRA
        results: dict[int, dict] = {}
        batches = [list(enumerate(texts))[i:i + self.batch_size]
                   for i in range(0, len(texts), self.batch_size)]

        for n, batch in enumerate(batches, 1):
            listing = "\n".join(f"{idx}. {title}" for idx, title in batch)
            body = self._request(listing, system_prompt=system_prompt,
                                 schema=_ANALYZE_SCHEMA, schema_name="headline_analyses")
            text = body["choices"][0]["message"]["content"]
            for item in json.loads(text)["analyses"]:
                idx = int(item["id"])
                if 0 <= idx < len(texts):
                    score, label, p_pos, p_neu, p_neg = _to_tuple(
                        item["label"], item.get("strength", 0.5))
                    category = item["category"] if item["category"] in CATEGORIES else "other"
                    results[idx] = {
                        "score": score, "label": label,
                        "p_pos": p_pos, "p_neu": p_neu, "p_neg": p_neg,
                        "category": category,
                        "relevance": max(0.0, min(1.0, float(item["relevance"]))),
                    }
            logger.info("LLM analyze: batch %d/%d done (%d/%d)",
                        n, len(batches), len(results), len(texts))

        missing = [i for i in range(len(texts)) if i not in results]
        if missing:
            logger.warning("LLM analyze: %d headline(s) missing — defaulting to "
                           "neutral/other/relevance 1.0: %s", len(missing), missing[:10])
            for i in missing:
                results[i] = {"score": 0.0, "label": "neutral", "p_pos": 0.0,
                              "p_neu": 1.0, "p_neg": 0.0, "category": "other",
                              "relevance": 1.0}

        return [results[i] for i in range(len(texts))]


# -- Module-level singleton (mirrors sentiment.get_scorer) -----------------------

_scorer: Optional[LLMSentimentScorer] = None


def get_scorer() -> LLMSentimentScorer:
    global _scorer
    if _scorer is None:
        _scorer = LLMSentimentScorer()
    return _scorer
