"""
Sentiment scoring with cardiffnlp/twitter-xlm-roberta-base-sentiment.

Model: XLM-RoBERTa fine-tuned on Twitter data in 8 languages including Turkish.
Labels: negative, neutral, positive
Output score: P(positive) - P(negative)  ->  range [-1, 1]

IMPORTANT - lazy imports
------------------------
torch and transformers are intentionally NOT imported at module level.
They are imported inside SentimentScorer._load(), which is called the
first time .score() is invoked. This means importing this module is
instant and never blocks commands like `status`, `scrape`, or `plot`
that have no need for the ML model.
"""

import logging
from typing import List, Optional, Tuple, Dict

from tqdm import tqdm

from config import (
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_MAX_LENGTH,
    SENTIMENT_MODEL,
    SENTIMENT_POSITIVE_THRESHOLD,
    SENTIMENT_NEGATIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# -- Score extraction ----------------------------------------------------------

def _extract_score(result: List[dict]) -> Tuple[float, str, float, float, float]:
    """
    Given the raw output for one text (list of {label, score} dicts),
    return (continuous_score, dominant_label, p_positive, p_neutral, p_negative).

    continuous_score = P(positive) - P(negative)  in [-1.0, 1.0]
    dominant_label   = label with the highest raw probability
    p_positive/neutral/negative = raw softmax probabilities (sum to ~1)
    """
    scores: Dict[str, float] = {}
    for item in result:
        lbl = item["label"].lower().strip()
        if lbl in ("positive", "pos", "label_2", "2"):
            scores["positive"] = item["score"]
        elif lbl in ("negative", "neg", "label_0", "0"):
            scores["negative"] = item["score"]
        elif lbl in ("neutral", "neu", "label_1", "1"):
            scores["neutral"] = item["score"]

    if not scores:
        # A model swap or transformers update changed the label names: failing
        # loudly here prevents silently scoring everything 0.0 / neutral.
        raise ValueError(f"Sentiment model returned unrecognised labels: {result!r}")

    p_pos = scores.get("positive", 0.0)
    p_neg = scores.get("negative", 0.0)
    p_neu = scores.get("neutral",  0.0)
    continuous = p_pos - p_neg
    # Threshold-based label (tuned 2026-06-08 on 198 human labels, accuracy 76.8%).
    # Wider neutral band fixes model's over-calling of negative on routine financial text.
    if continuous > SENTIMENT_POSITIVE_THRESHOLD:
        dominant = "positive"
    elif continuous < SENTIMENT_NEGATIVE_THRESHOLD:
        dominant = "negative"
    else:
        dominant = "neutral"
    return continuous, dominant, p_pos, p_neu, p_neg


# -- Scorer class --------------------------------------------------------------

class SentimentScorer:
    """
    Lazy-loading wrapper around the XLM-RoBERTa sentiment pipeline.

    The model is downloaded and loaded the first time .score() is called.
    Importing this module never triggers a torch import.

    Usage
    -----
    scorer = SentimentScorer()
    scores = scorer.score(["Borsa yukseldi", "Dolar dustu"])
    # -> [(0.72, 'positive'), (-0.41, 'negative')]
    """

    def __init__(
        self,
        model_name: str = SENTIMENT_MODEL,
        batch_size: int = SENTIMENT_BATCH_SIZE,
        max_length: int = SENTIMENT_MAX_LENGTH,
        device: Optional[int] = None,
    ):
        self.model_name  = model_name
        self.batch_size  = batch_size
        self.max_length  = max_length
        # None = auto-detect GPU/CPU when _load() is called.
        # Pass 0 to force GPU:0, -1 to force CPU.
        self._device_override = device
        self._pipe = None  # populated lazily on first .score() call

    # -- Lazy init -------------------------------------------------------------

    def _load(self) -> None:
        """Import torch + transformers and load the model. Called once."""
        if self._pipe is not None:
            return

        # Heavy imports happen HERE - not at module import time.
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline as hf_pipeline,
        )

        if self._device_override is not None:
            device = self._device_override
        else:
            device = 0 if torch.cuda.is_available() else -1

        self.device = device
        device_label = f"GPU:{device}" if device >= 0 else "CPU"

        logger.info("Loading sentiment model %s on %s ...", self.model_name, device_label)
        print(
            f"\nLoading sentiment model ({self.model_name})\n"
            f"    First run: ~1.1 GB download -- subsequent runs use cache.\n"
            f"    Running on: {device_label}\n"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model     = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        self._pipe = hf_pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,           # return ALL label scores
            device=device,
            truncation=True,
            max_length=self.max_length,
        )
        logger.info("Model loaded.")

    # -- Public API ------------------------------------------------------------

    def score(self, texts: List[str]) -> List[Tuple[float, str, float, float, float]]:
        """
        Score a list of texts.

        Returns a list of (score, label, p_positive, p_neutral, p_negative) tuples
        aligned with `texts`.
          score      in [-1.0, 1.0]
          label      in {'positive', 'neutral', 'negative'}
          p_positive / p_neutral / p_negative  are raw softmax probabilities
        """
        if not texts:
            return []

        self._load()
        results: List[Tuple[float, str, float, float, float]] = []

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        for batch in tqdm(batches, desc="Scoring", unit="batch", leave=False):
            raw = self._pipe(batch)  # type: ignore[misc]
            for item in raw:
                results.append(_extract_score(item))

        return results

    def score_df(self, df, text_col: str = "title"):
        """
        Convenience: score a DataFrame in-place, adding
        `sentiment_score`, `sentiment_label`, `p_positive`, `p_neutral`,
        `p_negative` columns.
        """
        import pandas as pd

        texts  = df[text_col].tolist()
        scored = self.score(texts)
        if scored:
            scores, labels, p_pos, p_neu, p_neg = zip(*scored)
        else:
            scores = labels = p_pos = p_neu = p_neg = []
        df = df.copy()
        df["sentiment_score"] = list(scores)
        df["sentiment_label"] = list(labels)
        df["p_positive"]      = list(p_pos)
        df["p_neutral"]       = list(p_neu)
        df["p_negative"]      = list(p_neg)
        return df


# -- Module-level singleton ----------------------------------------------------

_scorer: Optional[SentimentScorer] = None


def get_scorer() -> SentimentScorer:
    """Return a cached SentimentScorer instance (model not loaded until .score() is called)."""
    global _scorer
    if _scorer is None:
        _scorer = SentimentScorer()
    return _scorer
