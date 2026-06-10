"""
Benchmark an LLM (OpenAI or Gemini) against the human-labeled headline set.

Scores the same headlines that humans labeled, then reports:
  - LLM accuracy vs human labels (the number that matters)
  - XLM-RoBERTa accuracy vs human labels (current baseline, same rows)
  - Confusion matrix + per-category breakdown
  - Sample disagreements for manual review

This is a DECISION tool: if the LLM doesn't clearly beat the 76.8% XLM-R
baseline here, don't switch the production scorer.

Usage:
    # keys in .env:  OPENAI_API_KEY=sk-...  and/or  GEMINI_API_KEY=...
    python benchmark_llm.py <labels.csv> [--provider openai|gemini]
                            [--fewshot N] [--save]

Providers:
    openai  gpt-5-mini (falls back to gpt-4.1-mini / gpt-4o-mini) — paid,
            ~$0.02 per full run, no meaningful rate limits at this scale.
    gemini  gemini-2.5-flash — free tier, but capped at 20 requests/DAY.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

# Load .env if python-dotenv is available (same convention as config.py)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODELS = ["gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini"]  # first available wins
BATCH_SIZE = 50
GEMINI_SECONDS_BETWEEN_CALLS = 30   # free tier: 20 requests/day, aggressive throttling

SYSTEM_PROMPT = """\
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
even if phrased dryly. If a headline is genuinely ambiguous, choose neutral.

Return a JSON object with a "labels" array containing one entry per headline,
in the same order, each with the headline's "id" and your "label".
"""

FEWSHOT_HEADER = """

Here are examples labeled by our analyst — match their labeling style and judgment:

"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                },
                "required": ["id", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

# Gemini uses an OpenAPI-style schema variant (uppercase types, no additionalProperties)
GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "labels": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "label": {"type": "STRING", "enum": ["positive", "neutral", "negative"]},
                },
                "required": ["id", "label"],
            },
        }
    },
    "required": ["labels"],
}


# -- Providers -----------------------------------------------------------------

def pick_openai_model(api_key: str) -> str:
    """Return the first preferred model this key can access."""
    resp = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    available = {m["id"] for m in resp.json()["data"]}
    for model in OPENAI_MODELS:
        if model in available:
            return model
    raise RuntimeError(f"None of {OPENAI_MODELS} available on this key.")


def classify_batch_openai(api_key: str, model: str, batch: pd.DataFrame,
                          system_prompt: str) -> dict[int, str]:
    listing = "\n".join(f"{int(r.id)}. {r.title}" for r in batch.itertuples(index=False))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": listing},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "sentiment_labels", "strict": True, "schema": JSON_SCHEMA},
        },
        "max_completion_tokens": 8000,
    }
    if model.startswith("gpt-5"):
        payload["reasoning_effort"] = "low"   # classification needs little reasoning
    for attempt in range(3):
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=180,
        )
        if resp.status_code in (429, 500, 503):
            wait = 10 * (attempt + 1)
            print(f"    transient {resp.status_code} — waiting {wait}s ...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(text)
        return {int(item["id"]): item["label"] for item in data["labels"]}
    raise RuntimeError("Still failing after 3 attempts.")


def classify_batch_gemini(api_key: str, batch: pd.DataFrame,
                          system_prompt: str) -> dict[int, str]:
    listing = "\n".join(f"{int(r.id)}. {r.title}" for r in batch.itertuples(index=False))
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": listing}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCHEMA,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    for attempt in range(5):
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            params={"key": api_key},
            json=payload,
            timeout=120,
        )
        if resp.status_code in (429, 503):
            wait = 45 * (attempt + 1)
            print(f"    transient {resp.status_code} — waiting {wait}s ...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)
        return {int(item["id"]): item["label"] for item in data["labels"]}
    raise RuntimeError("Still failing after 5 attempts — free-tier quota likely exhausted.")


# -- Main ------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an LLM vs human labels")
    parser.add_argument("labels_csv", help="CSV with id, title, human_label, model_label columns")
    parser.add_argument("--provider", choices=["openai", "gemini"], default=None,
                        help="Default: openai if OPENAI_API_KEY is set, else gemini")
    parser.add_argument("--fewshot", type=int, default=0,
                        help="Use N labeled examples in the prompt (stratified; "
                             "those rows are excluded from evaluation)")
    parser.add_argument("--save", action="store_true",
                        help="Save per-row results to benchmark_llm_results.csv")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    provider = args.provider or ("openai" if openai_key else "gemini")

    if provider == "openai":
        if not openai_key:
            print("OPENAI_API_KEY is not set (env var or .env line).")
            sys.exit(1)
        model_used = pick_openai_model(openai_key)
        pacing = 0
    else:
        if not gemini_key:
            print("GEMINI_API_KEY is not set (env var or .env line).")
            sys.exit(1)
        model_used = GEMINI_MODEL
        pacing = GEMINI_SECONDS_BETWEEN_CALLS

    df = pd.read_csv(args.labels_csv)
    df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
    df = df[df["human_label"].isin(["positive", "neutral", "negative"])].copy()

    # -- Few-shot: pull stratified examples into the prompt, evaluate on the rest
    system_prompt = SYSTEM_PROMPT
    if args.fewshot:
        per_label = max(1, args.fewshot // 3)
        examples = pd.concat([
            df[df["human_label"] == lbl].sample(min(per_label, (df["human_label"] == lbl).sum()),
                                                random_state=42)
            for lbl in ["positive", "neutral", "negative"]
        ])
        df = df.drop(examples.index)
        example_lines = "\n".join(
            f'- "{r.title}" -> {r.human_label}' for r in examples.itertuples(index=False)
        )
        system_prompt = SYSTEM_PROMPT + FEWSHOT_HEADER + example_lines + "\n"
        print(f"Few-shot: {len(examples)} examples in prompt, "
              f"evaluating on the remaining {len(df)} headlines")

    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Benchmarking {len(df)} human-labeled headlines with {model_used} ({provider})\n")

    # -- Score in batches -------------------------------------------------------
    predictions: dict[int, str] = {}
    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    for i, batch in enumerate(batches, 1):
        try:
            if provider == "openai":
                predictions.update(classify_batch_openai(openai_key, model_used, batch, system_prompt))
            else:
                predictions.update(classify_batch_gemini(gemini_key, batch, system_prompt))
            print(f"  batch {i}/{len(batches)} done ({len(predictions)}/{len(df)})")
        except requests.HTTPError as exc:
            print(f"  batch {i} failed: {exc.response.status_code} {exc.response.text[:300]}")
            sys.exit(1)
        if pacing and i < len(batches):
            time.sleep(pacing)

    df["llm_label"] = df["id"].map(predictions)
    missing = df["llm_label"].isna().sum()
    if missing:
        print(f"  [!] {missing} headlines got no prediction — excluded from scoring")
        df = df.dropna(subset=["llm_label"])

    # -- Headline numbers ---------------------------------------------------------
    # model_label in the CSV is XLM-R's raw argmax. The production scorer applies
    # the tuned +-0.05 thresholds to model_score — recompute that here so the
    # baseline matches what actually runs in the pipeline.
    from config import SENTIMENT_POSITIVE_THRESHOLD as POS, SENTIMENT_NEGATIVE_THRESHOLD as NEG
    df["xlmr_tuned"] = df["model_score"].apply(
        lambda s: "positive" if s > POS else ("negative" if s < NEG else "neutral")
    )

    llm_acc        = (df["llm_label"]   == df["human_label"]).mean()
    xlmr_tuned_acc = (df["xlmr_tuned"]  == df["human_label"]).mean()
    xlmr_raw_acc   = (df["model_label"] == df["human_label"]).mean()

    print()
    print("=" * 58)
    print(f"  {model_used}" + (f"  ({args.fewshot}-shot)" if args.fewshot else "  (zero-shot)"))
    print(f"      accuracy vs human labels:   {llm_acc:.1%}   (n={len(df)})")
    print(f"  XLM-RoBERTa, tuned thresholds (production config)")
    print(f"      accuracy vs human labels:   {xlmr_tuned_acc:.1%}")
    print(f"  XLM-RoBERTa, raw argmax (for reference)")
    print(f"      accuracy vs human labels:   {xlmr_raw_acc:.1%}")
    print(f"  Delta vs production baseline: {llm_acc - xlmr_tuned_acc:+.1%}")
    print("=" * 58)

    # -- Confusion matrix -----------------------------------------------------------
    order = ["positive", "neutral", "negative"]
    print(f"\n  Confusion matrix (rows = human truth, cols = {model_used}):")
    print(f"  {'':>10} " + "".join(f"{c:>10}" for c in order))
    for truth in order:
        row = df[df["human_label"] == truth]
        counts = [int((row["llm_label"] == p).sum()) for p in order]
        print(f"  {truth:>10} " + "".join(f"{c:>10}" for c in counts))

    # -- Per-category accuracy ----------------------------------------------------------
    if "category" in df.columns:
        print(f"\n  Per-category accuracy (LLM vs XLM-R tuned):")
        for cat, grp in sorted(df.groupby("category"), key=lambda kv: len(kv[1]), reverse=True):
            g = (grp["llm_label"]  == grp["human_label"]).mean()
            x = (grp["xlmr_tuned"] == grp["human_label"]).mean()
            print(f"    {cat:<22} LLM {g:>6.1%}   XLM-R {x:>6.1%}   (n={len(grp)})")

    # -- Disagreement samples ---------------------------------------------------------------
    wrong = df[df["llm_label"] != df["human_label"]]
    print(f"\n  LLM disagreements with humans: {len(wrong)}  (showing up to 10)")
    for r in wrong.head(10).itertuples(index=False):
        title = str(r.title)[:70].encode("ascii", "replace").decode()
        print(f"    human={r.human_label:<8} llm={r.llm_label:<8} {title}")

    if args.save:
        out = "benchmark_llm_results.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n  Per-row results saved -> {out}")

    print("\n  Decision guide:")
    print("    LLM >= 85%        -> switch the scorer; clear win")
    print("    LLM 78-85%        -> switch only if the errors look more sensible")
    print("    LLM <= XLM-R      -> keep XLM-R; revisit labeling guidelines")


if __name__ == "__main__":
    main()
