"""
Robustness check for the media-slant finding (limitation L3: LLM scorer bias).

Re-scores a sample of the pro-gov and opposition headlines with a SECOND,
independent model (Gemini) using the same production prompt, then compares the
pro-gov vs opposition sentiment gap it finds to the one gpt-5-mini found on the
SAME headlines. If both independent models see the gap, it's in the text — not
one model's quirk.

Gemini free tier = ~20 requests/day, so this samples ~150 per camp (6 calls).

Usage:  GEMINI_API_KEY in .env, then  python replicate_slant.py
"""

import json
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy import stats

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from sentiment_llm import _build_system_prompt, _to_tuple
from polarization_analysis import PRO_GOV, OPPOSITION

GEMINI_MODEL = "gemini-2.5-flash"
SAMPLE_PER_CAMP = 150
BATCH = 50
PACING = 13

_SCHEMA = {
    "type": "OBJECT",
    "properties": {"labels": {"type": "ARRAY", "items": {"type": "OBJECT",
        "properties": {"id": {"type": "INTEGER"},
                       "label": {"type": "STRING", "enum": ["positive", "neutral", "negative"]},
                       "strength": {"type": "NUMBER"}},
        "required": ["id", "label", "strength"]}}},
    "required": ["labels"],
}


def gemini_score(api_key, titles, system_prompt):
    """Return {index: continuous_score} for a list of titles."""
    out = {}
    batches = [list(enumerate(titles))[i:i + BATCH] for i in range(0, len(titles), BATCH)]
    for bn, batch in enumerate(batches, 1):
        listing = "\n".join(f"{i}. {t}" for i, t in batch)
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": listing}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseSchema": _SCHEMA, "thinkingConfig": {"thinkingBudget": 0}},
        }
        for attempt in range(5):
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": api_key}, json=payload, timeout=120)
            if r.status_code in (429, 503):
                wait = 45 * (attempt + 1)
                print(f"    transient {r.status_code}, waiting {wait}s ...")
                time.sleep(wait); continue
            r.raise_for_status()
            data = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            for item in data["labels"]:
                idx = int(item["id"])
                if 0 <= idx < len(titles):
                    out[idx] = _to_tuple(item["label"], item.get("strength", 0.5))[0]
            break
        print(f"  batch {bn}/{len(batches)} done")
        if bn < len(batches):
            time.sleep(PACING)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("GEMINI_API_KEY not set (.env)."); sys.exit(1)

    con = sqlite3.connect("finance_sentiment.db")
    df = pd.read_sql_query(
        "SELECT source, title, sentiment_score FROM headlines WHERE sentiment_score IS NOT NULL", con)
    pg = df[df["source"].isin(PRO_GOV)].sample(min(SAMPLE_PER_CAMP, (df["source"].isin(PRO_GOV)).sum()),
                                               random_state=11)
    op = df[df["source"].isin(OPPOSITION)].sample(min(SAMPLE_PER_CAMP, (df["source"].isin(OPPOSITION)).sum()),
                                                  random_state=11)
    sample = pd.concat([pg, op]).reset_index(drop=True)
    print(f"Re-scoring {len(pg)} pro-gov + {len(op)} opposition headlines with {GEMINI_MODEL}\n")

    prompt = _build_system_prompt()
    scores = gemini_score(key, sample["title"].tolist(), prompt)
    sample["gemini"] = sample.index.map(scores)
    sample = sample.dropna(subset=["gemini"])

    pgm = sample[sample["source"].isin(PRO_GOV)]
    opm = sample[sample["source"].isin(OPPOSITION)]

    print("\n=== SLANT REPLICATION (same headlines, two independent models) ===")
    for model, col in [("gpt-5-mini (original)", "sentiment_score"), ("Gemini 2.5 Flash (check)", "gemini")]:
        gap = pgm[col].mean() - opm[col].mean()
        t, p = stats.ttest_ind(pgm[col], opm[col], equal_var=False)
        print(f"  {model:<26} pro-gov={pgm[col].mean():+.3f} opp={opm[col].mean():+.3f} "
              f"gap={gap:+.3f} (p={p:.1e})")
    # agreement between the two scorers on individual headlines
    r, _ = stats.pearsonr(sample["sentiment_score"], sample["gemini"])
    print(f"\n  Per-headline correlation between the two models: r={r:.2f}")
    print("  => if both gaps are large and same-signed, the political slant is in the")
    print("     text, not an artifact of one scorer.")


if __name__ == "__main__":
    main()
