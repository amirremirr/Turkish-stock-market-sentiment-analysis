"""
Framing vs selection (limitation L2) — same-story matching.

The slant finding could be "outlets frame the same events differently" (genuine
bias) or "outlets cover different events" (selection). To separate them, match
pro-gov and opposition headlines about the SAME event — same day (+/-1) and
enough shared content words — then compare sentiment WITHIN those matched pairs.

If pro-gov scores the *same stories* more positively than opposition, that is
framing, not selection.

This is a lightweight lexical matcher (shared significant tokens); the full
version is entity/event linking (see the migration). Read the example pairs to
judge match quality.

Usage:  python same_story_analysis.py
"""

import sqlite3
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

from scraper import _normalise
from polarization_analysis import PRO_GOV, OPPOSITION

MIN_SHARED = 2       # shared significant tokens to call it the same story
MIN_LEN = 5          # token length (bias toward content words / proper nouns)
WINDOW = 1           # +/- days

# very common tokens that co-occur across unrelated finance headlines
STOP = {"turkiye", "turkiyenin", "ekonomi", "ekonomik", "dolar", "euro", "borsa",
        "faiz", "enflasyon", "piyasa", "piyasalar", "milyar", "milyon", "yuzde",
        "acikladi", "aciklama", "sonra", "oldu", "olarak", "buyuk", "yeni",
        "bugun", "sonu", "gunu", "kadar", "daha", "icin", "ile", "rekor"}


def sig_tokens(title):
    toks = _normalise(str(title)).replace("'", " ").replace("-", " ").split()
    return {t for t in toks if len(t) >= MIN_LEN and t not in STOP and not t.isdigit()}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    con = sqlite3.connect("finance_sentiment.db")
    df = pd.read_sql_query(
        """SELECT source, published_at AS date, title, sentiment_score AS s
           FROM headlines WHERE sentiment_score IS NOT NULL AND published_at IS NOT NULL""", con)
    df["date"] = pd.to_datetime(df["date"])
    df["toks"] = df["title"].apply(sig_tokens)

    pg = df[df["source"].isin(PRO_GOV)].reset_index(drop=True)
    op = df[df["source"].isin(OPPOSITION)].reset_index(drop=True)

    pairs = []
    for _, o in op.iterrows():
        if not o["toks"]:
            continue
        cand = pg[(pg["date"] >= o["date"] - timedelta(days=WINDOW)) &
                  (pg["date"] <= o["date"] + timedelta(days=WINDOW))]
        best, best_n = None, MIN_SHARED - 1
        for _, c in cand.iterrows():
            shared = o["toks"] & c["toks"]
            if len(shared) > best_n:
                best_n, best = len(shared), (c, shared)
        if best is not None:
            c, shared = best
            pairs.append({"date": o["date"].date(), "op_title": o["title"], "op_s": o["s"],
                          "pg_title": c["title"], "pg_s": c["s"], "shared": shared})

    if len(pairs) < 5:
        print(f"Only {len(pairs)} same-story pairs found — too few to conclude.")
        print("(Opposition is still mostly one general-news outlet; matches grow as the")
        print(" broadened opposition economy feeds accumulate.)")
        return

    pr = pd.DataFrame(pairs)
    diff = pr["pg_s"] - pr["op_s"]                 # framing gap on the SAME story
    t, p = stats.ttest_rel(pr["pg_s"], pr["op_s"])
    print(f"=== SAME-STORY FRAMING TEST  (n={len(pr)} matched pairs) ===")
    print(f"  pro-gov mean on matched stories:    {pr['pg_s'].mean():+.3f}")
    print(f"  opposition mean on matched stories: {pr['op_s'].mean():+.3f}")
    print(f"  within-pair framing gap: {diff.mean():+.3f}  (paired t={t:.1f}, p={p:.1e})")
    share_pos = (diff > 0).mean()
    print(f"  share of pairs where pro-gov is more positive: {share_pos:.0%}")
    print("  => a positive, significant within-pair gap = FRAMING (same story, different")
    print("     spin), not merely different story selection.\n")

    print("  Example matched pairs (same story, the two camps' framing):")
    for _, r in pr.reindex(diff.abs().sort_values(ascending=False).index).head(5).iterrows():
        sh = ",".join(list(r["shared"])[:3])
        print(f"   [{r['date']}] shared: {sh}".encode("ascii", "replace").decode())
        print(f"     pro-gov ({r['pg_s']:+.2f}): {r['pg_title'][:72]}".encode("ascii", "replace").decode())
        print(f"     oppos.  ({r['op_s']:+.2f}): {r['op_title'][:72]}".encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
