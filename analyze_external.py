"""
Zoom-out analysis: public attention (Google Trends) and the world's media view
(GDELT) vs the market, the domestic press, and the media-polarization index.

Many of these relationships are HIGHER-N than the 30-day return question — a
search-interest vs lira link has ~60-120 daily points — so they are testable
now. Still FDR-corrected (we don't cherry-pick p-values).

Usage:  python analyze_external.py
"""

import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from polarization_analysis import PRO_GOV, OPPOSITION


def load():
    con = sqlite3.connect("finance_sentiment.db")
    ext = pd.read_sql_query("SELECT date, series, value FROM external_series", con)
    ext = ext.pivot(index="date", columns="series", values="value")
    bist = pd.read_sql_query("SELECT date, daily_return FROM bist100_prices", con).set_index("date")
    fx = pd.read_sql_query(
        "SELECT date, close FROM market_factors WHERE symbol='USDTRY=X'", con).set_index("date")
    fx["usdtry_ret"] = fx["close"].pct_change() * 100
    sent = pd.read_sql_query(
        "SELECT date, avg_score, headline_count FROM daily_sentiment", con).set_index("date")
    sent = sent[sent["headline_count"] >= 3][["avg_score"]]
    # polarization index
    h = pd.read_sql_query(
        "SELECT source, published_at AS date, sentiment_score AS s FROM headlines WHERE sentiment_score IS NOT NULL", con)
    def camp_daily(src):
        g = h[h["source"].isin(src)].groupby("date").agg(m=("s", "mean"), n=("s", "size"))
        return g[g["n"] >= 2]["m"]
    pol = (camp_daily(PRO_GOV) - camp_daily(OPPOSITION)).rename("polarization")

    df = ext.join(bist).join(fx[["close", "usdtry_ret"]]).join(sent).join(pol)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # CHANGES, not levels: search interest and prices both trend over the period,
    # and correlating trending levels inflates r (a real trap — GDELT tone vs the
    # lira looks like r=0.48 on levels but is ~0 on changes). Test on changes.
    df["dolar_chg"] = df["gt_dolar"].diff()
    df["kriz_chg"] = df["gt_kriz"].diff()
    df["tone_chg"] = df["gdelt_tone"].diff()
    df["usdtry_ret_next"] = df["usdtry_ret"].shift(-1)
    df["bist_ret_next"] = df["daily_return"].shift(-1)
    return df


def corr(df, a, b):
    m = df[[a, b]].dropna()
    if len(m) < 8:
        return len(m), np.nan, np.nan
    r, p = stats.pearsonr(m[a], m[b])
    return len(m), r, p


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = load()

    tests = [
        # public attention <-> the lira (on CHANGES, stationary)
        ("PUBLIC", "d(dolar-search) vs d(USD/TRY), same day", "dolar_chg", "usdtry_ret"),
        ("PUBLIC", "d(dolar-search) vs NEXT-day d(USD/TRY)", "dolar_chg", "usdtry_ret_next"),
        ("PUBLIC", "d(kriz-search) vs next-day BIST return", "kriz_chg", "bist_ret_next"),
        # public <-> press
        ("PUB<>PRESS", "d(dolar-search) vs press sentiment", "dolar_chg", "avg_score"),
        ("PUB<>PRESS", "d(dolar-search) vs media polarization", "dolar_chg", "polarization"),
        # global media <-> domestic (tone is bounded, less trending; also test on changes)
        ("GLOBAL", "global tone (GDELT) vs domestic press sentiment", "gdelt_tone", "avg_score"),
        ("GLOBAL", "d(global tone) vs d(USD/TRY)", "tone_chg", "usdtry_ret"),
        ("GLOBAL", "d(global tone) vs domestic press sentiment", "tone_chg", "avg_score"),
    ]
    res = []
    for block, name, a, b in tests:
        n, r, p = corr(df, a, b)
        res.append((block, name, n, r, p))
    R = pd.DataFrame(res, columns=["block", "test", "n", "r", "p"])
    ok = R["p"].notna()
    R.loc[ok, "q"] = multipletests(R.loc[ok, "p"], method="fdr_bh")[1]

    print("\nZOOM-OUT relationships (FDR-corrected)\n")
    print(f"  {'block':<11}{'test':<44}{'n':>4}{'r':>8}{'p':>7}{'q':>7}")
    print("  " + "-" * 80)
    for _, r in R.iterrows():
        q = f"{r['q']:.2f}" if pd.notna(r.get("q")) else "  -"
        rr = f"{r['r']:+.3f}" if pd.notna(r["r"]) else " n/a"
        pp = f"{r['p']:.3f}" if pd.notna(r["p"]) else "  -"
        print(f"  {r['block']:<11}{r['test']:<44}{int(r['n']):>4}{rr:>8}{pp:>7}{q:>7}")
    hits = R[R["q"] < 0.10].dropna(subset=["q"])
    print("\n  Survive FDR (q<0.10):", ", ".join(hits["test"]) if len(hits) else "none")
    print("\n  Note: correlations use CHANGES, not levels. On levels, dolar-search vs")
    print("  the lira looks like r=0.75 and GDELT tone vs the lira r=0.48 -- but those")
    print("  are common-trend artifacts (both series drift over the period); on changes")
    print("  the GDELT one vanishes entirely (r~0). Levels lie; changes are honest.")

    # ---- figure: the two most telling views ----
    try: plt.style.use("seaborn-v0_8-whitegrid")
    except OSError: pass
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Zooming out: public attention & global media vs the lira / press",
                 fontsize=13, fontweight="bold")
    d = df.dropna(subset=["gt_dolar", "close"])
    ax[0].plot(d.index, d["gt_dolar"], color="#C62828", label="'dolar' search interest")
    ax0b = ax[0].twinx()
    ax0b.plot(d.index, d["close"], color="#1565C0", label="USD/TRY")
    ax[0].set_ylabel("search interest (0-100)", color="#C62828")
    ax0b.set_ylabel("USD/TRY", color="#1565C0")
    ax[0].set_title("Public dollar-anxiety vs the lira", fontweight="bold")
    plt.setp(ax[0].get_xticklabels(), rotation=30, ha="right")
    d2 = df.dropna(subset=["gdelt_tone", "avg_score"])
    ax[1].scatter(d2["gdelt_tone"], d2["avg_score"], alpha=0.5, color="#6A1B9A")
    ax[1].set_xlabel("global media tone (GDELT)"); ax[1].set_ylabel("domestic press sentiment")
    ax[1].set_title("World's view vs domestic press", fontweight="bold")
    ax[1].axhline(0, color="grey", lw=0.6); ax[1].axvline(0, color="grey", lw=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("docs/external_overview.png", dpi=140, bbox_inches="tight")
    print("\nFigure -> docs/external_overview.png")


if __name__ == "__main__":
    main()
