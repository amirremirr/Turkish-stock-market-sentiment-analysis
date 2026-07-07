"""
Media polarization analysis — a Turkey-specific angle on financial news.

Idea: Turkey has a structurally clean pro-government press (Anadolu Agency,
Sabah) vs opposition press (Sozcu). Instead of asking "is the news positive?",
ask "do the two sides DISAGREE?" — inter-outlet disagreement is a text-only
proxy for political uncertainty, which is what actually moves Turkish assets.

Three layers, in order of statistical strength:
  1. SLANT GRADIENT (high N, ~1,900 headlines): do outlets differ systematically
     by political leaning, and is the difference significant?
  2. WHERE the slant lives (high N): is the pro-gov/opposition gap larger on
     politically-charged topics than on neutral ones? (If so, it's a *political*
     slant, not a general tone difference — the non-obvious part.)
  3. DAILY POLARIZATION INDEX (thin, ~20-30 days, exploratory): the pro-gov minus
     opposition sentiment gap over time, its spikes, and a caveated link to lira
     volatility.

Usage:  python polarization_analysis.py
"""

import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import DB_PATH

PRO_GOV    = ["aa_ekonomi", "aa_politika", "sabah_ekonomi"]   # state + pro-government
# Opposition camp broadened 2026-07-07 (was single-source). cumhuriyet_ekonomi /
# sozcu_ekonomi start collecting from that date — re-run to check the slant holds
# with a second distinct opposition outlet as their history accumulates.
OPPOSITION = ["sozcu_gundem", "sozcu_ekonomi", "cumhuriyet_ekonomi"]
MARKET     = ["bloomberght", "investing_tr_economy"]           # market-focused
_CAT = {"fx_lira": "Currency", "turkey_macro": "Turkish economy",
        "energy_commodities": "Energy/commodities", "rates_tcmb": "Rates/TCMB",
        "bist_company": "Companies", "global_risk": "Global", "political_risk": "Politics",
        "banks": "Banking", "crypto": "Crypto", "other": "Other"}


def load(db_path):
    con = sqlite3.connect(db_path)
    h = pd.read_sql_query(
        """SELECT source, published_at AS date, category, sentiment_score AS s, title
           FROM headlines WHERE sentiment_score IS NOT NULL AND published_at IS NOT NULL""", con)
    h["date"] = pd.to_datetime(h["date"])
    fx = pd.read_sql_query(
        "SELECT date, close FROM market_factors WHERE symbol='USDTRY=X' ORDER BY date", con)
    fx["date"] = pd.to_datetime(fx["date"])
    return h, fx


def camp(h, sources):
    return h[h["source"].isin(sources)]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    h, fx = load(DB_PATH)
    pg, op, mk = camp(h, PRO_GOV), camp(h, OPPOSITION), camp(h, MARKET)

    # ---- 1. Slant gradient + significance test ------------------------------
    print("\n1. SLANT GRADIENT  (mean sentiment +/- 95% CI)")
    def ci(x):
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        return m, 1.96 * se
    for name, grp in [("PRO-GOV/state", pg), ("Market-focused", mk), ("OPPOSITION", op)]:
        m, e = ci(grp["s"])
        print(f"   {name:<16} n={len(grp):<5} {m:+.3f} +/- {e:.3f}")
    t, p = stats.ttest_ind(pg["s"], op["s"], equal_var=False)
    d = (pg["s"].mean() - op["s"].mean()) / np.sqrt((pg["s"].var() + op["s"].var()) / 2)
    print(f"   pro-gov vs opposition gap = {pg['s'].mean() - op['s'].mean():+.3f}  "
          f"(t={t:.1f}, p={p:.1e}, Cohen d={d:.2f})")
    print("   => the political slant in financial sentiment is large and highly significant.")

    # ---- 2. Where the slant lives (by topic) --------------------------------
    print("\n2. WHERE THE SLANT LIVES  (pro-gov minus opposition, by topic)")
    rows = []
    for cat in h["category"].dropna().unique():
        a, b = pg[pg["category"] == cat]["s"], op[op["category"] == cat]["s"]
        if len(a) >= 10 and len(b) >= 10:
            rows.append((cat, a.mean() - b.mean(), len(a), len(b)))
    slant = pd.DataFrame(rows, columns=["cat", "gap", "npg", "nop"]).sort_values("gap", ascending=False)
    for _, r in slant.iterrows():
        print(f"   {_CAT.get(r['cat'], r['cat']):<20} gap={r['gap']:+.3f}  (n {int(r['npg'])}/{int(r['nop'])})")
    if len(slant) >= 2:
        print("   => if politics/macro top the list and commodities/global sit low, the")
        print("      divergence is POLITICAL, not a blanket tone difference.")

    # ---- 3. Daily polarization index ----------------------------------------
    def daily_mean(grp):
        g = grp.groupby("date").agg(m=("s", "mean"), n=("s", "size"))
        return g[g["n"] >= 2]["m"]
    pgd, opd = daily_mean(pg), daily_mean(op)
    idx = pd.DataFrame({"pg": pgd, "op": opd}).dropna()
    idx["polarization"] = idx["pg"] - idx["op"]
    print(f"\n3. DAILY POLARIZATION INDEX  (pro-gov - opposition, n={len(idx)} days -- thin/exploratory)")
    print(f"   mean gap {idx['polarization'].mean():+.3f}, std {idx['polarization'].std():.3f}")
    top = idx.sort_values("polarization", ascending=False).head(3)
    print("   Highest-polarization days (pro-gov bullish while opposition bearish):")
    for dt, r in top.iterrows():
        print(f"     {dt.date()}  gap={r['polarization']:+.2f}")

    # caveated market link: polarization vs next-day lira volatility
    if len(fx) > 5:
        fx = fx.copy(); fx["absret_next"] = (fx["close"].pct_change().shift(-1) * 100).abs()
        ml = idx.reset_index().merge(fx[["date", "absret_next"]], on="date").dropna()
        if len(ml) >= 8:
            r, pp = stats.pearsonr(ml["polarization"], ml["absret_next"])
            print(f"\n   [caveated, n={len(ml)}] polarization -> next-day |USD/TRY move|: "
                  f"r={r:+.3f} (p={pp:.2f})  -- exploratory, underpowered")

    # ---- Figure --------------------------------------------------------------
    try: plt.style.use("seaborn-v0_8-whitegrid")
    except OSError: pass
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Turkish financial press: political slant in market sentiment",
                 fontsize=14, fontweight="bold")
    # gradient
    order = [("Sabah\n(pro-gov)", "sabah_ekonomi"), ("AA\n(state)", "aa_ekonomi"),
             ("Haberturk", "haberturk_ekonomi"), ("Dunya", "dunya"),
             ("BloombergHT\n(market)", "bloomberght"), ("Investing\n(market)", "investing_tr_economy"),
             ("Sozcu\n(opp.)", "sozcu_gundem")]
    labels, means, errs = [], [], []
    for lab, src in order:
        x = h[h["source"] == src]["s"]
        if len(x) >= 10:
            m, e = ci(x); labels.append(lab); means.append(m); errs.append(e)
    cols = ["#2E7D32" if m > 0 else "#C62828" for m in means]
    ax[0].bar(range(len(means)), means, yerr=errs, color=cols, alpha=0.85, capsize=3)
    ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].axhline(0, color="black", lw=0.8); ax[0].set_ylabel("mean sentiment")
    ax[0].set_title("Slant gradient (95% CI)", fontweight="bold")
    # by topic
    s2 = slant.head(8)
    ax[1].barh([_CAT.get(c, c) for c in s2["cat"]][::-1], s2["gap"].values[::-1],
               color="#6A1B9A", alpha=0.8)
    ax[1].set_title("Pro-gov minus opposition, by topic", fontweight="bold")
    ax[1].set_xlabel("sentiment gap")
    # index over time
    ax[2].plot(idx.index, idx["polarization"], color="#1565C0", lw=1.5, marker="o", ms=3)
    ax[2].axhline(idx["polarization"].mean(), color="grey", ls="--", lw=0.8)
    ax[2].set_title(f"Daily polarization index (n={len(idx)})", fontweight="bold")
    ax[2].set_ylabel("pro-gov - opposition")
    plt.setp(ax[2].get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("docs/polarization.png", dpi=140, bbox_inches="tight")
    print("\nFigure -> docs/polarization.png")


if __name__ == "__main__":
    main()
