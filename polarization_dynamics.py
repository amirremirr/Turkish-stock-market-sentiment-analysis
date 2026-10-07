"""
Media polarization — dynamics and mechanism (deeper descriptive study).

Two questions:
  1. WHO drives the slant? Using the market-focused press as a neutral baseline,
     is the pro-gov/opposition split driven more by pro-government *optimism* or
     opposition *pessimism*? (High N, robust — bootstrap CIs.)
  2. Does the gap WIDEN under economic stress? A political-economy hypothesis:
     pro-gov media stays upbeat to defend the government while opposition
     amplifies bad news, so the divergence should be counter-cyclical — larger
     when the lira is weak / public anxiety is high / news is bad. (Weekly, thin,
     exploratory; scheduled macro events overlaid.)

Usage:  python polarization_dynamics.py
"""

import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import ECONOMIC_CALENDAR
from polarization_analysis import PRO_GOV, OPPOSITION, MARKET


def load():
    con = sqlite3.connect("finance_sentiment.db")
    h = pd.read_sql_query(
        "SELECT source, published_at AS date, sentiment_score AS s FROM headlines "
        "WHERE sentiment_score IS NOT NULL AND published_at IS NOT NULL", con)
    h["date"] = pd.to_datetime(h["date"])
    fx = pd.read_sql_query(
        "SELECT date, close FROM market_factors WHERE symbol='USDTRY=X'", con).set_index("date")
    ext = pd.read_sql_query(
        "SELECT date, value FROM external_series WHERE series='gt_dolar'", con).set_index("date")["value"]
    sent = pd.read_sql_query(
        "SELECT date, avg_score, headline_count FROM daily_sentiment", con)
    return h, fx, ext, sent


def date_cluster_bootstrap(h, camps, n=5000, seed=7):
    """Resample whole publication dates, not headlines.

    Headlines on the same date share stories and news flow, so an iid
    headline bootstrap overstates precision (see docs/POLARIZATION_METHODS.md).
    The market baseline is resampled with the camps rather than held fixed.
    Returns an (n, len(camps)) array of replicate camp means.
    """
    d = h[h["source"].isin(sum(camps.values(), []))].copy()
    d["camp"] = ""
    for name, sources in camps.items():
        d.loc[d["source"].isin(sources), "camp"] = name
    d["day"] = d["date"].dt.strftime("%Y-%m-%d")
    sums = d.pivot_table(index="day", columns="camp", values="s", aggfunc="sum").reindex(columns=list(camps)).fillna(0.0)
    cnts = d.pivot_table(index="day", columns="camp", values="s", aggfunc="size").reindex(columns=list(camps)).fillna(0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    return sums.values[idx].sum(axis=1) / cnts.values[idx].sum(axis=1), len(sums)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    np.random.seed(7)
    h, fx, ext, sent = load()
    pg = h[h["source"].isin(PRO_GOV)]["s"]
    op = h[h["source"].isin(OPPOSITION)]["s"]
    mk = h[h["source"].isin(MARKET)]["s"]

    # ---- 1. Who drives the slant (deviations from the market baseline) ------
    base = mk.mean()
    reps, n_days = date_cluster_bootstrap(h, {"pg": PRO_GOV, "op": OPPOSITION, "mk": MARKET})
    print("1. WHO DRIVES THE SLANT  (market-focused press = neutral baseline)")
    print(f"   window {h['date'].min():%Y-%m-%d} .. {h['date'].max():%Y-%m-%d}, "
          f"{n_days} publication dates (bootstrap resamples dates)")
    print(f"   market baseline sentiment: {base:+.3f}  (n={len(mk)})")
    for i, (name, grp) in enumerate([("pro-gov", pg), ("opposition", op)]):
        dev = reps[:, i] - reps[:, 2]
        lo, hi = np.percentile(dev, [2.5, 97.5])
        print(f"   {name:<11} {grp.mean():+.3f}  (n={len(grp)}, dev from baseline "
              f"{grp.mean()-base:+.3f}, 95% CI [{lo:+.3f},{hi:+.3f}])")
    pg_dev, op_dev = abs(pg.mean() - base), abs(op.mean() - base)
    diffs = np.abs(reps[:, 0] - reps[:, 2]) - np.abs(reps[:, 1] - reps[:, 2])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"   asymmetry |pro-gov dev| - |opp dev| = {pg_dev-op_dev:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
    verdict = "pro-gov OPTIMISM" if pg_dev > op_dev else "opposition PESSIMISM"
    strong = "and the CI excludes 0" if lo > 0 or hi < 0 else "but the CI includes 0"
    print(f"   => the slant is driven more by {verdict} ({strong}).")

    # ---- weekly polarization + stress hypothesis ----------------------------
    def weekly_camp(grp_src):
        d = h[h["source"].isin(grp_src)].copy()
        d["w"] = d["date"].dt.to_period("W").apply(lambda p: p.start_time)
        g = d.groupby("w").agg(m=("s", "mean"), n=("s", "size"))
        return g[g["n"] >= 5]["m"]
    wpg, wop = weekly_camp(PRO_GOV), weekly_camp(OPPOSITION)
    wk = pd.DataFrame({"pg": wpg, "op": wop}).dropna()
    wk["gap"] = wk["pg"] - wk["op"]
    # weekly stress
    fx.index = pd.to_datetime(fx.index); ext.index = pd.to_datetime(ext.index)
    fxw = fx["close"].resample("W").last().pct_change() * 100     # weekly lira depreciation %
    anxw = ext.resample("W").mean()                               # weekly dolar-search
    fxw.index = fxw.index.to_period("W").to_timestamp(); anxw.index = anxw.index.to_period("W").to_timestamp()
    wk = wk.join(fxw.rename("lira_deprec")).join(anxw.rename("anxiety"))

    print(f"\n2. STRESS HYPOTHESIS  (weekly, n={len(wk)} weeks)")
    tested = False
    for lbl, col in [("gap vs lira depreciation", "lira_deprec"), ("gap vs public anxiety", "anxiety")]:
        m = wk[["gap", col]].dropna()
        if len(m) >= 6:
            r, p = stats.pearsonr(m["gap"], m[col]); tested = True
            rho, p_rho = stats.spearmanr(m["gap"], m[col])
            print(f"   {lbl:<28} r={r:+.3f} (p={p:.2f}, n={len(m)})  "
                  f"rank rho={rho:+.3f} (p={p_rho:.2f})")
            # Few weeks, so one extreme week can carry Pearson on its own.
            # Read the hypothesis as supported only if the rank test agrees.
    if not tested:
        print("   Too few weeks to test — only a handful have >=5 opposition headlines/week")
        print("   (opposition is still mostly one outlet). DEFERRED until the broadened")
        print("   opposition feeds (Cumhuriyet, Sozcu-economy, added 2026-07-07) accumulate.")
    print("   (positive r = the pro-gov/opposition gap widens under stress)")

    # ---- event overlay ------------------------------------------------------
    events = {pd.Timestamp(d): lbl for d, lbl in ECONOMIC_CALENDAR.items()
              if h["date"].min() <= pd.Timestamp(d) <= h["date"].max()}
    print(f"\n3. Scheduled macro events in window: {len(events)} "
          f"({', '.join(sorted(set(events.values())))})")

    # ---- figure -------------------------------------------------------------
    try: plt.style.use("seaborn-v0_8-whitegrid")
    except OSError: pass
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Media polarization: who drives it, and does it widen under stress?",
                 fontsize=13, fontweight="bold")
    ax[0].bar(["pro-gov", "market\n(baseline)", "opposition"],
              [pg.mean(), base, op.mean()],
              color=["#2E7D32", "#9E9E9E", "#C62828"], alpha=0.85)
    ax[0].axhline(base, color="grey", ls="--", lw=0.8)
    ax[0].set_title("Deviation from the market baseline", fontweight="bold")
    ax[0].set_ylabel("mean sentiment")
    ax[1].plot(wk.index, wk["gap"], color="#1565C0", marker="o", ms=4, label="pro-gov - opposition")
    for d, lbl in events.items():
        ax[1].axvline(d, color="#C62828" if "PPK" in lbl else "#FF9800", ls=":", lw=1)
    ax[1].set_title("Weekly polarization gap (dotted = macro events)", fontweight="bold")
    ax[1].set_ylabel("gap"); plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("docs/polarization_dynamics.png", dpi=140, bbox_inches="tight")
    print("\nFigure -> docs/polarization_dynamics.png")


if __name__ == "__main__":
    main()
