"""
Exploratory signal analysis — targets and aggregations (2026-07-07).

EXPLORATORY ONLY, NOT INFERENCE. At ~30 overlap days this study is badly
underpowered (it can only detect |r| > ~0.5), and running many correlations
invites false positives — so this module does two disciplined things:

  Priority 1 (targets):      does the production daily sentiment relate to
                             next-day RETURN, VOLATILITY, FX, or ABNORMAL
                             return (BIST net of EM)?
  Priority 3 (aggregation):  does the choice of how we collapse a day's
                             headlines into one number change anything?
                             (mean vs confidence-weighted vs intensity vs
                             shock-count vs net-direction, all vs next return.)

Every test is pooled and corrected with Benjamini-Hochberg (FDR). The verdict
is about which target/aggregation to PRIORITISE once there is enough data —
NOT about whether a signal exists today.

Usage:  python explore_signal.py
"""

import sqlite3
import sys
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import DB_PATH, MINIMUM_HEADLINES_PER_DAY


def load(db_path: str):
    con = sqlite3.connect(db_path)
    h = pd.read_sql_query(
        """SELECT published_at AS date, sentiment_score AS s, sentiment_label AS lbl
           FROM headlines WHERE sentiment_score IS NOT NULL AND published_at IS NOT NULL""",
        con)
    h["date"] = pd.to_datetime(h["date"])
    prices = pd.read_sql_query("SELECT date, daily_return FROM bist100_prices ORDER BY date", con)
    prices["date"] = pd.to_datetime(prices["date"])
    em = pd.read_sql_query(
        "SELECT date, daily_return FROM market_factors WHERE symbol='EEM' ORDER BY date", con)
    em["date"] = pd.to_datetime(em["date"])
    fx = pd.read_sql_query("SELECT date, close FROM usdtry_rates ORDER BY date", con)
    fx["date"] = pd.to_datetime(fx["date"])
    prod = pd.read_sql_query(
        "SELECT date, avg_score, headline_count FROM daily_sentiment", con)
    prod["date"] = pd.to_datetime(prod["date"])
    return h, prices, em, fx, prod


def daily_features(h: pd.DataFrame) -> pd.DataFrame:
    """Alternative aggregations of each day's headlines."""
    def wmean(s):
        w = s.abs().clip(lower=0.10)
        return float(np.average(s, weights=w)) if w.sum() else 0.0
    g = h.groupby("date").agg(
        n=("s", "size"),
        mean=("s", "mean"),
        conf_weighted=("s", wmean),
        intensity=("s", lambda s: s.abs().mean()),
        shock_count=("s", lambda s: int((s.abs() >= 0.5).sum())),
        net_dir=("lbl", lambda l: ((l == "positive").sum() - (l == "negative").sum()) / len(l)),
    ).reset_index()
    return g[g["n"] >= MINIMUM_HEADLINES_PER_DAY]


def build_targets(prices, em, fx):
    """Next-day target series, matched on date."""
    p = prices.copy()
    p["ret_next"] = p["daily_return"].shift(-1)
    p["absret_next"] = p["ret_next"].abs()
    # abnormal return vs EM: residual of BIST on EM (contemporaneous beta)
    j = p.merge(em.rename(columns={"daily_return": "em"}), on="date", how="left")
    both = j.dropna(subset=["daily_return", "em"])
    if len(both) >= 10:
        beta = np.polyfit(both["em"], both["daily_return"], 1)[0]
    else:
        beta = 1.0
    j["abn"] = j["daily_return"] - beta * j["em"]
    j["abn_next"] = j["abn"].shift(-1)
    fx = fx.copy()
    fx["fx_next"] = fx["close"].pct_change().shift(-1) * 100
    return j[["date", "ret_next", "absret_next", "abn_next"]].merge(
        fx[["date", "fx_next"]], on="date", how="left"), beta


def corr(x, y):
    m = pd.concat([x, y], axis=1).dropna()
    if len(m) < 5:
        return len(m), np.nan, np.nan
    r, p = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
    return len(m), r, p


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    h, prices, em, fx, prod = load(DB_PATH)
    feats = daily_features(h)
    tgt, beta = build_targets(prices, em, fx)
    # production sentiment (confidence + time + relevance weighted) from daily_sentiment
    prod_rel = prod[prod["headline_count"] >= MINIMUM_HEADLINES_PER_DAY][["date", "avg_score"]]

    df = feats.merge(tgt, on="date").merge(prod_rel, on="date", how="left")

    tests = []
    # Priority 1 — production sentiment vs each target
    for tname, tcol in [("next return (direction)", "ret_next"),
                        ("next |return| (volatility)", "absret_next"),
                        ("next USD/TRY (FX)", "fx_next"),
                        ("next abnormal ret (vs EM)", "abn_next")]:
        n, r, p = corr(df["avg_score"], df[tcol])
        tests.append(("TARGET", f"sentiment -> {tname}", n, r, p))
    # Priority 3 — aggregation choice vs the primary target (next return)
    for aname, acol in [("simple mean", "mean"), ("confidence-weighted", "conf_weighted"),
                        ("intensity |score|", "intensity"), ("shock count", "shock_count"),
                        ("net direction", "net_dir")]:
        n, r, p = corr(df[acol], df["ret_next"])
        tests.append(("AGGREG", f"{aname} -> next return", n, r, p))

    res = pd.DataFrame(tests, columns=["block", "test", "n", "r", "p"])
    ok = res["p"].notna()
    res.loc[ok, "q_bh"] = multipletests(res.loc[ok, "p"], method="fdr_bh")[1]

    print("\nEXPLORATORY signal analysis (NOT inference — see power note below)\n")
    print(f"  EM beta (BIST on EEM) = {beta:.2f}\n")
    print(f"  {'block':<7}{'test':<38}{'n':>4}{'r':>8}{'p':>7}{'q(BH)':>8}")
    print("  " + "-" * 72)
    for _, row in res.iterrows():
        q = f"{row['q_bh']:.2f}" if pd.notna(row.get("q_bh")) else "  -"
        rr = f"{row['r']:+.3f}" if pd.notna(row["r"]) else "  n/a"
        pp = f"{row['p']:.2f}" if pd.notna(row["p"]) else "  -"
        print(f"  {row['block']:<7}{row['test']:<38}{int(row['n']):>4}{rr:>8}{pp:>7}{q:>8}")

    n_primary = int(res["n"].max())
    C = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    r_min = math.tanh(C / math.sqrt(max(n_primary, 4) - 3))
    sig = res["q_bh"].dropna().lt(0.10).sum()
    print("\n  Verdict:")
    print(f"    - Smallest |r| detectable at ~n={n_primary}, 80% power: ~{r_min:.2f} (large).")
    print(f"    - Tests surviving FDR correction (q<0.10): {int(sig)}.")
    print("    - Read raw p<0.05 with suspicion: with this many tests at this n, the")
    print("      occasional 'hit' is expected by chance (see the FX false positive).")
    print("    - Use this to see which target/aggregation to PRIORITISE as data grows,")
    print("      not to claim a signal today.")


if __name__ == "__main__":
    main()
