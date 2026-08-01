"""
Exploratory signal analysis — targets and aggregation sensitivity.

EXPLORATORY ONLY, NOT INFERENCE. At ~30 overlap days this study is badly
underpowered (it can only detect |r| > ~0.5), and running many correlations
invites false positives — so this module does two disciplined things:

  Priority 1 (targets):      does the session-aligned, unweighted sentiment
                             baseline relate to subsequent-session RETURN,
                             VOLATILITY, FX, or ABNORMAL return (BIST net of EM)?
  Priority 3 (sensitivity):  how do the pre-specified relevance/intensity/full
                             weighted variants compare with that baseline?

Every test is pooled and corrected with Benjamini-Hochberg (FDR). The verdict
is about which target or sensitivity check to PRIORITISE once there is enough
data — NOT about whether a signal exists today.

Usage:  python explore_signal.py
"""

import sqlite3
import sys
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import database as db
from config import DB_PATH, MINIMUM_HEADLINES_PER_DAY


def load(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        prices = pd.read_sql_query(
            "SELECT date, close, daily_return FROM bist100_prices ORDER BY date", con
        )
        em = pd.read_sql_query(
            "SELECT date, daily_return FROM market_factors "
            "WHERE symbol='EEM' ORDER BY date",
            con,
        )
        fx = pd.read_sql_query(
            "SELECT date, close FROM usdtry_rates ORDER BY date", con
        )
    finally:
        con.close()

    # Market-linked analysis has no publication-date fallback. avg_score is a
    # local compatibility alias for the pre-specified simple_mean baseline.
    prod = db.get_signal_variants(db_path=db_path).rename(
        columns={"simple_mean": "avg_score"}
    )
    for frame in (prices, em, fx, prod):
        frame["date"] = pd.to_datetime(frame["date"])
    return prices, em, fx, prod


def build_targets(prices, em, fx):
    """Build subsequent-observation targets before any signal-date join."""
    p = prices.copy().sort_values("date")
    # Recompute close-to-close returns on the complete ordered session table.
    # Shifting after a signal join would silently jump across missing signals.
    p["daily_return"] = p["close"].pct_change(fill_method=None) * 100.0
    p["ret_next"] = p["daily_return"].shift(-1)
    p["absret_next"] = p["ret_next"].abs()
    # abnormal return vs EM: residual of BIST on EM (contemporaneous beta)
    j = p.merge(
        em.rename(columns={"daily_return": "em"}), on="date", how="left"
    ).sort_values("date")
    both = j.dropna(subset=["daily_return", "em"])
    if len(both) >= 10:
        beta = np.polyfit(both["em"], both["daily_return"], 1)[0]
    else:
        beta = 1.0
    j["abn"] = j["daily_return"] - beta * j["em"]
    j["abn_next"] = j["abn"].shift(-1)
    fx = fx.copy().sort_values("date")
    fx["fx_next"] = fx["close"].pct_change(fill_method=None).shift(-1) * 100
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
    prices, em, fx, prod = load(DB_PATH)
    if prices.empty or prod.empty:
        print(
            "No session-aligned signal/price overlap is available. "
            "Run the aggregate and price steps first."
        )
        return
    tgt, beta = build_targets(prices, em, fx)
    variant_columns = [
        "avg_score",
        "relevance_weighted",
        "intensity_relevance_weighted",
        "full_weighted",
    ]
    prod_rel = prod[prod["headline_count"] >= MINIMUM_HEADLINES_PER_DAY][
        ["date", *variant_columns]
    ]
    df = prod_rel.merge(tgt, on="date", how="inner")

    tests = []
    # Priority 1 — session-aligned unweighted baseline vs each target.
    for tname, tcol in [
        ("subsequent-session return (direction)", "ret_next"),
        ("subsequent |return| (volatility)", "absret_next"),
        ("subsequent USD/TRY observation (FX)", "fx_next"),
        ("subsequent abnormal return (vs EM)", "abn_next"),
    ]:
        n, r, p = corr(df["avg_score"], df[tcol])
        tests.append(("TARGET", f"unweighted baseline -> {tname}", n, r, p))
    # Priority 3 — weighted variants are sensitivity views only.
    for aname, acol in [
        ("unweighted baseline", "avg_score"),
        ("relevance weighted", "relevance_weighted"),
        ("intensity + relevance weighted", "intensity_relevance_weighted"),
        ("full weighted", "full_weighted"),
    ]:
        n, r, p = corr(df[acol], df["ret_next"])
        tests.append(("SENS", f"{aname} -> subsequent return", n, r, p))

    res = pd.DataFrame(tests, columns=["block", "test", "n", "r", "p"])
    ok = res["p"].notna()
    res.loc[ok, "q_bh"] = multipletests(res.loc[ok, "p"], method="fdr_bh")[1]

    print("\nEXPLORATORY subsequent-session analysis (NOT inference)\n")
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
    print("    - Weighted variants are sensitivity checks, not alternate defaults.")
    print("    - Use this to pre-specify future work, not to claim a signal today.")


if __name__ == "__main__":
    main()
