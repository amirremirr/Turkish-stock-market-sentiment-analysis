"""README figure for the outlet-associated tone difference.

Reads the same rows as the inference report (``load_headlines``: scored,
non-excluded, one row per outlet per headline), so the figure and the numbers
quoted beside it cannot drift apart. Error bars resample whole publication
dates, matching docs/POLARIZATION_METHODS.md; headlines on one date share news
flow, so a per-headline interval would overstate precision.

Usage:  python -m analysis.polarization.figure [--db PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.corpus.report import _CAT_LABEL
from analysis.polarization.inference import (
    DEFAULT_OPPOSITION_SOURCES, DEFAULT_PRO_GOVERNMENT_SOURCES, OUTLET_OF_FEED,
    load_headlines,
)
from config import DB_PATH

_OUTLET_LABEL = {
    "sabah_ekonomi": "Sabah", "aa": "AA", "haberturk_ekonomi": "Haberturk",
    "dunya": "Dunya", "hurriyet_ekonomi": "Hurriyet", "ntv_ekonomi": "NTV",
    "bloomberght": "BloombergHT", "investing_tr_economy": "Investing",
    "cumhuriyet_ekonomi": "Cumhuriyet", "sozcu": "Sozcu",
}
_GREEN, _RED, _GREY, _BLUE = "#2E7D32", "#C62828", "#9E9E9E", "#1565C0"


def _outlet(feed: str) -> str:
    return OUTLET_OF_FEED.get(feed, feed)


def _date_cluster_ci(frame: pd.DataFrame, n: int = 2000, seed: int = 20260707):
    """95% interval for a mean, resampling publication dates."""

    by_day = frame.groupby("day")["sentiment"].agg(["sum", "size"])
    if len(by_day) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by_day), size=(n, len(by_day)))
    means = by_day["sum"].values[idx].sum(axis=1) / by_day["size"].values[idx].sum(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def build(db_path: str, out: str) -> None:
    h = load_headlines(db_path)
    h["day"] = pd.to_datetime(h["date"].astype(str).str[:10])
    h["outlet"] = h["source"].map(_outlet)
    pro = {_outlet(s) for s in DEFAULT_PRO_GOVERNMENT_SOURCES}
    opp = {_outlet(s) for s in DEFAULT_OPPOSITION_SOURCES}
    h["camp"] = np.select([h["outlet"].isin(pro), h["outlet"].isin(opp)],
                          ["pro", "opp"], default="other")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    first, last = h["day"].min(), h["day"].max()
    fig.suptitle(
        "Turkish financial press: outlet-associated tone differences "
        f"({first:%d %b} - {last:%d %b %Y}, {h['day'].nunique()} dates)",
        fontsize=14, fontweight="bold",
    )

    # 1. Outlet gradient with date-cluster intervals.
    rows = []
    for outlet, grp in h.groupby("outlet"):
        if len(grp) >= 30:
            lo, hi = _date_cluster_ci(grp)
            rows.append((outlet, grp["sentiment"].mean(), lo, hi, len(grp),
                         grp["camp"].iloc[0]))
    g = pd.DataFrame(rows, columns=["outlet", "m", "lo", "hi", "n", "camp"]).sort_values(
        "m", ascending=False)
    colors = g["camp"].map({"pro": _GREEN, "opp": _RED, "other": _GREY})
    ax[0].bar(range(len(g)), g["m"], color=colors, alpha=0.85,
              yerr=[g["m"] - g["lo"], g["hi"] - g["m"]], capsize=3)
    ax[0].set_xticks(range(len(g)))
    ax[0].set_xticklabels([f"{_OUTLET_LABEL.get(o, o)}\nn={n}" for o, n in zip(g["outlet"], g["n"])],
                          fontsize=8, rotation=40, ha="right")
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_ylabel("mean sentiment")
    ax[0].set_title("Outlet tone (green pro-gov, red opposition; 95% date-cluster CI)",
                    fontweight="bold", fontsize=10)

    # 2. Where the gap sits, by topic (descriptive; >=20 headlines per camp).
    topic = []
    for cat, grp in h[h["camp"] != "other"].groupby("category"):
        a, b = grp[grp["camp"] == "pro"]["sentiment"], grp[grp["camp"] == "opp"]["sentiment"]
        if len(a) >= 20 and len(b) >= 20:
            topic.append((_CAT_LABEL.get(cat, cat), a.mean() - b.mean()))
    t = pd.DataFrame(topic, columns=["cat", "gap"]).sort_values("gap")
    ax[1].barh(t["cat"], t["gap"], color="#6A1B9A", alpha=0.8)
    ax[1].axvline(0, color="black", lw=0.8)
    ax[1].set_xlabel("pro-gov minus opposition mean sentiment")
    ax[1].set_title("Gap by topic (>=20 headlines per camp)", fontweight="bold", fontsize=10)

    # 3. Weekly gap (weeks with >=5 headlines in each camp).
    camps = h[h["camp"] != "other"].copy()
    camps["week"] = camps["day"].dt.to_period("W").dt.start_time
    wk = camps.groupby(["week", "camp"])["sentiment"].agg(["mean", "size"]).unstack()
    wk = wk[(wk[("size", "pro")] >= 5) & (wk[("size", "opp")] >= 5)]
    gap = wk[("mean", "pro")] - wk[("mean", "opp")]
    ax[2].plot(gap.index, gap.values, color=_BLUE, marker="o", ms=3, lw=1.5)
    ax[2].axhline(gap.mean(), color="grey", ls="--", lw=0.8)
    ax[2].set_ylabel("pro-gov minus opposition")
    ax[2].set_title(f"Weekly gap (n={len(gap)} weeks)", fontweight="bold", fontsize=10)
    plt.setp(ax[2].get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Figure -> {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--out", default="docs/polarization.png")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    build(args.db, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
