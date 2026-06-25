"""
Descriptive analysis of the news corpus itself (not the market signal).

This is exploratory *description*, not prediction — it needs no minimum sample
of overlap days and carries no overfitting risk. It answers: what does Turkish
financial news look like, and do outlets and topics differ systematically?

Outputs:
  docs/corpus_overview.png   4-panel figure (topics, topic sentiment,
                             outlet comparison, coverage over time)
  docs/corpus_findings.md    auto-written summary of the headline findings

Usage:  python analyze_corpus.py
"""

import argparse
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DB_PATH

_CAT_LABEL = {
    "fx_lira": "Currency / lira", "turkey_macro": "Turkish economy",
    "energy_commodities": "Energy & commodities", "rates_tcmb": "Rates / TCMB",
    "bist_company": "Companies / BIST", "global_risk": "Global markets",
    "political_risk": "Political risk", "banks": "Banking", "crypto": "Crypto",
    "other": "Other",
}
_GREEN, _RED, _BLUE, _GREY = "#2E7D32", "#C62828", "#1565C0", "#9E9E9E"


def load(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT source, published_at, category, sentiment_score,
                  sentiment_label, COALESCE(relevance, 1.0) AS relevance
           FROM headlines
           WHERE sentiment_score IS NOT NULL AND published_at IS NOT NULL""",
        con,
    )
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def findings(df: pd.DataFrame) -> list[str]:
    out = []
    n, d0, d1 = len(df), df["published_at"].min().date(), df["published_at"].max().date()
    out.append(f"- **{n:,} scored headlines** span {d0} to {d1} "
               f"({df['source'].nunique()} sources, {df['category'].nunique()} topics).")

    cat = df["category"].value_counts()
    out.append(f"- Most-covered topic: **{_CAT_LABEL.get(cat.index[0], cat.index[0])}** "
               f"({cat.iloc[0]/n:.0%}); least: {_CAT_LABEL.get(cat.index[-1], cat.index[-1])}.")

    cs = df.groupby("category")["sentiment_score"].mean().sort_values()
    out.append(f"- Most *bearish* topic on average: **{_CAT_LABEL.get(cs.index[0], cs.index[0])}** "
               f"({cs.iloc[0]:+.2f}); most *bullish*: "
               f"**{_CAT_LABEL.get(cs.index[-1], cs.index[-1])}** ({cs.iloc[-1]:+.2f}).")

    src = df.groupby("source").agg(n=("source", "size"),
                                   sent=("sentiment_score", "mean"),
                                   rel=("relevance", "mean"))
    src = src[src["n"] >= 20].sort_values("rel")
    if len(src):
        lo, hi = src.iloc[0], src.iloc[-1]
        out.append(f"- Outlets differ markedly: **{src.index[0]}** is the noisiest "
                   f"(avg relevance {lo['rel']:.2f}), **{src.index[-1]}** the most "
                   f"on-topic ({hi['rel']:.2f}).")
        sb = src.sort_values("sent")
        out.append(f"- Most-bearish outlet: **{sb.index[0]}** ({sb['sent'].iloc[0]:+.2f}); "
                   f"most-bullish: **{sb.index[-1]}** ({sb['sent'].iloc[-1]:+.2f}).")
    return out


def figure(df: pd.DataFrame, out_path: str) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Turkish Financial News Corpus — Descriptive Overview",
                 fontsize=16, fontweight="bold", y=0.98)

    # A: topic coverage
    ax = axes[0, 0]
    cat = df["category"].value_counts()
    labels = [_CAT_LABEL.get(c, c) for c in cat.index]
    ax.barh(labels[::-1], cat.values[::-1], color=_BLUE, alpha=0.85)
    ax.set_title("What the news is about", fontweight="bold")
    ax.set_xlabel("headlines")

    # B: mean sentiment by topic
    ax = axes[0, 1]
    cs = df.groupby("category")["sentiment_score"].mean().sort_values()
    cols = [_GREEN if v >= 0 else _RED for v in cs.values]
    ax.barh([_CAT_LABEL.get(c, c) for c in cs.index], cs.values, color=cols, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Average mood by topic", fontweight="bold")
    ax.set_xlabel("mean sentiment  (-1 bearish … +1 bullish)")

    # C: outlet comparison — relevance vs sentiment, bubble = volume
    ax = axes[1, 0]
    src = df.groupby("source").agg(n=("source", "size"),
                                   sent=("sentiment_score", "mean"),
                                   rel=("relevance", "mean"))
    src = src[src["n"] >= 20]
    ax.scatter(src["rel"], src["sent"], s=src["n"] * 2.5, alpha=0.55,
               color=_BLUE, edgecolors="black", linewidths=0.5)
    for name, r in src.iterrows():
        ax.annotate(name, (r["rel"], r["sent"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_title("Outlets differ: noise vs mood (bubble = volume)", fontweight="bold")
    ax.set_xlabel("avg relevance  (1.0 = squarely market news)")
    ax.set_ylabel("avg sentiment")

    # D: coverage over time
    ax = axes[1, 1]
    daily = df.set_index("published_at").resample("D").size()
    ax.plot(daily.index, daily.values, color=_BLUE, lw=1.5)
    ax.fill_between(daily.index, daily.values, alpha=0.15, color=_BLUE)
    ax.set_title("News volume over time", fontweight="bold")
    ax.set_ylabel("headlines / day")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Descriptive corpus analysis")
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--output", default="docs/corpus_overview.png")
    args = p.parse_args()

    df = load(args.db)
    if df.empty:
        print("No scored headlines to analyse.")
        return

    lines = findings(df)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\nCorpus findings:")
    for ln in lines:
        print("  " + ln.replace("**", ""))

    figure(df, args.output)
    print(f"\nFigure -> {args.output}")

    with open("docs/corpus_findings.md", "w", encoding="utf-8") as f:
        f.write("# News corpus — descriptive findings\n\n")
        f.write("Auto-generated by `analyze_corpus.py` (exploratory description, "
                "not prediction).\n\n")
        f.write("\n".join(lines) + "\n\n")
        f.write(f"![corpus overview]({args.output.split('/')[-1]})\n")
    print("Findings -> docs/corpus_findings.md")


if __name__ == "__main__":
    main()
