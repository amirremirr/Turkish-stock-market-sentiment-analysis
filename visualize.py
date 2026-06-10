"""
Visualisation module — generates a three-panel figure:

  Panel 1  BIST 100 closing price (line)
  Panel 2  Daily sentiment score  (bar chart, coloured red/green)
           + 5-day rolling average (dashed line)
  Panel 3  Scatter: today's sentiment vs next-day BIST 100 return
           + OLS regression line + Pearson r annotation

Saves to PNG and optionally opens the interactive window.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

import database as db
from config import (
    DB_PATH, MINIMUM_HEADLINES_PER_DAY, MINIMUM_OVERLAP_DAYS,
    PLOT_DPI, PLOT_DAYS, PLOT_OUTPUT,
)

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────

_STYLE = "seaborn-v0_8-darkgrid"
_C_PRICE   = "#2C7BB6"   # blue for BIST 100
_C_POS     = "#4CAF50"   # green for positive sentiment bars
_C_NEG     = "#F44336"   # red for negative sentiment bars
_C_NEUTRAL = "#9E9E9E"   # grey
_C_ROLL    = "#FF9800"   # orange rolling average
_C_REG     = "#9C27B0"   # purple regression line


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data(
    db_path: str,
    days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (prices_df, sentiment_df, merged_df).
    All date columns are parsed as datetime.date objects.
    merged_df has columns: date, close, daily_return, avg_score, next_return.
    """
    start = (date.today() - timedelta(days=days)).isoformat()

    prices = db.get_prices(start=start, db_path=db_path)
    sent   = db.get_daily_sentiment(start=start, db_path=db_path)

    if prices.empty or sent.empty:
        return prices, sent, pd.DataFrame()

    prices["date"] = pd.to_datetime(prices["date"])
    sent["date"]   = pd.to_datetime(sent["date"])

    # next_return must be computed on the consecutive price series BEFORE the
    # merge: shifting after an inner join pairs a day with the next surviving
    # row, which is not the next trading day whenever the overlap has gaps.
    prices = prices.sort_values("date")
    prices["next_return"] = prices["daily_return"].shift(-1)  # true t+1 return

    merged = pd.merge(prices, sent, on="date", how="inner").sort_values("date")

    return prices, sent, merged


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_sentiment_vs_price(
    db_path: str = DB_PATH,
    days: int = PLOT_DAYS,
    output_path: str = PLOT_OUTPUT,
    show: bool = True,
) -> Optional[str]:
    """
    Generate the three-panel figure.

    Returns the output path on success, None if there's insufficient data.
    """
    try:
        plt.style.use(_STYLE)
    except OSError:
        pass  # fallback to default if style not found

    prices, sent, merged = _load_data(db_path, days)

    if merged.empty or len(merged) < 5:
        logger.warning(
            "Not enough overlapping data to plot "
            "(prices=%d rows, sentiment=%d rows, overlap=%d).",
            len(prices), len(sent), len(merged),
        )
        return None

    # Reliable days only for scatter/rolling: same gate as evaluate.py L5.
    # Thin days (< MINIMUM_HEADLINES_PER_DAY) stay in panels 1 & 2 (hatched)
    # but are excluded from OLS and rolling correlation to avoid noisy points.
    reliable_merged = merged[merged["headline_count"] >= MINIMUM_HEADLINES_PER_DAY]
    thin_excluded = len(merged) - len(reliable_merged)
    if thin_excluded:
        logger.info(
            "Scatter/rolling: excluded %d thin day(s) (< %d headlines)",
            thin_excluded, MINIMUM_HEADLINES_PER_DAY,
        )

    # Soft gate: flag low-overlap windows but still render panels 1 & 2.
    # Scatter / rolling correlation panels will show a PRELIMINARY watermark.
    insufficient = len(reliable_merged) < MINIMUM_OVERLAP_DAYS
    if insufficient:
        logger.warning(
            "Only %d reliable overlapping days (need %d for reliable signal stats). "
            "Scatter and rolling-correlation panels are marked PRELIMINARY.",
            len(reliable_merged), MINIMUM_OVERLAP_DAYS,
        )

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 13))
    gs  = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 1, 1.1],
        hspace=0.42,
        wspace=0.35,
    )

    ax_price   = fig.add_subplot(gs[0, :])    # full-width top
    ax_sent    = fig.add_subplot(gs[1, :])    # full-width middle
    ax_scatter = fig.add_subplot(gs[2, 0])    # bottom-left
    ax_roll    = fig.add_subplot(gs[2, 1])    # bottom-right

    dates = np.array(prices["date"].dt.to_pydatetime())
    sent_dates = np.array(sent["date"].dt.to_pydatetime())

    # ── Panel 1: BIST 100 price ───────────────────────────────────────────────
    ax_price.plot(
        prices["date"], prices["close"],
        color=_C_PRICE, linewidth=1.8, label="BIST 100 (kapanış)",
    )
    ax_price.fill_between(
        prices["date"], prices["close"],
        prices["close"].min() * 0.99,
        alpha=0.12, color=_C_PRICE,
    )
    ax_price.set_title("BIST 100 Kapanış Fiyatı", fontsize=13, fontweight="bold")
    ax_price.set_ylabel("Endeks")
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_price.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_price.legend(fontsize=9)

    # ── Panel 2: Daily sentiment bars ─────────────────────────────────────────
    scores = sent["avg_score"].values
    colors = [_C_POS if s >= 0 else _C_NEG for s in scores]
    bar_width = max(0.6, 0.8 * (days / 90))

    # Separate reliable days from thin days (headline_count < threshold)
    thin_mask    = sent["headline_count"] < MINIMUM_HEADLINES_PER_DAY
    reliable_sent = sent[~thin_mask]
    thin_sent     = sent[thin_mask]
    reliable_colors = [_C_POS if s >= 0 else _C_NEG for s in reliable_sent["avg_score"].values]
    thin_colors     = [_C_POS if s >= 0 else _C_NEG for s in thin_sent["avg_score"].values]

    ax_sent.bar(reliable_sent["date"], reliable_sent["avg_score"],
                color=reliable_colors, width=bar_width, alpha=0.85,
                label="Günlük ortalama duygu skoru")
    if not thin_sent.empty:
        ax_sent.bar(thin_sent["date"], thin_sent["avg_score"],
                    color=thin_colors, width=bar_width, alpha=0.45,
                    hatch="//", edgecolor="white",
                    label=f"Yetersiz veri (<{MINIMUM_HEADLINES_PER_DAY} haber)")

    # Rolling 5-day average
    if len(sent) >= 5:
        roll = sent["avg_score"].rolling(5, min_periods=3).mean()
        ax_sent.plot(
            sent["date"], roll,
            color=_C_ROLL, linewidth=2, linestyle="--",
            label="5 günlük ortalama", zorder=5,
        )

    ax_sent.axhline(0, color="white", linewidth=0.8, linestyle="-", alpha=0.5)
    ax_sent.set_title("Günlük Duygu Skoru  (XLM-RoBERTa)", fontsize=13, fontweight="bold")
    ax_sent.set_ylabel("Skor  [−1 … +1]")
    ax_sent.set_ylim(-1.05, 1.05)
    ax_sent.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_sent.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_sent.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_sent.legend(fontsize=9)

    # Headline count annotation on bars
    if len(sent) <= 60:
        for _, row in sent.iterrows():
            ax_sent.annotate(
                str(int(row["headline_count"])),
                xy=(row["date"], row["avg_score"]),
                xytext=(0, 4 if row["avg_score"] >= 0 else -10),
                textcoords="offset points",
                ha="center", fontsize=6, color="white", alpha=0.7,
            )

    # ── Panel 3a: Scatter sentiment → next-day return ─────────────────────────
    # Uses reliable_merged only (thin days excluded — same gate as evaluate.py L5)
    valid = reliable_merged.dropna(subset=["avg_score", "next_return"])

    if len(valid) >= 5:
        x = valid["avg_score"].values
        y = valid["next_return"].values

        ax_scatter.scatter(
            x, y,
            c=valid["next_return"].values,
            cmap="RdYlGn",
            s=50, alpha=0.7, edgecolors="white", linewidths=0.4,
        )

        # OLS regression
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax_scatter.plot(x_line, y_line, color=_C_REG, linewidth=1.8, label="OLS")

        sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
        thin_note = f"\n({thin_excluded} thin days excluded)" if thin_excluded else ""
        ax_scatter.annotate(
            f"Pearson r = {r_value:.3f}{sig}\np = {p_value:.3f}  (n={len(valid)}){thin_note}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )
        ax_scatter.legend(fontsize=9)
    else:
        ax_scatter.text(0.5, 0.5, "Yeterli veri yok", transform=ax_scatter.transAxes,
                        ha="center", va="center")

    ax_scatter.set_xlabel("Duygu Skoru (t)")
    ax_scatter.set_ylabel("BIST 100 Getirisi % (t+1)")
    ax_scatter.set_title("Duygu → Ertesi Gün Getiri", fontsize=11, fontweight="bold")
    ax_scatter.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax_scatter.axvline(0, color="grey", linewidth=0.6, linestyle="--")

    if insufficient:
        ax_scatter.text(
            0.5, 0.5,
            f"PRELIMINARY\n{len(reliable_merged)} gün  (min {MINIMUM_OVERLAP_DAYS} gerekli)",
            transform=ax_scatter.transAxes,
            ha="center", va="center", fontsize=11, color="red", alpha=0.55,
            rotation=15,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.6),
        )

    # ── Panel 3b: Rolling 30-day correlation ─────────────────────────────────
    # Also uses reliable_merged only — consistent with scatter panel and evaluate.py L5
    if len(valid) >= 15:
        roll_window = min(30, len(valid))
        roll_corr = (
            valid.set_index("date")[["avg_score", "next_return"]]
            .rolling(roll_window, min_periods=10)
            .corr()
            .unstack()["avg_score"]["next_return"]
            .dropna()
        )
        ax_roll.plot(
            roll_corr.index, roll_corr.values,
            color=_C_REG, linewidth=1.8,
        )
        ax_roll.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_roll.fill_between(
            roll_corr.index, roll_corr.values, 0,
            where=(roll_corr.values > 0), alpha=0.25, color=_C_POS,
        )
        ax_roll.fill_between(
            roll_corr.index, roll_corr.values, 0,
            where=(roll_corr.values < 0), alpha=0.25, color=_C_NEG,
        )
        ax_roll.set_ylabel("Pearson r")
        ax_roll.set_title(f"{roll_window} Günlük Yuvarlanan Korelasyon",
                          fontsize=11, fontweight="bold")
        ax_roll.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax_roll.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax_roll.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax_roll.set_ylim(-1.05, 1.05)
    else:
        ax_roll.text(0.5, 0.5, "Yeterli veri yok\n(min 15 gün gerekli)",
                     transform=ax_roll.transAxes, ha="center", va="center")
        ax_roll.set_title("30 Günlük Yuvarlanan Korelasyon", fontsize=11, fontweight="bold")

    if insufficient:
        ax_roll.text(
            0.5, 0.5,
            f"PRELIMINARY\n{len(reliable_merged)} gün  (min {MINIMUM_OVERLAP_DAYS} gerekli)",
            transform=ax_roll.transAxes,
            ha="center", va="center", fontsize=11, color="red", alpha=0.55,
            rotation=15,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.6),
        )

    # ── Supertitle ────────────────────────────────────────────────────────────
    data_range = (
        f"{sent['date'].min().strftime('%d %b %Y')} – "
        f"{sent['date'].max().strftime('%d %b %Y')}"
    )
    fig.suptitle(
        f"BIST 100 Piyasa Duygu Analizi\n{data_range}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    logger.info("Saved plot -> %s", output_path)
    print(f"\nPlot saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return output_path
