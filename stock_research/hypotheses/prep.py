"""Shared preparation: which events are admissible, and one studied row each.

Every filter here is counted. A hypothesis reports how many events it started
with and how many each rule removed, so a small final sample can be traced to
the rule that made it small.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from stock_research.calendar import CLEAN_REACTION_BUCKETS
from stock_research.config import EVENT_WINDOWS
from stock_research.eventstudy import STATUS_OK, event_table
from stock_research.hypotheses.common import Context, prior_features

NEWS_COLUMNS = ("event_id", "ticker", "day0", "bucket", "timing_ambiguous", "sentiment",
                "n_articles", "mention_type", "n_issuers", "link_confirmed")


def _count(counts: Dict[str, int], label: str, frame: pd.DataFrame) -> pd.DataFrame:
    counts[label] = int(len(frame))
    return frame


def admissible_news(ctx: Context, *, clean_timing: bool = True,
                    need_sentiment: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Company-specific, confirmed, single-issuer news with usable timing,
    collapsed to one row per ticker and day 0."""

    counts: Dict[str, int] = {}
    events = ctx.news_events
    if events is None or events.empty or not set(NEWS_COLUMNS) <= set(events.columns):
        return pd.DataFrame(columns=list(NEWS_COLUMNS)), {"input": 0}

    frame = _count(counts, "input", events)
    frame = _count(counts, "confirmed_link", frame[frame["link_confirmed"].astype(bool)])
    frame = _count(counts, "single_issuer", frame[frame["n_issuers"] == 1])
    frame = _count(counts, "material_mention", frame[frame["mention_type"] == "material"])
    frame = _count(counts, "day0_known", frame[frame["day0"].notna()])
    frame = _count(counts, "timing_unambiguous", frame[~frame["timing_ambiguous"].astype(bool)])
    if clean_timing:
        frame = _count(counts, "clean_reaction_bucket",
                       frame[frame["bucket"].isin(CLEAN_REACTION_BUCKETS)])
    if need_sentiment:
        frame = _count(counts, "has_sentiment", frame[frame["sentiment"].notna()])
    if frame.empty:
        return frame, counts

    collapsed = (frame.groupby(["ticker", "day0"], as_index=False)
                 .agg(event_id=("event_id", "first"), sentiment=("sentiment", "mean"),
                      n_articles=("n_articles", "sum"), n_events=("event_id", "size"),
                      bucket=("bucket", "first")))
    counts["ticker_days"] = int(len(collapsed))
    return collapsed, counts


def study(ctx: Context, events: pd.DataFrame,
          windows: Sequence[Tuple[int, int]] = EVENT_WINDOWS,
          *, with_features: bool = True) -> pd.DataFrame:
    """Event-study outputs and prior-only controls for each event row."""

    if events.empty:
        return events.assign(status=pd.Series(dtype=str))
    table = event_table(ctx.panel, events.to_dict("records"), windows=windows,
                        sealed=ctx.sealed)
    if with_features:
        features = [prior_features(ctx.panel, row["ticker"], row["day0"])
                    for row in events.to_dict("records")]
        table = pd.concat([table.reset_index(drop=True), pd.DataFrame(features)], axis=1)
    return table


def status_counts(table: pd.DataFrame) -> Dict[str, int]:
    if "status" not in table or table.empty:
        return {}
    return {str(k): int(v) for k, v in table["status"].value_counts().items()}


def size_tercile(ctx: Context, table: pd.DataFrame) -> pd.Series:
    """Market-cap tercile as of the last date strictly before day 0.

    Breakpoints are the terciles of all stocks' capitalisations on that same
    prior date, so neither a stock's later growth nor the later cross-section
    can leak in. Missing size stays missing.
    """

    result = pd.Series(pd.NA, index=table.index, dtype="object")
    if ctx.size is None or ctx.size.empty:
        return result
    size = ctx.size.dropna(subset=["market_cap"]).sort_values("date")
    by_date = {d: g.set_index("ticker")["market_cap"] for d, g in size.groupby("date")}
    dates = sorted(by_date)
    for index, row in table.iterrows():
        day0 = row.get("day0")
        if not isinstance(day0, str):
            continue
        position = np.searchsorted(dates, day0) - 1      # strictly before day 0
        if position < 0:
            continue
        cross = by_date[dates[position]]
        if row["ticker"] not in cross.index or len(cross) < 9:
            continue
        low, high = cross.quantile([1 / 3, 2 / 3])
        value = cross[row["ticker"]]
        result.loc[index] = "small" if value <= low else ("large" if value > high else "mid")
    return result
