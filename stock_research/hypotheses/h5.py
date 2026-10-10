"""H5 -- limit-down reversal conditional on news."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stock_research import costs, limits
from stock_research.config import MIN_EVENT_DATES_PER_GROUP, MIN_EVENTS_PER_GROUP
from stock_research.eventstudy import STATUS_OK
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    Context, Spec, base_result, coefficient, insufficient, mean_test,
    missing_requirements, primary_block, regress,
)

NEGATIVE = -0.10          # mean sentiment at or below this is "negative news"
WINDOWS = ((1, 3), (1, 1))

SPEC = Spec(
    id="H5",
    title="Limit-down reversal conditional on news",
    claim="Limit-down sessions with no company news reverse more over the next "
          "three sessions than limit-down sessions with negative news.",
    null="CAR(+1,+3) after a limit-down close is the same with and without news.",
    alternative="CAR(+1,+3) is higher after a limit-down close with no news.",
    eligibility="Sessions that closed locked at the lower price limit under a "
                "verified limit rule, with no corporate action that day, "
                "classified as 'no news' (no confirmed story for the ticker "
                "acting on that session or the one before) or 'negative news' "
                "(mean sentiment of that session's stories at or below -0.10). "
                "Sessions with only neutral or positive news are set aside.",
    event_time="Day 0 is the limit-down session. News is classified using only "
               "stories whose day 0 is that session or the prior one.",
    outcome="CAR(+1,+3), market model.",
    estimand="coefficient on the no-news indicator in CAR(+1,+3) ~ no_news + "
             "prior_ret_5 + volatility_20 + log_turnover_20 + market_ret_0",
    direction=+1,
    material_effect=0.005,
    benchmark="market model on XU100",
    controls=("prior_ret_5", "volatility_20", "log_turnover_20", "market_ret_0"),
    inference="OLS, standard errors clustered by day 0",
    sufficiency={"min_events_per_group": MIN_EVENTS_PER_GROUP,
                 "min_event_dates": 2 * MIN_EVENT_DATES_PER_GROUP},
    requires=("news_events", "complete_company_news"),
    sensitivity=("sessions at one price all day only", "CAR(+1,+1)"),
    execution_dependent=True,
)


def limit_down_events(ctx: Context) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ticker, frame in ctx.panel.bars.items():
        if ticker == ctx.panel.benchmark:
            continue
        flags = limits.detect(frame, ctx.limit_rules)
        for day in flags.index[flags["down_locked_close"].to_numpy()]:
            if ctx.sealed is not None and day >= ctx.sealed:
                continue                 # a limit day is itself an outcome
            rows.append({"ticker": ticker, "day0": day,
                         "one_price": bool(flags.loc[day, "one_price"]),
                         "market_ret_0": ctx.panel.market.loc[day, "ret"]})
    return pd.DataFrame(rows, columns=["ticker", "day0", "one_price", "market_ret_0"])


def classify_news(ctx: Context, events: pd.DataFrame) -> pd.Series:
    """'no_news', 'negative_news' or 'other_news' for each limit-down session,
    from stories that could act on that session or the one before it."""

    stories = ctx.news_events
    stories = stories[stories["link_confirmed"].astype(bool) & stories["day0"].notna()]
    by_key = stories.groupby(["ticker", "day0"])["sentiment"].mean()
    labels = []
    for row in events.itertuples():
        previous = ctx.panel.calendar.offset(row.day0, -1)
        today = by_key.get((row.ticker, row.day0))
        before = by_key.get((row.ticker, previous)) if previous else None
        if today is None and before is None:
            labels.append("no_news")
        elif today is not None and np.isfinite(today) and today <= NEGATIVE:
            labels.append("negative_news")
        else:
            labels.append("other_news")
    return pd.Series(labels, index=events.index, dtype=object)


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    events = limit_down_events(ctx)
    counts: Dict[str, Any] = {"limit_down_closes": int(len(events))}
    if blocked:
        result["limitations"].append(
            f"{len(events)} limit-down closes were detected from prices; their "
            "news status cannot be established, so no comparison is made.")
        return insufficient(result, blocked, counts)
    if events.empty:
        return insufficient(result, ["no limit-down closes detected"], counts)

    events["news"] = classify_news(ctx, events)
    counts["by_news"] = {str(k): int(v) for k, v in events["news"].value_counts().items()}
    table = prep.study(ctx, events, windows=WINDOWS)
    counts["studied"] = prep.status_counts(table)
    usable = table[(table["status"] == STATUS_OK)
                   & table["news"].isin(["no_news", "negative_news"])].copy()
    usable["no_news"] = (usable["news"] == "no_news").astype(float)
    sizes = usable.groupby("news").size()
    if sizes.reindex(["no_news", "negative_news"]).fillna(0).min() < MIN_EVENTS_PER_GROUP \
            or usable["day0"].nunique() < 2 * MIN_EVENT_DATES_PER_GROUP:
        return insufficient(result, [
            f"groups {sizes.to_dict()} on {usable['day0'].nunique()} dates; needs "
            f"{MIN_EVENTS_PER_GROUP} per group on {2 * MIN_EVENT_DATES_PER_GROUP} dates"], counts)
    result["sufficiency"]["counts"] = counts

    xs = ["no_news", "prior_ret_5", "volatility_20", "log_turnover_20", "market_ret_0"]
    fit = regress(usable, "car_p1_p3", xs, ["day0"])
    main = coefficient(fit, "no_news")
    result["primary"] = primary_block(
        SPEC, estimate=main["estimate"], se=main["se"], ci=main["ci"], p=main["p"],
        n=fit.get("n", 0), clusters=fit.get("clusters"),
        inference_status=fit.get("status", "ok"),
        extra={"n_dropped_missing": fit.get("n_dropped_missing")})
    for group, part in usable.groupby("news"):
        result["exploratory"].append({"test": f"mean CAR(+1,+3), {group}",
                                      **mean_test(part["car_p1_p3"], part["day0"])})
    result["sensitivity"].append({
        "variant": "one price all day",
        **coefficient(regress(usable[usable["one_price"].astype(bool)], "car_p1_p3", xs, ["day0"]), "no_news")})
    result["sensitivity"].append({
        "variant": "CAR(+1,+1)",
        **coefficient(regress(usable, "car_p1_p1", xs, ["day0"]), "no_news")})

    # The statistic above starts at the locked close. Whether an order there
    # would have traded cannot be seen in daily bars, so the executable version
    # enters at the next session's open, and only where that open was not
    # itself locked.
    quiet = usable[usable["no_news"] == 1]
    trades, fills = [], {}
    for row in quiet.itertuples():
        frame = ctx.panel.frame(row.ticker)
        entry, exit_ = (ctx.panel.calendar.offset(row.day0, 1),
                        ctx.panel.calendar.offset(row.day0, 3))
        fill = costs.entry_fill(frame, entry, limit=limits.limit_on(entry or "", ctx.limit_rules))
        fills[fill] = fills.get(fill, 0) + 1
        if fill == costs.FILL_OK and exit_ in frame.index and np.isfinite(frame.loc[exit_, "close"]):
            between = frame.loc[entry:exit_]
            if not between["corporate_action"].any():
                market = ctx.panel.market
                trades.append(frame.loc[exit_, "close"] / frame.loc[entry, "open"]
                              - market.loc[exit_, "close"] / market.loc[entry, "open"])
    result["execution"] = {
        "strategy": "after a no-news limit-down close, buy at the next open, "
                    "sell at the close two sessions later, hedged with the index",
        "assumption": "the limit-down close itself is not assumed tradable; "
                      "entry fills at the next opening print where it was not locked",
        "coverage": {"signals": int(len(quiet)), "entry_fills": fills},
        "verified": False,
        **costs.net_summary(trades),
    }
    result["limitations"] += [
        "No order-book data: fills at or near a price limit are unverified.",
        "'No news' means no story in the sources this project collects.",
    ]
    return result
