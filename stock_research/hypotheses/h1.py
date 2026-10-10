"""H1 -- small-cap news underreaction."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from stock_research import costs
from stock_research.config import MIN_EVENT_DATES_PER_GROUP, MIN_EVENTS_PER_GROUP
from stock_research.eventstudy import STATUS_OK
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    TWO_WAY, Context, Spec, base_result, coefficient, group_sufficient, insufficient,
    mean_test, missing_requirements, primary_block, regress,
)

SPEC = Spec(
    id="H1",
    title="Small-cap news underreaction",
    claim="Small-cap BIST equities show stronger sentiment-aligned drift after "
          "news than large-cap equities.",
    null="Signed abnormal drift over sessions +2..+5 is the same for small-cap "
         "and large-cap stocks.",
    alternative="Signed drift over +2..+5 is larger for small caps.",
    eligibility="Confirmed, single-issuer, material news with non-zero sentiment; "
                "published before the open, after the close, or on a non-session "
                "day; one row per ticker and day 0; market-model CAR available; "
                "point-in-time size tercile known.",
    event_time="Day 0 is the first session able to act on the news.",
    outcome="signed_drift = sign(sentiment) x CAR(+2,+5), market model.",
    estimand="coefficient on the small-cap indicator (large caps as the "
             "reference) in signed_drift ~ small + mid + prior_ret_5 + log_turnover_20",
    direction=+1,
    material_effect=0.005,
    benchmark="market model on XU100, estimation window -130..-11",
    controls=("mid-cap indicator", "prior_ret_5", "log_turnover_20"),
    inference="OLS, standard errors clustered two ways, by day 0 and by issuer",
    sufficiency={"min_events_per_tercile": MIN_EVENTS_PER_GROUP,
                 "min_event_dates": MIN_EVENT_DATES_PER_GROUP},
    requires=("news_events", "point_in_time_market_cap"),
    sensitivity=("clustering by day 0 only",
                 "market-adjusted drift instead of market-model",
                 "immediate reaction CAR(0,+1) by tercile",
                 "positive and negative events separately"),
    parameters={"drift_window": [2, 5], "reaction_window": [0, 1],
                "terciles": "of all stocks' capitalisation on the last date before day 0"},
)


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)

    events, counts = prep.admissible_news(ctx)
    if events.empty:
        return insufficient(result, ["no admissible news events"], counts)
    events = events[events["sentiment"] != 0]
    table = prep.study(ctx, events)
    table["tercile"] = prep.size_tercile(ctx, table)
    counts["studied"] = prep.status_counts(table)
    usable = table[(table["status"] == STATUS_OK) & table["tercile"].notna()
                   & table["car_p2_p5"].notna()].copy()
    usable["direction"] = np.sign(usable["sentiment"])
    usable["signed_drift"] = usable["direction"] * usable["car_p2_p5"]
    usable["signed_drift_mar"] = usable["direction"] * usable["mar_p2_p5"]
    usable["signed_reaction"] = usable["direction"] * usable["car_p0_p1"]
    usable["small"] = (usable["tercile"] == "small").astype(float)
    usable["mid"] = (usable["tercile"] == "mid").astype(float)

    by_tercile = usable.groupby("tercile").agg(events=("ticker", "size"),
                                               dates=("day0", "nunique"))
    counts["by_tercile"] = by_tercile.to_dict("index")
    short = [t for t in ("small", "mid", "large")
             if t not in by_tercile.index
             or by_tercile.loc[t, "events"] < MIN_EVENTS_PER_GROUP]
    if short or usable["day0"].nunique() < MIN_EVENT_DATES_PER_GROUP:
        return insufficient(result, [
            f"terciles below {MIN_EVENTS_PER_GROUP} events: {short or 'none'}; "
            f"{usable['day0'].nunique()} event dates"], counts)
    result["sufficiency"]["counts"] = counts

    xs = ["small", "mid", "prior_ret_5", "log_turnover_20"]
    fit = regress(usable, "signed_drift", xs, TWO_WAY)
    small = coefficient(fit, "small")
    result["primary"] = primary_block(
        SPEC, estimate=small["estimate"], se=small["se"], ci=small["ci"], p=small["p"],
        n=fit.get("n", 0), clusters=fit.get("clusters"),
        inference_status=fit.get("status", "ok"),
        extra={"n_dropped_missing": fit.get("n_dropped_missing")})

    for label, column in (("drift CAR(+2,+5)", "signed_drift"),
                          ("reaction CAR(0,+1)", "signed_reaction")):
        for tercile, group in usable.groupby("tercile"):
            test = mean_test(group[column], group["day0"], group["ticker"])
            result["exploratory"].append({"test": f"{label}, {tercile}", **test})
    if len(usable) >= 5 * MIN_EVENTS_PER_GROUP:
        # Descriptive only: breakpoints use the whole sample, so this table
        # is not a forecast and feeds no decision.
        quintile = pd.qcut(usable["sentiment"], 5, labels=False, duplicates="drop")
        for q, group in usable.groupby(quintile):
            result["exploratory"].append({
                "test": f"CAR(0,+5), sentiment quintile {int(q) + 1} (full-sample breakpoints)",
                **mean_test(group["car_p0_p5"], group["day0"], group["ticker"])})
    result["sensitivity"].append({
        "variant": "clustering by day 0 only",
        **coefficient(regress(usable, "signed_drift", xs, ["day0"]), "small")})
    alt = coefficient(regress(usable, "signed_drift_mar", xs, TWO_WAY), "small")
    result["sensitivity"].append({"variant": "market-adjusted drift", **alt})
    for name, mask in (("positive events", usable["direction"] > 0),
                       ("negative events", usable["direction"] < 0)):
        part = usable[mask]
        if part.groupby("tercile").size().reindex(["small", "large"]).fillna(0).min() \
                >= MIN_EVENTS_PER_GROUP:
            result["sensitivity"].append(
                {"variant": name, **coefficient(regress(part, "signed_drift", xs, TWO_WAY), "small")})

    # An executable version, long only: after positive news on a small cap, buy
    # at the close of day +1 and sell at the close of day +5, hedged with the
    # index. The mirror trade after negative news needs a short sale in a small
    # cap, which is not assumed to be available.
    longs = usable[(usable["tercile"] == "small") & (usable["direction"] > 0)]
    result["execution"] = {
        "strategy": "small caps, positive news only: buy at the close of day +1, "
                    "sell at the close of day +5, hedged with the index",
        "assumption": "fills at the closing auction on both days; an index hedge "
                      "is available; neither is verified",
        "not_assumed": "short sales after negative news",
        "coverage": {"small_cap_events": int((usable["tercile"] == "small").sum()),
                     "long_signals": int(len(longs))},
        **costs.net_summary(longs["mar_p2_p5"].tolist(), round_trips=2),
    }
    result["tables"]["by_tercile"] = by_tercile.reset_index().to_dict("records")
    result["limitations"].append(
        "Sentiment comes from a model validated on market-level headlines, not "
        "on company news.")
    return result
