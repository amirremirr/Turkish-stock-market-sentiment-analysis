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
    Context, Spec, base_result, coefficient, group_sufficient, insufficient,
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
    inference="OLS, standard errors clustered by day 0",
    sufficiency={"min_events_per_tercile": MIN_EVENTS_PER_GROUP,
                 "min_event_dates": MIN_EVENT_DATES_PER_GROUP},
    requires=("news_events", "point_in_time_market_cap"),
    sensitivity=("market-adjusted drift instead of market-model",
                 "immediate reaction CAR(0,+1) by tercile",
                 "positive and negative events separately"),
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
    fit = regress(usable, "signed_drift", xs, ["day0"])
    small = coefficient(fit, "small")
    result["primary"] = primary_block(
        SPEC, estimate=small["estimate"], se=small["se"], ci=small["ci"], p=small["p"],
        n=fit.get("n", 0), clusters=fit.get("clusters"),
        inference_status=fit.get("status", "ok"),
        extra={"n_dropped_missing": fit.get("n_dropped_missing")})

    for label, column in (("drift CAR(+2,+5)", "signed_drift"),
                          ("reaction CAR(0,+1)", "signed_reaction")):
        for tercile, group in usable.groupby("tercile"):
            test = mean_test(group[column], group["day0"])
            result["exploratory"].append({"test": f"{label}, {tercile}", **test})
    if len(usable) >= 5 * MIN_EVENTS_PER_GROUP:
        # Descriptive only: breakpoints use the whole sample, so this table
        # is not a forecast and feeds no decision.
        quintile = pd.qcut(usable["sentiment"], 5, labels=False, duplicates="drop")
        for q, group in usable.groupby(quintile):
            result["exploratory"].append({
                "test": f"CAR(0,+5), sentiment quintile {int(q) + 1} (full-sample breakpoints)",
                **mean_test(group["car_p0_p5"], group["day0"])})
    alt = coefficient(regress(usable, "signed_drift_mar", xs, ["day0"]), "small")
    result["sensitivity"].append({"variant": "market-adjusted drift", **alt})
    for name, mask in (("positive events", usable["direction"] > 0),
                       ("negative events", usable["direction"] < 0)):
        part = usable[mask]
        if part.groupby("tercile").size().reindex(["small", "large"]).fillna(0).min() \
                >= MIN_EVENTS_PER_GROUP:
            result["sensitivity"].append(
                {"variant": name, **coefficient(regress(part, "signed_drift", xs, ["day0"]), "small")})

    # An executable version: enter at the close of day +1, exit at the close of
    # day +5, in the direction of the news. Closing-auction fills are assumed.
    small_cap = usable[usable["tercile"] == "small"]
    result["execution"] = {
        "strategy": "small caps: trade in the sentiment direction from the close "
                    "of day +1 to the close of day +5, hedged with the index",
        "assumption": "fills at the closing auction on both days; not verified",
        **costs.net_summary(small_cap["signed_drift_mar"].tolist()),
    }
    result["tables"]["by_tercile"] = by_tercile.reset_index().to_dict("records")
    result["limitations"].append(
        "Sentiment comes from a model validated on market-level headlines, not "
        "on company news.")
    return result
