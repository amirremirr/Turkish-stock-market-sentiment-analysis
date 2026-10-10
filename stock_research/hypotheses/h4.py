"""H4 -- abnormal news coverage and reversal."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from stock_research.eventstudy import STATUS_OK
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    Context, Spec, base_result, coefficient, insufficient, missing_requirements,
    primary_block, regress,
)

LOOKBACK_SESSIONS = 60
MIN_COVERAGE_DAYS = 3
MIN_TICKER_DAYS = 200
MIN_DATES = 30

SPEC = Spec(
    id="H4",
    title="Abnormal news coverage and reversal",
    claim="Unexpectedly heavy coverage of a company is followed by negative "
          "abnormal returns over the next 5 to 20 sessions.",
    null="Abnormal coverage is unrelated to CAR(+1,+10).",
    alternative="Higher abnormal coverage is followed by lower CAR(+1,+10).",
    eligibility="Ticker-days with at least one confirmed news story, where the "
                "ticker had coverage on at least three of the prior 60 sessions; "
                "syndicated copies of one story count once.",
    event_time="Day 0 is the first session able to act on the day's stories.",
    outcome="CAR(+1,+10), market model.",
    estimand="coefficient on abnormal coverage, log(stories today / mean daily "
             "stories over the prior 60 sessions), in CAR(+1,+10) ~ abnormal_coverage "
             "+ mean_sentiment + prior_ret_5 + volatility_20 + log_turnover_20 + volume_ratio",
    direction=-1,
    material_effect=0.002,
    benchmark="market model on XU100",
    controls=("mean_sentiment", "prior_ret_5", "volatility_20", "log_turnover_20",
              "volume_ratio"),
    inference="OLS, standard errors clustered by day 0",
    sufficiency={"min_ticker_days": MIN_TICKER_DAYS, "min_dates": MIN_DATES,
                 "min_prior_coverage_days": MIN_COVERAGE_DAYS},
    requires=("news_events",),
    sensitivity=("CAR(+1,+5) and CAR(+1,+20)", "two-way clustering by date and ticker",
                 "excluding days with a KAP disclosure", "excluding scheduled macro days"),
)


def abnormal_coverage(stories: pd.DataFrame, calendar) -> pd.DataFrame:
    """Daily story counts per ticker and their excess over a prior-only baseline.

    ``stories`` has one row per deduplicated story and ticker. The baseline is
    the mean daily count over the 60 sessions strictly before day 0, zeros
    included. Fewer than three covered sessions in that window leaves the
    measure missing: a stock nobody wrote about has no "normal" to exceed.
    """

    daily = (stories.dropna(subset=["day0"]).groupby(["ticker", "day0"], as_index=False)
             .agg(stories=("event_id", "nunique"), mean_sentiment=("sentiment", "mean")))
    daily["_pos"] = daily["day0"].map(calendar.index)
    daily = daily.dropna(subset=["_pos"])
    daily["abnormal_coverage"] = np.nan
    for _, group in daily.groupby("ticker"):
        pos = group["_pos"].to_numpy()
        count = group["stories"].to_numpy(dtype=float)
        for i, index in enumerate(group.index):
            earlier = (pos < pos[i]) & (pos >= pos[i] - LOOKBACK_SESSIONS)
            if earlier.sum() < MIN_COVERAGE_DAYS:
                continue
            baseline = count[earlier].sum() / LOOKBACK_SESSIONS
            daily.loc[index, "abnormal_coverage"] = np.log(count[i] / baseline)
    return daily.drop(columns="_pos")


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)

    stories = ctx.news_events
    stories = stories[stories["link_confirmed"].astype(bool)
                      & ~stories["timing_ambiguous"].astype(bool)] if not stories.empty else stories
    counts = {"confirmed_stories": int(len(stories))}
    if stories.empty:
        return insufficient(result, ["no confirmed news stories"], counts)

    daily = abnormal_coverage(stories, ctx.panel.calendar)
    counts["ticker_days"] = int(len(daily))
    eligible = daily.dropna(subset=["abnormal_coverage"])
    counts["with_baseline"] = int(len(eligible))
    table = prep.study(ctx, eligible)
    counts["studied"] = prep.status_counts(table)
    usable = table[table["status"] == STATUS_OK].copy() if not table.empty else table
    if len(usable) < MIN_TICKER_DAYS or usable["day0"].nunique() < MIN_DATES:
        return insufficient(result, [
            f"{len(usable)} ticker-days on {usable['day0'].nunique() if len(usable) else 0} "
            f"dates; needs {MIN_TICKER_DAYS} on {MIN_DATES}"], counts)
    result["sufficiency"]["counts"] = counts

    xs = ["abnormal_coverage", "mean_sentiment", "prior_ret_5", "volatility_20",
          "log_turnover_20", "volume_ratio"]
    fit = regress(usable, "car_p1_p10", xs, ["day0"])
    main = coefficient(fit, "abnormal_coverage")
    result["primary"] = primary_block(
        SPEC, estimate=main["estimate"], se=main["se"], ci=main["ci"], p=main["p"],
        n=fit.get("n", 0), clusters=fit.get("clusters"),
        inference_status=fit.get("status", "ok"),
        extra={"n_dropped_missing": fit.get("n_dropped_missing")})

    for label, column in (("CAR(+2,+5)", "car_p2_p5"), ("CAR(+1,+20)", "car_p1_p20")):
        result["sensitivity"].append({
            "variant": label, **coefficient(regress(usable, column, xs, ["day0"]), "abnormal_coverage")})
    result["sensitivity"].append({
        "variant": "two-way clustering (date, ticker)",
        **coefficient(regress(usable, "car_p1_p10", xs, ["day0", "ticker"]), "abnormal_coverage")})

    if not ctx.kap_events.empty:
        disclosed = set(zip(ctx.kap_events["ticker"], ctx.kap_events["day0"]))
        quiet = usable[[key not in disclosed for key in zip(usable["ticker"], usable["day0"])]]
        result["sensitivity"].append({
            "variant": "excluding ticker-days with a sampled KAP disclosure",
            "n_kept": int(len(quiet)),
            **coefficient(regress(quiet, "car_p1_p10", xs, ["day0"]), "abnormal_coverage")})
    from config import ECONOMIC_CALENDAR
    macro_days = set(ECONOMIC_CALENDAR)
    calm = usable[~usable["day0"].isin(macro_days)]
    result["sensitivity"].append({
        "variant": "excluding scheduled macro days (config.ECONOMIC_CALENDAR)",
        "n_kept": int(len(calm)),
        **coefficient(regress(calm, "car_p1_p10", xs, ["day0"]), "abnormal_coverage")})

    result["limitations"] += [
        "Coverage is measured on the outlets this project scrapes, not on all media.",
        "An association would not show that retail attention is the mechanism; "
        "no attention measure independent of coverage is used.",
        "Windows of 10 and 20 sessions overlap for a frequently covered ticker; "
        "the two-way clustered variant addresses part of that.",
    ]
    return result
