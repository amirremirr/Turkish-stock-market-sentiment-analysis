"""H2 -- overnight news gap underreaction."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stock_research import costs
from stock_research.calendar import BUCKET_NON_SESSION, BUCKET_POST_CLOSE
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    Context, Spec, base_result, coefficient, insufficient, missing_requirements,
    primary_block, prior_features, regress,
)

MIN_EVENTS = 100
MIN_DATES = 30
#: An open this far from the prior close is at or near a price limit; the
#: opening print is then a constraint, not a price both sides chose.
LIMIT_OPEN = 0.095

SPEC = Spec(
    id="H2",
    title="Overnight news gap underreaction",
    claim="After company news published outside trading hours, the opening gap "
          "only partly reflects the adjustment; the move continues intraday.",
    null="Given the gap, the open-to-close return is unrelated to it.",
    alternative="The open-to-close return has the same sign as the gap.",
    eligibility="Confirmed, single-issuer, material news with sentiment, published "
                "after the close or on a non-session day with unambiguous timing; "
                "day 0 has an opening print, positive volume, no corporate action, "
                "and an opening gap inside the price limit.",
    event_time="Day 0 is the first session after publication.",
    outcome="intraday = log(close_0 / open_0); gap = log(open_0 / close_-1).",
    estimand="coefficient on gap in intraday ~ gap + sentiment + sentiment x gap "
             "+ market_gap + market_intraday + log_turnover_20 + volatility_20",
    direction=+1,
    material_effect=0.05,
    benchmark="index gap and index open-to-close return as regressors",
    controls=("sentiment", "sentiment x gap", "market_gap", "market_intraday",
              "log_turnover_20", "volatility_20"),
    inference="OLS, standard errors clustered by day 0",
    sufficiency={"min_events": MIN_EVENTS, "min_event_dates": MIN_DATES},
    requires=("news_events",),
    sensitivity=("excluding the smallest third of absolute gaps",
                 "sentiment-aligned gaps only"),
)


def _same_day(ctx: Context, events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    market = ctx.panel.market
    for event in events.to_dict("records"):
        frame, day0 = ctx.panel.frame(event["ticker"]), event["day0"]
        row = dict(event)
        if frame is None or day0 not in frame.index:
            row["exclusion"] = "no_bar"
        elif ctx.sealed is not None and day0 >= ctx.sealed:
            row["exclusion"] = "sealed_window"
        else:
            bar = frame.loc[day0]
            row.update({"gap": bar["gap"], "intraday": bar["intraday"],
                        "market_gap": market.loc[day0, "gap"],
                        "market_intraday": market.loc[day0, "intraday"],
                        "fill": costs.entry_fill(frame, day0, limit=0.10)})
            row.update(prior_features(ctx.panel, event["ticker"], day0))
            if bool(bar["corporate_action"]):
                row["exclusion"] = "corporate_action_on_day0"
            elif not np.isfinite(bar["gap"]) or not np.isfinite(bar["intraday"]):
                row["exclusion"] = "missing_open_or_close"
            elif not np.isfinite(bar["volume"]) or bar["volume"] <= 0:
                row["exclusion"] = "no_volume"
            elif abs(bar["gap"]) >= LIMIT_OPEN:
                row["exclusion"] = "opened_at_price_limit"
            else:
                row["exclusion"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)

    events, counts = prep.admissible_news(ctx)
    events = events[events["bucket"].isin([BUCKET_POST_CLOSE, BUCKET_NON_SESSION])] \
        if not events.empty else events
    counts["after_close_or_non_session"] = int(len(events))
    if events.empty:
        return insufficient(result, ["no admissible after-close news events"], counts)

    table = _same_day(ctx, events)
    counts["exclusions"] = {str(k): int(v) for k, v in
                            table["exclusion"].fillna("kept").value_counts().items()}
    usable = table[table["exclusion"].isna()].copy()
    if len(usable) < MIN_EVENTS or usable["day0"].nunique() < MIN_DATES:
        return insufficient(result, [
            f"{len(usable)} usable events on {usable['day0'].nunique()} dates; "
            f"needs {MIN_EVENTS} on {MIN_DATES}"], counts)
    result["sufficiency"]["counts"] = counts

    usable["sent_x_gap"] = usable["sentiment"] * usable["gap"]
    xs = ["gap", "sentiment", "sent_x_gap", "market_gap", "market_intraday",
          "log_turnover_20", "volatility_20"]
    fit = regress(usable, "intraday", xs, ["day0"])
    gap = coefficient(fit, "gap")
    result["primary"] = primary_block(
        SPEC, estimate=gap["estimate"], se=gap["se"], ci=gap["ci"], p=gap["p"],
        n=fit.get("n", 0), clusters=fit.get("clusters"),
        inference_status=fit.get("status", "ok"),
        extra={"n_dropped_missing": fit.get("n_dropped_missing")})
    for name in ("sentiment", "sent_x_gap"):
        result["exploratory"].append({"test": f"coefficient: {name}", **coefficient(fit, name)})

    bins = pd.qcut(usable["gap"], 10, duplicates="drop")
    result["tables"]["gap_bins"] = [
        {"gap": float(group["gap"].mean()), "intraday": float(group["intraday"].mean()),
         "n": int(len(group))} for _, group in usable.groupby(bins, observed=True)]
    cutoff = usable["gap"].abs().quantile(1 / 3)
    result["sensitivity"].append({
        "variant": "largest two thirds of absolute gaps",
        **coefficient(regress(usable[usable["gap"].abs() > cutoff], "intraday", xs, ["day0"]), "gap")})
    aligned = usable[np.sign(usable["gap"]) == np.sign(usable["sentiment"])]
    result["sensitivity"].append({
        "variant": "gap in the direction of sentiment",
        **coefficient(regress(aligned, "intraday", xs, ["day0"]), "gap")})

    # Statistical continuation is not tradable continuation. The executable
    # version buys or sells at the open in the gap's direction, hedges with the
    # index, and exits at the close -- where an opening fill was plausible.
    filled = usable[usable["fill"] == costs.FILL_OK]
    signed = np.sign(filled["gap"]) * (filled["intraday"] - filled["market_intraday"])
    result["execution"] = {
        "strategy": "trade in the gap's direction from the open to the close of "
                    "day 0, hedged with the index",
        "assumption": "fills at the opening print; short sales assumed possible",
        "coverage": {"usable_events": int(len(usable)), "filled": int(len(filled))},
        **costs.net_summary(signed.tolist()),
    }
    result["limitations"].append(
        "Daily bars cannot show whether the opening print was reachable in size.")
    return result
