"""H8 -- insider buying and subsequent returns."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from stock_research import costs
from stock_research.config import MIN_EVENT_DATES_PER_GROUP, MIN_EVENTS_PER_GROUP
from stock_research.eventstudy import STATUS_OK, collapse_same_day, flag_overlaps
from stock_research.events import INSIDER_TRADE
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    Context, Spec, base_result, coefficient, insufficient, mean_test,
    missing_requirements, primary_block, regress,
)
from stock_research.insider import PARSED

WINDOWS = ((1, 20), (0, 20), (1, 5), (1, 10), (0, 0))

SPEC = Spec(
    id="H8",
    title="Insider buying and subsequent returns",
    claim="Open-market insider purchases disclosed on KAP are followed by "
          "positive abnormal returns over 20 sessions, possibly more so for "
          "small companies.",
    null="Mean CAR(+1,+20) after a disclosed insider purchase is zero.",
    alternative="Mean CAR(+1,+20) after a disclosed insider purchase is positive.",
    eligibility="Share transaction notifications parsed as pure purchases, not "
                "flagged as transfers, off-exchange deals, option exercises or "
                "capital-increase allotments; not updates or corrections; one row "
                "per ticker and day 0, and none where the same issuer also "
                "disclosed a sale for that day 0.",
    event_time="Day 0 is the first session able to act on the disclosure's "
               "publication. The transaction date inside the form is not used: "
               "the market did not know it yet.",
    outcome="CAR(+1,+20), market model: from the close of day 0, the first close "
            "at which the disclosure was public for a full session.",
    estimand="mean CAR(+1,+20) across insider-purchase ticker-days",
    direction=+1,
    material_effect=0.01,
    benchmark="market model on XU100, estimation window -130..-11",
    controls=("prior_ret_5", "log_turnover_20", "volatility_20 (in the intensity regression)"),
    inference="mean with standard errors clustered by day 0",
    sufficiency={"min_events": MIN_EVENTS_PER_GROUP,
                 "min_event_dates": MIN_EVENT_DATES_PER_GROUP},
    requires=("kap_events",),
    sensitivity=("CAR(0,+20) including the first reaction", "two-way clustering",
                 "no earlier purchase by the same issuer within 20 sessions",
                 "small versus large companies (needs point-in-time market cap)"),
)


def prepare(ctx: Context) -> tuple[pd.DataFrame, Dict[str, Any]]:
    counts: Dict[str, Any] = {}
    events = ctx.kap_events
    if events.empty or "insider_parse_status" not in events.columns:
        return pd.DataFrame(), {"insider_notifications": 0}
    events = events[events["category"] == INSIDER_TRADE]
    counts["insider_notifications"] = int(len(events))
    counts["by_parse_status"] = {str(k): int(v) for k, v in
                                 events["insider_parse_status"].fillna("none").value_counts().items()}
    events = events[events["insider_parse_status"] == PARSED]
    counts["by_side"] = {str(k): int(v) for k, v in
                         events["insider_side"].fillna("none").value_counts().items()}
    events = events[events["ticker"].notna() & events["day0"].notna()
                    & ~events["timing_ambiguous"].astype(bool)]
    events = events[~(events["is_update"].eq(True) | events["is_correction"].eq(True))]
    counts["timed_and_resolved"] = int(len(events))
    if events.empty:
        return events, counts

    rows = []
    for (ticker, day0), group in events.groupby(["ticker", "day0"]):
        sides = set(group["insider_side"])
        if sides != {"buy"}:
            continue                       # a sale or a mixed day is not a purchase day
        values = group["insider_value"].dropna()
        rows.append({"ticker": ticker, "day0": day0, "event_id": group["event_id"].iloc[0],
                     "n_events": int(len(group)), "bucket": group["bucket"].iloc[0],
                     "insider_value": float(values.sum()) if len(values) == len(group) else None,
                     "insider_role": group["insider_role"].iloc[0]})
    counts["purchase_ticker_days"] = len(rows)
    return pd.DataFrame(rows), counts


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)

    events, counts = prepare(ctx)
    if events.empty:
        return insufficient(result, ["no parsed insider purchases"], counts)
    table = prep.study(ctx, events, windows=WINDOWS)
    counts["studied"] = prep.status_counts(table)
    usable = table[(table["status"] == STATUS_OK) & table["car_p1_p20"].notna()].copy()
    if len(usable) < MIN_EVENTS_PER_GROUP or usable["day0"].nunique() < MIN_EVENT_DATES_PER_GROUP:
        return insufficient(result, [
            f"{len(usable)} purchase ticker-days on {usable['day0'].nunique()} dates; "
            f"needs {MIN_EVENTS_PER_GROUP} on {MIN_EVENT_DATES_PER_GROUP}"], counts)
    result["sufficiency"]["counts"] = counts

    main = mean_test(usable["car_p1_p20"], usable["day0"])
    result["primary"] = primary_block(
        SPEC, estimate=main["mean"], se=main["se"], ci=main["ci"], p=main["p"],
        n=main["n"], clusters=main["clusters"], inference_status=main["status"])

    for label, column in (("CAR(0,0)", "car_p0_p0"), ("CAR(+1,+5)", "car_p1_p5"),
                          ("CAR(+1,+10)", "car_p1_p10")):
        result["exploratory"].append({"test": f"mean {label}",
                                      **mean_test(usable[column], usable["day0"])})

    result["sensitivity"].append({"variant": "CAR(0,+20), including the first reaction",
                                  **mean_test(usable["car_p0_p20"], usable["day0"])})
    result["sensitivity"].append({
        "variant": "two-way clustering (date, issuer)",
        **coefficient(regress(usable.assign(one=1.0), "car_p1_p20", [], ["day0", "ticker"]), "const")})
    alone = usable[~flag_overlaps(usable, 20, ctx.panel.calendar)]
    result["sensitivity"].append({
        "variant": "no earlier purchase by the same issuer within 20 sessions",
        **mean_test(alone["car_p1_p20"], alone["day0"])})
    result["sensitivity"].append({
        "variant": "small versus large companies", "status": "blocked",
        "reason": ctx.available("point_in_time_market_cap")[1]})

    sized = usable.dropna(subset=["insider_value", "log_turnover_20"]).copy()
    if len(sized) >= MIN_EVENTS_PER_GROUP:
        sized["log_intensity"] = np.log(sized["insider_value"]) - sized["log_turnover_20"]
        fit = regress(sized, "car_p1_p20",
                      ["log_intensity", "prior_ret_5", "volatility_20"], ["day0"])
        result["exploratory"].append({
            "test": "purchase intensity: log(value / mean daily turnover)",
            **coefficient(fit, "log_intensity"), "n": fit.get("n")})

    result["execution"] = {
        "strategy": "buy at the close of day 0 and hold 20 sessions, hedged with the index",
        "assumption": "fills at the closing auction on entry and exit; not verified",
        **costs.net_summary(usable["mar_p1_p20"].dropna().tolist()),
    }
    result["limitations"] += [
        "Purchases are identified by a text parser over the form's labelled "
        "fields; notifications it cannot read are left out, not guessed.",
        "The filer's role is read from free text and is not verified.",
        "Twenty-session windows overlap for issuers with repeated purchases.",
        "Survivorship: issuers delisted since the roster snapshot are missing.",
    ]
    return result
