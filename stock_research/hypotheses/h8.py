"""H8 -- insider buying and subsequent returns."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from stock_research import costs
from stock_research.config import MIN_EVENT_DATES_PER_GROUP, MIN_EVENTS_PER_GROUP
from stock_research.eventstudy import STATUS_OK, event_time_path, non_overlapping
from stock_research.events import INSIDER_TRADE
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    Context, Spec, base_result, coefficient, insufficient, mean_test,
    missing_requirements, primary_block, regress,
)
from stock_research.insider import PARSED, VENUE_ON, VENUE_UNKNOWN

WINDOWS = ((1, 20), (0, 20), (1, 5), (1, 10), (0, 0))

SPEC = Spec(
    id="H8",
    title="Insider buying and subsequent returns",
    claim="Open-market insider purchases disclosed on KAP are followed by "
          "positive abnormal returns over 20 sessions, possibly more so for "
          "small companies.",
    null="Mean CAR(+1,+20) after a disclosed insider purchase is zero.",
    alternative="Mean CAR(+1,+20) after a disclosed insider purchase is positive.",
    eligibility="Share transaction notifications parsed as pure purchases whose "
                "text states that the trade was made on the exchange (venue "
                "confirmed); not flagged as transfers, off-exchange deals, option "
                "exercises or capital-increase allotments; not updates or "
                "corrections; one row per ticker and day 0, and none where the "
                "same issuer also disclosed a sale, or a purchase with no stated "
                "venue, for that day 0.",
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
    inference="mean with standard errors clustered two ways, by day 0 and by "
              "issuer: a holder often buys on many consecutive days, and those "
              "20-session windows share most of their returns",
    sufficiency={"min_events": MIN_EVENTS_PER_GROUP,
                 "min_event_dates": MIN_EVENT_DATES_PER_GROUP},
    requires=("kap_events",),
    sensitivity=("including purchases whose venue is not stated",
                 "CAR(0,+20) including the first reaction",
                 "one purchase per issuer per 21 sessions",
                 "clustering by day 0 only",
                 "small versus large companies (needs point-in-time market cap)"),
    parameters={"windows": [list(w) for w in WINDOWS], "primary_window": [1, 20],
                "side": "buy only; a ticker-day with any sale is excluded",
                "venue": "on_exchange (stated in the text)",
                "venue_unknown": "excluded from the primary sample; a sensitivity analysis",
                "kinds": "holder trading another issuer, or issuer reporting a holder; "
                         "an issuer trading its own shares is a buyback"},
)


CONFIRMED = (VENUE_ON,)
ANY_STATED_OR_NOT = (VENUE_ON, VENUE_UNKNOWN)


def prepare(ctx: Context, venues: tuple = CONFIRMED) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Purchase ticker-days whose every purchase has a venue in *venues*.

    The primary sample passes ``CONFIRMED``: the text must say the trade was
    made on the exchange. A table with no venue column, from before venue was
    recorded, has no confirmed purchase at all.
    """

    counts: Dict[str, Any] = {}
    events = ctx.kap_events
    if events.empty or "insider_parse_status" not in events.columns:
        return pd.DataFrame(), {"insider_notifications": 0}
    events = events[events["category"] == INSIDER_TRADE]
    counts["insider_notifications"] = int(len(events))
    counts["by_parse_status"] = {str(k): int(v) for k, v in
                                 events["insider_parse_status"].fillna("none").value_counts().items()}
    events = events[events["insider_parse_status"] == PARSED].copy()
    counts["by_side"] = {str(k): int(v) for k, v in
                         events["insider_side"].fillna("none").value_counts().items()}
    if "insider_venue" not in events.columns:
        events["insider_venue"] = VENUE_UNKNOWN
    events["insider_venue"] = events["insider_venue"].fillna(VENUE_UNKNOWN)
    counts["by_venue"] = {str(k): int(v) for k, v in events.loc[
        events["insider_side"] == "buy", "insider_venue"].value_counts().items()}
    events = events[events["ticker"].notna() & events["day0"].notna()
                    & ~events["timing_ambiguous"].astype(bool)]
    events = events[~(events["is_update"].eq(True) | events["is_correction"].eq(True))]
    counts["timed_and_resolved"] = int(len(events))
    if events.empty:
        return events, counts

    rows = []
    any_venue = 0
    for (ticker, day0), group in events.groupby(["ticker", "day0"]):
        sides = set(group["insider_side"])
        if sides != {"buy"}:
            continue                       # a sale or a mixed day is not a purchase day
        any_venue += 1
        if not set(group["insider_venue"]) <= set(venues):
            continue                       # venue not shown for every purchase that day
        values = group["insider_value"].dropna()
        rows.append({"ticker": ticker, "day0": day0, "event_id": group["event_id"].iloc[0],
                     "n_events": int(len(group)), "bucket": group["bucket"].iloc[0],
                     "insider_value": float(values.sum()) if len(values) == len(group) else None,
                     "insider_role": group["insider_role"].iloc[0]})
    counts["purchase_ticker_days_any_venue"] = any_venue
    counts["purchase_ticker_days"] = len(rows)
    return pd.DataFrame(rows), counts


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)

    events, counts = prepare(ctx)
    if events.empty:
        return insufficient(result, [
            "no purchase whose text states it was made on the exchange (venue "
            f"confirmed); purchases by venue: {counts.get('by_venue', {})}"], counts)
    table = prep.study(ctx, events, windows=WINDOWS)
    counts["studied"] = prep.status_counts(table)
    usable = table[(table["status"] == STATUS_OK) & table["car_p1_p20"].notna()].copy()
    if len(usable) < MIN_EVENTS_PER_GROUP or usable["day0"].nunique() < MIN_EVENT_DATES_PER_GROUP:
        return insufficient(result, [
            f"{len(usable)} venue-confirmed purchase ticker-days on "
            f"{usable['day0'].nunique()} dates; needs {MIN_EVENTS_PER_GROUP} on "
            f"{MIN_EVENT_DATES_PER_GROUP}"], counts)
    result["sufficiency"]["counts"] = counts

    main = mean_test(usable["car_p1_p20"], usable["day0"], usable["ticker"])
    counts["distinct_issuers"] = int(usable["ticker"].nunique())
    result["primary"] = primary_block(
        SPEC, estimate=main["mean"], se=main["se"], ci=main["ci"], p=main["p"],
        n=main["n"], clusters=main["clusters"], inference_status=main["status"])

    for label, column in (("CAR(0,0)", "car_p0_p0"), ("CAR(+1,+5)", "car_p1_p5"),
                          ("CAR(+1,+10)", "car_p1_p10")):
        result["exploratory"].append({
            "test": f"mean {label}",
            **mean_test(usable[column], usable["day0"], usable["ticker"])})

    broader, _ = prepare(ctx, venues=ANY_STATED_OR_NOT)
    wide = prep.study(ctx, broader, windows=WINDOWS)
    wide = wide[(wide["status"] == STATUS_OK) & wide["car_p1_p20"].notna()]
    result["sensitivity"].append({
        "variant": "including purchases whose venue is not stated",
        **mean_test(wide["car_p1_p20"], wide["day0"], wide["ticker"])})
    result["sensitivity"].append({
        "variant": "CAR(0,+20), including the first reaction",
        **mean_test(usable["car_p0_p20"], usable["day0"], usable["ticker"])})
    spaced = usable[non_overlapping(usable, 20, ctx.panel.calendar)]
    result["sensitivity"].append({
        "variant": "one purchase per issuer per 21 sessions (no overlapping windows)",
        "n_kept": int(len(spaced)),
        **mean_test(spaced["car_p1_p20"], spaced["day0"])})
    result["sensitivity"].append({
        "variant": "clustering by day 0 only (overstates precision when windows overlap)",
        **mean_test(usable["car_p1_p20"], usable["day0"])})
    result["sensitivity"].append({
        "variant": "small versus large companies", "status": "blocked",
        "reason": ctx.available("point_in_time_market_cap")[1]})

    sized = usable.dropna(subset=["insider_value", "log_turnover_20"]).copy()
    if len(sized) >= MIN_EVENTS_PER_GROUP:
        sized["log_intensity"] = np.log(sized["insider_value"]) - sized["log_turnover_20"]
        fit = regress(sized, "car_p1_p20",
                      ["log_intensity", "prior_ret_5", "volatility_20"], ["day0", "ticker"])
        result["exploratory"].append({
            "test": "purchase intensity: log(value / mean daily turnover)",
            **coefficient(fit, "log_intensity"), "n": fit.get("n")})

    result["tables"]["event_time"] = {
        "insider purchases": event_time_path(ctx.panel, usable.to_dict("records"),
                                             sealed=ctx.sealed)}
    result["execution"] = {
        "strategy": "buy at the close of day 0 and hold 20 sessions, hedged with the index",
        "assumption": "fills at the closing auction on entry and exit; an index "
                      "hedge is available; neither is verified",
        **costs.net_summary(usable["mar_p1_p20"].dropna().tolist(), round_trips=2),
    }
    result["limitations"] += [
        "Purchases are identified by a text parser over the form's labelled "
        "fields; notifications it cannot read are left out, not guessed.",
        "No field of the form states the venue. A purchase is in the sample "
        "only if its free text names the exchange as where the trade was made; "
        "the common template that gives a price range and no venue is treated "
        "as venue unknown and left out, which makes the sample smaller and may "
        "make it unrepresentative of on-exchange purchases as a whole.",
        "'Insider' here means anyone with a notification duty for the issuer's "
        "shares: parents and large holders as well as directors.",
        "The filer's role is read from free text and is not verified.",
        "Twenty-session windows overlap for issuers with repeated purchases.",
        "Survivorship: issuers delisted since the roster snapshot are missing.",
    ]
    return result
