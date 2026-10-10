"""H7 -- KAP disclosure category drift."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stock_research import stats
from stock_research.calendar import CLEAN_REACTION_BUCKETS
from stock_research.config import EVENT_WINDOWS
from stock_research.events import UNUSUAL_PRICE_VOLUME
from stock_research.eventstudy import (
    STATUS_OK, collapse_same_day, event_time_path, non_overlapping, window_label,
)
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    TWO_WAY, Context, Spec, base_result, group_sufficient, insufficient, mean_test,
    missing_requirements, primary_block, regress,
)

MIN_CATEGORIES = 3
PRIMARY_WINDOW = (0, 5)
#: Categories whose filing is caused by the stock's own recent price move. A
#: company answering an exchange query about unusual trading files *because*
#: the price moved; any return pattern around it is the cause, not an effect.
PRICE_TRIGGERED = (UNUSUAL_PRICE_VOLUME,)

SPEC = Spec(
    id="H7",
    title="KAP disclosure category drift",
    claim="Different classes of KAP disclosure are followed by different "
          "abnormal-return patterns.",
    null="Mean CAR(0,+5) is the same for every disclosure category.",
    alternative="At least one category's mean CAR(0,+5) differs from the others.",
    eligibility="Sampled material-event disclosures from listed companies, "
                "published before the open, after the close or on a non-session "
                "day, so that day 0 starts from a price formed before "
                "publication; not updates or corrections; unambiguous timing; "
                "one row per ticker and day 0; rows where one issuer filed more "
                "than one category, or at more than one kind of time, for the "
                "same day 0 are excluded; price-triggered categories are "
                "excluded; categories with at least 30 events on at least 15 dates.",
    event_time="Day 0 is the first session able to act on the publication "
               "timestamp. Transaction, decision and record dates inside the "
               "disclosure are not used.",
    outcome="CAR(0,+5), market model.",
    estimand="spread between the highest and lowest category mean CAR(0,+5); "
             "tested jointly (all category means equal)",
    direction=+1,
    material_effect=0.01,
    benchmark="market model on XU100, estimation window -130..-11",
    controls=(),
    inference="OLS on category indicators, covariance clustered two ways, by "
              "day 0 and by issuer; joint F-test",
    sufficiency={"min_events_per_category": 30, "min_dates_per_category": 15,
                 "min_categories": MIN_CATEGORIES},
    requires=("kap_events",),
    sensitivity=("all timing buckets, including during-session disclosures",
                 "market-adjusted returns",
                 "one event per issuer per six sessions",
                 "clustering by day 0 only"),
    parameters={"primary_window": list(PRIMARY_WINDOW), "min_categories": MIN_CATEGORIES,
                "price_triggered_categories": list(PRICE_TRIGGERED),
                "clean_buckets": sorted(CLEAN_REACTION_BUCKETS)},
)


def prepare(ctx: Context, *, clean_timing: bool = True) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """The H7 sample, with a count at every step."""

    counts: Dict[str, Any] = {"input": int(len(ctx.kap_events))}
    events = ctx.kap_events
    if events.empty:
        return events, counts
    events = events[events["ticker"].notna()]
    counts["ticker_resolved"] = int(len(events))
    events = events[events["day0"].notna() & ~events["timing_ambiguous"].astype(bool)]
    counts["timing_usable"] = int(len(events))
    # A missing flag is not a "no": forms without the flag are kept, and the
    # count of them is reported so the choice is visible.
    repeat = events["is_update"].eq(True) | events["is_correction"].eq(True)
    counts["flag_absent"] = int(events["is_update"].isna().sum())
    events = events[~repeat]
    counts["not_update_or_correction"] = int(len(events))
    merged = pd.DataFrame(collapse_same_day(events.to_dict("records")))
    counts["ticker_days"] = int(len(merged))
    merged = merged[merged["n_categories"] == 1] if len(merged) else merged
    counts["single_category_ticker_days"] = int(len(merged))
    merged = merged[~merged["category"].isin(PRICE_TRIGGERED)] if len(merged) else merged
    counts["not_price_triggered"] = int(len(merged))
    if clean_timing and len(merged):
        merged = merged[merged["bucket"].isin(CLEAN_REACTION_BUCKETS)]
        counts["clean_reaction_timing"] = int(len(merged))
    return merged, counts


def _joint(frame: pd.DataFrame, categories: List[str], y: str, clusters: List[str]) -> Dict[str, Any]:
    reference = frame["category"].value_counts().idxmax()
    dummies = [c for c in categories if c != reference and (frame["category"] == c).any()]
    design = frame.copy()
    for c in dummies:
        design[f"is_{c}"] = (design["category"] == c).astype(float)
    fit = regress(design, y, [f"is_{c}" for c in dummies], clusters)
    restrictions = np.zeros((len(dummies), len(dummies) + 1))
    restrictions[:, 1:] = np.eye(len(dummies))
    means = frame.groupby("category")[y].mean()
    return {"test": stats.wald(fit, restrictions), "fit": fit,
            "spread": float(means.max() - means.min()),
            "highest": str(means.idxmax()), "lowest": str(means.idxmin()),
            "reference": reference, "n": int(len(frame))}


def _study(ctx: Context, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = prep.study(ctx, events, with_features=False)
    column = window_label(PRIMARY_WINDOW)
    return table, table[(table["status"] == STATUS_OK) & table[column].notna()].copy()


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)

    events, counts = prepare(ctx)
    if events.empty:
        return insufficient(result, ["no eligible disclosures"], counts)
    table, usable = _study(ctx, events)
    counts["studied"] = prep.status_counts(table)
    column = window_label(PRIMARY_WINDOW)

    sizes = usable.groupby("category").agg(events=("ticker", "size"), dates=("day0", "nunique"))
    counts["by_category"] = sizes.to_dict("index")
    kept = [c for c, row in sizes.iterrows() if group_sufficient(row["events"], row["dates"])]
    counts["sufficient_categories"] = kept
    if len(kept) < MIN_CATEGORIES:
        return insufficient(result, [
            f"{len(kept)} categories meet 30 events on 15 dates; needs {MIN_CATEGORIES}"], counts)
    result["sufficiency"]["counts"] = counts
    sample = usable[usable["category"].isin(kept)].copy()

    main = _joint(sample, kept, column, TWO_WAY)
    result["primary"] = primary_block(
        SPEC, estimate=main["spread"], se=None, ci=None, p=main["test"].get("p"),
        n=main["n"], clusters=main["fit"].get("clusters"),
        inference_status=main["test"].get("status", "ok"),
        extra={"joint_test": True, "f": main["test"].get("f"),
               "degrees_of_freedom": [main["test"].get("q"), main["test"].get("dof")],
               "highest_category": main["highest"], "lowest_category": main["lowest"]})

    # Exploratory: each category's mean in each window, FDR-adjusted together.
    cells = []
    for category in kept:
        part = sample[sample["category"] == category]
        for window in EVENT_WINDOWS:
            test = mean_test(part[window_label(window)], part["day0"], part["ticker"])
            cells.append({"test": f"{category}, CAR{window}", "category": category,
                          "window": list(window), **test})
    adjusted = stats.benjamini_hochberg([c["p"] for c in cells])
    for cell, p_bh in zip(cells, adjusted):
        cell["p_bh"] = p_bh
    result["exploratory"] = cells

    def variant(name: str, frame: pd.DataFrame, y: str, clusters: List[str]) -> None:
        enough = [c for c in kept if (frame["category"] == c).sum() >= 30]
        if len(enough) < MIN_CATEGORIES:
            result["sensitivity"].append({"variant": name, "status": "too_few_events",
                                          "n": int(len(frame))})
            return
        out = _joint(frame[frame["category"].isin(enough)], enough, y, clusters)
        result["sensitivity"].append({"variant": name, "spread": out["spread"],
                                      "n": out["n"], **out["test"]})

    every_bucket, _ = prepare(ctx, clean_timing=False)
    _, all_usable = _study(ctx, every_bucket)
    variant("all timing buckets, including during-session disclosures",
            all_usable[all_usable["category"].isin(kept)], column, TWO_WAY)
    variant("market-adjusted returns", sample.dropna(subset=["mar_p0_p5"]), "mar_p0_p5", TWO_WAY)
    spaced = non_overlapping(sample, PRIMARY_WINDOW[1], ctx.panel.calendar)
    variant("one event per issuer per six sessions", sample[spaced], column, TWO_WAY)
    variant("clustering by day 0 only", sample, column, ["day0"])

    result["tables"]["by_category"] = sizes.reset_index().to_dict("records")
    result["tables"]["event_time"] = {
        category: event_time_path(ctx.panel,
                                  sample[sample["category"] == category].to_dict("records"),
                                  sealed=ctx.sealed)
        for category in kept}
    result["limitations"] += [
        "The sample is a systematic 80-anchor sample of 2023 disclosures, not a census.",
        "Issuers are those in the provider's current roster with prices available: "
        "companies delisted since are missing (survivorship).",
        "Categories describe the form a filer chose. An association with returns "
        "is descriptive; it does not show the disclosure caused the return, and a "
        "bonus issue in particular changes share count, not company value.",
        "Disclosures published during the session are outside the primary sample, "
        "because their day-0 return includes whatever happened before publication.",
        "The spread between the highest and lowest category mean is biased upward "
        "by taking a maximum and a minimum of noisy means; it is reported as a "
        "size, and the decision rests on the joint test.",
    ]
    return result
