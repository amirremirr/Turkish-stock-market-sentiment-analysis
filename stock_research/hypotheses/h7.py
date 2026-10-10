"""H7 -- KAP disclosure category drift."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stock_research import stats
from stock_research.calendar import CLEAN_REACTION_BUCKETS
from stock_research.config import EVENT_WINDOWS
from stock_research.eventstudy import STATUS_OK, collapse_same_day, flag_overlaps, window_label
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    Context, Spec, base_result, group_sufficient, insufficient, mean_test,
    missing_requirements, primary_block, regress,
)

MIN_CATEGORIES = 3
PRIMARY_WINDOW = (0, 5)

SPEC = Spec(
    id="H7",
    title="KAP disclosure category drift",
    claim="Different classes of KAP disclosure are followed by different "
          "abnormal-return patterns.",
    null="Mean CAR(0,+5) is the same for every disclosure category.",
    alternative="At least one category's mean CAR(0,+5) differs from the others.",
    eligibility="Sampled material-event disclosures from listed companies, "
                "excluding updates and corrections and events with ambiguous "
                "timing; one row per ticker and day 0; rows where one issuer "
                "filed more than one category for the same day 0 are excluded; "
                "categories with at least 30 events on at least 15 dates.",
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
    inference="OLS on category indicators, covariance clustered by day 0, "
              "joint F-test with (categories - 1, dates - 1) degrees of freedom",
    sufficiency={"min_events_per_category": 30, "min_dates_per_category": 15,
                 "min_categories": MIN_CATEGORIES},
    requires=("kap_events",),
    sensitivity=("clean-reaction timing buckets only", "market-adjusted returns",
                 "excluding events within 5 sessions of an earlier event of the "
                 "same issuer", "two-way clustering by date and issuer"),
)


def prepare(ctx: Context) -> tuple[pd.DataFrame, Dict[str, Any]]:
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
    return merged, counts


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)

    events, counts = prepare(ctx)
    if events.empty:
        return insufficient(result, ["no eligible disclosures"], counts)
    table = prep.study(ctx, events, with_features=False)
    counts["studied"] = prep.status_counts(table)
    column = window_label(PRIMARY_WINDOW)
    usable = table[(table["status"] == STATUS_OK) & table[column].notna()].copy()

    sizes = usable.groupby("category").agg(events=("ticker", "size"), dates=("day0", "nunique"))
    counts["by_category"] = sizes.to_dict("index")
    kept = [c for c, row in sizes.iterrows() if group_sufficient(row["events"], row["dates"])]
    counts["sufficient_categories"] = kept
    if len(kept) < MIN_CATEGORIES:
        return insufficient(result, [
            f"{len(kept)} categories meet 30 events on 15 dates; needs {MIN_CATEGORIES}"], counts)
    result["sufficiency"]["counts"] = counts
    sample = usable[usable["category"].isin(kept)].copy()

    def joint(frame: pd.DataFrame, y: str, clusters: List[str]) -> Dict[str, Any]:
        reference = frame["category"].value_counts().idxmax()
        dummies = [c for c in kept if c != reference and (frame["category"] == c).any()]
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

    main = joint(sample, column, ["day0"])
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
            test = mean_test(part[window_label(window)], part["day0"])
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
        out = joint(frame[frame["category"].isin(enough)], y, clusters)
        result["sensitivity"].append({"variant": name, "spread": out["spread"],
                                      "n": out["n"], **out["test"]})

    variant("clean-reaction buckets only",
            sample[sample["bucket"].isin(CLEAN_REACTION_BUCKETS)], column, ["day0"])
    variant("market-adjusted returns", sample.dropna(subset=["mar_p0_p5"]), "mar_p0_p5", ["day0"])
    overlap = flag_overlaps(sample, 5, ctx.panel.calendar)
    variant("no earlier event of the same issuer within 5 sessions", sample[~overlap],
            column, ["day0"])
    variant("two-way clustering (date, issuer)", sample, column, ["day0", "ticker"])

    result["tables"]["by_category"] = sizes.reset_index().to_dict("records")
    result["limitations"] += [
        "The sample is a systematic 80-anchor sample of 2023 disclosures, not a census.",
        "Issuers are those in the provider's current roster with prices available: "
        "companies delisted since are missing (survivorship).",
        "Categories describe the form a filer chose. An association with returns "
        "is descriptive; it does not show the disclosure caused the return, and a "
        "bonus issue in particular changes share count, not company value.",
        "Day 0 for a during-session disclosure includes price moves before publication.",
    ]
    return result
