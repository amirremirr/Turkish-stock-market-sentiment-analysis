"""H7 -- KAP disclosure category drift.

Protocol v3. What changed from v2, and why, is in
``docs/stock_research/protocol_v3/CHANGES_FROM_V2.md``. In brief:

* The test is no longer "do any of the classes that happen to be large
  differ". It compares named classes of filing with economic content against
  a reference group of procedural filings, one contrast each, decided before
  any outcome is read.
* The joint p-value comes from a wild cluster bootstrap, not from an F
  distribution that the readiness audit found to reject a true null several
  times too often on this kind of design.
* An issuer's filings with overlapping outcome windows count once.
* A class is tested only if it has enough events, dates **and issuers**.
* The size of an effect is a contrast, not a maximum minus a minimum.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from stock_research import events as ev
from stock_research import stats
from stock_research.calendar import CLEAN_REACTION_BUCKETS
from stock_research.config import (
    ALPHA, EVENT_WINDOWS, H7_MIN_DATES, H7_MIN_EVENTS, H7_MIN_ISSUERS,
    H7_REFERENCE_MIN_DATES, H7_REFERENCE_MIN_EVENTS, H7_REFERENCE_MIN_ISSUERS,
    WILD_BOOTSTRAP_DRAWS, WILD_BOOTSTRAP_SEED,
)
from stock_research.eventstudy import (
    STATUS_OK, collapse_same_day, event_time_path, non_overlapping, window_label,
)
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    TWO_WAY, Context, Spec, base_result, insufficient, mean_test, missing_requirements,
    primary_block, regress,
)

PRIMARY_WINDOW = (0, 5)
REFERENCE = "reference"

#: Classes of filing whose reaction is the subject of the hypothesis. Each is
#: one pre-specified contrast against the reference group. Fixed before any
#: outcome was read; the list is the registered taxonomy for H7.
INTEREST = (
    ev.BUYBACK,             # the company reports buying back its own shares
    ev.DIVIDEND,            # dividend distribution
    ev.BONUS_ISSUE,         # bedelsiz: capital increase from internal resources
    ev.RIGHTS_ISSUE,        # bedelli: capital increase for cash
    ev.NEW_CONTRACT,        # a new business relationship, order or tender
    ev.ASSET_ACQUISITION,   # purchase of a financial or tangible fixed asset
    ev.ASSET_SALE,          # sale of one
    ev.DEBT_ISSUANCE,       # issue of debt instruments, or an issue ceiling
)
#: Procedural filings: general-assembly notices, auditor selection, changes to
#: the articles, registered-capital ceilings and the like. They are filed by
#: the same companies through the same channel and carry no transaction, so
#: their mean abnormal return is the level the benchmark leaves behind.
REFERENCE_CLASSES = (ev.ADMINISTRATIVE,)
#: Sensitivity: every class without a stated economic event as the reference.
POOLED_REFERENCE_CLASSES = (ev.ADMINISTRATIVE, ev.GENERAL_OTHER, ev.OTHER_FORM)
#: Sampled, counted, and outside the test, with the reason.
NOT_TESTED = {
    ev.GENERAL_OTHER: "a general form in which no registered phrase matched: content unknown",
    ev.OTHER_FORM: "a form name outside the taxonomy: content unknown",
    ev.CAPITAL_CHANGE_OTHER: "a capital form that is both rights and bonus, or unclear",
    ev.INSIDER_TRADE: "direction depends on the side of the trade; H8 tests purchases",
    ev.UNUSUAL_PRICE_VOLUME: "filed because the price moved: the return is the cause",
}
#: Kept from v2 under its old name: classes whose filing the price triggers.
PRICE_TRIGGERED = (ev.UNUSUAL_PRICE_VOLUME,)

SPEC = Spec(
    id="H7",
    title="KAP disclosure category drift",
    claim="Filings that report an economic event are followed by abnormal "
          "returns that differ from those following procedural filings.",
    null="Mean CAR(0,+5) of every tested class equals the mean of the reference "
         "group of procedural filings.",
    alternative="At least one tested class differs from the reference group.",
    eligibility="Every material-event disclosure of a listed company in the 2023 "
                "census, published before the open, after the close or on a "
                "non-session day; not an update or correction; unambiguous "
                "timing; not dated to a session that reopened the market after "
                "an unscheduled closure; one row per issuer and day 0, with "
                "rows of mixed class or mixed timing excluded; class in the "
                "registered list or the reference group; an issuer's rows "
                "thinned so that no two are within six sessions (the earliest "
                "is kept); a class is tested only with at least 30 events on 20 "
                "dates from 20 issuers, the reference group with 60 on 30 from 30.",
    event_time="Day 0 is the first session able to act on the publication "
               "timestamp. Transaction, decision and record dates inside the "
               "disclosure are not used.",
    outcome="CAR(0,+5), market model.",
    estimand="for each tested class, its mean CAR(0,+5) minus the reference "
             "group's; tested jointly (all contrasts zero). The reported size is "
             "the largest absolute contrast among those individually "
             "significant after Holm adjustment across the contrasts",
    direction=+1,
    material_effect=0.01,
    benchmark="market model on XU100, estimation window -130..-11",
    controls=(),
    inference="OLS on class indicators with the reference group as baseline; "
              "Wald statistic with covariance clustered two ways, by day 0 and "
              "by issuer; p-value by wild cluster bootstrap with the null "
              "imposed and Rademacher weights by day 0 (9,999 draws, fixed seed)",
    sufficiency={"min_events_per_class": H7_MIN_EVENTS, "min_dates_per_class": H7_MIN_DATES,
                 "min_issuers_per_class": H7_MIN_ISSUERS,
                 "reference_min_events": H7_REFERENCE_MIN_EVENTS,
                 "reference_min_dates": H7_REFERENCE_MIN_DATES,
                 "reference_min_issuers": H7_REFERENCE_MIN_ISSUERS,
                 "min_tested_classes": 1},
    requires=("kap_events",),
    sensitivity=("reference group pooled with unclassified filings",
                 "all timing buckets, including during-session disclosures",
                 "market-adjusted returns",
                 "no thinning of an issuer's overlapping events",
                 "events with no other sampled filing by the issuer within five sessions",
                 "bootstrap of the statistic clustered by day 0 only",
                 "CR2 covariance with Satterthwaite degrees of freedom, by day 0",
                 "two-way F test (the v2 primary test; known to over-reject)"),
    parameters={"primary_window": list(PRIMARY_WINDOW), "interest": list(INTEREST),
                "reference_classes": list(REFERENCE_CLASSES),
                "pooled_reference_classes": list(POOLED_REFERENCE_CLASSES),
                "not_tested": dict(NOT_TESTED),
                "clean_buckets": sorted(CLEAN_REACTION_BUCKETS),
                "thinning_sessions": PRIMARY_WINDOW[1],
                "bootstrap_draws": WILD_BOOTSTRAP_DRAWS, "bootstrap_seed": WILD_BOOTSTRAP_SEED,
                "contrast_alpha": ALPHA},
)


def group_of(category: Optional[str], reference: Sequence[str] = REFERENCE_CLASSES) -> Optional[str]:
    """The test group of a class: itself, the reference group, or none."""

    if category in reference:
        return REFERENCE
    return category if category in INTEREST else None


def prepare(ctx: Context, *, clean_timing: bool = True,
            reference: Sequence[str] = REFERENCE_CLASSES) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """The H7 sample before prices are consulted, with a count at every step."""

    counts: Dict[str, Any] = {"input": int(len(ctx.kap_events))}
    events = ctx.kap_events
    if events.empty:
        return events, counts
    events = events[events["ticker"].notna()]
    counts["ticker_resolved"] = int(len(events))
    events = events[events["day0"].notna() & ~events["timing_ambiguous"].astype(bool)]
    counts["timing_usable"] = int(len(events))
    if "closure_delayed" in events:
        events = events[~events["closure_delayed"].fillna(False).astype(bool)]
    counts["not_delayed_by_a_closure"] = int(len(events))
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
    if len(merged):
        counts["by_class_before_grouping"] = merged["category"].value_counts().to_dict()
        merged = merged.assign(group=merged["category"].map(lambda c: group_of(c, reference)))
        merged = merged[merged["group"].notna()]
    counts["in_registered_classes"] = int(len(merged))
    if clean_timing and len(merged):
        merged = merged[merged["bucket"].isin(CLEAN_REACTION_BUCKETS)]
        counts["clean_reaction_timing"] = int(len(merged))
    return merged, counts


def thin(table: pd.DataFrame, calendar) -> pd.Series:
    """True for the rows kept when each issuer's rows are thinned so that no
    two outcome windows overlap. The earliest row is kept, then the next one
    whose day 0 is more than five sessions later. Decided by dates alone."""

    return non_overlapping(table, PRIMARY_WINDOW[1], calendar)


def group_sizes(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(columns=["events", "dates", "issuers"])
    return table.groupby("group").agg(events=("ticker", "size"), dates=("day0", "nunique"),
                                      issuers=("ticker", "nunique"))


def sufficient(row: pd.Series, *, reference: bool) -> bool:
    need = ((H7_REFERENCE_MIN_EVENTS, H7_REFERENCE_MIN_DATES, H7_REFERENCE_MIN_ISSUERS)
            if reference else (H7_MIN_EVENTS, H7_MIN_DATES, H7_MIN_ISSUERS))
    return bool(row["events"] >= need[0] and row["dates"] >= need[1] and row["issuers"] >= need[2])


def design(frame: pd.DataFrame, tested: Sequence[str]):
    """Intercept (the reference group's mean) and one indicator per tested
    class; restrictions that set every indicator's coefficient to zero."""

    X = np.column_stack([np.ones(len(frame))]
                        + [(frame["group"] == c).to_numpy(dtype=float) for c in tested])
    R = np.zeros((len(tested), len(tested) + 1))
    R[:, 1:] = np.eye(len(tested))
    return X, R


def contrasts(frame: pd.DataFrame, tested: Sequence[str], y: str, *,
              draws: int = WILD_BOOTSTRAP_DRAWS, seed: int = WILD_BOOTSTRAP_SEED,
              each: bool = True) -> Dict[str, Any]:
    """The registered analysis on one sample: the joint bootstrap test and,
    with ``each``, a bootstrap test of every contrast on its own."""

    frame = frame.dropna(subset=[y, "day0", "ticker"])
    X, R = design(frame, tested)
    values = frame[y].to_numpy(dtype=float)
    dates, issuers = frame["day0"].to_numpy(), frame["ticker"].to_numpy()
    joint = stats.wild_cluster_wald(X, values, R, dates, other_clusters=issuers,
                                    draws=draws, seed=seed)
    fit = stats.ols_cluster(X, values, [dates, issuers])
    rows = []
    for position, name in enumerate(tested, start=1):
        row: Dict[str, Any] = {
            "class": name, "contrast": fit["coef"][position] if "coef" in fit else None,
            "se": fit["se"][position] if fit.get("se") else None, "p": None, "p_holm": None,
            "n": int((frame["group"] == name).sum())}
        if each:
            single = np.zeros((1, X.shape[1]))
            single[0, position] = 1.0
            row["p"] = stats.wild_cluster_wald(X, values, single, dates, other_clusters=issuers,
                                               draws=draws, seed=seed).get("p")
        rows.append(row)
    if each:
        for row, adjusted in zip(rows, stats.holm([r["p"] for r in rows])):
            row["p_holm"] = adjusted
    significant = [abs(r["contrast"]) for r in rows
                   if r["p_holm"] is not None and r["p_holm"] < ALPHA and r["contrast"] is not None]
    usable = [r for r in rows if r["contrast"] is not None and r["se"] is not None]
    corrected = None
    if usable and len(usable) == len(rows):
        # E[contrast^2] is the true contrast squared plus its variance, so the
        # mean of (contrast^2 - se^2) is unbiased for the mean squared contrast.
        excess = float(np.mean([r["contrast"] ** 2 - r["se"] ** 2 for r in usable]))
        corrected = float(np.sqrt(max(0.0, excess)))
    return {
        "joint": joint, "fit_status": fit.get("status"), "contrasts": rows,
        "reference_mean": fit["coef"][0] if "coef" in fit else None,
        "largest_significant_contrast": max(significant) if significant else 0.0,
        "significant_classes": [r["class"] for r in rows
                                if r["p_holm"] is not None and r["p_holm"] < ALPHA],
        "rms_contrast_bias_corrected": corrected,
        "n": int(len(frame)), "clusters": fit.get("clusters"),
    }


def _joint(frame: pd.DataFrame, categories: List[str], y: str, clusters: List[str]) -> Dict[str, Any]:
    """The v2 primary test: category indicators against the largest category,
    cluster-robust F. Kept so that v2's test can be reported beside v3's and
    so that the readiness audit's benchmarks stay reproducible."""

    reference = frame["category"].value_counts().idxmax()
    dummies = [c for c in categories if c != reference and (frame["category"] == c).any()]
    table = frame.copy()
    for c in dummies:
        table[f"is_{c}"] = (table["category"] == c).astype(float)
    fit = regress(table, y, [f"is_{c}" for c in dummies], clusters)
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


def _sample(ctx: Context, *, clean_timing: bool = True,
            reference: Sequence[str] = REFERENCE_CLASSES, thinned: bool = True):
    """Eligible rows with a usable CAR, thinned, and the counts along the way."""

    events, counts = prepare(ctx, clean_timing=clean_timing, reference=reference)
    if events.empty:
        return events, counts
    table, usable = _study(ctx, events)
    counts["studied"] = prep.status_counts(table)
    counts["with_car_0_5"] = int(len(usable))
    if thinned and len(usable):
        usable = usable[thin(usable, ctx.panel.calendar)]
        counts["after_issuer_thinning"] = int(len(usable))
    return usable, counts


def _isolated(ctx: Context, sample: pd.DataFrame) -> pd.Series:
    """True where the issuer has no other sampled filing, of any class, with a
    day 0 within five sessions of this one."""

    index = ctx.panel.calendar.index
    every = ctx.kap_events.dropna(subset=["ticker", "day0"])
    positions: Dict[str, np.ndarray] = {}
    for ticker, group in every.groupby("ticker"):
        found = sorted({index(d) for d in group["day0"]} - {None})
        positions[ticker] = np.array(found)
    keep = []
    for row in sample.itertuples():
        here = index(row.day0)
        others = positions.get(row.ticker, np.array([]))
        near = others[(np.abs(others - here) <= PRIMARY_WINDOW[1]) & (others != here)]
        keep.append(len(near) == 0)
    return pd.Series(keep, index=sample.index)


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)

    sample, counts = _sample(ctx)
    if sample.empty:
        return insufficient(result, ["no eligible disclosures"], counts)
    column = window_label(PRIMARY_WINDOW)

    sizes = group_sizes(sample)
    counts["by_group"] = sizes.to_dict("index")
    if REFERENCE not in sizes.index or not sufficient(sizes.loc[REFERENCE], reference=True):
        return insufficient(result, [
            f"the reference group needs {H7_REFERENCE_MIN_EVENTS} events on "
            f"{H7_REFERENCE_MIN_DATES} dates from {H7_REFERENCE_MIN_ISSUERS} issuers"], counts)
    tested = [c for c in INTEREST if c in sizes.index and sufficient(sizes.loc[c], reference=False)]
    counts["tested_classes"] = tested
    counts["classes_below_the_minimum"] = [c for c in INTEREST if c not in tested]
    if not tested:
        return insufficient(result, [
            f"no registered class has {H7_MIN_EVENTS} events on {H7_MIN_DATES} dates "
            f"from {H7_MIN_ISSUERS} issuers"], counts)
    result["sufficiency"]["counts"] = counts
    sample = sample[sample["group"].isin([REFERENCE, *tested])].copy()

    main = contrasts(sample, tested, column)
    joint = main["joint"]
    result["primary"] = primary_block(
        SPEC, estimate=main["largest_significant_contrast"], se=None, ci=None,
        p=joint.get("p"), n=main["n"], clusters=main["clusters"],
        inference_status=joint.get("status", "ok"),
        extra={"joint_test": True, "f": joint.get("f"),
               "bootstrap": {"draws": joint.get("draws"), "seed": joint.get("seed"),
                             "invalid_draws": joint.get("invalid_draws"),
                             "statistic": joint.get("statistic")},
               "tested_classes": tested, "reference_mean": main["reference_mean"],
               "contrasts": main["contrasts"],
               "significant_classes": main["significant_classes"],
               "rms_contrast_bias_corrected": main["rms_contrast_bias_corrected"]})

    # Exploratory: each group's mean in each window, FDR-adjusted together.
    cells = []
    for group in [REFERENCE, *tested]:
        part = sample[sample["group"] == group]
        for window in EVENT_WINDOWS:
            test = mean_test(part[window_label(window)], part["day0"], part["ticker"])
            cells.append({"test": f"{group}, CAR{window}", "category": group,
                          "window": list(window), **test})
    adjusted = stats.benjamini_hochberg([c["p"] for c in cells])
    for cell, p_bh in zip(cells, adjusted):
        cell["p_bh"] = p_bh
    result["exploratory"] = cells

    def variant(name: str, frame: pd.DataFrame, y: str = column, *, method: str = "registered") -> None:
        frame = frame.dropna(subset=[y])
        counted = group_sizes(frame)
        enough = [c for c in tested if c in counted.index and counted.loc[c, "events"] >= H7_MIN_EVENTS]
        if REFERENCE not in counted.index or not enough:
            result["sensitivity"].append({"variant": name, "status": "too_few_events",
                                          "n": int(len(frame))})
            return
        frame = frame[frame["group"].isin([REFERENCE, *enough])]
        entry: Dict[str, Any] = {"variant": name, "n": int(len(frame)), "tested_classes": enough}
        X, R = design(frame, enough)
        values = frame[y].to_numpy(dtype=float)
        dates, issuers = frame["day0"].to_numpy(), frame["ticker"].to_numpy()
        if method == "registered":
            out = contrasts(frame, enough, y, each=False)
            entry.update({"p": out["joint"].get("p"), "status": out["joint"].get("status"),
                          "contrasts": {r["class"]: r["contrast"] for r in out["contrasts"]}})
        elif method == "bootstrap_by_date":
            test = stats.wild_cluster_wald(X, values, R, dates, draws=WILD_BOOTSTRAP_DRAWS,
                                           seed=WILD_BOOTSTRAP_SEED)
            entry.update({"p": test.get("p"), "status": test.get("status")})
        elif method == "cr2":
            test = stats.cr2_wald(X, values, R, dates)
            entry.update({"p": test.get("p"), "status": test.get("status"),
                          "eta": test.get("eta")})
        elif method == "two_way_f":
            test = stats.wald(stats.ols_cluster(X, values, [dates, issuers]), R)
            entry.update({"p": test.get("p"), "status": test.get("status"),
                          "note": "the v2 reference distribution; rejects a true null too often"})
        result["sensitivity"].append(entry)

    pooled, _ = _sample(ctx, reference=POOLED_REFERENCE_CLASSES)
    variant("reference group pooled with unclassified filings", pooled)
    every_bucket, _ = _sample(ctx, clean_timing=False)
    variant("all timing buckets, including during-session disclosures", every_bucket)
    variant("market-adjusted returns", sample, "mar_p0_p5")
    unthinned, _ = _sample(ctx, thinned=False)
    variant("no thinning of an issuer's overlapping events", unthinned)
    variant("events with no other sampled filing by the issuer within five sessions",
            sample[_isolated(ctx, sample)])
    variant("bootstrap of the statistic clustered by day 0 only", sample,
            method="bootstrap_by_date")
    variant("CR2 covariance with Satterthwaite degrees of freedom, by day 0", sample,
            method="cr2")
    variant("two-way F test (the v2 primary test; known to over-reject)", sample,
            method="two_way_f")

    result["tables"]["by_category"] = sizes.reset_index().rename(
        columns={"group": "category"}).to_dict("records")
    result["tables"]["event_time"] = {
        group: event_time_path(ctx.panel, sample[sample["group"] == group].to_dict("records"),
                               sealed=ctx.sealed)
        for group in [REFERENCE, *tested]}
    result["limitations"] += [
        "The sample is the census of 2023 material-event disclosures the gateway "
        "serves, one year, and not a sample of other years.",
        "Issuers are those in the provider's current roster with prices available "
        "from two sources: companies delisted since are missing (survivorship).",
        "A contrast is a difference from procedural filings by listed companies. It "
        "does not show that the disclosure caused the return, and it is not an "
        "abnormal return against the market: the benchmark does not pin down a level.",
        "Classes describe the form a filer chose and the phrases the taxonomy knows. "
        "About four in ten sampled filings are unclassified and outside the test.",
        "Disclosures published during the session are outside the primary sample, "
        "because their day-0 return includes whatever happened before publication.",
        "The reported size is the largest contrast among those individually "
        "significant. Conditional on being selected it overstates the true contrast; "
        "the bias-corrected root-mean-square contrast is reported beside it.",
    ]
    return result
