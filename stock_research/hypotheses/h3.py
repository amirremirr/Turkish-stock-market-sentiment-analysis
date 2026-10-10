"""H3 -- sentiment surprise versus raw sentiment."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stock_research import stats
from stock_research.eventstudy import STATUS_OK
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    Context, Spec, base_result, chronological_split, insufficient,
    missing_requirements, primary_block,
)

LOOKBACK_SESSIONS = 60
MIN_HISTORY = 5
MIN_MATCHED = 200
MIN_TEST_DATES = 30
TRAIN_SHARE = 0.7
OUTCOME_WINDOW = (0, 5)
BASELINE = ["prior_ret_5", "log_turnover_20", "volatility_20"]

SPEC = Spec(
    id="H3",
    title="Sentiment surprise versus raw sentiment",
    claim="A stock's sentiment surprise, relative to its own recent news, "
          "predicts subsequent returns better than the raw sentiment level.",
    null="Out of sample, a model using the surprise has the same squared error "
         "as one using the raw level.",
    alternative="The surprise model has lower out-of-sample squared error.",
    eligibility="Admissible news events (confirmed, single-issuer, material, clean "
                "timing) whose ticker has at least five earlier news days in the "
                "prior 60 sessions with non-zero variance; CAR available.",
    event_time="Day 0 is the first session able to act on the news.",
    outcome="CAR(0,+5), market model.",
    estimand="mean out-of-sample difference in squared error, raw-sentiment model "
             "minus surprise model, on the same test events",
    direction=+1,
    material_effect=0.0,
    benchmark="a characteristics-only model (prior_ret_5, log_turnover_20, volatility_20)",
    controls=tuple(BASELINE),
    inference="events ordered by day 0; first 70% of dates train, last 30% test; "
              "training events whose six-session outcome window had not closed "
              "before the first test date are dropped; loss differences clustered "
              "two ways, by day 0 and by issuer",
    sufficiency={"min_matched_events": MIN_MATCHED, "min_test_dates": MIN_TEST_DATES},
    requires=("news_events",),
    sensitivity=("combined model versus baseline", "absolute error instead of squared",
                 "clustering by day 0 only"),
    parameters={"lookback_sessions": LOOKBACK_SESSIONS, "min_history": MIN_HISTORY,
                "min_matched": MIN_MATCHED, "min_test_dates": MIN_TEST_DATES,
                "train_share": TRAIN_SHARE, "outcome_window": list(OUTCOME_WINDOW),
                "embargo_sessions": OUTCOME_WINDOW[1], "baseline": BASELINE},
)


def sentiment_baseline(history: pd.DataFrame, calendar) -> pd.DataFrame:
    """Mean and spread of a ticker's sentiment on strictly earlier day 0s
    within the lookback, for every ticker-day in *history*.

    Both are missing when there are fewer than ``MIN_HISTORY`` earlier days or
    they have no variance: an undefined surprise is not a zero surprise.
    """

    out = history[["ticker", "day0"]].copy()
    out["baseline_mean"], out["baseline_sd"] = np.nan, np.nan
    position = history["day0"].map(lambda d: calendar.index(d) if isinstance(d, str) else None)
    for _, group in history.assign(_pos=position).dropna(subset=["_pos"]).groupby("ticker"):
        ordered = group.sort_values("_pos")
        pos = ordered["_pos"].to_numpy()
        sent = ordered["sentiment"].to_numpy(dtype=float)
        for i, index in enumerate(ordered.index):
            earlier = (pos < pos[i]) & (pos >= pos[i] - LOOKBACK_SESSIONS)
            past = sent[earlier]
            past = past[np.isfinite(past)]
            if len(past) < MIN_HISTORY:
                continue
            spread = past.std(ddof=1)
            if not np.isfinite(spread) or spread < 1e-9:
                continue
            out.loc[index, ["baseline_mean", "baseline_sd"]] = [past.mean(), spread]
    return out


def sentiment_surprise(history: pd.DataFrame, calendar) -> pd.Series:
    """z-score of each ticker-day's sentiment against its own baseline."""

    baseline = sentiment_baseline(history, calendar)
    return (history["sentiment"] - baseline["baseline_mean"]) / baseline["baseline_sd"]


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, xs: List[str], y: str) -> np.ndarray:
    X = np.column_stack([np.ones(len(train)), train[xs].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(X, train[y].to_numpy(dtype=float), rcond=None)
    return np.column_stack([np.ones(len(test)), test[xs].to_numpy(dtype=float)]) @ beta


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)

    events, counts = prep.admissible_news(ctx)
    if events.empty:
        return insufficient(result, ["no admissible news events"], counts)

    # The baseline is the stock's news in general: every confirmed mention,
    # not only the admissible ones. The value being compared with it is the
    # event's own sentiment -- the same number the raw model uses -- so the
    # two models differ only in whether that number is taken relative to the
    # stock's recent level.
    history = ctx.news_events[ctx.news_events["link_confirmed"].astype(bool)
                              & ctx.news_events["sentiment"].notna()
                              & ctx.news_events["day0"].notna()]
    history = history.groupby(["ticker", "day0"], as_index=False)["sentiment"].mean()
    baseline = sentiment_baseline(history, ctx.panel.calendar)
    events = events.merge(baseline, on=["ticker", "day0"], how="left")
    events["surprise"] = (events["sentiment"] - events["baseline_mean"]) / events["baseline_sd"]

    table = prep.study(ctx, events)
    counts["studied"] = prep.status_counts(table)
    columns = ["car_p0_p5", "sentiment", "surprise", *BASELINE]
    matched = table[(table["status"] == STATUS_OK)].dropna(subset=columns).sort_values("day0")
    counts["matched"] = int(len(matched))
    train, test, split = chronological_split(matched, "day0", TRAIN_SHARE,
                                             OUTCOME_WINDOW[1], ctx.panel.calendar)
    counts.update({"train": int(len(train)), "test": int(len(test)),
                   "test_dates": int(test["day0"].nunique()) if len(test) else 0, "split": split})
    if len(matched) < MIN_MATCHED or counts["test_dates"] < MIN_TEST_DATES or len(train) < 30:
        return insufficient(result, [
            f"{len(matched)} matched events, {counts['test_dates']} test dates; "
            f"needs {MIN_MATCHED} and {MIN_TEST_DATES}"], counts)
    result["sufficiency"]["counts"] = counts

    y = "car_p0_p5"
    models = {"baseline": BASELINE, "raw": BASELINE + ["sentiment"],
              "surprise": BASELINE + ["surprise"],
              "combined": BASELINE + ["sentiment", "surprise"]}
    actual = test[y].to_numpy(dtype=float)
    errors = {name: actual - _fit_predict(train, test, xs, y) for name, xs in models.items()}
    two_way = [test["day0"].to_numpy(), test["ticker"].to_numpy()]

    difference = stats.cluster_mean(errors["raw"] ** 2 - errors["surprise"] ** 2, two_way)
    result["primary"] = primary_block(
        SPEC, estimate=difference["mean"], se=difference["se"], ci=difference["ci"],
        p=difference["p"], n=difference["n"], clusters=difference["clusters"],
        inference_status=difference["status"])

    for name, error in errors.items():
        predicted = actual - error
        nonzero = actual != 0
        result["exploratory"].append({
            "test": f"model: {name}", "mse": float(np.mean(error ** 2)),
            "mae": float(np.mean(np.abs(error))),
            "directional_accuracy": float(np.mean((predicted[nonzero] > 0) == (actual[nonzero] > 0))),
            "n": int(len(error)),
        })
    result["sensitivity"].append({
        "variant": "combined model versus baseline (squared error)",
        **stats.cluster_mean(errors["baseline"] ** 2 - errors["combined"] ** 2, two_way)})
    result["sensitivity"].append({
        "variant": "absolute error, raw minus surprise",
        **stats.cluster_mean(np.abs(errors["raw"]) - np.abs(errors["surprise"]), two_way)})
    result["sensitivity"].append({
        "variant": "clustering by day 0 only",
        **stats.cluster_mean(errors["raw"] ** 2 - errors["surprise"] ** 2, two_way[:1])})
    result["limitations"].append(
        "One chronological split; no model selection was done on the test set.")
    return result
