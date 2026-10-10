"""H3 -- sentiment surprise versus raw sentiment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from stock_research import stats
from stock_research.eventstudy import STATUS_OK
from stock_research.hypotheses import prep
from stock_research.hypotheses.common import (
    Context, Spec, base_result, insufficient, missing_requirements, primary_block,
)

LOOKBACK_SESSIONS = 60
MIN_HISTORY = 5
MIN_MATCHED = 200
MIN_TEST_DATES = 30
TRAIN_SHARE = 0.7
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
                "timing) whose ticker has at least five earlier news events in "
                "the prior 60 sessions with non-zero variance; CAR available.",
    event_time="Day 0 is the first session able to act on the news.",
    outcome="CAR(0,+5), market model.",
    estimand="mean out-of-sample difference in squared error, raw-sentiment model "
             "minus surprise model, on the same test events",
    direction=+1,
    material_effect=0.0,
    benchmark="a characteristics-only model (prior_ret_5, log_turnover_20, volatility_20)",
    controls=tuple(BASELINE),
    inference="events ordered by day 0; first 70% of dates train, last 30% test; "
              "loss differences clustered by day 0",
    sufficiency={"min_matched_events": MIN_MATCHED, "min_test_dates": MIN_TEST_DATES,
                 "min_history": MIN_HISTORY, "lookback_sessions": LOOKBACK_SESSIONS},
    requires=("news_events",),
    sensitivity=("combined model versus baseline", "absolute error instead of squared"),
)


def sentiment_surprise(events: pd.DataFrame, calendar) -> pd.Series:
    """z-score of each event's sentiment against the same ticker's events on
    strictly earlier day 0s within the lookback. Missing when history is short
    or has no variance: an undefined surprise is not a zero surprise."""

    out = pd.Series(np.nan, index=events.index, dtype=float)
    position = events["day0"].map(lambda d: calendar.index(d) if isinstance(d, str) else None)
    for _, group in events.assign(_pos=position).dropna(subset=["_pos"]).groupby("ticker"):
        ordered = group.sort_values("_pos")
        pos = ordered["_pos"].to_numpy()
        sent = ordered["sentiment"].to_numpy(dtype=float)
        for i, index in enumerate(ordered.index):
            earlier = (pos < pos[i]) & (pos >= pos[i] - LOOKBACK_SESSIONS)
            history = sent[earlier]
            history = history[np.isfinite(history)]
            if len(history) < MIN_HISTORY:
                continue
            spread = history.std(ddof=1)
            if not np.isfinite(spread) or spread < 1e-9:
                continue
            out.loc[index] = (sent[i] - history.mean()) / spread
    return out


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

    # The baseline uses every confirmed mention of the ticker, not only the
    # admissible ones: the surprise is relative to all the stock's news.
    history = ctx.news_events[ctx.news_events["link_confirmed"].astype(bool)
                              & ctx.news_events["sentiment"].notna()
                              & ctx.news_events["day0"].notna()]
    history = history.groupby(["ticker", "day0"], as_index=False)["sentiment"].mean()
    history["surprise"] = sentiment_surprise(history, ctx.panel.calendar)
    events = events.merge(history[["ticker", "day0", "surprise"]], on=["ticker", "day0"], how="left")

    table = prep.study(ctx, events)
    counts["studied"] = prep.status_counts(table)
    columns = ["car_p0_p5", "sentiment", "surprise", *BASELINE]
    matched = table[(table["status"] == STATUS_OK)].dropna(subset=columns).sort_values("day0")
    counts["matched"] = int(len(matched))
    dates = sorted(matched["day0"].unique())
    cut = int(len(dates) * TRAIN_SHARE)
    train = matched[matched["day0"].isin(dates[:cut])]
    test = matched[matched["day0"].isin(dates[cut:])]
    counts.update({"train": int(len(train)), "test": int(len(test)),
                   "test_dates": int(test["day0"].nunique())})
    if len(matched) < MIN_MATCHED or test["day0"].nunique() < MIN_TEST_DATES:
        return insufficient(result, [
            f"{len(matched)} matched events, {test['day0'].nunique()} test dates; "
            f"needs {MIN_MATCHED} and {MIN_TEST_DATES}"], counts)
    result["sufficiency"]["counts"] = counts

    y = "car_p0_p5"
    models = {"baseline": BASELINE, "raw": BASELINE + ["sentiment"],
              "surprise": BASELINE + ["surprise"],
              "combined": BASELINE + ["sentiment", "surprise"]}
    actual = test[y].to_numpy(dtype=float)
    errors = {name: actual - _fit_predict(train, test, xs, y) for name, xs in models.items()}
    clusters = [test["day0"].to_numpy()]

    difference = stats.cluster_mean(errors["raw"] ** 2 - errors["surprise"] ** 2, clusters)
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
        **stats.cluster_mean(errors["baseline"] ** 2 - errors["combined"] ** 2, clusters)})
    result["sensitivity"].append({
        "variant": "absolute error, raw minus surprise",
        **stats.cluster_mean(np.abs(errors["raw"]) - np.abs(errors["surprise"]), clusters)})
    result["limitations"].append(
        "One chronological split; no model selection was done on the test set.")
    return result
