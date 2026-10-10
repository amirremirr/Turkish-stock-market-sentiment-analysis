"""H6 -- limit-up streak termination."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from stock_research import limits, stats
from stock_research.hypotheses.common import (
    Context, Spec, base_result, insufficient, missing_requirements, primary_block,
    prior_features,
)

MIN_AT_RISK = 300
MIN_PER_OUTCOME = 50
MIN_TEST_DATES = 20
TRAIN_SHARE = 0.7
RIDGE = 1e-3

BASELINE = ["streak_length", "market_ret_0", "volatility_20"]
NEWS = ["news_today", "news_count_3", "has_news", "positive_share_3", "sentiment_change"]
MARKET_STATE = ["log_volume_ratio", "turnover_change", "log_turnover_20"]

SPEC = Spec(
    id="H6",
    title="Limit-up streak termination",
    claim="A limit-up streak is more likely to end when news intensity and "
          "positive sentiment about the company weaken.",
    null="Adding news intensity and sentiment to a model of streak length and "
         "market conditions does not improve the out-of-sample probability "
         "forecast that a streak ends next session.",
    alternative="The model with news features has lower out-of-sample log loss.",
    eligibility="Every session that closed locked at the upper limit under a "
                "verified rule, where the next session has a bar for the ticker. "
                "A streak day whose next session is missing (halt, data gap, end "
                "of sample) is censored and not used.",
    event_time="Prediction cutoff is the close of the streak session; every "
               "feature uses that session or earlier ones.",
    outcome="ended = 1 if the next session does not close locked at the upper limit.",
    estimand="mean out-of-sample log-loss difference, baseline model minus "
             "baseline-plus-news model, on the same test sessions",
    direction=+1,
    material_effect=0.0,
    benchmark="logistic model on streak length, the index return that day and "
              "20-session volatility",
    controls=tuple(BASELINE + MARKET_STATE),
    inference="logistic regression with a small ridge penalty; first 70% of "
              "dates train, last 30% test; loss differences clustered by date",
    sufficiency={"min_at_risk": MIN_AT_RISK, "min_per_outcome": MIN_PER_OUTCOME,
                 "min_test_dates": MIN_TEST_DATES},
    requires=("news_events", "complete_company_news"),
    sensitivity=("Brier score difference", "volume and turnover features without news"),
)


# -- A small logistic regression ----------------------------------------------
def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE,
                 iterations: int = 100) -> Dict[str, np.ndarray]:
    """Newton-Raphson logit on standardised columns, intercept unpenalised."""

    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mean) / sd])
    beta = np.zeros(Z.shape[1])
    penalty = np.eye(Z.shape[1]) * ridge * len(X)
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
        gradient = Z.T @ (y - p) - penalty @ beta
        hessian = (Z * (p * (1 - p))[:, None]).T @ Z + penalty
        step = np.linalg.solve(hessian + 1e-9 * np.eye(len(beta)), gradient)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return {"beta": beta, "mean": mean, "sd": sd}


def predict_logistic(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones(len(X)), (X - model["mean"]) / model["sd"]])
    return 1 / (1 + np.exp(-np.clip(Z @ model["beta"], -30, 30)))


def log_loss_each(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    positives, negatives = p[y == 1], p[y == 0]
    if not len(positives) or not len(negatives):
        return None
    order = pd.Series(np.concatenate([positives, negatives])).rank().to_numpy()
    return float((order[:len(positives)].sum() - len(positives) * (len(positives) + 1) / 2)
                 / (len(positives) * len(negatives)))


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 5) -> List[Dict[str, Any]]:
    table = []
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (p >= low) & ((p < high) | (high == 1.0))
        if inside.any():
            table.append({"bin": f"[{low:.1f},{high:.1f})", "n": int(inside.sum()),
                          "mean_predicted": float(p[inside].mean()),
                          "observed_rate": float(y[inside].mean())})
    return table


# -- At-risk sessions ----------------------------------------------------------
def at_risk_sessions(ctx: Context) -> pd.DataFrame:
    """One row per limit-up session whose next session is observed."""

    stories = ctx.news_events
    if not stories.empty:
        stories = stories[stories["link_confirmed"].astype(bool) & stories["day0"].notna()]
    by_ticker = {t: g for t, g in stories.groupby("ticker")} if not stories.empty else {}
    sessions = ctx.panel.calendar.sessions
    rows: List[Dict[str, Any]] = []
    censored = 0

    for ticker, frame in ctx.panel.bars.items():
        if ticker == ctx.panel.benchmark:
            continue
        flags = limits.detect(frame, ctx.limit_rules)
        locked = flags["up_locked_close"]
        run = limits.streaks(locked)
        news = by_ticker.get(ticker)
        news_pos = (news["day0"].map(ctx.panel.calendar.index).to_numpy()
                    if news is not None else np.array([]))
        news_sent = news["sentiment"].to_numpy(dtype=float) if news is not None else np.array([])

        for i in np.flatnonzero(locked.to_numpy()):
            day = sessions[i]
            if ctx.sealed is not None and (day >= ctx.sealed or (
                    i + 1 < len(sessions) and sessions[i + 1] >= ctx.sealed)):
                continue
            if i + 1 >= len(sessions) or not np.isfinite(frame["close"].iloc[i + 1]) \
                    or not np.isfinite(frame["raw_ret"].iloc[i + 1]):
                censored += 1            # halt, gap, corporate action or sample end
                continue

            recent = (news_pos <= i) & (news_pos > i - 3)
            before = (news_pos <= i - 3) & (news_pos > i - 6)
            recent_sent = news_sent[recent][np.isfinite(news_sent[recent])]
            before_sent = news_sent[before][np.isfinite(news_sent[before])]
            volume = frame["volume"].to_numpy()
            turnover = frame["turnover"].to_numpy()
            prior_volume = volume[max(0, i - 20):i]
            prior_volume = prior_volume[np.isfinite(prior_volume) & (prior_volume > 0)]
            row = {
                "ticker": ticker, "date": day,
                "ended": float(not locked.iloc[i + 1]),
                "streak_length": float(run.iloc[i]),
                "market_ret_0": ctx.panel.market["ret"].iloc[i],
                "news_today": float((news_pos == i).sum()),
                "news_count_3": float(recent.sum()),
                "has_news": float(recent.any()),
                "positive_share_3": float((recent_sent > 0).mean()) if len(recent_sent) else 0.0,
                "sentiment_change": (float(recent_sent.mean() - before_sent.mean())
                                     if len(recent_sent) and len(before_sent) else 0.0),
                "log_volume_ratio": (float(np.log(volume[i] / prior_volume.mean()))
                                     if len(prior_volume) >= 10 and volume[i] > 0 else np.nan),
                "turnover_change": (float(np.log(turnover[i] / turnover[i - 1]))
                                    if i and turnover[i] > 0 and turnover[i - 1] > 0 else np.nan),
            }
            # prior_features looks strictly before the session it is given, so
            # asking for the next session's priors includes the cutoff session
            # itself and nothing after it.
            features = prior_features(ctx.panel, ticker, sessions[i + 1])
            row["volatility_20"] = features["volatility_20"]
            row["log_turnover_20"] = features["log_turnover_20"]
            rows.append(row)

    table = pd.DataFrame(rows)
    table.attrs["censored"] = censored
    return table


def _oos(train: pd.DataFrame, test: pd.DataFrame, xs: Sequence[str]) -> np.ndarray:
    model = fit_logistic(train[list(xs)].to_numpy(dtype=float), train["ended"].to_numpy(dtype=float))
    return predict_logistic(model, test[list(xs)].to_numpy(dtype=float))


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    table = at_risk_sessions(ctx)
    counts: Dict[str, Any] = {"at_risk_sessions": int(len(table)),
                              "censored": int(table.attrs.get("censored", 0))}
    if len(table):
        counts["ended"] = int(table["ended"].sum())
        counts["continued"] = int((1 - table["ended"]).sum())
    if blocked:
        result["limitations"].append(
            f"{len(table)} limit-up sessions with an observed next session were "
            "detected from prices; the news features cannot be measured.")
        return insufficient(result, blocked, counts)

    complete = table.dropna(subset=BASELINE + NEWS + MARKET_STATE).sort_values("date") \
        if len(table) else table
    counts["complete_rows"] = int(len(complete))
    dates = sorted(complete["date"].unique()) if len(complete) else []
    cut = int(len(dates) * TRAIN_SHARE)
    train = complete[complete["date"].isin(dates[:cut])] if len(complete) else complete
    test = complete[complete["date"].isin(dates[cut:])] if len(complete) else complete
    counts["test_dates"] = int(test["date"].nunique()) if len(test) else 0
    enough = (
        len(complete) >= MIN_AT_RISK
        and min(complete["ended"].sum(), (1 - complete["ended"]).sum()) >= MIN_PER_OUTCOME
        and counts["test_dates"] >= MIN_TEST_DATES
        and 0 < train["ended"].mean() < 1
    )
    if not enough:
        return insufficient(result, [
            f"{len(complete)} complete at-risk sessions, {counts.get('ended', 0)} ended, "
            f"{counts['test_dates']} test dates; needs {MIN_AT_RISK}, "
            f"{MIN_PER_OUTCOME} per outcome and {MIN_TEST_DATES}"], counts)
    result["sufficiency"]["counts"] = counts

    y = test["ended"].to_numpy(dtype=float)
    predictions = {
        "baseline": _oos(train, test, BASELINE),
        "baseline_plus_news": _oos(train, test, BASELINE + NEWS),
        "baseline_plus_market_state": _oos(train, test, BASELINE + MARKET_STATE),
        "full": _oos(train, test, BASELINE + MARKET_STATE + NEWS),
    }
    clusters = [test["date"].to_numpy()]
    difference = stats.cluster_mean(
        log_loss_each(y, predictions["baseline"]) - log_loss_each(y, predictions["baseline_plus_news"]),
        clusters)
    result["primary"] = primary_block(
        SPEC, estimate=difference["mean"], se=difference["se"], ci=difference["ci"],
        p=difference["p"], n=difference["n"], clusters=difference["clusters"],
        inference_status=difference["status"])

    for name, p in predictions.items():
        result["exploratory"].append({
            "test": f"model: {name}", "log_loss": float(log_loss_each(y, p).mean()),
            "brier": float(np.mean((p - y) ** 2)), "auc": auc(y, p), "n": int(len(y))})
    result["tables"]["calibration_baseline_plus_news"] = calibration(y, predictions["baseline_plus_news"])
    result["tables"]["calibration_baseline"] = calibration(y, predictions["baseline"])
    result["tables"]["hazard_by_length"] = [
        {"streak_length": int(length), "ended_rate": float(group["ended"].mean()),
         "n": int(len(group))}
        for length, group in complete.groupby(complete["streak_length"].clip(upper=6))]
    result["sensitivity"].append({
        "variant": "Brier score, baseline minus baseline-plus-news",
        **stats.cluster_mean((predictions["baseline"] - y) ** 2
                             - (predictions["baseline_plus_news"] - y) ** 2, clusters)})
    result["sensitivity"].append({
        "variant": "log loss, market-state model minus full model",
        **stats.cluster_mean(log_loss_each(y, predictions["baseline_plus_market_state"])
                             - log_loss_each(y, predictions["full"]), clusters)})
    result["limitations"] += [
        "A streak day followed by a missing session is censored, which removes "
        "halted stocks; halts are not random.",
        "Repeated streaks of one ticker are treated as separate at-risk spells.",
    ]
    return result
