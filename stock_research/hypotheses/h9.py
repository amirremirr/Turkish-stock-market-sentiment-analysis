"""H9 -- English versus Turkish news lead-lag."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stock_research import stats
from stock_research.hypotheses.common import (
    Context, Spec, base_result, insufficient, missing_requirements, primary_block,
)

MATCH_WINDOW_HOURS = 48
MIN_MATCHED = 100
MIN_DATES = 30
COLUMNS = ("story_id", "subject", "first_published_utc", "sentiment")

SPEC = Spec(
    id="H9",
    title="English versus Turkish news lead-lag",
    claim="English-language news sometimes reports developments about Turkish "
          "companies or the Turkish economy before Turkish outlets do.",
    null="For the same event, the English and Turkish first reports are equally "
         "likely to come first.",
    alternative="The English first report comes first more often than not.",
    eligibility="Events reported in both languages: the same subject (issuer or "
                "macro topic), first reports within 48 hours of each other, "
                "matched one to one; syndicated copies collapsed to the earliest.",
    event_time="First publication time in each language, in UTC.",
    outcome="english_first = 1 if the English first report precedes the Turkish one.",
    estimand="share of matched events where English is first, minus one half",
    direction=+1,
    material_effect=0.05,
    benchmark="one half: neither language systematically first",
    controls=(),
    inference="mean with standard errors clustered by the calendar date of the "
              "earlier report",
    sufficiency={"min_matched_events": MIN_MATCHED, "min_dates": MIN_DATES},
    requires=("english_news", "news_events"),
    sensitivity=("events at least one hour apart",),
    parameters={"match_window_hours": MATCH_WINDOW_HOURS, "min_matched": MIN_MATCHED,
                "min_dates": MIN_DATES, "ties": "excluded"},
)


def match_streams(english: pd.DataFrame, turkish: pd.DataFrame) -> pd.DataFrame:
    """One-to-one matches between first reports of the same subject.

    Within a subject, pairs are taken in order of how close they are in time,
    each story used at most once, and only inside the 48-hour window. The
    order of the two timestamps is the measurement; nothing about prices is
    read here.
    """

    rows: List[Dict[str, Any]] = []
    window = pd.Timedelta(hours=MATCH_WINDOW_HOURS)
    for subject, en in english.groupby("subject"):
        tr = turkish[turkish["subject"] == subject]
        if tr.empty:
            continue
        en_time = pd.to_datetime(en["first_published_utc"], utc=True)
        tr_time = pd.to_datetime(tr["first_published_utc"], utc=True)
        candidates = sorted(
            ((abs(te - tt), i, j) for i, te in en_time.items() for j, tt in tr_time.items()
             if abs(te - tt) <= window), key=lambda item: item[0])
        used_en, used_tr = set(), set()
        for _, i, j in candidates:
            if i in used_en or j in used_tr:
                continue
            used_en.add(i)
            used_tr.add(j)
            lag = (tr_time[j] - en_time[i]).total_seconds() / 3600
            rows.append({
                "subject": subject, "english_story": en.loc[i, "story_id"],
                "turkish_story": tr.loc[j, "story_id"],
                "english_time": en_time[i], "turkish_time": tr_time[j],
                "lag_hours": lag, "english_first": float(lag > 0),
                "tie": bool(lag == 0),
                "sentiment_divergence": (en.loc[i, "sentiment"] - tr.loc[j, "sentiment"]
                                         if pd.notna(en.loc[i, "sentiment"])
                                         and pd.notna(tr.loc[j, "sentiment"]) else np.nan),
                "date": min(en_time[i], tr_time[j]).strftime("%Y-%m-%d"),
            })
    return pd.DataFrame(rows)


def run(ctx: Context) -> Dict[str, Any]:
    result = base_result(SPEC, ctx)
    blocked = missing_requirements(SPEC, ctx)
    if blocked:
        return insufficient(result, blocked)
    english, turkish = ctx.english_events, ctx.news_events
    if english.empty or turkish.empty or not set(COLUMNS) <= set(english.columns) \
            or not set(COLUMNS) <= set(turkish.columns):
        return insufficient(result, ["a language stream is empty or lacks first-report times"],
                            {"english": int(len(english)), "turkish": int(len(turkish))})

    def first_reports(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.dropna(subset=["subject", "first_published_utc"])
        return frame.sort_values("first_published_utc").drop_duplicates("story_id")

    matched = match_streams(first_reports(english), first_reports(turkish))
    counts = {"english_stories": int(len(english)), "turkish_stories": int(len(turkish)),
              "matched": int(len(matched))}
    matched = matched[~matched["tie"]] if len(matched) else matched
    counts["matched_excluding_ties"] = int(len(matched))
    if len(matched) < MIN_MATCHED or matched["date"].nunique() < MIN_DATES:
        return insufficient(result, [
            f"{len(matched)} matched events on "
            f"{matched['date'].nunique() if len(matched) else 0} dates; needs "
            f"{MIN_MATCHED} on {MIN_DATES}"], counts)
    result["sufficiency"]["counts"] = counts

    test = stats.cluster_mean(matched["english_first"].to_numpy() - 0.5, [matched["date"].to_numpy()])
    result["primary"] = primary_block(
        SPEC, estimate=test["mean"], se=test["se"], ci=test["ci"], p=test["p"],
        n=test["n"], clusters=test["clusters"], inference_status=test["status"],
        extra={"english_first_share": float(matched["english_first"].mean())})

    lags = matched["lag_hours"]
    result["tables"]["lag_hours"] = {
        "median": float(lags.median()), "p10": float(lags.quantile(0.1)),
        "p90": float(lags.quantile(0.9)), "mean": float(lags.mean())}
    counts_by_bin, edges = np.histogram(lags.clip(-24, 24), bins=16, range=(-24, 24))
    result["tables"]["lag_histogram"] = {"edges": edges.tolist(), "counts": counts_by_bin.tolist()}
    divergence = matched["sentiment_divergence"].dropna()
    if len(divergence) >= 30:
        result["exploratory"].append({
            "test": "sentiment divergence (English minus Turkish)",
            **stats.cluster_mean(divergence.to_numpy(),
                                 [matched.loc[divergence.index, "date"].to_numpy()])})
    apart = matched[matched["lag_hours"].abs() >= 1]
    result["sensitivity"].append({
        "variant": "first reports at least one hour apart", "n": int(len(apart)),
        **stats.cluster_mean(apart["english_first"].to_numpy() - 0.5, [apart["date"].to_numpy()])})

    ok, reason = ctx.available("intraday_prices")
    result["limitations"] += [
        "Only the order of publication is tested. Whether either language's "
        "sentiment leads the other's, or leads prices, needs intraday data: "
        + ("available." if ok else f"not available ({reason})"),
        "A daily bar cannot place a price move before or after a report "
        "published the same day, so no information-transmission claim is made.",
        "Matching is by subject and time; two different events about one "
        "issuer inside 48 hours can be paired wrongly.",
    ]
    return result
