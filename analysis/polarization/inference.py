"""Observational inference for polarization in Turkish financial headlines.

The module deliberately separates two questions:

* ``selection`` describes which topics/stories each camp covers.
* ``framing`` compares sentiment while holding an explicitly shared event fixed.

The database's current ``events.event_id`` bridge is one event per headline.  It
therefore cannot identify shared stories.  An explicit repeated
``canonical_event_id`` (or ``shared_event_id``) is used when supplied; otherwise
the framing section is a clearly labelled lexical/date matching sensitivity.
No result produced here is a causal estimate of political bias.

Run from the repository root with::

    python -m analysis.polarization.inference --db finance_sentiment.db
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_PRO_GOVERNMENT_SOURCES = (
    "aa_ekonomi",
    "aa_politika",
    "sabah_ekonomi",
)
DEFAULT_OPPOSITION_SOURCES = (
    "sozcu_gundem",
    "sozcu_ekonomi",
    "cumhuriyet_ekonomi",
)

_MISSING_CATEGORY = "__missing_category__"
_CANONICAL_EVENT_COLUMNS = ("canonical_event_id", "shared_event_id")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "aciklama",
        "acikladi",
        "aciklandi",
        "bugun",
        "buyuk",
        "daha",
        "dolar",
        "ekonomi",
        "ekonomik",
        "enflasyon",
        "euro",
        "faiz",
        "icin",
        "kadar",
        "milyar",
        "milyon",
        "olarak",
        "oldu",
        "piyasa",
        "piyasalar",
        "rekor",
        "sonra",
        "turkiye",
        "turkiyenin",
        "yeni",
        "yuzde",
    }
)


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe records without leaking numpy scalar types."""

    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or (not isinstance(value, (list, dict, set)) and pd.isna(value)):
                converted[key] = None
            elif isinstance(value, np.integer):
                converted[key] = int(value)
            elif isinstance(value, np.floating):
                converted[key] = _finite_or_none(value)
            elif isinstance(value, set):
                converted[key] = sorted(str(item) for item in value)
            elif isinstance(value, pd.Timestamp):
                converted[key] = value.isoformat()
            else:
                converted[key] = value
        records.append(converted)
    return records


def _iso_date(value: Any) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat()


def _normalise_text(value: Any) -> str:
    text = str(value or "").casefold().replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(character for character in text if not unicodedata.combining(character))


def significant_tokens(title: Any, *, minimum_length: int = 5) -> frozenset[str]:
    """Extract deterministic content tokens for the fallback matcher."""

    tokens = _TOKEN_RE.findall(_normalise_text(title))
    return frozenset(
        token
        for token in tokens
        if len(token) >= minimum_length and token not in _STOPWORDS and not token.isdigit()
    )


def _prepare_headlines(
    raw: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    pro_government_sources: Sequence[str],
    opposition_sources: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    aliases = {
        "published_at": "date",
        "sentiment_score": "sentiment",
        "s": "sentiment",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame.rename(columns={source: target}, inplace=True)

    missing = [column for column in ("source", "date", "sentiment") if column not in frame]
    if missing:
        raise KeyError(f"polarization input missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["source"] = frame["source"].astype(str)
    pro_set = frozenset(str(item) for item in pro_government_sources)
    opposition_set = frozenset(str(item) for item in opposition_sources)
    overlap = sorted(pro_set & opposition_set)
    if overlap:
        raise ValueError(f"camp source lists overlap: {overlap}")

    frame["camp"] = np.select(
        [frame["source"].isin(pro_set), frame["source"].isin(opposition_set)],
        ["pro_government", "opposition"],
        default="outside_camps",
    )
    frame["sentiment"] = pd.to_numeric(frame["sentiment"], errors="coerce")
    frame["date"] = [_iso_date(value) for value in frame["date"]]

    initial_count = len(frame)
    outside_count = int((frame["camp"] == "outside_camps").sum())
    invalid_score = int((~np.isfinite(frame["sentiment"])).sum())
    missing_date = int(frame["date"].isna().sum())
    keep = (
        frame["camp"].isin(("pro_government", "opposition"))
        & np.isfinite(frame["sentiment"])
        & frame["date"].notna()
    )
    frame = frame.loc[keep].copy().reset_index(drop=True)
    if "title" not in frame:
        frame["title"] = ""
    frame["title"] = frame["title"].fillna("").astype(str)
    if "category" not in frame:
        frame["category"] = _MISSING_CATEGORY
    frame["category"] = frame["category"].fillna(_MISSING_CATEGORY).astype(str)
    frame["_row_key"] = [
        f"{value}:{position}"
        for position, value in enumerate(
            frame["headline_id"] if "headline_id" in frame else frame.index
        )
    ]

    diagnostics = {
        "input_rows": int(initial_count),
        "analyzed_rows": int(len(frame)),
        "outside_configured_camps": outside_count,
        "invalid_or_missing_sentiment": invalid_score,
        "missing_or_invalid_date": missing_date,
    }
    return frame, diagnostics


def _group_summary(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for value, group in frame.groupby(column, sort=True, dropna=False):
        scores = group["sentiment"]
        rows.append(
            {
                column: str(value),
                "count": int(len(scores)),
                "mean": _finite_or_none(scores.mean()),
                "standard_deviation": _finite_or_none(scores.std(ddof=1)),
            }
        )
    return rows


def _mean_difference(frame: pd.DataFrame) -> dict[str, Any]:
    pro = frame.loc[frame["camp"] == "pro_government", "sentiment"]
    opposition = frame.loc[frame["camp"] == "opposition", "sentiment"]
    result: dict[str, Any] = {
        "definition": "pro_government mean minus opposition mean",
        "pro_government_count": int(len(pro)),
        "opposition_count": int(len(opposition)),
        "estimate": None,
        "standardized_effect_size": None,
        "standardized_effect_definition": (
            "Cohen's d using the pooled within-camp sample standard deviation"
        ),
        "diagnostic": None,
    }
    if pro.empty or opposition.empty:
        result["diagnostic"] = "both camps need at least one scored headline"
        return result

    result["estimate"] = _finite_or_none(pro.mean() - opposition.mean())
    if len(pro) < 2 or len(opposition) < 2:
        result["diagnostic"] = "Cohen's d needs at least two observations in each camp"
        return result
    degrees = len(pro) + len(opposition) - 2
    pooled_variance = (
        (len(pro) - 1) * pro.var(ddof=1) + (len(opposition) - 1) * opposition.var(ddof=1)
    ) / degrees
    if not math.isfinite(float(pooled_variance)) or pooled_variance <= 0:
        result["diagnostic"] = "pooled within-camp variance is zero or undefined"
        return result
    result["standardized_effect_size"] = _finite_or_none(
        float(result["estimate"]) / math.sqrt(float(pooled_variance))
    )
    return result


def date_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    repetitions: int = 2_000,
    seed: int = 20260707,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap the camp mean difference by resampling whole publication dates."""

    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    required = {"date", "camp", "sentiment"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"bootstrap input missing required columns: {', '.join(missing)}")

    clusters = sorted(str(value) for value in frame["date"].dropna().unique())
    result: dict[str, Any] = {
        "status": "inadequate",
        "resampling_unit": "publication_date",
        "cluster_count": len(clusters),
        "repetitions_requested": int(repetitions),
        "repetitions_completed": 0,
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "lower": None,
        "upper": None,
        "diagnostic": None,
    }
    if len(clusters) < 2:
        result["diagnostic"] = "date-cluster bootstrap needs at least two dates"
        return result

    grouped = {date: frame.loc[frame["date"].astype(str) == date] for date in clusters}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        replicate = pd.concat([grouped[str(date)] for date in sampled], ignore_index=True)
        means = replicate.groupby("camp")["sentiment"].mean()
        if {"pro_government", "opposition"}.issubset(means.index):
            estimate = float(means["pro_government"] - means["opposition"])
            if math.isfinite(estimate):
                estimates.append(estimate)

    result["repetitions_completed"] = len(estimates)
    if not estimates:
        result["diagnostic"] = "no bootstrap replicate contained observations from both camps"
        return result
    alpha = (1.0 - confidence_level) / 2.0
    result.update(
        {
            "status": "ok",
            "lower": float(np.quantile(estimates, alpha)),
            "upper": float(np.quantile(estimates, 1.0 - alpha)),
            "diagnostic": (
                None
                if len(estimates) == repetitions
                else "some resamples contained only one camp and were omitted"
            ),
        }
    )
    return result


def _canonical_event_column(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in frame:
            raise KeyError(f"canonical event column not found: {requested}")
        return requested
    return next((column for column in _CANONICAL_EVENT_COLUMNS if column in frame), None)


def _shared_event_gaps(frame: pd.DataFrame, event_column: str) -> pd.DataFrame:
    eligible = frame.loc[frame[event_column].notna()].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=("event_id", "pro_government_mean", "opposition_mean", "gap", "pro_count", "opposition_count")
        )
    grouped = eligible.groupby([event_column, "camp"], sort=True)["sentiment"].agg(["mean", "size"])
    means = grouped["mean"].unstack("camp")
    sizes = grouped["size"].unstack("camp")
    if not {"pro_government", "opposition"}.issubset(means.columns):
        return pd.DataFrame(
            columns=("event_id", "pro_government_mean", "opposition_mean", "gap", "pro_count", "opposition_count")
        )
    shared = means.dropna(subset=["pro_government", "opposition"]).copy()
    result = pd.DataFrame(
        {
            "event_id": shared.index.astype(str),
            "pro_government_mean": shared["pro_government"].to_numpy(),
            "opposition_mean": shared["opposition"].to_numpy(),
            "gap": (shared["pro_government"] - shared["opposition"]).to_numpy(),
            "pro_count": sizes.loc[shared.index, "pro_government"].astype(int).to_numpy(),
            "opposition_count": sizes.loc[shared.index, "opposition"].astype(int).to_numpy(),
        }
    )
    return result.sort_values("event_id", kind="stable").reset_index(drop=True)


def lexical_date_pairs(
    frame: pd.DataFrame,
    *,
    minimum_shared_tokens: int = 2,
    window_days: int = 1,
    minimum_token_length: int = 5,
) -> pd.DataFrame:
    """Greedily select deterministic one-to-one lexical/date fallback pairs.

    Candidate edges are ranked globally, so neither camp's headline can be
    reused.  These are *inferred* matches, not verified common events.
    """

    if minimum_shared_tokens <= 0:
        raise ValueError("minimum_shared_tokens must be positive")
    if window_days < 0:
        raise ValueError("window_days cannot be negative")
    required = {"camp", "date", "sentiment", "title"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"matcher input missing required columns: {', '.join(missing)}")

    work = frame.copy()
    if "_row_key" not in work:
        work["_row_key"] = [str(index) for index in work.index]
    work["_match_date"] = pd.to_datetime(work["date"], errors="coerce")
    work["_tokens"] = [
        significant_tokens(title, minimum_length=minimum_token_length)
        for title in work["title"]
    ]
    pro = work.loc[work["camp"] == "pro_government"]
    opposition = work.loc[work["camp"] == "opposition"]
    candidates: list[dict[str, Any]] = []
    for pro_index, pro_row in pro.iterrows():
        if not pro_row["_tokens"] or pd.isna(pro_row["_match_date"]):
            continue
        for opposition_index, opposition_row in opposition.iterrows():
            if not opposition_row["_tokens"] or pd.isna(opposition_row["_match_date"]):
                continue
            date_gap = abs((pro_row["_match_date"] - opposition_row["_match_date"]).days)
            if date_gap > window_days:
                continue
            shared = pro_row["_tokens"] & opposition_row["_tokens"]
            if len(shared) < minimum_shared_tokens:
                continue
            union = pro_row["_tokens"] | opposition_row["_tokens"]
            candidates.append(
                {
                    "_pro_index": pro_index,
                    "_opposition_index": opposition_index,
                    "_pro_key": str(pro_row["_row_key"]),
                    "_opposition_key": str(opposition_row["_row_key"]),
                    "shared_token_count": len(shared),
                    "token_jaccard": len(shared) / len(union),
                    "date_gap_days": int(date_gap),
                    "shared_tokens": sorted(shared),
                }
            )
    candidates.sort(
        key=lambda row: (
            -row["shared_token_count"],
            -row["token_jaccard"],
            row["date_gap_days"],
            row["_pro_key"],
            row["_opposition_key"],
        )
    )

    used_pro: set[Any] = set()
    used_opposition: set[Any] = set()
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        pro_index = candidate["_pro_index"]
        opposition_index = candidate["_opposition_index"]
        if pro_index in used_pro or opposition_index in used_opposition:
            continue
        used_pro.add(pro_index)
        used_opposition.add(opposition_index)
        pro_row = work.loc[pro_index]
        opposition_row = work.loc[opposition_index]
        selected.append(
            {
                "match_id": f"lexical_{len(selected) + 1:04d}",
                "pro_row_key": candidate["_pro_key"],
                "opposition_row_key": candidate["_opposition_key"],
                "pro_date": str(pro_row["date"]),
                "opposition_date": str(opposition_row["date"]),
                "date_gap_days": candidate["date_gap_days"],
                "shared_token_count": candidate["shared_token_count"],
                "token_jaccard": candidate["token_jaccard"],
                "shared_tokens": candidate["shared_tokens"],
                "pro_title": str(pro_row["title"]),
                "opposition_title": str(opposition_row["title"]),
                "pro_government_sentiment": float(pro_row["sentiment"]),
                "opposition_sentiment": float(opposition_row["sentiment"]),
                "gap": float(pro_row["sentiment"] - opposition_row["sentiment"]),
            }
        )
    return pd.DataFrame(selected)


def _coverage_summary(
    frame: pd.DataFrame,
    *,
    match_method: str,
    matched_pro_rows: int,
    matched_opposition_rows: int,
    event_column: str | None,
    shared_event_count: int,
) -> dict[str, Any]:
    counts = (
        frame.groupby(["camp", "category"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = frame.groupby("camp").size()
    counts["within_camp_share"] = [
        count / totals[camp] for camp, count in zip(counts["camp"], counts["count"])
    ]
    pivot = counts.pivot(index="category", columns="camp", values="within_camp_share").fillna(0.0)
    for camp in ("pro_government", "opposition"):
        if camp not in pivot:
            pivot[camp] = 0.0
    total_variation = 0.5 * float(
        (pivot["pro_government"] - pivot["opposition"]).abs().sum()
    )

    camp_rows = []
    matched = {
        "pro_government": matched_pro_rows,
        "opposition": matched_opposition_rows,
    }
    for camp in ("pro_government", "opposition"):
        total = int(totals.get(camp, 0))
        matched_count = int(matched[camp])
        camp_rows.append(
            {
                "camp": camp,
                "headline_count": total,
                "matched_headline_count": matched_count,
                "unmatched_headline_count": total - matched_count,
                "matched_share": matched_count / total if total else None,
            }
        )

    if event_column is None:
        event_coverage: dict[str, Any] = {
            "status": "unavailable",
            "canonical_event_column": None,
            "pro_government_event_count": None,
            "opposition_event_count": None,
            "shared_event_count": 0,
            "pro_government_only_event_count": None,
            "opposition_only_event_count": None,
            "diagnostic": (
                "No explicit canonical/shared event identifier is available; bridge IDs "
                "that are unique per headline are deliberately not event-coverage evidence."
            ),
        }
    else:
        pro_events = set(
            frame.loc[
                (frame["camp"] == "pro_government") & frame[event_column].notna(),
                event_column,
            ].astype(str)
        )
        opposition_events = set(
            frame.loc[
                (frame["camp"] == "opposition") & frame[event_column].notna(),
                event_column,
            ].astype(str)
        )
        event_coverage = {
            "status": "ok",
            "canonical_event_column": event_column,
            "pro_government_event_count": len(pro_events),
            "opposition_event_count": len(opposition_events),
            "shared_event_count": len(pro_events & opposition_events),
            "pro_government_only_event_count": len(pro_events - opposition_events),
            "opposition_only_event_count": len(opposition_events - pro_events),
            "missing_event_id_headline_count": int(frame[event_column].isna().sum()),
            "diagnostic": (
                "Counts describe observed canonical-event coverage; they do not measure "
                "events that no configured source reported."
            ),
        }

    return {
        "estimand": "descriptive differences in topic and story coverage",
        "category_coverage": _records(counts),
        "category_share_total_variation": total_variation,
        "story_coverage_by_camp": camp_rows,
        "event_coverage": event_coverage,
        "matching_method": match_method,
        "canonical_event_column": event_column,
        "verified_shared_event_count": int(shared_event_count),
        "interpretation": (
            "Coverage differences are selection descriptives; they do not establish "
            "intent, political bias, or any causal mechanism."
        ),
    }


def _framing_summary(
    frame: pd.DataFrame,
    *,
    canonical_event_column: str | None,
    minimum_shared_tokens: int,
    window_days: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    event_gaps = (
        _shared_event_gaps(frame, canonical_event_column)
        if canonical_event_column is not None
        else pd.DataFrame()
    )
    if not event_gaps.empty:
        gaps = event_gaps["gap"].astype(float)
        matched_pro = int(event_gaps["pro_count"].sum())
        matched_opposition = int(event_gaps["opposition_count"].sum())
        matching = {
            "method": "canonical_event_id",
            "canonical_event_column": canonical_event_column,
            "verified_shared_events": int(len(event_gaps)),
            "fallback_used": False,
            "details": _records(event_gaps),
        }
        framing = {
            "status": "ok" if len(gaps) >= 2 else "descriptive_only",
            "estimand": "mean within-event pro_government minus opposition sentiment gap",
            "unit": "explicit repeated canonical event",
            "event_or_pair_count": int(len(gaps)),
            "mean_gap": _finite_or_none(gaps.mean()),
            "median_gap": _finite_or_none(gaps.median()),
            "standard_deviation": _finite_or_none(gaps.std(ddof=1)),
            "share_positive_gap": float((gaps > 0).mean()),
            "caveat": (
                "Within-event association is observational. Holding event ID fixed does "
                "not identify a causal political-bias effect."
            ),
        }
        return framing, {
            "matching": matching,
            "matched_pro_rows": matched_pro,
            "matched_opposition_rows": matched_opposition,
            "shared_event_count": int(len(event_gaps)),
        }

    lexical = lexical_date_pairs(
        frame,
        minimum_shared_tokens=minimum_shared_tokens,
        window_days=window_days,
    )
    if lexical.empty:
        gaps = pd.Series(dtype=float)
    else:
        gaps = lexical["gap"].astype(float)
    matching = {
        "method": "lexical_date_fallback",
        "canonical_event_column": canonical_event_column,
        "verified_shared_events": 0,
        "fallback_used": True,
        "one_to_one_no_reuse": True,
        "minimum_shared_tokens": int(minimum_shared_tokens),
        "window_days": int(window_days),
        "details": _records(lexical),
        "diagnostic": (
            "No repeated cross-camp canonical event IDs were available. These pairs "
            "are deterministic lexical/date candidates, not verified same events."
        ),
    }
    framing = {
        "status": "sensitivity_only" if len(gaps) >= 2 else "insufficient_matches",
        "estimand": "mean within-inferred-pair pro_government minus opposition sentiment gap",
        "unit": "unverified one-to-one lexical/date pair",
        "event_or_pair_count": int(len(gaps)),
        "mean_gap": _finite_or_none(gaps.mean()) if len(gaps) else None,
        "median_gap": _finite_or_none(gaps.median()) if len(gaps) else None,
        "standard_deviation": _finite_or_none(gaps.std(ddof=1)) if len(gaps) else None,
        "share_positive_gap": float((gaps > 0).mean()) if len(gaps) else None,
        "caveat": (
            "Lexical/date matches are a measurement sensitivity, not proof of common "
            "events or a causal political-bias effect; inspect pair quality."
        ),
    }
    return framing, {
        "matching": matching,
        "matched_pro_rows": int(len(lexical)),
        "matched_opposition_rows": int(len(lexical)),
        "shared_event_count": 0,
    }


def _cluster_result(
    fitted: Any,
    groups: pd.Series,
    *,
    coefficient_index: int,
    cluster_name: str,
) -> dict[str, Any]:
    cluster_count = int(groups.nunique(dropna=True))
    result: dict[str, Any] = {
        "cluster": cluster_name,
        "cluster_count": cluster_count,
        "status": "skipped",
        "coefficient": None,
        "standard_error": None,
        "p_value": None,
        "confidence_interval_95": [None, None],
        "diagnostic": None,
    }
    if groups.isna().any():
        result["diagnostic"] = "cluster identifiers contain missing values"
        return result
    if cluster_count < 2:
        result["diagnostic"] = "cluster-robust covariance needs at least two clusters"
        return result
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            robust = fitted.get_robustcov_results(
                cov_type="cluster",
                groups=groups.to_numpy(),
                use_correction=True,
            )
            interval = np.asarray(robust.conf_int(alpha=0.05))[coefficient_index]
            coefficient = _finite_or_none(robust.params[coefficient_index])
            standard_error = _finite_or_none(robust.bse[coefficient_index])
            p_value = _finite_or_none(robust.pvalues[coefficient_index])
        numerical_warnings = sorted(
            {str(item.message) for item in caught if issubclass(item.category, RuntimeWarning)}
        )
        diagnostics = []
        if cluster_count < 30:
            diagnostics.append("fewer than 30 clusters; small-cluster inference may be unstable")
        if numerical_warnings:
            diagnostics.append("statsmodels numerical warning: " + "; ".join(numerical_warnings))
        if standard_error is None or p_value is None:
            result.update(
                {
                    "status": "failed",
                    "coefficient": coefficient,
                    "diagnostic": "; ".join(diagnostics)
                    or "cluster covariance returned non-finite inference",
                }
            )
            return result
        result.update(
            {
                "status": "ok" if cluster_count >= 30 else "ok_with_few_clusters",
                "coefficient": coefficient,
                "standard_error": standard_error,
                "p_value": p_value,
                "confidence_interval_95": [
                    _finite_or_none(interval[0]),
                    _finite_or_none(interval[1]),
                ],
                "diagnostic": "; ".join(diagnostics) if diagnostics else None,
            }
        )
    except Exception as exc:  # statsmodels errors vary by version/design
        result["diagnostic"] = f"cluster covariance failed: {type(exc).__name__}: {exc}"
    return result


def _skipped_cluster_result(
    cluster_name: str,
    groups: pd.Series | None,
    diagnostic: str,
) -> dict[str, Any]:
    return {
        "cluster": cluster_name,
        "cluster_count": int(groups.nunique(dropna=True)) if groups is not None else 0,
        "status": "skipped",
        "coefficient": None,
        "standard_error": None,
        "p_value": None,
        "confidence_interval_95": [None, None],
        "diagnostic": diagnostic,
    }


def _regression(
    frame: pd.DataFrame,
    *,
    canonical_event_column: str | None,
) -> dict[str, Any]:
    initial_sensitivities = {
        "outlet": _skipped_cluster_result(
            "outlet",
            frame["source"] if "source" in frame else None,
            "cluster covariance was not run because the base regression was unavailable or inadequate",
        ),
        "date": _skipped_cluster_result(
            "date",
            frame["date"] if "date" in frame else None,
            "cluster covariance was not run because the base regression was unavailable or inadequate",
        ),
        "event": _skipped_cluster_result(
            "event",
            frame[canonical_event_column]
            if canonical_event_column is not None and canonical_event_column in frame
            else None,
            (
                "no explicit canonical/shared event identifier; the 1:1 headline-event "
                "bridge is not a shared-story cluster"
                if canonical_event_column is None
                else "cluster covariance was not run because the base regression was unavailable or inadequate"
            ),
        ),
    }
    result: dict[str, Any] = {
        "formula": "sentiment ~ camp_indicator + C(category) + C(date)",
        "camp_indicator": "1=pro_government, 0=opposition",
        "status": "unavailable",
        "observations": int(len(frame)),
        "coefficient": None,
        "standard_error_conventional": None,
        "p_value_conventional": None,
        "confidence_interval_95_conventional": [None, None],
        "design_rank": None,
        "design_columns": None,
        "residual_degrees_of_freedom": None,
        "diagnostics": [],
        "cluster_robust_sensitivities": initial_sensitivities,
        "interpretation": (
            "The camp coefficient is an adjusted observational association, not a "
            "causal estimate of political bias."
        ),
    }
    if frame.empty or frame["camp"].nunique() < 2:
        result["diagnostics"].append("regression needs observations from both camps")
        return result
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        result["diagnostics"].append(
            "statsmodels is not installed; install requirements.txt to run regression inference"
        )
        return result

    design = frame.copy()
    design["camp_indicator"] = (design["camp"] == "pro_government").astype(int)
    try:
        fitted = smf.ols(
            "sentiment ~ camp_indicator + C(category) + C(date)",
            data=design,
        ).fit()
    except Exception as exc:
        result["status"] = "failed"
        result["diagnostics"].append(f"regression failed: {type(exc).__name__}: {exc}")
        return result

    names = list(fitted.model.exog_names)
    coefficient_index = names.index("camp_indicator")
    rank = int(np.linalg.matrix_rank(fitted.model.exog))
    columns = int(fitted.model.exog.shape[1])
    residual_df = float(fitted.df_resid)
    result.update(
        {
            "status": "fit_complete",
            "coefficient": _finite_or_none(fitted.params["camp_indicator"]),
            "design_rank": rank,
            "design_columns": columns,
            "residual_degrees_of_freedom": residual_df,
        }
    )
    if rank < columns:
        result["status"] = "rank_deficient"
        result["diagnostics"].append(
            f"design matrix rank {rank} is below {columns} columns; robust inference skipped"
        )
        return result
    if residual_df <= 0:
        result["status"] = "inadequate_residual_degrees_of_freedom"
        result["diagnostics"].append(
            "no positive residual degrees of freedom; robust inference skipped"
        )
        return result

    interval = fitted.conf_int(alpha=0.05).loc["camp_indicator"]
    result.update(
        {
            "status": "ok",
            "standard_error_conventional": _finite_or_none(fitted.bse["camp_indicator"]),
            "p_value_conventional": _finite_or_none(fitted.pvalues["camp_indicator"]),
            "confidence_interval_95_conventional": [
                _finite_or_none(interval.iloc[0]),
                _finite_or_none(interval.iloc[1]),
            ],
        }
    )

    sensitivities = {
        "outlet": _cluster_result(
            fitted,
            design["source"],
            coefficient_index=coefficient_index,
            cluster_name="outlet",
        ),
        "date": _cluster_result(
            fitted,
            design["date"],
            coefficient_index=coefficient_index,
            cluster_name="date",
        ),
    }
    if canonical_event_column is None:
        sensitivities["event"] = _skipped_cluster_result(
            "event",
            None,
            (
                "no explicit canonical/shared event identifier; the 1:1 headline-event "
                "bridge is not a shared-story cluster"
            ),
        )
    else:
        shared = _shared_event_gaps(design, canonical_event_column)
        event_ids = design[canonical_event_column]
        if len(shared) < 2 or event_ids.isna().any():
            sensitivities["event"] = _skipped_cluster_result(
                "event",
                event_ids,
                (
                    "event clustering requires non-missing canonical IDs and at least "
                    "two repeated cross-camp events"
                ),
            )
        else:
            sensitivities["event"] = _cluster_result(
                fitted,
                event_ids.astype(str),
                coefficient_index=coefficient_index,
                cluster_name="event",
            )
    result["cluster_robust_sensitivities"] = sensitivities
    return result


def analyze_polarization(
    headlines: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    pro_government_sources: Sequence[str] = DEFAULT_PRO_GOVERNMENT_SOURCES,
    opposition_sources: Sequence[str] = DEFAULT_OPPOSITION_SOURCES,
    canonical_event_column: str | None = None,
    bootstrap_repetitions: int = 2_000,
    bootstrap_seed: int = 20260707,
    minimum_shared_tokens: int = 2,
    match_window_days: int = 1,
) -> dict[str, Any]:
    """Return a JSON-serializable, observational polarization report."""

    frame, input_diagnostics = _prepare_headlines(
        headlines,
        pro_government_sources=pro_government_sources,
        opposition_sources=opposition_sources,
    )
    event_column = _canonical_event_column(frame, canonical_event_column)
    framing, matching_info = _framing_summary(
        frame,
        canonical_event_column=event_column,
        minimum_shared_tokens=minimum_shared_tokens,
        window_days=match_window_days,
    )
    selection = _coverage_summary(
        frame,
        match_method=matching_info["matching"]["method"],
        matched_pro_rows=matching_info["matched_pro_rows"],
        matched_opposition_rows=matching_info["matched_opposition_rows"],
        event_column=event_column,
        shared_event_count=matching_info["shared_event_count"],
    )
    report = {
        "analysis_type": "observational_media_polarization",
        "causal_claim": False,
        "camp_definition": {
            "pro_government_sources": list(pro_government_sources),
            "opposition_sources": list(opposition_sources),
            "diagnostic": (
                "Camp assignments are researcher-specified groupings and should be "
                "subjected to alternative-definition sensitivity checks."
            ),
        },
        "input_diagnostics": input_diagnostics,
        "raw_descriptives": {
            "by_camp": _group_summary(frame, "camp"),
            "by_outlet": _group_summary(frame, "source"),
        },
        "mean_difference": _mean_difference(frame),
        "date_cluster_bootstrap": date_cluster_bootstrap(
            frame,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        ),
        "regression": _regression(frame, canonical_event_column=event_column),
        "selection": selection,
        "framing": framing,
        "matching_audit": matching_info["matching"],
        "limitations": [
            "All estimates are observational associations and do not identify causal political bias.",
            "Sentiment model measurement error is not corrected by these standard errors.",
            "Outlet camp assignments, source coverage, dates, and categories are researcher choices.",
            (
                "The current database event bridge is one event per headline; only explicit repeated "
                "canonical/shared identifiers qualify as verified shared events."
            ),
        ],
    }
    return report


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(connection, table):
        return set()
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def load_headlines(db_path: str | Path) -> pd.DataFrame:
    """Load eligible source observations without mutating the database.

    Stage 2 preserves source-distinct fetch observations even when a shared URL
    collapses to one canonical headline. Polarization is outlet-level analysis,
    so those source observations are the correct rows. Legacy canonical rows
    with no linked raw observation remain available through an explicit union.
    """

    connection = sqlite3.connect(str(db_path))
    try:
        headline_columns = _columns(connection, "headlines")
        if not headline_columns:
            raise ValueError("database has no headlines table")
        required = {"id", "source", "title", "published_at", "sentiment_score"}
        missing = sorted(required - headline_columns)
        if missing:
            raise ValueError(f"headlines table missing required columns: {', '.join(missing)}")

        headline_timestamp = (
            "COALESCE(h.published_timestamp, h.published_at)"
            if "published_timestamp" in headline_columns
            else "h.published_at"
        )
        category = "h.category" if "category" in headline_columns else "NULL"

        event_columns = _columns(connection, "events")
        event_join = ""
        event_select: list[str] = []
        if event_columns and {"event_id", "headline_id"}.issubset(event_columns):
            event_join = " LEFT JOIN events e ON e.headline_id = h.id"
            event_select.append("e.event_id AS bridge_event_id")
            if "canonical_event_id" in event_columns:
                event_select.append("e.canonical_event_id AS canonical_event_id")

        where = ["h.sentiment_score IS NOT NULL", "h.published_at IS NOT NULL"]
        if "processing_status" in headline_columns:
            where.append("h.processing_status = 'scored'")
        exclusion_columns = _columns(connection, "headline_exclusions")
        if "headline_id" in exclusion_columns:
            active = "x.restored_at IS NULL" if "restored_at" in exclusion_columns else "1=1"
            where.append(
                "NOT EXISTS (SELECT 1 FROM headline_exclusions x "
                f"WHERE x.headline_id=h.id AND {active})"
            )
        where_sql = " AND ".join(where)

        def canonical_select(*, only_without_raw: bool) -> str:
            select = [
                "NULL AS raw_observation_id",
                "h.id AS headline_id",
                "h.source AS source",
                "h.title AS title",
                f"{headline_timestamp} AS date",
                f"{category} AS category",
                "h.sentiment_score AS sentiment",
                *event_select,
            ]
            fallback = (
                " AND NOT EXISTS (SELECT 1 FROM raw_headline_observations r0 "
                "WHERE r0.headline_id=h.id)"
                if only_without_raw
                else ""
            )
            return (
                "SELECT " + ", ".join(select) + " FROM headlines h"
                + event_join + " WHERE " + where_sql + fallback
            )

        raw_columns = _columns(connection, "raw_headline_observations")
        raw_ready = {"observation_id", "headline_id", "source", "title"}.issubset(
            raw_columns
        )
        if raw_ready:
            raw_dates = []
            if "published_timestamp" in raw_columns:
                raw_dates.append("r.published_timestamp")
            if "published_at" in raw_columns:
                raw_dates.append("r.published_at")
            raw_dates.append(headline_timestamp)
            raw_timestamp = "COALESCE(" + ", ".join(raw_dates) + ")"
            raw_select = [
                "r.observation_id AS raw_observation_id",
                "h.id AS headline_id",
                "r.source AS source",
                "r.title AS title",
                f"{raw_timestamp} AS date",
                f"{category} AS category",
                "h.sentiment_score AS sentiment",
                *event_select,
            ]
            raw_query = (
                "SELECT " + ", ".join(raw_select)
                + " FROM raw_headline_observations r"
                + " JOIN headlines h ON h.id=r.headline_id"
                + event_join + " WHERE " + where_sql
            )
            query = (
                raw_query + " UNION ALL " + canonical_select(only_without_raw=True)
                + " ORDER BY date, headline_id, raw_observation_id"
            )
        else:
            query = canonical_select(only_without_raw=False) + " ORDER BY date, headline_id"
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()


def format_report(report: Mapping[str, Any]) -> str:
    """Render the core audit concisely; JSON retains every diagnostic/detail."""

    difference = report["mean_difference"]
    bootstrap = report["date_cluster_bootstrap"]
    regression = report["regression"]
    framing = report["framing"]
    selection = report["selection"]
    lines = [
        "POLARIZATION INFERENCE (OBSERVATIONAL; NO CAUSAL POLITICAL-BIAS CLAIM)",
        "",
        "Raw camp descriptives:",
    ]
    for row in report["raw_descriptives"]["by_camp"]:
        lines.append(f"  {row['camp']}: n={row['count']}, mean={row['mean']}")
    lines.extend(
        [
            "",
            (
                "Camp mean difference (pro-government - opposition): "
                f"{difference['estimate']} (Cohen d={difference['standardized_effect_size']})"
            ),
            (
                f"Date-cluster bootstrap: {bootstrap['status']}, dates={bootstrap['cluster_count']}, "
                f"95% interval=[{bootstrap['lower']}, {bootstrap['upper']}], seed={bootstrap['seed']}"
            ),
            (
                f"Adjusted regression: {regression['status']}, camp coefficient="
                f"{regression['coefficient']}"
            ),
            "",
            (
                f"Selection: category-share total variation="
                f"{selection['category_share_total_variation']}; method={selection['matching_method']}"
            ),
            (
                f"Framing: {framing['status']}, unit={framing['unit']}, "
                f"n={framing['event_or_pair_count']}, mean gap={framing['mean_gap']}"
            ),
            f"Framing caveat: {framing['caveat']}",
            "",
            "Full machine-readable diagnostics:",
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        ]
    )
    return "\n".join(lines)


def _console_safe(text: str, encoding: str | None = None) -> str:
    """Preserve UTF-8 output and escape only unsupported console characters."""
    active_encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(active_encoding, errors="backslashreplace").decode(active_encoding)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Observational selection-versus-framing polarization analysis"
    )
    parser.add_argument("--db", default="finance_sentiment.db", help="SQLite database path")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--minimum-shared-tokens", type=int, default=2)
    parser.add_argument("--match-window-days", type=int, default=1)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for the full JSON report (no file is written by default)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_headlines(args.db)
    report = analyze_polarization(
        frame,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.seed,
        minimum_shared_tokens=args.minimum_shared_tokens,
        match_window_days=args.match_window_days,
    )
    print(_console_safe(format_report(report)))
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
