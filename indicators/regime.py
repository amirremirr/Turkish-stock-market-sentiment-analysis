"""Deterministic news-regime and taxonomy-coverage reports.

Read-only: every number here is looked up from the stored indicator tables, so
the report describes what the pipeline computed rather than recomputing it a
second, possibly different way.

The output separates four things that are easy to conflate and mean different
things:

``level``       where tone sits now
``change``      how it moved over 5 and 20 sessions
``abnormal``    where it sits against its own prior history
``attention``   how much coverage there is, and from how many outlets

A high level with an unremarkable abnormal position is a calm market being
reported positively. A moderate level at the 99th percentile of its own history
is something else. Reporting them separately is the point.

No causal language is generated. The report says which outlets and headlines
carry the largest weight in a number; it never says a family moved *because* of
them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REGIME_REPORT_VERSION = "news-regime-v1"

# The domestic-only aggregate is stored in daily_family_signals under this key so
# it lives beside the families without being one of them.
DOMESTIC_COMPOSITE_KEY = "__domestic__"
# Recap tone restates the index move rather than describing an economic topic,
# so it is ranked separately from the economic families.
MARKET_RECAP_KEY = "market_recap"

_UNUSUAL_HIGH = 0.90
_UNUSUAL_LOW = 0.10
_ELEVATED_VOLUME_Z = 1.5


def _optional(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _latest_sessions(frame: pd.DataFrame, count: int) -> List[str]:
    if frame.empty or "signal_date" not in frame:
        return []
    return sorted(frame["signal_date"].dropna().unique())[-count:]


def _recap_share(current: pd.DataFrame) -> Optional[float]:
    """Share of the session's headlines that were market recap.

    Computed across every family rather than from the recap family's own row,
    because a recap headline is counted in whichever family it belongs to.
    """

    if current.empty or "market_recap_count" not in current:
        return None
    rows = current[current["signal_family"] != DOMESTIC_COMPOSITE_KEY]
    total = int(rows["headline_count"].sum()) if not rows.empty else 0
    if not total:
        return None
    return float(rows["market_recap_count"].sum()) / total


def build_regime_report(
    family_signals: pd.DataFrame,
    abnormal: pd.DataFrame,
    disagreement: pd.DataFrame,
    volume: pd.DataFrame,
    drivers: pd.DataFrame,
    *,
    as_of: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the regime summary from stored indicator tables."""

    if family_signals.empty:
        return {
            "version": REGIME_REPORT_VERSION,
            "as_of": None,
            "status": "no_data",
            "families": [],
            "notes": ["no family signals are stored yet"],
        }

    sessions = sorted(family_signals["signal_date"].dropna().unique())
    as_of = as_of or sessions[-1]
    current = family_signals[family_signals["signal_date"] == as_of]

    def _session_offset(offset: int) -> Optional[str]:
        if as_of not in sessions:
            return None
        index = sessions.index(as_of) - offset
        return sessions[index] if index >= 0 else None

    five_back, twenty_back = _session_offset(5), _session_offset(20)

    families: List[Dict[str, Any]] = []
    for _, row in current.sort_values("signal_family").iterrows():
        family = row["signal_family"]

        def _mean_at(date: Optional[str]) -> Optional[float]:
            if date is None:
                return None
            match = family_signals[
                (family_signals["signal_date"] == date)
                & (family_signals["signal_family"] == family)
            ]
            return _optional(match["simple_mean"].iloc[0]) if not match.empty else None

        level = _optional(row["simple_mean"])
        prior_5, prior_20 = _mean_at(five_back), _mean_at(twenty_back)

        abn = abnormal[
            (abnormal["signal_date"] == as_of)
            & (abnormal["scope"] == "family")
            & (abnormal["scope_key"] == family)
        ]
        dis = disagreement[
            (disagreement["signal_date"] == as_of)
            & (disagreement["signal_family"] == family)
        ]
        vol = volume[
            (volume["signal_date"] == as_of) & (volume["signal_family"] == family)
        ]

        percentile = _optional(abn["rolling_percentile"].iloc[0]) if not abn.empty else None
        volume_z = _optional(vol["volume_z"].iloc[0]) if not vol.empty else None

        families.append({
            "signal_family": family,
            "sample_sufficiency": row["sample_sufficiency"],
            "level": {
                "simple_mean": level,
                "relevance_weighted": _optional(row["relevance_weighted"]),
                "median": _optional(row["median_sentiment"]),
                "headline_count": int(row["headline_count"]),
                "source_count": int(row["source_count"]),
            },
            "change": {
                "vs_5_sessions": (
                    level - prior_5 if level is not None and prior_5 is not None else None
                ),
                "vs_20_sessions": (
                    level - prior_20 if level is not None and prior_20 is not None else None
                ),
                "reference_dates": {"five": five_back, "twenty": twenty_back},
            },
            "abnormal": {
                "abnormal_tone": _optional(abn["abnormal_tone"].iloc[0]) if not abn.empty else None,
                "rolling_z": _optional(abn["rolling_z"].iloc[0]) if not abn.empty else None,
                "rolling_percentile": percentile,
                "prior_count": int(abn["prior_count"].iloc[0]) if not abn.empty else 0,
                "is_unusual": (
                    percentile is not None
                    and (percentile >= _UNUSUAL_HIGH or percentile <= _UNUSUAL_LOW)
                ),
            },
            "disagreement": {
                "within_day_std": _optional(dis["within_day_std"].iloc[0]) if not dis.empty else None,
                "cross_outlet_std": _optional(dis["cross_outlet_std"].iloc[0]) if not dis.empty else None,
                "max_minus_min": _optional(dis["max_minus_min"].iloc[0]) if not dis.empty else None,
                "entropy": _optional(dis["sentiment_entropy"].iloc[0]) if not dis.empty else None,
                "camp_gap": _optional(dis["camp_gap"].iloc[0]) if not dis.empty else None,
                "min_sources_met": bool(dis["min_sources_met"].iloc[0]) if not dis.empty else False,
            },
            "attention": {
                "headline_count": int(vol["headline_count"].iloc[0]) if not vol.empty else 0,
                "observation_count": int(vol["observation_count"].iloc[0]) if not vol.empty else 0,
                "source_breadth": int(vol["source_breadth"].iloc[0]) if not vol.empty else 0,
                "volume_z": volume_z,
                "volume_percentile": _optional(vol["volume_percentile"].iloc[0]) if not vol.empty else None,
                "is_elevated": volume_z is not None and volume_z >= _ELEVATED_VOLUME_Z,
            },
            "quality": {
                "market_recap_count": int(row["market_recap_count"]),
                "market_recap_share": (
                    float(row["market_recap_count"]) / int(row["headline_count"])
                    if int(row["headline_count"]) else None
                ),
                "unknown_timing_count": int(row["unknown_timing_count"]),
                "timing_degraded": int(row["unknown_timing_count"]) > 0,
                "ambiguous_count": int(row["ambiguous_count"]),
            },
        })

    # Two families are excluded from the economic ranking, for different reasons.
    #
    # The domestic composite is an aggregate over families, so ranking it
    # against its own members would let it win by construction.
    #
    # market_recap is not an economic topic. Its tone restates the day's index
    # move ("Borsa yükselişle kapandı"), so on any strong session it wins or
    # loses the ranking mechanically and pushes out whatever news actually
    # moved. Its share is reported separately instead.
    ranked = [f for f in families
              if f["signal_family"] not in (DOMESTIC_COMPOSITE_KEY, MARKET_RECAP_KEY)
              and f["level"]["simple_mean"] is not None
              and f["sample_sufficiency"] == "sufficient"]
    by_level = sorted(ranked, key=lambda f: f["level"]["simple_mean"])
    moved = [f for f in ranked if f["change"]["vs_5_sessions"] is not None]
    by_move = sorted(moved, key=lambda f: abs(f["change"]["vs_5_sessions"]), reverse=True)

    driver_rows = drivers[drivers["signal_date"] == as_of] if not drivers.empty else pd.DataFrame()
    top_positive, top_negative = [], []
    if not driver_rows.empty:
        ordered = driver_rows.sort_values("sentiment_score", ascending=False)
        top_positive = _driver_records(ordered.head(5))
        top_negative = _driver_records(ordered.tail(5).iloc[::-1])

    return {
        "version": REGIME_REPORT_VERSION,
        "as_of": as_of,
        "status": "ok",
        "sessions_available": len(sessions),
        "families": families,
        "most_positive": by_level[-1]["signal_family"] if by_level else None,
        "most_negative": by_level[0]["signal_family"] if by_level else None,
        "largest_5_session_move": by_move[0]["signal_family"] if by_move else None,
        "unusual_percentiles": [
            f["signal_family"] for f in families if f["abnormal"]["is_unusual"]
        ],
        "elevated_disagreement": [
            f["signal_family"] for f in families
            if f["disagreement"]["cross_outlet_std"] is not None
            and f["disagreement"]["cross_outlet_std"] >= 0.30
        ],
        "elevated_volume": [
            f["signal_family"] for f in families if f["attention"]["is_elevated"]
        ],
        "insufficient_samples": [
            f["signal_family"] for f in families
            if f["sample_sufficiency"] != "sufficient"
        ],
        "domestic_only": next(
            (f for f in families if f["signal_family"] == DOMESTIC_COMPOSITE_KEY),
            None,
        ),
        # Reported beside the ranking rather than inside it, so a reader can see
        # how much of the day was the market talking about itself.
        "market_recap": next(
            (f for f in families if f["signal_family"] == MARKET_RECAP_KEY), None,
        ),
        "market_recap_share": _recap_share(current),
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "notes": [
            "Descriptive only. No causal claim is made and no result here is a "
            "validated predictive signal.",
            "Disagreement measures variation among observed news sources; it is "
            "not a measure of market uncertainty.",
        ],
    }


def _driver_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    fields = [
        "title", "source", "published_timestamp", "timing_bucket", "category",
        "signal_family", "sentiment_score", "relevance", "is_market_recap",
        "experiment_id",
    ]
    records = []
    for _, row in frame.iterrows():
        record = {}
        for field in fields:
            value = row.get(field)
            record[field] = None if value is None or pd.isna(value) else value
        records.append(record)
    return records


def build_coverage_report(headlines: pd.DataFrame) -> Dict[str, Any]:
    """Taxonomy coverage: what the rules assigned, and what they could not."""

    if headlines.empty:
        return {"version": REGIME_REPORT_VERSION, "total": 0, "families": []}

    total = len(headlines)
    by_family = headlines.groupby("signal_family").size().to_dict()
    by_category = headlines.groupby("category").size().to_dict()
    cross = (
        headlines.groupby(["category", "signal_family"]).size()
        .reset_index(name="n").to_dict("records")
    )
    rules = headlines.groupby("signal_family_rule").size().to_dict()
    ambiguous = headlines[headlines["signal_family_ambiguous"] == 1]
    recap = headlines[headlines["is_market_recap"] == 1]

    timing = (
        headlines.groupby(["signal_family", "timing_bucket"]).size()
        .reset_index(name="n").to_dict("records")
    )
    experiments = (
        headlines.groupby(["signal_family", "experiment_id"]).size()
        .reset_index(name="n").to_dict("records")
    )
    sources = (
        headlines.groupby(["signal_family"])["source"].nunique().to_dict()
    )

    examples: Dict[str, List[str]] = defaultdict(list)
    for family, group in headlines.groupby("signal_family"):
        examples[family] = group["title"].head(3).tolist()

    ambiguous_reasons = (
        ambiguous.groupby("signal_family_review").size().to_dict()
        if not ambiguous.empty else {}
    )

    return {
        "version": REGIME_REPORT_VERSION,
        "total": total,
        "by_signal_family": {
            family: {"count": int(count), "share": count / total}
            for family, count in sorted(by_family.items())
        },
        "by_detailed_category": {
            str(category): int(count) for category, count in sorted(by_category.items())
        },
        "category_to_family": cross,
        "assignment_rules": {str(rule): int(count) for rule, count in sorted(rules.items())},
        "ambiguous_count": int(len(ambiguous)),
        "ambiguous_share": len(ambiguous) / total,
        "ambiguous_reasons": {str(k): int(v) for k, v in ambiguous_reasons.items()},
        "ambiguous_examples": ambiguous["title"].head(10).tolist() if not ambiguous.empty else [],
        "other_family_share": by_family.get("other", 0) / total,
        "market_recap_count": int(len(recap)),
        "market_recap_share": len(recap) / total,
        "market_recap_examples": recap["title"].head(10).tolist() if not recap.empty else [],
        "sources_per_family": {str(k): int(v) for k, v in sorted(sources.items())},
        "family_by_timing": timing,
        "family_by_experiment": experiments,
        "examples_per_family": dict(examples),
    }


def write_reports(
    regime: Dict[str, Any],
    coverage: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    """Write JSON and CSV artifacts deterministically."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    regime_json = output_dir / "news_regime.json"
    regime_json.write_text(
        json.dumps(regime, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    written["regime_json"] = str(regime_json)

    coverage_json = output_dir / "taxonomy_coverage.json"
    coverage_json.write_text(
        json.dumps(coverage, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    written["coverage_json"] = str(coverage_json)

    if regime.get("families"):
        rows = []
        for family in regime["families"]:
            rows.append({
                "signal_family": family["signal_family"],
                "sample_sufficiency": family["sample_sufficiency"],
                "level_simple_mean": family["level"]["simple_mean"],
                "headline_count": family["level"]["headline_count"],
                "source_count": family["level"]["source_count"],
                "change_5": family["change"]["vs_5_sessions"],
                "change_20": family["change"]["vs_20_sessions"],
                "abnormal_tone": family["abnormal"]["abnormal_tone"],
                "rolling_percentile": family["abnormal"]["rolling_percentile"],
                "cross_outlet_std": family["disagreement"]["cross_outlet_std"],
                "volume_z": family["attention"]["volume_z"],
                "source_breadth": family["attention"]["source_breadth"],
                "market_recap_share": family["quality"]["market_recap_share"],
                "unknown_timing_count": family["quality"]["unknown_timing_count"],
            })
        regime_csv = output_dir / "news_regime.csv"
        pd.DataFrame(rows).to_csv(regime_csv, index=False)
        written["regime_csv"] = str(regime_csv)

    if coverage.get("by_signal_family"):
        coverage_csv = output_dir / "taxonomy_coverage.csv"
        pd.DataFrame([
            {"signal_family": family, "count": data["count"], "share": data["share"]}
            for family, data in coverage["by_signal_family"].items()
        ]).to_csv(coverage_csv, index=False)
        written["coverage_csv"] = str(coverage_csv)

    return written
