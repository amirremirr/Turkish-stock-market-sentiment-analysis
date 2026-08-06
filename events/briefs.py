"""Machine- and human-readable briefs for candidate events.

A brief answers: what appears to have happened, when it became public, when it
could first have been acted on, who reported it, how they differed, and what is
missing. It deliberately stops there. No trading recommendation is produced, and
the language never upgrades an algorithmic grouping into a verified event.

Data-quality warnings are part of the brief rather than a footnote, because the
single most useful thing a reader can know about a small event group is that it
came from one outlet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

BRIEF_VERSION = "event-brief-v1"

WARN_SINGLE_SOURCE = "single_source_group"
WARN_SINGLETON = "single_headline_group"
WARN_UNKNOWN_TIME = "unknown_publication_time"
WARN_RECAP_ONLY = "market_recap_only"
WARN_NO_ENTITY = "no_recognised_entity"
WARN_SPANS_SESSIONS = "spans_multiple_signal_dates"
WARN_UNREVIEWED = "grouping_not_human_reviewed"


def _quality_warnings(event: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if int(event.get("is_single_source") or 0):
        warnings.append(WARN_SINGLE_SOURCE)
    if int(event.get("is_singleton") or 0):
        warnings.append(WARN_SINGLETON)
    if int(event.get("unknown_timestamp_count") or 0):
        warnings.append(WARN_UNKNOWN_TIME)
    if (
        int(event.get("market_recap_count") or 0)
        and int(event.get("market_recap_count") or 0) == int(event.get("headline_count") or 0)
    ):
        warnings.append(WARN_RECAP_ONLY)
    if not event.get("primary_entity"):
        warnings.append(WARN_NO_ENTITY)
    if int(event.get("signal_date_span") or 1) > 1:
        warnings.append(WARN_SPANS_SESSIONS)
    if str(event.get("review_state", "unreviewed")) == "unreviewed":
        warnings.append(WARN_UNREVIEWED)
    return warnings


def build_event_brief(
    event: Dict[str, Any],
    headlines: Sequence[Dict[str, Any]],
    windows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble one machine-readable brief."""

    examples = sorted(
        headlines,
        key=lambda row: abs(float(row.get("sentiment_score") or 0.0)),
        reverse=True,
    )[:5]

    evaluable = [
        {
            "window_name": window["window_name"],
            "information_cutoff": window["information_cutoff"],
            "assumed_execution": window["assumed_execution"],
            "entry": f"{window.get('entry_price_field')} on {window.get('entry_date')}",
            "exit": f"{window.get('exit_price_field')} on {window.get('exit_date')}",
            "available": bool(window["is_available"]),
            "unavailable_reason": window.get("unavailable_reason"),
        }
        for window in windows
    ]

    return {
        "brief_version": BRIEF_VERSION,
        "group_key": event["group_key"],
        "algorithm_version": event["algorithm_version"],
        "status": "candidate_event_group",
        "status_note": (
            "Formed by transparent rules over entities, family, time proximity "
            "and title similarity. This is an algorithmic grouping, not a "
            "verified real-world event."
        ),
        "review_state": event.get("review_state", "unreviewed"),
        "event_type": event.get("event_type"),
        "signal_family": event.get("signal_family"),
        "primary_entity": event.get("primary_entity"),
        "entities": (event.get("entity_ids") or "").split(",") if event.get("entity_ids") else [],
        "first_seen_at": event.get("first_seen_at"),
        "first_reactable_session": event.get("signal_date"),
        "source_breadth": {
            "source_count": event.get("source_count"),
            "sources": (event.get("sources") or "").split(",") if event.get("sources") else [],
            "headline_count": event.get("headline_count"),
        },
        "tone": {
            "mean": event.get("mean_sentiment"),
            "median": event.get("median_sentiment"),
            "dispersion": event.get("sentiment_dispersion"),
            "cross_source_dispersion": event.get("cross_source_dispersion"),
            "strong_positive_count": event.get("strong_positive_count"),
            "strong_negative_count": event.get("strong_negative_count"),
            "dispersion_note": (
                "Dispersion measures disagreement among the outlets that "
                "covered this group. It is not market uncertainty."
            ),
        },
        "novelty": {
            "value": event.get("novelty"),
            "prior_entity_events": event.get("prior_entity_events"),
            "note": (
                "Novelty reflects how often this entity has already produced a "
                "candidate group, not whether the underlying news is new."
            ),
        },
        "headline_examples": [
            {
                "title": row.get("title"),
                "source": row.get("source"),
                "published_timestamp": row.get("published_timestamp"),
                "sentiment_score": row.get("sentiment_score"),
                "similarity_to_group": row.get("similarity"),
                "match_rule": row.get("match_rule"),
                "is_market_recap": bool(row.get("is_market_recap")),
            }
            for row in examples
        ],
        "market_windows_for_later_evaluation": evaluable,
        "data_quality_warnings": _quality_warnings(event),
        "disclaimer": (
            "Descriptive research output. No trading recommendation is made and "
            "no predictive relationship has been validated."
        ),
    }


def write_briefs(
    briefs: Sequence[Dict[str, Any]], output_dir: Path
) -> Dict[str, str]:
    """Write briefs deterministically as JSON and a flat CSV index."""

    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(briefs, key=lambda brief: brief["group_key"])
    json_path = output_dir / "event_briefs.json"
    json_path.write_text(
        json.dumps(ordered, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )

    index_path = output_dir / "event_briefs_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "group_key", "review_state", "event_type", "signal_family",
            "primary_entity", "first_seen_at", "first_reactable_session",
            "source_count", "headline_count", "mean_tone", "dispersion",
            "warnings",
        ])
        for brief in ordered:
            writer.writerow([
                brief["group_key"], brief["review_state"], brief["event_type"],
                brief["signal_family"], brief["primary_entity"],
                brief["first_seen_at"], brief["first_reactable_session"],
                brief["source_breadth"]["source_count"],
                brief["source_breadth"]["headline_count"],
                brief["tone"]["mean"], brief["tone"]["dispersion"],
                "|".join(brief["data_quality_warnings"]),
            ])
    return {"briefs_json": str(json_path), "briefs_index": str(index_path)}
