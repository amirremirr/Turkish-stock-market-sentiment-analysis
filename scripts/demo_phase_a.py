"""Offline Phase A demonstration: families, recap, abnormal tone, regime.

Runs entirely from a small committed fixture. No API key, no network, no model
download, no private database. It exists so a reader can see what the
descriptive layer actually produces without being given access to anything.

The fixture is deliberately small but shaped like the real thing: several
sessions so prior-only windows have history to work with, more than one outlet
so disagreement is defined, a recap headline so the exclusion is visible, and a
volume spike on the last session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from indicators.abnormal_tone import compute_abnormal_tone
from indicators.disagreement import compute_disagreement
from indicators.family_signals import compute_family_signal
from indicators.volume_shock import compute_volume_shocks
from events.briefs import build_event_brief
from events.clustering import (
    CLUSTER_ALGORITHM_VERSION, group_candidate_events, summarise_event,
)
from research.dataset import build_event_dataset, dataset_coverage
from research.modelling_unit import (
    attach_lagged_features, build_session_units, unit_counts,
)
from research.return_windows import PRIMARY_WINDOW
from research.timing import (
    SIGNAL_DATE_SEMANTICS, TIMING_RULE_VERSION, previous_session,
)
from taxonomy.market_recap import classify_market_recap
from taxonomy.signal_family import (
    DOMESTIC_FAMILIES, SIGNAL_FAMILY_VERSION, assign_signal_family,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPOSITORY_ROOT / "sample_data" / "demo_family_headlines.csv"

EXPERIMENT_ID = "demo-v1"


def _read_fixture(path: Path) -> List[Dict[str, Any]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    prepared: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        recap = classify_market_recap(row["title"])
        family = assign_signal_family(
            row["category"], row["title"], is_market_recap=recap.is_market_recap
        )
        prepared.append({
            "id": index,
            "title": row["title"],
            "source": row["source"],
            "signal_date": row["signal_date"],
            "category": row["category"],
            "sentiment_score": float(row["sentiment_score"]),
            "sentiment_label": row["sentiment_label"],
            "relevance": float(row["relevance"]),
            "timing_bucket": row["timing_bucket"],
            "event_id": row.get("event_id") or None,
            "signal_family": family.signal_family,
            "signal_family_rule": family.rule,
            "signal_family_ambiguous": 1 if family.ambiguous else 0,
            "signal_family_review": family.review_reason,
            "is_market_recap": recap.as_int,
            "market_recap_rule": recap.rule,
            "market_recap_confidence": recap.confidence,
        })
    return prepared


def run_demo(
    output_dir: Path | str = "demo_output_phase_a",
    *,
    fixture_path: Path | str = DEFAULT_FIXTURE,
) -> Dict[str, Path]:
    """Produce the Phase A demo artifacts deterministically."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _read_fixture(Path(fixture_path))

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            (record["signal_date"], record["signal_family"]), []
        ).append(record)

    family_rows = [
        compute_family_signal(
            group, signal_date=session, signal_family=family,
            experiment_id=EXPERIMENT_ID, family_version=SIGNAL_FAMILY_VERSION,
        )
        for (session, family), group in sorted(grouped.items())
    ]

    domestic: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        if record["signal_family"] in DOMESTIC_FAMILIES:
            domestic.setdefault(record["signal_date"], []).append(record)
    domestic_rows = [
        compute_family_signal(
            group, signal_date=session, signal_family="__domestic__",
            experiment_id=EXPERIMENT_ID, family_version=SIGNAL_FAMILY_VERSION,
        )
        for session, group in sorted(domestic.items())
    ]

    abnormal_rows = compute_abnormal_tone(
        records, window_sessions=5, min_history=2, experiment_id=EXPERIMENT_ID,
    )
    disagreement_rows = [
        compute_disagreement(
            group, signal_date=session, signal_family=family,
            experiment_id=EXPERIMENT_ID,
            pro_government_sources=["Sabah Ekonomi"],
            opposition_sources=["Sozcu Ekonomi"],
            min_sources=2,
        )
        for (session, family), group in sorted(grouped.items())
    ]
    volume_rows = compute_volume_shocks(
        records, window_sessions=5, min_history=2, experiment_id=EXPERIMENT_ID,
    )

    sessions = sorted({record["signal_date"] for record in records})
    latest = sessions[-1]

    recap_headlines = [r for r in records if r["is_market_recap"]]
    directional = [r for r in records if not r["is_market_recap"]]

    drivers = sorted(
        (r for r in records if r["signal_date"] == latest),
        key=lambda r: r["sentiment_score"], reverse=True,
    )

    # -- candidate events and timing-safe market windows --------------------
    groups = group_candidate_events(records)
    event_rows = []
    for group in groups:
        row = summarise_event(group)
        buckets = [m.get("timing_bucket") for m in group.members if m.get("timing_bucket")]
        order = {"during_session": 0, "unknown": 1, "weekend_or_holiday": 2,
                 "post_close": 3, "pre_open": 4}
        row["timing_bucket"] = (
            min(buckets, key=lambda b: order.get(b, 1)) if buckets else "unknown"
        )
        event_rows.append(row)

    demo_bars = [
        {"date": "2026-06-01", "open": 100.0, "close": 101.5, "bar_status": "complete"},
        {"date": "2026-06-02", "open": 101.5, "close": 100.8, "bar_status": "complete"},
        {"date": "2026-06-03", "open": 100.8, "close": 102.4, "bar_status": "complete"},
        {"date": "2026-06-04", "open": 102.4, "close": 101.1, "bar_status": "complete"},
        {"date": "2026-06-05", "open": 101.1, "close": 99.6, "bar_status": "complete"},
        # A provisional bar the window builder must refuse to use.
        {"date": "2026-06-08", "open": 99.6, "close": 99.9, "bar_status": "provisional"},
    ]
    built = build_event_dataset(
        event_rows, demo_bars, [], experiment_id=EXPERIMENT_ID,
        algorithm_version=CLUSTER_ALGORITHM_VERSION,
    )
    coverage = dataset_coverage(built["dataset"])

    windows_by_group = {}
    for window in built["windows"]:
        windows_by_group.setdefault(window["group_key"], []).append(window)

    # -- the statistical unit, and why it is the session ---------------------
    units = attach_lagged_features(build_session_units(built["dataset"]))
    counts = unit_counts(built["dataset"], units)

    # -- the timing convention, shown rather than asserted -------------------
    # One example per bucket, so a reader sees the buckets that *discriminate*
    # between the two readings of signal_date, not three copies of pre_open.
    bucket_notes = {
        "pre_open": "published before the open; traded at that open",
        "post_close": "published after the close; traded at the next open",
        "weekend_or_holiday": "published off-calendar; traded at the next open",
        "during_session": "blocked: no intraday price to enter at",
        "unknown": "blocked: publication time unknown",
    }
    timing_examples = []
    for bucket, note in bucket_notes.items():
        example = next(
            (r for r in records if r["timing_bucket"] == bucket), None,
        )
        if example is None:
            continue
        session = example["signal_date"]
        tradable = bucket in ("pre_open", "post_close", "weekend_or_holiday")
        timing_examples.append({
            "title": example["title"][:70],
            "timing_bucket": bucket,
            "signal_date_is": SIGNAL_DATE_SEMANTICS,
            "first_reactable_session": session,
            "previous_trading_session": previous_session(session),
            "entry": session if tradable else None,
            "exit": session if tradable else None,
            "note": note,
        })

    largest = max(event_rows, key=lambda row: row["headline_count"])
    largest_members = next(
        g for g in groups if g.group_key == largest["group_key"]
    ).members
    example_brief = build_event_brief(
        {**largest, "algorithm_version": CLUSTER_ALGORITHM_VERSION},
        largest_members,
        windows_by_group.get(largest["group_key"], []),
    )

    summary = {
        "as_of": latest,
        "sessions": sessions,
        "taxonomy_version": SIGNAL_FAMILY_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "families_present": sorted({r["signal_family"] for r in records}),
        "market_recap": {
            "count": len(recap_headlines),
            "share": len(recap_headlines) / len(records),
            "examples": [r["title"] for r in recap_headlines],
            "excluded_from_directional_sample": True,
            "directional_sample_size": len(directional),
        },
        "domestic_only_latest": next(
            (row for row in domestic_rows if row["signal_date"] == latest), None
        ),
        "family_signals_latest": [
            row for row in family_rows if row["signal_date"] == latest
        ],
        "abnormal_tone_latest": [
            row for row in abnormal_rows
            if row["signal_date"] == latest and row["scope"] == "family"
        ],
        "disagreement_latest": [
            row for row in disagreement_rows if row["signal_date"] == latest
        ],
        "volume_latest": [
            row for row in volume_rows if row["signal_date"] == latest
        ],
        "drivers_latest": [
            {
                "title": r["title"], "source": r["source"],
                "signal_family": r["signal_family"],
                "sentiment_score": r["sentiment_score"],
                "relevance": r["relevance"],
                "timing_bucket": r["timing_bucket"],
                "is_market_recap": bool(r["is_market_recap"]),
                "family_rule": r["signal_family_rule"],
            }
            for r in drivers
        ],
        "candidate_events": {
            "algorithm_version": CLUSTER_ALGORITHM_VERSION,
            "group_count": len(event_rows),
            "singleton_groups": sum(r["is_singleton"] for r in event_rows),
            "multi_source_groups": sum(
                1 for r in event_rows if (r["source_count"] or 0) > 1
            ),
            "status_note": (
                "Algorithmic groupings, not verified real-world events."
            ),
        },
        "example_event_brief": example_brief,
        "market_window_coverage": coverage,
        "timing_convention": {
            "signal_date_means": SIGNAL_DATE_SEMANTICS,
            "rule_version": TIMING_RULE_VERSION,
            "primary_window": PRIMARY_WINDOW,
            "examples": timing_examples,
            "note": (
                "Every tradable bucket executes at the open of the first "
                "reactable session. Windows anchored to the prior close measure "
                "reaction and could not have been traded."
            ),
        },
        "statistical_unit": {
            **counts,
            "note": (
                "Events sharing a reactable session share one index return, so "
                "the session is the unit. The duplication factor is how many "
                "event rows sit behind each independent outcome."
            ),
        },
        "notes": [
            "Descriptive only. Nothing here is a validated predictive signal.",
            "Candidate event groups are algorithmic, not verified real-world events.",
            "Provisional price bars are excluded from every return window.",
            "signal_date is the first session able to react, never the "
            "publication session.",
            "NULL means the value could not be defensibly computed, most often "
            "for want of prior history or independent sources.",
            "Market recap is retained for attention analysis and excluded from "
            "the directional sample.",
        ],
    }

    summary_path = output / "phase_a_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True,
                   default=str) + "\n",
        encoding="utf-8",
    )

    import csv as _csv

    families_path = output / "phase_a_family_signals.csv"
    all_family_rows = family_rows + domestic_rows
    with families_path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(all_family_rows[0]))
        writer.writeheader()
        for row in sorted(
            all_family_rows, key=lambda r: (r["signal_date"], r["signal_family"])
        ):
            writer.writerow(row)

    drivers_path = output / "phase_a_drivers.csv"
    with drivers_path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(
            handle, fieldnames=list(summary["drivers_latest"][0])
        )
        writer.writeheader()
        writer.writerows(summary["drivers_latest"])

    return {
        "summary": summary_path,
        "family_signals": families_path,
        "drivers": drivers_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output_phase_a"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = run_demo(args.output_dir)
    print("Offline Phase A demo complete: no key, network, model, or private DB.")
    print("Descriptive indicators only - no strategy, no alpha claim.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
