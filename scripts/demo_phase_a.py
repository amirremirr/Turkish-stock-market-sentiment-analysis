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
        "notes": [
            "Descriptive only. Nothing here is a validated predictive signal.",
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
