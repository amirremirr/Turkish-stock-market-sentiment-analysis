"""Refuse to publish a database snapshot that is older than the canonical one.

The daily workflow force-pushes the SQLite file to the ``data`` branch, so a
single run started from a stale copy would silently destroy newer production
state.  There is no history to recover from on a force-pushed orphan branch.

The guard compares monotonic freshness markers.  Each one only ever moves
forward in normal operation, so a candidate that scores lower on any of them
did not come from the canonical lineage:

    headline row count          rows are appended, never pruned
    max headlines.scraped_at    every run stamps newly fetched rows
    max headlines.published_at  the corpus frontier
    max bist100_prices.date     the market frontier
    max pipeline_runs.started_at  the run log

A missing table or column is treated as "unknown" and constrains nothing,
which keeps the guard usable against a pre-migration reference.

Usage::

    python -m scripts.guard_db_snapshot finance_sentiment.db --reference-git origin/data
    python -m scripts.guard_db_snapshot candidate.db --reference backups/canonical.db

Exit codes: ``0`` safe to publish, ``1`` regression detected, ``2`` bad input.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_FILENAME = "finance_sentiment.db"

# (metric, table, expression).  Counts and maxima are both monotonic.
_MARKERS = [
    ("headline_count", "headlines", "COUNT(*)"),
    ("max_scraped_at", "headlines", "MAX(scraped_at)"),
    ("max_published_at", "headlines", "MAX(published_at)"),
    ("max_price_date", "bist100_prices", "MAX(date)"),
    ("max_run_started_at", "pipeline_runs", "MAX(started_at)"),
]


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def snapshot_freshness(db_path: Path) -> Dict[str, Any]:
    """Read every freshness marker, tolerating absent legacy tables."""

    con = sqlite3.connect(str(db_path))
    try:
        markers: Dict[str, Any] = {}
        for name, table, expression in _MARKERS:
            if not _table_exists(con, table):
                markers[name] = None
                continue
            try:
                markers[name] = con.execute(
                    f"SELECT {expression} FROM {table}"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                markers[name] = None
        return markers
    finally:
        con.close()


def compare_snapshots(candidate: Path, reference: Path) -> Dict[str, Any]:
    """Return per-marker verdicts plus whether *candidate* may replace *reference*."""

    candidate_markers = snapshot_freshness(candidate)
    reference_markers = snapshot_freshness(reference)

    comparisons: List[Dict[str, Any]] = []
    for name, _, _ in _MARKERS:
        mine, theirs = candidate_markers[name], reference_markers[name]
        if theirs is None:
            verdict = "unconstrained"
        elif mine is None:
            verdict = "regression"
        elif mine < theirs:
            verdict = "regression"
        elif mine == theirs:
            verdict = "equal"
        else:
            verdict = "ahead"
        comparisons.append({
            "marker": name,
            "candidate": mine,
            "reference": theirs,
            "verdict": verdict,
        })

    regressions = [item for item in comparisons if item["verdict"] == "regression"]
    return {
        "candidate": str(candidate),
        "reference": str(reference),
        "comparisons": comparisons,
        "regressions": regressions,
        "safe_to_publish": not regressions,
    }


def extract_reference_from_git(
    revision: str, workdir: Path, *, filename: str = DEFAULT_DB_FILENAME
) -> Path:
    """Materialise ``<revision>:<filename>`` into *workdir*."""

    target = revision if ":" in revision else f"{revision}:{filename}"
    destination = workdir / "reference.db"
    completed = subprocess.run(
        ["git", "show", target],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise FileNotFoundError(
            f"cannot read {target}: {completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    destination.write_bytes(completed.stdout)
    return destination


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        f"candidate  {report['candidate']}",
        f"reference  {report['reference']}",
        "",
        f"  {'marker':<22}{'candidate':<26}{'reference':<26}verdict",
        "  " + "-" * 84,
    ]
    for item in report["comparisons"]:
        lines.append(
            f"  {item['marker']:<22}{str(item['candidate']):<26}"
            f"{str(item['reference']):<26}{item['verdict']}"
        )
    lines.append("")
    if report["safe_to_publish"]:
        lines.append("VERDICT: safe to publish - candidate is at or ahead of canonical")
    else:
        lines.append("VERDICT: REFUSED - candidate is behind the canonical snapshot on:")
        for item in report["regressions"]:
            lines.append(
                f"  - {item['marker']}: {item['candidate']} < {item['reference']}"
            )
        lines.append(
            "\nPublishing would destroy newer production state on a force-pushed"
            "\nbranch with no recoverable history. Refresh from the canonical"
            "\nsnapshot first, or pass --allow-regression with a written reason."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("candidate", type=Path, help="database about to be published")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", type=Path, help="canonical database file")
    source.add_argument(
        "--reference-git",
        help="canonical revision, e.g. 'origin/data' or 'origin/data:finance_sentiment.db'",
    )
    parser.add_argument(
        "--allow-regression",
        metavar="REASON",
        help="publish anyway; the reason is echoed into the output for the record",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.candidate.exists():
        print(f"candidate not found: {args.candidate}", file=sys.stderr)
        return 2

    temporary = None
    try:
        if args.reference is not None:
            if not args.reference.exists():
                print(f"reference not found: {args.reference}", file=sys.stderr)
                return 2
            reference = args.reference
        else:
            temporary = tempfile.TemporaryDirectory()
            try:
                reference = extract_reference_from_git(
                    args.reference_git, Path(temporary.name)
                )
            except FileNotFoundError as exc:
                # No canonical snapshot yet is a legitimate first-run state.
                print(f"no canonical reference available ({exc}); nothing to protect")
                return 0

        report = compare_snapshots(args.candidate, reference)
        print(format_report(report))
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
        if report["safe_to_publish"]:
            return 0
        if args.allow_regression:
            print(f"\nOVERRIDE ACCEPTED: {args.allow_regression}")
            return 0
        return 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
