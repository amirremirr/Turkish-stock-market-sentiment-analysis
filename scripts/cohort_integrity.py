"""Fingerprint the historical record so a migration cannot alter it unnoticed.

The reviewed legacy cohort is the 3465 headlines whose experiment identity was
reconstructed in Phase 0. Those rows are the part of the corpus most worth
protecting: their scores predate per-score provenance, they can never be
regenerated, and every later research claim rests on them.

Cohort membership is read from ``experiment_assignment_audit`` rather than from
a row-id range or a count, because that table is the authoritative record of
which headlines were reconstructed and it is append-only.

Every check here is a real comparison. A verification that cannot fail is worse
than none, since it reports success either way.

Usage::

    python -m scripts.cohort_integrity --db production.db --save baseline.json
    python -m scripts.cohort_integrity --db migrated.db --compare baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db

REVIEWED_METHOD = "reviewed_legacy_backfill"

# Reported findings and figures. These are dated research artifacts: a pipeline
# run must never regenerate them as a side effect.
REPORTED_ARTIFACTS = [
    "docs/corpus_findings.md",
    "docs/external_findings.md",
    "docs/polarization_findings.md",
    "docs/corpus_overview.png",
    "docs/external_overview.png",
    "docs/polarization.png",
    "docs/sample_output.png",
    "labels_validated.csv",
]


def _digest_rows(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(repr(tuple(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def reviewed_cohort_ids(db_path: str) -> List[int]:
    with db._conn(db_path) as con:
        return [
            int(row[0])
            for row in con.execute(
                "SELECT DISTINCT headline_id FROM experiment_assignment_audit "
                "WHERE assignment_method = ? ORDER BY headline_id",
                (REVIEWED_METHOD,),
            )
        ]


def fingerprint(db_path: str) -> Dict[str, Any]:
    """Capture the historical record's identity."""

    ids = reviewed_cohort_ids(db_path)
    with db._conn(db_path) as con:
        placeholders = ",".join("?" * len(ids)) if ids else "NULL"
        cohort = con.execute(
            f"""SELECT id, sentiment_score, sentiment_label, scored_at,
                       model_name, experiment_id
                FROM headlines WHERE id IN ({placeholders}) ORDER BY id""",
            ids,
        ).fetchall() if ids else []
        cohort_digest = _digest_rows(cohort)

        audit_digest = _digest_rows(con.execute(
            """SELECT assignment_id, headline_id, assigned_experiment_id,
                      assignment_method, evidence, reviewed_at, migration_version
               FROM experiment_assignment_audit ORDER BY assignment_id"""
        ))

        # Both digests below are scoped to the reviewed cohort rather than to
        # the whole table. The corpus grows every run, so a whole-table digest
        # would change on ordinary ingestion and the check would cry wolf until
        # nobody read it. Scoping to the fixed historical set means a change
        # here always means something was rewritten.
        observations_present = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='raw_headline_observations'"
        ).fetchone() is not None
        if observations_present:
            observations_digest = _digest_rows(con.execute(
                f"""SELECT observation_id, observation_key, headline_id, source,
                           title, url, published_at, observed_at
                    FROM raw_headline_observations
                    WHERE headline_id IN ({placeholders})
                    ORDER BY observation_id""",
                ids,
            )) if ids else None
            observations_count = int(con.execute(
                "SELECT COUNT(*) FROM raw_headline_observations"
            ).fetchone()[0])
        else:
            observations_digest = None
            observations_count = 0

        category_digest = _digest_rows(con.execute(
            f"SELECT id, category FROM headlines WHERE id IN ({placeholders})"
            " ORDER BY id",
            ids,
        )) if ids else _digest_rows([])
        headline_count = int(
            con.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
        )

    artifacts = {}
    for relative in REPORTED_ARTIFACTS:
        path = REPOSITORY_ROOT / relative
        artifacts[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        )

    return {
        "reviewed_cohort_size": len(ids),
        "reviewed_cohort_digest": cohort_digest,
        "experiment_assignment_audit_digest": audit_digest,
        "raw_headline_observations_digest": observations_digest,
        "raw_headline_observations_count": observations_count,
        "category_digest": category_digest,
        "headline_count": headline_count,
        "reported_artifacts": artifacts,
    }


def compare(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two fingerprints. Every entry is an explicit equality test."""

    checks: List[Dict[str, Any]] = []

    def check(name: str, expected: Any, actual: Any) -> None:
        checks.append({
            "check": name,
            "passed": expected == actual,
            "expected": expected,
            "actual": actual,
        })

    check(
        "reviewed cohort size",
        baseline["reviewed_cohort_size"], current["reviewed_cohort_size"],
    )
    check(
        "reviewed cohort digest (id, score, label, scored_at, model, experiment)",
        baseline["reviewed_cohort_digest"], current["reviewed_cohort_digest"],
    )
    check(
        "experiment_assignment_audit digest",
        baseline["experiment_assignment_audit_digest"],
        current["experiment_assignment_audit_digest"],
    )
    check(
        "raw observations for the reviewed cohort",
        baseline["raw_headline_observations_digest"],
        current["raw_headline_observations_digest"],
    )
    check(
        "detailed categories for the reviewed cohort",
        baseline["category_digest"], current["category_digest"],
    )
    # The corpus is expected to grow; it must never shrink.
    checks.append({
        "check": "corpus did not lose headlines",
        "passed": current["headline_count"] >= baseline["headline_count"],
        "expected": f">= {baseline['headline_count']}",
        "actual": current["headline_count"],
    })
    checks.append({
        "check": "raw observations were not removed",
        "passed": (
            current["raw_headline_observations_count"]
            >= baseline["raw_headline_observations_count"]
        ),
        "expected": f">= {baseline['raw_headline_observations_count']}",
        "actual": current["raw_headline_observations_count"],
    })
    for relative in sorted(baseline["reported_artifacts"]):
        check(
            f"reported artifact unchanged: {relative}",
            baseline["reported_artifacts"][relative],
            current["reported_artifacts"].get(relative),
        )

    return {
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "failed": [item["check"] for item in checks if not item["passed"]],
    }


def format_report(result: Dict[str, Any]) -> str:
    lines = []
    for item in result["checks"]:
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"  [{mark}] {item['check']}")
        if not item["passed"]:
            lines.append(f"         expected {item['expected']}")
            lines.append(f"         actual   {item['actual']}")
    lines.append("")
    lines.append(f"INTEGRITY: {'PASS' if result['passed'] else 'FAIL'}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--save", type=Path, help="write the fingerprint as baseline")
    parser.add_argument("--compare", type=Path, help="compare against a baseline")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.db).exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 2

    current = fingerprint(args.db)
    print(f"reviewed cohort      {current['reviewed_cohort_size']} headline(s)")
    print(f"cohort digest        {current['reviewed_cohort_digest']}")
    print(f"audit digest         {current['experiment_assignment_audit_digest'][:32]}")
    print(f"observations         {current['raw_headline_observations_count']} row(s)")
    print(f"headlines            {current['headline_count']}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(current, indent=2, sort_keys=True), "utf-8")
        print(f"\nbaseline written to {args.save}")

    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        result = compare(baseline, current)
        print()
        print(format_report(result))
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
