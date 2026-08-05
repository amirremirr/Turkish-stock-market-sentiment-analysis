"""Apply, inspect, or roll back the reviewed legacy provenance migration.

Scores written before ``headlines.experiment_id`` existed carry NULL provenance.
Aggregation represents each such group as a distinct model-scoped legacy
identity, so once a new score lands the eligible set spans two identities and
``aggregate_step`` refuses to run. That is the safeguard working as designed:
the fix is to establish the legacy identity, not to weaken the check.

This migration reconstructs it only where the stored evidence is unambiguous --
an exact model/prompt match with complete score components and no existing
assignment -- and records every assignment in the append-only
``experiment_assignment_audit`` table.

Nothing here modifies a score, label, timestamp, or model name.

Usage::

    python -m scripts.migrate_legacy_experiment_id --db copy.db --survey
    python -m scripts.migrate_legacy_experiment_id --db copy.db --apply
    python -m scripts.migrate_legacy_experiment_id --db copy.db --rollback
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
from config import DB_PATH


def _print_survey(survey: dict) -> None:
    print("Reviewed legacy provenance survey")
    print(f"  reviewed model    {survey['reviewed_model_name']}")
    print(f"  reviewed identity {survey['reviewed_experiment_id']}")
    print(f"  eligible          {survey['eligible']}")
    print(f"  already assigned  {survey['already_assigned']}")
    print(f"  blocked           {survey['blocked_total']}")
    for model_name, count in sorted(survey["blocked"].items()):
        print(f"    - {model_name}: {count} row(s) keep NULL provenance")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default=DB_PATH, help="database to operate on")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--survey", action="store_true",
                        help="classify rows without changing anything")
    action.add_argument("--apply", action="store_true",
                        help="assign the reviewed identity to eligible rows")
    action.add_argument("--rollback", action="store_true",
                        help="revert only assignments this migration made")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.db).exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 2

    if args.survey:
        result = db.survey_reviewed_legacy_candidates(db_path=args.db)
        _print_survey(result)
    elif args.apply:
        result = db.backfill_reviewed_legacy_experiment_id(db_path=args.db)
        _print_survey(result)
        print(f"\n  ASSIGNED          {result['assigned']} row(s)")
        print(f"  reviewed_at       {result['reviewed_at']}")
    else:
        result = db.rollback_reviewed_legacy_experiment_id(db_path=args.db)
        print("Reviewed legacy provenance rollback")
        print(f"  reverted          {result['reverted']} row(s)")
        print(f"  skipped diverged  {result['skipped_diverged']} row(s)")
        if result["skipped_diverged"]:
            print(
                "  note: diverged rows were reassigned after this migration "
                "and were left untouched"
            )

    identities = db.get_eligible_experiment_ids(db_path=args.db)
    print(f"\n  eligible identities: {identities}")
    if len(identities) > 1:
        print("  WARNING: aggregation remains blocked by mixed identities")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {"result": result, "eligible_experiment_ids": identities},
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
