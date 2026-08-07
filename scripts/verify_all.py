"""One command that checks everything a reader would otherwise have to trust.

Runs, in order, and reports each independently rather than stopping at the
first failure -- a partial pass tells you more than an early exit:

1. **schema**     every expected table and column exists, and the append-only
                  triggers actually refuse a write
2. **artifacts**  the frozen study re-hashes to its stored value, its conclusion
                  is verbatim, and the committed JSON matches the database
3. **integrity**  historical scores, categories, provenance and reported
                  findings are unchanged
4. **timing**     the ``signal_date`` convention still holds against real rows,
                  and every sampled window is aligned
5. **tests**      the full pytest suite
6. **demo**       the credential-free demo produces its artifacts and makes no
                  predictive claim

Nothing here needs an API key, a network connection or a private database,
except where a database path is supplied for checks 2-4.

Usage::

    python -m scripts.verify_all
    python -m scripts.verify_all --db finance_sentiment.db
    python -m scripts.verify_all --skip tests
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

#: Tables the schema check requires, with a column each check would notice the
#: absence of. Deliberately explicit rather than derived from the DDL, so a
#: table dropped from both would still fail here.
REQUIRED_TABLES = {
    "headlines": ("signal_date", "timing_bucket", "experiment_id"),
    "bist100_prices": ("bar_status", "source", "retrieved_at"),
    "market_factors": ("daily_return", "source", "transform_version"),
    "event_groups": ("first_reactable_session", "timing_conflict"),
    "event_return_windows": ("is_tradable", "rule_version"),
    "event_research_dataset": ("corpus_epoch", "is_tradable_window"),
    "session_modelling_units": ("corpus_epoch", "modelling_unit_version"),
    "validation_protocols": ("protocol_hash", "specification_json"),
    "validation_runs": ("protocol_hash", "verdict"),
    "validation_results": ("status", "binding_requirement"),
    "validation_predictions": ("actual", "predicted"),
    "frozen_research_results": ("artifact_hash", "conclusion"),
    "future_validation_definitions": ("first_eligible_session", "protocol_hash"),
    "future_validation_readiness": ("state", "eligible_to_run"),
    "event_review_sample": ("stratum", "group_key"),
    "experiment_assignment_audit": ("assigned_experiment_id", "evidence"),
    "event_group_audit": ("action", "actor"),
}

APPEND_ONLY_TABLES = (
    "frozen_research_results",
    "future_validation_definitions",
    "experiment_assignment_audit",
    "event_group_audit",
)


def _result(name: str, status: str, detail: Any = None) -> Dict[str, Any]:
    return {"check": name, "status": status, "detail": detail}


# ---------------------------------------------------------------------------
def check_schema(db_path: Optional[str]) -> Dict[str, Any]:
    """Tables, columns and append-only triggers, on a throwaway database.

    Built fresh from the DDL rather than inspected in place, so the check
    proves ``init_db`` produces the schema rather than proving that some
    existing file happens to have it.
    """

    import database as db

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "schema.db")
        db.init_db(db_path=path)
        connection = sqlite3.connect(path)
        try:
            missing: List[str] = []
            for table, columns in REQUIRED_TABLES.items():
                found = {
                    row[1] for row in connection.execute(
                        f"PRAGMA table_info({table})"
                    )
                }
                if not found:
                    missing.append(f"{table} (table absent)")
                    continue
                for column in columns:
                    if column not in found:
                        missing.append(f"{table}.{column}")

            # A DELETE against an empty table never fires a BEFORE DELETE
            # trigger -- there is no row to fire it for -- so an emptiness
            # check would pass on a table with no protection at all. Assert the
            # triggers exist, then prove one actually bites by inserting a row
            # and trying to remove it.
            triggers = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            }
            refused: List[str] = []
            for table in APPEND_ONLY_TABLES:
                for action in ("update", "delete"):
                    if f"trg_{table}_no_{action}" not in triggers:
                        refused.append(f"{table}: missing no_{action} trigger")

            connection.execute(
                """INSERT INTO event_group_audit
                   (group_key, algorithm_version, action, actor, performed_at)
                   VALUES ('probe', 'probe', 'annotate', 'verify_all', 'now')"""
            )
            for statement in (
                "UPDATE event_group_audit SET actor = 'x' WHERE group_key = 'probe'",
                "DELETE FROM event_group_audit WHERE group_key = 'probe'",
            ):
                try:
                    connection.execute(statement)
                except sqlite3.IntegrityError:
                    continue
                refused.append(f"event_group_audit accepted: {statement[:30]}")
        finally:
            connection.close()

    problems = {"missing_columns": missing, "mutable_append_only": refused}
    return _result(
        "schema", FAIL if (missing or refused) else PASS,
        problems if (missing or refused) else
        {"tables": len(REQUIRED_TABLES), "append_only": len(APPEND_ONLY_TABLES)},
    )


def check_artifacts(db_path: Optional[str]) -> Dict[str, Any]:
    """The frozen study still hashes to what it claims, verbatim."""

    if not db_path or not Path(db_path).exists():
        return _result("artifacts", SKIP, "no database supplied")

    from scripts.freeze_result import verify

    report = verify(db_path)
    if not report["count"]:
        return _result("artifacts", SKIP, "no frozen study yet")
    return _result(
        "artifacts", PASS if report["all_intact"] else FAIL,
        {"artifacts": report["count"], "checks": report["artifacts"]},
    )


def check_integrity(db_path: Optional[str]) -> Dict[str, Any]:
    """Historical scores, categories and provenance are unchanged."""

    if not db_path or not Path(db_path).exists():
        return _result("integrity", SKIP, "no database supplied")

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.cohort_integrity", "--db", db_path],
        cwd=str(REPOSITORY_ROOT), capture_output=True, text=True,
    )
    return _result(
        "integrity", PASS if completed.returncode == 0 else FAIL,
        (completed.stdout or completed.stderr).strip().splitlines()[-6:],
    )


def check_timing(db_path: Optional[str]) -> Dict[str, Any]:
    """The signal_date convention still holds against production rows."""

    if not db_path or not Path(db_path).exists():
        return _result("timing", SKIP, "no database supplied")

    from scripts.timing_audit import run_audit

    report = run_audit(db_path, per_bucket=25)
    semantics = report["semantics"]
    ok = report["all_passed"] and semantics["agrees_with_declared"]
    return _result(
        "timing", PASS if ok else FAIL,
        {
            "verdict": semantics["verdict"],
            "agrees_with_declared": semantics["agrees_with_declared"],
            "sampled_rows_passed": report["all_passed"],
            "per_bucket": report["sampled_per_bucket"],
        },
    )


def check_tests(_: Optional[str]) -> Dict[str, Any]:
    """The full deterministic suite."""

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=str(REPOSITORY_ROOT), capture_output=True, text=True,
    )
    tail = (completed.stdout or completed.stderr).strip().splitlines()[-3:]
    return _result(
        "tests", PASS if completed.returncode == 0 else FAIL, tail,
    )


def check_demo(_: Optional[str]) -> Dict[str, Any]:
    """The credential-free demo runs and makes no predictive claim."""

    from scripts.demo_phase_a import run_demo

    with tempfile.TemporaryDirectory() as directory:
        artifacts = run_demo(Path(directory) / "demo")
        summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
        sizes = {name: path.stat().st_size for name, path in artifacts.items()}

    notes = " ".join(summary.get("notes") or []).lower()
    problems = []
    if any(size <= 0 for size in sizes.values()):
        problems.append("an artifact is empty")
    if "nothing here is a validated predictive signal" not in notes:
        problems.append("the demo no longer denies a predictive claim")
    if summary.get("timing_convention", {}).get("signal_date_means") != (
        "first_reactable_session"
    ):
        problems.append("the demo does not state the timing convention")

    return _result(
        "demo", FAIL if problems else PASS,
        problems or {"artifacts": sizes,
                     "sessions": len(summary.get("sessions") or [])},
    )


CHECKS: Dict[str, Callable[[Optional[str]], Dict[str, Any]]] = {
    "schema": check_schema,
    "artifacts": check_artifacts,
    "integrity": check_integrity,
    "timing": check_timing,
    "tests": check_tests,
    "demo": check_demo,
}


def run_all(
    db_path: Optional[str] = None, *, skip: Sequence[str] = (),
) -> Dict[str, Any]:
    """Run every check, collecting failures rather than stopping at the first."""

    results: List[Dict[str, Any]] = []
    for name, check in CHECKS.items():
        if name in skip:
            results.append({**_result(name, SKIP, "skipped by request"),
                            "seconds": 0.0})
            continue
        started = time.perf_counter()
        try:
            outcome = check(db_path)
        except Exception as exc:                            # noqa: BLE001
            outcome = _result(name, FAIL, f"{type(exc).__name__}: {exc}")
        outcome["seconds"] = round(time.perf_counter() - started, 2)
        results.append(outcome)

    failed = [r for r in results if r["status"] == FAIL]
    return {
        "results": results,
        "passed": not failed,
        "failed": [r["check"] for r in failed],
        "database": db_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default=None,
                        help="database for the artifact, integrity and timing "
                             "checks; omit to skip those")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(CHECKS))
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_all(args.db, skip=args.skip)

    print("verification")
    print("-" * 60)
    for outcome in report["results"]:
        detail = outcome["detail"]
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail, default=str)[:120]
        print(f"  {outcome['status']:<5} {outcome['check']:<12} "
              f"{outcome['seconds']:>6.2f}s  {detail or ''}")
    print("-" * 60)
    print("ALL CHECKS PASSED" if report["passed"]
          else f"FAILED: {', '.join(report['failed'])}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8",
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
