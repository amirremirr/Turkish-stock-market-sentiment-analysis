"""Readiness of the untouched-future sample. Counts only, never performance.

This report answers "is there enough data yet?" and deliberately cannot answer
"is it working?". That restriction is the point.

Watching out-of-sample accuracy accumulate and running the evaluation when it
looks favourable is optional-stopping. It inflates the false-positive rate and
leaves no trace in the resulting p-value, standard error or interval — the
number looks exactly like a number obtained honestly. The only defence is not
to look, and the only way not to look is to build something that cannot show
you.

So this module reads the untouched partition's *inputs* — sessions, events,
family coverage, missingness, control availability — and never touches
``raw_return`` beyond counting how many distinct values exist, which is a
completeness statistic. :func:`database.record_future_readiness` rejects any
report carrying an accuracy, an error or a correlation.

Usage::

    python -m scripts.future_readiness --db finance_sentiment.db
    python -m scripts.future_readiness --db finance_sentiment.db --json-out r.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db  # noqa: E402
from research.future_validation import (  # noqa: E402
    EPOCH_UNTOUCHED, FUTURE_VALIDATION_VERSION, MINIMUM_DISTINCT_OUTCOMES,
    MINIMUM_HORIZON_DAYS, MINIMUM_SESSIONS, STATE_ACCUMULATING, STATE_ELIGIBLE,
    corpus_epoch, definition, definition_hash,
)
from research.protocol import FEATURE_SETS, protocol_hash  # noqa: E402
from research.return_windows import PRIMARY_WINDOW  # noqa: E402

#: Feature columns whose availability decides whether a specification could
#: even be attempted. Checked for presence, never for value.
_TRACKED_INPUTS = sorted({
    feature
    for specification in FEATURE_SETS.values()
    for feature in specification["features"]
})


def _stored_definition(db_path: str) -> Optional[Dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM future_validation_definitions "
            "ORDER BY registered_at DESC LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        connection.close()
    return dict(row) if row else None


def _rows(db_path: str, boundary: str) -> List[Dict[str, Any]]:
    """Untouched-side dataset rows for the primary window."""

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in connection.execute(
            """SELECT group_key, first_reactable_session, signal_date,
                      signal_family, eligibility_status, eligibility_reason,
                      is_tradable_window, timing_conflict, raw_return,
                      residual_em_lagged, residual_em_oil_fx_lagged,
                      source_count, headline_count
                 FROM event_research_dataset
                WHERE window_name = ?""",
            (PRIMARY_WINDOW,),
        )]
    finally:
        connection.close()

    return [
        row for row in rows
        if corpus_epoch(
            row.get("first_reactable_session") or row.get("signal_date"),
            boundary=boundary,
        ) == EPOCH_UNTOUCHED
    ]


def _session_inputs(db_path: str, sessions: Sequence[str]) -> Dict[str, Any]:
    """Presence of each modelled input on the untouched sessions."""

    if not sessions:
        return {name: {"present": 0, "missing": 0} for name in _TRACKED_INPUTS}

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        columns = {r[1] for r in connection.execute(
            "PRAGMA table_info(session_modelling_units)"
        )}
        units = [dict(r) for r in connection.execute(
            "SELECT * FROM session_modelling_units"
        )] if columns else []
    finally:
        connection.close()

    wanted = set(sessions)
    selected = [
        u for u in units if str(u.get("first_reactable_session")) in wanted
    ]
    parsed: List[Dict[str, Any]] = []
    for unit in selected:
        merged = dict(unit)
        try:
            merged.update(json.loads(unit.get("features_json") or "{}"))
        except (TypeError, ValueError):
            pass
        parsed.append(merged)

    report: Dict[str, Any] = {}
    for name in _TRACKED_INPUTS:
        present = sum(1 for row in parsed if row.get(name) is not None)
        report[name] = {
            "present": present,
            "missing": len(parsed) - present,
            "coverage": round(present / len(parsed), 4) if parsed else None,
        }
    return report


def build_report(db_path: str, *, now: Optional[str] = None) -> Dict[str, Any]:
    """Readiness and data quality for the untouched sample. No performance."""

    stored = _stored_definition(db_path)
    if stored is None:
        contract = definition(protocol_hash=protocol_hash())
        digest = definition_hash(contract)
        boundary = contract["first_eligible_session"]
        start = contract["validation_start"]
        required_sessions = MINIMUM_SESSIONS
        required_days = MINIMUM_HORIZON_DAYS
        registered = False
    else:
        contract = json.loads(stored["definition_json"])
        digest = stored["definition_hash"]
        boundary = stored["first_eligible_session"]
        start = stored["validation_start"]
        required_sessions = int(stored["minimum_sessions"])
        required_days = int(stored["minimum_horizon_days"])
        registered = True

    observed_at = now or datetime.now(timezone.utc).isoformat()
    # Negative while the window is still opening: the boundary is deliberately
    # set after the freeze, so "not started yet" is a valid state rather than a
    # clock error. Clamped for the horizon test so a not-yet-open window can
    # never read as satisfying it.
    elapsed = (
        date.fromisoformat(observed_at[:10]) - date.fromisoformat(start[:10])
    ).days
    window_open = elapsed >= 0
    elapsed = max(0, elapsed)

    rows = _rows(db_path, boundary)
    eligible = [
        row for row in rows
        if row["eligibility_status"] == "eligible"
        and row.get("is_tradable_window")
        and not int(row.get("timing_conflict") or 0)
        and row.get("raw_return") is not None
    ]
    sessions = sorted({
        str(row.get("first_reactable_session") or row.get("signal_date"))
        for row in eligible
    })
    # A completeness statistic, not a performance one: how many independent
    # numbers the sample actually contains.
    outcomes = {
        round(float(row["raw_return"]), 10) for row in eligible
    }

    families: Dict[str, int] = {}
    for row in eligible:
        key = str(row.get("signal_family") or "unclassified")
        families[key] = families.get(key, 0) + 1

    blocked: Dict[str, int] = {}
    for row in rows:
        if row["eligibility_status"] != "eligible":
            key = str(row.get("eligibility_reason") or "unspecified")
            blocked[key] = blocked.get(key, 0) + 1

    controls = {
        "residual_em_lagged": len({
            str(r["first_reactable_session"]) for r in eligible
            if r.get("residual_em_lagged") is not None
        }),
        "residual_em_oil_fx_lagged": len({
            str(r["first_reactable_session"]) for r in eligible
            if r.get("residual_em_oil_fx_lagged") is not None
        }),
        "sessions_total": len(sessions),
    }

    reasons: List[str] = []
    if not registered:
        reasons.append("future-validation definition is not registered")
    if not window_open:
        reasons.append(
            f"the untouched window opens at {start[:10]}; it has not started"
        )
    if len(sessions) < required_sessions:
        reasons.append(
            f"{len(sessions)} untouched sessions < {required_sessions} required"
        )
    if len(outcomes) < MINIMUM_DISTINCT_OUTCOMES:
        reasons.append(
            f"{len(outcomes)} distinct outcomes < {MINIMUM_DISTINCT_OUTCOMES} required"
        )
    if elapsed < required_days:
        reasons.append(
            f"{elapsed} days elapsed < {required_days} required"
        )

    return {
        "observed_at": observed_at,
        "definition_hash": digest,
        "definition_version": contract.get("version", FUTURE_VALIDATION_VERSION),
        "definition_registered": registered,
        "validation_start": start,
        "first_eligible_session": boundary,
        "state": STATE_ACCUMULATING if reasons else STATE_ELIGIBLE,
        "untouched_sessions": len(sessions),
        "required_sessions": required_sessions,
        "session_range": (
            {"first": sessions[0], "last": sessions[-1]} if sessions else None
        ),
        "eligible_events": len(eligible),
        "untouched_rows": len(rows),
        "distinct_outcomes": len(outcomes),
        "required_outcomes": MINIMUM_DISTINCT_OUTCOMES,
        "elapsed_days": elapsed,
        "required_days": required_days,
        "family_coverage": families,
        "blocked_reasons": blocked,
        "missingness": _session_inputs(db_path, sessions),
        "control_availability": controls,
        "eligible_to_run": not reasons,
        "blocking_reasons": reasons,
        "sealed_note": (
            "Counts and coverage only. No accuracy, error or correlation is "
            "computed for the untouched sample until it is eligible to run; "
            "inspecting performance while a sample accumulates and stopping "
            "when it looks favourable inflates the false-positive rate "
            "invisibly."
        ),
    }


def _text(report: Dict[str, Any]) -> str:
    lines = [
        f"future validation   {report['definition_version']} "
        f"{report['definition_hash'][:16]} registered={report['definition_registered']}",
        f"boundary            first eligible session "
        f"{report['first_eligible_session']} (start {report['validation_start'][:10]})",
        f"state               {report['state'].upper()}",
        "",
        f"untouched sessions  {report['untouched_sessions']} / "
        f"{report['required_sessions']} required",
        f"distinct outcomes   {report['distinct_outcomes']} / "
        f"{report['required_outcomes']} required",
        f"elapsed days        {report['elapsed_days']} / "
        f"{report['required_days']} required",
        f"eligible events     {report['eligible_events']} "
        f"(of {report['untouched_rows']} untouched rows)",
    ]
    if report["session_range"]:
        lines.append(
            f"session range       {report['session_range']['first']} .. "
            f"{report['session_range']['last']}"
        )
    lines += ["", "family coverage"]
    for family, count in sorted(report["family_coverage"].items(),
                                key=lambda kv: -kv[1]) or []:
        lines.append(f"  {family:<28} {count}")
    if not report["family_coverage"]:
        lines.append("  (none yet)")

    lines += ["", "blocked rows"]
    for reason, count in sorted(report["blocked_reasons"].items(),
                                key=lambda kv: -kv[1]) or []:
        lines.append(f"  {reason:<38} {count}")
    if not report["blocked_reasons"]:
        lines.append("  (none)")

    lines += ["", "control availability (sessions)"]
    for name, count in report["control_availability"].items():
        lines.append(f"  {name:<28} {count}")

    lines += ["", "input missingness (untouched sessions)"]
    for name, stats in sorted(report["missingness"].items()):
        lines.append(
            f"  {name:<28} present {stats['present']:>4}  "
            f"missing {stats['missing']:>4}"
        )

    lines += ["", f"eligible to run     {report['eligible_to_run']}"]
    for reason in report["blocking_reasons"]:
        lines.append(f"  blocked: {reason}")
    lines += ["", report["sealed_note"]]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default="finance_sentiment.db")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--now", default=None, help="override the clock (testing)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args.db, now=args.now)

    if not args.no_record:
        db.record_future_readiness(report, db_path=args.db)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    stream = getattr(sys.stdout, "buffer", None)
    text = _text(report)
    if stream is not None:
        stream.write(text.encode("utf-8", errors="backslashreplace"))
        stream.flush()
    else:                                                  # pragma: no cover
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
