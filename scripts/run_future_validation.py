"""Run the sealed untouched_future_v1 test -- once, and only when it is ready.

Order of operations, each step refusing before the next can run:

1. **Sealed inputs are intact.** The stored definition re-hashes to its key; the
   frozen retrospective artifact verifies; its protocol specification hashes to
   the definition's ``protocol_hash``; today's code reproduces that
   specification (the dataset version label aside, which the definition itself
   pins to the v3 dataset) and reproduces the definition hash.
2. **Not already run.** One result per definition, stored append-only.
3. **Readiness.** The same report the daily workflow records. Until every gate
   passes, no target on the untouched side is read by this script.
4. **Run and score.** The frozen specifications on untouched sessions only,
   scored by :mod:`research.future_scoring` (the sealed criteria as worded).
5. **Record.** Database row plus ``docs/frozen/untouched_future_v1_result.json``.

``--check`` stops after step 3 and never reads an outcome, which is what a
scheduled readiness check should call.

Reporting rule (sealed): a failure or inconclusive result is reported in full,
and the protocol is not re-run with different settings to obtain a different
answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db  # noqa: E402
from research.future_scoring import SCORING_VERSION, result_hash, score  # noqa: E402
from research.future_validation import (  # noqa: E402
    EPOCH_UNTOUCHED, definition, definition_hash, partition,
)
from research.protocol import (  # noqa: E402
    BASELINE_SETS, NEWS_SETS, _spec, protocol_hash,
)
from research.walkforward import (  # noqa: E402
    build_folds, evaluate_specification, fold_boundaries_are_safe,
)

RESULT_FILE = REPOSITORY_ROOT / "docs" / "frozen" / "untouched_future_v1_result.json"

#: The one field allowed to differ between the sealed protocol specification and
#: today's: the dataset version label. The retrospective run recorded v2; the
#: sealed future definition names v3 in ``allowed_feature_versions``. Every
#: rule -- features, models, thresholds, folds, success criteria -- must match.
_PERMITTED_SPEC_DIFFERENCE = ("versions", "dataset")


class Refused(RuntimeError):
    """A precondition failed; nothing past it ran."""


def _stored_definition(db_path: str) -> Dict[str, Any]:
    import sqlite3

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM future_validation_definitions ORDER BY registered_at"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 1:
        raise Refused(f"expected exactly one sealed definition, found {len(rows)}")
    return dict(rows[0])


def _frozen_specification(db_path: str, artifact_hash: str) -> Dict[str, Any]:
    for row in db.list_frozen_results(db_path=db_path):
        if row["artifact_hash"] == artifact_hash:
            return json.loads(row["artifact_json"])["study"]["protocol_specification"]
    raise Refused(f"frozen artifact {artifact_hash[:16]} is not in the database")


def check_sealed_inputs(db_path: str) -> Dict[str, Any]:
    """Step 1. Returns the parsed definition, or raises :class:`Refused`."""

    from scripts.freeze_result import verify

    stored = _stored_definition(db_path)
    contract = json.loads(stored["definition_json"])
    if definition_hash(contract) != stored["definition_hash"]:
        raise Refused("stored definition does not re-hash to its key")

    verification = verify(db_path)
    if not verification["all_intact"]:
        raise Refused(f"frozen retrospective artifact failed verification: {verification}")

    frozen = _frozen_specification(db_path, stored["frozen_artifact_hash"])
    if protocol_hash(frozen) != stored["protocol_hash"]:
        raise Refused("frozen protocol specification does not hash to the sealed protocol_hash")

    current = json.loads(json.dumps(_spec()))
    section, key = _PERMITTED_SPEC_DIFFERENCE
    allowed_dataset = contract["allowed_feature_versions"]["dataset_version"]
    if current[section][key] != allowed_dataset:
        raise Refused(
            f"code dataset version {current[section][key]!r} is not the sealed "
            f"{allowed_dataset!r}"
        )
    current[section][key] = frozen[section][key]
    if protocol_hash(current) != stored["protocol_hash"]:
        raise Refused("today's protocol rules differ from the sealed protocol")

    rebuilt = definition(
        protocol_hash=stored["protocol_hash"],
        frozen_artifact_hash=stored["frozen_artifact_hash"],
    )
    if definition_hash(rebuilt) != stored["definition_hash"]:
        raise Refused("today's code does not reproduce the sealed definition")

    return {"stored": stored, "definition": contract}


def check_readiness(db_path: str) -> Dict[str, Any]:
    """Step 3. Counts and coverage only; raises :class:`Refused` if not ready."""

    from scripts.future_readiness import build_report

    report = build_report(db_path)
    if not report.get("eligible_to_run"):
        reasons = report.get("blocking_reasons") or ["not eligible"]
        raise Refused("untouched sample not ready: " + "; ".join(reasons))
    return report


def _untouched_units(db_path: str, contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Session units built exactly as the walk-forward primary path builds
    them, then restricted to the untouched side of the boundary.

    Features are attached over the full ordered history first, so the first
    untouched session's lagged inputs come from strictly earlier sessions
    rather than being blanked; only the units themselves are partitioned.
    """

    from research.modelling_unit import attach_lagged_features, build_session_units
    from scripts.run_validation import (
        _load_abnormal_tone, _load_dataset, _load_factor_panel, _load_regimes,
    )

    specification = _spec()
    units = attach_lagged_features(
        build_session_units(
            _load_dataset(db_path), window_name=contract["target"]["window"],
            min_events=specification["sample"]["minimum_events_per_session"],
        ),
        factor_panel=_load_factor_panel(db_path),
        abnormal_tone=_load_abnormal_tone(db_path),
        regimes=_load_regimes(db_path),
    )
    return partition(units, boundary=contract["first_eligible_session"])[EPOCH_UNTOUCHED]


def evaluate_untouched(
    units: Sequence[Dict[str, Any]], contract: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 4. Fit the frozen specifications and score them. Pure on its inputs."""

    from scripts.run_validation import DEFAULT_MODELS, MODEL_FOR_SET

    geometry = contract["fold_geometry"]
    embargo = _spec()["folds"]["embargo_sessions"]
    folds = build_folds(
        [unit["first_reactable_session"] for unit in units],
        initial_train=geometry["initial_train_sessions"],
        test_size=geometry["test_sessions"],
        step=geometry["step_sessions"],
        embargo=embargo,
    )
    safety = fold_boundaries_are_safe(folds)
    if not safety["safe"]:
        raise Refused(f"unsafe fold design: {safety['violations']}")

    target = contract["target"]["column"]
    specifications = []
    for feature_set_name in list(BASELINE_SETS) + list(NEWS_SETS):
        for model_name in MODEL_FOR_SET.get(feature_set_name, DEFAULT_MODELS):
            specifications.append(evaluate_specification(
                units, feature_set_name=feature_set_name, model_name=model_name,
                target=target, folds=folds,
                minimum_sessions=geometry["initial_train_sessions"],
                minimum_test_sessions=geometry["minimum_test_sessions_per_fold"],
            ))

    scored = score(specifications, folds, contract["thresholds"])
    scored["untouched_sessions"] = len(units)
    scored["fold_sessions"] = [
        {"fold": f.index, "train": len(f.train), "test": len(f.test),
         "test_first": f.test[0], "test_last": f.test[-1]}
        for f in folds
    ]
    scored["specification_status"] = [
        {"feature_set": s["feature_set"], "model": s["model"], "kind": s["kind"],
         "status": s["status"],
         "binding_requirement": (s.get("gate") or {}).get("binding_requirement")}
        for s in specifications
    ]
    return scored


def run(db_path: str, *, check_only: bool = False, write_file: bool = True) -> Dict[str, Any]:
    sealed = check_sealed_inputs(db_path)
    stored, contract = sealed["stored"], sealed["definition"]

    db.init_db(db_path=db_path)
    existing = db.get_future_validation_result(stored["definition_hash"], db_path=db_path)
    if existing:
        raise Refused(
            f"already run: verdict {existing['verdict']!r} recorded at "
            f"{existing['completed_at']}; the sealed test runs once"
        )

    readiness = check_readiness(db_path)
    if check_only:
        return {"status": "eligible_not_run", "readiness": readiness}

    from scripts.run_validation import _code_commit, _snapshot

    scored = evaluate_untouched(_untouched_units(db_path, contract), contract)
    result = {
        "version": contract["version"],
        "definition_hash": stored["definition_hash"],
        "protocol_hash": stored["protocol_hash"],
        "frozen_artifact_hash": stored["frozen_artifact_hash"],
        "scoring_version": SCORING_VERSION,
        "verdict": scored["verdict"],
        "verdict_reason": scored["verdict_reason"],
        "untouched_sessions": scored["untouched_sessions"],
        "folds": scored["folds_fittable"],
        "news_specifications": scored["news_specifications"],
        "scoring": scored,
        "readiness_at_run": {
            k: readiness.get(k) for k in (
                "untouched_sessions", "distinct_outcomes", "elapsed_days",
                "observed_at",
            )
        },
        "code_commit": _code_commit(),
        "database_snapshot": _snapshot(db_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "reporting_rule": contract["success_criteria"]["reporting_rule"],
        "on_failure": contract["on_failure"],
    }
    result["result_hash"] = result_hash(result)
    db.record_future_validation_result(result, db_path=db_path)
    if write_file:
        RESULT_FILE.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return {"status": "completed", "result": result}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(db.DB_PATH))
    parser.add_argument(
        "--check", action="store_true",
        help="verify sealed inputs and readiness only; never reads an outcome",
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        outcome = run(args.db, check_only=args.check)
    except Refused as refusal:
        print(f"REFUSED  {refusal}")
        return 3

    if outcome["status"] == "eligible_not_run":
        print("sealed inputs intact; untouched sample is ELIGIBLE. Not run (--check).")
        return 0
    result = outcome["result"]
    print(f"verdict   {result['verdict'].upper()}")
    print(f"reason    {result['verdict_reason']}")
    print(f"sessions  {result['untouched_sessions']}   fittable folds {result['folds']}   "
          f"news specifications {result['news_specifications']}")
    print(f"result    {result['result_hash'][:16]}  -> {RESULT_FILE.relative_to(REPOSITORY_ROOT)}")
    print(result["reporting_rule"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
