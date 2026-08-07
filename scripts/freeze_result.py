"""Seal the completed retrospective study and open the untouched-future window.

Two things happen here, in this order and only in this order:

1. ``walk-forward-protocol-v1`` and its results become an immutable artifact --
   in an append-only table and in a committed JSON file, hashed so a later
   edit is detectable rather than merely discouraged.
2. ``untouched_future_v1`` is registered, referencing that artifact's hash. The
   boundary is therefore anchored to a study that is already sealed; it cannot
   be back-dated to include data the retrospective analysis had seen.

Re-running is safe: both steps are idempotent by content hash. Freezing the
same study twice proves identity instead of producing a second record.

Usage::

    python -m scripts.freeze_result --db finance_sentiment.db
    python -m scripts.freeze_result --db finance_sentiment.db --verify-only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db  # noqa: E402
from research.frozen_result import (  # noqa: E402
    RETROSPECTIVE_CONCLUSION, build_artifact, canonical_json, summary,
    verify_artifact,
)
from research.future_validation import (  # noqa: E402
    FUTURE_VALIDATION_VERSION, definition, definition_hash,
)

#: Committed alongside the code, so the record lives in git history too.
ARTIFACT_DIR = REPOSITORY_ROOT / "docs" / "frozen"


def _load_run(db_path: str, validation_run_id: Optional[int]) -> Dict[str, Any]:
    """Reassemble a stored validation run into the shape build_artifact wants."""

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        if validation_run_id is None:
            row = connection.execute(
                """SELECT * FROM validation_runs
                    WHERE specifications_run > 0
                    ORDER BY validation_run_id DESC LIMIT 1"""
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM validation_runs WHERE validation_run_id = ?",
                (validation_run_id,),
            ).fetchone()
        if row is None:
            raise SystemExit(
                "no validation run with fitted specifications is stored; "
                "run scripts.run_validation first"
            )
        run = dict(row)

        protocol = connection.execute(
            "SELECT * FROM validation_protocols WHERE protocol_hash = ?",
            (run["protocol_hash"],),
        ).fetchone()
        if protocol is None:
            raise SystemExit(f"protocol {run['protocol_hash']} is not stored")

        results = [dict(r) for r in connection.execute(
            "SELECT * FROM validation_results WHERE validation_run_id = ?",
            (run["validation_run_id"],),
        )]
        predictions = [dict(r) for r in connection.execute(
            "SELECT * FROM validation_predictions WHERE validation_run_id = ?",
            (str(run["validation_run_id"]),),
        )]
    finally:
        connection.close()

    by_specification: Dict[tuple, list] = {}
    for prediction in predictions:
        key = (prediction["feature_set"], prediction["model"], prediction["target"])
        by_specification.setdefault(key, []).append(prediction)

    specifications = []
    for result in results:
        key = (result["feature_set"], result["model"], result["target"])
        interval = {"lower": result.get("hit_lower"), "upper": result.get("hit_upper")}
        specifications.append({
            "feature_set": result["feature_set"],
            "model": result["model"],
            "target": result["target"],
            "kind": result["kind"],
            "status": result["status"],
            "gate": {
                "binding_requirement": result.get("binding_requirement"),
                "rows_complete": result.get("rows_complete"),
                "usable_sessions": result.get("usable_sessions"),
                "missing_by_column": None,
            },
            "stability": json.loads(result.get("stability_json") or "{}"),
            "subgroups": json.loads(result.get("subgroups_json") or "{}"),
            "pooled": ({
                "mae": result.get("mae"), "rmse": result.get("rmse"),
                "pearson_r": result.get("pearson_r"),
                "directional_accuracy": result.get("directional_accuracy"),
                "balanced_accuracy": result.get("balanced_accuracy"),
                "brier_score": result.get("brier_score"),
                "directional_hit_interval": interval,
            } if result["status"] == "fitted" else None),
            "predictions": sorted(
                by_specification.get(key, []),
                key=lambda p: str(p["first_reactable_session"]),
            ),
        })

    document = {
        "specification": json.loads(protocol["specification_json"]),
        "protocol_hash": run["protocol_hash"],
        "provenance": {
            "code_commit": run.get("code_commit"),
            "database_snapshot": run.get("database_snapshot"),
            "dataset_version": protocol["dataset_version"],
            "feature_version": protocol["feature_version"],
            "target_version": protocol["target_version"],
            "modelling_unit_version": None,
            "timing_rule_version": None,
            "return_window_version": None,
        },
    }
    from research.modelling_unit import MODELLING_UNIT_VERSION
    from research.return_windows import RETURN_WINDOW_VERSION
    from research.timing import TIMING_RULE_VERSION

    document["provenance"].update({
        "modelling_unit_version": MODELLING_UNIT_VERSION,
        "timing_rule_version": TIMING_RULE_VERSION,
        "return_window_version": RETURN_WINDOW_VERSION,
    })

    fitted = [s for s in specifications if s["status"] == "fitted"]
    from research.walkforward import compare_to_baselines

    comparison = compare_to_baselines(fitted)
    comparison["specifications_blocked"] = sum(
        1 for s in specifications if s["status"] != "fitted"
    )
    return {
        "run": run,
        "document": document,
        "specifications": specifications,
        "comparison": comparison,
        "counts": json.loads(run.get("counts_json") or "{}"),
    }


def freeze(
    db_path: str, *, validation_run_id: Optional[int] = None,
    write_file: bool = True,
) -> Dict[str, Any]:
    """Build, store and (optionally) commit the frozen artifact."""

    loaded = _load_run(db_path, validation_run_id)
    run = loaded["run"]

    artifact = build_artifact(
        protocol_document=loaded["document"],
        specifications=loaded["specifications"],
        comparison=loaded["comparison"],
        counts=loaded["counts"],
        folds=[],
        validation_run_id=run["validation_run_id"],
        frozen_at=datetime.now(timezone.utc).isoformat(),
    )

    stored = db.freeze_research_result(artifact, db_path=db_path)
    future = definition(
        protocol_hash=artifact["study"]["protocol_hash"],
        frozen_artifact_hash=artifact["artifact_hash"],
    )
    registered = db.register_future_validation(future, db_path=db_path)

    written = None
    if write_file:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        target = ARTIFACT_DIR / (
            f"{artifact['study']['protocol_version']}.json"
        )
        target.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False,
                       default=str) + "\n",
            encoding="utf-8",
        )
        contract = ARTIFACT_DIR / f"{FUTURE_VALIDATION_VERSION}.json"
        contract.write_text(
            json.dumps(
                {**future, "definition_hash": registered["definition_hash"]},
                indent=2, sort_keys=True, ensure_ascii=False, default=str,
            ) + "\n",
            encoding="utf-8",
        )
        written = [str(target), str(contract)]

    return {
        "artifact": artifact,
        "summary": summary(artifact),
        "verification": verify_artifact(artifact),
        "stored": stored,
        "future_validation": {**registered, "definition": future},
        "files": written,
    }


def verify(db_path: str) -> Dict[str, Any]:
    """Re-hash every stored artifact and re-check the committed files."""

    checks = []
    for row in db.list_frozen_results(db_path=db_path):
        artifact = json.loads(row["artifact_json"])
        result = verify_artifact(artifact)
        committed = ARTIFACT_DIR / (
            f"{artifact['study']['protocol_version']}.json"
        )
        file_matches = None
        if committed.exists():
            on_disk = json.loads(committed.read_text(encoding="utf-8"))
            file_matches = canonical_json(on_disk) == canonical_json(artifact)
        checks.append({
            "artifact_hash": row["artifact_hash"],
            "protocol_version": artifact["study"]["protocol_version"],
            "frozen_at": row["frozen_at"],
            "committed_file_matches": file_matches,
            **result,
        })
    return {
        "artifacts": checks,
        "all_intact": all(
            c["intact"] and c["conclusion_verbatim"]
            and c["committed_file_matches"] is not False
            for c in checks
        ) if checks else False,
        "count": len(checks),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default="finance_sentiment.db")
    parser.add_argument("--validation-run-id", type=int, default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-file", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.verify_only:
        report = verify(args.db)
        for check in report["artifacts"]:
            print(
                f"{check['protocol_version']}  intact={check['intact']}  "
                f"verbatim={check['conclusion_verbatim']}  "
                f"file_matches={check['committed_file_matches']}"
            )
        print(f"all_intact={report['all_intact']} ({report['count']} artifact(s))")
        return 0 if report["all_intact"] else 1

    result = freeze(
        args.db, validation_run_id=args.validation_run_id,
        write_file=not args.no_file,
    )
    brief = result["summary"]
    print(f"protocol      {brief['protocol_version']}  {brief['protocol_hash'][:16]}")
    print(f"artifact      {brief['artifact_hash'][:16]}  "
          f"already_frozen={result['stored']['already_frozen']}")
    print(f"commit        {brief['code_commit']}")
    print(f"db snapshot   {(brief['database_snapshot'] or '')[:16]}")
    print(f"sessions      {brief['independent_sessions']} independent "
          f"({brief['event_rows']} event rows, "
          f"{brief['duplication_factor']}x duplication)")
    print(f"specs         {brief['specifications_run']} run, "
          f"{brief['specifications_blocked']} blocked, "
          f"{brief['successful_news_specifications']} successful")
    print(f"verdict       {brief['verdict']}")
    print(f"\n{RETROSPECTIVE_CONCLUSION}\n")
    print(f"future        {FUTURE_VALIDATION_VERSION} "
          f"{result['future_validation']['definition_hash'][:16]} "
          f"already_registered={result['future_validation']['already_registered']}")
    print(f"              first eligible session "
          f"{result['future_validation']['definition']['first_eligible_session']}")
    for path in result["files"] or []:
        print(f"wrote         {path}")
    return 0 if result["verification"]["intact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
