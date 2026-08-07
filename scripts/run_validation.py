"""Execute the frozen walk-forward protocol against the research dataset.

Reads the modelling view, freezes (or re-confirms) the protocol, runs every
specification the sample gate permits, and writes results plus every
out-of-sample prediction back to the database so a reported metric can be
recomputed rather than trusted.

The protocol is frozen *before* results are read: the hash is computed and
stored first, and nothing in this script can alter the specification. If a
future run reports a different hash, the study changed.

Usage::

    python -m scripts.run_validation --db finance_sentiment.db
    python -m scripts.run_validation --db finance_sentiment.db --report out.md
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db  # noqa: E402
from research.modelling_unit import (  # noqa: E402
    MODELLING_UNIT_VERSION, attach_lagged_features, build_session_units,
    unit_counts,
)
from research.protocol import (  # noqa: E402
    BASELINE_SETS, FEATURE_SETS, NEWS_SETS, protocol_document, protocol_hash,
    select_geometry,
)
from research.return_windows import PRIMARY_WINDOW  # noqa: E402
from research.walkforward import (  # noqa: E402
    INSUFFICIENT, build_folds, compare_to_baselines, evaluate_specification,
    fold_boundaries_are_safe, subgroup_report,
)

#: Which model each feature set is evaluated with. Fixed by the protocol: a
#: feature set is not tried under several models until one of them works.
MODEL_FOR_SET = {
    "none": ("mean", "majority"),
    "previous_direction": ("mean", "majority"),
    "ar1": ("ridge", "logistic"),
    "headline_count_only": ("ridge", "logistic"),
    "net_tone_share": ("ridge", "logistic"),
    "market_controls_only": ("ridge", "logistic"),
}
DEFAULT_MODELS = ("ridge", "logistic")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPOSITORY_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:                                     # pragma: no cover
        return None


def _snapshot(db_path: str) -> Optional[str]:
    import hashlib

    try:
        return hashlib.sha256(Path(db_path).read_bytes()).hexdigest()
    except OSError:                                       # pragma: no cover
        return None


def _load_dataset(db_path: str) -> List[Dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM event_research_dataset"
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def _load_factor_panel(db_path: str) -> Dict[str, Dict[str, float]]:
    from research.controls import build_control_panel

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in connection.execute(
            "SELECT date, symbol, daily_return FROM market_factors ORDER BY date"
        )]
    finally:
        connection.close()
    return build_control_panel(rows)


def _load_abnormal_tone(db_path: str) -> Dict[str, Dict[str, float]]:
    """Prior-only tone surprise per session, for the ``abnormal_tone`` set.

    ``abnormal_tone_daily`` stores readings per family, outlet and
    outlet-family — there is no pre-aggregated session row — so the session
    value is the mean across families that produced a reading. Families with
    too little prior history are NULL there and are simply absent here; they are
    not counted as zero surprise, which would be a claim rather than a gap.

    ``domestic`` restricts to :data:`taxonomy.signal_family.DOMESTIC_FAMILIES`,
    excluding global risk and market recap.
    """

    from taxonomy.signal_family import DOMESTIC_FAMILIES

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in connection.execute(
            """SELECT signal_date, scope_key, abnormal_tone
                 FROM abnormal_tone_daily
                WHERE scope = 'family' AND abnormal_tone IS NOT NULL"""
        )]
    finally:
        connection.close()

    collected: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        session = str(row["signal_date"])
        value = float(row["abnormal_tone"])
        bucket = collected.setdefault(session, {"all": [], "domestic": []})
        bucket["all"].append(value)
        if str(row["scope_key"]) in DOMESTIC_FAMILIES:
            bucket["domestic"].append(value)

    return {
        session: {
            name: (sum(values) / len(values) if values else None)
            for name, values in scopes.items()
        }
        for session, scopes in collected.items()
    }


def _load_regimes(db_path: str) -> Dict[str, str]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        columns = {r[1] for r in connection.execute(
            "PRAGMA table_info(daily_signal_variants)"
        )}
        if "signal_date" not in columns:
            return {}
        rows = [dict(r) for r in connection.execute(
            "SELECT signal_date, headline_count FROM daily_signal_variants"
        )]
    finally:
        connection.close()

    counts = sorted(
        float(r["headline_count"]) for r in rows if r.get("headline_count")
    )
    if not counts:
        return {}
    median = counts[len(counts) // 2]
    # A coarse, pre-specified split. Named "attention" rather than "volatility"
    # because that is what it measures: how much was written, not how much the
    # market moved.
    return {
        str(r["signal_date"]): (
            "high_attention" if float(r["headline_count"] or 0) > median
            else "low_attention"
        )
        for r in rows if r.get("signal_date")
    }


#: Declared sensitivity analyses. Each re-runs one representative news
#: specification on a differently constructed sample. They exist to show how
#: fragile a result is, not to find a sample where one appears -- which is why
#: the set is fixed here and every one of them is reported.
SENSITIVITIES: Dict[str, Dict[str, Any]] = {
    "primary": {},
    "multi_source_events_only": {"multi_source_only": True},
    "singletons_removed": {"exclude_singletons": True},
    "timing_conflicts_included": {"exclude_conflicted": False},
    "descriptive_gap_window": {"__window__": "prior_close_to_reactable_open"},
    "descriptive_full_reaction_window": {
        "__window__": "prior_close_to_reactable_close",
    },
}

#: The specification carried through every sensitivity.
SENSITIVITY_SPECIFICATION = ("controls_plus_news", "ridge", "raw_return")


def _run_sensitivities(
    dataset: Sequence[Dict[str, Any]],
    db_path: str,
    *,
    window_name: str,
    specification: Dict[str, Any],
    geometry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Re-run one representative specification on each declared sample."""

    feature_set_name, model_name, target = SENSITIVITY_SPECIFICATION
    panel = _load_factor_panel(db_path)
    tone = _load_abnormal_tone(db_path)
    regimes = _load_regimes(db_path)

    reports: List[Dict[str, Any]] = []
    for name, filters in SENSITIVITIES.items():
        filters = dict(filters)
        window = filters.pop("__window__", window_name)
        # A descriptive window is not tradable, so the tradability filter that
        # protects the primary sample would empty it. Reported as descriptive.
        if window != window_name:
            filters["require_tradable"] = False

        units = attach_lagged_features(
            build_session_units(
                dataset, window_name=window,
                min_events=specification["sample"]["minimum_events_per_session"],
                **filters,
            ),
            factor_panel=panel, abnormal_tone=tone, regimes=regimes,
        )
        folds = (
            build_folds(
                [unit["first_reactable_session"] for unit in units],
                initial_train=geometry["initial_train_sessions"],
                test_size=geometry["test_sessions"],
                step=geometry["step_sessions"],
                embargo=specification["folds"]["embargo_sessions"],
            )
            if geometry["name"] != "none" else []
        )
        outcome = evaluate_specification(
            units, feature_set_name=feature_set_name, model_name=model_name,
            target=target, folds=folds,
            minimum_sessions=geometry["initial_train_sessions"] or 10**9,
            minimum_test_sessions=geometry["minimum_test_sessions_per_fold"] or 10**9,
        )
        pooled = outcome.get("pooled") or {}
        reports.append({
            "sensitivity": name,
            "window_name": window,
            "tradable": window == window_name,
            "sessions": len(units),
            "status": outcome["status"],
            "binding_requirement": outcome["gate"].get("binding_requirement"),
            "mae": pooled.get("mae"),
            "directional_accuracy": pooled.get("directional_accuracy"),
            "balanced_accuracy": pooled.get("balanced_accuracy"),
            "fitted_folds": (outcome.get("stability") or {}).get("fitted_folds", 0),
        })
    return reports


def run_validation(
    db_path: str,
    *,
    window_name: str = PRIMARY_WINDOW,
    persist: bool = True,
) -> Dict[str, Any]:
    """Run every protocol specification and return the full result set."""

    started = _now()
    document = protocol_document(
        code_commit=_code_commit(),
        database_snapshot=_snapshot(db_path),
        generated_at=started,
    )
    specification = document["specification"]
    if document["protocol_hash"] != protocol_hash():
        raise RuntimeError("protocol hash is not reproducible")

    dataset = _load_dataset(db_path)
    units = attach_lagged_features(
        build_session_units(
            dataset, window_name=window_name,
            min_events=specification["sample"]["minimum_events_per_session"],
        ),
        factor_panel=_load_factor_panel(db_path),
        abnormal_tone=_load_abnormal_tone(db_path),
        regimes=_load_regimes(db_path),
    )
    counts = unit_counts(dataset, units, window_name=window_name)

    # Geometry is chosen from the session count alone, before any model is
    # fitted and before any target has been read.
    geometry = select_geometry(len(units))
    folds: List[Any] = []
    if geometry["name"] != "none":
        folds = build_folds(
            [unit["first_reactable_session"] for unit in units],
            initial_train=geometry["initial_train_sessions"],
            test_size=geometry["test_sessions"],
            step=geometry["step_sessions"],
            embargo=specification["folds"]["embargo_sessions"],
        )
    safety = fold_boundaries_are_safe(folds)
    if not safety["safe"]:
        raise RuntimeError(f"unsafe fold design: {safety['violations']}")

    targets = [specification["target"]["primary"]["column"]] + [
        secondary["column"]
        for secondary in specification["target"]["secondary"]
        if secondary.get("window", window_name) == window_name
    ]
    targets = sorted(set(targets))

    results: List[Dict[str, Any]] = []
    for feature_set_name in list(BASELINE_SETS) + list(NEWS_SETS):
        for model_name in MODEL_FOR_SET.get(feature_set_name, DEFAULT_MODELS):
            for target in targets:
                # A tradable specification never sees a contemporaneous
                # control, on either side of the equation.
                if "contemporaneous" in target and FEATURE_SETS[
                    feature_set_name
                ]["kind"] == "news":
                    continue
                outcome = evaluate_specification(
                    units, feature_set_name=feature_set_name,
                    model_name=model_name, target=target, folds=folds,
                    minimum_sessions=geometry["initial_train_sessions"] or 10**9,
                    minimum_test_sessions=(
                        geometry["minimum_test_sessions_per_fold"] or 10**9
                    ),
                )
                outcome["subgroups"] = subgroup_report(outcome)
                results.append(outcome)

    sensitivities = _run_sensitivities(
        dataset, db_path, window_name=window_name, specification=specification,
        geometry=geometry,
    )

    comparison = compare_to_baselines(results)
    comparison["fold_geometry"] = geometry
    if not geometry.get("can_declare_success") and comparison["verdict"] == "success":
        # The reduced geometry is barred in advance from declaring success. A
        # win on four observations per parameter is a coin landing the same way
        # twice, and the protocol says so before the coin is tossed.
        comparison["verdict"] = "inconclusive"
        comparison["verdict_note"] = (
            f"{comparison['successes']} specification(s) cleared the thresholds, "
            f"but the {geometry['name']} fold geometry cannot declare success: "
            f"{geometry.get('reason')}"
        )

    run_id = None
    if persist:
        db.init_db(db_path=db_path)
        db.record_validation_protocol(document, db_path=db_path)
        run_id = db.record_validation_run(
            protocol_hash=document["protocol_hash"],
            code_commit=document["provenance"]["code_commit"],
            database_snapshot=document["provenance"]["database_snapshot"],
            experiment_id=(
                sorted({str(r.get("experiment_id")) for r in dataset
                        if r.get("experiment_id")})[:1] or [None]
            )[0],
            session_count=len(units), fold_count=len(folds),
            specifications=results, comparison=comparison, counts=counts,
            started_at=started, db_path=db_path,
        )

    return {
        "protocol": document,
        "validation_run_id": run_id,
        "counts": counts,
        "folds": [fold.as_row() for fold in folds],
        "fold_safety": safety,
        "targets": targets,
        "session_units": len(units),
        "specifications": results,
        "sensitivities": sensitivities,
        "comparison": comparison,
        "modelling_unit_version": MODELLING_UNIT_VERSION,
        "started_at": started,
        "finished_at": _now(),
    }


def _report(result: Dict[str, Any]) -> str:
    protocol = result["protocol"]
    counts = result["counts"]
    lines = [
        "# Walk-forward validation (retrospective exploration)",
        "",
        f"- protocol: `{protocol['specification']['protocol_version']}`",
        f"- protocol hash: `{protocol['protocol_hash']}`",
        f"- status: **{protocol['specification']['status']}**",
        f"- code commit: `{protocol['provenance']['code_commit']}`",
        f"- database snapshot: `{protocol['provenance']['database_snapshot']}`",
        "",
        "## Statistical unit",
        "",
        f"- event rows: {counts['event_rows']}",
        f"- distinct events: {counts['distinct_events']}",
        f"- distinct sessions: {counts['distinct_sessions']}",
        f"- distinct outcomes: {counts['distinct_outcomes']}",
        f"- session units modelled: {counts['session_units']}",
        f"- duplication factor: {counts['duplication_factor']}",
        "",
        "## Folds",
        "",
        "| fold | train | embargo | test | train range | test range |",
        "|---|---|---|---|---|---|",
    ]
    for fold in result["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['train_sessions']} | "
            f"{fold['embargoed_sessions']} | {fold['test_sessions']} | "
            f"{fold['train_start']}..{fold['train_end']} | "
            f"{fold['test_start']}..{fold['test_end']} |"
        )

    lines += ["", "## Specifications", "",
              "| feature set | model | target | status | folds | MAE | dir.acc | "
              "balanced | hit 95% CI | binding requirement |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for spec in result["specifications"]:
        pooled = spec.get("pooled") or {}
        interval = pooled.get("directional_hit_interval") or {}
        stability = spec.get("stability") or {}

        def _fmt(value, digits=4):
            return "n/a" if value is None else f"{value:.{digits}f}"

        ci = (
            f"[{_fmt(interval.get('lower'), 3)}, {_fmt(interval.get('upper'), 3)}]"
            if interval.get("lower") is not None else "n/a"
        )
        lines.append(
            f"| {spec['feature_set']} | {spec['model']} | {spec['target']} | "
            f"{spec['status']} | {stability.get('fitted_folds', 0)} | "
            f"{_fmt(pooled.get('mae'))} | "
            f"{_fmt(pooled.get('directional_accuracy'), 3)} | "
            f"{_fmt(pooled.get('balanced_accuracy'), 3)} | {ci} | "
            f"{spec['gate'].get('binding_requirement') or ''} |"
        )

    lines += [
        "", "## Sensitivity analyses",
        "",
        f"All re-run the same specification "
        f"(`{'/'.join(SENSITIVITY_SPECIFICATION)}`) on a differently "
        f"constructed sample.",
        "",
        "| sensitivity | window | tradable | sessions | status | folds | MAE | dir.acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["sensitivities"]:
        def _f(value, digits=4):
            return "n/a" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {row['sensitivity']} | {row['window_name']} | "
            f"{'yes' if row['tradable'] else 'no'} | {row['sessions']} | "
            f"{row['status']} | {row['fitted_folds']} | {_f(row['mae'])} | "
            f"{_f(row['directional_accuracy'], 3)} |"
        )
    lines += [
        "",
        "MAE is not comparable across windows: an overnight gap is a smaller "
        "number than a full session's range, so a lower error there reflects a "
        "narrower target, not a better model. The non-tradable rows measure "
        "reaction and cannot be earned.",
    ]

    comparison = result["comparison"]
    geometry = comparison.get("fold_geometry", {})
    lines += [
        "", "## Verdict", "",
        f"- fold geometry: **{geometry.get('name')}** "
        f"(can declare success: {geometry.get('can_declare_success')})",
        f"- specifications run: {comparison['specifications_run']}",
        f"- specifications blocked by the sample gate: "
        f"{comparison['specifications_blocked']}",
        f"- news specifications meeting every success criterion: "
        f"{comparison['successes']}",
        f"- **verdict: {comparison['verdict']}**",
        "",
        comparison["multiplicity_note"],
        "",
        "Retrospective walk-forward exploration on already-collected data. "
        "Not an untouched future test, and not a trading strategy.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default="finance_sentiment.db")
    parser.add_argument("--window", default=PRIMARY_WINDOW)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-persist", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_validation(
        args.db, window_name=args.window, persist=not args.no_persist,
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_report(result), encoding="utf-8")
        print(f"report written to {args.report}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"results written to {args.json}")

    comparison = result["comparison"]
    print(
        f"protocol_hash={result['protocol']['protocol_hash'][:16]} "
        f"sessions={result['session_units']} folds={len(result['folds'])} "
        f"run={comparison['specifications_run']} "
        f"blocked={comparison['specifications_blocked']} "
        f"verdict={comparison['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
