"""Seal a completed study as an immutable historical artifact.

A research result is only meaningful if it cannot quietly become a different
result. Once `walk-forward-protocol-v1` finished, its verdict stopped being a
live number and became a record of what was found, on which data, under which
rules, at which commit.

What "frozen" means here, concretely:

* the artifact is written once and never updated — the database table is
  append-only by trigger, and the JSON is committed to git, where a change is a
  diff someone has to justify;
* it carries everything needed to re-derive it: protocol hash, code commit,
  database snapshot digest, every version string, every specification result,
  every out-of-sample prediction, and the sample sizes behind each;
* the conclusion sentence is stored verbatim and is not regenerated from the
  numbers, so a later bug in a formatter cannot reword a finding.

A future protocol version that performs differently does **not** revise this.
It is a different study, and it gets its own artifact. The whole point of
freezing is that "we tried again and it worked" cannot be presented as "it
worked".
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

FROZEN_ARTIFACT_VERSION = "frozen-result-v1"

#: The retrospective conclusion, stored verbatim. Never recomputed from data.
RETROSPECTIVE_CONCLUSION = (
    "No evaluated news specification demonstrated reliable incremental "
    "out-of-sample predictive value under the pre-specified criteria in the "
    "current sample."
)

STATUS_FROZEN = "frozen"


def _round(value: Any, digits: int = 10) -> Any:
    """Round floats so a re-serialisation cannot change the digest."""

    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def build_artifact(
    *,
    protocol_document: Dict[str, Any],
    specifications: Sequence[Dict[str, Any]],
    comparison: Dict[str, Any],
    counts: Dict[str, Any],
    folds: Sequence[Dict[str, Any]],
    sensitivities: Sequence[Dict[str, Any]] = (),
    validation_run_id: Optional[int] = None,
    frozen_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the complete immutable record of one finished study."""

    specification = protocol_document["specification"]
    provenance = protocol_document["provenance"]

    results: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    for entry in specifications:
        pooled = entry.get("pooled") or {}
        interval = pooled.get("directional_hit_interval") or {}
        results.append(_round({
            "feature_set": entry["feature_set"],
            "model": entry["model"],
            "target": entry["target"],
            "kind": entry["kind"],
            "status": entry["status"],
            "binding_requirement": entry["gate"].get("binding_requirement"),
            "rows_complete": entry["gate"].get("rows_complete"),
            "usable_sessions": entry["gate"].get("usable_sessions"),
            "missing_by_column": entry["gate"].get("missing_by_column"),
            "fitted_folds": (entry.get("stability") or {}).get("fitted_folds", 0),
            "mae": pooled.get("mae"),
            "rmse": pooled.get("rmse"),
            "pearson_r": pooled.get("pearson_r"),
            "directional_accuracy": pooled.get("directional_accuracy"),
            "balanced_accuracy": pooled.get("balanced_accuracy"),
            "brier_score": pooled.get("brier_score"),
            "hit_rate_ci_lower": interval.get("lower"),
            "hit_rate_ci_upper": interval.get("upper"),
            "prediction_coverage": pooled.get("prediction_coverage"),
            "stability": entry.get("stability"),
            "subgroups": entry.get("subgroups"),
        }))
        for prediction in entry.get("predictions") or []:
            predictions.append(_round({
                "feature_set": entry["feature_set"],
                "model": entry["model"],
                "target": entry["target"],
                **{
                    key: prediction.get(key) for key in (
                        "fold", "first_reactable_session", "exit_date",
                        "actual", "predicted", "probability",
                    )
                },
            }))

    body = {
        "artifact_version": FROZEN_ARTIFACT_VERSION,
        "status": STATUS_FROZEN,
        "study": {
            "protocol_version": specification["protocol_version"],
            "protocol_status": specification["status"],
            "protocol_hash": protocol_document["protocol_hash"],
            "protocol_specification": specification,
        },
        "provenance": {
            "code_commit": provenance.get("code_commit"),
            "database_snapshot": provenance.get("database_snapshot"),
            "dataset_version": provenance.get("dataset_version"),
            "feature_version": provenance.get("feature_version"),
            "target_version": provenance.get("target_version"),
            "modelling_unit_version": provenance.get("modelling_unit_version"),
            "timing_rule_version": provenance.get("timing_rule_version"),
            "return_window_version": provenance.get("return_window_version"),
            "validation_run_id": validation_run_id,
        },
        "sample": _round(counts),
        "folds": _round(list(folds)),
        "specification_results": results,
        "out_of_sample_predictions": predictions,
        "sensitivities": _round(list(sensitivities)),
        "success_criteria": specification["success_criteria"],
        "decision_thresholds": specification["decision_thresholds"],
        "verdict": comparison.get("verdict"),
        "specifications_run": comparison.get("specifications_run"),
        "specifications_blocked": comparison.get("specifications_blocked"),
        "successes": comparison.get("successes"),
        "comparisons": _round(list(comparison.get("comparisons") or [])),
        "multiplicity_note": comparison.get("multiplicity_note"),
        "fold_geometry": comparison.get("fold_geometry"),
        "conclusion": RETROSPECTIVE_CONCLUSION,
        "immutability_note": (
            "This artifact records a completed study. A later protocol version "
            "that performs differently does not revise it; that is a different "
            "study with its own artifact."
        ),
        "frozen_at": frozen_at,
    }
    return {**body, "artifact_hash": artifact_hash(body)}


def canonical_json(artifact: Dict[str, Any]) -> str:
    """Stable serialisation used for hashing and for the committed file."""

    payload = {k: v for k, v in artifact.items()
               if k not in ("artifact_hash", "frozen_at")}
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    )


def artifact_hash(artifact: Dict[str, Any]) -> str:
    """SHA-256 of the artifact excluding its own hash and freeze timestamp.

    ``frozen_at`` is excluded so re-running the freeze on the same study proves
    identity rather than producing a new hash every time it is checked.
    """

    return hashlib.sha256(canonical_json(artifact).encode("utf-8")).hexdigest()


def verify_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Check a stored artifact still hashes to what it claims."""

    recomputed = artifact_hash(artifact)
    stored = artifact.get("artifact_hash")
    return {
        "intact": recomputed == stored,
        "stored_hash": stored,
        "recomputed_hash": recomputed,
        "conclusion_verbatim": artifact.get("conclusion") == RETROSPECTIVE_CONCLUSION,
        "predictions": len(artifact.get("out_of_sample_predictions") or []),
        "specifications": len(artifact.get("specification_results") or []),
    }


def summary(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """The handful of numbers a reader needs, without the full payload."""

    results = artifact.get("specification_results") or []
    fitted = [r for r in results if r["status"] == "fitted"]
    news = [r for r in fitted if r["kind"] == "news" and r.get("mae") is not None]
    baselines = [
        r for r in fitted if r["kind"] == "baseline" and r.get("mae") is not None
    ]
    best_news = min(news, key=lambda r: r["mae"], default=None)
    best_baseline = min(baselines, key=lambda r: r["mae"], default=None)

    return {
        "protocol_version": artifact["study"]["protocol_version"],
        "protocol_hash": artifact["study"]["protocol_hash"],
        "artifact_hash": artifact.get("artifact_hash"),
        "code_commit": artifact["provenance"]["code_commit"],
        "database_snapshot": artifact["provenance"]["database_snapshot"],
        "independent_sessions": artifact["sample"].get("distinct_sessions"),
        "event_rows": artifact["sample"].get("event_rows"),
        "distinct_outcomes": artifact["sample"].get("distinct_outcomes"),
        "duplication_factor": artifact["sample"].get("duplication_factor"),
        "specifications_run": artifact.get("specifications_run"),
        "specifications_blocked": artifact.get("specifications_blocked"),
        "successful_news_specifications": artifact.get("successes"),
        "best_news": (
            {k: best_news[k] for k in (
                "feature_set", "model", "target", "mae", "directional_accuracy",
                "hit_rate_ci_lower", "hit_rate_ci_upper",
            )} if best_news else None
        ),
        "best_baseline": (
            {k: best_baseline[k] for k in (
                "feature_set", "model", "target", "mae", "directional_accuracy",
            )} if best_baseline else None
        ),
        "verdict": artifact.get("verdict"),
        "conclusion": artifact.get("conclusion"),
    }
