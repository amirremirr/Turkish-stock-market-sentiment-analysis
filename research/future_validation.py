"""untouched_future_v1: a test on data that did not exist when it was written.

The retrospective study was honest about its own limit. Its folds ran forward
in time, but the corpus behind them had already been collected, inspected, and
used to build the very features being evaluated. No amount of chronological
discipline fixes that; the only cure is data nobody has seen.

So this module defines a boundary and then refuses to move it.

The boundary
------------
:data:`VALIDATION_START` is a timestamp strictly after the retrospective
analysis completed. An observation belongs to the untouched sample only if its
**first reactable session** begins at or after :data:`FIRST_ELIGIBLE_SESSION`.
Nothing before that line can enter; nothing after it may influence a design
decision.

What is sealed
--------------
Feature design, feature selection, model choice, hyperparameters, the target,
the thresholds and the success criteria were all fixed before the boundary and
are hashed into the protocol. This module adds the one rule the retrospective
study could not enforce: **the outcomes on the far side of the line are not
read until the minimum sample is reached.**

That is why :mod:`scripts.future_readiness` reports sessions, coverage and
missingness but never accuracy. Peeking at performance while a sample
accumulates, and stopping when it looks good, is optional-stopping — it inflates
the false-positive rate without leaving a trace in any single number.

Labelling
---------
Every row carries ``corpus_epoch``: ``retrospective`` for anything at or before
the boundary, ``untouched_future`` for anything after. The two are never pooled
silently.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

FUTURE_VALIDATION_VERSION = "untouched_future_v1"

#: The instant the retrospective study was sealed. Chosen as the first UTC
#: midnight strictly after the frozen artifact was produced, so the boundary is
#: a session boundary and not an arbitrary moment inside a trading day.
VALIDATION_START = "2026-08-08T00:00:00+00:00"

#: The earliest reaction session admissible into the untouched sample. Sessions
#: before this may have been observed while the protocol was being written.
FIRST_ELIGIBLE_SESSION = "2026-08-10"

EPOCH_RETROSPECTIVE = "retrospective"
EPOCH_UNTOUCHED = "untouched_future"

#: Minimum independent reaction sessions before the frozen evaluation may run.
#: Set from the protocol's own primary geometry (40 train + 1 embargo + 10
#: test), not from a guess about how long that will take.
MINIMUM_SESSIONS = 51

#: The evaluation may not run before this many calendar days have passed, even
#: if sessions accumulate faster than expected. A regime lasting six weeks is
#: not a test of a general relationship.
MINIMUM_HORIZON_DAYS = 120

#: Below this many distinct outcomes the sample is a handful of days wearing a
#: larger row count. Enforced separately from session count because a session
#: with no eligible event contributes nothing.
MINIMUM_DISTINCT_OUTCOMES = 51

STATE_ACCUMULATING = "accumulating"
STATE_ELIGIBLE = "eligible_to_run"
STATE_COMPLETED = "completed"


def definition(
    *,
    protocol_hash: str,
    frozen_artifact_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """The complete, hashable future-validation definition."""

    from research.controls import CONTROL_SET_VERSION, DEFAULT_MIN_OBSERVATIONS
    from research.dataset import DATASET_VERSION
    from research.modelling_unit import MODELLING_UNIT_VERSION
    from research.protocol import (
        FEATURE_SETS, FEATURE_VERSION, MODELS, TARGET_VERSION, _spec,
    )
    from research.return_windows import PRIMARY_WINDOW, RETURN_WINDOW_VERSION
    from research.timing import TIMING_RULE_VERSION, TRADABLE_BUCKETS

    specification = _spec()
    geometry = specification["folds"]["geometries"]["primary"]

    return {
        "version": FUTURE_VALIDATION_VERSION,
        "kind": "untouched_future_validation",
        "validation_start": VALIDATION_START,
        "first_eligible_session": FIRST_ELIGIBLE_SESSION,
        "boundary_rule": (
            "an observation is untouched only if its first reactable session is "
            "at or after first_eligible_session"
        ),

        "protocol_hash": protocol_hash,
        "frozen_retrospective_artifact_hash": frozen_artifact_hash,

        "allowed_feature_versions": {
            "feature_version": FEATURE_VERSION,
            "dataset_version": DATASET_VERSION,
            "modelling_unit_version": MODELLING_UNIT_VERSION,
            "timing_rule_version": TIMING_RULE_VERSION,
            "return_window_version": RETURN_WINDOW_VERSION,
            "control_set_version": CONTROL_SET_VERSION,
            "feature_sets": sorted(FEATURE_SETS),
        },
        "target": {
            "window": PRIMARY_WINDOW,
            "column": "raw_return",
            "version": TARGET_VERSION,
            "definition": (
                "percent return from the open to the close of the first session "
                "able to react to the news"
            ),
            "eligible_timing_buckets": sorted(TRADABLE_BUCKETS),
            "market_recap_excluded": True,
            "timing_conflicts_excluded": True,
        },
        "models": MODELS,
        "hyperparameters": {
            name: model["hyperparameters"] for name, model in MODELS.items()
        },
        "thresholds": specification["decision_thresholds"],
        "sample_size_requirement": {
            "minimum_sessions": MINIMUM_SESSIONS,
            "minimum_distinct_outcomes": MINIMUM_DISTINCT_OUTCOMES,
            "minimum_control_observations": DEFAULT_MIN_OBSERVATIONS,
            "derived_from": (
                "primary fold geometry: "
                f"{geometry['initial_train_sessions']} train + "
                f"{specification['folds']['embargo_sessions']} embargo + "
                f"{geometry['test_sessions']} test"
            ),
        },
        "minimum_evaluation_horizon_days": MINIMUM_HORIZON_DAYS,
        "fold_geometry": {"name": "primary", **geometry},
        "success_criteria": specification["success_criteria"],

        "sealed": [
            "feature design",
            "feature selection",
            "model selection",
            "hyperparameters",
            "target choice",
            "decision thresholds",
            "success criteria",
        ],
        "sealing_rule": (
            "No observation on or after first_eligible_session may influence any "
            "sealed item. Outcome statistics for the untouched sample are not "
            "computed or displayed until the sample-size requirement is met; "
            "inspecting performance while a sample accumulates and stopping when "
            "it looks favourable is optional-stopping, which inflates the "
            "false-positive rate invisibly."
        ),
        "on_failure": (
            "A failed future validation is reported as a failed future "
            "validation. It does not license a revised protocol presented as "
            "the same test."
        ),
    }


def definition_hash(document: Dict[str, Any]) -> str:
    """SHA-256 of the canonical definition."""

    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def corpus_epoch(
    first_reactable_session: Optional[str],
    *,
    boundary: str = FIRST_ELIGIBLE_SESSION,
) -> str:
    """Label one observation's side of the untouched boundary."""

    if not first_reactable_session:
        return EPOCH_RETROSPECTIVE
    return (
        EPOCH_UNTOUCHED if str(first_reactable_session) >= str(boundary)
        else EPOCH_RETROSPECTIVE
    )


def partition(
    rows: Sequence[Dict[str, Any]],
    *,
    boundary: str = FIRST_ELIGIBLE_SESSION,
    session_key: str = "first_reactable_session",
) -> Dict[str, List[Dict[str, Any]]]:
    """Split rows by epoch. The two sides are never pooled by accident."""

    grouped: Dict[str, List[Dict[str, Any]]] = {
        EPOCH_RETROSPECTIVE: [], EPOCH_UNTOUCHED: [],
    }
    for row in rows:
        grouped[corpus_epoch(row.get(session_key), boundary=boundary)].append(row)
    return grouped
