"""A predictive protocol frozen before any comparative result is seen.

Every choice below — target, sample, features, models, folds, metrics, missing
data, thresholds, and what counts as success — is fixed here and hashed. The
hash is what makes the freeze checkable: if a later run reports a different
hash, the protocol moved, and any comparison across the two is a comparison of
two different studies.

What this protocol is
---------------------
**Retrospective walk-forward exploration.** The folds run forward in time and
nothing in a training fold postdates its test fold, which removes look-ahead.
It does not remove the fact that this corpus was already collected, already
inspected, and already used to build features. A genuinely untouched test
requires data that did not exist when the protocol was written. The results this
protocol produces are therefore *exploratory* and are labelled that way
everywhere they appear.

Why the numbers are conservative
--------------------------------
- The primary unit is the session, not the event, because events sharing a
  session share one index return (see :mod:`research.modelling_unit`).
- Preprocessing is fitted on training folds only, applied to test folds.
- No feature is selected using test results; the feature sets are enumerated
  here, in advance.
- Contemporaneous controls are excluded from every tradable specification and
  are available only as declared descriptive sensitivities.
- A specification with too few observations is **not fitted**. It is recorded as
  ``insufficient_sample`` with the exact binding requirement, because a model
  fitted on 12 sessions is not a weaker result, it is a different kind of
  object.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from research.controls import CONTROL_SETS, DEFAULT_MIN_OBSERVATIONS
from research.dataset import DATASET_VERSION
from research.modelling_unit import (
    AGGREGATED_FEATURES, DEFAULT_EMBARGO_SESSIONS, MODELLING_UNIT_VERSION,
    UNIT_SESSION,
)
from research.return_windows import (
    PRIMARY_WINDOW, RETURN_WINDOW_VERSION, WINDOW_PRIOR_CLOSE_TO_CLOSE,
    WINDOW_PRIOR_CLOSE_TO_OPEN,
)
from research.timing import TIMING_RULE_VERSION, TRADABLE_BUCKETS

PROTOCOL_VERSION = "walk-forward-protocol-v1"
PROTOCOL_STATUS = "retrospective_walk_forward_exploration"
FEATURE_VERSION = "session-features-v1"
TARGET_VERSION = "reactable-open-to-close-v1"

# ---------------------------------------------------------------------------
# Feature sets. Enumerated in advance; nothing is added after seeing a result.
# ---------------------------------------------------------------------------
FEATURE_SETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "features": (),
        "description": "Intercept only; the unconditional baseline.",
        "kind": "baseline",
    },
    "previous_direction": {
        "features": ("prev_return_sign",),
        "description": "Sign of the previous session's target return.",
        "kind": "baseline",
    },
    "ar1": {
        "features": ("prev_return",),
        "description": "Previous session's target return (AR(1)).",
        "kind": "baseline",
    },
    "headline_count_only": {
        "features": ("headline_count",),
        "description": "Attention proxy with no tone information.",
        "kind": "baseline",
    },
    "net_tone_share": {
        "features": ("net_tone_share",),
        "description": "Positive minus negative share of events.",
        "kind": "baseline",
    },
    "market_controls_only": {
        "features": ("eem_lag1", "brent_lag1", "usdtry_lag1"),
        "description": "Lagged market factors, no news at all.",
        "kind": "baseline",
    },
    "family_signals": {
        "features": (
            "mean_tone", "breadth_weighted_tone", "tone_dispersion",
            "positive_share",
        ),
        "description": "Phase A family-level tone aggregates.",
        "kind": "news",
    },
    "abnormal_tone": {
        "features": ("abnormal_tone", "abnormal_tone_domestic"),
        "description": "Prior-only standardised tone surprise.",
        "kind": "news",
    },
    "disagreement": {
        "features": ("mean_cross_source_dispersion", "tone_dispersion"),
        "description": "Cross-outlet disagreement about the same events.",
        "kind": "news",
    },
    "attention_shock": {
        "features": ("headline_count", "source_breadth", "event_count"),
        "description": "Volume and breadth of coverage.",
        "kind": "news",
    },
    "event_tone_novelty": {
        "features": (
            "breadth_weighted_tone", "max_novelty", "multi_source_events",
        ),
        "description": "Event-level tone with entity novelty and corroboration.",
        "kind": "news",
    },
    "controls_plus_news": {
        "features": (
            "eem_lag1", "brent_lag1", "usdtry_lag1",
            "breadth_weighted_tone", "headline_count", "net_tone_share",
        ),
        "description": "Lagged market controls together with news features.",
        "kind": "news",
    },
}

BASELINE_SETS = tuple(
    name for name, spec in FEATURE_SETS.items() if spec["kind"] == "baseline"
)
NEWS_SETS = tuple(
    name for name, spec in FEATURE_SETS.items() if spec["kind"] == "news"
)

# ---------------------------------------------------------------------------
# Models. Regularised linear and logistic only, hyperparameters fixed here.
# ---------------------------------------------------------------------------
MODELS: Dict[str, Dict[str, Any]] = {
    "mean": {
        "family": "constant",
        "task": "regression",
        "hyperparameters": {},
        "description": "Training-fold mean; the null a model must beat.",
    },
    "ridge": {
        "family": "linear",
        "task": "regression",
        # Fixed in advance. Tuning alpha per fold on test data is the leak this
        # value exists to avoid; tuning it on training data is defensible but
        # would still be a choice made against this sample.
        "hyperparameters": {"alpha": 1.0, "standardise": True},
        "description": "L2-regularised least squares on standardised features.",
    },
    "logistic": {
        "family": "logistic",
        "task": "classification",
        "hyperparameters": {"l2": 1.0, "max_iterations": 200,
                            "learning_rate": 0.1, "standardise": True},
        "description": "L2-regularised logistic regression on the return sign.",
    },
    "majority": {
        "family": "constant",
        "task": "classification",
        "hyperparameters": {},
        "description": "Training-fold majority direction.",
    },
}


# Fold geometries, frozen. Applicability thresholds are derived in _spec() so a
# geometry can never be declared to apply at a sample size it cannot fit.
_GEOMETRIES: Dict[str, Dict[str, Any]] = {
    "primary": {
        "initial_train_sessions": 40,
        "test_sessions": 10,
        "step_sessions": 10,
        "minimum_test_sessions_per_fold": 8,
        "can_declare_success": True,
    },
    "reduced": {
        "initial_train_sessions": 25,
        "test_sessions": 6,
        "step_sessions": 6,
        "minimum_test_sessions_per_fold": 5,
        "can_declare_success": False,
        "reason": (
            "25 training sessions against up to six features is roughly four "
            "observations per parameter; the design can detect nothing it "
            "should be believed about, so its verdict is capped at inconclusive"
        ),
    },
}


def _spec() -> Dict[str, Any]:
    """The complete frozen specification, as a plain JSON-safe object."""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": PROTOCOL_STATUS,
        "status_note": (
            "Retrospective walk-forward exploration on already-collected data. "
            "Not an untouched future test."
        ),

        "target": {
            "primary": {
                "window": PRIMARY_WINDOW,
                "column": "raw_return",
                "definition": (
                    "percent return from the open to the close of the first "
                    "session able to react to the news"
                ),
                "version": TARGET_VERSION,
            },
            "secondary": [
                # residual_none is deliberately absent: for the empty control
                # set the residual *is* the raw return, so running it would add
                # a duplicate specification to the multiplicity count and no
                # information to the study.
                {"window": PRIMARY_WINDOW, "column": "residual_em_lagged"},
                {"window": PRIMARY_WINDOW, "column": "residual_em_oil_fx_lagged"},
                {"window": WINDOW_PRIOR_CLOSE_TO_OPEN, "column": "raw_return",
                 "tradable": False},
                {"window": WINDOW_PRIOR_CLOSE_TO_CLOSE, "column": "raw_return",
                 "tradable": False},
            ],
            "direction_label": "sign of the primary target, zero excluded",
        },

        "sample": {
            "modelling_unit": UNIT_SESSION,
            "modelling_unit_version": MODELLING_UNIT_VERSION,
            "eligible_timing_buckets": sorted(TRADABLE_BUCKETS),
            "excluded_timing_buckets": ["during_session", "unknown"],
            "market_recap_excluded": True,
            "market_recap_reason": (
                "recap tone follows the return by construction"
            ),
            "timing_conflicts_excluded_from_primary": True,
            "minimum_sources_per_event": 1,
            "minimum_sources_sensitivity": 2,
            "minimum_price_bar_status": ["complete", "corrected"],
            "minimum_control_history_sessions": DEFAULT_MIN_OBSERVATIONS,
            "minimum_events_per_session": 1,
        },

        "features": {
            "version": FEATURE_VERSION,
            "aggregated_from_events_by": list(AGGREGATED_FEATURES),
            "sets": {
                name: {
                    "features": list(spec["features"]),
                    "kind": spec["kind"],
                    "description": spec["description"],
                }
                for name, spec in FEATURE_SETS.items()
            },
            "standardisation": "training fold only, applied to test fold",
            "contemporaneous_controls_in_tradable_models": False,
        },

        "controls": {
            "version": "control-sets-v1",
            "tradable_sets": sorted(
                name for name, spec in CONTROL_SETS.items()
                if spec["kind"] == "tradable"
            ),
            "descriptive_sets": sorted(
                name for name, spec in CONTROL_SETS.items()
                if spec["kind"] != "tradable"
            ),
            "estimation": "rolling prior window, 60 sessions, minimum 30",
        },

        "models": MODELS,
        "baseline_feature_sets": list(BASELINE_SETS),
        "news_feature_sets": list(NEWS_SETS),

        "folds": {
            "design": "expanding-window walk-forward, chronological",
            # Two geometries, both fixed here, selected by the number of
            # available sessions and nothing else. Sample size is a property of
            # the data collection, knowable and known before any target was
            # read; selecting on it is not selecting on a result. The reduced
            # geometry exists so a small sample produces a stated null instead
            # of an empty report -- and it is barred from declaring success.
            "selection_rule": "by session count, before any model is fitted",
            # Each geometry's applicability threshold is *derived* from its own
            # parameters, never written down beside them. A hard-coded 50 against
            # 40 training + 1 embargo + 10 test is a geometry that applies at a
            # sample size where it cannot produce a single fold -- which is what
            # it did, on the first production run.
            "geometries": {
                name: {
                    **geometry,
                    "applies_when_sessions_at_least": (
                        geometry["initial_train_sessions"]
                        + DEFAULT_EMBARGO_SESSIONS
                        + geometry["test_sessions"]
                    ),
                }
                for name, geometry in _GEOMETRIES.items()
            },
            "below_minimum_sessions": (
                _GEOMETRIES["reduced"]["initial_train_sessions"]
                + DEFAULT_EMBARGO_SESSIONS
                + _GEOMETRIES["reduced"]["test_sessions"]
            ),
            "below_minimum_action": "report insufficient_sample for everything",
            "embargo_sessions": DEFAULT_EMBARGO_SESSIONS,
            "embargo_rationale": (
                "drops the session adjacent to each test fold so a story "
                "spanning two sessions cannot appear on both sides"
            ),
            "random_splitting": False,
            "shuffling": False,
        },

        "metrics": {
            "regression": ["mae", "rmse", "pearson_r", "r2_vs_train_mean"],
            "classification": [
                "directional_accuracy", "balanced_accuracy", "brier_score",
            ],
            "uncertainty": "session-cluster bootstrap, 2000 resamples",
            "reported_per": ["fold", "signal_family", "timing_bucket", "regime"],
        },

        "missing_values": {
            "policy": "drop the row for that specification; never impute zero",
            "rationale": (
                "a zero feature asserts neutrality that was never measured"
            ),
            "reported": "coverage and missingness per specification",
        },

        "decision_thresholds": {
            "direction_threshold": 0.0,
            "minimum_improvement_over_best_baseline_mae": 0.05,
            "minimum_directional_accuracy_over_majority": 0.05,
            "alpha": 0.05,
            "multiplicity": (
                "no specification is declared significant without an explicit "
                "correction for the number of specifications run"
            ),
        },

        "success_criteria": {
            "success": (
                "a news feature set beats every baseline on MAE and directional "
                "accuracy by the stated margins, in a majority of folds, with "
                "session-cluster-aware intervals excluding the baseline"
            ),
            "failure": (
                "no news feature set clears the margins, or the sample gate "
                "blocks the comparison"
            ),
            "inconclusive": (
                "margins cleared in fewer than a majority of folds, or fewer "
                "than three folds are fittable"
            ),
            "reporting_rule": (
                "failure and inconclusive results are reported in full; the "
                "protocol is not re-run with different settings to obtain a "
                "different answer"
            ),
        },

        "versions": {
            "timing_rule": TIMING_RULE_VERSION,
            "return_window": RETURN_WINDOW_VERSION,
            "dataset": DATASET_VERSION,
            "feature": FEATURE_VERSION,
            "target": TARGET_VERSION,
            "modelling_unit": MODELLING_UNIT_VERSION,
        },

        "prohibited": [
            "random train/test splitting",
            "full-sample normalisation",
            "fitting preprocessing on a test fold",
            "feature selection using test results",
            "contemporaneous controls in a tradable specification",
            "reporting significance without accounting for repeated sessions",
            "trading-strategy or transaction-cost evaluation",
        ],
    }


def select_geometry(session_count: int) -> Dict[str, Any]:
    """Pick the frozen fold geometry for a given number of sessions.

    The only input is the session count. No metric, target or model result is
    consulted, and the choice is made before the first fit.
    """

    geometries = _spec()["folds"]["geometries"]
    for name in ("primary", "reduced"):
        geometry = geometries[name]
        if session_count >= geometry["applies_when_sessions_at_least"]:
            return {"name": name, **geometry}
    return {
        "name": "none",
        "applies_when_sessions_at_least": None,
        "initial_train_sessions": None,
        "test_sessions": None,
        "step_sessions": None,
        "minimum_test_sessions_per_fold": None,
        "can_declare_success": False,
        "reason": (
            f"{session_count} sessions is below the minimum of "
            f"{_spec()['folds']['below_minimum_sessions']}"
        ),
    }


def protocol_document(
    *,
    code_commit: Optional[str] = None,
    database_snapshot: Optional[str] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """The frozen protocol plus its provenance.

    Provenance fields sit *outside* the hashed specification on purpose: the
    protocol is the same protocol whether it runs against yesterday's snapshot
    or today's, and a hash that changed with the database could not certify
    that the rules had stayed put.
    """

    specification = _spec()
    return {
        "specification": specification,
        "protocol_hash": protocol_hash(specification),
        "provenance": {
            "code_commit": code_commit,
            "database_snapshot": database_snapshot,
            "generated_at": generated_at,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "target_version": TARGET_VERSION,
            "modelling_unit_version": MODELLING_UNIT_VERSION,
            "timing_rule_version": TIMING_RULE_VERSION,
            "return_window_version": RETURN_WINDOW_VERSION,
        },
    }


def canonical_json(specification: Dict[str, Any]) -> str:
    """Stable serialisation: sorted keys, no incidental whitespace."""

    return json.dumps(
        specification, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def protocol_hash(specification: Optional[Dict[str, Any]] = None) -> str:
    """SHA-256 of the canonical specification."""

    payload = canonical_json(specification if specification is not None else _spec())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_set(name: str) -> List[str]:
    """The frozen feature list for *name*."""

    if name not in FEATURE_SETS:
        raise KeyError(f"unknown feature set: {name!r}")
    return list(FEATURE_SETS[name]["features"])


def is_tradable_specification(feature_set_name: str, target_column: str) -> bool:
    """Whether a specification may be described as execution-relevant."""

    if "contemporaneous" in target_column:
        return False
    return all(
        "contemporaneous" not in feature
        for feature in FEATURE_SETS[feature_set_name]["features"]
    )
