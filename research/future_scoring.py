"""Score untouched_future_v1 exactly as its sealed success criteria are worded.

Written 2026-10-08, while the untouched sample was still accumulating and before
any outcome on it had been computed. The git history of this file is the proof
of that ordering; docs/FUTURE_VALIDATION_SCORING.md records the same rules in
prose.

Why this module exists
----------------------
The retrospective study scored itself with
:func:`research.walkforward.compare_to_baselines`, which pools all folds and
declares success if any news specification clears the margins on the pooled
predictions. The sealed definition says something stricter:

    success       a news feature set beats every baseline on MAE and directional
                  accuracy by the stated margins, in a majority of folds, with
                  session-cluster-aware intervals excluding the baseline
    failure       no news feature set clears the margins, or the sample gate
                  blocks the comparison
    inconclusive  margins cleared in fewer than a majority of folds, or fewer
                  than three folds are fittable

plus a multiplicity rule: no specification is declared significant without an
explicit correction for the number of specifications run.

The retrospective code is left untouched: its result is frozen, and no
specification there cleared both margins, so its "failure" holds under either
reading. This module is what scores the untouched test.

Readings of the sealed text
---------------------------
Each phrase that admits more than one reading is resolved below, toward the
reading that makes success harder. None of these choices consults an outcome.

R1  "the margins": both pooled gains, measured against the best baseline on
    exactly the sessions the news specification predicted (MAE improvement
    >= 0.05, directional-accuracy improvement >= 0.05).
R2  "beats every baseline": beating the best baseline on each metric, chosen
    per metric on the matched sessions, is beating every baseline.
R3  "in a majority of folds": the margins are re-measured inside each fold, on
    that fold's matched sessions, and must be cleared in strictly more than half
    of the FITTABLE folds -- folds in which at least one baseline produced
    out-of-sample predictions, so a comparison exists. A fittable fold the news
    specification could not predict counts against it. (The first primary fold
    is never fittable: the one-session embargo leaves 39 training sessions
    against the 40-session minimum. Counting it would penalise every
    specification for a property of the geometry, not of the data.)
R4  "fewer than three folds are fittable": if fewer than three folds are
    fittable in the R3 sense, the verdict is inconclusive whatever the margins
    show. A specification with fewer than three fitted folds cannot succeed.
R5  "session-cluster-aware intervals excluding the baseline": paired bootstrap
    intervals, resampling sessions (one row per session, clustered on exit
    date), for (best-MAE baseline absolute error - news absolute error) and
    (news hit - best-accuracy baseline hit). Both lower bounds must be > 0.
R6  Multiplicity: the intervals are Bonferroni-corrected, two-sided at
    1 - alpha / m, where m is the number of news specifications compared.
R7  Precedence, resolving the overlap between "failure" and "inconclusive":
      1. no fitted news specification, or no fitted baseline -> failure
         (the sample gate blocks the comparison)
      2. fewer than three fittable folds                        -> inconclusive
      3. any specification meets R1 + R3 + R4 + R5/R6          -> success
      4. any specification clears the pooled margins (R1)      -> inconclusive
      5. otherwise                                             -> failure
    Step 4 is how "margins cleared in fewer than a majority of folds" is read:
    the margins were cleared overall but not consistently, or not beyond the
    corrected interval. Reading it as "cleared in zero or more folds" would make
    failure impossible, which cannot be what the text intends.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from research.walkforward import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, FITTED, Fold, _restrict,
)

SCORING_VERSION = "untouched-future-scoring-v1"
MINIMUM_FOLDS = 3
#: Matched sessions a baseline needs before it may stand as "the baseline";
#: the same floor compare_to_baselines uses.
MINIMUM_MATCHED = 3


def _best(
    baselines: Sequence[Dict[str, Any]], sessions: set, metric: str, *, lower: bool,
) -> Optional[Dict[str, Any]]:
    """The best baseline on *metric*, re-scored on exactly *sessions*."""

    scored = []
    for base in baselines:
        restricted = _restrict(base, sessions)
        if (
            restricted and restricted["n_predictions"] >= MINIMUM_MATCHED
            and restricted.get(metric) is not None
        ):
            scored.append((restricted[metric], base, restricted))
    if not scored:
        return None
    value, base, restricted = (min if lower else max)(scored, key=lambda t: t[0])
    return {"spec": base, "value": value, "metrics": restricted}


def _gains(
    spec: Dict[str, Any], baselines: Sequence[Dict[str, Any]], sessions: set,
) -> Dict[str, Any]:
    """Margins of *spec* over the best baselines on *sessions* (R1, R2)."""

    own = _restrict(spec, sessions)
    if not own:
        return {"evaluable": False}
    best_mae = _best(baselines, sessions, "mae", lower=True)
    best_acc = _best(baselines, sessions, "directional_accuracy", lower=False)
    mae_gain = (
        best_mae["value"] - own["mae"]
        if best_mae and own.get("mae") is not None else None
    )
    acc_gain = (
        own["directional_accuracy"] - best_acc["value"]
        if best_acc and own.get("directional_accuracy") is not None else None
    )
    return {
        "evaluable": mae_gain is not None and acc_gain is not None,
        "sessions": own["n_predictions"],
        "mae": own.get("mae"),
        "directional_accuracy": own.get("directional_accuracy"),
        "best_baseline_mae": best_mae["value"] if best_mae else None,
        "best_baseline_mae_feature_set": (
            f"{best_mae['spec']['feature_set']}/{best_mae['spec']['model']}"
            if best_mae else None
        ),
        "best_baseline_accuracy": best_acc["value"] if best_acc else None,
        "best_baseline_accuracy_feature_set": (
            f"{best_acc['spec']['feature_set']}/{best_acc['spec']['model']}"
            if best_acc else None
        ),
        "mae_gain": mae_gain,
        "accuracy_gain": acc_gain,
        "_best_mae_spec": best_mae["spec"] if best_mae else None,
        "_best_acc_spec": best_acc["spec"] if best_acc else None,
    }


def _clears(gains: Dict[str, Any], thresholds: Dict[str, Any]) -> bool:
    return bool(
        gains.get("evaluable")
        and gains["mae_gain"] >= thresholds["minimum_improvement_over_best_baseline_mae"]
        and gains["accuracy_gain"]
        >= thresholds["minimum_directional_accuracy_over_majority"]
    )


def paired_interval(
    differences: Sequence[float],
    clusters: Sequence[Any],
    *,
    level: float,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Percentile interval for a mean paired difference, resampling clusters (R5)."""

    import numpy as np

    grouped: Dict[Any, List[float]] = {}
    for value, cluster in zip(differences, clusters):
        grouped.setdefault(cluster, []).append(float(value))
    keys = sorted(grouped, key=str)
    result = {
        "mean": (sum(differences) / len(differences)) if differences else None,
        "lower": None, "upper": None, "clusters": len(keys), "level": level,
        "resamples": resamples,
    }
    if len(keys) < 2:
        return result
    rng = np.random.RandomState(seed)
    sums = np.array([sum(grouped[k]) for k in keys])
    counts = np.array([len(grouped[k]) for k in keys])
    picks = rng.randint(0, len(keys), size=(resamples, len(keys)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    result["lower"] = float(np.percentile(means, tail))
    result["upper"] = float(np.percentile(means, 100.0 - tail))
    return result


def _by_session(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(p["first_reactable_session"]): p for p in spec.get("predictions") or []}


def _intervals(
    spec: Dict[str, Any], gains: Dict[str, Any], *, level: float,
) -> Dict[str, Any]:
    """Paired, session-clustered intervals against the best baselines (R5, R6)."""

    own = _by_session(spec)
    out: Dict[str, Any] = {}

    base = gains.get("_best_mae_spec")
    if base is not None:
        other = _by_session(base)
        shared = sorted(set(own) & set(other))
        out["mae_gain"] = paired_interval(
            [abs(other[s]["actual"] - other[s]["predicted"])
             - abs(own[s]["actual"] - own[s]["predicted"]) for s in shared],
            [own[s].get("exit_date") or s for s in shared], level=level,
        )

    base = gains.get("_best_acc_spec")
    if base is not None:
        other = _by_session(base)
        shared = sorted(
            s for s in set(own) & set(other) if own[s]["actual"] != 0
        )

        def hit(p):
            return 1.0 if (p["actual"] > 0) == (p["predicted"] > 0) else 0.0

        out["accuracy_gain"] = paired_interval(
            [hit(own[s]) - hit(other[s]) for s in shared],
            [own[s].get("exit_date") or s for s in shared], level=level,
        )

    out["excludes_baseline"] = bool(
        all(
            out.get(name, {}).get("lower") is not None and out[name]["lower"] > 0
            for name in ("mae_gain", "accuracy_gain")
        )
    )
    return out


def score(
    specifications: Sequence[Dict[str, Any]],
    folds: Sequence[Fold],
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply R1-R7 to evaluated specifications. Pure: no I/O, no randomness
    beyond the fixed bootstrap seed."""

    fitted = [s for s in specifications if s.get("status") == FITTED and s.get("predictions")]
    baselines = [s for s in fitted if s["kind"] == "baseline"]
    news = [s for s in fitted if s["kind"] == "news"]
    m = len(news)
    baseline_sessions = {
        str(p["first_reactable_session"]) for base in baselines for p in base["predictions"]
    }
    fittable = [
        fold for fold in folds if baseline_sessions & {str(s) for s in fold.test}
    ]
    alpha = float(thresholds["alpha"])
    level = 1.0 - alpha / m if m else 1.0 - alpha

    rows: List[Dict[str, Any]] = []
    for spec in news:
        covered = {str(p["first_reactable_session"]) for p in spec["predictions"]}
        pooled = _gains(spec, baselines, covered)
        clears_pooled = _clears(pooled, thresholds)

        per_fold = []
        for fold in fittable:
            sessions = covered & {str(s) for s in fold.test}
            fold_gains = _gains(spec, baselines, sessions) if sessions else {"evaluable": False}
            per_fold.append({
                "fold": fold.index,
                "sessions": len(sessions),
                "mae_gain": fold_gains.get("mae_gain"),
                "accuracy_gain": fold_gains.get("accuracy_gain"),
                "clears_margins": _clears(fold_gains, thresholds),
            })
        fitted_folds = sum(1 for f in per_fold if f["sessions"] > 0)
        cleared = sum(1 for f in per_fold if f["clears_margins"])
        majority = cleared * 2 > len(fittable)

        intervals = _intervals(spec, pooled, level=level) if pooled.get("evaluable") else {
            "excludes_baseline": False,
        }
        success = bool(
            clears_pooled and majority and fitted_folds >= MINIMUM_FOLDS
            and intervals["excludes_baseline"]
        )
        rows.append({
            "feature_set": spec["feature_set"],
            "model": spec["model"],
            "target": spec["target"],
            "pooled": {k: v for k, v in pooled.items() if not k.startswith("_")},
            "clears_pooled_margins": clears_pooled,
            "folds": per_fold,
            "fitted_folds": fitted_folds,
            "folds_cleared": cleared,
            "majority_of_folds": majority,
            "intervals": intervals,
            "meets_success_criteria": success,
        })

    if not news or not baselines:
        verdict, reason = "failure", (
            "the sample gate blocked the comparison: "
            f"{len(news)} news and {len(baselines)} baseline specification(s) fitted"
        )
    elif len(fittable) < MINIMUM_FOLDS:
        verdict, reason = "inconclusive", (
            f"only {len(fittable)} of {len(folds)} fold(s) are fittable; the "
            f"criteria need at least {MINIMUM_FOLDS}"
        )
    elif any(r["meets_success_criteria"] for r in rows):
        winners = [f"{r['feature_set']}/{r['model']}" for r in rows if r["meets_success_criteria"]]
        verdict, reason = "success", (
            f"{', '.join(winners)} cleared both margins pooled and in a majority "
            f"of {len(fittable)} fittable folds, with Bonferroni-corrected intervals "
            f"(level {level:.4f}, m={m}) excluding the baseline"
        )
    elif any(r["clears_pooled_margins"] for r in rows):
        verdict, reason = "inconclusive", (
            "at least one news specification cleared the pooled margins, but "
            "not in a majority of folds or not beyond the corrected interval"
        )
    else:
        verdict, reason = "failure", (
            f"none of {m} news specifications cleared both margins"
        )

    return {
        "scoring_version": SCORING_VERSION,
        "verdict": verdict,
        "verdict_reason": reason,
        "folds_built": len(folds),
        "folds_fittable": len(fittable),
        "news_specifications": m,
        "baseline_specifications": len(baselines),
        "specifications_blocked": len(specifications) - len(fitted),
        "interval_level": level,
        "comparisons": rows,
    }


def result_hash(result: Dict[str, Any]) -> str:
    payload = json.dumps(
        {k: v for k, v in result.items() if k != "result_hash"},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
