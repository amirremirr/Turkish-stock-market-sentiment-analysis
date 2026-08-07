"""Chronological walk-forward evaluation under the frozen protocol.

Nothing here chooses anything. The target, the sample, the feature sets, the
models, the fold geometry and the thresholds all come from
:mod:`research.protocol`; this module executes them and reports what happened,
including when the honest answer is "not enough data to try".

Three properties are enforced structurally rather than by convention, because
each is a leak that a careful person still makes by accident:

**Folds never look forward.** Folds are built from an ordered session list by
index. A training fold is a prefix; a test fold follows it after an embargo gap.
There is no shuffling and no random split anywhere in this module.

**Preprocessing is fitted on training folds only.** The standardiser records the
training mean and standard deviation and applies them to the test fold. Fitting
it on the pooled sample would leak the test distribution into training — a small
effect that reliably flatters results on samples this size.

**A specification that cannot be fitted is not fitted.** The sample gate runs
*before* any model sees data and records ``insufficient_sample`` with the exact
binding requirement. Fitting a ridge on twelve sessions and reporting the number
would be worse than reporting nothing, because the number would be quoted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from research.protocol import FEATURE_SETS, MODELS, _spec

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260806

INSUFFICIENT = "insufficient_sample"
FITTED = "fitted"


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    """One chronological train/test split with an embargo between them."""

    index: int
    train: Tuple[str, ...]
    test: Tuple[str, ...]
    embargoed: Tuple[str, ...]

    def as_row(self) -> Dict[str, Any]:
        return {
            "fold": self.index,
            "train_sessions": len(self.train),
            "test_sessions": len(self.test),
            "embargoed_sessions": len(self.embargoed),
            "train_start": self.train[0] if self.train else None,
            "train_end": self.train[-1] if self.train else None,
            "test_start": self.test[0] if self.test else None,
            "test_end": self.test[-1] if self.test else None,
        }


def build_folds(
    sessions: Sequence[str],
    *,
    initial_train: int,
    test_size: int,
    step: int,
    embargo: int = 1,
) -> List[Fold]:
    """Expanding-window folds over an ordered session list.

    ``embargo`` sessions immediately before each test fold are removed from
    training. A one-row-per-session design already prevents an event from
    appearing on both sides; the embargo covers the remaining path, a story that
    spans two adjacent sessions.
    """

    ordered = sorted(set(str(session) for session in sessions))
    folds: List[Fold] = []
    index = 0
    train_end = initial_train

    while train_end + test_size <= len(ordered):
        cut = max(0, train_end - embargo)
        train = tuple(ordered[:cut])
        embargoed = tuple(ordered[cut:train_end])
        test = tuple(ordered[train_end:train_end + test_size])
        if train and test:
            index += 1
            folds.append(Fold(index=index, train=train, test=test,
                              embargoed=embargoed))
        train_end += step

    return folds


def fold_boundaries_are_safe(folds: Sequence[Fold]) -> Dict[str, Any]:
    """Assert the properties a fold design must have, and report violations."""

    violations: List[str] = []
    for fold in folds:
        if set(fold.train) & set(fold.test):
            violations.append(f"fold {fold.index}: train and test overlap")
        if fold.train and fold.test and max(fold.train) >= min(fold.test):
            violations.append(f"fold {fold.index}: training session at or after test start")
        if set(fold.embargoed) & set(fold.train):
            violations.append(f"fold {fold.index}: embargoed session left in training")
        if set(fold.embargoed) & set(fold.test):
            violations.append(f"fold {fold.index}: embargoed session leaked into test")
    return {"safe": not violations, "violations": violations,
            "fold_count": len(folds)}


# ---------------------------------------------------------------------------
# Preprocessing and models (numpy only; no scikit-learn by design)
# ---------------------------------------------------------------------------
@dataclass
class Standardiser:
    """Column means and deviations learned from a training fold only."""

    means: List[float] = field(default_factory=list)
    deviations: List[float] = field(default_factory=list)

    def fit(self, rows: Sequence[Sequence[float]]) -> "Standardiser":
        import numpy as np

        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        self.means = [float(v) for v in matrix.mean(axis=0)]
        deviations = matrix.std(axis=0)
        # A constant column carries no information; dividing by its zero spread
        # would produce infinities rather than a warning.
        self.deviations = [float(v) if v > 1e-12 else 1.0 for v in deviations]
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> List[List[float]]:
        return [
            [
                (float(value) - mean) / deviation
                for value, mean, deviation in zip(row, self.means, self.deviations)
            ]
            for row in rows
        ]


def _ridge_fit(
    features: Sequence[Sequence[float]], target: Sequence[float], alpha: float,
) -> Optional[List[float]]:
    """Closed-form ridge with an unpenalised intercept."""

    import numpy as np

    if not features or not target:
        return None
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    y = np.asarray(target, dtype=float)
    design = np.column_stack([np.ones(len(y)), matrix])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0                      # never shrink the intercept
    try:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:            # pragma: no cover - singular design
        return None
    return [float(value) for value in coefficients]


def _predict_linear(
    coefficients: Sequence[float], features: Sequence[Sequence[float]],
) -> List[float]:
    return [
        float(coefficients[0]) + math.fsum(
            float(beta) * float(value)
            for beta, value in zip(coefficients[1:], row)
        )
        for row in features
    ]


def _logistic_fit(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    l2: float,
    iterations: int,
    learning_rate: float,
) -> Optional[List[float]]:
    """L2-regularised logistic regression by full-batch gradient descent.

    Deterministic: fixed iteration count, fixed step, zero initialisation. No
    early stopping on a validation split, because carving one out of a sample
    this small would cost more than it buys.
    """

    import numpy as np

    if not features or not labels or len(set(labels)) < 2:
        return None
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    design = np.column_stack([np.ones(len(labels)), matrix])
    y = np.asarray(labels, dtype=float)
    weights = np.zeros(design.shape[1])

    for _ in range(int(iterations)):
        predictions = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30, 30)))
        gradient = design.T @ (predictions - y) / len(y)
        gradient[1:] += float(l2) * weights[1:] / len(y)
        weights -= float(learning_rate) * gradient
        if not np.all(np.isfinite(weights)):     # pragma: no cover
            return None
    return [float(value) for value in weights]


def _predict_probability(
    coefficients: Sequence[float], features: Sequence[Sequence[float]],
) -> List[float]:
    scores = _predict_linear(coefficients, features)
    return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score)))) for score in scores]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> Optional[float]:
    return math.fsum(values) / len(values) if values else None


def regression_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    reference: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """MAE, RMSE, correlation, and R^2 against a stated reference prediction."""

    if not actual:
        return {"n": 0, "mae": None, "rmse": None, "pearson_r": None,
                "r2_vs_reference": None}
    errors = [a - p for a, p in zip(actual, predicted)]
    mae = math.fsum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))

    correlation = None
    if len(actual) > 2:
        mean_a, mean_p = _mean(actual), _mean(predicted)
        cov = math.fsum((a - mean_a) * (p - mean_p) for a, p in zip(actual, predicted))
        var_a = math.fsum((a - mean_a) ** 2 for a in actual)
        var_p = math.fsum((p - mean_p) ** 2 for p in predicted)
        if var_a > 1e-12 and var_p > 1e-12:
            correlation = cov / math.sqrt(var_a * var_p)

    r2 = None
    if reference is not None:
        baseline_error = math.fsum((a - reference) ** 2 for a in actual)
        model_error = math.fsum(e * e for e in errors)
        if baseline_error > 1e-12:
            r2 = 1.0 - model_error / baseline_error

    return {"n": len(actual), "mae": mae, "rmse": rmse,
            "pearson_r": correlation, "r2_vs_reference": r2}


def direction_metrics(
    actual: Sequence[float], predicted: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Directional and balanced accuracy, ignoring exactly-zero outcomes.

    Balanced accuracy is reported alongside raw accuracy because a sample with
    58% up-sessions makes "always up" look like skill.
    """

    pairs = [
        (1 if a > 0 else 0, 1 if p > 0 else 0)
        for a, p in zip(actual, predicted) if a != 0
    ]
    if not pairs:
        return {"n": 0, "directional_accuracy": None,
                "balanced_accuracy": None, "positive_rate": None}

    correct = sum(1 for a, p in pairs if a == p)
    ups = [(a, p) for a, p in pairs if a == 1]
    downs = [(a, p) for a, p in pairs if a == 0]
    sensitivity = (
        sum(1 for a, p in ups if p == 1) / len(ups) if ups else None
    )
    specificity = (
        sum(1 for a, p in downs if p == 0) / len(downs) if downs else None
    )
    balanced = (
        (sensitivity + specificity) / 2.0
        if sensitivity is not None and specificity is not None else None
    )
    return {
        "n": len(pairs),
        "directional_accuracy": correct / len(pairs),
        "balanced_accuracy": balanced,
        "positive_rate": len(ups) / len(pairs),
    }


def brier_score(
    actual: Sequence[float], probabilities: Sequence[float],
) -> Optional[float]:
    pairs = [(1.0 if a > 0 else 0.0, p) for a, p in zip(actual, probabilities) if a != 0]
    if not pairs:
        return None
    return math.fsum((p - a) ** 2 for a, p in pairs) / len(pairs)


def calibration_table(
    actual: Sequence[float], probabilities: Sequence[float], bins: int = 4,
) -> List[Dict[str, Any]]:
    """Predicted vs realised positive rate, in equal-width probability bins."""

    pairs = [(1.0 if a > 0 else 0.0, p) for a, p in zip(actual, probabilities) if a != 0]
    table: List[Dict[str, Any]] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [
            (a, p) for a, p in pairs
            if (p >= low and p < high) or (index == bins - 1 and p == 1.0)
        ]
        if not bucket:
            continue
        table.append({
            "bin": f"[{low:.2f},{high:.2f})",
            "n": len(bucket),
            "mean_predicted": _mean([p for _, p in bucket]),
            "observed_rate": _mean([a for a, _ in bucket]),
        })
    return table


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[Any],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> Dict[str, Optional[float]]:
    """Percentile interval for a mean, resampling whole clusters.

    Resampling rows would treat two events on the same session as two
    independent observations of the same index return. Resampling clusters --
    here, exit sessions -- keeps the dependence intact.
    """

    import numpy as np

    if not values:
        return {"mean": None, "lower": None, "upper": None, "clusters": 0}

    grouped: Dict[Any, List[float]] = {}
    for value, cluster in zip(values, clusters):
        grouped.setdefault(cluster, []).append(float(value))
    keys = sorted(grouped, key=str)
    if len(keys) < 2:
        return {"mean": _mean(list(values)), "lower": None, "upper": None,
                "clusters": len(keys)}

    rng = np.random.RandomState(seed)
    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        picked = rng.randint(0, len(keys), size=len(keys))
        pooled = [v for position in picked for v in grouped[keys[position]]]
        means[index] = float(np.mean(pooled))

    tail = (1.0 - level) / 2.0 * 100.0
    return {
        "mean": _mean(list(values)),
        "lower": float(np.percentile(means, tail)),
        "upper": float(np.percentile(means, 100.0 - tail)),
        "clusters": len(keys),
        "resamples": resamples,
    }


# ---------------------------------------------------------------------------
# Sample-size gate
# ---------------------------------------------------------------------------
def sample_gate(
    rows: Sequence[Dict[str, Any]],
    features: Sequence[str],
    target: str,
    *,
    minimum_sessions: int,
    folds: Sequence[Fold],
    minimum_test_sessions: int,
) -> Dict[str, Any]:
    """Decide whether a specification may be fitted at all.

    Runs before any model touches the data and names the exact binding
    requirement when it refuses. Loosening a rule to raise coverage -- for
    instance dropping the 30-observation control minimum -- is not available
    here; that rule protects the residuals themselves.
    """

    complete = [
        row for row in rows
        if row.get(target) is not None
        and all(row.get(name) is not None for name in features)
    ]
    usable_sessions = {str(row["first_reactable_session"]) for row in complete}

    fittable = [
        fold for fold in folds
        if len(usable_sessions & set(fold.train)) >= minimum_sessions
        and len(usable_sessions & set(fold.test)) >= minimum_test_sessions
    ]

    binding: Optional[str] = None
    if len(complete) < minimum_sessions:
        binding = (
            f"{len(complete)} usable rows < minimum_sessions_to_fit "
            f"{minimum_sessions}"
        )
    elif not folds:
        binding = "no folds could be built from the available session range"
    elif not fittable:
        binding = (
            f"no fold has >= {minimum_sessions} training and "
            f">= {minimum_test_sessions} test sessions with complete features"
        )

    missing = {
        name: sum(1 for row in rows if row.get(name) is None)
        for name in list(features) + [target]
    }

    return {
        "status": INSUFFICIENT if binding else FITTED,
        "binding_requirement": binding,
        "rows_total": len(rows),
        "rows_complete": len(complete),
        "usable_sessions": len(usable_sessions),
        "fittable_folds": [fold.index for fold in fittable],
        "missing_by_column": missing,
        "coverage": (
            round(len(complete) / len(rows), 4) if rows else None
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _matrix(rows: Sequence[Dict[str, Any]], features: Sequence[str]):
    return [[float(row[name]) for name in features] for row in rows]


def evaluate_specification(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_set_name: str,
    model_name: str,
    target: str,
    folds: Sequence[Fold],
    minimum_sessions: int,
    minimum_test_sessions: int,
) -> Dict[str, Any]:
    """Run one (feature set, model, target) specification across all folds."""

    features = list(FEATURE_SETS[feature_set_name]["features"])
    model = MODELS[model_name]
    gate = sample_gate(
        rows, features, target, minimum_sessions=minimum_sessions,
        folds=folds, minimum_test_sessions=minimum_test_sessions,
    )

    specification = {
        "feature_set": feature_set_name,
        "model": model_name,
        "target": target,
        "task": model["task"],
        "kind": FEATURE_SETS[feature_set_name]["kind"],
        "gate": gate,
        "folds": [],
        "predictions": [],
    }
    if gate["status"] == INSUFFICIENT:
        specification["status"] = INSUFFICIENT
        return specification

    complete = [
        row for row in rows
        if row.get(target) is not None
        and all(row.get(name) is not None for name in features)
    ]
    by_session = {str(row["first_reactable_session"]): row for row in complete}

    predictions: List[Dict[str, Any]] = []
    fold_reports: List[Dict[str, Any]] = []

    for fold in folds:
        train = [by_session[s] for s in fold.train if s in by_session]
        test = [by_session[s] for s in fold.test if s in by_session]
        if len(train) < minimum_sessions or len(test) < minimum_test_sessions:
            fold_reports.append({
                **fold.as_row(), "status": INSUFFICIENT,
                "usable_train": len(train), "usable_test": len(test),
            })
            continue

        train_target = [float(row[target]) for row in train]
        test_target = [float(row[target]) for row in test]
        train_mean = _mean(train_target) or 0.0

        if model["family"] == "constant" and model["task"] == "regression":
            predicted = [train_mean] * len(test)
            probabilities = None
        elif model["family"] == "constant":
            majority = 1 if sum(1 for v in train_target if v > 0) * 2 >= len(train_target) else 0
            predicted = [1.0 if majority else -1.0] * len(test)
            probabilities = [float(majority)] * len(test)
        else:
            train_x = _matrix(train, features)
            test_x = _matrix(test, features)
            if model["hyperparameters"].get("standardise"):
                scaler = Standardiser().fit(train_x)
                train_x = scaler.transform(train_x)
                test_x = scaler.transform(test_x)

            if model["family"] == "linear":
                coefficients = _ridge_fit(
                    train_x, train_target, model["hyperparameters"]["alpha"],
                )
                if coefficients is None:
                    fold_reports.append({**fold.as_row(), "status": "degenerate_design"})
                    continue
                predicted = _predict_linear(coefficients, test_x)
                probabilities = None
            else:
                labels = [1 if value > 0 else 0 for value in train_target]
                coefficients = _logistic_fit(
                    train_x, labels,
                    l2=model["hyperparameters"]["l2"],
                    iterations=model["hyperparameters"]["max_iterations"],
                    learning_rate=model["hyperparameters"]["learning_rate"],
                )
                if coefficients is None:
                    fold_reports.append({**fold.as_row(), "status": "degenerate_design"})
                    continue
                probabilities = _predict_probability(coefficients, test_x)
                predicted = [p - 0.5 for p in probabilities]

        report = {
            **fold.as_row(), "status": FITTED,
            "usable_train": len(train), "usable_test": len(test),
            "train_mean": train_mean,
            **regression_metrics(test_target, predicted, reference=train_mean),
            **direction_metrics(test_target, predicted),
        }
        if probabilities is not None:
            report["brier_score"] = brier_score(test_target, probabilities)
        fold_reports.append(report)

        for position, (row, actual, prediction) in enumerate(
            zip(test, test_target, predicted)
        ):
            predictions.append({
                "fold": fold.index,
                "first_reactable_session": row["first_reactable_session"],
                "exit_date": row.get("exit_date"),
                "signal_family": row.get("dominant_family"),
                "timing_bucket": row.get("dominant_timing_bucket"),
                "regime": row.get("regime"),
                "actual": actual,
                "predicted": prediction,
                "probability": (
                    probabilities[position] if probabilities else None
                ),
            })

    specification["folds"] = fold_reports
    specification["predictions"] = predictions
    specification["status"] = FITTED if predictions else INSUFFICIENT
    if not predictions:
        specification["gate"] = {
            **gate, "status": INSUFFICIENT,
            "binding_requirement": (
                gate["binding_requirement"]
                or "no fold produced out-of-sample predictions"
            ),
        }
        return specification

    actual = [p["actual"] for p in predictions]
    predicted = [p["predicted"] for p in predictions]
    clusters = [p["exit_date"] or p["first_reactable_session"] for p in predictions]
    pooled_reference = _mean(actual)

    testable = {session for fold in folds for session in fold.test}
    specification["pooled"] = {
        **regression_metrics(actual, predicted, reference=pooled_reference),
        **direction_metrics(actual, predicted),
        "predicted_sessions": len(predictions),
        "testable_sessions": len(testable),
        # What fraction of the sessions a fold offered actually got a
        # prediction. A specification that predicts a third of them is not
        # comparable to one that predicts all of them, and the number says so.
        "prediction_coverage": (
            round(len(predictions) / len(testable), 4) if testable else None
        ),
        "missing_by_column": gate["missing_by_column"],
        "absolute_error_interval": cluster_bootstrap(
            [abs(a - p) for a, p in zip(actual, predicted)], clusters,
        ),
        "directional_hit_interval": cluster_bootstrap(
            [1.0 if (a > 0) == (p > 0) else 0.0
             for a, p in zip(actual, predicted) if a != 0],
            [c for c, a in zip(clusters, actual) if a != 0],
        ),
    }
    if any(p.get("probability") is not None for p in predictions):
        probabilities = [p["probability"] for p in predictions]
        specification["pooled"]["brier_score"] = brier_score(actual, probabilities)
        specification["calibration"] = calibration_table(actual, probabilities)

    fitted_folds = [f for f in fold_reports if f.get("status") == FITTED]
    maes = [f["mae"] for f in fitted_folds if f.get("mae") is not None]
    accuracies = [
        f["directional_accuracy"] for f in fitted_folds
        if f.get("directional_accuracy") is not None
    ]
    specification["stability"] = {
        "fitted_folds": len(fitted_folds),
        "mae_mean": _mean(maes),
        "mae_min": min(maes) if maes else None,
        "mae_max": max(maes) if maes else None,
        "mae_spread": (max(maes) - min(maes)) if len(maes) > 1 else None,
        "accuracy_mean": _mean(accuracies),
        "accuracy_min": min(accuracies) if accuracies else None,
        "accuracy_max": max(accuracies) if accuracies else None,
        "folds_above_half": sum(1 for a in accuracies if a > 0.5),
    }
    return specification


def subgroup_report(
    specification: Dict[str, Any], *, minimum: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    """Out-of-sample performance split by family, timing bucket and regime.

    Subgroups below *minimum* observations are reported with their count and no
    metrics. A directional accuracy computed on four sessions is a coin flip
    described to three decimal places.
    """

    report: Dict[str, List[Dict[str, Any]]] = {}
    predictions = specification.get("predictions") or []
    for dimension in ("signal_family", "timing_bucket", "regime"):
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for prediction in predictions:
            key = prediction.get(dimension)
            if key:
                buckets.setdefault(str(key), []).append(prediction)

        rows: List[Dict[str, Any]] = []
        for key in sorted(buckets):
            group = buckets[key]
            actual = [p["actual"] for p in group]
            predicted = [p["predicted"] for p in group]
            if len(group) < minimum:
                rows.append({dimension: key, "n": len(group),
                             "status": "below_reporting_minimum"})
                continue
            rows.append({
                dimension: key, "status": "reported",
                **regression_metrics(actual, predicted, reference=_mean(actual)),
                **direction_metrics(actual, predicted),
            })
        report[dimension] = rows
    return report


def _restrict(
    specification: Dict[str, Any], sessions: set,
) -> Optional[Dict[str, Any]]:
    """Re-score a fitted specification on a given subset of test sessions."""

    predictions = [
        p for p in specification.get("predictions") or []
        if p["first_reactable_session"] in sessions
    ]
    if not predictions:
        return None
    actual = [p["actual"] for p in predictions]
    predicted = [p["predicted"] for p in predictions]
    return {
        **regression_metrics(actual, predicted, reference=_mean(actual)),
        **direction_metrics(actual, predicted),
        # Both metric helpers emit "n" and the second wins; keep an unambiguous
        # count so the matching test below cannot read the directional one.
        "n_predictions": len(predictions),
    }


def compare_to_baselines(
    specifications: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Rank news specifications against the best baseline on the same target."""

    protocol = _spec()
    thresholds = protocol["decision_thresholds"]
    fitted = [s for s in specifications if s.get("status") == FITTED and s.get("pooled")]

    comparisons: List[Dict[str, Any]] = []
    for target in sorted({s["target"] for s in fitted}):
        on_target = [s for s in fitted if s["target"] == target]
        baselines = [s for s in on_target if s["kind"] == "baseline"]
        if not baselines:
            continue

        for spec in (s for s in on_target if s["kind"] == "news"):
            # Specifications differ in coverage: a feature with more missing
            # values predicts fewer sessions. Comparing its MAE against a
            # baseline scored on a larger, different set is not a comparison of
            # models, it is a comparison of samples. Every baseline is therefore
            # re-scored on exactly the sessions this specification predicted.
            covered = {p["first_reactable_session"] for p in spec["predictions"]}
            matched = [
                (base, _restrict(base, covered)) for base in baselines
            ]
            usable = [
                (base, m) for base, m in matched
                if m and m["n_predictions"] >= 3
            ]

            mae = spec["pooled"]["mae"]
            accuracy = spec["pooled"]["directional_accuracy"]
            best_mae = min(
                (pair for pair in usable if pair[1]["mae"] is not None),
                key=lambda pair: pair[1]["mae"], default=None,
            )
            best_accuracy = max(
                (pair for pair in usable
                 if pair[1]["directional_accuracy"] is not None),
                key=lambda pair: pair[1]["directional_accuracy"], default=None,
            )
            mae_gain = (
                best_mae[1]["mae"] - mae
                if best_mae and mae is not None else None
            )
            accuracy_gain = (
                accuracy - best_accuracy[1]["directional_accuracy"]
                if best_accuracy and accuracy is not None else None
            )
            beats_mae = bool(
                mae_gain is not None
                and mae_gain >= thresholds["minimum_improvement_over_best_baseline_mae"]
            )
            beats_direction = bool(
                accuracy_gain is not None
                and accuracy_gain
                >= thresholds["minimum_directional_accuracy_over_majority"]
            )
            interval = spec["pooled"]["directional_hit_interval"]
            excludes_chance = bool(
                interval.get("lower") is not None and interval["lower"] > 0.5
            )
            comparisons.append({
                "target": target,
                "feature_set": spec["feature_set"],
                "model": spec["model"],
                "predicted_sessions": len(covered),
                "mae": mae,
                "best_baseline_mae": best_mae[1]["mae"] if best_mae else None,
                "best_baseline_feature_set": (
                    best_mae[0]["feature_set"] if best_mae else None
                ),
                "baselines_matched_on_sessions": len(covered),
                "mae_improvement": mae_gain,
                "directional_accuracy": accuracy,
                "best_baseline_accuracy": (
                    best_accuracy[1]["directional_accuracy"]
                    if best_accuracy else None
                ),
                "best_baseline_accuracy_feature_set": (
                    best_accuracy[0]["feature_set"] if best_accuracy else None
                ),
                "accuracy_improvement": accuracy_gain,
                "beats_mae_threshold": beats_mae,
                "beats_direction_threshold": beats_direction,
                "hit_rate_interval_excludes_chance": excludes_chance,
                "meets_success_criteria": (
                    beats_mae and beats_direction and excludes_chance
                ),
            })

    successes = [c for c in comparisons if c["meets_success_criteria"]]
    return {
        "comparisons": comparisons,
        "specifications_run": len(fitted),
        "specifications_blocked": sum(
            1 for s in specifications if s.get("status") == INSUFFICIENT
        ),
        "successes": len(successes),
        "verdict": (
            "success" if successes
            else "failure" if comparisons
            else "inconclusive"
        ),
        "multiplicity_note": (
            f"{len(comparisons)} news specifications were compared; at alpha "
            f"{thresholds['alpha']} roughly "
            f"{max(1, round(len(comparisons) * thresholds['alpha']))} would clear "
            "a nominal threshold by chance alone. No result below is described "
            "as significant."
        ),
    }
