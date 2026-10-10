"""Inference that does not mistake rows for independent observations.

Stock-level events are dependent in two directions. Events on the same day
share market and sector shocks; events for the same issuer share whatever is
persistent about that issuer. A plain t-test on event rows ignores both and
reports confidence it has not earned -- the same error the index study caught
when 731 event rows turned out to carry 49 outcomes.

The primary test here clusters by **event date**. Issuer clustering and
two-way clustering (Cameron, Gelbach and Miller, 2011) are available as
sensitivities. Small numbers of clusters are handled with the CR1 correction
and a t reference distribution with ``G - 1`` degrees of freedom; below
``MIN_CLUSTERS`` no p-value is produced at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as _scipy

MIN_CLUSTERS = 10


def _codes(labels: Sequence[Any]) -> np.ndarray:
    _, codes = np.unique(np.asarray([str(v) for v in labels]), return_inverse=True)
    return codes


def _meat(X: np.ndarray, resid: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, int]:
    scores = X * resid[:, None]
    groups = int(codes.max()) + 1
    summed = np.zeros((groups, X.shape[1]))
    np.add.at(summed, codes, scores)
    return summed.T @ summed, groups


def ols_cluster(X: np.ndarray, y: np.ndarray, clusters: Sequence[Sequence[Any]],
                names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """OLS with one-way or two-way cluster-robust standard errors.

    ``clusters`` is a list of one or two label sequences. Rows with any
    non-finite value must be removed by the caller; this function refuses them
    rather than dropping them silently.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("non-finite values reached the estimator")
    n, k = X.shape
    names = list(names) if names is not None else [f"x{i}" for i in range(k)]
    out: Dict[str, Any] = {"n": n, "k": k, "names": names, "status": "ok"}
    if n <= k:
        return {**out, "status": "too_few_observations"}

    xtx = X.T @ X
    if np.linalg.matrix_rank(xtx) < k:
        return {**out, "status": "collinear_design"}
    bread = np.linalg.inv(xtx)
    beta = bread @ X.T @ y
    resid = y - X @ beta

    code_sets = [_codes(c) for c in clusters]
    groups = []
    variance = np.zeros((k, k))

    def piece(codes: np.ndarray) -> np.ndarray:
        meat, g = _meat(X, resid, codes)
        groups.append(g)
        scale = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 else np.nan
        return scale * bread @ meat @ bread

    if len(code_sets) == 1:
        variance = piece(code_sets[0])
    elif len(code_sets) == 2:
        first, second = piece(code_sets[0]), piece(code_sets[1])
        both = _codes([f"{a}|{b}" for a, b in zip(code_sets[0], code_sets[1])])
        variance = first + second - piece(both)
        groups = groups[:2]
    else:
        raise ValueError("one or two clusterings are supported")

    min_groups = min(groups)
    out.update({"coef": beta.tolist(), "clusters": groups, "min_clusters": min_groups})
    diagonal = np.diag(variance)
    if min_groups < MIN_CLUSTERS or not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
        # Two-way variance can be non-positive in small samples; that is a
        # reason to report no inference, not to clip it to something usable.
        return {**out, "status": "too_few_clusters" if min_groups < MIN_CLUSTERS
                else "variance_not_positive", "se": None, "t": None, "p": None, "ci": None}

    se = np.sqrt(diagonal)
    t = beta / se
    dof = min_groups - 1
    p = 2 * _scipy.t.sf(np.abs(t), dof)
    critical = _scipy.t.ppf(0.975, dof)
    out.update({
        "se": se.tolist(), "t": t.tolist(), "p": p.tolist(), "dof": dof,
        "ci": [[float(b - critical * s), float(b + critical * s)] for b, s in zip(beta, se)],
    })
    return out


def cluster_mean(values: Sequence[float], clusters: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    """Mean with cluster-robust inference: an intercept-only regression."""

    y = np.asarray(values, dtype=float)
    fit = ols_cluster(np.ones((len(y), 1)), y, clusters, names=["mean"])
    result = {"n": fit["n"], "status": fit["status"], "clusters": fit.get("clusters"),
              "mean": None, "se": None, "t": None, "p": None, "ci": None}
    if "coef" in fit:
        result["mean"] = fit["coef"][0]
    if fit["status"] == "ok":
        result.update({"se": fit["se"][0], "t": fit["t"][0], "p": fit["p"][0],
                       "ci": fit["ci"][0], "dof": fit["dof"]})
    return result


def cluster_bootstrap(values: Sequence[float], clusters: Sequence[Any], *,
                      resamples: int, seed: int, level: float = 0.95) -> Dict[str, Any]:
    """Percentile interval for a mean, resampling whole clusters."""

    y = np.asarray(values, dtype=float)
    codes = _codes(clusters)
    groups = int(codes.max()) + 1 if len(codes) else 0
    if groups < 2:
        return {"mean": float(y.mean()) if len(y) else None, "lower": None,
                "upper": None, "clusters": groups}
    sums = np.bincount(codes, weights=y, minlength=groups)
    counts = np.bincount(codes, minlength=groups)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, groups, size=(resamples, groups))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1 - level) / 2 * 100
    return {"mean": float(y.mean()), "lower": float(np.percentile(means, tail)),
            "upper": float(np.percentile(means, 100 - tail)), "clusters": groups,
            "resamples": resamples, "seed": seed}


# -- Multiple testing ---------------------------------------------------------
def holm(p_values: Sequence[Optional[float]], family_size: Optional[int] = None) -> List[Optional[float]]:
    """Holm step-down adjusted p-values.

    ``family_size`` fixes the number of hypotheses in the family. It defaults
    to the number of p-values supplied, but a registered family should pass its
    registered size: a test that could not be run still belongs to the family
    it was registered in, and dropping it would make the correction lighter
    exactly when data are thin.
    """

    present = [(i, p) for i, p in enumerate(p_values) if p is not None]
    m = family_size if family_size is not None else len(present)
    if m < len(present):
        raise ValueError("family_size cannot be smaller than the number of tests run")
    adjusted: List[Optional[float]] = [None] * len(p_values)
    running = 0.0
    for rank, (index, p) in enumerate(sorted(present, key=lambda item: item[1])):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg adjusted p-values (false-discovery rate)."""

    present = sorted(((p, i) for i, p in enumerate(p_values) if p is not None), reverse=True)
    m = len(present)
    adjusted: List[Optional[float]] = [None] * len(p_values)
    running = 1.0
    for position, (p, index) in enumerate(present):
        rank = m - position
        running = min(running, p * m / rank)
        adjusted[index] = running
    return adjusted
