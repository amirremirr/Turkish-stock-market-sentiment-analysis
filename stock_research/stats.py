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
    out["cov"] = variance.tolist()
    t = beta / se
    dof = min_groups - 1
    p = 2 * _scipy.t.sf(np.abs(t), dof)
    critical = _scipy.t.ppf(0.975, dof)
    out.update({
        "se": se.tolist(), "t": t.tolist(), "p": p.tolist(), "dof": dof,
        "ci": [[float(b - critical * s), float(b + critical * s)] for b, s in zip(beta, se)],
    })
    return out


def wald(fit: Dict[str, Any], restrictions: np.ndarray) -> Dict[str, Any]:
    """Joint test of ``R beta = 0`` from a cluster-robust fit.

    Uses an F reference with ``(q, G - 1)`` degrees of freedom, which is the
    conservative small-cluster choice; ``q`` cannot exceed ``G - 1``.
    """

    if fit.get("status") != "ok":
        return {"status": fit.get("status"), "p": None}
    R = np.atleast_2d(np.asarray(restrictions, dtype=float))
    beta, cov = np.asarray(fit["coef"]), np.asarray(fit["cov"])
    q, dof = R.shape[0], fit["dof"]
    middle = R @ cov @ R.T
    if q > dof or np.linalg.matrix_rank(middle) < q:
        return {"status": "too_few_clusters_for_joint_test", "p": None, "q": q, "dof": dof}
    statistic = float((R @ beta) @ np.linalg.solve(middle, R @ beta) / q)
    return {"status": "ok", "f": statistic, "q": q, "dof": dof,
            "p": float(_scipy.f.sf(statistic, q, dof))}


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


# -- Joint tests that hold their level on unbalanced designs -------------------
def _restricted(X: np.ndarray, y: np.ndarray, R: np.ndarray):
    """OLS with ``R beta = 0`` imposed: (bread, unrestricted beta, restricted beta)."""

    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    middle = R @ bread @ R.T
    return bread, beta, beta - bread @ R.T @ np.linalg.solve(middle, R @ beta)


def wild_cluster_wald(X: np.ndarray, y: np.ndarray, restrictions: np.ndarray,
                      boot_clusters: Sequence[Any], *,
                      other_clusters: Optional[Sequence[Any]] = None,
                      draws: int, seed: int, chunk: int = 64) -> Dict[str, Any]:
    """Wild cluster bootstrap p-value for the joint null ``R beta = 0``.

    The statistic is the cluster-robust Wald statistic of :func:`wald`:
    clustered by ``boot_clusters`` alone, or two ways when ``other_clusters``
    is given. Its reference distribution is not F. It is built by imposing the
    null, flipping the sign of every restricted residual in a cluster together
    (Rademacher weights, one per ``boot_clusters`` cluster), and recomputing
    the statistic on each such sample (Cameron, Gelbach and Miller 2008; for a
    two-way statistic with weights along one dimension, MacKinnon, Nielsen and
    Webb 2021).

    ``p = (1 + number of draws with a statistic at least as large) /
    (1 + draws)``. A draw whose two-way variance is not positive definite
    counts as at least as large: that can only raise p.

    The F reference of :func:`wald` takes its denominator degrees of freedom
    from all clusters. When a regressor lives on a few of them, as a small
    category's indicator does, the statistic is more variable than that F and
    the test rejects a true null several times too often
    (docs/stock_research/readiness/INFERENCE_CALIBRATION.md). The bootstrap
    carries the design's imbalance into its own reference distribution.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    R = np.atleast_2d(np.asarray(restrictions, dtype=float))
    n, k = X.shape
    q = R.shape[0]
    two_way = other_clusters is not None
    fit = ols_cluster(X, y, [boot_clusters] + ([other_clusters] if two_way else []))
    observed = wald(fit, R)
    out: Dict[str, Any] = {
        "status": observed.get("status"), "p": None, "f": observed.get("f"), "q": q,
        "draws": int(draws), "seed": int(seed), "clusters": fit.get("clusters"),
        "statistic": "two_way" if two_way else "one_way", "invalid_draws": 0,
    }
    if observed.get("status") != "ok":
        return out

    bread, _, restricted = _restricted(X, y, R)
    resid = y - X @ restricted                       # residuals with the null imposed
    g = _codes(boot_clusters)
    groups = int(g.max()) + 1
    scores = X * resid[:, None]
    outer = X[:, :, None] * X[:, None, :]
    S = np.zeros((groups, k))
    np.add.at(S, g, scores)
    XX = np.zeros((groups, k, k))
    np.add.at(XX, g, outer)
    adjust = (n - 1) / (n - k)
    scale_g = groups / (groups - 1) * adjust

    if two_way:
        # Cells of the two clusterings, sorted so that one issuer's cells are
        # contiguous: issuer sums are then a ``reduceat`` over cell sums.
        i = _codes(other_clusters)
        key = i.astype(np.int64) * groups + g
        order = np.argsort(key, kind="stable")
        keys, first = np.unique(key[order], return_index=True)
        cell_g = (keys % groups).astype(int)
        cell_i = keys // groups
        A = np.add.reduceat(scores[order], first, axis=0)
        D_cell = np.add.reduceat(outer[order], first, axis=0)
        issuer_first = np.flatnonzero(np.r_[True, cell_i[1:] != cell_i[:-1]])
        issuers, cells = len(issuer_first), len(keys)
        scale_i = issuers / (issuers - 1) * adjust
        scale_cell = cells / (cells - 1) * adjust

    rng = np.random.default_rng(seed)
    exceed = invalid = 0
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        W = rng.integers(0, 2, size=(size, groups)) * 2.0 - 1.0
        delta = (W @ S) @ bread                       # beta* - restricted beta
        Sg = W[:, :, None] * S[None] - np.einsum("gkl,bl->bgk", XX, delta)
        meat = scale_g * np.einsum("bgk,bgl->bkl", Sg, Sg)
        if two_way:
            Sc = W[:, cell_g, None] * A[None] - np.einsum("pkl,bl->bpk", D_cell, delta)
            Si = np.add.reduceat(Sc, issuer_first, axis=1)
            meat = (meat + scale_i * np.einsum("bik,bil->bkl", Si, Si)
                    - scale_cell * np.einsum("bpk,bpl->bkl", Sc, Sc))
        middle = R @ (bread @ meat @ bread) @ R.T     # (size, q, q)
        estimate = delta @ R.T                        # R beta*, since R beta_restricted = 0
        try:
            solved = np.linalg.solve(middle, estimate[:, :, None])[:, :, 0]
            f = np.einsum("bq,bq->b", estimate, solved) / q
        except np.linalg.LinAlgError:                 # a singular draw: one at a time
            f = np.full(size, np.nan)
            for b in range(size):
                try:
                    f[b] = estimate[b] @ np.linalg.solve(middle[b], estimate[b]) / q
                except np.linalg.LinAlgError:
                    pass
        if two_way:
            # The difference of three variances need not be positive definite.
            f[np.linalg.eigvalsh(middle).min(axis=1) <= 0] = np.nan
        bad = ~np.isfinite(f) | (f < 0)
        invalid += int(bad.sum())
        exceed += int((bad | (f >= observed["f"])).sum())
    out.update({"status": "ok", "p": (1 + exceed) / (draws + 1), "invalid_draws": invalid})
    return out


def cr2_wald(X: np.ndarray, y: np.ndarray, restrictions: np.ndarray,
             clusters: Sequence[Any]) -> Dict[str, Any]:
    """Joint test of ``R beta = 0`` with the CR2 covariance and Satterthwaite-type
    degrees of freedom (the "AHT" test of Pustejovsky and Tipton 2018).

    CR2 rescales each cluster's residuals by ``(I - H_gg)^(-1/2)`` so that the
    covariance is unbiased under independent, equal-variance errors. The Wald
    statistic ``Q`` is then referred to Hotelling's T-squared with ``eta``
    degrees of freedom, where ``eta`` matches the variability of the estimated
    covariance of ``R beta`` under that working model:

        ``F = (eta - q + 1) / (eta q) * Q  ~  F(q, eta - q + 1)``.

    One-way clustering only. With one restriction this is the Bell-McCaffrey
    t-test; for a mean with equal clusters ``eta`` is ``G - 1``.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    R = np.atleast_2d(np.asarray(restrictions, dtype=float))
    n, k = X.shape
    q = R.shape[0]
    codes = _codes(clusters)
    groups = int(codes.max()) + 1
    out: Dict[str, Any] = {"status": "ok", "p": None, "q": q, "clusters": groups}
    if groups < MIN_CLUSTERS:
        return {**out, "status": "too_few_clusters"}
    xtx = X.T @ X
    if np.linalg.matrix_rank(xtx) < k:
        return {**out, "status": "collinear_design"}
    bread = np.linalg.inv(xtx)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    values, vectors = np.linalg.eigh(R @ bread @ R.T)
    if values.min() <= 0:
        return {**out, "status": "restrictions_not_full_rank"}
    C = (vectors * values ** -0.5) @ vectors.T @ R @ bread      # Omega^(-1/2) R (X'X)^-1

    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(groups + 1))
    meat = np.zeros((k, k))
    E = np.zeros((groups, q, k))
    BB = np.zeros((groups, q, q))
    for c in range(groups):
        rows = order[bounds[c]:bounds[c + 1]]
        Xg = X[rows]
        values, vectors = np.linalg.eigh(np.eye(len(rows)) - Xg @ bread @ Xg.T)
        keep = values > 1e-10
        A = (vectors[:, keep] * values[keep] ** -0.5) @ vectors[:, keep].T
        score = Xg.T @ (A @ resid[rows])
        meat += np.outer(score, score)
        Bg = C @ Xg.T @ A
        BB[c] = Bg @ Bg.T
        E[c] = Bg @ Xg
    middle = R @ (bread @ meat @ bread) @ R.T
    if np.linalg.matrix_rank(middle) < q:
        return {**out, "status": "variance_not_positive"}
    estimate = R @ beta
    Q = float(estimate @ np.linalg.solve(middle, estimate))

    # Covariances between the clusters' contributions to the estimated
    # covariance of R beta, under the working model.
    P = -np.einsum("gqk,hrk->ghqr", E @ bread, E)
    P[np.arange(groups), np.arange(groups)] += BB
    traces = np.einsum("ghqq->gh", P)
    eta = q * (q + 1) / float((traces ** 2).sum() + np.einsum("ghqr,ghrq->", P, P))
    dof = eta - q + 1
    out.update({"eta": eta, "dof": dof, "wald": Q})
    if dof <= 0:
        return {**out, "status": "too_few_effective_clusters"}
    f = dof / (eta * q) * Q
    out.update({"f": float(f), "p": float(_scipy.f.sf(f, q, dof))})
    return out


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
