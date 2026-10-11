"""Calibration of the registered inference, and of alternatives to it.

Everything here is a null simulation on synthetic returns. The procedures
compared are *not* swapped into the registered study; the registered estimator
is called as it is, and the alternatives are implemented here, outside the
fingerprinted package, as candidates for a future protocol.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as sps

from readiness_audit import power
from stock_research import stats

LEVELS = (0.05, power.HOLM_ALPHA)


def _codes(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels, return_inverse=True)[1]


class Procedures:
    """Several joint tests of "all category means are equal" on one design."""

    def __init__(self, design: power.Design, categories: Optional[Sequence[str]] = None):
        import pandas as pd

        keep = np.isin(design.category, list(categories)) if categories is not None \
            else np.ones(len(design.category), dtype=bool)
        self.keep = keep
        category = design.category[keep]
        counts = pd.Series(category).value_counts()
        self.reference = counts.idxmax()
        self.names = [c for c in counts.index if c != self.reference]
        self.X = np.column_stack([np.ones(keep.sum())]
                                 + [(category == c).astype(float) for c in self.names])
        self.q = len(self.names)
        self.R = np.zeros((self.q, self.q + 1))
        self.R[:, 1:] = np.eye(self.q)
        self.date = _codes(design.date[keep])
        self.issuer = _codes(design.issuer[keep])
        self.both = _codes(np.array([f"{a}|{b}" for a, b in zip(self.date, self.issuer)]))
        self.bread = np.linalg.inv(self.X.T @ self.X)
        self.n, self.k = self.X.shape
        self.category = category
        # Fewest clusters carrying any one category: the binding count for a
        # test about that category's mean.
        self.min_dates = min(len(np.unique(self.date[category == c])) for c in counts.index)
        self.min_issuers = min(len(np.unique(self.issuer[category == c])) for c in counts.index)

    # -- pieces ----------------------------------------------------------------
    def _cr1(self, resid: np.ndarray, codes: np.ndarray) -> np.ndarray:
        scores = self.X * resid[:, None]
        g = int(codes.max()) + 1
        summed = np.zeros((g, self.k))
        np.add.at(summed, codes, scores)
        scale = (g / (g - 1)) * ((self.n - 1) / (self.n - self.k))
        return scale * self.bread @ (summed.T @ summed) @ self.bread

    def _wald(self, beta: np.ndarray, cov: np.ndarray) -> float:
        middle = self.R @ cov @ self.R.T
        try:
            return float((self.R @ beta) @ np.linalg.solve(middle, self.R @ beta) / self.q)
        except np.linalg.LinAlgError:
            return np.nan

    def __call__(self, y_full: np.ndarray, *, wild: int = 0,
                 rng: Optional[np.random.Generator] = None) -> Dict[str, Optional[float]]:
        y = y_full[self.keep]
        beta = self.bread @ self.X.T @ y
        resid = y - self.X @ beta
        v_date, v_issuer = self._cr1(resid, self.date), self._cr1(resid, self.issuer)
        v_both = self._cr1(resid, self.both)
        g_date, g_issuer = int(self.date.max()) + 1, int(self.issuer.max()) + 1
        g_min = min(g_date, g_issuer)
        out: Dict[str, Optional[float]] = {}

        def p_from(cov: np.ndarray, dof: float) -> Optional[float]:
            f = self._wald(beta, cov)
            if not np.isfinite(f) or f < 0 or dof < 1:
                return None
            return float(sps.f.sf(f, self.q, dof))

        # The registered test, through the registered functions.
        fit = stats.ols_cluster(self.X, y, [self.date, self.issuer])
        out["registered_two_way"] = stats.wald(fit, self.R).get("p")
        out["date_only"] = p_from(v_date, g_date - 1)
        out["issuer_only"] = p_from(v_issuer, g_issuer - 1)
        # Two-way without subtracting the intersection term: larger by
        # construction, never negative.
        out["two_way_sum"] = p_from(v_date + v_issuer, g_min - 1)
        # The registered variance, but with degrees of freedom set by the
        # category with the fewest date clusters rather than by all dates.
        out["two_way_min_category_dof"] = p_from(v_date + v_issuer - v_both,
                                                 min(self.min_dates, self.min_issuers) - self.q)
        out["date_only_min_category_dof"] = p_from(v_date, self.min_dates - self.q)

        # Delete-one-date jackknife (CV3) of the category means.
        f_jack = self._jackknife(y)
        out["jackknife_date"] = (float(sps.f.sf(f_jack, self.q, g_date - 1))
                                 if np.isfinite(f_jack) else None)

        if wild:
            out["wild_cluster_date"] = self._wild(y, wild, rng)
        return out

    def _jackknife(self, y: np.ndarray) -> float:
        g = int(self.date.max()) + 1
        xtx = self.X.T @ self.X
        xty = self.X.T @ y
        betas = np.empty((g, self.k))
        order = np.argsort(self.date, kind="stable")
        bounds = np.searchsorted(self.date[order], np.arange(g + 1))
        for c in range(g):
            rows = order[bounds[c]:bounds[c + 1]]
            Xg, yg = self.X[rows], y[rows]
            try:
                betas[c] = np.linalg.solve(xtx - Xg.T @ Xg, xty - Xg.T @ yg)
            except np.linalg.LinAlgError:
                return np.nan
        beta = np.linalg.solve(xtx, xty)
        centred = betas - beta
        cov = (g - 1) / g * centred.T @ centred
        return self._wald(beta, cov)

    def _wild(self, y: np.ndarray, draws: int, rng: np.random.Generator) -> Optional[float]:
        """Restricted wild cluster bootstrap by date (Rademacher weights) of
        the date-clustered Wald statistic: Cameron, Gelbach and Miller (2008)."""

        beta = self.bread @ self.X.T @ y
        observed = self._wald(beta, self._cr1(y - self.X @ beta, self.date))
        if not np.isfinite(observed):
            return None
        restricted_mean = y.mean()                 # the null: one common mean
        u = y - restricted_mean
        g = int(self.date.max()) + 1
        exceed = 0
        for _ in range(draws):
            signs = rng.choice((-1.0, 1.0), size=g)[self.date]
            y_star = restricted_mean + u * signs
            b = self.bread @ self.X.T @ y_star
            f = self._wald(b, self._cr1(y_star - self.X @ b, self.date))
            exceed += bool(np.isfinite(f) and f >= observed)
        return (exceed + 1) / (draws + 1)


def synthetic_design(sizes: Sequence[int], dates: Sequence[int], *, seed: int = 0,
                     sigma: float = 0.03, sessions: int = 440) -> power.Design:
    """Independent events on a made-up layout: a different issuer for every
    event, category *k* holding ``sizes[k]`` events spread over ``dates[k]``
    sessions. Nothing in it comes from the sample."""

    rng = np.random.default_rng(seed)
    category, position = [], []
    for k, (size, days) in enumerate(zip(sizes, dates)):
        category += [f"c{k}"] * size
        position += list(140 + rng.integers(0, days, size) * 3 + k % 3)
    n = len(category)
    return power.Design(issuer=np.arange(n), position=np.array(position),
                        date=np.array(position), category=np.array(category),
                        kept=[f"c{k}" for k in range(len(sizes))],
                        sigma=np.full(n, sigma), sessions=sessions, anchors=0)


def small_category_experiment(simulations: int = 3000, seed: int = 31) -> List[Dict[str, Any]]:
    """Size of the registered test as one category shrinks, everything else
    held fixed: four categories of 300 independent events on 80 dates each,
    plus a fifth of the size and date count given."""

    rng = np.random.default_rng(seed)
    rows = []
    for size, days in ((30, 15), (40, 22), (60, 30), (90, 45), (150, 60), (300, 80)):
        design = synthetic_design([300, 300, 300, 300, size], [80, 80, 80, 80, days],
                                  seed=int(rng.integers(1 << 30)))
        test = power.JointTest(design)
        hits = {level: 0 for level in LEVELS}
        corrected = {level: 0 for level in LEVELS}
        for _ in range(simulations):
            outcome = test(power.simulate_cars(design, rng, common_share=0.0, ar1=0.0))
            for level in LEVELS:
                hits[level] += bool(outcome["p"] is not None and outcome["p"] < level)
                corrected[level] += bool(outcome["p_corrected"] is not None
                                         and outcome["p_corrected"] < level)
        row: Dict[str, Any] = {"fifth_category_events": size, "fifth_category_dates": days,
                               "simulations": simulations}
        for level in LEVELS:
            for name, tally in (("registered", hits), ("min_category_dof", corrected)):
                rate = tally[level] / simulations
                half = 1.96 * np.sqrt(rate * (1 - rate) / simulations)
                row[f"{name}_rate_at_{level:.4f}"] = rate
                row[f"{name}_ci_at_{level:.4f}"] = [max(0.0, rate - half), rate + half]
        rows.append(row)
    return rows


def null_rates(structure, calibration, *, sessions: int, simulations: int, projections: int,
               recurrence: str = "crp", common_share: Optional[float] = None,
               ar1: Optional[float] = None, categories: Optional[int] = None,
               wild: int = 0, anchors_total: int = 80, seed: int = 7,
               event_vol: float = 1.0,
               ingested: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Rejection rates under the null for every procedure.

    ``categories``: keep only the largest *categories* of those that meet the
    registered size rule (None keeps them all).
    """

    import pandas as pd

    rng = np.random.default_rng(seed)
    common_share = calibration["common_daily_share"] if common_share is None else common_share
    ar1 = calibration["mean_ar1"] if ar1 is None else ar1
    per = int(np.ceil(simulations / projections))
    hits: Dict[str, Dict[float, int]] = {}
    valid: Dict[str, int] = {}
    shape: List[Dict[str, float]] = []
    for _ in range(projections):
        design = power.project(structure, calibration, anchors_total=anchors_total,
                               frame_anchors=80, sessions=sessions, rng=rng, recurrence=recurrence,
                               ingested=ingested)
        order = pd.Series(design.category).value_counts().index.tolist()
        chosen = order[:categories] if categories else order
        proc = Procedures(design, chosen)
        counts = pd.Series(proc.category).value_counts()
        shape.append({"events": proc.n, "categories": len(chosen),
                      "smallest_category": int(counts.min()),
                      "dates": int(proc.date.max()) + 1, "issuers": int(proc.issuer.max()) + 1,
                      "min_dates_in_a_category": proc.min_dates,
                      "min_issuers_in_a_category": proc.min_issuers})
        for _ in range(per):
            y = power.simulate_cars(design, rng, common_share=common_share, ar1=ar1,
                                    event_vol=event_vol)
            for name, p in proc(y, wild=wild, rng=rng).items():
                if p is None:
                    continue
                valid[name] = valid.get(name, 0) + 1
                tally = hits.setdefault(name, {level: 0 for level in LEVELS})
                for level in LEVELS:
                    tally[level] += p < level
    rows = {}
    for name, tally in hits.items():
        n = valid[name]
        rows[name] = {"valid": n}
        for level in LEVELS:
            rate = tally[level] / n
            half = 1.96 * np.sqrt(rate * (1 - rate) / n)
            rows[name][f"rate_at_{level:.4f}"] = rate
            rows[name][f"ci_at_{level:.4f}"] = [max(0.0, rate - half), rate + half]
    return {"design": pd.DataFrame(shape).mean().to_dict(), "procedures": rows,
            "settings": {"recurrence": recurrence, "common_share": common_share, "ar1": ar1,
                         "categories": categories, "simulations": simulations,
                         "anchors": anchors_total, "wild_draws": wild, "event_vol": event_vol}}
