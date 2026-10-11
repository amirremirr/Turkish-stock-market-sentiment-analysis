"""Power of the registered H7 test, by simulation on the real sample structure.

What is real: which issuer filed in which category on which session, taken
from the snapshot without reading a return, and the eligibility rules of the
registered protocol. What is synthetic: every return. Residual volatility,
the share of variance common to all stocks on a day, serial correlation and
tail weight are calibrated on 2022 only, before any sampled disclosure.

The test applied is the registered one, called directly:
``stats.ols_cluster`` on category indicators with two-way clustering,
``stats.wald`` for the joint null, Holm with a family of nine, and the
registered material effect on the spread between category means.

Projection to the full frame
----------------------------
The snapshot holds some of the 80 anchors. Each missing anchor gets a session
by interpolating anchor number against the sessions of the anchors already
seen (disclosure numbers rise with time), and a set of events by resampling an
observed anchor. Twenty such projections are drawn and averaged over, so the
result does not hang on one guess about the anchors still to come.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from stock_research import stats
from stock_research.config import (
    ALPHA, ESTIMATION_WINDOW, MIN_EVENT_DATES_PER_GROUP, MIN_EVENTS_PER_GROUP,
)
from stock_research.hypotheses import common, h7

HOLM_ALPHA = ALPHA / common.FAMILY_SIZE        # the lone-test case: p * 9 < 0.05
MATERIAL = h7.SPEC.material_effect
WINDOW = h7.PRIMARY_WINDOW
EFFECTS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05)


@dataclass
class Design:
    """One projected sample: per-event arrays plus what the test needs."""

    issuer: np.ndarray          # issuer code per event (0..J-1)
    position: np.ndarray        # day-0 session index per event
    date: np.ndarray            # cluster label per event (day 0)
    category: np.ndarray        # category name per event
    kept: List[str]             # categories meeting the registered size rule
    sigma: np.ndarray           # residual sd per issuer
    sessions: int
    anchors: int


def project(structure: pd.DataFrame, calibration: Dict[str, Any], *, anchors_total: int,
            frame_anchors: int, sessions: int, rng: np.random.Generator,
            recurrence: str = "crp", ingested: Optional[Sequence[int]] = None) -> Design:
    """A full-size sample built from the observed anchors.

    ``ingested`` lists every anchor that has been fully ingested, including
    those that yield no usable event (an anchor whose fifty disclosures were
    all published during the session gives H7's primary sample nothing). They
    belong in the pool an unseen anchor is drawn from; leaving them out would
    project a sample larger than the frame will deliver.

    ``recurrence`` says who files in an anchor not yet seen:

    ``crp``       (default) each event's issuer is a new one with probability
                  equal to the share of issuers seen once in that category so
                  far (a Good-Turing estimate), and otherwise an issuer already
                  seen in that category, drawn in proportion to how often. New
                  issuers join the pool, so frequent filers stay frequent and
                  the number of distinct issuers grows as it would.
    ``resample``  the resampled anchor's own issuers again. Overstates how
                  often an issuer recurs; kept as a pessimistic bound.
    ``fresh``     every event a different issuer: no recurrence at all.
    """

    seen = sorted(structure["anchor"].unique())
    by_anchor = {a: g for a, g in structure.groupby("anchor")}
    first_pos = {a: int(g["position"].min()) for a, g in by_anchor.items()}
    # Anchor k of a sample of `anchors_total` sits at this point of the
    # registered frame's 0..frame_anchors-1 axis; its session is interpolated
    # from the anchors already seen (disclosure numbers rise with time).
    numbers = np.arange(anchors_total) * (frame_anchors / anchors_total)
    interpolated = np.interp(numbers, seen, [first_pos[a] for a in seen])
    pool_of_anchors = sorted(set(seen) | {int(a) for a in (ingested or [])})

    # Issuers seen so far, by category, with how often each filed.
    pools: Dict[str, Dict[str, int]] = {}
    for category, group in structure.groupby("category"):
        pools[category] = group["ticker"].value_counts().to_dict()
    fresh_count = [0]

    rows = []
    for k in range(anchors_total):
        number = numbers[k]
        observed = anchors_total == frame_anchors and int(number) in pool_of_anchors
        source = int(number) if observed else int(rng.choice(pool_of_anchors))
        if source not in by_anchor:
            continue                                  # an anchor with no usable event
        start = first_pos[source] if observed else int(round(interpolated[k]))
        block = by_anchor[source]
        offset = block["position"].to_numpy() - first_pos[source]
        tickers = block["ticker"].to_numpy().astype(object)
        if not observed and recurrence != "resample":
            tickers = tickers.copy()
            for i, category in enumerate(block["category"].to_numpy()):
                pool = pools.setdefault(category, {})
                total = sum(pool.values())
                singles = sum(1 for v in pool.values() if v == 1)
                new_share = 1.0 if (recurrence == "fresh" or total == 0) else singles / total
                if rng.random() < new_share:
                    fresh_count[0] += 1
                    name = f"NEW{fresh_count[0]:05d}"
                else:
                    names = list(pool)
                    weights = np.array([pool[n] for n in names], dtype=float)
                    name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
                pool[name] = pool.get(name, 0) + 1
                tickers[i] = name
        rows.append(pd.DataFrame({
            "ticker": tickers, "category": block["category"].to_numpy(),
            "position": np.clip(start + offset, 136, sessions - 21),
        }))
    events = pd.concat(rows, ignore_index=True)
    # One row per issuer and day 0, as the registered study collapses them.
    events = events.drop_duplicates(["ticker", "position"]).reset_index(drop=True)

    sizes = events.groupby("category").agg(n=("ticker", "size"), d=("position", "nunique"))
    kept = [c for c, r in sizes.iterrows()
            if r["n"] >= MIN_EVENTS_PER_GROUP and r["d"] >= MIN_EVENT_DATES_PER_GROUP]
    events = events[events["category"].isin(kept)].reset_index(drop=True)

    tickers, issuer = np.unique(events["ticker"], return_inverse=True)
    known = calibration["residual_sd_by_ticker"]
    pool = np.array([v for v in known.values() if np.isfinite(v)])
    sigma = np.array([known[t] if t in known and np.isfinite(known[t]) else rng.choice(pool)
                      for t in tickers])
    return Design(issuer, events["position"].to_numpy(), events["position"].to_numpy(),
                  events["category"].to_numpy(), kept, sigma, sessions, anchors_total)


def simulate_cars(design: Design, rng: np.random.Generator, *, common_share: float,
                  ar1: float, df: float = 9.0, event_vol: float = 1.0,
                  window: Optional[tuple] = None) -> np.ndarray:
    """Noise CAR(0,+5) for every event: no effect in it.

    Daily residuals share a market-wide shock (``common_share`` of variance),
    are serially correlated (``ar1``), heavy-tailed (Student t), and scaled by
    each issuer's own volatility. The expected-return intercept is estimated
    on the registered window and subtracted, so its estimation error is in.
    Events of one issuer with overlapping windows share returns because they
    are sums over the same simulated days.
    """

    issuers, sessions = len(design.sigma), design.sessions
    unit = np.sqrt((df - 2) / df)
    shock = rng.standard_t(df, size=sessions) * unit
    own = rng.standard_t(df, size=(issuers, sessions)) * unit
    e = np.sqrt(common_share) * shock[None, :] + np.sqrt(1 - common_share) * own
    if ar1:
        x = np.empty_like(e)
        x[:, 0] = e[:, 0]
        scale = np.sqrt(1 - ar1 ** 2)
        for t in range(1, sessions):
            x[:, t] = ar1 * x[:, t - 1] + scale * e[:, t]
    else:
        x = e
    ar = x * design.sigma[:, None]
    total = np.concatenate([np.zeros((issuers, 1)), np.cumsum(ar, axis=1)], axis=1)
    a, b = window or WINDOW
    j, p = design.issuer, design.position
    summed = total[j, p + b + 1] - total[j, p + a]
    est_lo, est_hi = p + ESTIMATION_WINDOW[0], p + ESTIMATION_WINDOW[1] + 1
    alpha_hat = (total[j, est_hi] - total[j, est_lo]) / (est_hi - est_lo)
    return event_vol * summed - (b - a + 1) * alpha_hat


class JointTest:
    """The registered H7 joint test on a fixed design (see ``h7._joint``)."""

    def __init__(self, design: Design):
        counts = pd.Series(design.category).value_counts()
        self.reference = counts.idxmax()
        self.dummies = [c for c in design.kept if c != self.reference]
        self.X = np.column_stack(
            [np.ones(len(design.category))]
            + [(design.category == c).astype(float) for c in self.dummies])
        self.R = np.zeros((len(self.dummies), len(self.dummies) + 1))
        self.R[:, 1:] = np.eye(len(self.dummies))
        self.clusters = [design.date, design.issuer]
        self.masks = {c: design.category == c for c in design.kept}
        self.design = design
        # Fewest date or issuer clusters carrying any one category. A candidate
        # correction (not registered) takes its degrees of freedom from here.
        self.small_cluster_dof = min(
            min(len(np.unique(design.date[m])), len(np.unique(design.issuer[m])))
            for m in self.masks.values()) - len(self.dummies)

    def __call__(self, y: np.ndarray) -> Dict[str, Any]:
        from scipy import stats as sps

        fit = stats.ols_cluster(self.X, y, self.clusters)
        test = stats.wald(fit, self.R)
        means = np.array([y[m].mean() for m in self.masks.values()])
        corrected = None
        if test.get("p") is not None and self.small_cluster_dof >= 1:
            corrected = float(sps.f.sf(test["f"], test["q"], self.small_cluster_dof))
        return {"p": test.get("p"), "p_corrected": corrected,
                "spread": float(means.max() - means.min()), "status": test.get("status")}

    def category_mean(self, y: np.ndarray, category: str) -> Optional[float]:
        mask = self.masks[category]
        result = stats.cluster_mean(y[mask], [self.design.date[mask], self.design.issuer[mask]])
        return result["p"]


def shifts(test: JointTest, scenario: str, effect: float) -> np.ndarray:
    """Planted category means for a scenario.

    ``one:<category>``  that category's mean is *effect*, the rest zero.
    ``linear``          means spaced evenly from -effect/2 to +effect/2 over
                        the kept categories, in order of size.
    Either way the true spread between the highest and lowest category mean is
    exactly *effect*.
    """

    design = test.design
    out = np.zeros(len(design.category))
    if scenario.startswith("one:"):
        out[test.masks[scenario[4:]]] = effect
    elif scenario == "linear":
        order = pd.Series(design.category).value_counts().index.tolist()
        levels = np.linspace(-effect / 2, effect / 2, len(order))
        for category, level in zip(order, levels):
            out[test.masks[category]] = level
    else:
        raise ValueError(scenario)
    return out


def run(structure: pd.DataFrame, calibration: Dict[str, Any], *, sessions: int,
        frame_anchors: int = 80, anchors_total: int = 80, simulations: int = 2000,
        projections: int = 20, common_share: Optional[float] = None, ar1: Optional[float] = None,
        event_vol: float = 1.0, recurrence: str = "crp", effects: Sequence[float] = EFFECTS,
        scenarios: Optional[Sequence[str]] = None, seed: int = 20261011,
        ingested: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Rejection rates of the registered test for each scenario and effect.

    Each simulation draws one noise vector and reuses it across effects and
    scenarios (common random numbers), so differences between rows are not
    simulation noise.
    """

    rng = np.random.default_rng(seed)
    common_share = calibration["common_daily_share"] if common_share is None else common_share
    ar1 = calibration["mean_ar1"] if ar1 is None else ar1
    per_projection = int(np.ceil(simulations / projections))
    tallies: Dict[tuple, Dict[str, float]] = {}
    described = []

    for _ in range(projections):
        design = project(structure, calibration, anchors_total=anchors_total,
                         frame_anchors=frame_anchors, sessions=sessions, rng=rng,
                         recurrence=recurrence, ingested=ingested)
        test = JointTest(design)
        counts = pd.Series(design.category).value_counts()
        described.append({"events": int(len(design.category)),
                          "dates": int(len(np.unique(design.date))),
                          "issuers": int(len(design.sigma)), "categories": len(design.kept),
                          **{f"n_{c}": int(counts[c]) for c in design.kept}})
        todo = list(scenarios) if scenarios is not None else (
            ["linear"] + [f"one:{c}" for c in design.kept])
        planted = {(s, e): shifts(test, s, e) for s in todo for e in effects
                   if s == "linear" or s[4:] in test.masks}
        for _ in range(per_projection):
            noise = simulate_cars(design, rng, common_share=common_share, ar1=ar1,
                                  event_vol=event_vol)
            for (scenario, effect), shift in planted.items():
                y = noise + shift
                outcome = test(y)
                tally = tallies.setdefault((scenario, effect), {
                    "runs": 0, "no_inference": 0, "nominal": 0, "holm": 0, "supported": 0,
                    "corrected_nominal": 0, "corrected_holm": 0, "corrected_supported": 0,
                    "spread_material": 0, "category_nominal": 0, "category_runs": 0})
                tally["runs"] += 1
                p = outcome["p"]
                if p is None:
                    tally["no_inference"] += 1
                    continue
                tally["nominal"] += p < ALPHA
                tally["holm"] += p < HOLM_ALPHA
                material = outcome["spread"] >= MATERIAL
                tally["spread_material"] += material
                tally["supported"] += (p < HOLM_ALPHA) and material
                pc = outcome["p_corrected"]
                if pc is not None:
                    tally["corrected_nominal"] += pc < ALPHA
                    tally["corrected_holm"] += pc < HOLM_ALPHA
                    tally["corrected_supported"] += (pc < HOLM_ALPHA) and material
                if scenario.startswith("one:"):
                    own = test.category_mean(y, scenario[4:])
                    tally["category_runs"] += 1
                    tally["category_nominal"] += bool(own is not None and own < ALPHA)

    rows = []
    for (scenario, effect), t in sorted(tallies.items()):
        n = t["runs"]
        row = {"scenario": scenario, "true_spread": effect, "anchors": anchors_total,
               "common_share": common_share, "ar1": ar1, "event_vol": event_vol,
               "recurrence": recurrence,
               "simulations": n, "no_inference_share": t["no_inference"] / n}
        for key in ("nominal", "holm", "supported", "corrected_nominal", "corrected_holm",
                    "corrected_supported", "spread_material"):
            rate = t[key] / n
            row[f"power_{key}"] = rate
            row[f"se_{key}"] = float(np.sqrt(rate * (1 - rate) / n))
        if t["category_runs"]:
            rate = t["category_nominal"] / t["category_runs"]
            row["power_category_own_test"] = rate
            row["se_category_own_test"] = float(np.sqrt(rate * (1 - rate) / t["category_runs"]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["designs"] = pd.DataFrame(described).mean(numeric_only=True).to_dict()
    return table


def minimum_detectable(table: pd.DataFrame, scenario: str,
                       column: str = "power_corrected_supported",
                       target: float = 0.8) -> Optional[float]:
    """Smallest true spread with power at least *target*, by linear
    interpolation between simulated effects. None if never reached."""

    part = table[table["scenario"] == scenario].sort_values("true_spread")
    x, y = part["true_spread"].to_numpy(), part[column].to_numpy()
    for i in range(1, len(x)):
        if y[i] >= target > y[i - 1]:
            return float(x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return float(x[0]) if len(y) and y[0] >= target else None
