"""What every hypothesis shares: its registered form, its inputs, and the rule
that turns an estimate into a verdict.

The verdict rule is the same for all nine and is fixed here, before any
outcome is read:

``data_insufficient``
    A registered data requirement or sample-size rule is not met. Nothing is
    estimated. This is the answer whenever the honest alternative would be a
    number nobody should believe.
``supported``
    The primary estimate has the hypothesised sign, its **Holm-adjusted**
    p-value (family of nine, whether or not all nine could run) is below
    ``ALPHA``, and it is at least the registered material effect.
``execution_not_verifiable``
    As ``supported``, for a hypothesis whose claim depends on trading at a
    price the data cannot confirm was tradable.
``not_supported``
    The 95% interval, read in the hypothesised direction, lies entirely below
    the material effect: an effect that size can be ruled out.
``inconclusive``
    Everything else, including a significant effect too small to matter and an
    unadjusted p-value below 0.05 that does not survive adjustment.

A run on synthetic or fixture data, or on an incomplete sample frame, can
produce any of these, but ``claim_allowed`` is false and the report says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stock_research import stats
from stock_research.config import (
    ALPHA, MIN_EVENT_DATES_PER_GROUP, MIN_EVENTS_PER_GROUP,
)
from stock_research.data.prices import ORIGIN_REAL, Panel

SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
INCONCLUSIVE = "inconclusive"
DATA_INSUFFICIENT = "data_insufficient"
EXECUTION_NOT_VERIFIABLE = "execution_not_verifiable"
STATUSES = (SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE, DATA_INSUFFICIENT, EXECUTION_NOT_VERIFIABLE)

FAMILY_SIZE = 9          # H1..H9: the registered primary family


@dataclass(frozen=True)
class Spec:
    """A hypothesis as registered. Hashed into the protocol."""

    id: str
    title: str
    claim: str
    null: str
    alternative: str
    eligibility: str
    event_time: str
    outcome: str
    estimand: str
    direction: int                 # +1 or -1: sign of the primary estimate under H1
    material_effect: float
    benchmark: str
    controls: Tuple[str, ...]
    inference: str
    sufficiency: Dict[str, Any]
    requires: Tuple[str, ...]      # dataset keys that must be available
    multiple_testing: str = (
        "primary test: Holm across the nine registered hypotheses (family size "
        "fixed at nine); exploratory tests: Benjamini-Hochberg within the hypothesis"
    )
    sensitivity: Tuple[str, ...] = ()
    execution_dependent: bool = False
    kind: str = "primary"
    #: Every module-level constant the hypothesis uses. In the spec so that
    #: changing one changes the protocol hash.
    parameters: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Context:
    """Everything a hypothesis may read. Nothing else is in scope."""

    panel: Panel
    kap_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    news_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    english_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    size: Optional[pd.DataFrame] = None        # point-in-time: ticker, date, market_cap
    availability: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sealed: Optional[str] = None
    frame_complete: bool = True
    limit_rules: Optional[List[Dict[str, Any]]] = None

    @property
    def origin(self) -> str:
        return self.panel.origin

    @property
    def claim_allowed(self) -> bool:
        return self.origin == ORIGIN_REAL and self.frame_complete

    def available(self, key: str) -> Tuple[bool, str]:
        entry = self.availability.get(key)
        if entry is None:
            return False, f"{key}: no availability record"
        return bool(entry.get("available")), str(entry.get("reason", ""))


def base_result(spec: Spec, ctx: Context) -> Dict[str, Any]:
    return {
        "hypothesis": spec.id, "title": spec.title, "claim": spec.claim,
        "data_origin": ctx.origin, "frame_complete": ctx.frame_complete,
        "claim_allowed": ctx.claim_allowed, "status": None,
        "sufficiency": {"met": True, "reasons": [], "counts": {}},
        "primary": None, "exploratory": [], "sensitivity": [],
        "execution": None, "limitations": [], "tables": {},
    }


def missing_requirements(spec: Spec, ctx: Context) -> List[str]:
    reasons = []
    for key in spec.requires:
        ok, reason = ctx.available(key)
        if not ok:
            reasons.append(f"{key} unavailable: {reason}")
    return reasons


def insufficient(result: Dict[str, Any], reasons: Sequence[str],
                 counts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result["status"] = DATA_INSUFFICIENT
    result["sufficiency"] = {"met": False, "reasons": list(reasons),
                             "counts": dict(counts or {})}
    return result


def group_sufficient(n_events: int, n_dates: int) -> bool:
    return n_events >= MIN_EVENTS_PER_GROUP and n_dates >= MIN_EVENT_DATES_PER_GROUP


def primary_block(spec: Spec, *, estimate: Optional[float], se: Optional[float],
                  ci: Optional[Sequence[float]], p: Optional[float], n: int,
                  clusters: Any, inference_status: str = "ok",
                  extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "estimand": spec.estimand, "direction": spec.direction,
        "material_effect": spec.material_effect, "estimate": estimate, "se": se,
        "ci": list(ci) if ci is not None else None, "p": p, "p_holm": None,
        "n": n, "clusters": clusters, "inference_status": inference_status,
        **(extra or {}),
    }


def decide(spec: Spec, primary: Optional[Dict[str, Any]]) -> str:
    """The registered verdict rule. Needs ``p_holm`` to be filled in first."""

    if primary is None or primary.get("estimate") is None or primary.get("p_holm") is None:
        return INCONCLUSIVE
    if primary.get("joint_test"):
        # A joint test has no interval to rule an effect out with, so it can
        # support a difference or leave it open; it cannot reject one.
        large_enough = primary["estimate"] >= spec.material_effect
        return SUPPORTED if primary["p_holm"] < ALPHA and large_enough else INCONCLUSIVE
    if primary.get("ci") is None:
        return INCONCLUSIVE
    signed = spec.direction * primary["estimate"]
    bounds = sorted(spec.direction * b for b in primary["ci"])
    if primary["p_holm"] < ALPHA and signed >= spec.material_effect:
        return EXECUTION_NOT_VERIFIABLE if spec.execution_dependent else SUPPORTED
    if bounds[1] < spec.material_effect:
        return NOT_SUPPORTED
    return INCONCLUSIVE


# -- Controls known strictly before day 0 --------------------------------------
def prior_features(panel: Panel, ticker: str, day0: Optional[str]) -> Dict[str, Optional[float]]:
    """Controls built only from sessions before day 0."""

    blank = {"prior_ret_5": None, "volatility_20": None, "log_turnover_20": None,
             "volume_ratio": None}
    frame = panel.frame(ticker)
    position = panel.calendar.index(day0) if day0 else None
    if frame is None or position is None or position < 21:
        return blank
    ret = frame["ret"].to_numpy()[position - 20:position]
    turnover = frame["turnover"].to_numpy()[position - 20:position]
    volume = frame["volume"].to_numpy()[position - 20:position]
    last5 = ret[-5:]
    finite_turnover = turnover[np.isfinite(turnover) & (turnover > 0)]
    finite_volume = volume[np.isfinite(volume) & (volume > 0)]
    return {
        "prior_ret_5": float(last5.sum()) if np.isfinite(last5).all() else None,
        "volatility_20": float(np.nanstd(ret, ddof=1)) if np.isfinite(ret).sum() >= 10 else None,
        "log_turnover_20": float(np.log(finite_turnover.mean())) if len(finite_turnover) >= 10 else None,
        "volume_ratio": (float(volume[-1] / finite_volume.mean())
                         if len(finite_volume) >= 10 and np.isfinite(volume[-1]) else None),
    }


def regress(table: pd.DataFrame, y: str, xs: Sequence[str], cluster: Sequence[str],
            *, intercept: bool = True) -> Dict[str, Any]:
    """Cluster-robust OLS on the complete rows of *table*.

    Rows with a missing value in any used column are dropped, and the number
    dropped is returned: a silent complete-case filter is how a sample quietly
    stops being the sample that was described.
    """

    columns = [y, *xs, *cluster]
    usable = table.dropna(subset=columns)
    names = (["const"] if intercept else []) + list(xs)
    info = {"n_input": int(len(table)), "n_dropped_missing": int(len(table) - len(usable)),
            "y": y, "names": names}
    if usable.empty:
        return {**info, "status": "no_complete_rows", "n": 0}
    X = usable[list(xs)].to_numpy(dtype=float)
    if intercept:
        X = np.column_stack([np.ones(len(usable)), X])
    fit = stats.ols_cluster(X, usable[y].to_numpy(dtype=float),
                            [usable[c].to_numpy() for c in cluster], names=names)
    return {**info, **fit}


def coefficient(fit: Dict[str, Any], name: str) -> Dict[str, Any]:
    """One coefficient's row from a fit, with None where inference is absent."""

    out = {"name": name, "estimate": None, "se": None, "ci": None, "p": None}
    if "coef" not in fit or name not in fit.get("names", []):
        return out
    index = fit["names"].index(name)
    out["estimate"] = fit["coef"][index]
    if fit.get("status") == "ok":
        out.update({"se": fit["se"][index], "ci": fit["ci"][index], "p": fit["p"][index]})
    return out


def mean_test(values: pd.Series, dates: pd.Series,
              tickers: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Clustered mean of the non-missing values.

    With *tickers*, clustering is two-way: by date and by issuer. That is the
    form to use whenever one issuer can contribute several events whose
    outcome windows overlap, because those outcomes share returns. Clustering
    by date alone then rejects a true null about half the time (measured in
    ``test_overlapping_windows_need_issuer_clustering``).
    """

    keep = values.notna() & dates.notna()
    clusters = [dates[keep].to_numpy()]
    if tickers is not None:
        keep = keep & tickers.notna()
        clusters = [dates[keep].to_numpy(), tickers[keep].to_numpy()]
    result = stats.cluster_mean(values[keep].to_numpy(dtype=float), clusters)
    result["dates"] = int(dates[keep].nunique())
    result["clustering"] = "date and issuer" if tickers is not None else "date"
    return result


#: Clustering used for a primary test whenever outcome windows can overlap.
TWO_WAY = ["day0", "ticker"]


def chronological_split(table: pd.DataFrame, date_column: str, train_share: float,
                        embargo_sessions: int, calendar) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Train on the early dates, test on the late ones, with a gap between.

    An event's outcome is not known on its day 0; it is known when its window
    closes. A model fitted "as of" the first test date may therefore only use
    training events whose windows had closed by then. ``embargo_sessions`` is
    the length of the outcome window: training events within that many
    sessions before the first test date are dropped.
    """

    dates = sorted(table[date_column].unique())
    cut = int(len(dates) * train_share)
    info = {"train_dates": cut, "test_dates": len(dates) - cut,
            "embargo_sessions": embargo_sessions, "embargoed_rows": 0}
    if cut == 0 or cut >= len(dates):
        return table.iloc[0:0], table.iloc[0:0], info
    first_test = calendar.index(dates[cut])
    position = table[date_column].map(calendar.index)
    is_test = table[date_column].isin(dates[cut:])
    usable_train = ~is_test & (position <= first_test - 1 - embargo_sessions)
    info["embargoed_rows"] = int((~is_test & ~usable_train).sum())
    return table[usable_train], table[is_test], info
