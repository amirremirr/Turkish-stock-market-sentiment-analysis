"""Versioned control sets and residual (excess) returns.

The distinction this module exists to enforce: a control is only *tradable* if
its value was observable at the moment the position was assumed to open.

EEM and Brent close at US session times, hours after Borsa Istanbul. Their
same-day close is therefore **not** observable when a Turkish position opens the
next morning at 10:00 Istanbul — using it would import tomorrow's information
into today's model and produce a residual that no one could have computed.

So control sets come in two clearly separated kinds:

``tradable``       every input lagged to a value published before the execution
                   timestamp. These may be used for execution-sensitive work.
``contemporaneous`` same-session values, useful for describing how much of a
                   move was market-wide, and explicitly **not** tradable. They
                   are labelled that way in the schema so a later stage cannot
                   pick them up by accident.

Betas are estimated on a **rolling prior window** — never the full sample. A
full-sample beta applied to its own estimation period leaks the future into
every historical residual, which is the same defect as a full-sample mean in
abnormal tone.

No model selection happens here, and no split is random. This module builds
inputs; choosing among them is a later phase's job under a frozen protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

CONTROL_SET_VERSION = "control-sets-v1"

KIND_TRADABLE = "tradable"
KIND_CONTEMPORANEOUS = "contemporaneous_descriptive"

# Pre-specified control sets. Fixed in advance so a later stage cannot choose
# whichever set happens to flatter a result.
CONTROL_SETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "kind": KIND_TRADABLE,
        "controls": (),
        "description": "Raw return with no adjustment; the baseline comparison.",
    },
    "em_lagged": {
        "kind": KIND_TRADABLE,
        "controls": ("EEM_lag1",),
        "description": (
            "Emerging-market equity return through the previous US close, "
            "which is public before a Turkish open."
        ),
    },
    "em_oil_fx_lagged": {
        "kind": KIND_TRADABLE,
        "controls": ("EEM_lag1", "BZ=F_lag1", "USDTRY=X_lag1"),
        "description": (
            "Emerging-market equity, Brent and USD/TRY, each lagged one "
            "session so every input predates the assumed execution."
        ),
    },
    "em_contemporaneous": {
        "kind": KIND_CONTEMPORANEOUS,
        "controls": ("EEM",),
        "description": (
            "Same-session emerging-market return. Describes how much of a move "
            "was market-wide. NOT tradable: the US close postdates the Turkish "
            "session."
        ),
    },
    "em_oil_fx_contemporaneous": {
        "kind": KIND_CONTEMPORANEOUS,
        "controls": ("EEM", "BZ=F", "USDTRY=X"),
        "description": (
            "Same-session emerging-market, Brent and USD/TRY returns. "
            "Descriptive risk adjustment only; NOT tradable."
        ),
    },
}

DEFAULT_ESTIMATION_WINDOW = 60
DEFAULT_MIN_OBSERVATIONS = 30


def is_tradable(control_set_id: str) -> bool:
    """Whether a control set may be used for execution-sensitive research."""

    definition = CONTROL_SETS.get(control_set_id)
    return bool(definition) and definition["kind"] == KIND_TRADABLE


@dataclass(frozen=True)
class ControlAvailability:
    """Whether a control's value was observable at the execution moment."""

    control_name: str
    value: Optional[float]
    available: bool
    reason: Optional[str]


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_control_panel(
    factor_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Return ``date -> {control_name: return}`` including lagged variants.

    Lagged names are materialised explicitly rather than computed at use time,
    so a caller cannot silently read an unlagged value where a lagged one was
    intended.
    """

    by_symbol: Dict[str, List[Tuple[str, float]]] = {}
    for row in factor_rows:
        value = _finite(row.get("daily_return"))
        date = row.get("date")
        symbol = row.get("symbol")
        if value is None or not date or not symbol:
            continue
        by_symbol.setdefault(str(symbol), []).append((str(date), value))

    panel: Dict[str, Dict[str, float]] = {}
    for symbol, series in by_symbol.items():
        series.sort()
        for index, (date, value) in enumerate(series):
            panel.setdefault(date, {})[symbol] = value
            if index + 1 < len(series):
                # The value observable on the *next* session.
                next_date = series[index + 1][0]
                panel.setdefault(next_date, {})[f"{symbol}_lag1"] = value
    return panel


def control_availability(
    control_set_id: str,
    date: Optional[str],
    panel: Dict[str, Dict[str, float]],
) -> List[ControlAvailability]:
    """Report each control's value and whether it was observable."""

    definition = CONTROL_SETS.get(control_set_id)
    if definition is None:
        raise KeyError(f"unknown control set: {control_set_id!r}")

    available: List[ControlAvailability] = []
    row = panel.get(str(date), {}) if date else {}
    for name in definition["controls"]:
        value = row.get(name)
        available.append(ControlAvailability(
            control_name=name,
            value=value,
            available=value is not None,
            reason=None if value is not None else "control_value_missing",
        ))
    return available


def _ols(y: List[float], columns: List[List[float]]) -> Optional[List[float]]:
    """Least squares with an intercept, via numpy. None if degenerate."""

    import numpy as np

    if not y or len(y) <= len(columns) + 1:
        return None
    design = np.column_stack([np.ones(len(y))] + [np.asarray(c) for c in columns])
    target = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(target)):
        return None
    try:
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return [float(value) for value in coefficients]


def compute_residual_returns(
    market_returns: Sequence[Tuple[str, float]],
    panel: Dict[str, Dict[str, float]],
    control_set_id: str,
    *,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> Dict[str, Dict[str, Any]]:
    """Residual returns from a rolling **prior-window** control model.

    For each date the model is fitted on the preceding ``estimation_window``
    sessions and applied to that date only. The date being described never
    contributes to its own coefficients; below ``min_observations`` the residual
    is NULL rather than estimated from too little history.
    """

    definition = CONTROL_SETS.get(control_set_id)
    if definition is None:
        raise KeyError(f"unknown control set: {control_set_id!r}")
    names = list(definition["controls"])

    ordered = sorted(market_returns)
    results: Dict[str, Dict[str, Any]] = {}

    if not names:
        # The "none" set: residual is the raw return, stated explicitly.
        for date, value in ordered:
            results[date] = {
                "control_set_id": control_set_id,
                "kind": definition["kind"],
                "raw_return": value,
                "fitted": 0.0,
                "residual": value,
                "coefficients": None,
                "estimation_observations": 0,
                "estimation_window_end": None,
                "version": CONTROL_SET_VERSION,
            }
        return results

    for index, (date, value) in enumerate(ordered):
        history = ordered[max(0, index - estimation_window):index]
        rows = [
            (day, ret) for day, ret in history
            if all(panel.get(day, {}).get(name) is not None for name in names)
        ]
        current = [panel.get(date, {}).get(name) for name in names]

        if len(rows) < min_observations or any(v is None for v in current):
            results[date] = {
                "control_set_id": control_set_id,
                "kind": definition["kind"],
                "raw_return": value,
                "fitted": None,
                "residual": None,
                "coefficients": None,
                "estimation_observations": len(rows),
                "estimation_window_end": rows[-1][0] if rows else None,
                "version": CONTROL_SET_VERSION,
            }
            continue

        y = [ret for _, ret in rows]
        columns = [
            [panel[day][name] for day, _ in rows] for name in names
        ]
        coefficients = _ols(y, columns)
        if coefficients is None:
            results[date] = {
                "control_set_id": control_set_id, "kind": definition["kind"],
                "raw_return": value, "fitted": None, "residual": None,
                "coefficients": None,
                "estimation_observations": len(rows),
                "estimation_window_end": rows[-1][0],
                "version": CONTROL_SET_VERSION,
            }
            continue

        fitted = coefficients[0] + sum(
            beta * float(control)
            for beta, control in zip(coefficients[1:], current)
        )
        results[date] = {
            "control_set_id": control_set_id,
            "kind": definition["kind"],
            "raw_return": value,
            "fitted": fitted,
            "residual": value - fitted,
            "coefficients": ",".join(f"{c:.8f}" for c in coefficients),
            "estimation_observations": len(rows),
            "estimation_window_end": rows[-1][0],
            "version": CONTROL_SET_VERSION,
        }
    return results
