"""Exploratory sensitivity analysis for session-aligned sentiment signals.

This module compares every published signal variant on equal footing.  It is
descriptive and exploratory: it does not select a preferred specification,
tune a threshold, or represent a trading strategy.

The predictive target is the fractional close-to-close return from a signal's
session to the immediately following market session.  Targets are constructed
on the complete, ordered price table *before* signals are joined, so gaps in
the signal series cannot accidentally turn a multi-session move into a
"next-session" return.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import database


VARIANTS = (
    "simple_mean",
    "relevance_weighted",
    "intensity_relevance_weighted",
    "full_weighted",
)
LOW_SAMPLE_THRESHOLD = 30
EXPLORATORY_LABEL = "exploratory; not a validated predictive model or strategy"


def _date_string(value: Any, *, field: str) -> str:
    """Normalize a database date-like value without changing its timezone."""

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        raise ValueError(f"{field} contains a missing date")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains an invalid date: {value!r}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field} contains an invalid date: {value!r}")
    return timestamp.date().isoformat()


def _signal_date_column(frame: pd.DataFrame) -> str:
    for candidate in ("signal_date", "session_date", "date"):
        if candidate in frame.columns:
            return candidate
    raise KeyError("signal variants need a signal_date, session_date, or date column")


def _prepare_signals(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw)
    date_column = _signal_date_column(raw)
    missing = [variant for variant in VARIANTS if variant not in raw.columns]
    if missing:
        raise KeyError(f"signal variants missing required columns: {', '.join(missing)}")

    signals = raw[[date_column, *VARIANTS]].copy()
    signals.rename(columns={date_column: "signal_date"}, inplace=True)
    signals["signal_date"] = [
        _date_string(value, field="signal_date") for value in signals["signal_date"]
    ]
    if signals["signal_date"].duplicated().any():
        duplicates = sorted(
            signals.loc[signals["signal_date"].duplicated(False), "signal_date"].unique()
        )
        raise ValueError(f"signal variants contain duplicate session dates: {duplicates}")

    for variant in VARIANTS:
        signals[variant] = pd.to_numeric(signals[variant], errors="coerce")
        signals.loc[~np.isfinite(signals[variant]), variant] = np.nan
    return signals.sort_values("signal_date", kind="stable").reset_index(drop=True)


def _prepare_price_targets(raw: pd.DataFrame) -> pd.DataFrame:
    """Build next-session targets before any signal merge occurs."""

    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw)
    missing = [column for column in ("date", "close") if column not in raw.columns]
    if missing:
        raise KeyError(f"prices missing required columns: {', '.join(missing)}")

    prices = raw[["date", "close"]].copy()
    prices["price_date"] = [
        _date_string(value, field="price date") for value in prices.pop("date")
    ]
    if prices["price_date"].duplicated().any():
        duplicates = sorted(
            prices.loc[prices["price_date"].duplicated(False), "price_date"].unique()
        )
        raise ValueError(f"prices contain duplicate market dates: {duplicates}")

    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    invalid_close = ~np.isfinite(prices["close"]) | (prices["close"] <= 0)
    if invalid_close.any():
        invalid_dates = prices.loc[invalid_close, "price_date"].tolist()
        raise ValueError(f"prices contain missing, non-finite, or non-positive closes: {invalid_dates}")

    prices.sort_values("price_date", kind="stable", inplace=True)
    prices.reset_index(drop=True, inplace=True)
    prices["next_session_date"] = prices["price_date"].shift(-1)
    prices["next_session_close"] = prices["close"].shift(-1)
    prices["next_session_return"] = prices["next_session_close"] / prices["close"] - 1.0
    return prices


def _finite_pair(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    return left_values[mask], right_values[mask]


def _pearson(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = np.asarray(list(left), dtype=float)
    y = np.asarray(list(right), dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _sample_fields(n: int) -> dict[str, Any]:
    low = n < LOW_SAMPLE_THRESHOLD
    return {
        "low_sample_size": low,
        "sample_size_label": (
            f"low sample size (n={n} < {LOW_SAMPLE_THRESHOLD})"
            if low
            else f"n={n}; above the report's descriptive low-sample flag"
        ),
    }


def _signal_correlations(signals: pd.DataFrame) -> dict[str, Any]:
    matrix: dict[str, dict[str, float | None]] = {variant: {} for variant in VARIANTS}
    sample_sizes: dict[str, dict[str, int]] = {variant: {} for variant in VARIANTS}
    pairs: list[dict[str, Any]] = []

    for left_index, left_name in enumerate(VARIANTS):
        for right_index, right_name in enumerate(VARIANTS):
            left, right = _finite_pair(signals[left_name], signals[right_name])
            matrix[left_name][right_name] = _pearson(left, right)
            sample_sizes[left_name][right_name] = len(left)
            if right_index > left_index:
                pairs.append(
                    {
                        "variant_a": left_name,
                        "variant_b": right_name,
                        "n": len(left),
                        "pearson_r": _pearson(left, right),
                        **_sample_fields(len(left)),
                    }
                )
    return {"matrix": matrix, "sample_sizes": sample_sizes, "pairs": pairs}


def _directional_agreement(signals: pd.DataFrame) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(VARIANTS):
        for right_name in VARIANTS[left_index + 1 :]:
            left, right = _finite_pair(signals[left_name], signals[right_name])
            left_direction = np.sign(left)
            right_direction = np.sign(right)
            n = len(left)
            agreement_count = int(np.sum(left_direction == right_direction))
            nonzero = (left_direction != 0) & (right_direction != 0)
            nonzero_n = int(np.sum(nonzero))
            nonzero_agreement_count = int(
                np.sum(left_direction[nonzero] == right_direction[nonzero])
            )
            results.append(
                {
                    "variant_a": left_name,
                    "variant_b": right_name,
                    "n_comparable": n,
                    "agreement_count_including_zero": agreement_count,
                    "agreement_rate_including_zero": agreement_count / n if n else None,
                    "n_both_nonzero": nonzero_n,
                    "agreement_count_both_nonzero": nonzero_agreement_count,
                    "agreement_rate_both_nonzero": (
                        nonzero_agreement_count / nonzero_n if nonzero_n else None
                    ),
                    **_sample_fields(n),
                }
            )
    return results


def _distributions(signals: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        values = signals[variant].dropna().astype(float)
        n = len(values)
        row: dict[str, Any] = {
            "variant": variant,
            "n": n,
            "mean": float(values.mean()) if n else None,
            "sample_std": float(values.std(ddof=1)) if n > 1 else None,
            "min": float(values.min()) if n else None,
            "q25": float(values.quantile(0.25)) if n else None,
            "median": float(values.median()) if n else None,
            "q75": float(values.quantile(0.75)) if n else None,
            "max": float(values.max()) if n else None,
            "positive_share": float((values > 0).mean()) if n else None,
            "negative_share": float((values < 0).mean()) if n else None,
            "zero_share": float((values == 0).mean()) if n else None,
            **_sample_fields(n),
        }
        rows.append(row)
    return rows


def _predictive_metrics(aligned: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        signals, returns = _finite_pair(aligned[variant], aligned["next_session_return"])
        n = len(signals)
        signal_direction = np.sign(signals)
        return_direction = np.sign(returns)
        actionable = (signal_direction != 0) & (return_direction != 0)
        n_directional = int(np.sum(actionable))
        hits = int(np.sum(signal_direction[actionable] == return_direction[actionable]))
        rows.append(
            {
                "variant": variant,
                "analysis_type": EXPLORATORY_LABEL,
                "target": "next-session fractional close-to-close return",
                "n_correlation": n,
                "pearson_r": _pearson(signals, returns),
                "n_directional": n_directional,
                "directional_hits": hits,
                "directional_hit_rate": hits / n_directional if n_directional else None,
                "zero_signal_count": int(np.sum(signal_direction == 0)),
                "zero_return_count": int(np.sum(return_direction == 0)),
                "directional_rule": (
                    "sign(signal) must equal sign(next return); zero signals and "
                    "zero returns are excluded"
                ),
                **_sample_fields(n),
            }
        )
    return rows


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values to strict-JSON-compatible Python values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def run_sensitivity_analysis(
    db_path: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run and optionally persist the four-variant exploratory report.

    The returned dictionary is strict-JSON-compatible.  ``output_path`` always
    receives UTF-8 JSON, irrespective of its filename extension.
    """

    signals = _prepare_signals(database.get_signal_variants(db_path=db_path))
    price_targets = _prepare_price_targets(database.get_prices(db_path=db_path))

    # This is deliberately the first point where the two datasets meet.
    aligned = signals.merge(
        price_targets,
        left_on="signal_date",
        right_on="price_date",
        how="left",
        validate="one_to_one",
    )
    aligned.sort_values("signal_date", kind="stable", inplace=True)

    report = {
        "metadata": {
            "analysis_type": EXPLORATORY_LABEL,
            "preferred_variant": None,
            "variants": list(VARIANTS),
            "return_definition": (
                "next session close / current session close - 1; computed on the "
                "complete ordered price table before joining signals"
            ),
            "return_units": "fractional (0.01 means 1%)",
            "low_sample_threshold": LOW_SAMPLE_THRESHOLD,
            "signal_rows": len(signals),
            "price_rows": len(price_targets),
            "signals_matching_a_price_session": int(aligned["price_date"].notna().sum()),
            "signals_with_next_session_target": int(
                aligned["next_session_return"].notna().sum()
            ),
            "interpretation_note": (
                "All variants are reported without model selection. Correlations and "
                "hit rates are exploratory and do not establish predictability."
            ),
        },
        "signal_correlations": _signal_correlations(signals),
        "directional_agreement": _directional_agreement(signals),
        "distributions": _distributions(signals),
        "predictive": _predictive_metrics(aligned),
        "aligned_observations": aligned[
            [
                "signal_date",
                "price_date",
                "next_session_date",
                "close",
                "next_session_close",
                "next_session_return",
                *VARIANTS,
            ]
        ].to_dict("records"),
    }
    safe_report = _json_safe(report)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(safe_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return safe_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare all session-aligned sentiment signal variants."
    )
    parser.add_argument(
        "--db",
        default=getattr(database, "DB_PATH", "finance_sentiment.db"),
        help="SQLite database path (default: database.DB_PATH)",
    )
    parser.add_argument(
        "--output",
        help="Optional UTF-8 JSON report path; JSON is printed when omitted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_sensitivity_analysis(args.db, args.output)
    if args.output is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"Sensitivity report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
