"""Run a small, deterministic, fully offline sentiment-signal demo.

The demo deliberately consumes cached sentiment fields from committed sample
files.  It never initializes a scorer, reads the private project database, or
makes a network request.  Its primary result is the unweighted ``simple_mean``;
the three weighted variants are displayed only as sensitivity calculations.

Run from the repository root::

    python -m scripts.demo

Use ``--output-dir`` to keep generated artifacts somewhere other than the
default ``demo_output`` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

# Select the non-interactive backend before importing pyplot.  This makes the
# command work in CI and on machines without a display server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from aggregation.signals import compute_signal_variants
from trading_calendar import assign_trading_session


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEADLINES = REPOSITORY_ROOT / "sample_data" / "demo_headlines.csv"
DEFAULT_PRICES = REPOSITORY_ROOT / "sample_data" / "demo_bist100_prices.csv"
VARIANTS = (
    "simple_mean",
    "relevance_weighted",
    "intensity_relevance_weighted",
    "full_weighted",
)
RESULT_COLUMNS = (
    "signal_date",
    *VARIANTS,
    "headline_count",
    "positive_share",
    "negative_share",
    "neutral_share",
    "dispersion",
    "source_count",
    "event_count",
    "close",
    "same_session_return",
    "next_session_return",
)
CLASSIFICATIONS = (
    "excluded",
    "failed",
    "missing",
    "explicit_neutral",
    "scored_non_neutral",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _optional_float(value: object) -> float | None:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise ValueError(f"expected a number, received {value!r}") from exc
    return parsed


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _classify(row: dict[str, Any]) -> str:
    """Return one mutually exclusive audit state for a headline."""

    if row["is_excluded"]:
        return "excluded"
    if row["processing_status"] == "failed":
        return "failed"
    if row["processing_status"] != "scored" or not _finite(row["sentiment_score"]):
        return "missing"
    if row["sentiment_label"] == "neutral" and row["sentiment_score"] == 0.0:
        return "explicit_neutral"
    return "scored_non_neutral"


def _prepare_headlines(path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_csv(path)
    required = {
        "observation_id",
        "title",
        "source",
        "published_timestamp",
        "published_date",
        "category",
        "event_id",
        "processing_status",
        "sentiment_score",
        "sentiment_label",
        "p_positive",
        "p_neutral",
        "p_negative",
        "relevance",
        "is_excluded",
        "exclusion_reason",
        "model_name",
    }
    missing_columns = sorted(required - set(raw_rows[0] if raw_rows else ()))
    if missing_columns:
        raise ValueError(f"headline sample is missing columns: {', '.join(missing_columns)}")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        timestamp = raw["published_timestamp"].strip() or None
        published_date = raw["published_date"].strip() or None
        assignment = assign_trading_session(timestamp, published_date)
        local_timestamp = assignment.published_at_istanbul
        row: dict[str, Any] = {
            **raw,
            "processing_status": raw["processing_status"].strip().lower(),
            "sentiment_label": raw["sentiment_label"].strip().lower(),
            "sentiment_score": _optional_float(raw["sentiment_score"]),
            "p_positive": _optional_float(raw["p_positive"]),
            "p_neutral": _optional_float(raw["p_neutral"]),
            "p_negative": _optional_float(raw["p_negative"]),
            "relevance": _optional_float(raw["relevance"]),
            "is_excluded": _truthy(raw["is_excluded"]),
            "signal_date": assignment.signal_date,
            "timing_bucket": assignment.timing_bucket,
            "published_hour": local_timestamp.hour if local_timestamp else None,
        }
        row["audit_classification"] = _classify(row)
        rows.append(row)
    return rows


def _score_component_violations(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Validate legacy-shaped components without treating them as probabilities."""
    violations: list[str] = []
    for row in rows:
        if row["processing_status"] != "scored":
            continue
        components = (row["p_positive"], row["p_neutral"], row["p_negative"])
        valid = all(_finite(value) and 0.0 <= value <= 1.0 for value in components)
        if not valid or not math.isclose(sum(components), 1.0, abs_tol=1e-9):
            violations.append(str(row["observation_id"]))
    return violations


def _eligible_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["audit_classification"] in {"explicit_neutral", "scored_non_neutral"}
    ]


def _session_signals(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _eligible_rows(rows):
        grouped.setdefault(row["signal_date"], []).append(row)

    signals: list[dict[str, Any]] = []
    for session_date in sorted(grouped):
        result = compute_signal_variants(grouped[session_date])
        signals.append({"signal_date": session_date, **result})
    return signals


def _load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, dtype={"date": "string"})
    required = {"date", "close"}
    if not required.issubset(prices.columns):
        raise ValueError("price sample must contain date and close columns")
    prices = prices.loc[:, ["date", "close"]].copy()
    prices["date"] = prices["date"].astype(str)
    prices["close"] = pd.to_numeric(prices["close"], errors="raise")
    prices = prices.sort_values("date", kind="stable").reset_index(drop=True)
    if prices.empty:
        raise ValueError("price sample is empty")
    if prices["date"].duplicated().any():
        raise ValueError("price sample contains duplicate session dates")
    if (~prices["close"].map(math.isfinite)).any() or (prices["close"] <= 0).any():
        raise ValueError("price closes must be finite and positive")

    # This ordering is methodologically important: derive D+1 from every
    # consecutive exchange session, then join the sparse news-signal dates.
    prices["same_session_return"] = prices["close"].pct_change()
    prices["next_session_return"] = prices["close"].shift(-1) / prices["close"] - 1.0
    return prices


def _align_signals_and_returns(
    signals: list[dict[str, Any]], prices: pd.DataFrame
) -> pd.DataFrame:
    signal_frame = pd.DataFrame(signals)
    if signal_frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    aligned = signal_frame.merge(
        prices,
        left_on="signal_date",
        right_on="date",
        how="left",
        validate="one_to_one",
    ).drop(columns="date")
    return aligned.loc[:, RESULT_COLUMNS].sort_values("signal_date").reset_index(drop=True)


def _audit(
    rows: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    prices: pd.DataFrame,
    aligned: pd.DataFrame,
) -> dict[str, Any]:
    classifications = Counter(row["audit_classification"] for row in rows)
    classification_counts = {
        name: int(classifications.get(name, 0)) for name in CLASSIFICATIONS
    }
    timing_counts = dict(sorted(Counter(row["timing_bucket"] for row in rows).items()))
    duplicate_ids = len(rows) - len({row["observation_id"] for row in rows})
    component_violations = _score_component_violations(rows)
    eligible = _eligible_rows(rows)
    eligible_ids = {row["observation_id"] for row in eligible}
    excluded_from_signal_ids = {
        row["observation_id"] for row in rows if row["audit_classification"] == "excluded"
    }
    unresolved_ids = {
        row["observation_id"]
        for row in rows
        if row["audit_classification"] in {"failed", "missing"}
    }
    signal_input_count = sum(int(signal["input_count"]) for signal in signals)
    mapped_price_dates = set(prices["date"].astype(str))
    missing_price_sessions = sorted(
        str(date) for date in aligned.loc[aligned["close"].isna(), "signal_date"]
    )
    missing_target_sessions = sorted(
        str(date)
        for date in aligned.loc[aligned["next_session_return"].isna(), "signal_date"]
    )

    checks = [
        {
            "name": "unique_observation_ids",
            "passed": duplicate_ids == 0,
            "details": {"duplicate_count": duplicate_ids},
        },
        {
            "name": "cached_score_component_shape",
            "passed": not component_violations,
            "details": {
                "violating_observation_ids": component_violations,
                "interpretation": "synthetic compatibility fields, not calibrated probabilities",
            },
        },
        {
            "name": "all_publications_assigned_to_sessions",
            "passed": all(bool(row["signal_date"]) for row in rows),
            "details": {"timing_bucket_counts": timing_counts},
        },
        {
            "name": "only_complete_nonexcluded_scores_enter_signals",
            "passed": signal_input_count == len(eligible_ids)
            and not (excluded_from_signal_ids & eligible_ids)
            and not (unresolved_ids & eligible_ids),
            "details": {
                "eligible_count": len(eligible_ids),
                "aggregated_input_count": signal_input_count,
            },
        },
        {
            "name": "explicit_neutral_is_observed_not_missing",
            "passed": classification_counts["explicit_neutral"] > 0,
            "details": {
                "explicit_neutral_count": classification_counts["explicit_neutral"],
                "missing_count": classification_counts["missing"],
            },
        },
        {
            "name": "signal_sessions_have_market_prices",
            "passed": not missing_price_sessions
            and all(str(signal["signal_date"]) in mapped_price_dates for signal in signals),
            "details": {"missing_price_sessions": missing_price_sessions},
        },
        {
            "name": "subsequent_session_target_available",
            "passed": not missing_target_sessions,
            "details": {
                "missing_target_sessions": missing_target_sessions,
                "construction": "complete price series shift(-1) before sparse signal join",
            },
        },
    ]
    return {
        "demo_contract": {
            "offline": True,
            "sentiment_source": "committed cached model outputs",
            "score_component_interpretation": (
                "synthetic compatibility fields, not calibrated probabilities"
            ),
            "primary_signal": "simple_mean",
            "weighted_variants_role": "sensitivity_only",
            "predictive_target": "subsequent-session fractional close-to-close return",
            "claim_scope": "descriptive reproducibility demo; no strategy or alpha claim",
        },
        "inputs": {
            "headline_rows": len(rows),
            "price_sessions": len(prices),
            "signal_sessions": len(signals),
        },
        "record_classification": classification_counts,
        "timing_bucket_counts": timing_counts,
        "checks": checks,
        "all_checks_passed": all(bool(check["passed"]) for check in checks),
    }


def _write_chart(aligned: pd.DataFrame, path: Path) -> None:
    dates = pd.to_datetime(aligned["signal_date"], format="%Y-%m-%d")
    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)

    axis.plot(
        dates,
        aligned["simple_mean"],
        color="#17365D",
        linewidth=3.0,
        marker="o",
        markersize=6,
        label="simple_mean (primary baseline)",
        zorder=5,
    )
    sensitivity_styles = {
        "relevance_weighted": ("#4E79A7", "--"),
        "intensity_relevance_weighted": ("#F28E2B", "-."),
        "full_weighted": ("#777777", ":"),
    }
    for variant, (color, line_style) in sensitivity_styles.items():
        axis.plot(
            dates,
            aligned[variant],
            color=color,
            linestyle=line_style,
            linewidth=1.35,
            alpha=0.9,
            label=f"{variant} (sensitivity)",
        )
    axis.axhline(0.0, color="#222222", linewidth=0.7, alpha=0.6)
    axis.set_ylabel("Sentiment score")
    axis.set_xlabel("First session able to react")
    axis.grid(axis="y", alpha=0.18)

    returns_axis = axis.twinx()
    return_percent = aligned["next_session_return"] * 100.0
    colors = ["#59A14F" if value >= 0 else "#E15759" for value in return_percent]
    returns_axis.bar(
        dates,
        return_percent,
        width=0.45,
        color=colors,
        alpha=0.16,
        label="subsequent-session return",
        zorder=0,
    )
    returns_axis.set_ylabel("Subsequent-session return (%)")

    handles, labels = axis.get_legend_handles_labels()
    return_handles, return_labels = returns_axis.get_legend_handles_labels()
    axis.legend(handles + return_handles, labels + return_labels, loc="best", fontsize=8)
    axis.set_title("Offline BIST news-sentiment demo")
    figure.suptitle(
        "Simple mean is the baseline; weighted variants are sensitivity checks only",
        fontsize=10,
        y=0.955,
    )
    figure.text(
        0.5,
        0.005,
        "Descriptive alignment only — no strategy or alpha claim.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.savefig(path, dpi=140, metadata={"Software": "Turkish stock news sentiment demo"})
    plt.close(figure)


def run_demo(
    output_dir: Path | str = "demo_output",
    *,
    headlines_path: Path | str = DEFAULT_HEADLINES,
    prices_path: Path | str = DEFAULT_PRICES,
) -> dict[str, Path]:
    """Run the offline demo and return the three generated artifact paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    headline_file = Path(headlines_path)
    price_file = Path(prices_path)

    rows = _prepare_headlines(headline_file)
    signals = _session_signals(rows)
    prices = _load_prices(price_file)
    aligned = _align_signals_and_returns(signals, prices)
    audit = _audit(rows, signals, prices, aligned)
    if not audit["all_checks_passed"]:
        failed = [check["name"] for check in audit["checks"] if not check["passed"]]
        raise ValueError(f"offline demo audit failed: {', '.join(failed)}")

    results_path = output / "signal_results.csv"
    audit_path = output / "audit.json"
    chart_path = output / "signal_variants.png"
    aligned.to_csv(results_path, index=False, float_format="%.10f", encoding="utf-8")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_chart(aligned, chart_path)
    return {"results": results_path, "audit": audit_path, "chart": chart_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_output"),
        help="artifact directory (default: ./demo_output)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = run_demo(args.output_dir)
    print("Offline demo complete: no API key, network, or private database used.")
    print("Primary signal: simple_mean; weighted variants are sensitivity checks only.")
    for name in ("results", "audit", "chart"):
        print(f"{name.capitalize()}: {artifacts[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
