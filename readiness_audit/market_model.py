"""How well the registered expected-return model predicts, before the sample.

    python -m readiness_audit.market_model

Uses sessions up to 2022-12-30 only: no sampled disclosure had been published.
Each stock's model is fitted on the first half of that period and judged on
the second half, so the comparison is out of sample. No event is involved.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from readiness_audit import snapshot as snap


def _fit(y: np.ndarray, x: np.ndarray):
    beta = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    return y.mean() - beta * x.mean(), beta


def summarise(db=None) -> Dict[str, Any]:
    panel, _, _ = snap.load(db or snap.SNAPSHOT_DB)
    sessions = [s for s in panel.calendar.sessions if s <= snap.PRE_SAMPLE_END]
    half = len(sessions) // 2
    fit_days, test_days = sessions[:half], sessions[half:]
    market = panel.market["ret"]
    stocks = {t: f["ret"] for t, f in panel.bars.items() if t != panel.benchmark}
    returns = pd.DataFrame(stocks).loc[sessions]
    # Equal-weighted return of every stock in the panel that traded that day.
    equal = returns.mean(axis=1)

    rows = []
    for ticker, series in returns.items():
        a = pd.concat([series, market, equal], axis=1, keys=["r", "m", "e"])
        fit, test = a.loc[fit_days].dropna(), a.loc[test_days].dropna()
        if len(fit) < 60 or len(test) < 40:
            continue
        alpha, beta = _fit(fit["r"].to_numpy(), fit["m"].to_numpy())
        alpha_e, beta_e = _fit(fit["r"].to_numpy(), fit["e"].to_numpy())
        _, beta_late = _fit(test["r"].to_numpy(), test["m"].to_numpy())
        volume = panel.bars[ticker].loc[sessions, "volume"]
        rows.append({
            "ticker": ticker, "beta_first_half": beta, "beta_second_half": beta_late,
            "alpha_daily": alpha,
            "sd_raw": test["r"].std(),
            "sd_market_model": (test["r"] - alpha - beta * test["m"]).std(),
            "sd_market_adjusted": (test["r"] - test["m"]).std(),
            "sd_equal_weighted_model": (test["r"] - alpha_e - beta_e * test["e"]).std(),
            "bias_market_model": (test["r"] - alpha - beta * test["m"]).mean(),
            "bias_market_adjusted": (test["r"] - test["m"]).mean(),
            "bias_equal_weighted_model": (test["r"] - alpha_e - beta_e * test["e"]).mean(),
            "sessions_without_trade_share": float((~(volume > 0)).mean()),
        })
    table = pd.DataFrame(rows)

    def med(column: str) -> float:
        return float(table[column].median())

    out = {
        "period": [sessions[0], sessions[-1]], "fit_sessions": len(fit_days),
        "test_sessions": len(test_days), "stocks": int(len(table)),
        "median_out_of_sample_residual_sd": {
            "raw_return": med("sd_raw"), "market_model_xu100": med("sd_market_model"),
            "market_adjusted_xu100": med("sd_market_adjusted"),
            "equal_weighted_panel_model": med("sd_equal_weighted_model")},
        "cross_sectional_sd_of_mean_daily_abnormal_return": {
            "market_model_xu100": float(table["bias_market_model"].std()),
            "market_adjusted_xu100": float(table["bias_market_adjusted"].std()),
            "equal_weighted_panel_model": float(table["bias_equal_weighted_model"].std())},
        "mean_daily_abnormal_return_all_stocks": {
            "market_model_xu100": float(table["bias_market_model"].mean()),
            "market_adjusted_xu100": float(table["bias_market_adjusted"].mean()),
            "equal_weighted_panel_model": float(table["bias_equal_weighted_model"].mean())},
        "beta": {
            "median_first_half": med("beta_first_half"),
            "median_second_half": med("beta_second_half"),
            "correlation_between_halves": float(
                table["beta_first_half"].corr(table["beta_second_half"])),
            "median_absolute_change": float(
                (table["beta_first_half"] - table["beta_second_half"]).abs().median())},
        "stocks_with_over_5pct_sessions_without_trade": int(
            (table["sessions_without_trade_share"] > 0.05).sum()),
    }
    (snap.AUDIT_OUT / "market_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(summarise(), indent=1))
