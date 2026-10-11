"""Markdown tables for the readiness documents, straight from the result files.

    python -m readiness_audit.tables

Printed so that no number in a document is typed by hand.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from readiness_audit import power
from readiness_audit.run_simulations import DOCS

LABELS = {
    "linear": "evenly spaced over all categories",
    "one:buyback": "only buyback differs",
    "one:general_other": "only general_other differs",
    "one:other_form": "only other_form differs",
    "one:administrative": "only administrative differs",
    "one:debt_issuance": "only debt_issuance differs",
    "one:new_contract": "only new_contract differs",
    "one:asset_acquisition": "only asset_acquisition differs",
}


def _pct(value: float, digits: int = 1) -> str:
    return "–" if pd.isna(value) else f"{100 * value:.{digits}f}%"


def _mde(value) -> str:
    return "above 5%" if value is None else f"{100 * value:.1f}%"


def power_table(table: pd.DataFrame, column: str, analysis: str = "main", anchors: int = 80,
                scenarios=None) -> str:
    part = table[(table["analysis"] == analysis) & (table["anchors"] == anchors)]
    effects = sorted(part["true_spread"].unique())
    lines = ["| Where the effect sits | " + " | ".join(_pct(e) for e in effects) + " | 80% power at |",
             "| --- |" + " --- |" * (len(effects) + 1)]
    for scenario in scenarios or [s for s in LABELS if s in set(part["scenario"])]:
        rows = part[part["scenario"] == scenario].sort_values("true_spread")
        if rows.empty:
            continue
        note = "" if rows["simulations"].min() >= 1000 else f" ({int(rows['simulations'].min())} runs)"
        lines.append(f"| {LABELS.get(scenario, scenario)}{note} | "
                     + " | ".join(_pct(v) for v in rows[column]) + " | "
                     + _mde(power.minimum_detectable(rows, scenario, column)) + " |")
    return "\n".join(lines)


def variant_table(table: pd.DataFrame, analyses, column: str, scenario: str) -> str:
    rows = table[table["analysis"].isin(analyses) & (table["scenario"] == scenario)]
    keys = ["analysis", "anchors", "common_share", "ar1", "event_vol", "recurrence"]
    effects = sorted(rows["true_spread"].unique())
    lines = ["| Variant | " + " | ".join(_pct(e) for e in effects) + " | 80% power at |",
             "| --- |" + " --- |" * (len(effects) + 1)]
    for key, part in rows.groupby(keys, sort=False):
        part = part.sort_values("true_spread")
        analysis, anchors, share, ar1, vol, recurrence = key
        name = {"main": "registered frame, 80 anchors",
                "sample_size": f"{anchors} anchors",
                "common_share": f"common daily share {share:.0%}",
                "event_volatility": f"event-window volatility x{vol:g}",
                "issuer_recurrence_high": "issuers recur as in the observed anchors",
                "issuer_recurrence_none": "no issuer files twice",
                "serial_correlation_high": f"serial correlation {ar1:.2f}"}.get(analysis, analysis)
        lines.append(f"| {name} | " + " | ".join(_pct(v) for v in part[column]) + " | "
                     + _mde(power.minimum_detectable(part, scenario, column)) + " |")
    return "\n".join(lines)


def calibration_table(results: dict, names, procedures) -> str:
    lines = ["| Design | Categories | Smallest category | Fewest dates in a category | Procedure "
             "| Runs | Rate at 5% (95% CI) | Rate at 0.56% (95% CI) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        if name not in results:
            continue
        design = results[name]["design"]
        for procedure in procedures:
            row = results[name]["procedures"].get(procedure)
            if row is None:
                continue
            a, b = row["ci_at_0.0500"], row["ci_at_0.0056"]
            lines.append(
                f"| {name.replace('_', ' ')} | {design['categories']:.1f} | "
                f"{design['smallest_category']:.0f} | {design['min_dates_in_a_category']:.0f} | "
                f"{procedure.replace('_', ' ')} | {row['valid']} | "
                f"{_pct(row['rate_at_0.0500'])} ({_pct(a[0])} to {_pct(a[1])}) | "
                f"{_pct(row['rate_at_0.0056'], 2)} ({_pct(b[0], 2)} to {_pct(b[1], 2)}) |")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    table = pd.read_csv(DOCS / "POWER_RESULTS.csv")
    designs = json.loads((DOCS / "power_designs.json").read_text(encoding="utf-8"))
    print("## designs\n")
    for name, design in designs.items():
        print(name, {k: round(v, 1) for k, v in design.items()})
    null = table[(table["analysis"] == "main") & (table["true_spread"] == 0)
                 & (table["scenario"] == "linear")].iloc[0]
    print("\n## null row (main)\n")
    for column in ("power_nominal", "power_holm", "power_supported", "power_corrected_nominal",
                   "power_corrected_holm", "power_corrected_supported", "power_spread_material",
                   "no_inference_share"):
        se = null.get(column.replace("power_", "se_"), float("nan"))
        print(f"{column}: {_pct(null[column], 2)}  (se {_pct(se, 2)})")
    for title, column in (("registered test, verdict supported", "power_supported"),
                          ("registered test, nominal 5%", "power_nominal"),
                          ("registered test, Holm level", "power_holm"),
                          ("size-corrected, verdict supported", "power_corrected_supported"),
                          ("size-corrected, nominal 5%", "power_corrected_nominal"),
                          ("category's own mean test, nominal 5%", "power_category_own_test")):
        print(f"\n## main: {title}\n")
        print(power_table(table, column))
    for scenario in ("linear", "one:buyback", "one:administrative", "one:new_contract"):
        for title, column in (("registered supported", "power_supported"),
                              ("corrected supported", "power_corrected_supported")):
            print(f"\n## by sample size: {scenario}, {title}\n")
            print(variant_table(table, ["main", "sample_size"], column, scenario))
            print(f"\n## sensitivity: {scenario}, {title}\n")
            print(variant_table(table, ["main", "common_share", "event_volatility",
                                        "issuer_recurrence_high", "issuer_recurrence_none",
                                        "serial_correlation_high"], column, scenario))
    print("\n## null rates of sensitivity variants (linear, effect 0)\n")
    zero = table[(table["true_spread"] == 0) & (table["scenario"] == "linear")]
    print(zero[["analysis", "anchors", "common_share", "ar1", "event_vol", "recurrence",
                "simulations", "power_nominal", "power_holm", "power_corrected_nominal",
                "power_corrected_holm"]].round(4).to_string(index=False))

    results = json.loads((DOCS / "calibration_results.json").read_text(encoding="utf-8"))
    procedures = ["registered_two_way", "date_only", "issuer_only", "jackknife_date",
                  "two_way_min_category_dof", "date_only_min_category_dof", "two_way_sum",
                  "wild_cluster_date"]
    print("\n## calibration\n")
    print(calibration_table(results, list(results), procedures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
