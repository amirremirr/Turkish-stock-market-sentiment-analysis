"""Figures for the readiness report, from the simulation result files.

    python -m readiness_audit.figures

Every curve is a simulation on synthetic returns. None is a research result.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from readiness_audit import power
from readiness_audit.run_simulations import DOCS

FIGURES = DOCS / "figures"
NOTE = "Synthetic returns on the projected sample structure. Not a research result."


def _finish(fig, name: str) -> None:
    fig.text(0.01, 0.01, NOTE, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / name, dpi=140)
    plt.close(fig)


def power_curves(table: pd.DataFrame) -> None:
    """Left: the registered test's verdict. Right: the same effects under a
    procedure that holds its level, next to the registered test, on common
    simulated samples."""

    main = table[table["analysis"] == "main"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    ax = axes[0]
    for scenario, part in main.groupby("scenario"):
        if part["simulations"].min() < 800:
            continue                                   # category absent from most projections
        part = part.sort_values("true_spread")
        label = "evenly spaced" if scenario == "linear" else f"only {scenario[4:]}"
        ax.errorbar(part["true_spread"] * 100, part["power_supported"],
                    yerr=1.96 * part["se_supported"], marker="o", ms=3, lw=1.2, capsize=2,
                    label=label)
    ax.axhline(0.8, color="0.5", lw=0.8, ls="--")
    ax.axhline(power.HOLM_ALPHA, color="0.7", lw=0.8, ls=":")
    ax.set_title("Registered test: probability of verdict 'supported'", fontsize=10)
    ax.set_xlabel("true spread between category means, CAR(0,+5), %")
    ax.set_ylabel("probability")
    ax.legend(fontsize=7, title="where the effect sits", title_fontsize=7)

    ax = axes[1]
    compared = pd.read_csv(DOCS / "calibration_power_comparison.csv")
    styles = {"registered_two_way": ("registered (level not held)", "--"),
              "wild_cluster_date": ("wild cluster bootstrap", "-")}
    for (scenario, procedure), part in compared.groupby(["scenario", "procedure"]):
        if procedure not in styles:
            continue
        name, line = styles[procedure]
        where = "evenly spaced" if scenario == "linear" else f"only {scenario[4:]}"
        ax.errorbar(part["true_spread"] * 100, part["power_at_0.0056"],
                    yerr=1.96 * part["se_at_0.0056"], marker="o", ms=3, lw=1.2, ls=line,
                    capsize=2, label=f"{where}: {name}")
    ax.axhline(0.8, color="0.5", lw=0.8, ls="--")
    ax.set_title("Rejection at the Holm level, same simulated samples", fontsize=10)
    ax.set_xlabel("true spread, %")
    ax.legend(fontsize=7)
    _finish(fig, "power_curves.png")


def nominal_against_holm(table: pd.DataFrame) -> None:
    part = table[(table["analysis"] == "main") & (table["scenario"] == "linear")].sort_values("true_spread")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for column, label in (("power_nominal", "registered, p < 0.05"),
                          ("power_holm", "registered, Holm over nine (p < 0.0056)"),
                          ("power_corrected_nominal", "size-corrected, p < 0.05"),
                          ("power_corrected_holm", "size-corrected, Holm over nine")):
        ax.plot(part["true_spread"] * 100, part[column], marker="o", ms=3, label=label)
    ax.axhline(0.8, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("true spread, evenly spaced over categories, %")
    ax.set_ylabel("rejection probability")
    ax.legend(fontsize=7)
    _finish(fig, "nominal_vs_holm.png")


def sample_size(table: pd.DataFrame) -> None:
    """Registered verdict by sample size (lines), and the wild bootstrap at the
    Holm level where it was run (crosses)."""

    rows = table[table["analysis"].isin(["main", "sample_size"])]
    wild = []
    for name in ("calibration_power_comparison.csv",
                 "calibration_power_comparison_240_anchors.csv",
                 "calibration_power_comparison_345_anchors.csv"):
        if (DOCS / name).exists():
            part = pd.read_csv(DOCS / name)
            part["anchors"] = part.get("anchors", 80)
            wild.append(part[part["procedure"] == "wild_cluster_date"])
    wild = pd.concat(wild, ignore_index=True) if wild else pd.DataFrame()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    colors = {}
    for ax, scenario in zip(axes, ("linear", "one:buyback", "one:administrative")):
        for anchors, part in rows[rows["scenario"] == scenario].groupby("anchors"):
            part = part.sort_values("true_spread")
            line, = ax.plot(part["true_spread"] * 100, part["power_supported"], marker="o", ms=3,
                            lw=1.2, label=f"{anchors} anchors, registered")
            colors[anchors] = line.get_color()
        if len(wild):
            for anchors, part in wild[wild["scenario"] == scenario].groupby("anchors"):
                ax.plot(part["true_spread"] * 100, part["power_at_0.0056"], marker="x", ms=7,
                        lw=0, color=colors.get(anchors, "k"),
                        label=f"{anchors} anchors, wild bootstrap")
        ax.axhline(0.8, color="0.5", lw=0.8, ls="--")
        ax.set_title({"linear": "evenly spaced over all tested categories",
                      "one:buyback": "effect only in buyback",
                      "one:administrative": "effect only in administrative"}[scenario],
                     fontsize=10)
        ax.set_xlabel("true spread, %")
    axes[0].set_ylabel("probability of rejection at the Holm level")
    axes[0].legend(fontsize=6.5)
    _finish(fig, "power_by_sample_size.png")


def calibration_chart() -> None:
    results = json.loads((DOCS / "calibration_results.json").read_text(encoding="utf-8"))
    names = [n for n in ("realistic_all_categories", "independent_all_categories",
                         "realistic_four_largest", "realistic_two_largest",
                         "stress_issuer_recurrence", "realistic_240_anchors",
                         "realistic_census_345_anchors") if n in results]
    procedures = ["registered_two_way", "date_only", "jackknife_date",
                  "two_way_min_category_dof", "date_only_min_category_dof"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, level in zip(axes, ("0.0500", "0.0056")):
        width = 0.8 / len(procedures)
        for k, procedure in enumerate(procedures):
            rates = [results[n]["procedures"][procedure][f"rate_at_{level}"] for n in names]
            low = [results[n]["procedures"][procedure][f"ci_at_{level}"][0] for n in names]
            high = [results[n]["procedures"][procedure][f"ci_at_{level}"][1] for n in names]
            x = [i + k * width for i in range(len(names))]
            ax.bar(x, rates, width, label=procedure.replace("_", " "),
                   yerr=[[r - l for r, l in zip(rates, low)], [h - r for r, h in zip(rates, high)]],
                   capsize=1.5, error_kw={"lw": 0.6})
        ax.axhline(float(level), color="k", lw=0.8, ls="--")
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(names))])
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
        ax.set_title(f"Rejection rate under the null at level {float(level):.4f}", fontsize=10)
    axes[0].legend(fontsize=7)
    _finish(fig, "null_rejection_rates.png")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(DOCS / "POWER_RESULTS.csv")
    power_curves(table)
    nominal_against_holm(table)
    sample_size(table)
    calibration_chart()
    print(sorted(p.name for p in FIGURES.glob("*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
