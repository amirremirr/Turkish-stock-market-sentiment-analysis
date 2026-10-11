"""Run every simulation of the readiness audit and write the result files.

    python -m readiness_audit.run_simulations            # everything
    python -m readiness_audit.run_simulations power_main # one part

Reads the sample structure and the pre-sample calibration written by
``make_structure``; generates only synthetic returns.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

from readiness_audit import calibration as cal
from readiness_audit import power
from readiness_audit import snapshot as snap
from stock_research import stats
from stock_research.config import KAP_FRAME_VERSION, REPOSITORY_ROOT
from stock_research.data import kap

DOCS = REPOSITORY_ROOT / "docs" / "stock_research" / "readiness"
SESSIONS = 440
SIMS = 2000
CENSUS_ANCHORS = 345


INGESTED: list = []          # fully ingested anchors, set by ``_inputs``


def _inputs():
    structure = pd.read_csv(snap.AUDIT_OUT / "h7_structure.csv")
    calibration = json.loads((snap.AUDIT_OUT / "calibration.json").read_text(encoding="utf-8"))
    # Anchors complete in the copy the structure was read from, including the
    # ones that yield no usable event.
    con = sqlite3.connect(f"file:{snap.SNAPSHOT_DB.as_posix()}?mode=ro", uri=True)
    try:
        INGESTED[:] = [r[0] for r in con.execute(
            "SELECT block_seq FROM sr_kap_frame WHERE frame_version=? AND status IN (?,?) "
            "ORDER BY block_seq", (KAP_FRAME_VERSION, *kap.ATTEMPTED))]
    finally:
        con.close()
    return structure, calibration


def _focus(structure):
    return ["linear"] + [f"one:{c}" for c in ("buyback", "administrative", "new_contract")
                         if c in structure["category"].unique()]


def part_power_main(structure, calibration) -> None:
    started = time.time()
    table = power.run(structure, calibration, sessions=SESSIONS, ingested=INGESTED,
                      simulations=SIMS)
    table["analysis"] = "main"
    table.to_csv(snap.AUDIT_OUT / "power_main.csv", index=False)
    (snap.AUDIT_OUT / "power_designs_main.json").write_text(
        json.dumps({"80_anchors": table.attrs["designs"]}, indent=1), encoding="utf-8")
    print("main", round(time.time() - started), "s", flush=True)


def part_power_size(structure, calibration) -> None:
    started = time.time()
    frames, designs = [], {}
    # 345 anchors is about every material-event disclosure of the year: a census.
    for anchors, sims in ((160, SIMS), (240, SIMS), (CENSUS_ANCHORS, 1000)):
        table = power.run(structure, calibration, sessions=SESSIONS, ingested=INGESTED,
                          simulations=sims, anchors_total=anchors,
                          scenarios=_focus(structure))
        table["analysis"] = "sample_size"
        frames.append(table)
        designs[f"{anchors}_anchors"] = table.attrs["designs"]
        pd.concat(frames, ignore_index=True).to_csv(snap.AUDIT_OUT / "power_size.csv", index=False)
        (snap.AUDIT_OUT / "power_designs_size.json").write_text(
            json.dumps(designs, indent=1), encoding="utf-8")
        print("anchors", anchors, round(time.time() - started), "s", flush=True)


def part_power_sensitivity(structure, calibration, chunk: int) -> None:
    """Two variants per chunk, so the sweep can run as parallel processes."""

    started = time.time()
    frames = []
    focus = _focus(structure)
    variants = [("common_share", dict(common_share=share)) for share in (0.0, 0.10, 0.20)]
    variants += [("event_volatility", dict(event_vol=vol)) for vol in (1.5, 2.0)]
    variants += [("issuer_recurrence_high", dict(recurrence="resample")),
                 ("issuer_recurrence_none", dict(recurrence="fresh")),
                 ("serial_correlation_high", dict(ar1=0.25))]
    for name, kwargs in variants[2 * chunk:2 * chunk + 2]:
        table = power.run(structure, calibration, sessions=SESSIONS, ingested=INGESTED,
                          simulations=SIMS, scenarios=focus, **kwargs)
        table["analysis"] = name
        frames.append(table)
        pd.concat(frames, ignore_index=True).to_csv(
            snap.AUDIT_OUT / f"power_sensitivity_{chunk}.csv", index=False)
        print(name, kwargs, round(time.time() - started), "s", flush=True)


def part_merge() -> None:
    frames = [pd.read_csv(snap.AUDIT_OUT / f"power_{name}.csv")
              for name in ("main", "size", *(f"sensitivity_{k}" for k in range(4)))]
    pd.concat(frames, ignore_index=True).to_csv(
        DOCS / "POWER_RESULTS.csv", index=False, float_format="%.5f")
    designs = {}
    for name in ("main", "size"):
        designs.update(json.loads(
            (snap.AUDIT_OUT / f"power_designs_{name}.json").read_text(encoding="utf-8")))
    (DOCS / "power_designs.json").write_text(json.dumps(designs, indent=1), encoding="utf-8")


def part_calibration(structure, calibration) -> None:
    started = time.time()
    out = {}
    settings = {
        "realistic_all_categories": dict(simulations=5000, projections=25),
        "independent_all_categories": dict(simulations=5000, projections=25, recurrence="fresh",
                                           common_share=0.0, ar1=0.0),
        "realistic_four_largest": dict(simulations=5000, projections=25, categories=4),
        "realistic_three_largest": dict(simulations=5000, projections=25, categories=3),
        "realistic_two_largest": dict(simulations=5000, projections=25, categories=2),
        "stress_common_share_20pct": dict(simulations=3000, projections=15, common_share=0.20),
        "stress_issuer_recurrence": dict(simulations=3000, projections=15, recurrence="resample"),
        "stress_serial_correlation": dict(simulations=3000, projections=15, ar1=0.25),
        "stress_event_volatility_2x": dict(simulations=3000, projections=15, event_vol=2.0),
        "realistic_240_anchors": dict(simulations=3000, projections=15, anchors_total=240),
        "realistic_census_345_anchors": dict(simulations=2000, projections=10, anchors_total=345),
    }
    for name, kwargs in settings.items():
        out[name] = cal.null_rates(structure, calibration, sessions=SESSIONS,
                                   ingested=INGESTED, **kwargs)
        print(name, round(time.time() - started), "s", flush=True)
        (DOCS / "calibration_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def part_small_category() -> None:
    rows = cal.small_category_experiment()
    (DOCS / "calibration_small_category.json").write_text(json.dumps(rows, indent=1),
                                                          encoding="utf-8")
    print(json.dumps(rows, indent=1))


def part_wild_null(structure, calibration) -> None:
    """The wild cluster bootstrap again, with enough draws and runs to speak
    about the Holm level."""

    started = time.time()
    out = cal.null_rates(structure, calibration, sessions=SESSIONS, ingested=INGESTED,
                         simulations=3000, projections=15, wild=999, seed=17)
    (DOCS / "calibration_wild_bootstrap.json").write_text(json.dumps(out, indent=1),
                                                          encoding="utf-8")
    print("wild null", round(time.time() - started), "s", flush=True)


WILD_CASES = [("linear", 0.02), ("linear", 0.03), ("linear", 0.05),
              ("one:buyback", 0.015), ("one:buyback", 0.02), ("one:buyback", 0.03)]
WILD_CASES_LARGE = [("linear", 0.03), ("one:buyback", 0.01), ("one:buyback", 0.015),
                    ("one:buyback", 0.02), ("one:administrative", 0.02),
                    ("one:administrative", 0.03)]


def part_wild_power(structure, calibration, *, anchors: int = 80, cases=WILD_CASES,
                    projections: int = 10, per: int = 60, draws: int = 499) -> None:
    """Power of the candidate procedures next to the registered test, on the
    same simulated samples."""

    started = time.time()
    rng = np.random.default_rng(23 + anchors)
    names = ("registered_two_way", "date_only_min_category_dof", "wild_cluster_date")
    hits = {case: {name: {level: 0 for level in cal.LEVELS} for name in names} for case in cases}
    runs = {case: 0 for case in cases}
    for _ in range(projections):
        design = power.project(structure, calibration, anchors_total=anchors, frame_anchors=80,
                               sessions=SESSIONS, rng=rng, ingested=INGESTED)
        procedures, test = cal.Procedures(design), power.JointTest(design)
        planted = {case: power.shifts(test, *case) for case in cases
                   if case[0] == "linear" or case[0][4:] in test.masks}
        for _ in range(per):
            noise = power.simulate_cars(design, rng,
                                        common_share=calibration["common_daily_share"],
                                        ar1=calibration["mean_ar1"])
            for case, shift in planted.items():
                result = procedures(noise + shift, wild=draws, rng=rng)
                runs[case] += 1
                for name in names:
                    p = result.get(name)
                    for level in cal.LEVELS:
                        hits[case][name][level] += bool(p is not None and p < level)
    out = []
    for case in cases:
        n = runs[case]
        if not n:
            continue
        for name in names:
            row = {"anchors": anchors, "scenario": case[0], "true_spread": case[1],
                   "procedure": name, "runs": n, "wild_draws": draws}
            for level in cal.LEVELS:
                rate = hits[case][name][level] / n
                row[f"power_at_{level:.4f}"] = rate
                row[f"se_at_{level:.4f}"] = float(np.sqrt(rate * (1 - rate) / n))
            out.append(row)
    name = "calibration_power_comparison" + ("" if anchors == 80 else f"_{anchors}_anchors")
    pd.DataFrame(out).to_csv(DOCS / f"{name}.csv", index=False, float_format="%.5f")
    print("wild power", anchors, round(time.time() - started), "s", flush=True)


def part_mean_test() -> None:
    """The earlier audit's design: twelve issuers, ten filings each on nearby
    sessions, CAR(+1,+20), no effect. It reported 6.7% from 300 runs."""

    rng = np.random.default_rng(11)
    runs, results = 5000, {}
    rates = {"date_only": 0, "two_way": 0, "issuer_only": 0}
    covered, widths = 0, []
    for _ in range(runs):
        starts = rng.integers(150, 240, size=12)
        issuer = np.repeat(np.arange(12), 10)
        position = (starts[:, None] + 2 * np.arange(10)[None, :]).ravel()
        design = power.Design(issuer, position, position, np.array(["x"] * 120), ["x"],
                              rng.uniform(0.02, 0.04, size=12), 300, 12)
        y = power.simulate_cars(design, rng, common_share=0.0, ar1=0.0, df=30.0, window=(1, 20))
        one = stats.cluster_mean(y, [position])
        two = stats.cluster_mean(y, [position, issuer])
        firm = stats.cluster_mean(y, [issuer])
        rates["date_only"] += one["p"] < 0.05
        rates["two_way"] += bool(two["p"] is not None and two["p"] < 0.05)
        rates["issuer_only"] += firm["p"] < 0.05
        if two["ci"] is not None:
            covered += two["ci"][0] <= 0.0 <= two["ci"][1]
            widths.append(two["ci"][1] - two["ci"][0])
    for name, hits in rates.items():
        rate = hits / runs
        half = 1.96 * np.sqrt(rate * (1 - rate) / runs)
        results[name] = {"rejection_rate": rate, "ci": [rate - half, rate + half]}
    results["two_way_ci_coverage"] = covered / runs
    results["runs"] = runs
    results["design"] = {"issuers": 12, "events": 120, "window": [1, 20],
                         "issuer_clusters": 12, "date_clusters": "about 90"}
    (DOCS / "mean_test_calibration.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parts = set((argv or sys.argv[1:]) or [
        "mean", "calibration", "small_category", "wild_null", "wild_power",
        "wild_power_large", "power_main", "power_size",
        "power_sensitivity", "merge"])
    DOCS.mkdir(parents=True, exist_ok=True)
    structure, calibration = _inputs()
    if "mean" in parts:
        part_mean_test()
    if "calibration" in parts:
        part_calibration(structure, calibration)
    if "power_main" in parts:
        part_power_main(structure, calibration)
    if "power_size" in parts:
        part_power_size(structure, calibration)
    for chunk in range(4):
        if f"power_sensitivity_{chunk}" in parts or "power_sensitivity" in parts:
            part_power_sensitivity(structure, calibration, chunk)
    if "small_category" in parts:
        part_small_category()
    if "wild_null" in parts:
        part_wild_null(structure, calibration)
    if "wild_power" in parts:
        part_wild_power(structure, calibration)
    for anchors in (240, CENSUS_ANCHORS):
        if f"wild_power_{anchors}" in parts or "wild_power_large" in parts:
            part_wild_power(structure, calibration, anchors=anchors, cases=WILD_CASES_LARGE,
                            projections=6, per=50, draws=299)
    if "merge" in parts:
        part_merge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
