"""Rejection rates of the candidate joint tests, under the null and with an
effect planted, on the designs of ``v3_validation.designs``.

    python -m v3_validation.calibrate run <design> <seed> <simulations> [draws] [mode]
    python -m v3_validation.calibrate merge

``mode`` is ``null`` (default), ``power`` or ``verdict``. Each run writes one
piece to the output directory; ``merge`` adds the pieces up and writes
``docs/stock_research/protocol_v3/validation/*.json``. Pieces with different
seeds are independent, so a design can be spread over several processes.

Every return is synthetic. The procedures are the registered functions of
``stock_research.stats``, called as the study would call them.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from readiness_audit import power
from stock_research import stats
from stock_research.config import ALPHA, REPOSITORY_ROOT
from stock_research.hypotheses import common, h7
from v3_validation import designs
from v3_validation import snapshot as snap

HOLM = ALPHA / common.FAMILY_SIZE
LEVELS = (ALPHA, HOLM)
PIECES = snap.OUT / "pieces"
DOCS = REPOSITORY_ROOT / "docs" / "stock_research" / "protocol_v3" / "validation"
PROCEDURES = ("v2_two_way_f", "wild_date_statistic", "wild_two_way_statistic", "cr2_satterthwaite_date")
POWER_CASES = [("one:buyback", 0.005), ("one:buyback", 0.01), ("one:buyback", 0.015), ("one:buyback", 0.02),
               ("one:buyback", 0.03), ("one:new_contract", 0.02), ("one:new_contract", 0.03),
               ("one:dividend", 0.02), ("one:dividend", 0.03), ("one:debt_issuance", 0.02),
               ("one:debt_issuance", 0.03), ("one:rights_issue", 0.03), ("one:bonus_issue", 0.03),
               ("one:asset_acquisition", 0.03), ("all", 0.01), ("all", 0.02)]


def _persistence(meta: Dict[str, Any], factor: float = 1.0, **fixed: float) -> Dict[str, float]:
    measured = {g: (v["share"] or 0.0) for g, v in meta["persistence"].items()}
    out = {g: min(0.95, factor * share) for g, share in measured.items()}
    out.update(fixed)
    return out


def _builders() -> Dict[str, Dict[str, Any]]:
    """Each design: how to draw it, and the noise it is simulated with."""

    return {
        "audit_v2_rules": {"audit": True},
        "v3_anchor80": {"anchors_total": 80},
        # Half of all events keep the issuer of the anchor they were copied
        # from: with 345 anchors drawn from about 30, those issuers recur a
        # dozen times each. (Copying every issuer leaves fewer distinct issuers
        # than the registered minimum, so the study would not run at all.)
        "v3_census_recurrence": {"anchors_total": designs.CENSUS_ANCHORS, "recurrence": "mixed",
                                 "persistence": 2.0, "fixed": {"buyback": 0.9}},
        "v3_census": {"anchors_total": designs.CENSUS_ANCHORS},
        "v3_census_persistent": {"anchors_total": designs.CENSUS_ANCHORS, "persistence": 2.0},
        "v3_census_programmes": {"anchors_total": designs.CENSUS_ANCHORS, "persistence": 2.0,
                                 "fixed": {"buyback": 0.9}},
        "v3_census_programmes_unthinned": {"anchors_total": designs.CENSUS_ANCHORS,
                                           "persistence": 2.0, "fixed": {"buyback": 0.9},
                                           "thinning": False},
        "v3_census_common20": {"anchors_total": designs.CENSUS_ANCHORS, "persistence": 2.0,
                               "fixed": {"buyback": 0.9}, "common_share": 0.20},
        "v3_census_serial25": {"anchors_total": designs.CENSUS_ANCHORS, "persistence": 2.0,
                               "fixed": {"buyback": 0.9}, "ar1": 0.25},
        # The real census structure, once it exists (no projection).
        "census_actual": {"actual": True},
        "census_actual_common20": {"actual": True, "common_share": 0.20},
        "census_actual_serial25": {"actual": True, "ar1": 0.25},
        "controlled_minimum": {"controlled": {"small": (30, 20, 20)}},
        "controlled_programme": {"controlled": {"small": (120, 40, 20), "programme": True}},
    }


def draw_design(name: str, rng: np.random.Generator):
    spec = _builders()[name]
    if spec.get("audit"):
        calibration = json.loads((designs.AUDIT_OUT / "calibration.json").read_text(encoding="utf-8"))
        return designs.audit_design(rng), calibration, spec
    rows, calibration, meta = designs.inputs()
    if spec.get("actual"):
        design = designs.actual(rows, calibration, rng)
        if design is None:
            raise RuntimeError("the real structure does not meet the registered minimums")
        return design, calibration, spec
    if "controlled" in spec:
        return designs.controlled(calibration, rng, **spec["controlled"]), calibration, spec
    persistence = None
    if "persistence" in spec:
        persistence = _persistence(meta, spec["persistence"], **spec.get("fixed", {}))
    for _ in range(50):
        design = designs.project(
            rows, calibration, anchors_total=spec["anchors_total"],
            ingested=meta["anchors_complete"], rng=rng,
            recurrence=spec.get("recurrence", "crp"), persistence=persistence,
            thinning=spec.get("thinning", True))
        if design is not None:
            return design, calibration, spec
    raise RuntimeError(f"{name}: the reference group never reached its minimum")


class Tests:
    """The candidate procedures on one design."""

    def __init__(self, design: power.Design):
        counts = pd.Series(design.category).value_counts()
        self.baseline = h7.REFERENCE if h7.REFERENCE in counts.index else counts.idxmax()
        self.tested = [g for g in design.kept if g != self.baseline]
        self.X = np.column_stack([np.ones(len(design.category))]
                                 + [(design.category == g).astype(float) for g in self.tested])
        self.R = np.zeros((len(self.tested), len(self.tested) + 1))
        self.R[:, 1:] = np.eye(len(self.tested))
        self.date, self.issuer = design.date, design.issuer
        self.masks = {g: design.category == g for g in design.kept}

    def __call__(self, y: np.ndarray, draws: int, seed: int) -> Dict[str, Optional[float]]:
        fit = stats.ols_cluster(self.X, y, [self.date, self.issuer])
        return {
            "v2_two_way_f": stats.wald(fit, self.R).get("p"),
            "wild_date_statistic": stats.wild_cluster_wald(
                self.X, y, self.R, self.date, draws=draws, seed=seed).get("p"),
            "wild_two_way_statistic": stats.wild_cluster_wald(
                self.X, y, self.R, self.date, other_clusters=self.issuer,
                draws=draws, seed=seed).get("p"),
            "cr2_satterthwaite_date": stats.cr2_wald(self.X, y, self.R, self.date).get("p"),
        }

    def shift(self, scenario: str, effect: float) -> Optional[np.ndarray]:
        out = np.zeros(len(self.date))
        if scenario == "all":
            for k, group in enumerate(self.tested):
                out[self.masks[group]] = effect if k % 2 == 0 else -effect
            return out
        group = scenario[4:]
        if group not in self.tested:
            return None
        out[self.masks[group]] = effect
        return out


def run(name: str, seed: int, simulations: int, draws: int = 999, mode: str = "null",
        per_design: int = 50) -> Dict[str, Any]:
    started = time.time()
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    cases = [("null", 0.0)] if mode != "power" else POWER_CASES
    tallies: Dict[str, Dict[str, Dict[str, int]]] = {}
    shapes: List[Dict[str, Any]] = []
    done = 0
    while done < simulations:
        design, calibration, spec = draw_design(name, rng)
        tests = Tests(design)
        shapes.append({**designs.describe(design), "tested": len(tests.tested)})
        common_share = spec.get("common_share", calibration["common_daily_share"])
        ar1 = spec.get("ar1", calibration["mean_ar1"])
        frame = pd.DataFrame({"group": design.category, "day0": design.date.astype(str),
                              "ticker": design.issuer.astype(str)})
        for _ in range(min(per_design, simulations - done)):
            noise = power.simulate_cars(design, rng, common_share=common_share, ar1=ar1)
            boot_seed = int(rng.integers(1 << 31))
            for scenario, effect in cases:
                planted = tests.shift(scenario, effect) if mode == "power" else np.zeros(len(noise))
                if planted is None:
                    continue
                y = noise + planted
                key = f"{scenario}@{effect}"
                cell = tallies.setdefault(key, {})
                if mode == "verdict":
                    frame["y"] = y
                    out = h7.contrasts(frame, tests.tested, "y", draws=draws, seed=boot_seed)
                    p = out["joint"].get("p")
                    entry = cell.setdefault("verdict", {"valid": 0, "joint_holm": 0, "supported": 0,
                                                        "any_contrast_significant": 0})
                    if p is not None:
                        entry["valid"] += 1
                        entry["joint_holm"] += p < HOLM
                        entry["supported"] += (p < HOLM
                                               and out["largest_significant_contrast"]
                                               >= h7.SPEC.material_effect)
                        entry["any_contrast_significant"] += bool(out["significant_classes"])
                    continue
                for procedure, p in tests(y, draws, boot_seed).items():
                    entry = cell.setdefault(procedure, {"valid": 0, "at_0.05": 0, "at_holm": 0})
                    if p is None:
                        continue
                    entry["valid"] += 1
                    entry["at_0.05"] += p < ALPHA
                    entry["at_holm"] += p < HOLM
            done += 1
    piece = {"design": name, "seed": seed, "mode": mode, "simulations": done, "draws": draws,
             "seconds": round(time.time() - started), "tallies": tallies,
             "shape": pd.DataFrame(shapes).mean(numeric_only=True).round(2).to_dict()}
    PIECES.mkdir(parents=True, exist_ok=True)
    (PIECES / f"{mode}_{name}_{seed}.json").write_text(json.dumps(piece, indent=1), encoding="utf-8")
    return piece


def _rate(hits: int, n: int) -> Dict[str, Any]:
    if not n:
        return {"rate": None, "ci": None}
    rate = hits / n
    half = 1.96 * np.sqrt(rate * (1 - rate) / n)
    return {"rate": rate, "ci": [max(0.0, rate - half), min(1.0, rate + half)]}


def merge() -> Dict[str, Any]:
    merged: Dict[str, Dict[str, Any]] = {}
    for path in sorted(PIECES.glob("*.json")):
        piece = json.loads(path.read_text(encoding="utf-8"))
        slot = merged.setdefault(piece["mode"], {}).setdefault(piece["design"], {
            "simulations": 0, "draws": piece["draws"], "pieces": 0, "seconds": 0,
            "shape": piece["shape"], "tallies": {}})
        slot["simulations"] += piece["simulations"]
        slot["pieces"] += 1
        slot["seconds"] += piece["seconds"]
        for case, cell in piece["tallies"].items():
            for procedure, entry in cell.items():
                total = slot["tallies"].setdefault(case, {}).setdefault(procedure, {})
                for key, value in entry.items():
                    total[key] = total.get(key, 0) + value
    DOCS.mkdir(parents=True, exist_ok=True)
    for mode, by_design in merged.items():
        out: Dict[str, Any] = {}
        for name, slot in by_design.items():
            rows: Dict[str, Any] = {}
            for case, cell in slot["tallies"].items():
                for procedure, entry in cell.items():
                    n = entry["valid"]
                    row = {"valid": n}
                    for key, value in entry.items():
                        if key != "valid":
                            row[key] = _rate(value, n)
                    rows.setdefault(case, {})[procedure] = row
            out[name] = {"simulations": slot["simulations"], "bootstrap_draws": slot["draws"],
                         "design": slot["shape"], "results": rows}
        (DOCS / f"{mode}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return merged


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] == "merge":
        merged = merge()
        print({mode: {name: slot["simulations"] for name, slot in designs_.items()}
               for mode, designs_ in merged.items()})
        return 0
    name, seed, simulations = argv[1], int(argv[2]), int(argv[3])
    draws = int(argv[4]) if len(argv) > 4 else 999
    mode = argv[5] if len(argv) > 5 else "null"
    piece = run(name, seed, simulations, draws, mode)
    print(json.dumps({k: v for k, v in piece.items() if k != "tallies"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
