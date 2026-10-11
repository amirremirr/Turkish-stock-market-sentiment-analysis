"""Sample designs for validating the v3 test: who files what, when.

A design is structure only. Returns are simulated on top of it by the
readiness audit's ``simulate_cars`` (issuer volatility, a market-wide daily
shock, serial correlation, heavy tails, an estimated intercept; events of one
issuer share the simulated days their windows have in common).

Three sources of design:

``audit_design``     the readiness audit's own projection under v2's rules.
                     The audit's rejection rates on it are the benchmark.
``project``          the v3 rules (test groups, reference group, issuer
                     thinning, issuer minimum) on the anchors ingested so far,
                     projected to 80 anchors or to a census.
``controlled``       made-up layouts that isolate one difficulty.

Issuer recurrence is the dependence a date-level bootstrap cannot see, so the
projection has a knob for it. ``persistence`` is, per group, the chance that
an event's issuer is one that filed in the same group within the last ten
sessions: a company running a buyback programme files day after day.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from readiness_audit import power
from stock_research.config import (
    H7_MIN_DATES, H7_MIN_EVENTS, H7_MIN_ISSUERS, H7_REFERENCE_MIN_DATES,
    H7_REFERENCE_MIN_EVENTS, H7_REFERENCE_MIN_ISSUERS, REPOSITORY_ROOT,
)
from stock_research.hypotheses import h7
from v3_validation import snapshot as snap

SESSIONS = 440
CENSUS_ANCHORS = 345          # about every material-event disclosure of 2023
#: The readiness audit's inputs, in the main checkout (not committed).
AUDIT_OUT = REPOSITORY_ROOT.parent / "finance" / "outputs" / "stock_research" / "readiness"
AUDIT_INGESTED = [0, 9, 11, 13, 24, 26, 37, 39, 50, 52, 63, 65, 76, 78]


def inputs():
    rows = pd.read_csv(snap.OUT / "structure.csv")
    calibration = json.loads((snap.OUT / "calibration.json").read_text(encoding="utf-8"))
    meta = json.loads((snap.OUT / "structure_meta.json").read_text(encoding="utf-8"))
    return rows, calibration, meta


def audit_design(rng: np.random.Generator) -> power.Design:
    """The readiness audit's projected 80-anchor sample, v2 rules, unchanged."""

    structure = pd.read_csv(AUDIT_OUT / "h7_structure.csv")
    calibration = json.loads((AUDIT_OUT / "calibration.json").read_text(encoding="utf-8"))
    return power.project(structure, calibration, anchors_total=80, frame_anchors=80,
                         sessions=SESSIONS, rng=rng, ingested=AUDIT_INGESTED)


def _thin(events: pd.DataFrame) -> pd.DataFrame:
    """One row per issuer per six sessions: the earliest, then the next more
    than five sessions later. The rule of ``h7.thin``, on session positions."""

    keep = np.zeros(len(events), dtype=bool)
    order = np.lexsort((events["position"].to_numpy(), events["ticker"].to_numpy()))
    tickers = events["ticker"].to_numpy()[order]
    positions = events["position"].to_numpy()[order]
    last_ticker, last_kept = None, -10 ** 9
    for where, ticker, position in zip(order, tickers, positions):
        if ticker != last_ticker:
            last_ticker, last_kept = ticker, -10 ** 9
        if position - last_kept > h7.PRIMARY_WINDOW[1]:
            keep[where] = True
            last_kept = position
    return events[keep]


def qualify(events: pd.DataFrame) -> Optional[List[str]]:
    """Tested classes under the registered minimums, or None if the reference
    group itself falls short."""

    sizes = events.groupby("group").agg(n=("ticker", "size"), d=("position", "nunique"),
                                        i=("ticker", "nunique"))
    if h7.REFERENCE not in sizes.index:
        return None
    ref = sizes.loc[h7.REFERENCE]
    if not (ref["n"] >= H7_REFERENCE_MIN_EVENTS and ref["d"] >= H7_REFERENCE_MIN_DATES
            and ref["i"] >= H7_REFERENCE_MIN_ISSUERS):
        return None
    return [c for c in h7.INTEREST if c in sizes.index
            and sizes.loc[c, "n"] >= H7_MIN_EVENTS and sizes.loc[c, "d"] >= H7_MIN_DATES
            and sizes.loc[c, "i"] >= H7_MIN_ISSUERS]


def project(rows: pd.DataFrame, calibration: Dict[str, Any], *, anchors_total: int,
            ingested: Sequence[int], rng: np.random.Generator, recurrence: str = "crp",
            persistence: Optional[Dict[str, float]] = None, thinning: bool = True,
            frame_anchors: int = 80, sessions: int = SESSIONS) -> Optional[power.Design]:
    """A full-size v3 sample built from the anchors ingested so far.

    Anchors already seen keep their events when the target is the 80-anchor
    frame. Every other anchor gets a session by interpolation and the events
    of a randomly drawn ingested anchor (empty ones included). Who files them
    is decided by ``recurrence`` as in the readiness audit (``crp``,
    ``resample``, ``fresh``; ``mixed`` keeps the copied anchor's issuer for
    half the events), after ``persistence`` has had its say.
    Returns None when the reference group does not reach its minimum.
    """

    seen = sorted(rows["anchor"].unique())
    by_anchor = {a: g for a, g in rows.groupby("anchor")}
    first_pos = {a: int(g["position"].min()) for a, g in by_anchor.items()}
    numbers = np.arange(anchors_total) * (frame_anchors / anchors_total)
    interpolated = np.interp(numbers, seen, [first_pos[a] for a in seen])
    pool_of_anchors = sorted(set(seen) | {int(a) for a in ingested})
    pools: Dict[str, Dict[str, int]] = {
        group: part["ticker"].value_counts().to_dict() for group, part in rows.groupby("group")}
    recent: Dict[str, List[tuple]] = {}           # group -> [(position, issuer)]
    persistence = persistence or {}
    fresh = [0]

    parts = []
    for k in range(anchors_total):
        number = numbers[k]
        observed = anchors_total == frame_anchors and int(number) in pool_of_anchors
        source = int(number) if observed else int(rng.choice(pool_of_anchors))
        if source not in by_anchor:
            continue
        block = by_anchor[source]
        start = first_pos[source] if observed else int(round(interpolated[k]))
        positions = np.clip(start + block["position"].to_numpy() - first_pos[source],
                            136, sessions - 21)
        tickers = block["ticker"].to_numpy().astype(object)
        groups = block["group"].to_numpy()
        if not observed and recurrence != "resample":
            tickers = tickers.copy()
            for i, group in enumerate(groups):
                pool = pools.setdefault(group, {})
                if recurrence == "mixed" and rng.random() < 0.5:
                    continue                      # the copied anchor's own issuer again
                active = [issuer for position, issuer in recent.get(group, [])
                          if 0 < positions[i] - position <= 10]
                if active and rng.random() < persistence.get(group, 0.0):
                    name = active[int(rng.integers(len(active)))]
                else:
                    total = sum(pool.values())
                    singles = sum(1 for v in pool.values() if v == 1)
                    new = 1.0 if (recurrence == "fresh" or total == 0) else singles / total
                    if rng.random() < new:
                        fresh[0] += 1
                        name = f"NEW{fresh[0]:05d}"
                    else:
                        names = list(pool)
                        weights = np.array([pool[n] for n in names], dtype=float)
                        name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
                pool[name] = pool.get(name, 0) + 1
                tickers[i] = name
        for group, position, ticker in zip(groups, positions, tickers):
            recent.setdefault(group, []).append((int(position), ticker))
        parts.append(pd.DataFrame({"ticker": tickers, "group": groups, "position": positions}))
    events = pd.concat(parts, ignore_index=True)
    events = events.drop_duplicates(["ticker", "position"]).reset_index(drop=True)
    if thinning:
        events = _thin(events).reset_index(drop=True)
    return _design(events, calibration, rng, anchors_total, sessions)


def _design(events: pd.DataFrame, calibration: Dict[str, Any], rng: np.random.Generator,
            anchors: int, sessions: int) -> Optional[power.Design]:
    tested = qualify(events)
    if not tested:
        return None
    kept = [h7.REFERENCE, *tested]
    events = events[events["group"].isin(kept)].reset_index(drop=True)
    tickers, issuer = np.unique(events["ticker"], return_inverse=True)
    known = calibration["residual_sd_by_ticker"]
    pool = np.array([v for v in known.values() if np.isfinite(v)])
    sigma = np.array([known[t] if t in known and np.isfinite(known[t]) else rng.choice(pool)
                      for t in tickers])
    position = events["position"].to_numpy()
    return power.Design(issuer, position, position, events["group"].to_numpy(), kept, sigma,
                        sessions, anchors)


def controlled(calibration: Dict[str, Any], rng: np.random.Generator, *,
               small=(30, 20, 20), programme: bool = False) -> power.Design:
    """A reference group and three large classes, plus one class of the given
    (events, dates, issuers). With ``programme`` the small class's issuers
    file on consecutive sessions, so their windows overlap."""

    rows = []
    layout = [(h7.REFERENCE, 200, 80, 150), (h7.INTEREST[0], 300, 80, 200),
              (h7.INTEREST[1], 300, 80, 200), (h7.INTEREST[2], 300, 80, 200),
              (h7.INTEREST[3], *small)]
    for k, (group, events, dates, issuers) in enumerate(layout):
        days = 140 + np.sort(rng.choice(260, size=dates, replace=False))
        names = [f"{group}_{i}" for i in range(issuers)]
        for e in range(events):
            issuer = names[e % issuers]
            if programme and group == h7.INTEREST[3]:
                # Each issuer's events on a run of consecutive sessions.
                day = 140 + (names.index(issuer) * 7) % 200 + (e // issuers)
            else:
                day = int(days[rng.integers(dates)])
            rows.append({"ticker": issuer, "group": group, "position": day})
    events = pd.DataFrame(rows).drop_duplicates(["ticker", "position"]).reset_index(drop=True)
    tickers, issuer = np.unique(events["ticker"], return_inverse=True)
    pool = np.array([v for v in calibration["residual_sd_by_ticker"].values() if np.isfinite(v)])
    position = events["position"].to_numpy()
    return power.Design(issuer, position, position, events["group"].to_numpy(),
                        [g for g, *_ in layout], rng.choice(pool, size=len(tickers)),
                        SESSIONS, 0)


def describe(design: power.Design) -> Dict[str, Any]:
    counts = pd.Series(design.category).value_counts()
    frame = pd.DataFrame({"g": design.category, "d": design.date, "i": design.issuer})
    per_issuer = frame.groupby("i").size()
    return {"events": int(len(design.category)), "dates": int(frame["d"].nunique()),
            "issuers": int(frame["i"].nunique()), "groups": len(design.kept),
            "smallest_group": int(counts.min()),
            "fewest_dates_in_a_group": int(frame.groupby("g")["d"].nunique().min()),
            "fewest_issuers_in_a_group": int(frame.groupby("g")["i"].nunique().min()),
            "share_of_events_from_repeat_issuers": float(
                per_issuer[per_issuer > 1].sum() / len(frame)),
            "largest_issuer": int(per_issuer.max())}
