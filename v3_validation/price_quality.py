"""What the v3 price rules do to the real price data. A data-quality check.

    python -m v3_validation.price_quality

Works on the copy of the research database. It fetches the second source for
every sampled issuer code into the copy, builds the panel twice (v2 rules; v3
rules with reconciliation) and counts what changed. Single-session returns of
every stock on every session are compared; no event window is built and no
market model is fitted. For sampled events it reports only whether the returns
a window needs still *exist*.

Writes counts, dates and tickers (no prices) to
``docs/stock_research/protocol_v3/validation/price_reconciliation.json``.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from readiness_audit import snapshot as audit
from stock_research import pipeline
from stock_research.config import BENCHMARK_TICKER, RECONCILE_TOLERANCE
from stock_research.data import prices, secondary
from v3_validation import snapshot as snap
from v3_validation.calibrate import DOCS


def run(db=snap.COPY) -> Dict[str, Any]:
    db = str(db)
    fetched = pipeline.ingest_secondary_prices(db)
    coverage = pipeline.price_coverage(db)
    primary = pipeline.latest_snapshot(db)
    second_id = pipeline.latest_secondary_snapshot(db)
    plain = prices.load_panel(primary, BENCHMARK_TICKER, db)
    checked = prices.load_panel(primary, BENCHMARK_TICKER, db, secondary_snapshot=second_id)
    second, actions = secondary.load(second_id, db)

    # Where the disagreements are, by date, and how large.
    rows = []
    for ticker, frame in plain.bars.items():
        if ticker not in second:
            continue
        close = second[ticker]["close_adjusted"].reindex(frame.index).astype(float)
        other = close / close.shift(1) - 1
        both = pd.DataFrame({"ticker": ticker, "date": frame.index, "first": frame["ret"].to_numpy(),
                             "second": other.to_numpy()}).dropna()
        rows.append(both)
    compared = pd.concat(rows, ignore_index=True)
    compared["gap"] = (compared["first"] - compared["second"]).abs()
    off = compared[compared["gap"] > RECONCILE_TOLERANCE]
    by_date = off.groupby("date").size()
    total_by_date = compared.groupby("date").size()
    mass = {d: f"{int(n)} of {int(total_by_date[d])}" for d, n in by_date.items() if n >= 10}
    stocks = compared[compared["ticker"] != BENCHMARK_TICKER]

    # Sampled events: does the window's data still exist under each rule set?
    table = pd.read_pickle(snap.OUT / "kap_events.pkl")
    structure = pd.read_csv(snap.OUT / "structure.csv")
    events = structure[structure["kept_by_thinning"]]
    still = [all(audit.availability(checked, r.ticker, r.day0).values()) for r in events.itertuples()]
    lost = events[~np.array(still)]

    consistency = prices.action_consistency(checked, actions)
    wrong_factor = [
        {"ticker": row["ticker"], "date": row["date"], "kind": row["kind"], "return": row["return"]}
        for row in consistency["rows"]
        if row["listed_by"] == "second" and row["return"] != "verified"
        and row["kind"] in secondary.CAPITAL_KINDS]
    out = {
        "copy_of_database_with_anchors_complete": len(json.loads(
            (snap.OUT / "structure_meta.json").read_text(encoding="utf-8"))["anchors_complete"]),
        "second_source_fetch": fetched, "price_coverage": coverage,
        "tolerance": RECONCILE_TOLERANCE,
        "returns_compared": int(len(compared)),
        "absolute_difference_quantiles": {
            q: float(stocks["gap"].quantile(float(q))) for q in ("0.5", "0.9", "0.975", "0.99", "0.999")},
        "share_differing_by_more_than": {
            f"{t:.3f}": float((stocks["gap"] > t).mean()) for t in (0.002, 0.005, 0.01, 0.02)},
        "disagreements_over_tolerance": int(len(off)),
        "dates_with_ten_or_more_disagreements": mass,
        "disagreements_on_those_dates": int(sum(n for n in by_date if n >= 10)),
        "disagreements_elsewhere": int(sum(n for n in by_date if n < 10)),
        "disagreements_in_2023": int(off["date"].str.startswith("2023").sum()),
        "benchmark_disagreements": int((off["ticker"] == BENCHMARK_TICKER).sum()),
        "reconciliation_summary": {k: (v if k != "tickers_without_second_source" else len(v))
                                   for k, v in checked.reconciliation.items()},
        "tickers_without_second_source": checked.reconciliation["tickers_without_second_source"][:40],
        "returns_kept_under_v2_rules": int(sum(int(f["ret"].notna().sum()) for f in plain.bars.values())),
        "returns_kept_under_v3_rules": int(sum(int(f["ret"].notna().sum()) for f in checked.bars.values())),
        "corporate_action_consistency": consistency["summary"],
        "capital_increases_whose_return_is_not_verified": wrong_factor,
        "action_date_limit_flags": sorted(
            (t, i["date"]) for t, found in checked.issues.items() for i in found
            if i["issue"] == prices.ISSUE_ACTION_LIMIT),
        "calendar_schedule_mismatches": checked.calendar.schedule_mismatches(
            pipeline.SAMPLE_YEAR, known_gaps=checked.calendar_gaps),
        "sampled_events": {
            "eligible_thinned_with_data_under_v2_rules": int(len(events)),
            "of_which_still_have_their_data_under_v3_rules": int(sum(still)),
            "lost_by_group": lost["group"].value_counts().to_dict(),
            "lost_by_day0": lost["day0"].value_counts().sort_index().to_dict(),
            "sampled_disclosures_with_day0_on_a_mass_disagreement_date": int(
                table["day0"].isin(list(mass)).sum()),
        },
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "price_reconciliation.json").write_text(json.dumps(out, indent=1, default=str),
                                                    encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(run(), indent=1, default=str))
