"""Completeness, eligibility and missingness of the census. Counts only.

    python -m v3_validation.census_report [path-to-database]

Reads a database (by default the copy) and reports where the census stands
and what happens to its disclosures on the way to the H7 sample: by class, by
issuer and by date. For each eligible row it asks whether the returns its
window and its estimation window need *exist* under the v3 price rules
(``notna()``), and which rule removed them when they do not. No return value
is read, no market model is fitted, no CAR is computed.

Also rewrites ``structure.csv`` in the output directory from the same panel,
so the final calibration runs on the sample the registered run would use.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from readiness_audit import snapshot as audit
from stock_research import events as sr_events
from stock_research import pipeline
from stock_research.config import (
    BENCHMARK_TICKER, ESTIMATION_WINDOW, KAP_FRAME_VERSION,
)
from stock_research.data import kap, prices
from stock_research.hypotheses import h7
from stock_research.hypotheses.common import Context
from v3_validation import snapshot as snap
from v3_validation.calibrate import DOCS

WINDOW_ISSUES = (prices.ISSUE_DISAGREEMENT, prices.ISSUE_UNVERIFIED, prices.ISSUE_ACTION_LIMIT,
                 "move_beyond_price_limit", "unexplained_jump")


def _why_missing(panel, ticker: str, day0: str) -> str:
    """Which rule left an eligible row without usable data."""

    flags = audit.availability(panel, ticker, day0)
    if all(flags.values()):
        return "has_data"
    for name in ("in_panel", "day0_in_calendar", "horizon"):
        if not flags[name]:
            return name
    position = panel.calendar.index(day0)
    sessions = panel.calendar.sessions
    window = set(sessions[position:position + 6])
    found = {i["issue"] for i in panel.issues.get(ticker, []) if i["date"] in window}
    if not flags["window_0_5"]:
        for issue in WINDOW_ISSUES:
            if issue in found:
                return f"window: {issue}"
        return "window: no trade or no bar"
    lo = max(0, position + ESTIMATION_WINDOW[0])
    estimation = set(sessions[lo:max(0, position + ESTIMATION_WINDOW[1] + 1)])
    unverified = sum(1 for i in panel.issues.get(ticker, [])
                     if i["date"] in estimation and i["issue"] == prices.ISSUE_UNVERIFIED)
    return ("estimation: second source missing" if unverified > 30
            else "estimation: fewer than 60 returns")


def run(db: Path = snap.COPY) -> Dict[str, Any]:
    db = str(db)
    progress = kap.progress(db)
    verified = kap.verify_details(db)
    coverage = pipeline.price_coverage(db)
    primary, second = pipeline.latest_snapshot(db), pipeline.latest_secondary_snapshot(db)
    panel = prices.load_panel(primary, BENCHMARK_TICKER, db, secondary_snapshot=second)
    details = sr_events.load_details(db)
    table = sr_events.build_kap_events(details, panel)
    table["page"] = details["block_seq"].to_numpy()

    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    try:
        listing = pd.read_sql_query(
            "SELECT i.disclosure_index, i.block_seq, i.is_target, l.disclosure_class, "
            "l.company_id, l.frame_version AS first_listed_by, d.published_raw "
            "FROM sr_kap_frame_items i JOIN sr_raw_kap_listing l USING (disclosure_index) "
            "LEFT JOIN sr_raw_kap_detail d USING (disclosure_index) WHERE i.frame_version = ?",
            con, params=(KAP_FRAME_VERSION,))
        failures = pd.read_sql_query(
            "SELECT disclosure_index, http_status, substr(attempted_at, 1, 10) AS day "
            "FROM sr_kap_detail_failures WHERE frame_version = ?", con,
            params=(KAP_FRAME_VERSION,))
        log = [tuple(r) for r in con.execute(
            "SELECT event, at, frame_version, substr(ingester_sha256, 1, 16), code_commit "
            "FROM sr_ingest_log ORDER BY event_id")]
        protocols = [tuple(r) for r in con.execute(
            "SELECT version, substr(protocol_hash, 1, 16), registered_at FROM sr_protocols "
            "ORDER BY registered_at")]
        results = con.execute("SELECT COUNT(*) FROM sr_results").fetchone()[0]
        raw_by_frame = dict(con.execute(
            "SELECT frame_version, COUNT(*) FROM sr_raw_kap_listing GROUP BY 1").fetchall())
    finally:
        con.close()
    targets = listing[listing["is_target"] == 1]
    missing = targets[targets["published_raw"].isna()]

    # The road from a sampled disclosure to an H7 observation.
    eligible, counts = h7.prepare(Context(panel=panel, kap_events=table))
    eligible = eligible.copy()
    eligible["state"] = [_why_missing(panel, r.ticker, r.day0) for r in eligible.itertuples()]
    usable = eligible[eligible["state"] == "has_data"].copy()
    usable["position"] = usable["day0"].map(panel.calendar.index).astype(int)
    usable["kept_by_thinning"] = h7.thin(usable, panel.calendar)
    usable["anchor"] = usable["event_id"].map(table.set_index("event_id")["page"])
    final = usable[usable["kept_by_thinning"]]
    sizes = h7.group_sizes(final)
    sizes["tested"] = [h7.sufficient(row, reference=(name == h7.REFERENCE))
                       for name, row in sizes.iterrows()]

    table["resolved"] = table["ticker"].notna()
    table["group"] = table["category"].map(h7.group_of)
    by_class = table.groupby("category").agg(
        sampled=("event_id", "size"), ticker_resolved=("resolved", "sum"),
        closure_delayed=("closure_delayed", "sum"),
        update_or_correction=("is_update", lambda s: int((s == True).sum())),          # noqa: E712
        during_session=("bucket", lambda s: int((s == "during_session").sum())))
    by_class["in_h7"] = [h7.group_of(c) or "outside the test" for c in by_class.index]
    by_class["eligible_clean_timing"] = eligible.groupby("category").size()
    by_class["with_data"] = usable.groupby("category").size()
    by_class["after_thinning"] = final.groupby("category").size()
    by_class = by_class.fillna(0).astype({c: int for c in by_class.columns if c != "in_h7"})

    per_issuer = final.groupby("ticker").size()
    unresolved = table[~table["resolved"]]
    by_date = final.groupby("day0").size()
    sessions_2023 = [s for s in panel.calendar.sessions if s.startswith("2023")]
    month = lambda frame, column: frame[column].str.slice(0, 7).value_counts().sort_index().to_dict()  # noqa: E731

    out = {
        "frame": progress, "details_verified": {k: (v if isinstance(v, (bool, int)) else len(v))
                                                for k, v in verified.items()},
        "raw_listing_rows_by_frame_that_first_stored_them": raw_by_frame,
        "census_members": {
            "listed": int(len(listing)), "targets": int(len(targets)),
            "targets_first_listed_by": targets["first_listed_by"].value_counts().to_dict(),
            "not_targets": int((listing["is_target"] == 0).sum())},
        "missing_details": {
            "count": int(len(missing)), "disclosure_indices": missing["disclosure_index"].tolist(),
            "by_page": missing["block_seq"].value_counts().to_dict(),
            "by_filer": missing["company_id"].value_counts().to_dict(),
            "refusals_logged": int(len(failures)),
            "refusals_by_http_status": failures["http_status"].value_counts().to_dict(),
            "refusal_days_per_disclosure": failures.groupby("disclosure_index")["day"]
            .nunique().to_dict()},
        "ingestion_log": log, "protocols_registered": protocols, "results_stored": int(results),
        "price_coverage": coverage, "price_reconciliation": {
            k: (v if k != "tickers_without_second_source" else len(v))
            for k, v in panel.reconciliation.items()},
        "calendar_schedule_mismatches": panel.calendar.schedule_mismatches(
            pipeline.SAMPLE_YEAR, known_gaps=panel.calendar_gaps),
        "sampled_disclosures": int(len(table)),
        "eligibility_counts": counts,
        "eligible_rows_by_data_state": eligible["state"].value_counts().to_dict(),
        "data_state_by_group": {g: part["state"].value_counts().to_dict()
                                for g, part in eligible.groupby("group")},
        "by_class": by_class.reset_index().to_dict("records"),
        "h7_groups": sizes.reset_index().to_dict("records"),
        "tested_classes": [c for c in h7.INTEREST if c in sizes.index and sizes.loc[c, "tested"]],
        "reference_sufficient": bool(h7.REFERENCE in sizes.index and sizes.loc[h7.REFERENCE, "tested"]),
        "by_issuer": {
            "issuers_in_final_sample": int(len(per_issuer)),
            "events_per_issuer_quantiles": {q: float(per_issuer.quantile(float(q)))
                                            for q in ("0.5", "0.9", "0.99", "1.0")},
            "share_of_events_from_the_ten_largest": float(per_issuer.nlargest(10).sum() / max(1, len(final))),
            "events_removed_by_thinning": int((~usable["kept_by_thinning"]).sum()),
            "thinned_by_group": usable[~usable["kept_by_thinning"]]["group"].value_counts().to_dict(),
            "unresolved_disclosures": int(len(unresolved)),
            "unresolved_filers": int(unresolved["sender_id"].nunique()),
            "unresolved_by_class": unresolved["category"].value_counts().to_dict()},
        "by_date": {
            "sessions_2023": len(sessions_2023),
            "sessions_with_a_final_event": int(by_date.index.str.startswith("2023").sum()),
            "events_per_session_quantiles": {q: float(by_date.quantile(float(q)))
                                             for q in ("0.1", "0.5", "0.9", "1.0")},
            "final_events_by_month": month(final, "day0"),
            "sampled_by_publication_month": month(table.dropna(subset=["published_local"]),
                                                  "published_local"),
            "eligible_without_data_by_month": month(eligible[eligible["state"] != "has_data"], "day0"),
            "closure_delayed": int(table["closure_delayed"].sum()),
            "final_events_on_2023_01_05_or_06": int(final["day0"].isin(
                ["2023-01-05", "2023-01-06"]).sum())},
    }
    usable[["event_id", "anchor", "ticker", "day0", "position", "group", "category", "bucket",
            "kept_by_thinning"]].to_csv(snap.OUT / "structure.csv", index=False)
    calibration = audit.pre_sample_calibration(panel)
    (snap.OUT / "calibration.json").write_text(json.dumps(calibration, indent=1, default=str),
                                               encoding="utf-8")
    meta = {"anchors_complete": [], "persistence": snap.persistence(usable),
            "source": "census_report"}
    (snap.OUT / "structure_meta.json").write_text(json.dumps(meta, indent=1, default=str),
                                                  encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "census_report.json").write_text(json.dumps(out, indent=1, default=str),
                                             encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = run(Path(sys.argv[1]) if len(sys.argv) > 1 else snap.COPY)
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_class",)}, indent=1,
                     default=str)[:6000])
