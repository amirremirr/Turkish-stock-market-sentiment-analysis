"""A read-only copy of the research database, and the v3 sample structure.

    python -m v3_validation.snapshot [path-to-stock_research.db]

The research database lives in the main checkout and is never opened for
writing from here. It is copied with SQLite's backup API; everything else
works on the copy, which is where the v3 tables are created and where the
anchor sample's rows are entered as census members so that the v3 code can be
exercised on real metadata before the census exists.

"Structure" is: which issuer, which first actionable session, which test
group, which anchor, and whether the returns an event would need *exist*
(``notna()`` only). No event-window return is read.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from readiness_audit import snapshot as audit
from stock_research import events as sr_events
from stock_research import pipeline, store
from stock_research.config import BENCHMARK_TICKER, KAP_EARLIER_FRAMES, KAP_FRAME_VERSION, REPOSITORY_ROOT
from stock_research.data import kap, prices
from stock_research.hypotheses import h7
from stock_research.hypotheses.common import Context

OUT = REPOSITORY_ROOT / "outputs" / "stock_research" / "v3"
COPY = OUT / "copy.db"
#: The main checkout sits beside this worktree.
DEFAULT_SOURCE = REPOSITORY_ROOT.parent / "finance" / "stock_research.db"


def take_copy(source: Path = DEFAULT_SOURCE, target: Path = COPY) -> Path:
    return audit.take_snapshot(Path(source), target)


def adopt_anchor_sample(db: Path = COPY) -> Dict[str, int]:
    """Enter the anchor sample's listing rows as census members, in the copy.

    The real census does this page by page as its listing reaches each
    disclosure. Here it is done at once, for the rows that exist, so that the
    v3 event table can be built from them.
    """

    store.init_db(str(db))
    with store.connect(str(db)) as con:
        snapshot = con.execute(
            "SELECT snapshot_id FROM sr_raw_members ORDER BY retrieved_at DESC LIMIT 1").fetchone()[0]
    listed = kap._listed_member_ids(str(db), snapshot)
    marks = ",".join("?" * len(KAP_EARLIER_FRAMES))
    with store.connect(str(db)) as con:
        rows = con.execute(
            f"SELECT disclosure_index, block_seq, payload_json FROM sr_raw_kap_listing "
            f"WHERE frame_version IN ({marks})", KAP_EARLIER_FRAMES).fetchall()
        for row in rows:
            con.execute("INSERT OR IGNORE INTO sr_kap_frame_items VALUES (?,?,?,?,?)",
                        (KAP_FRAME_VERSION, row[0], row[1],
                         int(kap.is_target(json.loads(row[2]), listed)), store.now_iso()))
        targets = con.execute("SELECT COUNT(*) FROM sr_kap_frame_items WHERE is_target=1").fetchone()[0]
    return {"listing_rows": len(rows), "targets": int(targets)}


def load(db: Path = COPY):
    snapshot_id = pipeline.latest_snapshot(str(db))
    panel = prices.load_panel(snapshot_id, BENCHMARK_TICKER, str(db))
    details = sr_events.load_details(str(db))
    table = sr_events.build_kap_events(details, panel)
    table["anchor"] = details["block_seq"].to_numpy()
    return panel, table


def structure(panel, table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Rows the v3 H7 test could use, before thinning, by eligibility and by
    the existence of the returns they need."""

    eligible, counts = h7.prepare(Context(panel=panel, kap_events=table))
    anchors = table.set_index("event_id")["anchor"]
    eligible = eligible.copy()
    eligible["anchor"] = eligible["event_id"].map(anchors)
    flags = pd.DataFrame([audit.availability(panel, r.ticker, r.day0) for r in eligible.itertuples()],
                         index=eligible.index)
    usable = eligible[flags.all(axis=1)].copy()
    counts["price_and_estimation_available"] = int(len(usable))
    usable["position"] = usable["day0"].map(panel.calendar.index).astype(int)
    kept = h7.thin(usable, panel.calendar)
    counts["after_issuer_thinning"] = int(kept.sum())
    usable["kept_by_thinning"] = kept
    return usable[["event_id", "anchor", "ticker", "day0", "position", "group", "category",
                   "bucket", "kept_by_thinning"]], counts


def persistence(rows: pd.DataFrame, horizon: int = 10) -> Dict[str, Any]:
    """How often an event's issuer already filed in the same group at an
    earlier ingested anchor no more than *horizon* sessions before.

    A lower bound on recurrence: an anchor is fifty consecutive filings, part
    of a day, and the days between anchors are not sampled at all.
    """

    out: Dict[str, Any] = {}
    for group, part in rows.groupby("group"):
        hits = total = 0
        by_issuer = {t: np.sort(g["position"].unique()) for t, g in part.groupby("ticker")}
        anchor_positions = np.sort(part.groupby("anchor")["position"].min().to_numpy())
        for row in part.itertuples():
            # Only events that have an earlier anchor in range can recur at all.
            earlier = anchor_positions[(anchor_positions < row.position - 1)
                                       & (anchor_positions >= row.position - horizon)]
            if len(earlier) == 0:
                continue
            total += 1
            own = by_issuer[row.ticker]
            hits += bool(((own < row.position - 1) & (own >= row.position - horizon)).any())
        out[group] = {"events_with_an_earlier_anchor_in_range": total, "issuer_filed_before": hits,
                      "share": (hits / total) if total else None}
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = argv if argv is not None else sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    db = take_copy(Path(argv[0]) if argv else DEFAULT_SOURCE)
    adopted = adopt_anchor_sample(db)
    fetched = pipeline.ingest_prices(str(db))               # writes to the copy only
    panel, table = load(db)
    rows, counts = structure(panel, table)
    calibration = audit.pre_sample_calibration(panel)
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        complete = [r[0] for r in con.execute(
            "SELECT block_seq FROM sr_kap_frame WHERE frame_version IN (%s) AND status='complete' "
            "ORDER BY block_seq" % ",".join("?" * len(KAP_EARLIER_FRAMES)), KAP_EARLIER_FRAMES)]
    finally:
        con.close()
    rows.to_csv(OUT / "structure.csv", index=False)
    table.drop(columns=["summary"], errors="ignore").to_pickle(OUT / "kap_events.pkl")
    thinned = rows[rows["kept_by_thinning"]]
    meta = {
        "source": "copy of the research database (SQLite backup API, source opened read-only)",
        "adopted": adopted, "prices_fetched_into_copy": fetched,
        "anchors_complete": complete, "sampled_disclosures": int(len(table)),
        "eligibility_counts": counts,
        "usable_by_group": rows["group"].value_counts().to_dict(),
        "thinned_by_group": thinned["group"].value_counts().to_dict(),
        "thinned_issuers_by_group": thinned.groupby("group")["ticker"].nunique().to_dict(),
        "thinned_dates_by_group": thinned.groupby("group")["day0"].nunique().to_dict(),
        "closure_delayed": int(table.get("closure_delayed", pd.Series(dtype=bool)).sum()),
        "persistence": persistence(rows),
        "calendar_schedule_mismatches": panel.calendar.schedule_mismatches(
            "2023", known_gaps=panel.calendar_gaps),
        "calibration": {k: v for k, v in calibration.items() if k != "residual_sd_by_ticker"},
    }
    (OUT / "structure_meta.json").write_text(json.dumps(meta, indent=1, default=str),
                                             encoding="utf-8")
    (OUT / "calibration.json").write_text(json.dumps(calibration, indent=1, default=str),
                                          encoding="utf-8")
    print(json.dumps(meta, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
