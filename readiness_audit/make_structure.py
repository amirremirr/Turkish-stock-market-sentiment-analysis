"""Snapshot the database, fetch bars for new issuers into the *copy*, and write
the sample structure and the pre-sample calibration to the audit directory.

    python -m readiness_audit.make_structure
"""

from __future__ import annotations

import json
import sys

from readiness_audit import snapshot as snap
from stock_research import pipeline
from stock_research.data import kap


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    db = snap.take_snapshot()
    fetched = pipeline.ingest_prices(str(db))          # writes to the copy only
    panel, table, details = snap.load(db)
    structure, counts = snap.h7_structure(panel, table)
    calibration = snap.pre_sample_calibration(panel)
    progress = kap.progress(str(db))

    structure.to_csv(snap.AUDIT_OUT / "h7_structure.csv", index=False)
    table.drop(columns=["summary"], errors="ignore").to_pickle(snap.AUDIT_OUT / "kap_events.pkl")
    meta = {
        "snapshot_taken_from": "stock_research.db (read-only, SQLite backup API)",
        "frame": progress, "prices_fetched_into_copy": fetched,
        "sampled_disclosures": int(len(table)),
        "h7_eligibility_counts": counts,
        "h7_usable_by_category": structure["category"].value_counts().to_dict(),
        "h7_usable_dates_by_category": structure.groupby("category")["day0"].nunique().to_dict(),
        "h7_usable_issuers_by_category": structure.groupby("category")["ticker"].nunique().to_dict(),
        "h7_usable_anchors": int(structure["anchor"].nunique()),
        "h7_usable_dates": int(structure["day0"].nunique()),
        "h7_usable_issuers": int(structure["ticker"].nunique()),
        "calibration": {k: v for k, v in calibration.items() if k != "residual_sd_by_ticker"},
    }
    (snap.AUDIT_OUT / "structure_meta.json").write_text(
        json.dumps(meta, indent=1, default=str), encoding="utf-8")
    (snap.AUDIT_OUT / "calibration.json").write_text(
        json.dumps(calibration, indent=1, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
