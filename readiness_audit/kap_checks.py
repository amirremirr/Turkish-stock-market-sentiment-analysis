"""Completeness and provenance of the KAP sample, from the snapshot copy.

    python -m readiness_audit.kap_checks

Counts, identifiers, timestamps and digests only. Nothing here joins a
disclosure to a price.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any, Dict

import pandas as pd

from readiness_audit import snapshot as snap
from stock_research import calendar as cal
from stock_research import pipeline
from stock_research.config import KAP_FRAME_VERSION, KAP_TARGET_CLASSES
from stock_research.data import kap


def summarise(db=None) -> Dict[str, Any]:
    db = str(db or snap.SNAPSHOT_DB)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out: Dict[str, Any] = {"frame": kap.progress(db)}

    frame = pd.read_sql_query(
        "SELECT block_seq, block_start, process_order, status, listed, targets, details_done, "
        "updated_at FROM sr_kap_frame WHERE frame_version=? ORDER BY block_seq", con,
        params=(KAP_FRAME_VERSION,))
    done = frame[frame["status"].isin(kap.ATTEMPTED)]
    out["anchors"] = {
        "total": int(len(frame)), "by_status": frame["status"].value_counts().to_dict(),
        "attempted_block_seq": done["block_seq"].tolist(),
        "listed_items_per_attempted_anchor": {
            "min": int(done["listed"].min()), "max": int(done["listed"].max())},
        "targets_per_attempted_anchor": {
            "min": int(done["targets"].min()), "median": float(done["targets"].median()),
            "max": int(done["targets"].max())},
    }

    snapshot = con.execute(
        "SELECT snapshot_id FROM sr_raw_members ORDER BY retrieved_at DESC LIMIT 1").fetchone()[0]
    listed = kap._listed_member_ids(db, snapshot)
    members = pd.read_sql_query("SELECT member_id, member_type, stock_codes FROM sr_raw_members "
                                "WHERE snapshot_id=?", con, params=(snapshot,))
    out["members"] = {
        "snapshot_id": snapshot, "rows": int(len(members)),
        "by_type": members["member_type"].value_counts().to_dict(),
        "listed_with_stock_code": len(listed),
    }

    listing = pd.read_sql_query(
        "SELECT l.disclosure_index, l.block_seq, l.disclosure_class, l.company_id, "
        "d.published_raw, d.body_text, d.subject_tr FROM sr_raw_kap_listing l "
        "LEFT JOIN sr_raw_kap_detail d USING (disclosure_index) "
        "JOIN sr_kap_frame f ON f.frame_version=l.frame_version AND f.block_seq=l.block_seq "
        "WHERE l.frame_version=? AND f.status IN (?,?)", con,
        params=(KAP_FRAME_VERSION, *kap.ATTEMPTED))
    in_class = listing["disclosure_class"].isin(KAP_TARGET_CLASSES)
    is_listed = listing["company_id"].astype(str).isin(listed)
    target = in_class & is_listed
    out["listing"] = {
        "items": int(len(listing)), "by_class": listing["disclosure_class"].value_counts().to_dict(),
        "targets": int(target.sum()),
        "not_target_wrong_class": int((~in_class).sum()),
        "not_target_filer_not_in_listed_roster": int((in_class & ~is_listed).sum()),
        "distinct_filers_not_in_listed_roster": int(
            listing.loc[in_class & ~is_listed, "company_id"].nunique()),
    }
    targets = listing[target]
    stamps = targets["published_raw"].map(cal.parse_kap_time)
    out["details"] = {
        "targets": int(len(targets)),
        "with_detail": int(targets["published_raw"].notna().sum()),
        "without_detail": int(targets["published_raw"].isna().sum()),
        "unavailable_indices": kap.unavailable_details(db),
        "timestamp_unparseable": int((targets["published_raw"].notna() & stamps.isna()).sum()),
        "empty_body_text": int((targets["published_raw"].notna()
                                & (targets["body_text"].fillna("").str.len() == 0)).sum()),
        "missing_subject": int((targets["published_raw"].notna()
                                & targets["subject_tr"].isna()).sum()),
    }
    # Disclosure numbers are issued in order of publication, so within the
    # sample a later number should never carry an earlier timestamp.
    ordered = targets.assign(stamp=stamps).dropna(subset=["stamp"]).sort_values("disclosure_index")
    out["details"]["timestamps_out_of_order"] = int((ordered["stamp"].diff().dt.total_seconds() < 0).sum())

    failures = pd.read_sql_query("SELECT disclosure_index, http_status FROM sr_kap_detail_failures",
                                 con)
    out["failures"] = {
        "refused_requests": int(len(failures)),
        "distinct_disclosures": int(failures["disclosure_index"].nunique()),
        "by_http_status": failures["http_status"].value_counts().to_dict(),
        "max_attempts_on_one_disclosure": int(failures.groupby("disclosure_index").size().max())
        if len(failures) else 0,
    }
    out["ingestion_log"] = [dict(r) for r in con.execute(
        "SELECT run_id, event, at, ingester_sha256, code_commit FROM sr_ingest_log ORDER BY event_id")]
    out["ingester_sha256_now"] = kap.ingester_sha256()
    con.close()

    provenance = pipeline.dataset_provenance(db)
    out["digests_at_snapshot"] = {k: provenance[k] for k in (
        "kap_sample_digest", "price_snapshot_id", "price_snapshot_digest",
        "members_snapshot_digest", "counts")}

    table = pd.read_pickle(snap.AUDIT_OUT / "kap_events.pkl")
    resolved = table["ticker"].notna()
    by_category = table.assign(resolved=resolved).groupby("category")["resolved"].agg(
        events="size", resolved="sum")
    by_category["unresolved_share"] = 1 - by_category["resolved"] / by_category["events"]
    out["issuer_mapping"] = {
        "sampled_disclosures": int(len(table)), "ticker_resolved": int(resolved.sum()),
        "unresolved": int((~resolved).sum()),
        "unresolved_share": float((~resolved).mean()),
        "unresolved_distinct_filers": int(table.loc[~resolved, "sender_id"].nunique()),
        "by_category": by_category.sort_values("events", ascending=False).round(3)
        .reset_index().to_dict("records"),
    }
    out["classification"] = {
        "by_rule": table["category_rule"].value_counts().to_dict(),
        "by_category": table["category"].value_counts().to_dict(),
        "insider_parse_status": table.loc[table["category"] == "insider_trade",
                                          "insider_parse_status"].value_counts().to_dict(),
    }
    (snap.AUDIT_OUT / "kap_checks.json").write_text(
        json.dumps(out, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(summarise(), indent=1, default=str, ensure_ascii=False))
