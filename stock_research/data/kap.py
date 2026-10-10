"""Resumable KAP sample ingestion from the MKK API Portal.

The listing endpoint returns 50 disclosures per call with no timestamp; the
time, subject and body need one detail call each, at 6 calls a minute. So the
study does not take a census. It takes ``KAP_FRAME_BLOCKS`` anchor indices
spaced evenly across the available range and, at each, the next 50
material-event disclosures. The anchors are fixed by constants and chosen
without reference to any price.

Blocks are processed in a strided order, so an interrupted run still covers the
whole year thinly rather than its first weeks thickly. Every listing row and
every detail is committed as it arrives; a rerun skips what is already stored.

Nothing here reads a price or computes a return.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

from stock_research import store
from stock_research.config import (
    KAP_FRAME_BLOCKS, KAP_FRAME_FIRST_INDEX, KAP_FRAME_LAST_INDEX,
    KAP_FRAME_STRIDE, KAP_FRAME_VERSION, KAP_LISTED_MEMBER_MARK,
    KAP_LISTING_TYPES, KAP_TARGET_CLASSES, KAP_THROTTLE_SECONDS, REPOSITORY_ROOT,
)

logger = logging.getLogger("stock_research.kap")
SOURCE = "mkk-api-portal-vyk-dev"


def frame_blocks() -> List[Dict[str, int]]:
    """The fixed sample: evenly spaced anchors and their processing order.

    The last anchor sits a little before the end of the range so that 50
    filtered disclosures still follow it.
    """

    span = KAP_FRAME_LAST_INDEX - 1500 - KAP_FRAME_FIRST_INDEX
    blocks = []
    for seq in range(KAP_FRAME_BLOCKS):
        start = KAP_FRAME_FIRST_INDEX + round(seq * span / (KAP_FRAME_BLOCKS - 1))
        blocks.append({"block_seq": seq, "block_start": start})
    order = sorted(range(KAP_FRAME_BLOCKS), key=lambda k: (k * KAP_FRAME_STRIDE) % KAP_FRAME_BLOCKS)
    for position, seq in enumerate(order):
        blocks[seq]["process_order"] = position
    return blocks


def _base_url() -> str:
    from config import KAP_BASE_URL      # loads .env as a side effect
    return KAP_BASE_URL


def _auth():
    key, secret = os.environ.get("MKK_API_KEY", ""), os.environ.get("MKK_API_SECRET", "")
    if not (key and secret):
        raise RuntimeError("MKK_API_KEY / MKK_API_SECRET not set (env or .env)")
    return key, secret


def api_get(path: str, params: Optional[dict] = None, *, attempts: int = 5) -> Any:
    """GET with the plan's throttle. Sleeps after every call, success or not."""

    last_error = None
    for _ in range(attempts):
        try:
            response = requests.get(_base_url() + path, params=params,
                                    auth=_auth(), timeout=90)
        except requests.RequestException as exc:
            last_error = repr(exc)
            time.sleep(KAP_THROTTLE_SECONDS * 3)
            continue
        if response.status_code == 429:
            last_error = "429"
            time.sleep(65)
            continue
        time.sleep(KAP_THROTTLE_SECONDS)
        if response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            time.sleep(30)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"KAP request failed after {attempts} attempts: {path} ({last_error})")


def body_text(detail: Dict[str, Any], limit: int = 20000) -> str:
    """Plain Turkish text of a disclosure body (base64 HTML in the payload)."""

    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    parts = []
    for message in detail.get("htmlMessages") or []:
        encoded = (message or {}).get("tr") if isinstance(message, dict) else None
        if not encoded:
            continue
        try:
            html = base64.b64decode(encoded).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            continue
        soup = BeautifulSoup(html, "lxml")
        # English cells sit in the same document; keep the Turkish ones.
        for node in soup.select(".content-en"):
            node.decompose()
        parts.append(re.sub(r"\s+", " ", soup.get_text(" ")).strip())
    return " ".join(parts)[:limit]


def detail_row(detail: Dict[str, Any], raw_bytes: bytes) -> Dict[str, Any]:
    metadata = {k: v for k, v in detail.items() if k != "htmlMessages"}
    return {
        "disclosure_index": int(detail["disclosureIndex"]),
        "sender_id": detail.get("senderId"),
        "sender_title": detail.get("senderTitle"),
        "sender_codes": store.dumps(detail.get("senderExchCodes") or []),
        "related_stocks": store.dumps(detail.get("relatedStocks") or []),
        "disclosure_type": detail.get("disclosureType"),
        "disclosure_class": detail.get("disclosureClass"),
        "disclosure_reason": detail.get("disclosureReason"),
        "subject_tr": (detail.get("subject") or {}).get("tr"),
        "subject_en": (detail.get("subject") or {}).get("en"),
        "summary_tr": (detail.get("summary") or {}).get("tr"),
        "published_raw": detail.get("time"),
        "event_type_code": detail.get("eventType"),
        "body_text": body_text(detail),
        "metadata_json": store.dumps(metadata),
        "payload_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source": SOURCE,
        "retrieved_at": store.now_iso(),
    }


def ensure_frame(db_path=None) -> None:
    store.init_db(db_path)
    with store.connect(db_path) as con:
        for block in frame_blocks():
            con.execute(
                """INSERT OR IGNORE INTO sr_kap_frame
                   (frame_version, block_seq, block_start, process_order)
                   VALUES (?,?,?,?)""",
                (KAP_FRAME_VERSION, block["block_seq"], block["block_start"],
                 block["process_order"]),
            )


def ensure_members(db_path=None) -> str:
    """Store one member snapshot; returns its id."""

    with store.connect(db_path) as con:
        row = con.execute(
            "SELECT snapshot_id FROM sr_raw_members ORDER BY retrieved_at DESC LIMIT 1"
        ).fetchone()
    if row:
        return row[0]
    members = api_get("/members")
    snapshot = "members-" + store.now_iso()
    with store.connect(db_path) as con:
        seen = set()
        for member in members:
            if member.get("id") in seen:
                continue
            seen.add(member.get("id"))
            con.execute(
                "INSERT INTO sr_raw_members VALUES (?,?,?,?,?,?,?)",
                (snapshot, member.get("id"), member.get("title"),
                 member.get("stockCode"), member.get("memberType"), SOURCE,
                 store.now_iso()),
            )
    return snapshot


def _listed_member_ids(db_path, snapshot: str) -> set:
    with store.connect(db_path) as con:
        return {
            r["member_id"] for r in con.execute(
                "SELECT member_id, member_type, stock_codes FROM sr_raw_members "
                "WHERE snapshot_id = ?", (snapshot,))
            if KAP_LISTED_MEMBER_MARK in (r["member_type"] or "") and r["stock_codes"]
        }


def is_target(item: Dict[str, Any], listed: set) -> bool:
    return (
        item.get("disclosureClass") in KAP_TARGET_CLASSES
        and str(item.get("companyId")) in listed
    )


def _store_detail(db_path, seq: int, index: int, detail: Dict[str, Any]) -> None:
    raw = json.dumps(detail, sort_keys=True, ensure_ascii=False).encode("utf-8")
    with store.connect(db_path) as con:
        con.execute(
            "INSERT OR IGNORE INTO sr_raw_kap_detail VALUES "
            "(:disclosure_index,:sender_id,:sender_title,:sender_codes,"
            ":related_stocks,:disclosure_type,:disclosure_class,"
            ":disclosure_reason,:subject_tr,:subject_en,:summary_tr,"
            ":published_raw,:event_type_code,:body_text,:metadata_json,"
            ":payload_sha256,:source,:retrieved_at)", detail_row(detail, raw),
        )
        con.execute(
            "UPDATE sr_kap_frame SET details_done=details_done+1, updated_at=? "
            "WHERE frame_version=? AND block_seq=?",
            (store.now_iso(), KAP_FRAME_VERSION, seq),
        )


def run(db_path=None, *, max_hours: float = 8.0,
        max_blocks: Optional[int] = None) -> Dict[str, Any]:
    ensure_frame(db_path)
    snapshot = ensure_members(db_path)
    listed = _listed_member_ids(db_path, snapshot)
    deadline = time.time() + max_hours * 3600
    done_blocks = 0

    while time.time() < deadline:
        with store.connect(db_path) as con:
            block = con.execute(
                "SELECT * FROM sr_kap_frame WHERE frame_version=? AND status!='complete' "
                "ORDER BY process_order LIMIT 1", (KAP_FRAME_VERSION,),
            ).fetchone()
        if block is None:
            break
        seq, start = block["block_seq"], block["block_start"]

        if block["status"] == "pending":
            listing = api_get("/disclosures", {
                "disclosureIndex": str(start),
                "disclosureTypes": ",".join(KAP_LISTING_TYPES),
            })
            with store.connect(db_path) as con:
                for item in listing:
                    con.execute(
                        "INSERT OR IGNORE INTO sr_raw_kap_listing VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (int(item["disclosureIndex"]), KAP_FRAME_VERSION, seq,
                         item.get("disclosureType"), item.get("disclosureClass"),
                         str(item.get("companyId")), item.get("title"),
                         store.dumps(item), SOURCE, store.now_iso()),
                    )
                con.execute(
                    "UPDATE sr_kap_frame SET status='listed', listed=?, targets=?, "
                    "updated_at=? WHERE frame_version=? AND block_seq=?",
                    (len(listing), sum(1 for i in listing if is_target(i, listed)),
                     store.now_iso(), KAP_FRAME_VERSION, seq),
                )

        with store.connect(db_path) as con:
            rows = [dict(r) for r in con.execute(
                "SELECT l.disclosure_index, l.payload_json FROM sr_raw_kap_listing l "
                "LEFT JOIN sr_raw_kap_detail d USING (disclosure_index) "
                "WHERE l.frame_version=? AND l.block_seq=? AND d.disclosure_index IS NULL "
                "ORDER BY l.disclosure_index", (KAP_FRAME_VERSION, seq))]
        pending = [r for r in rows if is_target(json.loads(r["payload_json"]), listed)]

        out_of_time = False
        for row in pending:
            if time.time() >= deadline:
                out_of_time = True
                break
            index = row["disclosure_index"]
            try:
                detail = api_get(f"/disclosureDetail/{index}", {"fileType": "html"})
            except requests.HTTPError as exc:
                logger.warning("detail %s unavailable: %s", index, exc)
                detail = {"disclosureIndex": str(index), "unavailable": str(exc)[:200]}
            _store_detail(db_path, seq, index, detail)
        if out_of_time:
            break

        with store.connect(db_path) as con:
            con.execute(
                "UPDATE sr_kap_frame SET status='complete', updated_at=? "
                "WHERE frame_version=? AND block_seq=?",
                (store.now_iso(), KAP_FRAME_VERSION, seq),
            )
        done_blocks += 1
        logger.info("block %s (start %s) complete: %s details", seq, start, len(pending))
        if max_blocks and done_blocks >= max_blocks:
            break

    return progress(db_path)


def progress(db_path=None) -> Dict[str, Any]:
    with store.connect(db_path) as con:
        rows = con.execute(
            "SELECT status, COUNT(*) n, COALESCE(SUM(targets),0) t, "
            "COALESCE(SUM(details_done),0) d FROM sr_kap_frame "
            "WHERE frame_version=? GROUP BY status", (KAP_FRAME_VERSION,),
        ).fetchall()
    by_status = {r["status"]: {"blocks": r["n"], "targets": r["t"], "details": r["d"]}
                 for r in rows}
    complete = by_status.get("complete", {}).get("blocks", 0)
    return {
        "frame_version": KAP_FRAME_VERSION,
        "blocks_total": KAP_FRAME_BLOCKS,
        "blocks_complete": complete,
        "frame_complete": complete == KAP_FRAME_BLOCKS,
        "by_status": by_status,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="KAP block-sample ingestion")
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--db")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(REPOSITORY_ROOT / "stock_research_ingest.log",
                                      encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    if args.command == "run":
        result = run(args.db, max_hours=args.max_hours, max_blocks=args.max_blocks)
    else:
        store.init_db(args.db)
        result = progress(args.db)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
