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

Two kinds of completeness are kept apart:

* **Ingestion** is complete when every anchor has been listed and every target
  disclosure has been asked for.
* **Details** are complete when every one of those requests was answered.

A disclosure the gateway refuses is logged in ``sr_kap_detail_failures`` and
asked for again on later runs, up to ``KAP_MAX_DETAIL_ATTEMPTS`` times. It
never gets a row in ``sr_raw_kap_detail``, and its anchor is
``incomplete_details``, not ``complete``.

This module decides which disclosures are in the sample (``is_target``) and
what text the taxonomy and the insider parser read (``body_text``), so it is
part of the protocol fingerprint. Each run logs the hash of this file.

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
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from stock_research import store
from stock_research.config import (
    KAP_FRAME_BLOCKS, KAP_FRAME_FIRST_INDEX, KAP_FRAME_LAST_INDEX,
    KAP_FRAME_STRIDE, KAP_FRAME_VERSION, KAP_LISTED_MEMBER_MARK,
    KAP_LISTING_TYPES, KAP_MAX_DETAIL_ATTEMPTS, KAP_MAX_UNAVAILABLE_SHARE,
    KAP_TARGET_CLASSES, KAP_THROTTLE_SECONDS, REPOSITORY_ROOT,
)

logger = logging.getLogger("stock_research.kap")
SOURCE = "mkk-api-portal-vyk-dev"
MAX_DETAIL_ATTEMPTS = KAP_MAX_DETAIL_ATTEMPTS

STATUS_PENDING = "pending"
STATUS_LISTED = "listed"
STATUS_COMPLETE = "complete"                     # every target detail stored
STATUS_INCOMPLETE = "incomplete_details"         # every target asked for, some refused
ATTEMPTED = (STATUS_COMPLETE, STATUS_INCOMPLETE)


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


#: Pacing (added 2026-10-11 for the last anchors of the v2 sample, after the
#: gateway began refusing about one call in four at the fixed spacing). It
#: decides how fast the sample arrives, never what is in it: the frame,
#: ``is_target``, ``body_text`` and ``detail_row`` are unchanged.
THROTTLE_MAX_SECONDS = 30
THROTTLE_STEP_SECONDS = 2
THROTTLE_RECOVERY_CALLS = 30
RATE_LIMIT_WAIT_SECONDS = 65


class Pace:
    """Seconds to leave between calls, adjusted to what the gateway allows."""

    def __init__(self) -> None:
        self.spacing = float(KAP_THROTTLE_SECONDS)
        self.accepted_in_a_row = 0
        self.refusals = 0

    def accepted(self) -> float:
        self.accepted_in_a_row += 1
        if self.accepted_in_a_row >= THROTTLE_RECOVERY_CALLS:
            self.accepted_in_a_row = 0
            self.spacing = max(float(KAP_THROTTLE_SECONDS), self.spacing - 1.0)
        return self.spacing

    def refused(self, retry_after: Optional[str] = None) -> float:
        self.refusals += 1
        self.accepted_in_a_row = 0
        self.spacing = min(float(THROTTLE_MAX_SECONDS), self.spacing + THROTTLE_STEP_SECONDS)
        try:
            named = float(retry_after) if retry_after is not None else None
        except (TypeError, ValueError):
            named = None
        if named is None or not (0 < named <= 600):
            return float(RATE_LIMIT_WAIT_SECONDS)
        return max(named, self.spacing)


PACE = Pace()


def api_get(path: str, params: Optional[dict] = None, *, attempts: int = 5) -> Any:
    """GET at the pace the gateway allows. Sleeps after every call. A
    rate-limit refusal does not use up an attempt."""

    last_error = None
    tries = refusals = 0
    while tries < attempts:
        tries += 1
        try:
            response = requests.get(_base_url() + path, params=params,
                                    auth=_auth(), timeout=90)
        except requests.RequestException as exc:
            last_error = repr(exc)
            time.sleep(KAP_THROTTLE_SECONDS * 3)
            continue
        if response.status_code == 429:
            last_error = "429"
            refusals += 1
            wait = PACE.refused(response.headers.get("Retry-After"))
            logger.info("rate limited (Retry-After=%s): waiting %.0f s, spacing now %.0f s",
                        response.headers.get("Retry-After"), wait, PACE.spacing)
            time.sleep(wait)
            if refusals <= 20:
                tries -= 1
            continue
        time.sleep(PACE.accepted())
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


def ingester_sha256() -> str:
    """SHA-256 of this file with line endings normalised: the same value the
    protocol fingerprint records for ``data/kap.py``."""

    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _code_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPOSITORY_ROOT),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                   # pragma: no cover
        return None


def log_event(db_path, run_id: str, event: str, detail: Optional[Dict[str, Any]] = None,
              *, ingester_hash: Optional[str] = None, code_commit: Optional[str] = None,
              at: Optional[str] = None) -> None:
    with store.connect(db_path) as con:
        con.execute(
            "INSERT INTO sr_ingest_log (run_id, event, at, frame_version, ingester_sha256, "
            "code_commit, detail_json) VALUES (?,?,?,?,?,?,?)",
            (run_id, event, at or store.now_iso(), KAP_FRAME_VERSION,
             ingester_hash or ingester_sha256(),
             code_commit if code_commit is not None else _code_commit(),
             store.dumps(detail) if detail is not None else None),
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


def _record_failure(db_path, seq: int, index: int, exc: requests.HTTPError) -> None:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    with store.connect(db_path) as con:
        con.execute(
            "INSERT INTO sr_kap_detail_failures (disclosure_index, frame_version, block_seq, "
            "http_status, error, source, attempted_at) VALUES (?,?,?,?,?,?,?)",
            (index, KAP_FRAME_VERSION, seq, status, str(exc)[:300], SOURCE, store.now_iso()),
        )


def _target_state(db_path, seq: int, listed: set) -> Dict[str, List[int]]:
    """Targets of one anchor by what is known about their detail.

    ``stored``: a real detail. ``never_stored``: no row yet, so it can be asked
    for. ``placeholder``: a row with no timestamp, written by the ingester as
    it was before 2026-10-11 when a request was refused; the raw table is
    append-only, so that disclosure stays unavailable.
    """

    with store.connect(db_path) as con:
        rows = [dict(r) for r in con.execute(
            "SELECT l.disclosure_index, l.payload_json, d.disclosure_index AS has_row, "
            "d.published_raw FROM sr_raw_kap_listing l "
            "LEFT JOIN sr_raw_kap_detail d USING (disclosure_index) "
            "WHERE l.frame_version=? AND l.block_seq=? ORDER BY l.disclosure_index",
            (KAP_FRAME_VERSION, seq))]
    state: Dict[str, List[int]] = {"stored": [], "never_stored": [], "placeholder": []}
    for row in rows:
        if not is_target(json.loads(row["payload_json"]), listed):
            continue
        if row["has_row"] is None:
            state["never_stored"].append(row["disclosure_index"])
        elif row["published_raw"] is None:
            state["placeholder"].append(row["disclosure_index"])
        else:
            state["stored"].append(row["disclosure_index"])
    return state


def _attempts(db_path, indices: List[int]) -> Dict[int, int]:
    if not indices:
        return {}
    marks = ",".join("?" * len(indices))
    with store.connect(db_path) as con:
        return {r[0]: r[1] for r in con.execute(
            f"SELECT disclosure_index, COUNT(*) FROM sr_kap_detail_failures "
            f"WHERE disclosure_index IN ({marks}) GROUP BY disclosure_index", indices)}


def _next_block(db_path, listed: set, visited: set):
    """The next anchor with work: unfinished ones first, then anchors with a
    refused detail that may still be asked for again."""

    with store.connect(db_path) as con:
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM sr_kap_frame WHERE frame_version=? AND status!=? "
            "ORDER BY (status=?) , process_order",
            (KAP_FRAME_VERSION, STATUS_COMPLETE, STATUS_INCOMPLETE))]
    for block in rows:
        if block["block_seq"] in visited:
            continue
        if block["status"] != STATUS_INCOMPLETE:
            return block
        missing = _target_state(db_path, block["block_seq"], listed)["never_stored"]
        attempts = _attempts(db_path, missing)
        if any(attempts.get(i, 0) < MAX_DETAIL_ATTEMPTS for i in missing):
            return block
    return None


def run(db_path=None, *, max_hours: float = 8.0,
        max_blocks: Optional[int] = None) -> Dict[str, Any]:
    ensure_frame(db_path)
    snapshot = ensure_members(db_path)
    listed = _listed_member_ids(db_path, snapshot)
    deadline = time.time() + max_hours * 3600
    run_id = uuid.uuid4().hex
    # Taken once, at the start: the code that is running, not whatever is on
    # disk when the run ends.
    identity = {"ingester_hash": ingester_sha256(), "code_commit": _code_commit() or ""}
    log_event(db_path, run_id, "start", {"max_hours": max_hours, "max_blocks": max_blocks,
                                         "members_snapshot": snapshot}, **identity)
    done_blocks, visited = 0, set()

    while time.time() < deadline:
        block = _next_block(db_path, listed, visited)
        if block is None:
            break
        seq, start = block["block_seq"], block["block_start"]
        visited.add(seq)

        if block["status"] == STATUS_PENDING:
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
                    "UPDATE sr_kap_frame SET status=?, listed=?, targets=?, "
                    "updated_at=? WHERE frame_version=? AND block_seq=?",
                    (STATUS_LISTED, len(listing), sum(1 for i in listing if is_target(i, listed)),
                     store.now_iso(), KAP_FRAME_VERSION, seq),
                )

        missing = _target_state(db_path, seq, listed)["never_stored"]
        attempts = _attempts(db_path, missing)
        out_of_time = False
        for index in missing:
            if attempts.get(index, 0) >= MAX_DETAIL_ATTEMPTS:
                continue
            if time.time() >= deadline:
                out_of_time = True
                break
            try:
                detail = api_get(f"/disclosureDetail/{index}", {"fileType": "html"})
            except requests.HTTPError as exc:
                # The gateway answered and refused. Logged as a gap; the anchor
                # cannot be complete while it stands.
                logger.warning("detail %s unavailable: %s", index, exc)
                _record_failure(db_path, seq, index, exc)
                continue
            _store_detail(db_path, seq, index, detail)
        if out_of_time:
            break

        state = _target_state(db_path, seq, listed)
        gaps = len(state["never_stored"]) + len(state["placeholder"])
        status = STATUS_COMPLETE if gaps == 0 else STATUS_INCOMPLETE
        with store.connect(db_path) as con:
            con.execute(
                "UPDATE sr_kap_frame SET status=?, updated_at=? "
                "WHERE frame_version=? AND block_seq=?",
                (status, store.now_iso(), KAP_FRAME_VERSION, seq),
            )
        done_blocks += 1
        logger.info("block %s (start %s) %s: %s stored, %s unavailable", seq, start, status,
                    len(state["stored"]), gaps)
        if max_blocks and done_blocks >= max_blocks:
            break

    result = progress(db_path)
    log_event(db_path, run_id, "finish", result, **identity)
    return result


def unavailable_details(db_path=None) -> List[int]:
    """Target disclosures of attempted anchors that have no real detail."""

    with store.connect(db_path) as con:
        snapshot = con.execute(
            "SELECT snapshot_id FROM sr_raw_members ORDER BY retrieved_at DESC LIMIT 1").fetchone()
        blocks = [r[0] for r in con.execute(
            "SELECT block_seq FROM sr_kap_frame WHERE frame_version=? AND status IN (?,?)",
            (KAP_FRAME_VERSION, *ATTEMPTED))]
    if snapshot is None:
        return []
    listed = _listed_member_ids(db_path, snapshot[0])
    missing: List[int] = []
    for seq in blocks:
        state = _target_state(db_path, seq, listed)
        missing += state["never_stored"] + state["placeholder"]
    return sorted(missing)


def progress(db_path=None) -> Dict[str, Any]:
    """Where the sample stands, with ingestion and detail completeness apart.

    ``frame_complete`` is the gate the analysis uses: every anchor attempted
    **and** the share of unavailable details within the registered tolerance.
    """

    with store.connect(db_path) as con:
        frame = [dict(r) for r in con.execute(
            "SELECT block_seq, status, COALESCE(targets, 0) AS targets FROM sr_kap_frame "
            "WHERE frame_version=?", (KAP_FRAME_VERSION,))]
        stored = {r[0]: r[1] for r in con.execute(
            "SELECT l.block_seq, COUNT(*) FROM sr_raw_kap_detail d "
            "JOIN sr_raw_kap_listing l USING (disclosure_index) "
            "WHERE l.frame_version=? AND d.published_raw IS NOT NULL GROUP BY l.block_seq",
            (KAP_FRAME_VERSION,))}
    by_status: Dict[str, Dict[str, int]] = {}
    targets = details_stored = unavailable = 0
    for block in frame:
        have = stored.get(block["block_seq"], 0)
        entry = by_status.setdefault(block["status"], {"blocks": 0, "targets": 0, "details": 0})
        entry["blocks"] += 1
        entry["targets"] += block["targets"]
        entry["details"] += have
        details_stored += have
        if block["status"] in ATTEMPTED:
            targets += block["targets"]
            unavailable += max(0, block["targets"] - have)
    complete = by_status.get(STATUS_COMPLETE, {}).get("blocks", 0)
    attempted = complete + by_status.get(STATUS_INCOMPLETE, {}).get("blocks", 0)
    ingestion_complete = bool(frame) and attempted == len(frame) == KAP_FRAME_BLOCKS
    share = (unavailable / targets) if targets else 0.0
    within = share <= KAP_MAX_UNAVAILABLE_SHARE
    return {
        "frame_version": KAP_FRAME_VERSION,
        "blocks_total": KAP_FRAME_BLOCKS,
        "blocks_complete": complete,
        "blocks_attempted": attempted,
        "ingestion_complete": ingestion_complete,
        "targets": targets,
        "details_stored": details_stored,
        "details_unavailable": unavailable,
        "unavailable_share": share,
        "max_unavailable_share": KAP_MAX_UNAVAILABLE_SHARE,
        "within_tolerance": within,
        "details_complete": ingestion_complete and unavailable == 0,
        "frame_complete": ingestion_complete and within,
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
