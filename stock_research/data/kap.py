"""Resumable KAP census ingestion from the MKK API Portal.

The listing endpoint returns up to 50 disclosures per call, in ascending
disclosure number from the number asked for, with no timestamp; the time,
subject and body need one detail call each, at 6 calls a minute.

Protocol v3 takes a **census** of the frame's range: every material-event
disclosure of a listed company between ``KAP_FRAME_FIRST_INDEX`` and
``KAP_FRAME_LAST_INDEX``. The range is read page by page. A page starts where
the one before it ended, so the pages cannot be laid out in advance: the
frame table grows by one row per page until a page comes back short or
reaches the end of the range. The range and the membership rule are constants,
fixed without reference to any price.

Nothing stored under an earlier frame is rewritten. A listing row or a detail
that already exists is kept as it is and becomes a member of the census
through ``sr_kap_frame_items``, which records, for this frame, which
disclosure numbers each page held and which of them are targets.

Two kinds of completeness are kept apart:

* **Ingestion** is complete when the last page has been read and every target
  disclosure has been asked for.
* **Details** are complete when every one of those requests was answered.

A disclosure the gateway refuses is logged in ``sr_kap_detail_failures`` and
asked for again on later days, once a day, up to ``KAP_MAX_DETAIL_ATTEMPTS``
times. It never gets a row in ``sr_raw_kap_detail``, and its page is
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
from stock_research.calendar import parse_kap_time
from stock_research.config import (
    KAP_EARLIER_FRAMES, KAP_FRAME_FIRST_INDEX, KAP_FRAME_LAST_INDEX, KAP_FRAME_VERSION,
    KAP_LISTED_MEMBER_MARK, KAP_LISTING_PAGE, KAP_LISTING_TYPES, KAP_MAX_DETAIL_ATTEMPTS,
    KAP_MAX_UNAVAILABLE_SHARE, KAP_MIN_ATTEMPT_DAYS, KAP_RATE_LIMIT_WAIT_SECONDS,
    KAP_TARGET_CLASSES, KAP_THROTTLE_MAX_SECONDS, KAP_THROTTLE_RECOVERY_CALLS,
    KAP_THROTTLE_SECONDS, KAP_THROTTLE_STEP_SECONDS, REPOSITORY_ROOT,
)

logger = logging.getLogger("stock_research.kap")
SOURCE = "mkk-api-portal-vyk-dev"
MAX_DETAIL_ATTEMPTS = KAP_MAX_DETAIL_ATTEMPTS

STATUS_PENDING = "pending"
STATUS_LISTED = "listed"
STATUS_COMPLETE = "complete"                     # every target detail stored
STATUS_INCOMPLETE = "incomplete_details"         # every target asked for, some refused
ATTEMPTED = (STATUS_COMPLETE, STATUS_INCOMPLETE)
_END_KEY = f"kap_frame_last_page:{KAP_FRAME_VERSION}"


def _base_url() -> str:
    from config import KAP_BASE_URL      # loads .env as a side effect
    return KAP_BASE_URL


def _auth():
    key, secret = os.environ.get("MKK_API_KEY", ""), os.environ.get("MKK_API_SECRET", "")
    if not (key and secret):
        raise RuntimeError("MKK_API_KEY / MKK_API_SECRET not set (env or .env)")
    return key, secret


class Pace:
    """Seconds to leave between calls, adjusted to what the gateway allows.

    A refused call (HTTP 429) costs far more than a slightly wider spacing,
    so each refusal widens the spacing by a step, up to a ceiling, and a run
    of accepted calls narrows it by a second, down to the plan's own spacing.
    """

    def __init__(self) -> None:
        self.spacing = float(KAP_THROTTLE_SECONDS)
        self.accepted_in_a_row = 0
        self.refusals = 0

    def accepted(self) -> float:
        self.accepted_in_a_row += 1
        if self.accepted_in_a_row >= KAP_THROTTLE_RECOVERY_CALLS:
            self.accepted_in_a_row = 0
            self.spacing = max(float(KAP_THROTTLE_SECONDS), self.spacing - 1.0)
        return self.spacing

    def refused(self, retry_after: Optional[str] = None) -> float:
        """Seconds to wait before asking again."""

        self.refusals += 1
        self.accepted_in_a_row = 0
        self.spacing = min(float(KAP_THROTTLE_MAX_SECONDS),
                           self.spacing + KAP_THROTTLE_STEP_SECONDS)
        try:
            named = float(retry_after) if retry_after is not None else None
        except (TypeError, ValueError):
            named = None
        if named is None or not (0 < named <= 600):
            return float(KAP_RATE_LIMIT_WAIT_SECONDS)
        return max(named, self.spacing)


PACE = Pace()


def api_get(path: str, params: Optional[dict] = None, *, attempts: int = 5) -> Any:
    """GET at the pace the gateway allows. Sleeps after every call, accepted
    or not. A rate-limit refusal does not use up an attempt: it is the
    gateway asking for patience, not a failure of the request."""

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
            logger.info("rate limited: waiting %.0f s, spacing now %.0f s", wait, PACE.spacing)
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


# -- The frame -----------------------------------------------------------------
def ensure_frame(db_path=None) -> None:
    """The first page of the census. Later pages are added as each is read."""

    store.init_db(db_path)
    with store.connect(db_path) as con:
        con.execute(
            "INSERT OR IGNORE INTO sr_kap_frame (frame_version, block_seq, block_start, "
            "process_order) VALUES (?,?,?,?)", (KAP_FRAME_VERSION, 0, KAP_FRAME_FIRST_INDEX, 0))


def record_page(db_path, seq: int, items: List[Dict[str, Any]], listed: set) -> Dict[str, Any]:
    """Store one listing page: its raw rows, its members, and the page after it.

    A raw listing row that already exists (from an earlier frame) is left as
    it is. Membership of this frame is recorded separately, so nothing about
    the earlier sample is rewritten. Disclosures past the end of the range are
    not members.
    """

    members = [item for item in items if int(item["disclosureIndex"]) <= KAP_FRAME_LAST_INDEX]
    numbers = [int(item["disclosureIndex"]) for item in items]
    last_page = len(items) < KAP_LISTING_PAGE or max(numbers, default=0) >= KAP_FRAME_LAST_INDEX
    now = store.now_iso()
    with store.connect(db_path) as con:
        for item in members:
            number = int(item["disclosureIndex"])
            con.execute(
                "INSERT OR IGNORE INTO sr_raw_kap_listing VALUES (?,?,?,?,?,?,?,?,?,?)",
                (number, KAP_FRAME_VERSION, seq, item.get("disclosureType"),
                 item.get("disclosureClass"), str(item.get("companyId")), item.get("title"),
                 store.dumps(item), SOURCE, now))
            con.execute(
                "INSERT OR IGNORE INTO sr_kap_frame_items VALUES (?,?,?,?,?)",
                (KAP_FRAME_VERSION, number, seq, int(is_target(item, listed)), now))
        con.execute(
            "UPDATE sr_kap_frame SET status=?, listed=?, targets=?, updated_at=? "
            "WHERE frame_version=? AND block_seq=?",
            (STATUS_LISTED, len(members), sum(1 for i in members if is_target(i, listed)), now,
             KAP_FRAME_VERSION, seq))
        if last_page:
            con.execute("INSERT OR REPLACE INTO sr_meta (key, value) VALUES (?, ?)",
                        (_END_KEY, str(seq)))
        else:
            con.execute(
                "INSERT OR IGNORE INTO sr_kap_frame (frame_version, block_seq, block_start, "
                "process_order) VALUES (?,?,?,?)",
                (KAP_FRAME_VERSION, seq + 1, max(numbers) + 1, seq + 1))
    return {"members": len(members), "last_page": last_page}


def last_page(db_path=None) -> Optional[int]:
    """Number of the final page, once the listing has reached the end."""

    with store.connect(db_path) as con:
        row = con.execute("SELECT value FROM sr_meta WHERE key = ?", (_END_KEY,)).fetchone()
    return int(row[0]) if row else None


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


def _target_state(db_path, seq: Optional[int] = None) -> Dict[str, List[int]]:
    """Targets, of one page or of the whole frame, by what is known about
    their detail.

    ``stored``: a real detail, whichever frame fetched it. ``never_stored``:
    no row yet, so it can be asked for. ``placeholder``: a row with no
    timestamp, written by the ingester as it was before 2026-10-11 when a
    request was refused; the raw table is append-only, so that disclosure
    stays unavailable.
    """

    where = "WHERE i.frame_version=? AND i.is_target=1"
    params: List[Any] = [KAP_FRAME_VERSION]
    if seq is not None:
        where += " AND i.block_seq=?"
        params.append(seq)
    with store.connect(db_path) as con:
        rows = con.execute(
            "SELECT i.disclosure_index, d.disclosure_index AS has_row, d.published_raw "
            f"FROM sr_kap_frame_items i LEFT JOIN sr_raw_kap_detail d USING (disclosure_index) "
            f"{where} ORDER BY i.disclosure_index", params).fetchall()
    state: Dict[str, List[int]] = {"stored": [], "never_stored": [], "placeholder": []}
    for row in rows:
        if row["has_row"] is None:
            state["never_stored"].append(row["disclosure_index"])
        elif row["published_raw"] is None:
            state["placeholder"].append(row["disclosure_index"])
        else:
            state["stored"].append(row["disclosure_index"])
    return state


def _attempts(db_path, indices: List[int]) -> Dict[int, Dict[str, Any]]:
    """Refusals of each disclosure: how many, and on which days."""

    out: Dict[int, Dict[str, Any]] = {}
    with store.connect(db_path) as con:
        for start in range(0, len(indices), 500):
            part = indices[start:start + 500]
            marks = ",".join("?" * len(part))
            for row in con.execute(
                    f"SELECT disclosure_index, attempted_at FROM sr_kap_detail_failures "
                    f"WHERE disclosure_index IN ({marks})", part):
                entry = out.setdefault(row[0], {"count": 0, "days": set()})
                entry["count"] += 1
                entry["days"].add(str(row[1])[:10])
    return out


def _may_ask(attempts: Dict[int, Dict[str, Any]], index: int, today: str) -> bool:
    """One request a day, and no more than the registered number in all."""

    seen = attempts.get(index)
    return seen is None or (seen["count"] < MAX_DETAIL_ATTEMPTS and today not in seen["days"])


def _next_block(db_path, visited: set):
    """The next page with work: unfinished ones first, in order, then pages
    with a refused detail that may be asked for again today."""

    with store.connect(db_path) as con:
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM sr_kap_frame WHERE frame_version=? AND status!=? "
            "ORDER BY (status=?) , process_order",
            (KAP_FRAME_VERSION, STATUS_COMPLETE, STATUS_INCOMPLETE))]
    today = store.now_iso()[:10]
    for block in rows:
        if block["block_seq"] in visited:
            continue
        if block["status"] != STATUS_INCOMPLETE:
            return block
        missing = _target_state(db_path, block["block_seq"])["never_stored"]
        attempts = _attempts(db_path, missing)
        if any(_may_ask(attempts, i, today) for i in missing):
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
        block = _next_block(db_path, visited)
        if block is None:
            break
        seq, start = block["block_seq"], block["block_start"]
        visited.add(seq)

        if block["status"] == STATUS_PENDING:
            listing = api_get("/disclosures", {
                "disclosureIndex": str(start),
                "disclosureTypes": ",".join(KAP_LISTING_TYPES),
            })
            record_page(db_path, seq, listing, listed)

        missing = _target_state(db_path, seq)["never_stored"]
        attempts = _attempts(db_path, missing)
        today = store.now_iso()[:10]
        out_of_time = False
        for index in missing:
            if not _may_ask(attempts, index, today):
                continue
            if time.time() >= deadline:
                out_of_time = True
                break
            try:
                detail = api_get(f"/disclosureDetail/{index}", {"fileType": "html"})
            except requests.HTTPError as exc:
                # The gateway answered and refused. Logged as a gap; the page
                # cannot be complete while it stands.
                logger.warning("detail %s unavailable: %s", index, exc)
                _record_failure(db_path, seq, index, exc)
                continue
            _store_detail(db_path, seq, index, detail)
        if out_of_time:
            break

        state = _target_state(db_path, seq)
        gaps = len(state["never_stored"]) + len(state["placeholder"])
        status = STATUS_COMPLETE if gaps == 0 else STATUS_INCOMPLETE
        with store.connect(db_path) as con:
            con.execute(
                "UPDATE sr_kap_frame SET status=?, updated_at=? "
                "WHERE frame_version=? AND block_seq=?",
                (status, store.now_iso(), KAP_FRAME_VERSION, seq),
            )
        done_blocks += 1
        logger.info("page %s (start %s) %s: %s stored, %s unavailable", seq, start, status,
                    len(state["stored"]), gaps)
        if max_blocks and done_blocks >= max_blocks:
            break

    result = progress(db_path)
    log_event(db_path, run_id, "finish", result, **identity)
    return result


# -- Completeness ---------------------------------------------------------------
def _attempted_pages(db_path) -> List[int]:
    with store.connect(db_path) as con:
        return [r[0] for r in con.execute(
            "SELECT block_seq FROM sr_kap_frame WHERE frame_version=? AND status IN (?,?)",
            (KAP_FRAME_VERSION, *ATTEMPTED))]


def unavailable_details(db_path=None) -> List[int]:
    """Target disclosures of attempted pages that have no real detail."""

    missing: List[int] = []
    for seq in _attempted_pages(db_path):
        state = _target_state(db_path, seq)
        missing += state["never_stored"] + state["placeholder"]
    return sorted(missing)


def verify_details(db_path=None) -> Dict[str, Any]:
    """Check every stored target detail, not just count it.

    A detail whose timestamp does not parse has no day 0 and would drop out of
    the study without a trace; one with no subject cannot be classified. Both
    are listed by number, and the run refuses while either list is non-empty.
    """

    with store.connect(db_path) as con:
        rows = con.execute(
            "SELECT d.disclosure_index, d.published_raw, d.subject_tr, d.payload_sha256 "
            "FROM sr_kap_frame_items i JOIN sr_raw_kap_detail d USING (disclosure_index) "
            "WHERE i.frame_version=? AND i.is_target=1 AND d.published_raw IS NOT NULL",
            (KAP_FRAME_VERSION,)).fetchall()
    bad_time = sorted(r[0] for r in rows if parse_kap_time(r[1]) is None)
    no_subject = sorted(r[0] for r in rows if not (r[2] or "").strip())
    no_hash = sorted(r[0] for r in rows if not re.fullmatch(r"[0-9a-f]{64}", r[3] or ""))
    return {"details_checked": len(rows), "timestamp_unparseable": bad_time,
            "subject_missing": no_subject, "payload_hash_missing": no_hash,
            "verified": not (bad_time or no_subject or no_hash)}


def progress(db_path=None) -> Dict[str, Any]:
    """Where the census stands, with ingestion and detail completeness apart.

    ``frame_complete`` is the gate the analysis uses: the listing has reached
    the end of the range, every page has been attempted, every stored detail
    verifies, and any disclosure still missing was refused on the registered
    number of separate days and is within the registered share.
    """

    with store.connect(db_path) as con:
        frame = [dict(r) for r in con.execute(
            "SELECT block_seq, block_start, status FROM sr_kap_frame WHERE frame_version=? "
            "ORDER BY block_seq", (KAP_FRAME_VERSION,))]
        reused = con.execute(
            "SELECT COUNT(*) FROM sr_kap_frame_items i JOIN sr_raw_kap_listing l "
            "USING (disclosure_index) WHERE i.frame_version=? AND l.frame_version!=?",
            (KAP_FRAME_VERSION, KAP_FRAME_VERSION)).fetchone()[0]
    end = last_page(db_path)
    by_status: Dict[str, int] = {}
    for block in frame:
        by_status[block["status"]] = by_status.get(block["status"], 0) + 1
    attempted = sum(by_status.get(s, 0) for s in ATTEMPTED)
    listing_complete = end is not None
    ingestion_complete = listing_complete and attempted == len(frame) == end + 1

    state = _target_state(db_path)
    attempted_pages = set(_attempted_pages(db_path))
    with store.connect(db_path) as con:
        page_of = {r[0]: r[1] for r in con.execute(
            "SELECT disclosure_index, block_seq FROM sr_kap_frame_items WHERE frame_version=? "
            "AND is_target=1", (KAP_FRAME_VERSION,))}
    asked = [i for i in state["never_stored"] if page_of[i] in attempted_pages]
    attempts = _attempts(db_path, asked)
    exhausted = [i for i in asked
                 if attempts.get(i, {"count": 0})["count"] >= MAX_DETAIL_ATTEMPTS
                 and len(attempts[i]["days"]) >= KAP_MIN_ATTEMPT_DAYS]
    unavailable = len(asked) + len(state["placeholder"])
    accepted = len(exhausted) + len(state["placeholder"])
    targets = len(state["stored"]) + len(state["never_stored"]) + len(state["placeholder"])
    targets_attempted = len(state["stored"]) + unavailable
    share = (unavailable / targets_attempted) if targets_attempted else 0.0
    within = unavailable == accepted and share <= KAP_MAX_UNAVAILABLE_SHARE
    covered = max((b["block_start"] for b in frame), default=KAP_FRAME_FIRST_INDEX)
    span = KAP_FRAME_LAST_INDEX - KAP_FRAME_FIRST_INDEX
    return {
        "frame_version": KAP_FRAME_VERSION,
        "kind": "census",
        "range": [KAP_FRAME_FIRST_INDEX, KAP_FRAME_LAST_INDEX],
        "range_listed_share": 1.0 if listing_complete else max(
            0.0, min(1.0, (covered - KAP_FRAME_FIRST_INDEX) / span)),
        "listing_complete": listing_complete,
        "blocks_total": (end + 1) if listing_complete else None,
        "blocks_known": len(frame),
        "blocks_complete": by_status.get(STATUS_COMPLETE, 0),
        "blocks_attempted": attempted,
        "ingestion_complete": ingestion_complete,
        "targets": targets,
        "targets_reused_from_earlier_frames": int(reused),
        "details_stored": len(state["stored"]),
        "details_unavailable": unavailable,
        "details_awaiting_another_attempt": unavailable - accepted,
        "unavailable_share": share,
        "max_unavailable_share": KAP_MAX_UNAVAILABLE_SHARE,
        "attempts_required": MAX_DETAIL_ATTEMPTS,
        "attempt_days_required": KAP_MIN_ATTEMPT_DAYS,
        "within_tolerance": within,
        "details_complete": ingestion_complete and unavailable == 0,
        "frame_complete": ingestion_complete and within,
        "earlier_frames": list(KAP_EARLIER_FRAMES),
        "by_status": by_status,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="KAP census ingestion")
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
