"""
KAP Tier-A disclosure ingestion via the MKK API Portal (migration Phase 3).

Creates events directly (no headline) from KAP material-event (ODA) and
financial-report (FR) disclosures:
    source='kap', source_tier='A', credibility=1.0,
    external_id='kap:<disclosureIndex>' (dedup),
    title = '<company>: <subject/summary>',
    relatedStocks -> event_entities (ticker rows) — Phase 6 head start.

Incremental cursor: kv_state['kap_cursor'] holds the last processed
disclosureIndex; each run lists from there. First run seeds the cursor from
/lastDisclosureIndex WITHOUT backfilling (we only want disclosures published
from now on).

API notes (docs/kap_api_notes.md): HTTP Basic auth (MKK_API_KEY/SECRET in
.env), free plan throttled to 6 calls/min — we sleep KAP_THROTTLE_SECONDS
between calls and cap detail fetches per run.

⚠ The dev gateway (apigwdev) serves a HISTORICAL SAMPLE dataset (late 2023).
KAP_ENABLED stays False until production access; use --dry-run to validate.
"""

import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import requests

import database as db
from config import (
    DB_PATH,
    KAP_BASE_URL,
    KAP_DISCLOSURE_TYPES,
    KAP_MAX_DETAILS_PER_RUN,
    KAP_THROTTLE_SECONDS,
)
from trading_calendar import signal_date as _signal_date

logger = logging.getLogger(__name__)

_CURSOR_KEY = "kap_cursor"


# -- Small helpers ----------------------------------------------------------------

def parse_kap_time(raw: str) -> Tuple[Optional[str], Optional[int]]:
    """'29.12.2023 18:23:08' -> ('2023-12-29', 18). Returns (None, None) on junk."""
    try:
        dt = datetime.strptime(raw.strip(), "%d.%m.%Y %H:%M:%S")
        return dt.date().isoformat(), dt.hour
    except (ValueError, AttributeError):
        return None, None


def build_title(detail: dict) -> str:
    """Human-readable event title: company + subject/summary (Turkish)."""
    company = detail.get("senderTitle") or detail.get("behalfSenderTitle") or "?"
    subject = (detail.get("subject") or {}).get("tr") or ""
    summary = (detail.get("summary") or {}).get("tr") or ""
    body = summary or subject or detail.get("disclosureType", "")
    return f"{company}: {body}"[:300]


def _auth() -> Tuple[str, str]:
    key = os.environ.get("MKK_API_KEY", "")
    secret = os.environ.get("MKK_API_SECRET", "")
    if not (key and secret):
        raise RuntimeError("MKK_API_KEY / MKK_API_SECRET not set (env or .env)")
    return key, secret


def _get(path: str, params: Optional[dict] = None) -> requests.Response:
    """GET with throttle + one retry on transient failures."""
    for attempt in range(2):
        try:
            resp = requests.get(f"{KAP_BASE_URL}{path}", params=params,
                                auth=_auth(), timeout=60)
        except requests.RequestException as exc:
            logger.warning("KAP: %s — retrying once", type(exc).__name__)
            time.sleep(KAP_THROTTLE_SECONDS)
            continue
        if resp.status_code == 429:
            logger.warning("KAP: throttled — waiting 60s")
            time.sleep(60)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"KAP: request failed twice: {path}")


def _get_cursor(db_path: str) -> Optional[int]:
    with db._conn(db_path) as con:
        row = con.execute("SELECT value FROM kv_state WHERE key=?", (_CURSOR_KEY,)).fetchone()
    return int(row[0]) if row else None


def _set_cursor(value: int, db_path: str) -> None:
    with db._conn(db_path) as con:
        con.execute("INSERT OR REPLACE INTO kv_state (key, value) VALUES (?, ?)",
                    (_CURSOR_KEY, str(value)))


# -- Ingestion ----------------------------------------------------------------------

def ingest(db_path: str = DB_PATH, dry_run: bool = False,
           max_details: int = KAP_MAX_DETAILS_PER_RUN) -> dict:
    """
    Fetch new ODA/FR disclosures past the cursor and insert them as Tier-A
    events. Returns {'new_events': n, 'cursor': last_index, 'samples': [...]}.

    dry_run: fetch and parse but write NOTHING (no events, no cursor move) —
    use against the dev gateway's sample data.
    """
    cursor = _get_cursor(db_path)
    if cursor is None:
        last = int(_get("/lastDisclosureIndex").json()["lastDisclosureIndex"])
        if not dry_run:
            _set_cursor(last, db_path)
            logger.info("KAP: cursor seeded at %d — ingestion starts next run", last)
            return {"new_events": 0, "cursor": last, "samples": []}
        # dry-run with no cursor: look back a little so there is data to show
        cursor = last - 30

    time.sleep(KAP_THROTTLE_SECONDS)
    listing = _get("/disclosures", params={
        "disclosureIndex": str(cursor),
        "disclosureTypes": ",".join(KAP_DISCLOSURE_TYPES),
    }).json()
    new_items = [it for it in listing if int(it["disclosureIndex"]) > cursor]
    new_items = new_items[:max_details]

    samples, inserted, last_index = [], 0, cursor
    for item in new_items:
        idx = int(item["disclosureIndex"])
        time.sleep(KAP_THROTTLE_SECONDS)
        detail = _get(f"/disclosureDetail/{idx}", params={"fileType": "html"}).json()

        pub_date, pub_hour = parse_kap_time(detail.get("time", ""))
        if not pub_date:
            logger.warning("KAP %d: unparseable time %r — skipped", idx, detail.get("time"))
            last_index = idx
            continue

        event = {
            "external_id":  f"kap:{idx}",
            "source":       "kap",
            "source_tier":  "A",
            "published_at": pub_date,
            "signal_date":  _signal_date(pub_date, pub_hour),
            "title":        build_title(detail),
            "event_type":   f"{detail.get('disclosureType')}/{detail.get('disclosureClass')}",
            "credibility":  1.0,
            "tickers":      [s.get("code") for s in (detail.get("relatedStocks") or [])
                             if isinstance(s, dict) and s.get("code")],
        }
        samples.append(event)

        if not dry_run:
            with db._conn(db_path) as con:
                cur = con.execute(
                    """INSERT OR IGNORE INTO events
                       (external_id, source_tier, source, published_at, signal_date,
                        title, event_type, credibility, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                    (event["external_id"], "A", "kap", event["published_at"],
                     event["signal_date"], event["title"], event["event_type"], 1.0),
                )
                if cur.rowcount:
                    inserted += 1
                    event_id = con.execute(
                        "SELECT event_id FROM events WHERE external_id=?",
                        (event["external_id"],),
                    ).fetchone()[0]
                    con.executemany(
                        "INSERT OR IGNORE INTO event_entities (event_id, entity_type, entity_id) "
                        "VALUES (?, 'ticker', ?)",
                        [(event_id, t) for t in event["tickers"]],
                    )
        last_index = idx

    if not dry_run and last_index > cursor:
        _set_cursor(last_index, db_path)

    logger.info("KAP ingest: %d new event(s), cursor %d -> %d%s",
                inserted, cursor, last_index, " [DRY RUN]" if dry_run else "")
    return {"new_events": inserted, "cursor": last_index, "samples": samples}
