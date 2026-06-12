"""
Headline -> Event bridge (migration Phase 2).

The event is the unit of analysis in the new research path; headlines are raw
input. During migration every scored headline is mirrored 1:1 into the events
table (dual-write). Tier A sources (KAP, TCMB — Phase 3) will create events
natively with no headline_id.

Bridge semantics (temporary, until Phase 4 structured extraction):
    direction       = sentiment_score        (continuous, [-1, +1])
    magnitude       = |sentiment_score|
    credibility     = source-tier default (A=1.0, B=0.75, C=0.5)
    event_type      = NULL (Phase 4)
    novelty         = NULL (Phase 4)
    session_window  = NULL (Phase 5)

Idempotent: unique index on headline_id; existing events are left untouched
(re-syncing after a re-score updates sentiment fields).
"""

import logging

import database as db
from config import DB_PATH, SOURCE_TIERS, DEFAULT_SOURCE_TIER

logger = logging.getLogger(__name__)

_TIER_CREDIBILITY = {"A": 1.0, "B": 0.75, "C": 0.5}


def source_tier(source: str) -> str:
    return SOURCE_TIERS.get(source, DEFAULT_SOURCE_TIER)


def sync(db_path: str = DB_PATH) -> int:
    """Mirror scored headlines into events. Returns number of new events."""
    with db._conn(db_path) as con:
        inserted = con.execute(
            """
            INSERT INTO events
                (headline_id, source_tier, source, published_at, signal_date,
                 title, direction, magnitude, credibility,
                 sentiment_score, sentiment_label, model_version, created_at)
            SELECT h.id,
                   ?,                      -- placeholder, fixed up below
                   h.source,
                   h.published_at,
                   h.signal_date,
                   h.title,
                   h.sentiment_score,
                   ABS(h.sentiment_score),
                   ?,
                   h.sentiment_score,
                   h.sentiment_label,
                   h.model_name,
                   datetime('now')
            FROM headlines h
            WHERE h.sentiment_score IS NOT NULL
              AND h.published_at    IS NOT NULL
              AND h.signal_date     IS NOT NULL
              AND h.id NOT IN (SELECT headline_id FROM events
                               WHERE headline_id IS NOT NULL)
            """,
            (DEFAULT_SOURCE_TIER, _TIER_CREDIBILITY[DEFAULT_SOURCE_TIER]),
        ).rowcount

        # Per-source tier/credibility (single UPDATE per tier keeps this fast)
        for tier in ("A", "B", "C"):
            sources = [s for s, t in SOURCE_TIERS.items() if t == tier]
            if sources:
                ph = ",".join("?" * len(sources))
                con.execute(
                    f"UPDATE events SET source_tier=?, credibility=? "
                    f"WHERE source IN ({ph})",
                    [tier, _TIER_CREDIBILITY[tier], *sources],
                )

        # Keep sentiment fields in step with the headline scorer (re-scores)
        con.execute(
            """
            UPDATE events SET
                direction       = (SELECT sentiment_score FROM headlines h
                                   WHERE h.id = events.headline_id),
                magnitude       = ABS((SELECT sentiment_score FROM headlines h
                                       WHERE h.id = events.headline_id)),
                sentiment_score = (SELECT sentiment_score FROM headlines h
                                   WHERE h.id = events.headline_id),
                sentiment_label = (SELECT sentiment_label FROM headlines h
                                   WHERE h.id = events.headline_id),
                model_version   = (SELECT model_name FROM headlines h
                                   WHERE h.id = events.headline_id),
                signal_date     = (SELECT signal_date FROM headlines h
                                   WHERE h.id = events.headline_id)
            WHERE headline_id IS NOT NULL
            """
        )

    if inserted:
        logger.info("events bridge: %d new event(s) from headlines", inserted)
    return inserted
