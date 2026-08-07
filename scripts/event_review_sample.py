"""Draw a stratified sample of candidate groups for human review.

Grouping quality has so far been argued from the algorithm's own rules. This
draws a sample a person can actually read and judge, and it is built to keep
that judgement clean:

**No market data is consulted.** The draw never reads a return, a residual or a
target. A reviewer shown "this group preceded a 2% rally" is no longer judging
whether the headlines describe one story.

**It is deterministic.** Ordering is by a stable hash of the group key, not by
``random``, so the same corpus yields the same sample on any machine and the
draw can be re-derived rather than trusted.

**It is stratified toward the boundaries.** A uniform sample of a corpus that is
94% singletons is 94% singletons, which tells you almost nothing about the
decisions the algorithm actually makes. The strata below deliberately
over-sample the places where the rules are closest to flipping — similarities
just above and just below each threshold, and near-neighbour pairs that did
*not* merge, which is the only way a missed merge can ever be seen.

**It cannot change anything.** Reviewing produces a filled-in sheet. Thresholds
are not tuned from it, and any later algorithm revision takes a new version
string, so reviewed and unreviewed groupings never blur together.

Usage::

    python -m scripts.event_review_sample --db finance_sentiment.db --out review.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db  # noqa: E402
from events.clustering import (  # noqa: E402
    DEFAULT_SIMILARITY_WITH_ENTITY, DEFAULT_SIMILARITY_WITHOUT_ENTITY,
    jaccard, title_tokens,
)

SAMPLE_VERSION = "event-review-sample-v1"

#: How many groups to draw from each stratum. Small enough that a person can
#: actually finish the sheet, which matters more than statistical power here.
DEFAULT_PER_STRATUM = 15

#: Similarity band counted as "near the threshold" on either side.
THRESHOLD_BAND = 0.08

STRATA = (
    "multi_source",
    "multi_headline_single_source",
    "high_similarity",
    "threshold_near",
    "entity_based",
    "title_only",
    "cross_session",
    "near_miss_pair",
)

#: What a reviewer may record. Free text goes in `notes`, not in the verdict.
REVIEW_VERDICTS = ("correct_group", "false_merge", "missed_merge", "uncertain")

REVIEW_COLUMNS = [
    "stratum", "group_key", "comparison_group_key", "similarity", "rationale",
    "headline_count", "source_count", "sources", "signal_family", "event_type",
    "primary_entity", "first_reactable_session", "titles",
    "verdict", "reviewer", "reviewed_at", "notes",
]


def _order_key(value: str) -> str:
    """Stable ordering independent of insertion order or Python's hash seed."""

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _load(db_path: str) -> Dict[str, Any]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        groups = [dict(r) for r in connection.execute(
            """SELECT group_key, algorithm_version, signal_family, event_type,
                      primary_entity, entity_ids, headline_count, source_count,
                      sources, first_reactable_session, signal_date_span,
                      is_singleton, is_single_source, timing_conflict
                 FROM event_groups"""
        )]
        mappings = [dict(r) for r in connection.execute(
            """SELECT m.group_key, m.headline_id, m.similarity, m.match_rule,
                      m.entity_overlap, h.title, h.source
                 FROM event_headline_map m
                 JOIN headlines h ON h.id = m.headline_id"""
        )]
    finally:
        connection.close()

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for row in mappings:
        by_group.setdefault(row["group_key"], []).append(row)
    return {"groups": groups, "members": by_group}


def _members_similarity(members: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Lowest non-seed similarity in a group: the weakest link that held."""

    scores = [
        float(m["similarity"]) for m in members
        if m.get("match_rule") != "seed" and m.get("similarity") is not None
    ]
    return min(scores) if scores else None


def _near_miss_pairs(
    groups: Sequence[Dict[str, Any]],
    members: Dict[str, List[Dict[str, Any]]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Singleton pairs that were each other's nearest neighbour and did not merge.

    This is the only stratum that can surface a *missed* merge. Everything else
    inspects decisions the algorithm made; this inspects one it declined to
    make. Restricted to same-family, same-session singleton pairs, which is
    where a missed merge is both plausible and checkable by reading two titles.
    """

    singletons = [g for g in groups if int(g.get("is_singleton") or 0)]
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for group in singletons:
        key = (group.get("signal_family"), group.get("first_reactable_session"))
        if all(key):
            buckets.setdefault(key, []).append(group)

    candidates: List[Dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        if len(bucket) < 2:
            continue
        titled = []
        for group in sorted(bucket, key=lambda g: _order_key(g["group_key"])):
            rows = members.get(group["group_key"]) or []
            if rows:
                titled.append((group, rows[0].get("title") or "",
                               title_tokens(rows[0].get("title"))))
        for index, (left, left_title, left_tokens) in enumerate(titled):
            best = None
            for right, right_title, right_tokens in titled[index + 1:]:
                score = jaccard(left_tokens, right_tokens)
                if best is None or score > best[0]:
                    best = (score, right, right_title)
            if best is None or best[0] <= 0.0:
                continue
            score, right, right_title = best
            # Only pairs that came close enough to be arguable.
            if score < DEFAULT_SIMILARITY_WITH_ENTITY - THRESHOLD_BAND:
                continue
            candidates.append({
                "group_key": left["group_key"],
                "comparison_group_key": right["group_key"],
                "algorithm_version": left["algorithm_version"],
                "similarity": round(score, 4),
                "rationale": (
                    f"nearest neighbour in {key[0]} on {key[1]} at Jaccard "
                    f"{score:.2f}; did not merge"
                ),
                "_titles": f"{left_title} || {right_title}",
                "_group": left,
            })

    candidates.sort(key=lambda c: (-c["similarity"], _order_key(c["group_key"])))
    return candidates[:limit]


def draw_sample(
    db_path: str, *, per_stratum: int = DEFAULT_PER_STRATUM,
) -> List[Dict[str, Any]]:
    """Build the stratified draw. Deterministic, and blind to market data."""

    loaded = _load(db_path)
    groups = loaded["groups"]
    members = loaded["members"]
    index = {g["group_key"]: g for g in groups}

    def _pick(predicate, stratum: str, rationale: str) -> List[Dict[str, Any]]:
        chosen = sorted(
            (g for g in groups if predicate(g)),
            key=lambda g: _order_key(g["group_key"]),
        )[:per_stratum]
        return [{
            "stratum": stratum,
            "group_key": g["group_key"],
            "algorithm_version": g["algorithm_version"],
            "comparison_group_key": None,
            "similarity": _members_similarity(members.get(g["group_key"], [])),
            "rationale": rationale,
            "_group": g,
        } for g in chosen]

    def _rules(group) -> set:
        return {
            m.get("match_rule") for m in members.get(group["group_key"], [])
            if m.get("match_rule") and m["match_rule"] != "seed"
        }

    rows: List[Dict[str, Any]] = []
    rows += _pick(
        lambda g: int(g.get("source_count") or 0) > 1,
        "multi_source", "carried by more than one outlet",
    )
    rows += _pick(
        lambda g: int(g.get("headline_count") or 0) > 1
        and int(g.get("source_count") or 0) <= 1,
        "multi_headline_single_source",
        "several headlines, one outlet: repetition or a real story?",
    )
    rows += _pick(
        lambda g: (_members_similarity(members.get(g["group_key"], [])) or 0) >= 0.8,
        "high_similarity", "near-identical titles; check for syndication",
    )
    rows += _pick(
        lambda g: (
            lambda s: s is not None and (
                abs(s - DEFAULT_SIMILARITY_WITH_ENTITY) <= THRESHOLD_BAND
                or abs(s - DEFAULT_SIMILARITY_WITHOUT_ENTITY) <= THRESHOLD_BAND
            )
        )(_members_similarity(members.get(g["group_key"], []))),
        "threshold_near", "similarity within one band of a decision threshold",
    )
    rows += _pick(
        lambda g: "entity+family+time+title" in _rules(g),
        "entity_based", "merged on a shared entity",
    )
    rows += _pick(
        lambda g: "family+time+title_only" in _rules(g),
        "title_only", "merged on title overlap with no shared entity",
    )
    rows += _pick(
        lambda g: int(g.get("signal_date_span") or 1) > 1,
        "cross_session", "members react on different sessions",
    )

    for candidate in _near_miss_pairs(groups, members, per_stratum):
        rows.append({
            "stratum": "near_miss_pair",
            "group_key": candidate["group_key"],
            "algorithm_version": candidate["algorithm_version"],
            "comparison_group_key": candidate["comparison_group_key"],
            "similarity": candidate["similarity"],
            "rationale": candidate["rationale"],
            "_group": candidate["_group"],
            "_titles": candidate["_titles"],
        })

    for row in rows:
        group = row.pop("_group")
        titles = row.pop("_titles", None)
        if titles is None:
            titles = " || ".join(
                str(m.get("title") or "")
                for m in sorted(members.get(group["group_key"], []),
                                key=lambda m: m["headline_id"])
            )
        row.update({
            "headline_count": group.get("headline_count"),
            "source_count": group.get("source_count"),
            "sources": group.get("sources"),
            "signal_family": group.get("signal_family"),
            "event_type": group.get("event_type"),
            "primary_entity": group.get("primary_entity"),
            "first_reactable_session": group.get("first_reactable_session"),
            "titles": titles[:600],
        })
    return rows


def write_sheet(rows: Sequence[Dict[str, Any]], path: Path) -> Path:
    """Write the review sheet: one row per group, verdict columns left blank."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row, "verdict": "", "reviewer": "", "reviewed_at": "",
                "notes": "",
            })
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default="finance_sentiment.db")
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    parser.add_argument("--out", type=Path,
                        default=Path("docs/event_review_sheet.csv"))
    parser.add_argument("--no-store", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = draw_sample(args.db, per_stratum=args.per_stratum)

    if not args.no_store:
        db.replace_event_review_sample(
            rows, sample_version=SAMPLE_VERSION, db_path=args.db,
        )
    written = write_sheet(rows, args.out)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    print(f"{SAMPLE_VERSION}: {len(rows)} group(s) drawn")
    for stratum in STRATA:
        print(f"  {stratum:<30} {counts.get(stratum, 0)}")
    print(f"\nreview sheet: {written}")
    print(f"verdicts: {', '.join(REVIEW_VERDICTS)}")
    print("Thresholds are not tuned from this review; a revised algorithm takes "
          "a new version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
