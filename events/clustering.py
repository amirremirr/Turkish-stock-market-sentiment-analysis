"""Deterministic candidate-event grouping.

Transparent rules, not learned similarity. Two headlines join the same candidate
group when they agree on all of:

* a shared normalized entity (or both have none, in which case entity evidence
  is absent and title similarity must carry the decision alone);
* the same signal family;
* publication within a bounded time window;
* normalized-title Jaccard similarity at or above a threshold.

Every pairwise decision keeps its similarity score, the rule that produced it,
and the algorithm version, so any grouping can be re-derived and argued with
afterwards. Nothing here is a verified real-world event: these are *candidate*
groups, and the vocabulary stays that way until a human reviews one.

Determinism matters as much as accuracy. Grouping runs by ascending headline id
with a single linear pass per candidate, so the same corpus always yields the
same groups regardless of dictionary ordering or row arrival order.

Source-independent duplication: a group formed entirely from one outlet is a
single voice repeating itself, and is flagged rather than counted as corroborated
coverage.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from events.entities import classify_event_type, extract_entities
from taxonomy.signal_family import normalize
from trading_calendar import ISTANBUL

CLUSTER_ALGORITHM_VERSION = "event-cluster-transparent-v1"

# Two headlines can join a group only within this many hours of each other.
DEFAULT_TIME_WINDOW_HOURS = 48
# Normalized-title Jaccard threshold when a shared entity is present.
DEFAULT_SIMILARITY_WITH_ENTITY = 0.30
# Higher bar when no entity is shared: title overlap is the only evidence.
DEFAULT_SIMILARITY_WITHOUT_ENTITY = 0.60
# Tokens shorter than this carry little signal in Turkish headlines.
MIN_TOKEN_LENGTH = 4

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very common Turkish news words that would inflate similarity between
# unrelated headlines.
_STOPWORDS: FrozenSet[str] = frozenset({
    "icin", "olarak", "sonra", "once", "ile", "daha", "kadar", "gibi", "ancak",
    "ayrica", "yeni", "buyuk", "genel", "bugun", "dun", "yarin", "sonrasi",
    "aciklama", "acikladi", "oldu", "olacak", "yapti", "yapacak", "dedi",
    "belirtti", "ifade", "soyledi", "gore", "uzere", "dair", "hakkinda",
})


def title_tokens(title: Any) -> FrozenSet[str]:
    """Return the significant tokens of a headline."""

    text = normalize(title)
    return frozenset(
        token for token in _TOKEN_RE.findall(text)
        if len(token) >= MIN_TOKEN_LENGTH and token not in _STOPWORDS
    )


def jaccard(left: FrozenSet[str], right: FrozenSet[str]) -> float:
    """Jaccard similarity; 0.0 when either side has no significant tokens."""

    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _parse_moment(value: Any) -> Optional[datetime]:
    """Parse to a timezone-aware UTC moment.

    The corpus mixes tz-aware timestamps with date-only publication dates. A
    naive value is read as Istanbul local, matching the convention used by
    trading_calendar for the same feeds, then normalized to UTC so every
    comparison in this module is between like and like.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(raw[:10])
            except ValueError:
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ISTANBUL)
    return parsed.astimezone(timezone.utc)


@dataclass
class CandidateEvent:
    """An algorithmic event group. Not a verified real-world event."""

    group_key: str
    signal_family: Optional[str]
    event_type: Optional[str]
    primary_entity: Optional[str]
    entities: Set[Tuple[str, str]] = field(default_factory=set)
    members: List[Dict[str, Any]] = field(default_factory=list)
    seed_tokens: FrozenSet[str] = frozenset()
    seed_headline_id: Optional[int] = None

    @property
    def headline_ids(self) -> List[int]:
        return [int(member["headline_id"]) for member in self.members]

    @property
    def sources(self) -> Set[str]:
        return {
            str(member["source"]) for member in self.members
            if member.get("source")
        }

    @property
    def is_singleton(self) -> bool:
        return len(self.members) == 1

    @property
    def is_single_source(self) -> bool:
        return len(self.sources) <= 1


def _member_record(
    record: Dict[str, Any],
    *,
    similarity: float,
    rule: str,
    entity_overlap: Sequence[str],
) -> Dict[str, Any]:
    return {
        "headline_id": int(record["id"]),
        "source": record.get("source"),
        "title": record.get("title"),
        "published_timestamp": record.get("published_timestamp"),
        "published_at": record.get("published_at"),
        "signal_date": record.get("signal_date"),
        "timing_bucket": record.get("timing_bucket"),
        "sentiment_score": record.get("sentiment_score"),
        "relevance": record.get("relevance"),
        "is_market_recap": record.get("is_market_recap"),
        "similarity": similarity,
        "match_rule": rule,
        "entity_overlap": ",".join(sorted(entity_overlap)) or None,
    }


def group_candidate_events(
    records: Sequence[Dict[str, Any]],
    *,
    time_window_hours: int = DEFAULT_TIME_WINDOW_HOURS,
    similarity_with_entity: float = DEFAULT_SIMILARITY_WITH_ENTITY,
    similarity_without_entity: float = DEFAULT_SIMILARITY_WITHOUT_ENTITY,
    algorithm_version: str = CLUSTER_ALGORITHM_VERSION,
) -> List[CandidateEvent]:
    """Group headlines into candidate events.

    Records are processed in ascending headline id, and each is compared against
    open groups only. A group closes once the candidate is beyond the time
    window, which keeps the pass linear in practice and the result independent
    of input ordering.
    """

    ordered = sorted(records, key=lambda record: int(record["id"]))
    window = timedelta(hours=time_window_hours)

    groups: List[CandidateEvent] = []
    # (anchor_moment, group). The anchor is the group's FIRST member and never
    # moves. Advancing it to the newest member would let a daily drumbeat -- a
    # recap headline printed every morning, each within the window of the last --
    # chain across months into one candidate event. Anchoring bounds a group's
    # total duration to the window, which is what the window is meant to mean.
    open_groups: List[Tuple[datetime, CandidateEvent]] = []

    for record in ordered:
        moment = (
            _parse_moment(record.get("published_timestamp"))
            or _parse_moment(record.get("published_at"))
        )
        extraction = extract_entities(record.get("title"))
        tokens = title_tokens(record.get("title"))
        family = record.get("signal_family")
        event_type, _ = classify_event_type(record.get("title"))

        best: Optional[Tuple[float, str, List[str], CandidateEvent]] = None
        still_open: List[Tuple[datetime, CandidateEvent]] = []

        session = record.get("signal_date")
        for anchor, group in open_groups:
            if moment is not None and anchor is not None:
                if abs(moment - anchor) > window:
                    continue                     # group has aged out
            still_open.append((anchor, group))

            if group.signal_family != family:
                continue

            # Without a timestamp on either side there is no proximity
            # evidence, so fall back to requiring the same reaction session.
            # Otherwise a recurring headline with no publication time -- "Borsa
            # opened higher", printed most mornings -- would merge months of
            # separate daily stories into one candidate event.
            if moment is None or anchor is None:
                group_sessions = {
                    member.get("signal_date") for member in group.members
                }
                if session is None or group_sessions != {session}:
                    continue

            overlap = sorted(group.entities & extraction.entities)
            overlap_ids = [entity_id for _, entity_id in overlap]
            similarity = jaccard(group.seed_tokens, tokens)
            threshold = (
                similarity_with_entity if overlap_ids else similarity_without_entity
            )
            if similarity < threshold:
                continue

            rule = (
                "entity+family+time+title" if overlap_ids
                else "family+time+title_only"
            )
            if best is None or similarity > best[0]:
                best = (similarity, rule, overlap_ids, group)

        open_groups = still_open

        if best is not None:
            similarity, rule, overlap_ids, group = best
            group.members.append(
                _member_record(record, similarity=similarity, rule=rule,
                               entity_overlap=overlap_ids)
            )
            group.entities |= extraction.entities
            if group.event_type is None:
                group.event_type = event_type
            continue

        group = CandidateEvent(
            group_key=f"{algorithm_version}:{int(record['id'])}",
            signal_family=family,
            event_type=event_type,
            primary_entity=extraction.primary_entity,
            entities=set(extraction.entities),
            seed_tokens=tokens,
            seed_headline_id=int(record["id"]),
        )
        group.members.append(
            _member_record(record, similarity=1.0, rule="seed", entity_overlap=[])
        )
        groups.append(group)
        open_groups.append((moment, group))

    return groups


@dataclass(frozen=True)
class EventAggregate:
    """Event-level descriptive summary. Not a verified real-world event."""

    values: Dict[str, Any]


def summarise_event(
    group: CandidateEvent,
    *,
    prior_entity_events: int = 0,
    strong_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute event-level metadata, tone and disagreement.

    ``novelty`` is 1/(1 + prior candidate groups for the same entity) over the
    preceding history. It measures how often this entity has already generated a
    group, not whether the news itself is new -- the name says candidate-level
    repetition, and nothing more should be read into it.
    """

    scores = [
        float(member["sentiment_score"]) for member in group.members
        if member.get("sentiment_score") is not None
        and _finite(member["sentiment_score"])
    ]
    sources = sorted(group.sources)
    timestamps = [
        _parse_moment(member.get("published_timestamp"))
        or _parse_moment(member.get("published_at"))
        for member in group.members
    ]
    known = [moment for moment in timestamps if moment is not None]
    signal_dates = sorted(
        {member["signal_date"] for member in group.members if member.get("signal_date")}
    )

    by_source: Dict[str, List[float]] = {}
    for member in group.members:
        score = member.get("sentiment_score")
        if member.get("source") and score is not None and _finite(score):
            by_source.setdefault(str(member["source"]), []).append(float(score))
    source_means = [
        math.fsum(values) / len(values) for values in by_source.values()
    ]

    mean = math.fsum(scores) / len(scores) if scores else None
    median = _median(scores) if scores else None
    dispersion = _pstdev(scores) if len(scores) > 1 else None
    # Cross-source disagreement needs at least two distinct voices; one outlet
    # cannot disagree with itself.
    cross_source = _pstdev(source_means) if len(source_means) > 1 else None

    return {
        "group_key": group.group_key,
        "algorithm_version": CLUSTER_ALGORITHM_VERSION,
        "signal_family": group.signal_family,
        "event_type": group.event_type,
        "primary_entity": group.primary_entity,
        "entity_ids": ",".join(sorted({e for _, e in group.entities})) or None,
        "headline_count": len(group.members),
        "source_count": len(sources),
        "sources": ",".join(sources) or None,
        "first_seen_at": min(known).isoformat() if known else None,
        "last_seen_at": max(known).isoformat() if known else None,
        "unknown_timestamp_count": sum(
            1 for moment in timestamps if moment is None
        ),
        "signal_date": signal_dates[0] if signal_dates else None,
        "signal_date_span": len(signal_dates),
        "mean_sentiment": mean,
        "median_sentiment": median,
        "sentiment_dispersion": dispersion,
        "cross_source_dispersion": cross_source,
        "strong_positive_count": sum(
            1 for score in scores if score >= strong_threshold
        ),
        "strong_negative_count": sum(
            1 for score in scores if score <= -strong_threshold
        ),
        "market_recap_count": sum(
            1 for member in group.members if member.get("is_market_recap")
        ),
        "is_singleton": 1 if group.is_singleton else 0,
        "is_single_source": 1 if group.is_single_source else 0,
        "novelty": 1.0 / (1.0 + prior_entity_events),
        "prior_entity_events": prior_entity_events,
        "review_state": "unreviewed",
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _pstdev(values: List[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
