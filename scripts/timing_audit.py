"""Prove the timing convention against production records.

The question this answers is not "do the tests pass" but "what does
``signal_date`` actually mean in the rows we have shipped". Two hypotheses are
stated, and real records are asked to refute one of them:

**A.** ``signal_date`` is the publication / associated market session.
**B.** ``signal_date`` is the first session capable of reacting.

``pre_open`` and ``during_session`` cannot tell them apart — for those the
publication session *is* the first reactable session. The verdict therefore
rests entirely on ``post_close``, ``weekend_or_holiday`` and ``unknown``, and
the report says so rather than quietly averaging over buckets that carry no
information.

For each sampled headline the audit derives the expected entry and exit dates
from the hypothesis under test, asks the production window builder what it
generates, and records pass/fail with a reason. A shift of exactly one session
in the post-close family is what the v1 defect looked like.

Usage::

    python -m scripts.timing_audit --db finance_sentiment.db --per-bucket 25
    python -m scripts.timing_audit --db finance_sentiment.db --format markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.return_windows import (  # noqa: E402
    PRIMARY_WINDOW, PriceSeries, build_return_windows,
)
from research.timing import (  # noqa: E402
    ALL_BUCKETS, BUCKET_DURING, BUCKET_PRE_OPEN, BUCKET_UNKNOWN,
    SIGNAL_DATE_SEMANTICS, TIMING_RULE_VERSION, expected_publication_session,
    expected_signal_date, previous_session,
)

DEFAULT_PER_BUCKET = 25

#: Buckets where hypotheses A and B make different predictions.
DISCRIMINATING_BUCKETS = ("post_close", "weekend_or_holiday", "unknown")

HYPOTHESIS_A = "publication_or_associated_session"
HYPOTHESIS_B = "first_reactable_session"

AUDIT_COLUMNS = [
    "headline_id", "title", "published_at_istanbul", "timing_bucket",
    "stored_signal_date", "previous_trading_session", "first_reactable_session",
    "expected_entry_date", "expected_exit_date",
    "generated_entry_date", "generated_exit_date",
    "result", "reason",
]


def _load_headlines(db_path: str) -> List[Dict[str, Any]]:
    import sqlite3

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT id, title, published_at, published_timestamp,
                      published_hour, timing_bucket, signal_date
                 FROM headlines
                WHERE signal_date IS NOT NULL
                ORDER BY id"""
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def _load_prices(db_path: str) -> List[Dict[str, Any]]:
    import sqlite3

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT date, open, close, bar_status FROM bist100_prices ORDER BY date"
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def _timestamp_for(record: Dict[str, Any]) -> Optional[str]:
    """The best publication timestamp available for a row.

    Legacy rows predate the ``published_timestamp`` column but kept
    ``published_hour``; reconstructing the timestamp from it is what the
    ingestion path itself does, so the audit sees the same input production saw.
    """

    stored = record.get("published_timestamp")
    if stored:
        return str(stored)
    hour = record.get("published_hour")
    day = record.get("published_at")
    if day and hour is not None:
        return f"{day}T{int(hour):02d}:00:00+03:00"
    return None


def _sample(records: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Spread the sample across the whole corpus, deterministically.

    Taking the first N would sample one week of one scraper's behaviour. An
    even stride covers the full date range without a random seed.
    """

    total = len(records)
    if total <= limit:
        return list(records)
    step = total / float(limit)
    return [records[int(index * step)] for index in range(limit)]


def classify_semantics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Test both hypotheses against every usable production row."""

    per_bucket: Dict[str, Dict[str, int]] = {
        bucket: {"rows": 0, "matches_a": 0, "matches_b": 0, "a_defined": 0}
        for bucket in ALL_BUCKETS
    }

    for record in records:
        bucket = str(record.get("timing_bucket") or BUCKET_UNKNOWN)
        if bucket not in per_bucket:
            continue
        timestamp = _timestamp_for(record)
        published = record.get("published_at")
        stored = str(record.get("signal_date"))

        derived_b = expected_signal_date(timestamp, published)
        derived_a = expected_publication_session(timestamp, published)
        if derived_b is None:
            continue

        stats = per_bucket[bucket]
        stats["rows"] += 1
        stats["matches_b"] += int(stored == derived_b)
        if derived_a is not None:
            stats["a_defined"] += 1
            stats["matches_a"] += int(stored == derived_a)

    discriminating = {
        bucket: stats for bucket, stats in per_bucket.items()
        if bucket in DISCRIMINATING_BUCKETS and stats["rows"]
    }
    a_holds = all(
        stats["matches_a"] == stats["a_defined"] and stats["a_defined"] == stats["rows"]
        for stats in discriminating.values()
    ) and bool(discriminating)
    b_holds = all(
        stats["matches_b"] == stats["rows"] for stats in per_bucket.values()
        if stats["rows"]
    ) and any(stats["rows"] for stats in per_bucket.values())

    if b_holds and not a_holds:
        verdict = HYPOTHESIS_B
    elif a_holds and not b_holds:
        verdict = HYPOTHESIS_A
    else:
        verdict = "unresolved"

    return {
        "per_bucket": per_bucket,
        "discriminating_buckets": sorted(discriminating),
        "hypothesis_a_holds": a_holds,
        "hypothesis_b_holds": b_holds,
        "verdict": verdict,
        "declared_semantics": SIGNAL_DATE_SEMANTICS,
        "agrees_with_declared": verdict == SIGNAL_DATE_SEMANTICS,
        "rule_version": TIMING_RULE_VERSION,
    }


def _expected_window(
    record: Dict[str, Any], prices: PriceSeries,
) -> Dict[str, Optional[str]]:
    """Expected primary-window entry/exit under the proven convention.

    Derived from the timing bucket and the calendar alone, independently of the
    window builder being audited.
    """

    bucket = str(record.get("timing_bucket") or BUCKET_UNKNOWN)
    session = str(record.get("signal_date"))

    if bucket == BUCKET_DURING:
        return {"entry": None, "exit": None, "note": "blocked: intraday"}
    if bucket == BUCKET_UNKNOWN:
        return {"entry": None, "exit": None, "note": "blocked: unknown time"}
    if prices.get(session) is None:
        return {"entry": None, "exit": None, "note": "no settled bar"}
    # Every tradable bucket executes at the open of the first reactable session
    # and exits at its close. pre_open and post_close differ in what came
    # before, not in where the position is opened.
    return {"entry": session, "exit": session, "note": None}


def audit_rows(
    records: Sequence[Dict[str, Any]],
    prices: PriceSeries,
    *,
    per_bucket: int = DEFAULT_PER_BUCKET,
) -> List[Dict[str, Any]]:
    """Build the per-headline audit table."""

    by_bucket: Dict[str, List[Dict[str, Any]]] = {b: [] for b in ALL_BUCKETS}
    for record in records:
        bucket = str(record.get("timing_bucket") or BUCKET_UNKNOWN)
        if bucket in by_bucket:
            by_bucket[bucket].append(record)

    audited: List[Dict[str, Any]] = []
    for bucket in ALL_BUCKETS:
        for record in _sample(by_bucket[bucket], per_bucket):
            timestamp = _timestamp_for(record)
            stored = str(record.get("signal_date"))
            derived = expected_signal_date(timestamp, record.get("published_at"))
            expected = _expected_window(record, prices)

            windows = build_return_windows(stored, bucket, prices)
            primary = next(
                (w for w in windows if w.window_name == PRIMARY_WINDOW), None
            )
            generated_entry = primary.entry_date if primary and primary.is_available else None
            generated_exit = primary.exit_date if primary and primary.is_available else None

            failures: List[str] = []
            if derived is None:
                failures.append("publication date unusable")
            elif stored != derived:
                failures.append(
                    f"stored signal_date {stored} != first reactable {derived}"
                )
            if expected["entry"] != generated_entry:
                failures.append(
                    f"entry {generated_entry} != expected {expected['entry']}"
                )
            if expected["exit"] != generated_exit:
                failures.append(
                    f"exit {generated_exit} != expected {expected['exit']}"
                )

            audited.append({
                "headline_id": record["id"],
                "title": (str(record.get("title") or ""))[:90],
                "published_at_istanbul": timestamp or "",
                "timing_bucket": bucket,
                "stored_signal_date": stored,
                "previous_trading_session": previous_session(stored) or "",
                "first_reactable_session": derived or "",
                "expected_entry_date": expected["entry"] or "",
                "expected_exit_date": expected["exit"] or "",
                "generated_entry_date": generated_entry or "",
                "generated_exit_date": generated_exit or "",
                "result": "FAIL" if failures else "PASS",
                "reason": "; ".join(failures) or (expected["note"] or "aligned"),
            })
    return audited


def run_audit(
    db_path: str, *, per_bucket: int = DEFAULT_PER_BUCKET,
) -> Dict[str, Any]:
    """Full audit: semantics verdict plus the per-headline table."""

    records = _load_headlines(db_path)
    prices = PriceSeries(_load_prices(db_path))
    semantics = classify_semantics(records)
    table = audit_rows(records, prices, per_bucket=per_bucket)

    coverage = {bucket: 0 for bucket in ALL_BUCKETS}
    failures = {bucket: 0 for bucket in ALL_BUCKETS}
    for row in table:
        coverage[row["timing_bucket"]] += 1
        if row["result"] == "FAIL":
            failures[row["timing_bucket"]] += 1

    return {
        "database": db_path,
        "semantics": semantics,
        "rows": table,
        "sampled_per_bucket": coverage,
        "failures_per_bucket": failures,
        "requested_per_bucket": per_bucket,
        "meets_sample_requirement": all(
            count >= per_bucket for count in coverage.values()
        ),
        "all_passed": not any(failures.values()),
    }


def _markdown(result: Dict[str, Any]) -> str:
    semantics = result["semantics"]
    lines = [
        "# Timing audit",
        "",
        f"- database: `{result['database']}`",
        f"- verdict: **{semantics['verdict']}**",
        f"- hypothesis A (publication session) holds: {semantics['hypothesis_a_holds']}",
        f"- hypothesis B (first reactable session) holds: {semantics['hypothesis_b_holds']}",
        f"- discriminating buckets: {', '.join(semantics['discriminating_buckets']) or 'none'}",
        f"- all sampled rows pass: {result['all_passed']}",
        "",
        "| bucket | rows | matches A | A defined | matches B |",
        "|---|---|---|---|---|",
    ]
    for bucket, stats in semantics["per_bucket"].items():
        lines.append(
            f"| {bucket} | {stats['rows']} | {stats['matches_a']} | "
            f"{stats['a_defined']} | {stats['matches_b']} |"
        )
    lines += ["", "| " + " | ".join(AUDIT_COLUMNS) + " |",
              "|" + "---|" * len(AUDIT_COLUMNS)]
    for row in result["rows"]:
        lines.append(
            "| " + " | ".join(str(row[c]).replace("|", "/") for c in AUDIT_COLUMNS) + " |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default="finance_sentiment.db")
    parser.add_argument("--per-bucket", type=int, default=DEFAULT_PER_BUCKET)
    parser.add_argument("--format", choices=("json", "markdown", "csv"),
                        default="markdown")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_audit(args.db, per_bucket=args.per_bucket)

    if args.format == "json":
        text = json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n"
    elif args.format == "markdown":
        text = _markdown(result)
    else:
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(result["rows"])
        text = buffer.getvalue()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"timing audit written to {args.out}")
    else:
        # Turkish headlines carry characters the Windows console codepage
        # cannot encode. Losing a diacritic in a terminal dump is acceptable;
        # crashing the audit over one is not. Files are always written UTF-8.
        stream = getattr(sys.stdout, "buffer", None)
        if stream is not None:
            stream.write(text.encode("utf-8", errors="backslashreplace"))
            stream.flush()
        else:                                             # pragma: no cover
            sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))

    semantics = result["semantics"]
    print(
        f"\nverdict={semantics['verdict']} "
        f"agrees_with_declared={semantics['agrees_with_declared']} "
        f"all_passed={result['all_passed']}",
        file=sys.stderr,
    )
    return 0 if (result["all_passed"] and semantics["agrees_with_declared"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
