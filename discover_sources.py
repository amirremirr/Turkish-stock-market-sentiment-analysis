"""
discover_sources.py — Evaluate RSS feeds before adding them to config.py.

Usage
-----
  # Test a single new URL
  python discover_sources.py --url https://www.hurriyet.com.tr/rss/ekonomi

  # Test all currently configured feeds (baseline)
  python discover_sources.py --all

  # Test a batch of candidate URLs from a file (one URL per line)
  python discover_sources.py --file candidates.txt

  # Test with verbose sample headlines
  python discover_sources.py --url <URL> --sample 10

What it measures
----------------
  Reachability     Does the feed return HTTP 200?
  Parseable        Is the XML valid?
  Volume           How many items are in the feed right now?
  Date coverage    What date range do items span?
  Date completeness% What fraction of items have a parseable date?
  Turkish chars%   What fraction of titles contain Turkish characters?
                   Low value (<20%) often means encoding corruption.
  Relevance rate%  What fraction pass the current relevance filter?
                   Below 30% means the feed is mostly off-topic noise.
  Category spread  Which categories the headlines land in.
  Duplicates       How much overlap with headlines already in the DB?
  Verdict          Add / Marginal / Skip — based on the above metrics.

Decision guide
--------------
  Add    : reachable, parseable, ≥10 items, ≥40% relevant, ≥20% Turkish chars
  Marginal: reachable but low relevance or low volume — inspect sample manually
  Skip   : unreachable, unparseable, or <10% relevant
"""

import argparse
import sys
import time
import textwrap
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import List, Optional

# Force UTF-8 output so Turkish characters print correctly in any terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import requests

# Project imports
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CRAWL_DELAY,
    REQUEST_TIMEOUT,
    RSS_FEEDS,
    RELEVANCE_FILTER_ENABLED,
)
from scraper import (
    _normalise,
    _is_relevant,
    _parse_date,
    _cdata_strip,
    classify_headline,
    RSSFeedScraper,
)
import database as db
from config import DB_PATH

# ── Formatting helpers ────────────────────────────────────────────────────────

W = 66

def _hdr(title: str) -> None:
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")

def _row(label: str, value, note: str = "", warn: bool = False) -> None:
    tag = "  [!]" if warn else ""
    print(f"  {label:<38} {str(value):<14} {note}{tag}")

def _ok(msg: str)   -> None: print(f"  [OK]  {msg}")
def _warn(msg: str) -> None: print(f"  [!!]  {msg}")
def _info(msg: str) -> None: print(f"  [ ]   {msg}")


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_feed(
    url: str,
    label: str = "",
    sample_n: int = 5,
    db_path: str = DB_PATH,
) -> dict:
    """
    Fetch a single RSS URL, measure its quality, print a structured report.
    Returns a result dict with the key metrics.
    """
    label = label or url
    _hdr(f"Feed: {label}")
    print(f"  URL: {url}\n")

    result = {
        "url":         url,
        "label":       label,
        "reachable":   False,
        "parseable":   False,
        "item_count":  0,
        "verdict":     "SKIP",
    }

    # ── Fetch ────────────────────────────────────────────────────────────────
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    }
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        status = resp.status_code
        result["reachable"] = (status == 200)
        _row("HTTP status", status, warn=(status != 200))
        if status != 200:
            _warn(f"Feed returned HTTP {status} — skip.")
            return result
    except Exception as exc:
        _warn(f"Connection failed: {exc}")
        return result

    result["reachable"] = True

    # ── Parse XML ────────────────────────────────────────────────────────────
    try:
        scraper = RSSFeedScraper()
        root = scraper._parse_xml(resp.content)
        result["parseable"] = True
    except ET.ParseError as exc:
        _warn(f"XML parse failed: {exc}")
        return result

    items = root.findall(".//item") or root.findall(
        ".//{http://www.w3.org/2005/Atom}entry"
    )
    n = len(items)
    result["item_count"] = n
    _row("Items in feed", n, warn=(n < 5))
    if n == 0:
        _warn("Feed parsed OK but contains 0 items.")
        return result

    # ── Extract titles and dates ─────────────────────────────────────────────
    titles = []
    dates  = []
    for item in items:
        title = (
            _cdata_strip(item.findtext("title") or "")
            or _cdata_strip(
                item.findtext("{http://www.w3.org/2005/Atom}title") or ""
            )
        ).strip()
        if title:
            titles.append(title)

        raw_date = (
            item.findtext("pubDate")
            or item.findtext("{http://www.w3.org/2005/Atom}published")
            or item.findtext("{http://www.w3.org/2005/Atom}updated")
            or ""
        )
        d = _parse_date(raw_date)
        dates.append(d)

    has_date  = [d for d in dates if d is not None]
    date_pct  = len(has_date) / n * 100 if n else 0
    _row("Items with parseable date", f"{len(has_date)}/{n}", f"({date_pct:.0f}%)",
         warn=(date_pct < 50))

    if has_date:
        oldest = min(has_date)
        newest = max(has_date)
        span   = (newest - oldest).days
        _row("Date range", f"{oldest} .. {newest}")
        _row("Span (days)", span)

        days_stale = (date.today() - newest).days
        _row("Most recent item age (days)", days_stale,
             warn=(days_stale > 3))
        if days_stale > 3:
            _warn("Feed appears stale — last item is more than 3 days old")

    # ── Turkish character encoding ────────────────────────────────────────────
    def _has_turkish(t: str) -> bool:
        return any(c in "şçğıöüŞÇĞİÖÜ" for c in t)

    tr_count = sum(1 for t in titles if _has_turkish(t))
    tr_pct   = tr_count / len(titles) * 100 if titles else 0
    _row("Titles with Turkish chars", f"{tr_count}/{len(titles)}",
         f"({tr_pct:.0f}%)", warn=(tr_pct < 20))
    if tr_pct < 20:
        _warn("Very few Turkish characters — possible encoding corruption or non-Turkish source")

    # ── Relevance filtering ───────────────────────────────────────────────────
    relevant   = [t for t in titles if _is_relevant(t)]
    rel_pct    = len(relevant) / len(titles) * 100 if titles else 0
    irrelevant = [t for t in titles if not _is_relevant(t)]
    _row("Headlines passing relevance filter", f"{len(relevant)}/{len(titles)}",
         f"({rel_pct:.0f}%)", warn=(rel_pct < 30))

    if rel_pct < 30:
        _warn("Low relevance rate — most headlines are off-topic for BIST analysis")

    # ── Category spread ───────────────────────────────────────────────────────
    if relevant:
        from collections import Counter
        cats = Counter(classify_headline(t) for t in relevant)
        print()
        print(f"  Category breakdown ({len(relevant)} relevant headlines):")
        for cat, cnt in cats.most_common():
            bar = "█" * cnt
            warn = cat == "other" and cnt / len(relevant) > 0.4
            print(f"    {cat:<22} {cnt:>3}  {bar}{'  [!] high other' if warn else ''}")

    # ── Duplicate check against DB ───────────────────────────────────────────
    try:
        with db._conn(db_path) as con:
            db_titles_raw = con.execute(
                "SELECT title FROM headlines ORDER BY scraped_at DESC LIMIT 500"
            ).fetchall()
        db_titles_norm = {_normalise(r["title"][:80]) for r in db_titles_raw}
        overlaps = sum(
            1 for t in relevant if _normalise(t[:80]) in db_titles_norm
        )
        overlap_pct = overlaps / len(relevant) * 100 if relevant else 0
        _row("Overlap with existing DB headlines",
             f"{overlaps}/{len(relevant)}", f"({overlap_pct:.0f}%)")
        if overlap_pct > 70:
            _info("High overlap — this feed may largely duplicate an existing source")
    except Exception:
        pass  # DB may not exist yet

    # ── Sample headlines ─────────────────────────────────────────────────────
    if sample_n > 0 and relevant:
        print(f"\n  Sample relevant headlines (first {min(sample_n, len(relevant))}):")
        for t in relevant[:sample_n]:
            cat = classify_headline(t)
            short = textwrap.shorten(t, width=60, placeholder="...")
            print(f"    [{cat:16}]  {short}")

    if irrelevant and sample_n > 0:
        print(f"\n  Sample FILTERED OUT headlines (first {min(3, len(irrelevant))}):")
        for t in irrelevant[:3]:
            short = textwrap.shorten(t, width=60, placeholder="...")
            print(f"    [filtered]  {short}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    if not result["reachable"] or not result["parseable"]:
        verdict = "SKIP"
        reason  = "Unreachable or unparseable"
    elif n < 5:
        verdict = "SKIP"
        reason  = "Too few items"
    elif rel_pct < 15:
        verdict = "SKIP"
        reason  = f"Only {rel_pct:.0f}% relevant — mostly off-topic noise"
    elif rel_pct < 35 or tr_pct < 20:
        verdict = "MARGINAL"
        reason  = "Low relevance or encoding issues — inspect sample manually"
    elif days_stale > 5 if has_date else False:
        verdict = "MARGINAL"
        reason  = "Feed is stale — may not update regularly"
    else:
        verdict = "ADD"
        reason  = f"{rel_pct:.0f}% relevant, {tr_pct:.0f}% Turkish chars, {n} items"

    result.update({
        "verdict":    verdict,
        "rel_pct":    rel_pct,
        "tr_pct":     tr_pct,
        "days_stale": days_stale if has_date else None,
    })

    verdict_icon = {"ADD": "[OK]", "MARGINAL": "[ ]", "SKIP": "[!!]"}[verdict]
    print(f"  {verdict_icon}  VERDICT: {verdict}  —  {reason}")

    if verdict == "ADD":
        # Suggest the config.py key name from the domain
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "").split(".")[0]
        print(f"\n  To add to config.py:")
        print(f'      "{domain}": "{url}",')
        print(f"  (inside RSS_FEEDS dict in config.py)")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test RSS feeds before adding them to the pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  help="Single RSS URL to evaluate")
    group.add_argument("--all",  action="store_true",
                       help="Test all currently configured RSS feeds")
    group.add_argument("--file", help="Text file with one RSS URL per line")
    parser.add_argument("--sample", type=int, default=5,
                        help="Number of sample headlines to print (default: 5)")
    parser.add_argument("--db", default=DB_PATH, help="SQLite DB path")
    args = parser.parse_args()

    results = []

    if args.url:
        r = evaluate_feed(args.url, sample_n=args.sample, db_path=args.db)
        results.append(r)

    elif args.all:
        print(f"Testing {len(RSS_FEEDS)} configured feeds...\n")
        for key, url in RSS_FEEDS.items():
            r = evaluate_feed(url, label=key, sample_n=args.sample, db_path=args.db)
            results.append(r)
            time.sleep(CRAWL_DELAY)

    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"Testing {len(urls)} candidate URLs from {args.file}...\n")
        for url in urls:
            r = evaluate_feed(url, sample_n=args.sample, db_path=args.db)
            results.append(r)
            time.sleep(CRAWL_DELAY)

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'=' * W}")
        print("  SUMMARY")
        print(f"{'=' * W}")
        print(f"  {'Feed':<28} {'Items':>6}  {'Rel%':>5}  {'TR%':>4}  {'Verdict'}")
        print(f"  {'-'*28}  {'-'*6}  {'-'*5}  {'-'*4}  {'-'*8}")
        for r in results:
            icon = {"ADD": "✓", "MARGINAL": "~", "SKIP": "✗"}.get(r.get("verdict",""), "?")
            print(
                f"  {r['label']:<28}  "
                f"{r.get('item_count', 0):>6}  "
                f"{r.get('rel_pct', 0):>4.0f}%  "
                f"{r.get('tr_pct', 0):>3.0f}%  "
                f"{icon} {r.get('verdict', '?')}"
            )
        adds = [r for r in results if r.get("verdict") == "ADD"]
        if adds:
            print(f"\n  {len(adds)} feed(s) recommended to add.")
        print()


if __name__ == "__main__":
    main()
