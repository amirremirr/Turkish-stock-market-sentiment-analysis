"""
News scraper - pulls Turkish financial headlines from RSS feeds (primary)
with an HTML fallback for tr.investing.com.

Returned headline dicts always contain:
    title        str
    url          str | None
    published_at datetime.date | None
    source       str   (feed/source key name)
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from config import (
    CRAWL_DELAY,
    DEFAULT_LOOKBACK_DAYS,
    HTML_SOURCES,
    KEYWORD_RELEVANCE_RULE_VERSION,
    NEWS_CATEGORIES,
    RELEVANCE_BLOCKLIST,
    RELEVANCE_FILTER_ENABLED,
    RELEVANCE_HARD_BLOCKLIST,
    RELEVANCE_KEYWORDS,
    RELEVANCE_STRONG,
    REQUEST_TIMEOUT,
    RSS_FEEDS,
)

logger = logging.getLogger(__name__)


# -- Relevance filter ----------------------------------------------------------

# Full Turkish -> ASCII fold table, expressed as Unicode codepoints so this
# source file contains zero non-ASCII characters.
#
# Problem: Python .lower() maps:
#   U+0130 (capital dotted-I / "I with dot above") -> "i̇" (i + combining dot)
#   U+0131 (lowercase dotless-i)                   -> stays U+0131, not ASCII 'i'
#   U+015E/015F (S/s with cedilla = ş)             -> stays ş, not 's'
#   etc.
#
# Our keywords use ASCII-folded forms ("turkiye", "doviz", "bankacilik"),
# so the input text must be folded the same way before matching.
#
# Mapping (codepoint -> replacement char):
#   U+0130  I-dot-above   -> i   (capital dotted I)
#   U+015E  S-cedilla     -> s   (capital S with cedilla = S)
#   U+015F  s-cedilla     -> s   (small s with cedilla = s)
#   U+00C7  C-cedilla     -> c   (capital C with cedilla = C)
#   U+00E7  c-cedilla     -> c   (small c with cedilla = c)
#   U+011E  G-breve       -> g   (capital G with breve = G)
#   U+011F  g-breve       -> g   (small g with breve = g)
#   U+00D6  O-diaeresis   -> o   (capital O with diaeresis = O)
#   U+00F6  o-diaeresis   -> o   (small o with diaeresis = o)
#   U+00DC  U-diaeresis   -> u   (capital U with diaeresis = U)
#   U+00FC  u-diaeresis   -> u   (small u with diaeresis = u)
#   U+0131  dotless-i     -> i   (Turkish small dotless i)
#
_TR_ASCII_FOLD = str.maketrans({
    0x130: "i",   # capital I with dot above
    0x15E: "s",  0x15F: "s",   # S/s with cedilla
    0x00C7: "c", 0x00E7: "c",  # C/c with cedilla
    0x11E: "g",  0x11F: "g",   # G/g with breve
    0x00D6: "o", 0x00F6: "o",  # O/o with diaeresis
    0x00DC: "u", 0x00FC: "u",  # U/u with diaeresis
    0x131: "i",                 # dotless lowercase i
})


def _normalise(text: str) -> str:
    """
    ASCII-fold all Turkish diacritics then lowercase.

    After this, "Turkiye", "turkiye", and "TURKIYE" all become "turkiye".
    Keywords in config.py must use these ASCII-folded forms.
    """
    return text.translate(_TR_ASCII_FOLD).lower()


def classify_headline(title: str) -> str:
    """
    Classify a headline into one of the NEWS_CATEGORIES buckets.

    Uses the same ASCII-folded normalisation as the relevance filter so that
    Turkish diacritics in live headlines match ASCII keywords.

    Returns the category slug of the first matching rule, or "other".
    """
    t = _normalise(title)
    for category, keywords in NEWS_CATEGORIES:
        if any(kw in t for kw in keywords):
            return category
    return "other"


def _is_relevant(title: str) -> bool:
    """
    Three-tier relevance filter:

    Tier 0 - Hard blocklist: topics so off-topic that even a Turkey mention
              cannot save them (lottery, World Cup, water levels, concerts…).
              Applied BEFORE the strong-marker bypass.

    Tier 1 - Soft blocklist: if a blocklist term is found AND no strong Turkey
              marker is present, exclude the headline from derived analysis.
              Catches Bitcoin/crypto, NYSE, Nikkei, Indian rupee, etc.

    Tier 2 - Keyword match: at least one keyword from RELEVANCE_KEYWORDS
              must be present.

    Returns True (include) or False (exclude). Collection preserves both.
    """
    return _relevance_exclusion_rule(title) is None


def _relevance_exclusion_rule(title: str) -> Optional[str]:
    """Return the keyword rule that excludes ``title``, or ``None``.

    Keeping the rule decision separate from :func:`_is_relevant` lets the
    collector preserve off-topic observations while retaining the legacy
    boolean predicate for existing callers.
    """
    t = _normalise(title)

    # Tier 0 — always exclude, no exceptions.
    if any(hbl in t for hbl in RELEVANCE_HARD_BLOCKLIST):
        return "hard_blocklist"

    has_strong = any(s in t for s in RELEVANCE_STRONG)

    # Tier 1 — a strong Turkey-market marker overrides the soft blocklist.
    if not has_strong and any(bl in t for bl in RELEVANCE_BLOCKLIST):
        return "soft_blocklist"

    # Tier 2 — observations with no market keyword remain auditable.
    if not any(kw in t for kw in RELEVANCE_KEYWORDS):
        return "missing_relevance_keyword"

    return None


def _with_exclusion_metadata(headline: Dict) -> Dict:
    """Return an observation with explicit, reversible exclusion metadata.

    ``excluded_at`` belongs to the persistence layer: collection identifies
    the rule outcome, while the database records when it was stored.
    """
    observation = dict(headline)
    rule = (
        _relevance_exclusion_rule(str(observation.get("title", "")))
        if RELEVANCE_FILTER_ENABLED else None
    )
    observation.update({
        "is_excluded": rule is not None,
        "exclusion_reason": "off_topic" if rule is not None else None,
        "exclusion_rule": rule,
        "exclusion_version": (
            KEYWORD_RELEVANCE_RULE_VERSION if rule is not None else None
        ),
    })
    return observation


def _observation_identity(headline: Dict) -> tuple:
    """Return a stable, source-scoped identity for in-run deduplication.

    URLs are preferred. URL-less observations fall back to normalised title and
    publication date. Including the source ensures identical headlines from
    different outlets remain distinct raw observations.
    """
    source = str(headline.get("source") or "unknown")
    url = str(headline.get("url") or "").strip()
    if url:
        return source, "url", url
    published = headline.get("published_at")
    return (
        source,
        "title_date",
        _normalise(str(headline.get("title", "")))[:120],
        str(published),
    )


# -- HTTP session factory ------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    # Intentionally omit 'br' (Brotli): requests auto-decompresses gzip/deflate
    # but NOT Brotli. Advertising 'br' causes servers like dunya.com to send
    # Brotli-compressed bytes that arrive as unparseable binary XML.
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


# -- Date parsing -------------------------------------------------------------

_DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT",
    "%a, %d %b %Y %H:%M:%S +0000",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y",
]

# Formats that represent UTC time despite having no timezone object in the parsed result
_UTC_LITERAL_FORMATS = {
    "%a, %d %b %Y %H:%M:%S GMT",
    "%a, %d %b %Y %H:%M:%S +0000",
    "%Y-%m-%dT%H:%M:%SZ",
}
# Formats with no time component — hour is undefined
_DATE_ONLY_FORMATS = {"%Y-%m-%d", "%d.%m.%Y"}

# Turkey is UTC+3 year-round, but an IANA zone keeps the conversion contract
# explicit and ensures the publication date and hour cross midnight together.
_ISTANBUL = ZoneInfo("Europe/Istanbul")


def _parse_datetime(raw: str) -> tuple[Optional[datetime], bool]:
    """Return an Istanbul-local datetime and whether a time was supplied."""
    if not raw:
        return None, False
    raw = raw.strip()

    def _try(value: str):
        for fmt in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(value, fmt)
            except ValueError:
                continue
            has_time = fmt not in _DATE_ONLY_FORMATS
            if not has_time:
                return parsed.replace(tzinfo=_ISTANBUL), False
            if parsed.tzinfo is not None:
                return parsed.astimezone(_ISTANBUL), True
            if fmt in _UTC_LITERAL_FORMATS:
                return parsed.replace(tzinfo=timezone.utc).astimezone(_ISTANBUL), True
            return parsed.replace(tzinfo=_ISTANBUL), True
        return None

    parsed = _try(raw)
    if parsed is not None:
        return parsed
    cleaned = re.sub(r"\s+[A-Z]{2,5}$", "", raw).strip()
    if cleaned != raw:
        parsed = _try(cleaned)
        if parsed is not None:
            return parsed
    logger.debug("Could not parse date: %r", raw)
    return None, False


def _parse_date(raw: str) -> Optional[date]:
    parsed, _ = _parse_datetime(raw)
    return parsed.date() if parsed is not None else None


def _parse_hour(raw: str) -> Optional[int]:
    """Return the Istanbul-local hour, or ``None`` for date-only input."""
    parsed, has_time = _parse_datetime(raw)
    return parsed.hour if parsed is not None and has_time else None


def _parse_timestamp(raw: str) -> Optional[str]:
    """Return an Istanbul ISO timestamp, or ``None`` for date-only input."""
    parsed, has_time = _parse_datetime(raw)
    return parsed.isoformat() if parsed is not None and has_time else None


def _cdata_strip(text: str) -> str:
    """Remove XML CDATA wrappers if present."""
    return re.sub(r"<!\[CDATA\[|\]\]>", "", text).strip()


# -- RSS scraper ---------------------------------------------------------------

class RSSFeedScraper:
    """Fetches and parses RSS 2.0 / Atom feeds."""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or _make_session()
        # Populated by scrape_all(); maps each source to detailed included /
        # excluded / duplicate counts, or to "failed: <message>".
        self.source_status: Dict[str, str] = {}

    @staticmethod
    def _parse_xml(content: bytes) -> ET.Element:
        """
        Parse XML content robustly:
          1. Strip UTF-8 BOM (0xEF 0xBB 0xBF) - causes 'invalid token at line 1, col 0'
          2. Try common Turkish encodings on parse failure
        """
        # Strip UTF-8 BOM if present
        if content.startswith(b"\xef\xbb\xbf"):
            content = content[3:]

        try:
            return ET.fromstring(content)
        except ET.ParseError:
            # Some Turkish sites declare windows-1254 but serve mixed encodings
            for enc in ("utf-8", "iso-8859-9", "windows-1254", "latin-1"):
                try:
                    return ET.fromstring(content.decode(enc, errors="replace").encode("utf-8"))
                except (ET.ParseError, UnicodeDecodeError):
                    continue
            raise  # re-raise original so caller logs it

    def fetch(self, url: str, source_key: str) -> List[Dict]:
        headlines: List[Dict] = []
        try:
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            except requests.exceptions.SSLError:
                # Some Turkish news sites have incomplete certificate chains.
                # Retry without verification so the feed isn't silently lost.
                logger.warning("[RSS] SSL error for %s — retrying without verification", url)
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT, verify=False)
            resp.raise_for_status()
            # ElementTree handles both RSS and Atom
            root = self._parse_xml(resp.content)
            items = root.findall(".//item")  # RSS 2.0
            if not items:
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")  # Atom

            for item in items:
                # -- title -------------------------------------------------
                title = (
                    _cdata_strip(item.findtext("title") or "")
                    or _cdata_strip(
                        item.findtext("{http://www.w3.org/2005/Atom}title") or ""
                    )
                ).strip()
                if not title:
                    continue

                # -- url ---------------------------------------------------
                link = (
                    item.findtext("link")
                    or (item.find("{http://www.w3.org/2005/Atom}link") or {}).get("href", "")  # type: ignore[union-attr]
                    or ""
                ).strip()

                # -- date --------------------------------------------------
                raw_date = (
                    item.findtext("pubDate")
                    or item.findtext("{http://www.w3.org/2005/Atom}published")
                    or item.findtext("{http://www.w3.org/2005/Atom}updated")
                    or ""
                )
                pub_date = _parse_date(raw_date)
                pub_hour = _parse_hour(raw_date)
                pub_timestamp = _parse_timestamp(raw_date)
                source_item_id = (item.findtext("guid") or "").strip() or None

                headlines.append(
                    {
                        "title":          title,
                        "url":            link or None,
                        "published_at":   pub_date,
                        "published_hour": pub_hour,
                        "published_timestamp": pub_timestamp,
                        "source":         source_key,
                        "source_item_id": source_item_id,
                        "category":       classify_headline(title),
                        "raw_payload":    {"published_raw": raw_date},
                    }
                )

            logger.info("[RSS] %s -> %d items", source_key, len(headlines))
        except ET.ParseError as exc:
            logger.warning("[RSS] XML parse error for %s: %s", url, exc)
            raise   # re-raise so scrape_all can record the failure
        except requests.RequestException as exc:
            logger.warning("[RSS] Request failed for %s: %s", url, exc)
            raise   # re-raise so scrape_all can record the failure

        return headlines

    def scrape_all(
        self,
        feeds: Optional[Dict[str, str]] = None,
        since: Optional[date] = None,
    ) -> List[Dict]:
        feeds = feeds or RSS_FEEDS
        all_headlines: List[Dict] = []
        seen: set = set()

        total_fetched = 0
        total_skipped_date = 0
        total_excluded_relevance = 0
        total_skipped_dedup = 0
        total_included = 0

        self.source_status = {}

        for key, url in feeds.items():
            try:
                items = self.fetch(url, key)
            except Exception as exc:
                self.source_status[key] = f"failed: {exc}"
                logger.warning("[RSS] Source %s failed: %s", key, exc)
                time.sleep(CRAWL_DELAY)
                continue

            total_fetched += len(items)
            source_old = 0
            source_duplicates = 0
            source_excluded = 0
            source_included = 0
            for h in items:
                if since and h["published_at"] and h["published_at"] < since:
                    source_old += 1
                    total_skipped_date += 1
                    continue

                # Deduplicate only a stable identity within the same source.
                # Identical text from different outlets is a distinct observation.
                dedup_key = _observation_identity(h)
                if dedup_key in seen:
                    source_duplicates += 1
                    total_skipped_dedup += 1
                    continue
                seen.add(dedup_key)

                observation = _with_exclusion_metadata(h)
                if observation["is_excluded"]:
                    source_excluded += 1
                    total_excluded_relevance += 1
                    logger.debug(
                        "[filter] retained as excluded (%s): %s",
                        observation["exclusion_rule"],
                        observation["title"][:80],
                    )
                else:
                    source_included += 1
                    total_included += 1
                all_headlines.append(observation)

            source_returned = source_included + source_excluded
            self.source_status[key] = (
                f"ok (fetched={len(items)}, returned={source_returned}, "
                f"included={source_included}, excluded={source_excluded}, "
                f"too_old={source_old}, duplicates={source_duplicates})"
            )
            time.sleep(CRAWL_DELAY)

        failed = [k for k, v in self.source_status.items() if v.startswith("failed")]
        if failed:
            logger.warning("[RSS] %d/%d sources failed: %s", len(failed), len(feeds), failed)
        if len(failed) == len(feeds):
            logger.critical("[RSS] ALL sources failed - no headlines collected")

        logger.info(
            "RSS pipeline: %d fetched | %d too-old | %d same-source dedup | "
            "%d returned (%d included, %d excluded)",
            total_fetched,
            total_skipped_date,
            total_skipped_dedup,
            len(all_headlines),
            total_included,
            total_excluded_relevance,
        )
        return all_headlines


# -- HTML fallback scraper -----------------------------------------------------

class InvestingTRScraper:
    """
    HTML scraper for tr.investing.com/news/stock-market-news.
    Used only when RSS feeds are unavailable or blocked.
    """

    BASE = "https://tr.investing.com"

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or _make_session()

    def scrape(self, max_pages: int = 3) -> List[Dict]:
        headlines: List[Dict] = []
        seen: set = set()
        total_fetched = 0
        total_included = 0
        total_excluded = 0
        total_duplicates = 0

        for page in range(1, max_pages + 1):
            url = (
                f"{self.BASE}/news/stock-market-news"
                if page == 1
                else f"{self.BASE}/news/stock-market-news/{page}"
            )
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")

                # Try multiple selector patterns (Investing.com changes layout)
                articles = (
                    soup.select("article.js-article-item")
                    or soup.select("div.largeTitle article")
                    or soup.select(".newsSectionWrapper article")
                    or soup.select("li.js-stream-content")
                )

                if not articles:
                    logger.warning("[HTML] No articles found on page %d, stopping.", page)
                    break

                for art in articles:
                    title_el = (
                        art.select_one("a.title")
                        or art.select_one("h3 a")
                        or art.select_one("a[data-id]")
                        or art.select_one("a")
                    )
                    if not title_el:
                        continue

                    title = title_el.get_text(strip=True)
                    href = title_el.get("href", "")
                    link = (
                        href if href.startswith("http")
                        else (self.BASE + href if href else "")
                    )
                    total_fetched += 1

                    date_el = art.select_one("time") or art.select_one(".date")
                    raw_date = (
                        (date_el.get("datetime") or date_el.get_text(strip=True))
                        if date_el
                        else ""
                    )
                    pub_date = _parse_date(raw_date)
                    pub_hour = _parse_hour(raw_date)
                    pub_timestamp = _parse_timestamp(raw_date)

                    raw_observation = {
                        "title":        title,
                        "url":          link or None,
                        "published_at": pub_date,
                        "published_hour": pub_hour,
                        "published_timestamp": pub_timestamp,
                        "source":       "investing_tr_html",
                        "category":     classify_headline(title),
                        "raw_payload":  {"published_raw": raw_date},
                    }
                    dedup_key = _observation_identity(raw_observation)
                    if dedup_key in seen:
                        total_duplicates += 1
                        continue
                    seen.add(dedup_key)

                    observation = _with_exclusion_metadata(raw_observation)
                    if observation["is_excluded"]:
                        total_excluded += 1
                    else:
                        total_included += 1
                    headlines.append(observation)

                logger.info(
                    "[HTML] page %d -> %d observations so far "
                    "(%d included, %d excluded, %d same-source duplicates)",
                    page, len(headlines), total_included, total_excluded,
                    total_duplicates,
                )
                time.sleep(CRAWL_DELAY * 1.5)  # slightly slower for HTML pages

            except requests.RequestException as exc:
                logger.warning("[HTML] Request failed (page %d): %s", page, exc)
                break

        logger.info(
            "HTML pipeline: %d fetched | %d same-source dedup | "
            "%d returned (%d included, %d excluded)",
            total_fetched, total_duplicates, len(headlines), total_included,
            total_excluded,
        )
        return headlines


# -- Public entry point --------------------------------------------------------

def get_headlines(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    use_rss: bool = True,
    use_html_fallback: bool = True,
) -> List[Dict]:
    """
    Fetch headlines from all configured sources.

    Returns source-distinct headline observations filtered to the last
    ``lookback_days`` days (headlines with no date are always included).
    Relevance exclusions remain in the list with explicit metadata.
    """
    since = date.today() - timedelta(days=lookback_days)
    session = _make_session()
    headlines: List[Dict] = []

    if use_rss:
        rss = RSSFeedScraper(session)
        headlines = rss.scrape_all(since=since)

    if not headlines and use_html_fallback:
        logger.info("RSS returned nothing - falling back to HTML scraper")
        html = InvestingTRScraper(session)
        raw = html.scrape(max_pages=5)
        headlines = [
            h for h in raw
            if h["published_at"] is None or h["published_at"] >= since
        ]

    excluded = sum(1 for h in headlines if h.get("is_excluded"))
    logger.info(
        "Total observations returned: %d (%d included, %d excluded)",
        len(headlines), len(headlines) - excluded, excluded,
    )
    return headlines
