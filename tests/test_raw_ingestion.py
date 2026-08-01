"""Raw-ingestion regression tests with no live network access."""

from __future__ import annotations

from dataclasses import dataclass

import scraper as scraper_module
from scraper import InvestingTRScraper, RSSFeedScraper


@dataclass
class _FakeResponse:
    content: bytes = b""
    text: str = ""

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses[url]


def _rss(*items: tuple[str, str, str]) -> bytes:
    body = []
    for title, link, published in items:
        body.append(
            "<item>"
            f"<title>{title}</title>"
            f"<link>{link}</link>"
            f"<pubDate>{published}</pubDate>"
            "</item>"
        )
    return ("<rss><channel>" + "".join(body) + "</channel></rss>").encode()


def test_rss_returns_off_topic_observations_with_exclusion_metadata(monkeypatch):
    url = "https://example.test/feed"
    session = _FakeSession({
        url: _FakeResponse(content=_rss(
            (
                "Bitcoin yeni bir rekora ulasti",
                "https://example.test/bitcoin",
                "Wed, 10 Jun 2026 08:00:00 +0300",
            ),
            (
                "BIST 100 endeksi yukselisle kapandi",
                "https://example.test/bist",
                "Wed, 10 Jun 2026 09:00:00 +0300",
            ),
        )),
    })
    monkeypatch.setattr(scraper_module.time, "sleep", lambda _seconds: None)

    collector = RSSFeedScraper(session)
    observations = collector.scrape_all(feeds={"outlet_a": url})

    assert len(observations) == 2
    excluded = next(row for row in observations if "Bitcoin" in row["title"])
    included = next(row for row in observations if "BIST" in row["title"])

    assert excluded["is_excluded"] is True
    assert excluded["exclusion_reason"] == "off_topic"
    assert excluded["exclusion_rule"] == "soft_blocklist"
    assert excluded["exclusion_version"]
    assert included["is_excluded"] is False
    assert included["exclusion_reason"] is None
    assert included["exclusion_rule"] is None
    assert included["exclusion_version"] is None

    # Existing caller-facing keys remain present on both observations.
    for row in observations:
        expected_keys = {
            "title", "url", "published_at", "published_hour", "source", "category",
        }
        assert expected_keys <= row.keys()

    status = collector.source_status["outlet_a"]
    assert "returned=2" in status
    assert "included=1" in status
    assert "excluded=1" in status


def test_rss_preserves_identical_headlines_from_distinct_sources(monkeypatch):
    title = "BIST 100 endeksi yukselisle kapandi"
    published = "Wed, 10 Jun 2026 09:00:00 +0300"
    url_a = "https://outlet-a.test/feed"
    url_b = "https://outlet-b.test/feed"
    shared_article_url = "https://wire.test/shared-story"
    session = _FakeSession({
        url_a: _FakeResponse(content=_rss((title, shared_article_url, published))),
        url_b: _FakeResponse(content=_rss((title, shared_article_url, published))),
    })
    monkeypatch.setattr(scraper_module.time, "sleep", lambda _seconds: None)

    observations = RSSFeedScraper(session).scrape_all(
        feeds={"outlet_a": url_a, "outlet_b": url_b},
    )

    assert len(observations) == 2
    assert {row["source"] for row in observations} == {"outlet_a", "outlet_b"}


def test_rss_deduplicates_stable_identity_within_one_source(monkeypatch):
    feed_url = "https://outlet.test/feed"
    article_url = "https://outlet.test/story/1"
    item = (
        "BIST 100 endeksi yukselisle kapandi",
        article_url,
        "Wed, 10 Jun 2026 09:00:00 +0300",
    )
    session = _FakeSession({feed_url: _FakeResponse(content=_rss(item, item))})
    monkeypatch.setattr(scraper_module.time, "sleep", lambda _seconds: None)

    collector = RSSFeedScraper(session)
    observations = collector.scrape_all(feeds={"outlet": feed_url})

    assert len(observations) == 1
    assert "duplicates=1" in collector.source_status["outlet"]


def test_html_returns_off_topic_observation_marked_excluded(monkeypatch):
    page_url = InvestingTRScraper.BASE + "/news/stock-market-news"
    html = """
    <html><body>
      <article class="js-article-item">
        <a class="title" href="/news/bitcoin">Bitcoin yeni bir rekora ulasti</a>
        <time datetime="2026-06-10T08:00:00+03:00"></time>
      </article>
      <article class="js-article-item">
        <a class="title" href="/news/bist">BIST 100 endeksi yukselisle kapandi</a>
        <time datetime="2026-06-10T09:00:00+03:00"></time>
      </article>
      <article class="js-article-item">
        <a class="title" href="/news/bist">BIST 100 endeksi yukselisle kapandi</a>
        <time datetime="2026-06-10T09:00:00+03:00"></time>
      </article>
    </body></html>
    """
    session = _FakeSession({page_url: _FakeResponse(text=html)})
    monkeypatch.setattr(scraper_module.time, "sleep", lambda _seconds: None)

    observations = InvestingTRScraper(session).scrape(max_pages=1)

    assert len(observations) == 2
    excluded = next(row for row in observations if "Bitcoin" in row["title"])
    included = next(row for row in observations if "BIST" in row["title"])
    assert excluded["is_excluded"] is True
    assert excluded["exclusion_rule"] == "soft_blocklist"
    assert included["is_excluded"] is False
