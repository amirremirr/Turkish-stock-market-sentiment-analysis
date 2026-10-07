"""Atom feeds carry the article URL in <link href=...>, not as element text."""

from types import SimpleNamespace

from scraper import RSSFeedScraper

ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ntv.com.tr</title>
  <entry>
    <id>https://www.ntv.com.tr/ekonomi/ornek-haber-1</id>
    <title type="text">Merkez Bankasi faiz kararini acikladi</title>
    <link rel="alternate" href="https://www.ntv.com.tr/ekonomi/ornek-haber-1" />
    <published>2026-10-05T10:15:00+03:00</published>
  </entry>
</feed>"""


class _Session:
    def get(self, url, timeout=None, verify=True):
        return SimpleNamespace(content=ATOM, raise_for_status=lambda: None)


def test_atom_entry_keeps_its_href():
    items = RSSFeedScraper(session=_Session()).fetch("https://x/rss", "ntv_ekonomi")

    assert len(items) == 1
    assert items[0]["url"] == "https://www.ntv.com.tr/ekonomi/ornek-haber-1"
