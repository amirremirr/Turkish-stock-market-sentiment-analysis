"""Normalized entity and event-type extraction from Turkish headlines.

Entities are the strongest available grouping key: two outlets covering the
same TCMB decision will disagree on wording but both name TCMB. Extraction is a
dictionary lookup over known institutions, listed issuers and macro concepts --
not named-entity recognition. That is a deliberate limit: a curated list is
auditable and deterministic, and an entity it misses produces a smaller group
rather than a wrong one.

Event types are assigned from the same transparent vocabulary. They describe
what kind of announcement a headline appears to be, not what happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from taxonomy.signal_family import normalize

ENTITY_RULE_VERSION = "entity-rules-v1"
EVENT_TYPE_RULE_VERSION = "event-type-rules-v1"

# entity_id -> (entity_type, surface forms). The id is canonical and stable;
# surface forms are what actually appears in Turkish headlines.
ENTITY_DICTIONARY: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    # institutions
    "TCMB": ("institution", ("tcmb", "merkez bankasi", "merkez bankasinin")),
    "TUIK": ("institution", ("tuik", "turkiye istatistik kurumu")),
    "BDDK": ("institution", ("bddk", "bankacilik duzenleme")),
    "SPK": ("institution", ("spk", "sermaye piyasasi kurulu")),
    "HAZINE": ("institution", ("hazine", "hazine ve maliye")),
    "BORSA_ISTANBUL": ("institution", ("borsa istanbul", "bist", "bist 100", "xu100")),
    "TMSF": ("institution", ("tmsf",)),
    "FED": ("institution", ("fed", "federal reserve", "fomc")),
    "ECB": ("institution", ("ecb", "avrupa merkez bankasi")),
    "IMF": ("institution", ("imf",)),
    "NATO": ("institution", ("nato",)),
    "OPEC": ("institution", ("opec", "opec+")),
    # rating agencies
    "MOODYS": ("agency", ("moodys", "moody's")),
    "FITCH": ("agency", ("fitch",)),
    "SP_GLOBAL": ("agency", ("s&p", "standard & poor")),
    # listed issuers (major BIST constituents)
    "THYAO": ("issuer", ("thy", "turk hava yollari", "thyao")),
    "GARAN": ("issuer", ("garanti", "garanti bbva", "garan")),
    "AKBNK": ("issuer", ("akbank", "akbnk")),
    "ISCTR": ("issuer", ("is bankasi", "isbank", "isctr")),
    "YKBNK": ("issuer", ("yapi kredi", "yapikredi", "ykbnk")),
    "VAKBN": ("issuer", ("vakifbank", "vakbn")),
    "HALKB": ("issuer", ("halkbank", "halkb")),
    "ZIRAAT": ("issuer", ("ziraat bankasi", "ziraat")),
    "ASELS": ("issuer", ("aselsan", "asels")),
    "EREGL": ("issuer", ("eregli", "erdemir", "eregl")),
    "TUPRS": ("issuer", ("tupras", "tuprs")),
    "KCHOL": ("issuer", ("koc holding", "kchol")),
    "SAHOL": ("issuer", ("sabanci holding", "sahol")),
    "BIMAS": ("issuer", ("bim ", "bimas")),
    "SISE": ("issuer", ("sisecam", "sise cam")),
    "PGSUS": ("issuer", ("pegasus", "pgsus")),
    "TCELL": ("issuer", ("turkcell", "tcell")),
    "FROTO": ("issuer", ("ford otosan", "froto")),
    "TOASO": ("issuer", ("tofas", "toaso")),
    # macro concepts
    "ENFLASYON": ("macro", ("enflasyon", "tufe", "ufe")),
    "FAIZ": ("macro", ("politika faizi", "faiz karari", "ppk")),
    "ISSIZLIK": ("macro", ("issizlik", "istihdam")),
    "BUYUME": ("macro", ("buyume", "gsyh")),
    "CARI_ACIK": ("macro", ("cari acik", "cari denge")),
    "BUTCE": ("macro", ("butce",)),
    "DIS_TICARET": ("macro", ("ihracat", "ithalat", "dis ticaret")),
    # instruments
    "USDTRY": ("instrument", ("dolar/tl", "dolar kuru", "usd/try", "dolar")),
    "EUR": ("instrument", ("euro", "avro")),
    "ALTIN": ("instrument", ("altin", "gram altin", "ons altin")),
    "PETROL": ("instrument", ("petrol", "brent")),
    "DOGALGAZ": ("instrument", ("dogalgaz",)),
}

# event_type -> ordered markers. First match wins, so the most specific
# vocabulary is listed first.
EVENT_TYPE_RULES: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("rate_decision", ("politika faizi", "faiz karari", "ppk", "faiz kararini")),
    ("data_release", ("enflasyon", "tufe", "ufe", "issizlik", "buyume", "gsyh",
                      "cari acik", "butce", "veri acikland", "aciklandi")),
    ("rating_action", ("kredi notu", "not artirimi", "not indirimi", "gorunum")),
    ("earnings", ("bilanco", "finansal sonuc", "net kar", "ceyrek kar",
                  "kar acikladi", "zarar acikladi")),
    ("corporate_action", ("temettu", "kar payi", "sermaye artirim", "bedelli",
                          "bedelsiz", "halka arz", "geri alim")),
    ("m_and_a", ("satin al", "devralma", "birlesme", "hisse devri")),
    ("regulatory_action", ("duzenleme", "yonetmelik", "teblig", "karar aldi",
                           "sorusturma", "para cezasi", "islem yasagi")),
    ("appointment", ("atandi", "atama", "gorevden al", "istifa", "secildi")),
    ("guidance", ("bekleniyor", "beklentisi", "tahmin", "projeksiyon",
                  "hedef fiyat")),
    ("market_move", ("yukseldi", "dustu", "geriledi", "kapandi", "tamamladi",
                     "deger kazandi", "deger kaybetti")),
    ("geopolitical", ("savas", "yaptirim", "jeopolitik", "askeri", "saldiri")),
)


@dataclass(frozen=True)
class EntityExtraction:
    """Entities found in one headline."""

    entities: FrozenSet[Tuple[str, str]]      # (entity_type, entity_id)
    primary_entity: Optional[str]
    rule_version: str

    @property
    def entity_ids(self) -> FrozenSet[str]:
        return frozenset(entity_id for _, entity_id in self.entities)


def _word_bounded(text: str, surface: str) -> bool:
    """Match a surface form on token boundaries where the form is a word.

    A bare substring test would let 'bist' match 'bistro'. Forms containing a
    space or punctuation are matched directly, since they are already specific.
    """
    if not surface.isalnum():
        return surface in text
    return re.search(rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])", text) is not None


def extract_entities(title: Any) -> EntityExtraction:
    """Extract normalized entities from a headline.

    The primary entity prefers a listed issuer, then an institution, then a
    macro concept, then an instrument -- most specific first, because a headline
    naming both an issuer and an index is about the issuer.
    """

    text = normalize(title)
    if not text:
        return EntityExtraction(frozenset(), None, ENTITY_RULE_VERSION)

    found: List[Tuple[str, str]] = []
    for entity_id, (entity_type, surfaces) in ENTITY_DICTIONARY.items():
        for surface in surfaces:
            if _word_bounded(text, surface):
                found.append((entity_type, entity_id))
                break

    priority = {"issuer": 0, "institution": 1, "agency": 1, "macro": 2, "instrument": 3}
    primary = None
    if found:
        primary = sorted(found, key=lambda item: (priority.get(item[0], 9), item[1]))[0][1]
    return EntityExtraction(frozenset(found), primary, ENTITY_RULE_VERSION)


def classify_event_type(title: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(event_type, matched_marker)`` for a headline.

    An unmatched headline returns ``(None, None)`` rather than a catch-all
    label: an unknown type is information, a fabricated one is not.
    """

    text = normalize(title)
    if not text:
        return (None, None)
    for event_type, markers in EVENT_TYPE_RULES:
        for marker in markers:
            if marker in text:
                return (event_type, marker)
    return (None, None)
