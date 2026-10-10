"""News-to-ticker linking that prefers a missed link to a wrong one.

A wrong ticker puts another company's return into an event study, and no
amount of clustering repairs that. So evidence is ranked, every link stores
what it matched and why, and anything below the confirmation threshold goes to
a review queue instead of into a sample.

Evidence, strongest first
-------------------------
1. **Issuer identifier** from the source itself (KAP sender). Not handled
   here: a KAP disclosure never needs text matching.
2. **Ticker token** written in capitals in mixed-case text (``THYAO``,
   ``(ASELS)``).
3. **Unambiguous alias**: a name that means only the company (``turkcell``,
   ``tupras``).
4. **Ambiguous alias with a corporate cue**: a name that is also an ordinary
   word or a family name (``sok``, ``koc``, ``is``), accepted only next to a
   word that makes it a company (``sok market``, ``koc holding``).
5. An ambiguous alias with no cue is **not a link**. It is queued for review.

What a link is not
------------------
A link says the text names the company. Whether the text is a material,
company-specific announcement is a separate judgement, ``mention_type``, and
only ``material`` single-issuer mentions are admissible as events.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LINKER_VERSION = "entity-linker-v1"
CONFIRM_THRESHOLD = 0.80

METHOD_TICKER = "ticker_token"
METHOD_ALIAS = "alias"
METHOD_ALIAS_CUE = "ambiguous_alias_with_cue"
METHOD_ALIAS_BARE = "ambiguous_alias_no_cue"

MENTION_MATERIAL = "material"
MENTION_CASUAL = "casual"
MENTION_SECTOR = "sector"
MENTION_MULTI = "multi_issuer"

_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c", "â": "a",
    "î": "i", "û": "u", "’": "'", "‘": "'",
})


def fold(text: Any) -> str:
    """Lower-case ASCII form used for matching. ``I`` folds to ``i`` so that
    Turkish capitals and English capitals compare equal."""

    folded = str(text or "").translate(_FOLD).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&+'.]+", " ", folded)).strip()


# ticker -> (unambiguous aliases, ambiguous aliases, cue words for the ambiguous ones)
# Folded forms. Hand-checked; an alias belongs in the second tuple if a Turkish
# reader could meet the word in a headline that is not about the company.
CURATED: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]] = {
    "THYAO": (("turk hava yollari", "thy"), (), ()),
    "PGSUS": (("pegasus hava", "pegasus havayollari"), ("pegasus",), ("hisse", "yolcu", "ucak", "ucus")),
    "GARAN": (("garanti bbva", "garanti bankasi"), ("garanti",), ("bbva", "bankasi", "banka", "hisse")),
    "AKBNK": (("akbank",), (), ()),
    "ISCTR": (("is bankasi", "isbank", "turkiye is bankasi"), (), ()),
    "YKBNK": (("yapi kredi", "yapi ve kredi bankasi"), (), ()),
    "VAKBN": (("vakifbank", "vakiflar bankasi"), (), ()),
    "HALKB": (("halkbank", "halk bankasi"), (), ()),
    "TSKB": (("tskb", "turkiye sinai kalkinma bankasi"), (), ()),
    "SKBNK": (("sekerbank",), (), ()),
    "ALBRK": (("albaraka turk",), ("albaraka",), ("katilim", "banka", "hisse")),
    "ASELS": (("aselsan",), (), ()),
    "EREGL": (("erdemir", "eregli demir"), (), ()),
    "KRDMD": (("kardemir",), (), ()),
    "TUPRS": (("tupras",), (), ()),
    "PETKM": (("petkim",), (), ()),
    "KCHOL": (("koc holding",), ("koc",), ("holding", "grubu", "toplulugu")),
    "SAHOL": (("sabanci holding",), ("sabanci",), ("holding", "grubu", "toplulugu")),
    "DOHOL": (("dogan holding",), ("dogan",), ("holding", "sirketler grubu")),
    "ALARK": (("alarko holding", "alarko"), (), ()),
    "TKFEN": (("tekfen holding", "tekfen"), (), ()),
    "ENKAI": (("enka insaat",), ("enka",), ("insaat", "hisse", "holding")),
    "BIMAS": (("bim birlesik magazalar", "bim magazalar"), ("bim",), ("market", "magaza", "hisse", "indirim")),
    "MGROS": (("migros",), (), ()),
    "SOKM": (("sok marketler",), ("sok",), ("market", "magaza")),
    "SISE": (("sisecam",), (), ()),
    "TCELL": (("turkcell",), (), ()),
    "TTKOM": (("turk telekom",), (), ()),
    "FROTO": (("ford otosan",), (), ()),
    "TOASO": (("tofas",), (), ()),
    "DOAS": (("dogus otomotiv",), (), ()),
    "OTKAR": (("otokar",), (), ()),
    "TTRAK": (("turk traktor",), (), ()),
    "ARCLK": (("arcelik",), (), ()),
    "VESTL": (("vestel",), (), ()),
    "SASA": (("sasa polyester",), ("sasa",), ("polyester", "hisse", "yatirim")),
    "KOZAL": (("koza altin",), (), ()),
    "EKGYO": (("emlak konut",), (), ()),
    "ULKER": (("ulker biskuvi",), ("ulker",), ("biskuvi", "hisse", "gida")),
    "TAVHL": (("tav havalimanlari",), ("tav",), ("havalimani", "havalimanlari", "hisse")),
    "AEFES": (("anadolu efes",), (), ()),
    "CCOLA": (("coca cola icecek", "coca-cola icecek"), (), ()),
    "HEKTS": (("hektas",), (), ()),
    "GUBRF": (("gubre fabrikalari", "gubretas"), (), ()),
    "ZOREN": (("zorlu enerji",), (), ()),
    "AKSEN": (("aksa enerji",), (), ()),
    "ENJSA": (("enerjisa",), (), ()),
    "ODAS": (("odas elektrik", "odas enerji"), (), ()),
    "MAVI": (("mavi giyim",), ("mavi",), ("giyim", "jeans", "magaza")),
    "LOGO": (("logo yazilim",), ("logo",), ("yazilim",)),
    "CIMSA": (("cimsa",), (), ()),
    "AKSA": (("aksa akrilik",), (), ()),
    "BRISA": (("brisa",), (), ()),
    "KONTR": (("kontrolmatik",), (), ()),
    "ASTOR": (("astor enerji",), ("astor",), ("enerji", "hisse", "trafo")),
}

_MATERIAL_CUES = (
    "kap'a", "kap a bildir", "kap aciklama", "bildirdi", "acikladi", "duyurdu",
    "bilanco", "net kar", "net zarar", "kar acikladi", "zarar acikladi", "ciro",
    "temettu", "kar payi", "bedelsiz", "bedelli", "sermaye artirim", "geri alim",
    "pay geri", "ihale", "sozlesme", "anlasma imzala", "siparis", "satin al",
    "devral", "birlesme", "halka arz", "yatirim karari", "fabrika", "tesis",
    "para cezasi", "sorusturma", "konkordato", "iflas", "ortaklik", "hisse devri",
    "kredi notu", "tahvil ihrac", "genel mudur", "istifa", "atandi",
)
_STOCK_CUES = ("hisse", "hisseleri", "hissesi", "paylari", "senedi", "hedef fiyat", "kap")
_SECTOR_CUES = (
    "sektoru", "sektorunde", "sektorde", "sektor", "bankalar", "banka hisseleri",
    "bankacilik endeksi", "holdingler", "havayollari sirketleri", "perakendeciler",
    "otomotiv ureticileri", "enerji sirketleri", "gyo'lar", "endeksi",
)


@dataclass(frozen=True)
class Link:
    ticker: str
    method: str
    confidence: float
    matched: str
    evidence: str
    confirmed: bool


@dataclass
class LinkResult:
    links: List[Link] = field(default_factory=list)        # confirmed
    review: List[Link] = field(default_factory=list)       # not confirmed
    mention_type: Optional[str] = None
    material_cues: Tuple[str, ...] = ()

    @property
    def tickers(self) -> List[str]:
        return sorted({link.ticker for link in self.links})


def _word(text: str, phrase: str) -> Optional[int]:
    match = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text)
    return match.start() if match else None


def _has_cue(text: str, position: int, alias: str, cues: Sequence[str], window: int = 28) -> Optional[str]:
    lo, hi = max(0, position - window), position + len(alias) + window
    nearby = text[lo:hi]
    for cue in cues:
        if _word(nearby, cue) is not None:
            return cue
    return None


class Linker:
    """Links text to tickers using curated aliases and, optionally, the tickers
    of a member roster (for the capitalised-token rule only)."""

    def __init__(self, known_tickers: Optional[Iterable[str]] = None,
                 curated: Optional[Dict[str, tuple]] = None):
        self.curated = dict(curated if curated is not None else CURATED)
        self.tickers = {t.upper() for t in (known_tickers or [])} | set(self.curated)

    def link(self, text: Any) -> LinkResult:
        original = str(text or "")
        folded = fold(original)
        result = LinkResult()
        found: Dict[str, Link] = {}
        queued: Dict[str, Link] = {}

        # 2. Ticker tokens, only where capitals carry information.
        letters = [c for c in original if c.isalpha()]
        mixed_case = bool(letters) and sum(c.isupper() for c in letters) / len(letters) < 0.6
        if mixed_case:
            for token in re.findall(r"(?<![A-Za-z0-9ÇĞİÖŞÜçğıöşü])[A-Z]{4,5}(?![A-Za-z0-9ÇĞİÖŞÜçğıöşü])", original):
                if token not in self.tickers:
                    continue
                if token in self.curated:
                    found.setdefault(token, Link(
                        token, METHOD_TICKER, 0.97, token,
                        "capitalised ticker token in mixed-case text", True))
                    continue
                # A roster code can be an ordinary word in capitals (ALTIN,
                # METRO). Outside the curated list it needs to look like a
                # ticker: in brackets, or beside a word about shares.
                bracketed = f"({token})" in original
                position = _word(folded, token.lower())
                cue = (_has_cue(folded, position, token.lower(), _STOCK_CUES)
                       if position is not None else None)
                if bracketed or cue:
                    found.setdefault(token, Link(
                        token, METHOD_TICKER, 0.90, token,
                        "roster ticker " + ("in brackets" if bracketed else f"beside '{cue}'"),
                        True))
                else:
                    queued.setdefault(token, Link(
                        token, METHOD_TICKER, 0.50, token,
                        "roster ticker token with no share cue", False))

        for ticker, (plain, ambiguous, cues) in self.curated.items():
            for alias in plain:
                if _word(folded, alias) is not None:
                    found.setdefault(ticker, Link(
                        ticker, METHOD_ALIAS, 0.93, alias, "unambiguous alias", True))
                    break
            if ticker in found:
                continue
            for alias in ambiguous:
                position = _word(folded, alias)
                if position is None:
                    continue
                cue = _has_cue(folded, position, alias, cues)
                if cue:
                    found[ticker] = Link(ticker, METHOD_ALIAS_CUE, 0.85, alias,
                                         f"ambiguous alias next to cue '{cue}'", True)
                else:
                    queued[ticker] = Link(ticker, METHOD_ALIAS_BARE, 0.40, alias,
                                          "ambiguous alias with no corporate cue", False)
                break

        result.links = sorted(found.values(), key=lambda l: l.ticker)
        result.review = sorted((l for t, l in queued.items() if t not in found),
                               key=lambda l: l.ticker)
        result.material_cues = tuple(c for c in _MATERIAL_CUES if _word(folded, c) is not None)
        sector = any(_word(folded, c) is not None for c in _SECTOR_CUES)

        if not result.links:
            result.mention_type = MENTION_SECTOR if sector else None
        elif len(result.links) > 1:
            result.mention_type = MENTION_MULTI
        elif sector and not result.material_cues:
            result.mention_type = MENTION_SECTOR
        elif result.material_cues:
            result.mention_type = MENTION_MATERIAL
        else:
            result.mention_type = MENTION_CASUAL
        return result


def evaluate(examples: Sequence[Dict[str, Any]], linker: Optional[Linker] = None) -> Dict[str, Any]:
    """Score the linker on labelled examples.

    Each example has ``text`` and ``tickers`` (the gold set, possibly empty).
    Counts are per (example, ticker) pair. ``false_assignments`` is the number
    of confirmed links to a ticker the example is not about -- the error that
    matters most -- and ``unresolved_rate`` is the share of gold pairs the
    linker neither confirmed nor queued.
    """

    linker = linker or Linker()
    tp = fp = fn = queued_gold = 0
    errors: List[Dict[str, Any]] = []
    for example in examples:
        gold = set(example["tickers"])
        outcome = linker.link(example["text"])
        predicted = set(outcome.tickers)
        review = {link.ticker for link in outcome.review}
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        queued_gold += len((gold - predicted) & review)
        if predicted != gold:
            errors.append({"text": example["text"], "gold": sorted(gold),
                           "predicted": sorted(predicted), "review": sorted(review)})
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else None)
    gold_pairs = tp + fn
    return {
        "linker_version": LINKER_VERSION, "examples": len(examples),
        "gold_pairs": gold_pairs, "true_positive": tp, "false_assignments": fp,
        "missed": fn, "precision": precision, "recall": recall, "f1": f1,
        "unresolved_rate": ((fn - queued_gold) / gold_pairs) if gold_pairs else None,
        "queued_for_review": queued_gold, "errors": errors,
    }
