"""Detect Turkish headlines that merely report a price move already observed.

A recap carries no new information about the future: "BIST closed lower" tells
you what the market did, not what it learned. Mixing recaps into a directional
sentiment signal creates a reverse-causality trap -- the tone follows the return
by construction, so any apparent predictive relationship is the return
predicting itself.

Recaps are still worth keeping. They measure attention and are the right sample
for reverse-causality checks, so nothing is deleted or hidden: the flag simply
makes the distinction available.

This is deliberately a **rules** classifier, not an LLM category. Adding a
category to the scoring prompt would change the prompt version, which changes
the stored model identity, which splits the experiment and invalidates the
held-out validation. A separate versioned column costs none of that.

Detection shape
---------------
A recap needs BOTH a market subject (an index, a venue, a traded instrument or a
sector's shares) AND a movement predicate in a reporting frame (rose, fell,
closed, started the day, gained value). Requiring both is what keeps "the lira
weakened after the decision" -- a causal report of new information -- from
matching.

Then an exemption pass removes headlines that announce genuinely new
information even though they use market vocabulary: a new index, a listing, an
appointment, a regulatory decision, a company disclosure. Those are checked
after the positive match because they are the harder, rarer case, and an
explicit exemption is easier to audit than an ever-longer negative pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from taxonomy.signal_family import normalize

MARKET_RECAP_VERSION = "market-recap-rules-v1"

# Subjects whose movement can be recapped.
_MARKET_SUBJECTS: Tuple[str, ...] = (
    "borsa", "bist", "bist 100", "bist100", "xu100", "xu030", "endeks",
    "dolar", "euro", "avro", "sterlin", "kur", "doviz", "lira", "tl",
    "altin", "gram altin", "ons altin", "gumus", "petrol", "brent",
    "hisse", "hisseler", "hisse senedi", "hisse senetleri",
    "piyasa", "piyasalar", "borsalar", "faiz", "tahvil", "kripto", "bitcoin",
    "bankacilik endeksi", "sanayi endeksi",
)

# Movement predicates in a reporting frame.
_MOVEMENT_TERMS: Tuple[str, ...] = (
    "yukseldi", "yukselis", "dustu", "dusus", "geriledi", "gerileme",
    "arttı", "artti", "azaldi", "kazandi", "kaybetti", "deger kazandi",
    "deger kaybetti", "tirmandi", "cakildi", "sicradi", "zayifladi",
    "guclendi", "toparlandi", "sert dustu", "sert yukseldi",
    "rekor kirdi", "zirve gordu", "dip gordu",
    "kapandi", "kapatti", "tamamladi", "basladi", "acildi",
    "gunu", "gune", "haftayi", "haftaya", "seansi", "seans",
    "yatay seyretti", "karisik seyretti", "primli", "ekside", "artida",
)

# Phrases that are recaps on their own: the frame is unambiguous.
_STRONG_RECAP_PATTERNS: Tuple[str, ...] = (
    r"\bgune\s+\w*\s*(basladi|yukselisle|dususle|yatay)",
    r"\bgunu\s+\w*\s*(tamamladi|kapatti|yukselisle|dususle)",
    r"\bhaftayi\s+\w*\s*(tamamladi|kapatti)",
    r"\bkapanis(ta|ini)?\b",
    r"\bgun\s+icinde\b",
    r"\bseans\w*\s+(sonunda|boyunca)",
    r"\bguncel\s+(dolar|euro|altin)\s+(fiyat|kur)",
    r"\b(dolar|euro|altin|gumus)\s+ne\s+kadar\b",
)

# Genuinely new information that happens to use market vocabulary. Checked
# after a positive match, so an exemption always wins over a recap frame.
_NEW_INFORMATION_TERMS: Tuple[str, ...] = (
    # market structure
    "yeni endeks", "endeks olusturul", "endekse dahil", "endeksten cikar",
    "endeks kural", "yeni pazar", "pazar degisik",
    # listings and issuance
    "halka arz", "halka arzi", "borsada islem gormeye", "islem gormeye baslad",
    "ihrac", "tahvil ihrac", "sermaye artirim", "bedelli", "bedelsiz",
    # governance
    "atandi", "atama", "gorevden al", "istifa", "yonetim kurulu",
    "genel mudur", "ceo", "baskan sec",
    # regulation and policy decisions
    "spk", "bddk", "duzenleme", "karar aldi", "yonetmelik", "teblig",
    "sorusturma", "inceleme baslat", "para cezasi", "islem yasagi",
    "faiz karari", "politika faizi", "ppk",
    # company disclosures
    "ozel durum", "kap aciklama", "bilanco", "finansal sonuc", "temettu",
    "satin alma", "birlesme", "ihale", "sozlesme imzala", "anlasma imzala",
    "yatirim karari", "tesis", "fabrika",
    # forward-looking commentary is analysis, not a recap of what happened
    "bekleniyor", "beklentisi", "tahmin", "hedef fiyat", "onerisi",
    "olabilir", "projeksiyon",
)

_STRONG_RECAP_RE = tuple(re.compile(pattern) for pattern in _STRONG_RECAP_PATTERNS)


@dataclass(frozen=True)
class RecapClassification:
    """Whether a headline is a market recap, and why."""

    is_market_recap: bool
    version: str
    rule: str
    evidence: Optional[str]
    confidence: float

    @property
    def as_int(self) -> int:
        return 1 if self.is_market_recap else 0


def _first_match(text: str, terms: Sequence[str]) -> Optional[str]:
    for term in terms:
        if term in text:
            return term
    return None


def classify_market_recap(title: Any) -> RecapClassification:
    """Classify one headline.

    Confidence reports which rule fired, not a calibrated probability: 0.9 for
    an unambiguous recap frame, 0.7 for subject-plus-movement, 0.0 when an
    exemption or a missing element rules it out.
    """

    text = normalize(title)
    if not text:
        return RecapClassification(False, MARKET_RECAP_VERSION, "empty_title", None, 0.0)

    exemption = _first_match(text, _NEW_INFORMATION_TERMS)

    strong = next(
        (regex.pattern for regex in _STRONG_RECAP_RE if regex.search(text)), None
    )
    subject = _first_match(text, _MARKET_SUBJECTS)
    movement = _first_match(text, _MOVEMENT_TERMS)

    if strong is not None and subject is not None:
        if exemption is not None:
            return RecapClassification(
                False, MARKET_RECAP_VERSION, "exempt_new_information",
                f"{exemption} (over recap frame)", 0.0,
            )
        return RecapClassification(
            True, MARKET_RECAP_VERSION, "strong_recap_frame",
            f"{subject}+{strong}", 0.9,
        )

    if subject is not None and movement is not None:
        if exemption is not None:
            return RecapClassification(
                False, MARKET_RECAP_VERSION, "exempt_new_information",
                f"{exemption} (over {subject}+{movement})", 0.0,
            )
        return RecapClassification(
            True, MARKET_RECAP_VERSION, "subject_and_movement",
            f"{subject}+{movement}", 0.7,
        )

    missing = "no_market_subject" if subject is None else "no_movement_predicate"
    return RecapClassification(False, MARKET_RECAP_VERSION, missing, subject, 0.0)
