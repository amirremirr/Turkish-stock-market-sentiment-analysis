"""Map frozen detailed categories onto economically distinct signal families.

The detailed ``category`` field is a measurement input: it was assigned by a
specific prompt version and any edit would silently redefine the historical
record. This layer never touches it. Families are *derived* from that frozen
category plus transparent headline rules, stamped with their own version, so
the mapping can be revised without disturbing anything underneath.

Why families at all: the existing categories are topical, not economic. A rate
decision and a company earnings release move a portfolio through different
channels, and averaging them into one number discards exactly the structure a
reader needs. The families below are chosen to separate those channels.

Two categories cannot be resolved from the category alone:

``banks``
    The approved boundary is **entity specificity, not industry**. Sector- and
    system-level banking news (credit growth, deposit rates, BDDK regulation,
    liquidity, sector profitability) is ``banking_financial_sector``; a named
    listed bank's earnings, dividend, acquisition or KAP disclosure is
    ``company_kap``, the same as any other named issuer. Because the rule keys
    on whether a specific listed entity is named, it applies uniformly across
    sectors instead of special-casing banks.

``bist_company``
    Splits between ``company_kap`` (a named issuer's disclosure) and
    ``market_recap`` (a summary of a price move that already happened).

Anything the rules cannot resolve confidently is reported as ambiguous rather
than forced into a family. ``other`` is a genuine bucket, not a dumping ground:
a headline lands there because no family applies, and the coverage report exists
to make that visible.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

SIGNAL_FAMILY_VERSION = "signal-family-v1"

MONETARY_POLICY = "monetary_policy"
INFLATION_MACRO = "inflation_macro"
POLITICAL_REGULATORY_RISK = "political_regulatory_risk"
FX_LIRA = "fx_lira"
BANKING_FINANCIAL_SECTOR = "banking_financial_sector"
COMPANY_KAP = "company_kap"
GLOBAL_RISK = "global_risk"
MARKET_RECAP = "market_recap"
MEDIA_NARRATIVE = "media_narrative"
OTHER = "other"

SIGNAL_FAMILIES: Tuple[str, ...] = (
    MONETARY_POLICY,
    INFLATION_MACRO,
    POLITICAL_REGULATORY_RISK,
    FX_LIRA,
    BANKING_FINANCIAL_SECTOR,
    COMPANY_KAP,
    GLOBAL_RISK,
    MARKET_RECAP,
    MEDIA_NARRATIVE,
    OTHER,
)

# Families whose sentiment describes Turkish domestic conditions. global_risk is
# excluded from the domestic aggregate: a Fed decision is real information, but
# folding it into a Turkish domestic tone series conflates two different
# economies. market_recap is excluded because it reports moves that already
# happened rather than describing conditions.
DOMESTIC_FAMILIES: Tuple[str, ...] = (
    MONETARY_POLICY,
    INFLATION_MACRO,
    POLITICAL_REGULATORY_RISK,
    FX_LIRA,
    BANKING_FINANCIAL_SECTOR,
    COMPANY_KAP,
)

# Categories that map to exactly one family with no headline inspection needed.
DIRECT_CATEGORY_MAP: Mapping[str, str] = {
    "rates_tcmb": MONETARY_POLICY,
    "turkey_macro": INFLATION_MACRO,
    "political_risk": POLITICAL_REGULATORY_RISK,
    "fx_lira": FX_LIRA,
    "global_risk": GLOBAL_RISK,
    "energy_commodities": GLOBAL_RISK,
    "crypto": OTHER,
    "other": OTHER,
}

# Categories needing a headline rule to choose between families.
RESOLVED_CATEGORIES: Tuple[str, ...] = ("banks", "bist_company")


def normalize(text: Any) -> str:
    """ASCII-fold and lowercase, matching the scraper's keyword convention."""

    if text is None:
        return ""
    folded = unicodedata.normalize("NFKD", str(text))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    replacements = {"ı": "i", "İ": "i", "ş": "s", "ğ": "g", "ç": "c", "ö": "o", "ü": "u"}
    for source, target in replacements.items():
        folded = folded.replace(source, target)
    return " ".join(folded.lower().split())


# -- Banking: sector/system level vs a named issuer -----------------------------

# System-level vocabulary. These describe conditions across the banking system
# rather than one bank's own business.
_SECTOR_BANKING_TERMS: Tuple[str, ...] = (
    "bankacilik sektor", "bankacilik sektoru", "sektorun", "sektorde",
    "kredi buyume", "kredi buyumesi", "kredi hacmi", "kredi musluk",
    "mevduat faiz", "mevduat oran", "mevduat maliyet",
    "bddk", "tmsf", "tasarruf mevduati sigorta",
    "zorunlu karsilik", "sermaye yeterlilik", "likidite",
    "takipteki alacak", "kredi riski", "aktif kalitesi",
    "bankalarin karlilig", "sektor karlilig", "bankacilik karlilig",
    "kredi kart", "ticari kredi", "tuketici kredi", "konut kredi",
    "faiz indirim", "kredi faiz", "kredi maliyet",
    "bankacilik duzenleme", "bankalara yonelik", "banka kredileri",
    "finansal istikrar", "sistemik",
)

# Named listed banks. A specific issuer moves this to company_kap.
_NAMED_BANKS: Tuple[str, ...] = (
    "garanti", "akbank", "yapi kredi", "yapikredi", "isbank", "is bankasi",
    "vakifbank", "halkbank", "ziraat", "denizbank", "qnb", "teb",
    "sekerbank", "alternatif bank", "icbc", "odeabank", "fibabanka",
    "albaraka", "kuveyt turk", "vakif katilim", "ziraat katilim",
    "emlak katilim", "turkiye finans", "anadolubank", "burgan",
)

# Issuer-level event vocabulary: what a specific company announces about itself.
_COMPANY_EVENT_TERMS: Tuple[str, ...] = (
    "bilanco", "finansal sonuc", "finansal rapor", "ceyrek kar", "net kar",
    "kar acikladi", "zarar acikladi", "temettu", "kar payi",
    "sermaye artirim", "bedelli", "bedelsiz", "halka arz", "ihrac",
    "satin alma", "satin aldi", "devralma", "birlesme", "hisse devri",
    "ozel durum", "kap aciklama", "kap'a", "kaba bildir",
    "genel kurul", "yonetim kurulu", "atama", "atandi", "istifa etti",
    "ceo", "genel mudur", "yatirim karari", "tesis yatirim",
    "ihale kazandi", "sozlesme imzala", "anlasma imzala",
)


# Explicit commentary and press-about-press framing. Kept deliberately narrow:
# ordinary analytical reporting is still news about its subject, so only headlines
# that foreground the commentary itself land in media_narrative.
_MEDIA_NARRATIVE_TERMS: Tuple[str, ...] = (
    "kose yazisi", "kose yazar", "yorum:", "analiz:", "degerlendirme:",
    "manset", "mansetler", "gazetelerin", "basinda bugun", "basin ozeti",
    "medyada", "sosyal medyada", "gundem yaratti", "tepki cekti",
    "elestiri oku", "iddiasi gundem",
)


def _contains_any(text: str, terms: Sequence[str]) -> Optional[str]:
    for term in terms:
        if term in text:
            return term
    return None


@dataclass(frozen=True)
class FamilyAssignment:
    """One headline's family, with the evidence that produced it."""

    signal_family: str
    signal_family_version: str
    rule: str
    evidence: Optional[str]
    ambiguous: bool
    review_reason: Optional[str]


def _resolve_banks(text: str) -> FamilyAssignment:
    named = _contains_any(text, _NAMED_BANKS)
    event = _contains_any(text, _COMPANY_EVENT_TERMS)
    sector = _contains_any(text, _SECTOR_BANKING_TERMS)

    # A named bank announcing its own event is an issuer disclosure.
    if named and event:
        return FamilyAssignment(
            COMPANY_KAP, SIGNAL_FAMILY_VERSION,
            "banks:named_issuer_event", f"{named}+{event}", False, None,
        )
    if sector:
        return FamilyAssignment(
            BANKING_FINANCIAL_SECTOR, SIGNAL_FAMILY_VERSION,
            "banks:sector_vocabulary", sector, False, None,
        )
    if named:
        # A named bank without an issuer event: sector reporting that happens to
        # cite a bank. Assign to the sector, but flag it -- the boundary here is
        # genuinely uncertain and belongs in the review report.
        return FamilyAssignment(
            BANKING_FINANCIAL_SECTOR, SIGNAL_FAMILY_VERSION,
            "banks:named_without_issuer_event", named, True,
            "named bank without an issuer-level event; sector assumed",
        )
    return FamilyAssignment(
        BANKING_FINANCIAL_SECTOR, SIGNAL_FAMILY_VERSION,
        "banks:sector_default", None, False, None,
    )


def _resolve_bist_company(text: str, *, is_market_recap: bool) -> FamilyAssignment:
    if is_market_recap:
        return FamilyAssignment(
            MARKET_RECAP, SIGNAL_FAMILY_VERSION,
            "bist_company:market_recap", None, False, None,
        )
    event = _contains_any(text, _COMPANY_EVENT_TERMS)
    if event:
        return FamilyAssignment(
            COMPANY_KAP, SIGNAL_FAMILY_VERSION,
            "bist_company:issuer_event", event, False, None,
        )
    if "spk" in text or "sermaye piyasas" in text or "borsa istanbul" in text:
        return FamilyAssignment(
            POLITICAL_REGULATORY_RISK, SIGNAL_FAMILY_VERSION,
            "bist_company:market_regulation", "spk/borsa istanbul", False, None,
        )
    # Equity-market news that is neither a recap nor a named issuer event.
    return FamilyAssignment(
        COMPANY_KAP, SIGNAL_FAMILY_VERSION,
        "bist_company:default_issuer", None, True,
        "equity news without an explicit issuer event or recap marker",
    )


def assign_signal_family(
    category: Any,
    title: Any = None,
    *,
    is_market_recap: bool = False,
) -> FamilyAssignment:
    """Derive a signal family from the frozen category plus headline rules.

    ``is_market_recap`` comes from the separate recap classifier. A recap
    outranks its topical family for *any* category: a headline whose content is
    "this already moved" describes the market's past, not current conditions,
    whatever subject it nominally covers.
    """

    text = normalize(title)
    slug = normalize(category)

    if is_market_recap:
        return FamilyAssignment(
            MARKET_RECAP, SIGNAL_FAMILY_VERSION,
            "market_recap_override", None, False, None,
        )

    narrative = _contains_any(text, _MEDIA_NARRATIVE_TERMS)
    if narrative is not None:
        return FamilyAssignment(
            MEDIA_NARRATIVE, SIGNAL_FAMILY_VERSION,
            "media_narrative:commentary_framing", narrative, False, None,
        )

    if slug in RESOLVED_CATEGORIES:
        if slug == "banks":
            return _resolve_banks(text)
        return _resolve_bist_company(text, is_market_recap=False)

    if slug in DIRECT_CATEGORY_MAP:
        family = DIRECT_CATEGORY_MAP[slug]
        # A macro headline dominated by inflation vocabulary keeps
        # inflation_macro; one about the policy rate belongs to monetary policy.
        if family == INFLATION_MACRO and _contains_any(
            text, ("politika faizi", "faiz karari", "ppk", "para politikasi")
        ):
            return FamilyAssignment(
                MONETARY_POLICY, SIGNAL_FAMILY_VERSION,
                "turkey_macro:monetary_override", "policy-rate vocabulary",
                False, None,
            )
        return FamilyAssignment(
            family, SIGNAL_FAMILY_VERSION,
            f"category_map:{slug}", slug, False, None,
        )

    if not slug:
        return FamilyAssignment(
            OTHER, SIGNAL_FAMILY_VERSION, "unclassified_category", None, True,
            "headline has no detailed category",
        )

    return FamilyAssignment(
        OTHER, SIGNAL_FAMILY_VERSION, "unmapped_category", slug, True,
        f"detailed category {slug!r} has no family mapping",
    )


def is_domestic(family: str) -> bool:
    """Whether a family belongs in the Turkish domestic-only aggregate."""

    return family in DOMESTIC_FAMILIES
