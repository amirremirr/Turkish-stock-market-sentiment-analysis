"""Reading a "Pay Alım Satım Bildirimi" (share purchase/sale notification).

Written against real forms from the 2023 sample. Three things about them are
easy to get wrong and each would corrupt H8:

1. **The filer is often not the issuer.** A parent or shareholder files the
   form and names the company whose shares it traded in the "related
   companies" field. The traded stock is that related company, not the filer.
2. **Companies report buybacks on this form too.** A filer trading its own
   shares ("kendi paylarının geri alımı") is a buyback, which H8 excludes.
3. **The numbers are in a table after a block of headers,** one row per
   transaction date: date, nominal bought, nominal sold, net, then holdings.

What is parsed: the side (from the summed table rows), the nominal amounts,
the mid-point of the disclosed price range, and who the counterparty is in
relation to the issuer. Anything not read with confidence stays unparsed, and
an unparsed form is not an event.

Not open-market purchases, excluded where the text shows it: transfers and
off-exchange deals, option exercises, capital-increase allotments, inheritance
and gifts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from stock_research.entities import fold

INSIDER_PARSER_VERSION = "insider-parser-v2"

PARSED = "parsed"
UNPARSED = "unparsed"
EXCLUDED = "excluded_not_open_market"

KIND_THIRD_PARTY = "holder_trading_another_issuer"    # filer != issuer
KIND_OWN_SHARES = "issuer_trading_own_shares"         # a buyback
KIND_REPORTED_BY_ISSUER = "issuer_reporting_a_holder" # filer == issuer, not own shares

_EXCLUSION_STEMS = (
    "borsa disi", "borsa disinda", "devir", "devri", "devral", "virman", "opsiyon",
    "bedelsiz", "bedelli", "sermaye artirim", "miras", "bagis", "hibe",
)
_OWN_SHARE_PHRASES = ("kendi paylari", "geri alim", "geri alinan", "pay geri al")
_ROLE_PHRASES = (
    ("yonetim kurulu baskani", "board_chair"),
    ("yonetim kurulu uyesi", "board_member"),
    ("genel mudur", "general_manager"),
    ("yonetici", "executive"),
    ("ortakligimizca", "filer_is_holder"),
    ("ortagi", "shareholder"),
)
_ROW = re.compile(
    r"(\d{2}[/.]\d{2}[/.]\d{4})\s+(-?[\d.]+(?:,\d+)?)\s+(-?[\d.]+(?:,\d+)?)\s+(-?[\d.]+(?:,\d+)?)"
)
# Prices are read from the original text: folding turns the decimal comma
# into a space.
_PRICE_RANGE = re.compile(
    r"(\d+(?:,\d+)?)\s*-\s*(\d+(?:,\d+)?)\s*TL\s*fiyat\s*aral", re.IGNORECASE)
_SINGLE_PRICE = re.compile(r"(\d+(?:,\d+)?)\s*TL\s*(?:ortalama\s*)?fiyat(?!\s*aral)", re.IGNORECASE)


def turkish_number(text: str) -> Optional[float]:
    """'1.234.567,89' -> 1234567.89. None if it is not a number."""

    cleaned = text.strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def transaction_rows(body: Optional[str]) -> List[Dict[str, Any]]:
    """Rows of the transaction table: date, nominal bought, nominal sold.

    Only text after the table's last header is searched, so a date and
    figures in the free-text explanation are not read as a row.
    """

    text = body or ""
    marker = "Pay Alım Satım Bilgileri İşlem Tarihi"
    start = text.rfind(marker)
    if start < 0:
        return []
    table = text[start:]
    header_end = table.rfind("(%)")
    table = table[header_end + 3:] if header_end >= 0 else table
    rows = []
    for match in _ROW.finditer(table):
        bought, sold = turkish_number(match.group(2)), turkish_number(match.group(3))
        if bought is None or sold is None:
            continue
        rows.append({"date": match.group(1), "bought": bought, "sold": sold})
    return rows


def transaction_kind(body: Optional[str], summary: Optional[str],
                     sender_codes: Sequence[str], related_codes: Sequence[str]) -> str:
    """Whose shares were traded, relative to who filed."""

    other = [code for code in related_codes if code not in set(sender_codes)]
    if other:
        return KIND_THIRD_PARTY
    text = fold(f"{summary or ''} {body or ''}")
    if any(phrase in text for phrase in _OWN_SHARE_PHRASES):
        return KIND_OWN_SHARES
    return KIND_REPORTED_BY_ISSUER


def parse_insider(body: Optional[str], summary: Optional[str] = None,
                  sender_codes: Sequence[str] = (), related_codes: Sequence[str] = ()) -> Dict[str, Any]:
    """Side, size and counterparty of a share transaction notification."""

    out: Dict[str, Any] = {
        "insider_parser_version": INSIDER_PARSER_VERSION,
        "insider_parse_status": UNPARSED, "insider_kind": None, "insider_side": None,
        "insider_buy_nominal": None, "insider_sell_nominal": None,
        "insider_price_mid": None, "insider_value": None, "insider_role": None,
        "insider_exclusion": None,
    }
    folded = fold(body)
    if not folded:
        return out
    out["insider_kind"] = transaction_kind(body, summary, sender_codes, related_codes)

    explanation = folded.split("pay alim satim bilgileri islem tarihi")[0]
    for stem in _EXCLUSION_STEMS:
        if re.search(rf"(?<![a-z]){re.escape(stem)}", explanation):
            out.update({"insider_parse_status": EXCLUDED, "insider_exclusion": stem})
            return out
    for phrase, role in _ROLE_PHRASES:
        if phrase in explanation:
            out["insider_role"] = role
            break

    rows = transaction_rows(body)
    if not rows:
        return out
    bought, sold = sum(r["bought"] for r in rows), sum(r["sold"] for r in rows)
    out.update({"insider_buy_nominal": bought, "insider_sell_nominal": sold})
    if bought > 0 and sold == 0:
        out["insider_side"] = "buy"
    elif sold > 0 and bought == 0:
        out["insider_side"] = "sell"
    elif bought > 0 and sold > 0:
        out["insider_side"] = "mixed"
    else:
        return out

    price = None
    raw_explanation = (body or "").split("Pay Alım Satım Bilgileri İşlem Tarihi")[0]
    span = _PRICE_RANGE.search(raw_explanation)
    if span:
        low, high = turkish_number(span.group(1)), turkish_number(span.group(2))
        if low and high and low <= high:
            price = (low + high) / 2
    else:
        single = _SINGLE_PRICE.search(raw_explanation)
        if single:
            price = turkish_number(single.group(1))
    if price and out["insider_side"] == "buy":
        # Nominal value is 1 TL per share on Borsa Istanbul, so nominal bought
        # is the share count; value is an estimate at the range mid-point.
        out.update({"insider_price_mid": price, "insider_value": bought * price})
    out["insider_parse_status"] = PARSED
    return out
