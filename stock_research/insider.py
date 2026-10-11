"""Reading a "Pay Alım Satım Bildirimi" (share purchase/sale notification).

Written against real forms from the 2023 sample. Four things about them are
easy to get wrong, and each would corrupt H8:

1. **The filer is often not the issuer.** A parent or shareholder files the
   form and names the company whose shares it traded in the "related
   companies" field. The traded stock is that related company, not the filer.
2. **Companies report buybacks on this form too.** A filer trading its own
   shares ("kendi paylarının geri alımı") is a buyback, which H8 excludes.
3. **The numbers are in a table after a block of headers,** one row per
   transaction date: date, nominal bought, nominal sold, net, holdings at the
   start of the day, holdings at the end, then percentages.
4. **A blank cell disappears.** In the extracted text an empty "bought" or
   "sold" cell leaves no trace, so a sale of 190,000 reads
   ``190.000 190.000 22.966.000 22.776.000`` -- which looks like a purchase
   and a sale of 190,000 each. The side is therefore never read from the
   position of a number. It is read from the **change in holdings**, and a
   row is accepted only if its amounts reconcile with that change.

What is parsed: the side and nominal amounts (from reconciled rows), the
midpoint of the disclosed price range, and who the counterparty is in relation
to the issuer. Anything not read with confidence stays unparsed, and an
unparsed form is not an event.

Not open-market purchases, excluded where the text shows it: transfers and
off-exchange deals, option exercises, capital-increase allotments, inheritance
and gifts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from stock_research.entities import fold

INSIDER_PARSER_VERSION = "insider-parser-v3"

PARSED = "parsed"
UNPARSED = "unparsed"
EXCLUDED = "excluded_not_open_market"

KIND_THIRD_PARTY = "holder_trading_another_issuer"    # filer != issuer
KIND_OWN_SHARES = "issuer_trading_own_shares"         # a buyback
KIND_REPORTED_BY_ISSUER = "issuer_reporting_a_holder" # filer == issuer, not own shares

#: Stems that mark a transaction as not an on-exchange purchase or sale.
#: "bedelli" alone is not one: "190.000 TL nominal bedelli" means "with a
#: nominal value of", so the capital-increase stems name the increase.
_EXCLUSION_STEMS = (
    "borsa disi", "borsa disinda", "devir", "devri", "devral", "virman", "opsiyon",
    "bedelli sermaye", "bedelsiz sermaye", "bedelsiz pay", "sermaye artirim", "ruchan",
    "miras", "bagis", "hibe",
)
_OWN_SHARE_PHRASES = ("kendi paylari", "geri alim", "geri alinan", "pay geri al")
_VENUE_PHRASES = ("borsa istanbul", "borsa'da", "borsada", "bias")
_ROLE_PHRASES = (
    ("yonetim kurulu baskani", "board_chair"),
    ("yonetim kurulu uyesi", "board_member"),
    ("genel mudur", "general_manager"),
    ("yonetici", "executive"),
    ("ortakligimizca", "filer_is_holder"),
    ("hissedar", "shareholder"),
    ("pay sahibi", "shareholder"),
    ("ortagi", "shareholder"),
)
_TABLE_MARKER = "Pay Alım Satım Bilgileri İşlem Tarihi"
_DATE = re.compile(r"\d{2}[/.]\d{2}[/.]\d{4}")
_NUMBER = re.compile(r"-?\d[\d.]*(?:,\d+)?")
# Prices are read from the original text: folding turns the decimal comma
# into a space.
_PRICE_RANGE = re.compile(
    r"(\d+(?:,\d+)?)\s*-\s*(\d+(?:,\d+)?)\s*TL\s*fiyat\s*aral", re.IGNORECASE)
_SINGLE_PRICE = re.compile(r"(\d+(?:,\d+)?)\s*TL\s*(?:ortalama\s*)?fiyat(?!\s*aral)", re.IGNORECASE)
#: Relative tolerance when reconciling amounts with the change in holdings.
RECONCILE_TOLERANCE = 0.01


def turkish_number(text: str) -> Optional[float]:
    """'1.234.567,89' -> 1234567.89. None if it is not a number."""

    cleaned = text.strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RECONCILE_TOLERANCE * max(abs(a), abs(b), 1.0)


def transaction_rows(body: Optional[str]) -> List[Dict[str, Any]]:
    """Reconciled rows of the transaction table.

    Each row is the text from one date to the next. The figures before the
    first percentage sign are, at most: bought, sold, net, holdings at the
    start, holdings at the end. The last two are always present. A row is
    returned only if what precedes them agrees with ``end - start``; otherwise
    it is marked ``reconciled: False`` and the form will not be parsed.
    """

    text = body or ""
    start = text.rfind(_TABLE_MARKER)
    if start < 0:
        return []
    table = text[start:]
    header_end = table.rfind("(%)")
    table = table[header_end + 3:] if header_end >= 0 else table

    dates = list(_DATE.finditer(table))
    rows: List[Dict[str, Any]] = []
    for position, match in enumerate(dates):
        end = dates[position + 1].start() if position + 1 < len(dates) else len(table)
        cell_text = table[match.end():end].split("%")[0]
        numbers = [turkish_number(n) for n in _NUMBER.findall(cell_text)]
        numbers = [n for n in numbers if n is not None]
        row: Dict[str, Any] = {"date": match.group(0), "bought": None, "sold": None,
                               "reconciled": False}
        if len(numbers) >= 3:
            begin, finish = numbers[-2], numbers[-1]
            change = finish - begin
            amounts = numbers[:-2]
            if len(amounts) >= 3:                 # bought, sold, net all present
                bought, sold = amounts[0], amounts[1]
                if _close(bought - sold, change):
                    row.update({"bought": bought, "sold": sold, "reconciled": True})
            elif amounts and _close(abs(amounts[0]), abs(change)) and change != 0:
                # One of the two cells was blank; the holdings say which.
                amount = abs(amounts[0])
                row.update({"bought": amount if change > 0 else 0.0,
                            "sold": amount if change < 0 else 0.0, "reconciled": True})
        rows.append(row)
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
        "insider_exclusion": None, "insider_venue_stated": None,
    }
    folded = fold(body)
    if not folded:
        return out
    out["insider_kind"] = transaction_kind(body, summary, sender_codes, related_codes)

    explanation = folded.split(fold(_TABLE_MARKER))[0]
    out["insider_venue_stated"] = any(phrase in explanation for phrase in _VENUE_PHRASES)
    for stem in _EXCLUSION_STEMS:
        if re.search(rf"(?<![a-z]){re.escape(stem)}", explanation):
            out.update({"insider_parse_status": EXCLUDED, "insider_exclusion": stem})
            return out
    for phrase, role in _ROLE_PHRASES:
        if phrase in explanation:
            out["insider_role"] = role
            break

    rows = transaction_rows(body)
    if not rows or not all(row["reconciled"] for row in rows):
        return out                      # no table, or a row that does not add up
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
    raw_explanation = (body or "").split(_TABLE_MARKER)[0]
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
        # is the share count; value is an estimate at the range midpoint.
        out.update({"insider_price_mid": price, "insider_value": bought * price})
    out["insider_parse_status"] = PARSED
    return out
