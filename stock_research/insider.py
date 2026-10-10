"""Reading a "Pay Alım Satım Bildirimi" (share purchase/sale notification).

The form reports transactions in a company's shares by people and entities
with a notification duty. What H8 needs from it is narrow: was the net
transaction a purchase, by whom, and how large.

The parser reads the form's own labelled fields. Anything it cannot read with
confidence it leaves unparsed, and an unparsed notification is not an event:
guessing the side of an insider trade would put sales into a sample of
purchases.

Not open-market purchases, and excluded where the text shows it:
transfers and off-exchange deals ("borsa dışı", "devir", "virman"), option
exercises, capital-increase allotments, and buybacks by the company itself
(those are filed on a different form and never reach this parser).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from stock_research.entities import fold

INSIDER_PARSER_VERSION = "insider-parser-v1"

PARSED = "parsed"
UNPARSED = "unparsed"
EXCLUDED = "excluded_not_open_market"

_EXCLUSION_PHRASES = (
    "borsa disi", "borsa disinda", "devir", "virman", "opsiyon", "bedelsiz",
    "bedelli", "sermaye artirim", "miras", "bagis", "takas",
)
_ROLE_PHRASES = (
    ("yonetim kurulu baskani", "board_chair"),
    ("yonetim kurulu uyesi", "board_member"),
    ("genel mudur", "general_manager"),
    ("yonetici", "executive"),
    ("ana ortak", "controlling_shareholder"),
    ("ortagi", "shareholder"),
    ("ortak", "shareholder"),
)


def turkish_number(text: str) -> Optional[float]:
    """'1.234.567,89' -> 1234567.89. None if it is not a number."""

    cleaned = text.strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _amounts_after(label: str, folded: str) -> List[float]:
    values = []
    for match in re.finditer(re.escape(label) + r"[^0-9]{0,60}([0-9][0-9.]*(?:,[0-9]+)?)", folded):
        number = turkish_number(match.group(1))
        if number is not None:
            values.append(number)
    return values


def parse_insider(body: Optional[str]) -> Dict[str, Any]:
    """Side, size and filer role of a share transaction notification."""

    out: Dict[str, Any] = {
        "insider_parser_version": INSIDER_PARSER_VERSION,
        "insider_parse_status": UNPARSED, "insider_side": None,
        "insider_buy_nominal": None, "insider_sell_nominal": None,
        "insider_value": None, "insider_role": None, "insider_exclusion": None,
    }
    folded = fold(body)
    if not folded:
        return out

    for phrase in _EXCLUSION_PHRASES:
        if re.search(rf"(?<![a-z]){re.escape(phrase)}", folded):
            out.update({"insider_parse_status": EXCLUDED, "insider_exclusion": phrase})
            return out
    for phrase, role in _ROLE_PHRASES:
        if phrase in folded:
            out["insider_role"] = role
            break

    buys = _amounts_after("alim islemine konu paylarin toplam nominal tutari", folded)
    sells = _amounts_after("satim islemine konu paylarin toplam nominal tutari", folded)
    if not buys and not sells:
        return out
    bought, sold = sum(buys), sum(sells)
    out.update({"insider_buy_nominal": bought, "insider_sell_nominal": sold})
    if bought > 0 and sold == 0:
        out["insider_side"] = "buy"
    elif sold > 0 and bought == 0:
        out["insider_side"] = "sell"
    elif bought > 0 and sold > 0:
        out["insider_side"] = "mixed"
    else:
        return out

    prices = _amounts_after("islem fiyati", folded)
    if out["insider_side"] == "buy" and len(prices) == len(buys) and prices:
        out["insider_value"] = float(sum(n * p for n, p in zip(buys, prices)))
    out["insider_parse_status"] = PARSED
    return out
