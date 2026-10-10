"""KAP disclosures -> categorised, timed, ticker-resolved events.

Taxonomy
--------
KAP files disclosures under standard form names. The category is taken from
that name first, because it is the filer's own classification and it is exact.
More than half of all material-event disclosures use the catch-all form
"Özel Durum Açıklaması (Genel)"; for those, and only those, a short list of
phrases in the filer's one-line summary assigns a category, and everything
else stays ``general_other``. The phrase list is deliberately narrow: a
missed buyback lands in ``general_other`` and dilutes it slightly, while a
wrong assignment would put noise into the category being tested.

The taxonomy was fixed from form names and summaries alone, before any price
was joined to an event.

Dates
-----
Every event is anchored to its **publication** timestamp, the moment the
market could know. A disclosure about a transaction, a board decision or a
record date carries those other dates in its body; they are not used to place
the event, because they were not public when they happened.

Updates and corrections
-----------------------
Forms carry "is this an update" and "is this a correction" flags. Such a
disclosure repeats an event that already had its reaction, so the primary
samples exclude it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from stock_research import store
from stock_research.calendar import SessionCalendar, parse_kap_time
from stock_research.config import KAP_FRAME_VERSION
from stock_research.entities import fold

TAXONOMY_VERSION = "kap-taxonomy-v1"

NEW_CONTRACT = "new_contract"
BUYBACK = "buyback"
INSIDER_TRADE = "insider_trade"
BONUS_ISSUE = "bonus_issue"
RIGHTS_ISSUE = "rights_issue"
CAPITAL_CHANGE_OTHER = "capital_change_other"
DIVIDEND = "dividend"
ASSET_ACQUISITION = "asset_acquisition"
ASSET_SALE = "asset_sale"
DEBT_ISSUANCE = "debt_issuance"
UNUSUAL_PRICE_VOLUME = "unusual_price_volume"
ADMINISTRATIVE = "administrative"
GENERAL_OTHER = "general_other"
OTHER_FORM = "other_form"

#: Form name -> category. Exact match on KAP's own form titles, compared after
#: folding (case, Turkish letters and punctuation removed).
_SUBJECT_CATEGORIES_RAW: Dict[str, str] = {
    "yeni is iliskisi": NEW_CONTRACT,
    "paylarin geri alinmasina iliskin bildirim": BUYBACK,
    "kar payi dagitim islemlerine iliskin bildirim": DIVIDEND,
    "finansal duran varlik edinimi": ASSET_ACQUISITION,
    "maddi duran varlik alimi": ASSET_ACQUISITION,
    "finansal duran varlik satisi": ASSET_SALE,
    "maddi duran varlik satimi": ASSET_SALE,
    "pay disinda sermaye piyasasi araci islemlerine iliskin bildirim (faiz iceren)": DEBT_ISSUANCE,
    "pay disinda sermaye piyasasi araci islemlerine iliskin bildirim (faizsiz)": DEBT_ISSUANCE,
    "ihrac tavanina iliskin bildirim": DEBT_ISSUANCE,
    "olagan disi fiyat ve miktar hareketleri": UNUSUAL_PRICE_VOLUME,
    "genel kurul islemlerine iliskin bildirim": ADMINISTRATIVE,
    "genel kurul bildirimi": ADMINISTRATIVE,
    "esas sozlesme tadili": ADMINISTRATIVE,
    "bagimsiz denetim kurulusunun belirlenmesi": ADMINISTRATIVE,
    "sirket merkezi degisikligi": ADMINISTRATIVE,
    "kayitli sermaye tavani islemlerine iliskin bildirim": ADMINISTRATIVE,
    "sirket genel bilgi formu": ADMINISTRATIVE,
    "kurumsal yonetim uyum raporu": ADMINISTRATIVE,
    "sorumluluk beyani": ADMINISTRATIVE,
}
SUBJECT_CATEGORIES: Dict[str, str] = {fold(k): v for k, v in _SUBJECT_CATEGORIES_RAW.items()}
CAPITAL_FORM = fold("Sermaye Artırımı - Azaltımı İşlemlerine İlişkin Bildirim")
SHARE_TRANSACTION_FORM = fold("Pay Alım Satım Bildirimi")
GENERAL_FORM = fold("Özel Durum Açıklaması (Genel)")

#: Phrases in the summary of a general-form disclosure. Checked in order.
GENERAL_SUMMARY_RULES: Sequence[tuple] = (
    (BUYBACK, ("geri alim", "geri alinan pay", "pay geri al", "paylarin geri alin")),
    (NEW_CONTRACT, ("yeni is iliskisi", "sozlesme imza", "siparis alin", "siparis al",
                    "ihale sonuc", "ihaleyi kazan", "ihale kazan", "is alinmasi")),
    (DIVIDEND, ("kar payi dagit", "temettu")),
    (BONUS_ISSUE, ("bedelsiz sermaye artirim", "bedelsiz pay")),
    (RIGHTS_ISSUE, ("bedelli sermaye artirim",)),
)

_FLAG_PATTERNS = {
    "is_update": r"UpdateAnnouncementFlag\|[^|]*?\b(Evet|Hayır)\b",
    "is_correction": r"CorrectionAnnouncementFlag\|[^|]*?\b(Evet|Hayır)\b",
    "is_delayed": r"DelayedAnnouncementFlag\|[^|]*?\b(Evet|Hayır)\b",
}


def parse_flags(body: Optional[str]) -> Dict[str, Optional[bool]]:
    """The form's own update / correction / delayed flags. None if absent."""

    out: Dict[str, Optional[bool]] = {}
    for name, pattern in _FLAG_PATTERNS.items():
        match = re.search(pattern, body or "")
        out[name] = (match.group(1) == "Evet") if match else None
    return out


def classify(subject: Optional[str], summary: Optional[str], body: Optional[str],
             sender: Sequence[str] = (), related: Sequence[str] = ()) -> Dict[str, str]:
    """Category and the rule that assigned it."""

    form = fold(subject)
    if not form:
        return {"category": OTHER_FORM, "rule": "no_subject"}
    if form == SHARE_TRANSACTION_FORM:
        # The same form carries a holder's trade in another issuer, an issuer
        # reporting a holder's trade, and an issuer's own buyback.
        from stock_research.insider import KIND_OWN_SHARES, transaction_kind

        kind = transaction_kind(body, summary, sender, related)
        if kind == KIND_OWN_SHARES:
            return {"category": BUYBACK, "rule": "share_form_own_shares"}
        return {"category": INSIDER_TRADE, "rule": f"share_form:{kind}"}
    if form in SUBJECT_CATEGORIES:
        return {"category": SUBJECT_CATEGORIES[form], "rule": "form_name"}
    if form == CAPITAL_FORM:
        text = fold(f"{summary or ''} {body or ''}")
        bonus = "bedelsiz" in text and not re.search(r"bedelsiz[^.]{0,80}(yok|bulunmamaktadir)", text)
        rights = bool(re.search(r"bedelli(?!siz)", text))
        if bonus and not rights:
            return {"category": BONUS_ISSUE, "rule": "capital_form_bedelsiz_only"}
        if rights and not bonus:
            return {"category": RIGHTS_ISSUE, "rule": "capital_form_bedelli_only"}
        return {"category": CAPITAL_CHANGE_OTHER, "rule": "capital_form_mixed_or_unclear"}
    if form == GENERAL_FORM:
        text = fold(summary)
        for category, phrases in GENERAL_SUMMARY_RULES:
            for phrase in phrases:
                if phrase in text:
                    return {"category": category, "rule": f"general_summary:{phrase}"}
        return {"category": GENERAL_OTHER, "rule": "general_form_no_phrase"}
    return {"category": OTHER_FORM, "rule": "unmapped_form_name"}


# -- Tickers -------------------------------------------------------------------
def yahoo_symbol(code: str) -> str:
    return f"{code}.IS"


def sender_codes(raw: Optional[str]) -> List[str]:
    try:
        codes = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [c for c in codes if isinstance(c, str) and re.fullmatch(r"[A-Z0-9]{3,6}", c)]


def related_codes(raw: Optional[str]) -> List[str]:
    """Codes in the form's "related companies" field (a list of {"code": ...})."""

    try:
        items = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    codes = [item.get("code") for item in items if isinstance(item, dict)]
    return [c for c in codes if isinstance(c, str) and re.fullmatch(r"[A-Z0-9]{3,6}", c)]


def candidate_symbols(db_path=None) -> List[str]:
    """Every sender code among sampled disclosures, as a provider symbol."""

    with store.connect(db_path) as con:
        rows = con.execute(
            "SELECT DISTINCT d.sender_codes, d.related_stocks FROM sr_raw_kap_detail d "
            "JOIN sr_raw_kap_listing l USING (disclosure_index) "
            "WHERE l.frame_version = ?", (KAP_FRAME_VERSION,),
        ).fetchall()
    codes = sorted({code for row in rows
                    for code in sender_codes(row[0]) + related_codes(row[1])})
    return [yahoo_symbol(code) for code in codes]


def primary_symbol(codes: Sequence[str], panel, day0: Optional[str]) -> Optional[str]:
    """Which of an issuer's listed lines stands for it at *day0*.

    An issuer can have several lines (share classes, a warrant issuer code).
    The line used is the one with the highest median turnover over the 60
    sessions before day 0 -- a rule that looks only backwards and needs no
    hand-made table. No line with enough history means no ticker.
    """

    position = panel.calendar.index(day0) if day0 else None
    if position is None:
        return None
    best, best_turnover = None, -1.0
    for code in codes:
        frame = panel.frame(yahoo_symbol(code))
        if frame is None:
            continue
        window = frame["turnover"].to_numpy()[max(0, position - 60):position]
        window = window[np.isfinite(window)]
        if len(window) < 20:
            continue
        median = float(np.median(window))
        if median > best_turnover:
            best, best_turnover = yahoo_symbol(code), median
    return best


# -- Building the event table ---------------------------------------------------
def load_details(db_path=None) -> pd.DataFrame:
    with store.connect(db_path) as con:
        return pd.read_sql_query(
            "SELECT d.*, l.block_seq FROM sr_raw_kap_detail d "
            "JOIN sr_raw_kap_listing l USING (disclosure_index) "
            "WHERE l.frame_version = ? AND d.published_raw IS NOT NULL "
            "ORDER BY d.disclosure_index", con, params=(KAP_FRAME_VERSION,),
        )


def build_kap_events(details: pd.DataFrame, panel) -> pd.DataFrame:
    """One row per sampled disclosure with timing, category and ticker."""

    from stock_research.insider import parse_insider

    rows: List[Dict[str, Any]] = []
    for detail in details.to_dict("records"):
        action = panel.calendar.actionable(parse_kap_time(detail["published_raw"]))
        codes = sender_codes(detail["sender_codes"])
        related = related_codes(detail["related_stocks"])
        other = [c for c in related if c not in set(codes)]
        label = classify(detail["subject_tr"], detail["summary_tr"], detail["body_text"],
                         codes, related)
        # Whose stock the event is about. Normally the filer's. On a share
        # transaction filed by a holder it is the related company's, and if
        # the form names more than one the target is not knowable.
        if label["category"] == INSIDER_TRADE and other:
            target = other if len(other) == 1 else []
        else:
            target = codes
        row = {
            "event_id": f"kap:{detail['disclosure_index']}",
            "disclosure_index": int(detail["disclosure_index"]),
            "sender_id": detail["sender_id"],
            "sender_title": detail["sender_title"],
            "sender_codes": codes,
            "related_codes": related,
            "names_other_issuer": bool(other),
            "ticker": primary_symbol(target, panel, action.day0),
            "published_local": action.published_local,
            "published_utc": action.published_utc,
            "day0": action.day0,
            "bucket": action.bucket,
            "timing_ambiguous": action.timing_ambiguous,
            "disclosure_type": detail["disclosure_type"],
            "subject": detail["subject_tr"],
            "summary": detail["summary_tr"],
            "category": label["category"],
            "category_rule": label["rule"],
            "taxonomy_version": TAXONOMY_VERSION,
            **parse_flags(detail["body_text"]),
            "source": detail["source"],
        }
        if row["category"] == INSIDER_TRADE:
            row.update(parse_insider(detail["body_text"], detail["summary_tr"], codes, related))
        rows.append(row)
    return pd.DataFrame(rows)
