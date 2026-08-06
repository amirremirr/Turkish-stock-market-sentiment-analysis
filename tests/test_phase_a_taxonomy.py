"""Signal-family taxonomy and market-recap classification contracts.

The taxonomy is a derived layer. Its whole safety argument is that it reads the
frozen category and title and writes only its own columns, so these tests spend
most of their effort proving that nothing underneath moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
from taxonomy.market_recap import MARKET_RECAP_VERSION, classify_market_recap
from taxonomy.signal_family import (
    BANKING_FINANCIAL_SECTOR,
    COMPANY_KAP,
    DOMESTIC_FAMILIES,
    FX_LIRA,
    GLOBAL_RISK,
    INFLATION_MACRO,
    MARKET_RECAP,
    MEDIA_NARRATIVE,
    MONETARY_POLICY,
    OTHER,
    POLITICAL_REGULATORY_RISK,
    SIGNAL_FAMILIES,
    SIGNAL_FAMILY_VERSION,
    assign_signal_family,
    is_domestic,
)


# -- Every detailed category maps somewhere ------------------------------------

# The full production category set, from config.NEWS_CATEGORIES.
ALL_CATEGORIES = [
    "bist_company", "rates_tcmb", "political_risk", "turkey_macro", "crypto",
    "global_risk", "fx_lira", "banks", "energy_commodities", "other",
]


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_every_detailed_category_maps_to_a_known_family(category):
    result = assign_signal_family(category, "Nötr bir başlık")
    assert result.signal_family in SIGNAL_FAMILIES
    assert result.signal_family_version == SIGNAL_FAMILY_VERSION
    assert result.rule


@pytest.mark.parametrize(
    "category, expected",
    [
        ("rates_tcmb", MONETARY_POLICY),
        ("turkey_macro", INFLATION_MACRO),
        ("political_risk", POLITICAL_REGULATORY_RISK),
        ("fx_lira", FX_LIRA),
        ("global_risk", GLOBAL_RISK),
        ("energy_commodities", GLOBAL_RISK),
        ("crypto", OTHER),
        ("other", OTHER),
    ],
)
def test_direct_category_mappings(category, expected):
    assert assign_signal_family(category, "Başlık").signal_family == expected


def test_unmapped_category_is_other_and_flagged_ambiguous():
    result = assign_signal_family("brand_new_category", "Başlık")
    assert result.signal_family == OTHER
    assert result.ambiguous
    assert "no family mapping" in result.review_reason


def test_missing_category_is_reported_not_guessed():
    result = assign_signal_family(None, "Başlık")
    assert result.signal_family == OTHER
    assert result.ambiguous


def test_policy_rate_vocabulary_moves_macro_to_monetary_policy():
    result = assign_signal_family("turkey_macro", "TCMB politika faizi kararı")
    assert result.signal_family == MONETARY_POLICY


def test_commentary_framing_lands_in_media_narrative():
    result = assign_signal_family("turkey_macro", "Köşe yazısı: enflasyon nereye")
    assert result.signal_family == MEDIA_NARRATIVE


# -- The approved banking boundary: entity specificity, not industry -----------

@pytest.mark.parametrize(
    "title",
    [
        "Bankacılık sektöründe kredi büyümesi yavaşladı",
        "BDDK bankalara yönelik yeni düzenleme yayımladı",
        "Mevduat faiz oranları geriledi",
        "Bankaların sermaye yeterlilik rasyosu güçlü",
        "Takipteki alacaklar arttı",
        "Konut kredisi faizleri değişti",
    ],
)
def test_sector_level_banking_is_banking_financial_sector(title):
    result = assign_signal_family("banks", title)
    assert result.signal_family == BANKING_FINANCIAL_SECTOR
    assert not result.ambiguous


@pytest.mark.parametrize(
    "title",
    [
        "Garanti BBVA bilanço açıkladı",
        "Akbank temettü kararı aldı",
        "Yapı Kredi sermaye artırımı yapacak",
        "Halkbank net kar açıkladı",
        "İş Bankası satın alma anlaşması imzaladı",
    ],
)
def test_named_bank_issuer_events_are_company_kap(title):
    """A named bank's own event is an issuer disclosure, like any other issuer."""

    result = assign_signal_family("banks", title)
    assert result.signal_family == COMPANY_KAP
    assert not result.ambiguous


def test_named_bank_without_an_issuer_event_is_flagged_for_review():
    result = assign_signal_family("banks", "Garanti BBVA şubelerini yeniledi")
    assert result.signal_family == BANKING_FINANCIAL_SECTOR
    assert result.ambiguous
    assert "named bank" in result.review_reason


def test_bist_company_issuer_event_is_company_kap():
    result = assign_signal_family("bist_company", "THY bilanço açıkladı")
    assert result.signal_family == COMPANY_KAP
    assert not result.ambiguous


def test_bist_company_market_regulation_is_political_regulatory():
    result = assign_signal_family("bist_company", "SPK yeni karar aldı")
    assert result.signal_family == POLITICAL_REGULATORY_RISK


def test_unresolved_equity_news_is_reported_ambiguous():
    result = assign_signal_family("bist_company", "Şirket hakkında genel bilgi")
    assert result.ambiguous
    assert result.review_reason


# -- Domestic-only membership --------------------------------------------------

def test_global_risk_is_not_domestic():
    assert not is_domestic(GLOBAL_RISK)
    assert GLOBAL_RISK not in DOMESTIC_FAMILIES


def test_market_recap_is_not_domestic():
    assert not is_domestic(MARKET_RECAP)


@pytest.mark.parametrize("family", DOMESTIC_FAMILIES)
def test_domestic_families_are_domestic(family):
    assert is_domestic(family)


# -- Market recap: positives ----------------------------------------------------

@pytest.mark.parametrize(
    "title",
    [
        "Borsa güne yükselişle başladı",
        "BIST 100 günü düşüşle tamamladı",
        "Bankacılık hisseleri değer kazandı",
        "Dolar gün içinde yükseldi",
        "BIST 100 haftayı yükselişle tamamladı",
        "Altın gün içinde geriledi",
        "Piyasalar yatay seyretti",
        "Borsa günü artıda kapattı",
        "Dolar ne kadar? Güncel dolar kuru",
        "Euro seans sonunda geriledi",
    ],
)
def test_market_recap_positive_fixtures(title):
    result = classify_market_recap(title)
    assert result.is_market_recap, f"{title!r} -> {result.rule}"
    assert result.version == MARKET_RECAP_VERSION
    assert result.confidence > 0


# -- Market recap: false positives ----------------------------------------------

@pytest.mark.parametrize(
    "title",
    [
        # market vocabulary, genuinely new information
        "Borsa İstanbul yeni endeks oluşturdu",
        "Şirketin halka arzı onaylandı",
        "Garanti BBVA genel müdür atadı",
        "SPK yeni düzenleme kararı aldı",
        "THY bilanço açıkladı",
        "TCMB politika faizi kararını açıkladı",
        "BDDK bankalara yönelik yönetmelik yayımladı",
        "Borsada işlem görmeye başladı",
        "Şirket sermaye artırımı kararı aldı",
        "Merkez Bankası faiz kararı bekleniyor",
        "Hisse için hedef fiyat açıklandı",
        "Borsa İstanbul'a yeni CEO atandı",
    ],
)
def test_market_recap_false_positive_guards(title):
    result = classify_market_recap(title)
    assert not result.is_market_recap, f"{title!r} matched via {result.rule}"


def test_recap_exemption_is_reported_as_such():
    result = classify_market_recap("Borsa yükseldi, yeni endeks oluşturuldu")
    assert not result.is_market_recap
    assert result.rule == "exempt_new_information"


def test_empty_title_is_not_a_recap():
    assert not classify_market_recap("").is_market_recap
    assert not classify_market_recap(None).is_market_recap


def test_market_recap_overrides_topical_family():
    result = assign_signal_family("fx_lira", "Dolar gün içinde yükseldi",
                                  is_market_recap=True)
    assert result.signal_family == MARKET_RECAP
    assert result.rule == "market_recap_override"


# -- Storage: version stamping and non-mutation --------------------------------

@pytest.fixture
def classified_db(tmp_path):
    path = str(tmp_path / "tax.db")
    db.init_db(path)
    rows = [
        (1, "aa_ekonomi", "Bankacılık sektöründe kredi büyümesi yavaşladı", "banks"),
        (2, "dunya", "Borsa güne yükselişle başladı", "bist_company"),
        (3, "sozcu_gundem", "TCMB politika faizi kararını açıkladı", "rates_tcmb"),
        (4, "bloomberght", "Fed faiz kararını açıkladı", "global_risk"),
        (5, "sabah_ekonomi", "Garanti BBVA bilanço açıkladı", "banks"),
    ]
    with db._conn(path) as con:
        for index, source, title, category in rows:
            con.execute(
                """INSERT INTO headlines (id, source, title, url, published_at,
                   scraped_at, sentiment_score, sentiment_label, scored_at,
                   p_positive, p_neutral, p_negative, model_name, experiment_id,
                   category, relevance, signal_date, timing_bucket,
                   processing_status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'scored')""",
                (index, source, title, f"u{index}", "2026-07-20",
                 "2026-07-20T09:00:00Z", 0.2, "positive", "2026-07-20T09:05:00Z",
                 0.6, 0.3, 0.1, "gpt-5-mini-2025-08-07/p3", "v1-p3", category,
                 0.9, "2026-07-20", "pre_open"),
            )
    return path


def _frozen_state(path):
    with db._conn(path) as con:
        return con.execute(
            "SELECT id, sentiment_score, sentiment_label, scored_at, model_name,"
            " experiment_id, category FROM headlines ORDER BY id"
        ).fetchall()


def test_classification_never_mutates_scores_categories_or_experiment(classified_db):
    before = _frozen_state(classified_db)
    db.classify_signal_families(db_path=classified_db)
    assert _frozen_state(classified_db) == before


def test_classification_stamps_both_versions(classified_db):
    db.classify_signal_families(db_path=classified_db)
    with db._conn(classified_db) as con:
        rows = con.execute(
            "SELECT signal_family_version, market_recap_version FROM headlines"
        ).fetchall()
    assert all(row[0] == SIGNAL_FAMILY_VERSION for row in rows)
    assert all(row[1] == MARKET_RECAP_VERSION for row in rows)


def test_classification_is_idempotent(classified_db):
    first = db.classify_signal_families(db_path=classified_db)
    second = db.classify_signal_families(db_path=classified_db)
    assert first["classified"] == 5
    assert second["classified"] == 0, "a repeat call must reclassify nothing"


def test_classification_assigns_the_expected_families(classified_db):
    db.classify_signal_families(db_path=classified_db)
    with db._conn(classified_db) as con:
        families = dict(con.execute("SELECT id, signal_family FROM headlines"))
    assert families[1] == BANKING_FINANCIAL_SECTOR
    assert families[2] == MARKET_RECAP
    assert families[3] == MONETARY_POLICY
    assert families[4] == GLOBAL_RISK
    assert families[5] == COMPANY_KAP


def test_recap_flag_and_evidence_are_stored(classified_db):
    db.classify_signal_families(db_path=classified_db)
    with db._conn(classified_db) as con:
        row = con.execute(
            "SELECT is_market_recap, market_recap_rule, market_recap_confidence"
            " FROM headlines WHERE id = 2"
        ).fetchone()
    assert row["is_market_recap"] == 1
    assert row["market_recap_rule"]
    assert row["market_recap_confidence"] > 0
