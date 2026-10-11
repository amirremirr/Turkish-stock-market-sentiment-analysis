"""Entity linking, the KAP taxonomy, insider forms, price limits, costs, guards."""

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_research import costs, events, fixtures, guards, insider, limits, store
from stock_research.data import kap, prices
from stock_research.entities import (
    CONFIRM_THRESHOLD, Linker, MENTION_CASUAL, MENTION_MATERIAL, MENTION_MULTI,
    MENTION_SECTOR, evaluate, fold,
)

HERE = Path(__file__).parent


# -- Entity linking -------------------------------------------------------------
@pytest.fixture(scope="module")
def labelled():
    with open(HERE / "entity_eval.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    return [{"text": r["text"], "tickers": [t for t in r["tickers"].split(";") if t]} for r in rows]


def test_linker_makes_no_false_assignment_on_the_labelled_set(labelled):
    report = evaluate(labelled)
    assert report["false_assignments"] == 0, report["errors"]
    assert report["precision"] == 1.0
    assert report["recall"] >= 0.95
    assert report["unresolved_rate"] <= 0.02


def test_ambiguous_names_need_a_cue_and_otherwise_go_to_review():
    linker = Linker()
    bare = linker.link("Koç burcu için bu hafta finansal uyarı")
    assert bare.tickers == [] and [l.ticker for l in bare.review] == ["KCHOL"]
    assert all(l.confidence < CONFIRM_THRESHOLD for l in bare.review)
    assert linker.link("Koç Holding'den dev satın alma").tickers == ["KCHOL"]


def test_roster_ticker_tokens_need_brackets_or_a_share_cue():
    linker = Linker(known_tickers=["ALTIN", "FRIGO"])
    assert linker.link("ALTIN fiyatları yükselişte").tickers == []
    assert linker.link("FRIGO hisseleri tavan yaptı").tickers == ["FRIGO"]
    assert linker.link("Frigo-Pak (FRIGO) yeni sözleşme imzaladı").tickers == ["FRIGO"]
    # In an all-capitals headline a capitalised token carries no information.
    assert linker.link("THYAO VE ASELS İÇİN YENİ HEDEF FİYAT").tickers == []


def test_mention_types_separate_material_casual_sector_and_multi():
    linker = Linker()
    assert linker.link("Aselsan 250 milyon dolarlık sözleşme imzaladı").mention_type == MENTION_MATERIAL
    assert linker.link("Akbank Sanat'ta yeni sergi açıldı").mention_type == MENTION_CASUAL
    assert linker.link("Bankalar endeksi yükseldi, Akbank öne çıktı").mention_type == MENTION_SECTOR
    assert linker.link("Ford Otosan ve Tofaş üretime ara veriyor").mention_type == MENTION_MULTI
    assert linker.link("Merkez Bankası faizi sabit tuttu").mention_type is None


def test_fold_handles_turkish_capitals():
    assert fold("TÜRKİYE İŞ BANKASI A.Ş.") == "turkiye is bankasi a.s."
    assert fold("Şişecam’ın") == "sisecam'in"


# -- KAP taxonomy ----------------------------------------------------------------
def test_category_comes_from_the_form_name_first():
    assert events.classify("Yeni İş İlişkisi", None, None)["category"] == events.NEW_CONTRACT
    assert events.classify("Kar Payı Dağıtım İşlemlerine İlişkin Bildirim", "", "")["category"] == events.DIVIDEND
    unknown = events.classify("Bilinmeyen Yeni Form", "geri alım", "")
    assert unknown == {"category": events.OTHER_FORM, "rule": "unmapped_form_name"}


def test_general_form_uses_only_the_summary_phrase_list():
    general = "Özel Durum Açıklaması (Genel)"
    assert events.classify(general, "Pay Geri Alım İşlemleri Hakkında", "")["category"] == events.BUYBACK
    assert events.classify(general, "Yatırımcı İlişkileri Uzmanının Ayrılması", "")["category"] == events.GENERAL_OTHER
    # A phrase in the body, not the summary, does not assign a category.
    assert events.classify(general, "Bilgilendirme", "geri alım yapılmıştır")["category"] == events.GENERAL_OTHER


def test_capital_form_distinguishes_bonus_from_rights_and_admits_doubt():
    form = "Sermaye Artırımı - Azaltımı İşlemlerine İlişkin Bildirim"
    assert events.classify(form, "Bedelsiz sermaye artırımı", "")["category"] == events.BONUS_ISSUE
    assert events.classify(form, "Bedelli sermaye artırımı", "")["category"] == events.RIGHTS_ISSUE
    assert events.classify(form, "Bedelli ve bedelsiz sermaye artırımı", "")["category"] == events.CAPITAL_CHANGE_OTHER
    assert events.classify(form, "Sermaye azaltımı", "")["category"] == events.CAPITAL_CHANGE_OTHER


def test_update_and_correction_flags_are_read_or_left_unknown():
    body = ("oda_UpdateAnnouncementFlag| Yapılan Açıklama Güncelleme mi? Evet (Yes) "
            "oda_CorrectionAnnouncementFlag| Yapılan Açıklama Düzeltme mi? Hayır (No)")
    flags = events.parse_flags(body)
    assert (flags["is_update"], flags["is_correction"], flags["is_delayed"]) == (True, False, None)
    assert events.parse_flags("") == {"is_update": None, "is_correction": None, "is_delayed": None}


# -- Share transaction forms -----------------------------------------------------
HEADERS = (
    "Pay Alım Satım Bilgileri İşlem Tarihi Alım İşlemine Konu Payların Toplam Nominal Tutarı (TL) "
    "Satım İşlemine Konu Payların Toplam Nominal Tutarı (TL) İşlemlerin Net Nominal Tutarı (TL) "
    "Sahip Olunan Payların Gün Başı Nominal Tutarı (TL) Sahip Olunan Payların Gün Sonu Nominal "
    "Tutarı (TL) Sahip Olunan Payların Gün Başı Bakiyesinin Sermayeye Oranı (%) Sahip Olunan Oy "
    "Haklarının Gün Sonu Bakiyesinin Oy Haklarına Oranı (%) "
)


def _form(explanation, rows):
    return f"oda_ExplanationTextBlock| {explanation} {HEADERS}{rows} Yukarıdaki açıklamalarımızın"


def test_holder_purchase_is_attributed_to_the_related_issuer():
    body = _form("16.02.2023 tarihinde B Şirketi payları ile ilgili olarak 5,42-5,54 TL fiyat "
                 "aralığından 650.000 TL toplam nominal tutarlı alış işlemi Ortaklığımızca "
                 "gerçekleştirilmiştir.",
                 "16/02/2023 650.000 0 650.000 211.581.301,69 212.231.301,69 % 30,43 % 30,52")
    label = events.classify("Pay Alım Satım Bildirimi", "Pay Alım", body, ["AAAAA"], ["BBBBB"])
    assert label["category"] == events.INSIDER_TRADE
    parsed = insider.parse_insider(body, "Pay Alım", ["AAAAA"], ["BBBBB"])
    assert parsed["insider_parse_status"] == insider.PARSED
    assert parsed["insider_kind"] == insider.KIND_THIRD_PARTY
    assert (parsed["insider_side"], parsed["insider_buy_nominal"]) == ("buy", 650000.0)
    assert parsed["insider_price_mid"] == pytest.approx(5.48)
    assert parsed["insider_value"] == pytest.approx(650000 * 5.48)


def test_an_issuers_own_buyback_on_the_share_form_is_not_an_insider_trade():
    body = _form("Şirketimizin kendi paylarının geri alımı ile ilgili olarak 9,71 - 9,97 TL fiyat "
                 "aralığından 470.000 TL toplam nominal tutarlı alış işlemi gerçekleştirilmiştir.",
                 "16/02/2023 470.000 0 470.000 89.264.862,98 89.734.862,98 % 15,83 % 15,91")
    label = events.classify("Pay Alım Satım Bildirimi", "kendi paylarının geri alımı", body,
                            ["NTHOL"], [])
    assert label == {"category": events.BUYBACK, "rule": "share_form_own_shares"}


def test_sales_mixed_days_transfers_and_unreadable_forms():
    sale = insider.parse_insider(_form("satış işlemi gerçekleştirilmiştir.",
                                       "01/03/2023 0 120.000 -120.000 500.000 380.000 % 5 % 4"))
    assert sale["insider_side"] == "sell" and sale["insider_value"] is None
    mixed = insider.parse_insider(_form("işlemler gerçekleştirilmiştir.",
                                        "01/03/2023 10.000 0 10.000 50.000 60.000 % 1 % 1 "
                                        "02/03/2023 0 4.000 -4.000 60.000 56.000 % 1 % 1"))
    assert (mixed["insider_side"], mixed["insider_buy_nominal"], mixed["insider_sell_nominal"]) == (
        "mixed", 10000.0, 4000.0)
    transfer = insider.parse_insider(_form("Borsa dışı devir işlemi ile paylar alınmıştır.",
                                           "01/03/2023 10.000 0 10.000 50.000 60.000 % 1 % 1"))
    assert transfer["insider_parse_status"] == insider.EXCLUDED
    assert insider.parse_insider("Serbest metin, tablo yok")["insider_parse_status"] == insider.UNPARSED
    # A date and figures in the explanation are not a table row.
    stray = insider.parse_insider("01/03/2023 10.000 0 10.000 tarihli işlem")
    assert stray["insider_parse_status"] == insider.UNPARSED
    # A table with headers and no rows (details in an attachment) is unparsed.
    assert insider.parse_insider(_form("açıklama ekte yer almaktadır.", ""))[
        "insider_parse_status"] == insider.UNPARSED


def test_a_blank_table_cell_cannot_turn_a_sale_into_a_purchase():
    """Real form 1205583: the "bought" cell is empty and leaves no trace, so the
    row reads as two equal amounts. Holdings fell, so it is a sale."""

    body = _form("Şirketimiz hissedarlarından Sn. X, 190.000 TL nominal bedelli 190.000 adedinin "
                 "satışını Borsa İstanbul A.Ş. nezdinde gerçekleştirmiş olup",
                 "12/10/2023 190.000 190.000 22.966.000 22.776.000 % 44,17 % 44,17 % 43,8 % 43,8")
    parsed = insider.parse_insider(body, "Pay Alım Satım Bildirimi", ["SRVGY"], [])
    # "nominal bedelli" is "with a nominal value of", not a rights issue.
    assert parsed["insider_parse_status"] == insider.PARSED and parsed["insider_exclusion"] is None
    assert (parsed["insider_side"], parsed["insider_sell_nominal"], parsed["insider_buy_nominal"]) == (
        "sell", 190000.0, 0.0)
    assert parsed["insider_venue_stated"] and parsed["insider_role"] == "shareholder"
    # The mirror case: one amount, holdings rose.
    bought = insider.parse_insider(_form("alış işlemi", "12/10/2023 5.000 5.000 100.000 105.000 % 1 % 1"))
    assert (bought["insider_side"], bought["insider_buy_nominal"]) == ("buy", 5000.0)


def test_a_row_that_does_not_reconcile_with_holdings_is_not_parsed():
    # Claims 10,000 bought, but holdings went from 50,000 to 50,300.
    odd = insider.parse_insider(_form("alış işlemi", "01/03/2023 10.000 0 10.000 50.000 50.300 % 1 % 1"))
    assert odd["insider_parse_status"] == insider.UNPARSED and odd["insider_side"] is None
    rows = insider.transaction_rows(_form("x", "01/03/2023 10.000 0 10.000 50.000 50.300 % 1 % 1"))
    assert rows == [{"date": "01/03/2023", "bought": None, "sold": None, "reconciled": False}]
    # A capital-increase allotment is excluded by name.
    allot = insider.parse_insider(_form("Bedelli sermaye artırımı kapsamında alınan paylar",
                                        "01/03/2023 10.000 0 10.000 50.000 60.000 % 1 % 1"))
    assert allot["insider_parse_status"] == insider.EXCLUDED


def test_event_builder_uses_related_issuer_for_holder_trades_and_prior_liquidity():
    panel = fixtures.synthetic_panel(tickers=3, sessions=260, seed=4)
    panel.bars["AAAAA.IS"], panel.bars["BBBBB.IS"] = panel.bars["S000"], panel.bars["S001"]
    panel.bars["BBBBX.IS"] = panel.bars["S002"].assign(turnover=panel.bars["S002"]["turnover"] * 1e-3)
    day = panel.calendar.sessions[200]
    stamp = f"{day[8:10]}.{day[5:7]}.{day[:4]} 11:00:00"
    body = _form("5,00-5,20 TL fiyat aralığından alış işlemi Ortaklığımızca gerçekleştirilmiştir.",
                 "01/03/2023 1.000 0 1.000 20.000 21.000 % 1 % 1")
    details = pd.DataFrame([
        {"disclosure_index": 1, "sender_id": "9", "sender_title": "A", "sender_codes": '["AAAAA"]',
         "related_stocks": '[{"code": "BBBBB"}]', "published_raw": stamp, "disclosure_type": "ODA",
         "subject_tr": "Pay Alım Satım Bildirimi", "summary_tr": "Pay alım", "body_text": body,
         "source": "t"},
        {"disclosure_index": 2, "sender_id": "8", "sender_title": "B", "sender_codes": '["BBBBX", "BBBBB"]',
         "related_stocks": "[]", "published_raw": stamp, "disclosure_type": "ODA",
         "subject_tr": "Yeni İş İlişkisi", "summary_tr": "Sözleşme", "body_text": "", "source": "t"},
    ])
    built = events.build_kap_events(details, panel).set_index("disclosure_index")
    assert built.loc[1, "ticker"] == "BBBBB.IS" and built.loc[1, "names_other_issuer"]
    assert built.loc[1, "insider_side"] == "buy"
    # Two lines for one issuer: the one with more turnover before day 0 wins.
    assert built.loc[2, "ticker"] == "BBBBB.IS"
    assert built.loc[2, "day0"] == day and built.loc[2, "bucket"] == "during_session"


# -- Price limits -----------------------------------------------------------------
def _bar(days, rows):
    frame = pd.DataFrame(rows, index=days, columns=["open", "high", "low", "close"]).astype(float)
    frame["adj_close"], frame["volume"] = frame["close"], 1000.0
    frame["dividend"], frame["split_ratio"] = 0.0, 0.0
    return frame


def test_touching_a_limit_is_not_closing_locked_at_it():
    days = fixtures.weekdays("2023-01-02", 5)
    stock = _bar(days, [
        [100, 100, 100, 100],
        [100, 110, 100, 110],      # closes locked at +10%
        [110, 121, 108, 112],      # touches +10% and falls back
        [112, 112, 100.8, 100.8],  # closes locked at -10%
        [90.72, 90.72, 90.72, 90.72],  # one price all day at -10%
    ])
    market = _bar(days, [[100, 100, 100, 100]] * 5)
    panel = prices.build_panel({"MKT": market, "X": stock}, "MKT", origin="synthetic", snapshot_id="t")
    flags = limits.detect(panel.frame("X"))
    assert flags["up_locked_close"].tolist() == [False, True, False, False, False]
    assert flags["up_touched"].tolist() == [False, True, True, False, False]
    assert flags["down_locked_close"].tolist() == [False, False, False, True, True]
    assert flags["one_price"].tolist()[-1] and not flags["one_price"].tolist()[3]
    assert limits.streaks(flags["down_locked_close"]).tolist() == [0, 0, 0, 1, 2]


def test_no_limit_is_detected_without_a_verified_rule_or_on_action_dates():
    days = fixtures.weekdays("2019-06-03", 3)
    stock = _bar(days, [[100, 100, 100, 100], [100, 110, 100, 110], [110, 121, 110, 121]])
    market = _bar(days, [[100, 100, 100, 100]] * 3)
    panel = prices.build_panel({"MKT": market, "X": stock}, "MKT", origin="synthetic", snapshot_id="t")
    assert limits.limit_on("2019-06-04") is None
    assert not limits.detect(panel.frame("X"))["up_locked_close"].any()

    days = fixtures.weekdays("2023-01-02", 3)
    stock = _bar(days, [[100, 100, 100, 100], [100, 110, 100, 110], [110, 121, 110, 121]])
    stock.loc[days[1], "split_ratio"] = 2.0
    market = _bar(days, [[100, 100, 100, 100]] * 3)
    panel = prices.build_panel({"MKT": market, "X": stock}, "MKT", origin="synthetic", snapshot_id="t")
    assert limits.detect(panel.frame("X"))["up_locked_close"].tolist() == [False, False, True]


def test_empirical_check_counts_moves_beyond_the_rule():
    context = fixtures.limit_context(limit_downs=30, streak_starts=10)
    check = limits.empirical_check(context.panel)
    assert check["at_limit"] >= 40 and check["beyond_limit"] == 0


# -- Costs and fills ---------------------------------------------------------------
def test_fill_rules_err_toward_not_filled():
    days = fixtures.weekdays("2023-01-02", 4)
    stock = _bar(days, [[100, 100, 100, 100], [110, 110, 110, 110], [110, 112, 109, 111],
                        [111, 112, 110, 111]])
    stock.loc[days[3], "volume"] = 0.0
    market = _bar(days, [[100, 100, 100, 100]] * 4)
    panel = prices.build_panel({"MKT": market, "X": stock}, "MKT", origin="synthetic", snapshot_id="t")
    frame = panel.frame("X")
    assert costs.entry_fill(frame, days[1], limit=0.10) == costs.FILL_LOCKED
    assert costs.entry_fill(frame, days[2], limit=0.10) == costs.FILL_OK
    # A zero-volume bar is a carried-forward price: in the panel it is no bar.
    assert costs.entry_fill(frame, days[3], limit=0.10) == costs.FILL_NO_BAR
    assert costs.entry_fill(frame, "2030-01-01") == costs.FILL_NO_BAR
    assert costs.entry_fill(None, days[1]) == costs.FILL_NO_BAR


def test_net_returns_subtract_each_registered_scenario():
    summary = costs.net_summary([0.01, 0.03, None, float("nan")])
    assert summary["trades"] == 2 and summary["gross_mean"] == pytest.approx(0.02)
    assert summary["net_mean"] == pytest.approx(
        {"round_trip_30bp": 0.017, "round_trip_40bp": 0.016, "round_trip_50bp": 0.015})
    assert costs.CostModel().round_trip == pytest.approx(0.004)
    assert costs.max_drawdown([0.1, -0.2, 0.05]) == pytest.approx(-0.2)
    assert costs.net_summary([])["gross_mean"] is None


# -- Guards and the store ------------------------------------------------------------
def test_sealed_guard_lifts_only_when_the_index_result_exists(tmp_path):
    path = tmp_path / "index.db"
    assert guards.sealed_from(path) == "2026-08-10"            # no database
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE future_validation_results (definition_hash TEXT)")
    connection.commit()
    assert guards.sealed_from(path) == "2026-08-10"            # table, no result
    connection.execute("INSERT INTO future_validation_results VALUES ('x')")
    connection.commit()
    connection.close()
    assert guards.sealed_from(path) is None
    with pytest.raises(guards.SealedWindowError):
        guards.assert_outcome_allowed("2026-08-10", sealed="2026-08-10")
    guards.assert_outcome_allowed("2026-08-07", sealed="2026-08-10")


def test_raw_tables_are_append_only_and_separate_from_the_index_database(tmp_path):
    path = str(tmp_path / "s.db")
    store.init_db(path)
    frames = {"X": fixtures.synthetic_panel(tickers=1, sessions=30, seed=1).bars["S000"]}
    provider = prices.FrameProvider(frames)
    first = prices.ingest(["X", "MISSING"], "2022-01-01", "2023-01-01", provider, "snap", path)
    again = prices.ingest(["X", "MISSING"], "2022-01-01", "2023-01-01", provider, "snap", path)
    assert (first["ok"], first["no_data"]) == (1, 1) and again["skipped"] == 2
    connection = sqlite3.connect(path)
    try:
        status = dict(connection.execute("SELECT ticker, status FROM sr_price_availability"))
        assert status == {"X": "ok", "MISSING": "no_data"}
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE sr_raw_price_bars SET close = 1")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM sr_raw_price_bars")
        names = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert all(name.startswith("sr_") for name in names)
    finally:
        connection.close()


def test_kap_frame_is_fixed_evenly_spaced_and_fully_ordered():
    blocks = kap.frame_blocks()
    starts = [b["block_start"] for b in blocks]
    gaps = {b - a for a, b in zip(starts, starts[1:])}
    assert len(blocks) == 80 and starts == sorted(starts) and max(gaps) - min(gaps) <= 1
    assert sorted(b["process_order"] for b in blocks) == list(range(80))
    # The first twenty processed already span most of the range.
    early = sorted(b["block_seq"] for b in blocks if b["process_order"] < 20)
    assert early[0] <= 4 and early[-1] >= 75


def test_kap_body_text_keeps_turkish_and_drops_english():
    import base64

    html = ('<html><body><td class="content-tr">Türkçe açıklama</td>'
            '<td class="content-en">English text</td></body></html>')
    detail = {"htmlMessages": [{"tr": base64.b64encode(html.encode()).decode(), "en": None}]}
    text = kap.body_text(detail)
    assert "Türkçe açıklama" in text and "English" not in text
    assert kap.body_text({"htmlMessages": [{"tr": "not-base64!!", "en": None}]}) == ""
