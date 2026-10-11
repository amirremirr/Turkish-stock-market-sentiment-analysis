"""Regression tests for the follow-up source review of 2026-10-11.

Three confirmed problems, each reproduced here:

1. the protocol fingerprint left out the ingester, which decides the sample;
2. an anchor was marked complete although a disclosure detail was unavailable;
3. H8 treated a purchase as on-exchange unless the text said otherwise.

Nothing here touches the real database or computes a real return.
"""

import json
import sqlite3

import pandas as pd
import pytest
import requests

from stock_research import fixtures as F
from stock_research import insider, pipeline, store
from stock_research.data import kap
from stock_research.hypotheses import common, h8, registry
from stock_research.hypotheses.common import DATA_INSUFFICIENT

from tests.stock_research.test_components import _form
from tests.stock_research.test_pipeline import _database


# -- 1. The fingerprint covers everything that decides the sample ---------------
def test_protocol_fingerprint_covers_the_ingester():
    files = registry.code_fingerprint()
    assert "data/kap.py" in files
    # Only files that display or dispatch are left out.
    assert registry._NOT_RESULT_BEARING == {"fixtures.py", "reporting.py", "cli.py",
                                            "registry_doc.py"}
    constants = registry.protocol_document()["constants"]["kap_frame"]
    assert constants["listed_member_mark"] == "IGS"
    assert "max_unavailable_share" in constants


def test_new_protocol_version_and_changelog_leave_the_old_registration_alone(tmp_path):
    path = str(tmp_path / "s.db")
    store.init_db(path)
    old_hash, old_time = "9" * 64, "2026-10-11T00:15:04+00:00"
    with store.connect(path) as con:
        con.execute("INSERT INTO sr_protocols VALUES (?,?,?,?)",
                    (old_hash, "stock-research-protocol-v1", "{}", old_time))
    assert registry.PROTOCOL_VERSION == "stock-research-protocol-v2"
    document = registry.protocol_document()
    assert document["supersedes"]["protocol_version"] == "stock-research-protocol-v1"
    assert len(document["changes_from_previous"]) >= 3
    registry.register(path)
    with store.connect(path) as con:
        rows = {r["protocol_hash"]: (r["version"], r["registered_at"])
                for r in con.execute("SELECT * FROM sr_protocols")}
        assert rows[old_hash] == ("stock-research-protocol-v1", old_time)
        assert rows[registry.protocol_hash()][0] == "stock-research-protocol-v2"
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("DELETE FROM sr_protocols WHERE protocol_hash = ?", (old_hash,))


# -- 2. Ingestion completeness is not detail completeness -----------------------
class _Gateway:
    """A stand-in for the KAP API: one anchor's listing, some details failing."""

    def __init__(self, failing=()):
        self.failing, self.calls = set(failing), []

    def __call__(self, path, params=None, **_):
        self.calls.append(path)
        if path == "/members":
            return [{"id": "1", "title": "A A.Ş.", "stockCode": "AAAAA", "memberType": "IGS"},
                    {"id": "2", "title": "Fon", "stockCode": "FFF", "memberType": "FK"}]
        if path == "/disclosures":
            start = int(params["disclosureIndex"])
            return [{"disclosureIndex": str(start + k), "disclosureType": "ODA",
                     "disclosureClass": "ODA", "companyId": "1" if k < 3 else "2",
                     "title": "A"} for k in range(4)]
        index = int(path.rsplit("/", 1)[1])
        if index in self.failing:
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError("404 Client Error", response=response)
        return {"disclosureIndex": str(index), "senderId": "1", "senderTitle": "A A.Ş.",
                "senderExchCodes": ["AAAAA"], "relatedStocks": [], "disclosureType": "ODA",
                "disclosureClass": "ODA", "subject": {"tr": "Yeni İş İlişkisi"},
                "summary": {"tr": "x"}, "time": "02.01.2023 19:00:00", "htmlMessages": []}


@pytest.fixture
def one_anchor(monkeypatch, tmp_path):
    monkeypatch.setattr(kap, "KAP_FRAME_BLOCKS", 2)
    monkeypatch.setattr(kap, "KAP_FRAME_STRIDE", 1)
    path = str(tmp_path / "k.db")
    first = kap.frame_blocks()[0]["block_start"]
    return path, first


def _frame(path):
    with store.connect(path) as con:
        return {r["block_seq"]: dict(r) for r in con.execute("SELECT * FROM sr_kap_frame")}


def test_an_anchor_with_an_unavailable_detail_is_not_complete(monkeypatch, one_anchor):
    path, first = one_anchor
    gateway = _Gateway(failing={first + 1})
    monkeypatch.setattr(kap, "api_get", gateway)
    kap.run(path, max_blocks=1)

    block = _frame(path)[0]
    assert block["targets"] == 3                          # the fund filing is not a target
    assert block["status"] == "incomplete_details"        # was 'complete' before the fix
    with store.connect(path) as con:
        stored = [r[0] for r in con.execute("SELECT disclosure_index FROM sr_raw_kap_detail")]
        failures = [dict(r) for r in con.execute("SELECT * FROM sr_kap_detail_failures")]
    assert sorted(stored) == [first, first + 2]           # no placeholder row for the gap
    assert [(f["disclosure_index"], f["http_status"]) for f in failures] == [(first + 1, 404)]

    progress = kap.progress(path)
    assert progress["details_stored"] == 2 and progress["details_unavailable"] == 1
    assert progress["blocks_complete"] == 0 and progress["blocks_attempted"] == 1
    assert not progress["details_complete"]
    assert kap.unavailable_details(path) == [first + 1]


def test_ingestion_complete_and_details_complete_are_reported_separately(monkeypatch, one_anchor):
    path, first = one_anchor
    monkeypatch.setattr(kap, "api_get", _Gateway(failing={first + 1}))
    kap.run(path)                                          # both anchors, one gap
    progress = kap.progress(path)
    assert progress["ingestion_complete"] and not progress["details_complete"]
    assert progress["details_unavailable"] == 1 and progress["targets"] == 6
    assert progress["unavailable_share"] == pytest.approx(1 / 6)
    assert not progress["within_tolerance"] and not progress["frame_complete"]


def test_a_later_run_fills_the_gap_and_gives_up_after_three_attempts(monkeypatch, one_anchor):
    path, first = one_anchor
    failing = _Gateway(failing={first + 1})
    monkeypatch.setattr(kap, "api_get", failing)
    for _ in range(5):                                     # 1 + retries, capped
        kap.run(path, max_blocks=1)
    attempts = [c for c in failing.calls if c.endswith(f"/{first + 1}")]
    assert len(attempts) == kap.MAX_DETAIL_ATTEMPTS == 3
    assert _frame(path)[0]["status"] == "incomplete_details"

    # A different database where the gateway recovers before the cap.
    path2 = path.replace("k.db", "k2.db")
    monkeypatch.setattr(kap, "api_get", _Gateway(failing={first + 1}))
    kap.run(path2, max_blocks=1)
    monkeypatch.setattr(kap, "api_get", _Gateway())
    kap.run(path2)                                         # finishes anchor 1, then retries
    assert _frame(path2)[0]["status"] == "complete"
    assert kap.progress(path2)["details_unavailable"] == 0


def test_a_placeholder_row_from_the_old_ingester_counts_as_unavailable(monkeypatch, one_anchor):
    """Before the fix a failed detail was stored as a row with no timestamp and
    the anchor was marked complete. Such rows may exist; they are gaps."""

    path, first = one_anchor
    monkeypatch.setattr(kap, "api_get", _Gateway())
    kap.run(path)
    assert kap.progress(path)["details_complete"]
    kap.ensure_frame(path)
    with store.connect(path) as con:                       # an extra target, served as a stub
        con.execute("UPDATE sr_kap_frame SET targets = targets + 1 WHERE block_seq = 0")
        con.execute("INSERT INTO sr_raw_kap_listing VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (first + 40, kap.KAP_FRAME_VERSION, 0, "ODA", "ODA", "1", "A",
                     json.dumps({"disclosureIndex": str(first + 40), "disclosureClass": "ODA",
                                 "companyId": "1"}), "t", "now"))
        row = kap.detail_row({"disclosureIndex": str(first + 40), "unavailable": "404"}, b"x")
        con.execute(
            "INSERT INTO sr_raw_kap_detail VALUES (:disclosure_index,:sender_id,:sender_title,"
            ":sender_codes,:related_stocks,:disclosure_type,:disclosure_class,:disclosure_reason,"
            ":subject_tr,:subject_en,:summary_tr,:published_raw,:event_type_code,:body_text,"
            ":metadata_json,:payload_sha256,:source,:retrieved_at)", row)
    progress = kap.progress(path)
    assert progress["details_unavailable"] == 1 and not progress["details_complete"]
    assert kap.unavailable_details(path) == [first + 40]


def test_real_run_tolerates_a_small_recorded_gap_and_refuses_a_large_one(tmp_path):
    small = _database(str(tmp_path / "small.db"))
    with store.connect(small) as con:                      # 300 stored of 302 targets
        con.execute("UPDATE sr_kap_frame SET targets = 0")
        con.execute("UPDATE sr_kap_frame SET targets = 302 WHERE block_seq = 0")
    registry.register(small)
    outcome = pipeline.run_real(small, index_db=tmp_path / "none.db")
    sample = outcome["manifest"]["kap_sample"]
    assert (sample["targets"], sample["details_stored"], sample["details_unavailable"]) == (302, 300, 2)
    assert sample["unavailable_share"] == pytest.approx(2 / 302)
    assert outcome["manifest"]["frame_complete"]

    large = _database(str(tmp_path / "large.db"))
    with store.connect(large) as con:                      # 300 stored of 310: over 1%
        con.execute("UPDATE sr_kap_frame SET targets = 0")
        con.execute("UPDATE sr_kap_frame SET targets = 310 WHERE block_seq = 0")
    registry.register(large)
    with pytest.raises(pipeline.NotReady, match="unavailable"):
        pipeline.run_real(large, index_db=tmp_path / "none.db")


# -- Provenance -----------------------------------------------------------------
def test_manifest_fingerprints_the_data_it_was_computed_from(tmp_path):
    first = _database(str(tmp_path / "a.db"))
    same = _database(str(tmp_path / "b.db"))
    other = _database(str(tmp_path / "c.db"), seed=4)
    one, two, three = (pipeline.dataset_provenance(p) for p in (first, same, other))
    for key in ("kap_sample_digest", "price_snapshot_digest", "members_snapshot_digest"):
        assert len(one[key]) == 64 and one[key] == two[key]
    assert one["kap_sample_digest"] != three["kap_sample_digest"]
    assert one["price_snapshot_digest"] != three["price_snapshot_digest"]
    assert one["counts"]["price_bars"] > 0 and one["counts"]["kap_details"] == 300

    registry.register(first)
    manifest = pipeline.run_real(first, index_db=tmp_path / "none.db")["manifest"]
    assert manifest["provenance"]["kap_sample_digest"] == one["kap_sample_digest"]


def test_each_ingestion_run_is_logged_with_the_ingesters_own_hash(monkeypatch, one_anchor):
    path, _ = one_anchor
    monkeypatch.setattr(kap, "api_get", _Gateway())
    kap.run(path)
    with store.connect(path) as con:
        events = [dict(r) for r in con.execute("SELECT * FROM sr_ingest_log ORDER BY event_id")]
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("DELETE FROM sr_ingest_log")
    assert [e["event"] for e in events] == ["start", "finish"]
    assert events[0]["run_id"] == events[1]["run_id"]
    assert events[0]["ingester_sha256"] == registry.code_fingerprint()["data/kap.py"]
    assert json.loads(events[1]["detail_json"])["details_stored"] == 6


# -- 3. Venue must be shown, not assumed ----------------------------------------
ROW = "16/02/2023 650.000 0 650.000 1.000.000 1.650.000 % 1 % 1"


@pytest.mark.parametrize("explanation,venue", [
    # The standard template: a price range, no venue. Unknown, not on-exchange.
    ("16.02.2023 tarihinde B payları ile ilgili olarak 5,42-5,54 TL fiyat aralığından 650.000 TL "
     "toplam nominal tutarlı alış işlemi Ortaklığımızca gerçekleştirilmiştir.", "unknown"),
    ("650.000 adet payın alımını Borsa İstanbul A.Ş. nezdinde gerçekleştirmiş olup", "on_exchange"),
    ("Borsa İstanbul'da 5,42 TL fiyattan alış işlemi gerçekleştirilmiştir.", "on_exchange"),
    ("BİAŞ Pay Piyasası'nda alım işlemi yapılmıştır.", "on_exchange"),
    # Saying where the shares are listed is not saying where the trade happened.
    ("Borsa İstanbul'da işlem gören şirket paylarından 650.000 adet alınmıştır.", "unknown"),
])
def test_venue_is_on_exchange_only_when_the_text_says_so(explanation, venue):
    parsed = insider.parse_insider(_form(explanation, ROW))
    assert parsed["insider_parse_status"] == insider.PARSED and parsed["insider_side"] == "buy"
    assert parsed["insider_venue"] == venue
    assert (parsed["insider_venue_evidence"] is not None) == (venue == "on_exchange")


@pytest.mark.parametrize("explanation", [
    "Borsa dışı işlem ile 650.000 adet pay alınmıştır.",
    "Borsa İstanbul dışında gerçekleştirilen işlem ile pay alınmıştır.",
    "Borsa İstanbul A.Ş. dışında pay devri gerçekleşmiştir.",
])
def test_off_exchange_trades_are_excluded_whatever_else_the_text_says(explanation):
    parsed = insider.parse_insider(_form(explanation, ROW))
    assert parsed["insider_parse_status"] == insider.EXCLUDED
    assert parsed["insider_venue"] == "off_exchange"


def _with_venue(ctx, venues):
    events = ctx.kap_events.copy()
    buys = events.index[events["insider_side"] == "buy"]
    events.loc[buys, "insider_venue"] = [venues[i % len(venues)] for i in range(len(buys))]
    return common.Context(panel=ctx.panel, kap_events=events, availability=ctx.availability)


def test_h8_primary_sample_is_venue_confirmed_purchases_only():
    base = F.kap_context(effect=True, insider_buys=120)
    mixed = _with_venue(base, ["on_exchange", "on_exchange", "unknown"])
    events, counts = h8.prepare(mixed)
    assert counts["by_venue"] == {"on_exchange": 80, "unknown": 40}
    assert len(events) == counts["purchase_ticker_days"] == 80
    result = h8.run(mixed)
    assert result["primary"]["n"] <= 80
    broader = [s for s in result["sensitivity"] if "venue is not stated" in s["variant"]][0]
    assert broader["n"] > result["primary"]["n"]
    assert "venue" in registry.SPECS["H8"].eligibility
    assert registry.SPECS["H8"].parameters["venue"] == "on_exchange (stated in the text)"


def test_h8_is_insufficient_when_no_purchase_has_a_confirmed_venue():
    unknown = _with_venue(F.kap_context(effect=True), ["unknown"])
    result = h8.run(unknown)
    assert result["status"] == DATA_INSUFFICIENT and result["primary"] is None
    assert result["sufficiency"]["counts"]["by_venue"] == {"unknown": 90}
    assert "venue" in result["sufficiency"]["reasons"][0]
    # A table built before venue was recorded has no such column: all unknown.
    legacy = F.kap_context(effect=True)
    legacy.kap_events.drop(columns=["insider_venue"], inplace=True, errors="ignore")
    assert h8.run(legacy)["status"] == DATA_INSUFFICIENT
