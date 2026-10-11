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
    assert registry.PROTOCOL_VERSION == "stock-research-protocol-v3"
    document = registry.protocol_document()
    assert document["supersedes"]["protocol_version"] == "stock-research-protocol-v2"
    assert [e["protocol_version"] for e in document["lineage"]] == [
        "stock-research-protocol-v1", "stock-research-protocol-v2"]
    assert len(document["changes_from_previous"]) >= 3
    registry.register(path)
    with store.connect(path) as con:
        rows = {r["protocol_hash"]: (r["version"], r["registered_at"])
                for r in con.execute("SELECT * FROM sr_protocols")}
        assert rows[old_hash] == ("stock-research-protocol-v1", old_time)
        assert rows[registry.protocol_hash()][0] == "stock-research-protocol-v3"
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("DELETE FROM sr_protocols WHERE protocol_hash = ?", (old_hash,))


# -- 2. Ingestion completeness is not detail completeness -----------------------
class _Gateway:
    """A stand-in for the KAP API: a range of two listing pages of four
    disclosures each (three of them targets), some details failing."""

    def __init__(self, failing=(), first=None):
        self.failing, self.calls = set(failing), []
        self.first = kap.KAP_FRAME_FIRST_INDEX if first is None else first

    def __call__(self, path, params=None, **_):
        self.calls.append(path)
        if path == "/members":
            return [{"id": "1", "title": "A A.Ş.", "stockCode": "AAAAA", "memberType": "IGS"},
                    {"id": "2", "title": "Fon", "stockCode": "FFF", "memberType": "FK"}]
        if path == "/disclosures":
            start = int(params["disclosureIndex"])
            if start >= self.first + 8:                    # past the end of the range
                return []
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
def census(monkeypatch, tmp_path):
    monkeypatch.setattr(kap, "KAP_LISTING_PAGE", 4)        # a full page is four items here
    return str(tmp_path / "k.db"), kap.KAP_FRAME_FIRST_INDEX


def _frame(path):
    with store.connect(path) as con:
        return {r["block_seq"]: dict(r) for r in con.execute("SELECT * FROM sr_kap_frame")}


def _clock(monkeypatch, day):
    monkeypatch.setattr(store, "now_iso", lambda: f"{day}T10:00:00+00:00")


def test_a_page_with_an_unavailable_detail_is_not_complete(monkeypatch, census):
    path, first = census
    gateway = _Gateway(failing={first + 1})
    monkeypatch.setattr(kap, "api_get", gateway)
    kap.run(path, max_blocks=1)

    block = _frame(path)[0]
    assert block["targets"] == 3                          # the fund filing is not a target
    assert block["status"] == "incomplete_details"
    with store.connect(path) as con:
        stored = [r[0] for r in con.execute("SELECT disclosure_index FROM sr_raw_kap_detail")]
        failures = [dict(r) for r in con.execute("SELECT * FROM sr_kap_detail_failures")]
    assert sorted(stored) == [first, first + 2]           # no placeholder row for the gap
    assert [(f["disclosure_index"], f["http_status"]) for f in failures] == [(first + 1, 404)]

    progress = kap.progress(path)
    assert progress["details_stored"] == 2 and progress["details_unavailable"] == 1
    assert progress["blocks_complete"] == 0 and progress["blocks_attempted"] == 1
    assert not progress["listing_complete"] and not progress["details_complete"]
    assert kap.unavailable_details(path) == [first + 1]


def test_the_census_walks_the_range_page_by_page_to_its_end(monkeypatch, census):
    path, first = census
    gateway = _Gateway()
    monkeypatch.setattr(kap, "api_get", gateway)
    kap.run(path)
    frame = _frame(path)
    # Each page starts where the one before it ended; the empty page ends it.
    assert [frame[k]["block_start"] for k in sorted(frame)] == [first, first + 4, first + 8]
    assert kap.last_page(path) == 2
    progress = kap.progress(path)
    assert progress["kind"] == "census" and progress["listing_complete"]
    assert progress["ingestion_complete"] and progress["details_complete"]
    assert progress["frame_complete"] and progress["targets"] == 6
    assert progress["range_listed_share"] == 1.0 and progress["blocks_total"] == 3
    assert kap.verify_details(path)["verified"]


def test_ingestion_complete_and_details_complete_are_reported_separately(monkeypatch, census):
    path, first = census
    monkeypatch.setattr(kap, "api_get", _Gateway(failing={first + 1}))
    kap.run(path)                                          # the whole range, one gap
    progress = kap.progress(path)
    assert progress["ingestion_complete"] and not progress["details_complete"]
    assert progress["details_unavailable"] == 1 and progress["targets"] == 6
    assert progress["unavailable_share"] == pytest.approx(1 / 6)
    assert progress["details_awaiting_another_attempt"] == 1
    assert not progress["within_tolerance"] and not progress["frame_complete"]


def test_a_refused_detail_is_asked_for_once_a_day_and_three_times_in_all(monkeypatch, census):
    path, first = census
    failing = _Gateway(failing={first + 1})
    monkeypatch.setattr(kap, "api_get", failing)
    attempts = lambda: len([c for c in failing.calls if c.endswith(f"/{first + 1}")])   # noqa: E731

    _clock(monkeypatch, "2026-11-01")
    kap.run(path)
    kap.run(path)                                          # same day: not asked again
    assert attempts() == 1
    for day in ("2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"):
        _clock(monkeypatch, day)
        kap.run(path)
    assert attempts() == kap.MAX_DETAIL_ATTEMPTS == 3      # capped
    progress = kap.progress(path)
    # Refused on three separate days: it may now be accepted as unavailable,
    # but one in six is far above the registered share.
    assert progress["details_awaiting_another_attempt"] == 0
    assert progress["details_unavailable"] == 1 and not progress["within_tolerance"]
    monkeypatch.setattr(kap, "KAP_MAX_UNAVAILABLE_SHARE", 0.2)
    assert kap.progress(path)["within_tolerance"] and kap.progress(path)["frame_complete"]


def test_a_later_run_fills_the_gap(monkeypatch, census):
    path, first = census
    _clock(monkeypatch, "2026-11-01")
    monkeypatch.setattr(kap, "api_get", _Gateway(failing={first + 1}))
    kap.run(path)
    _clock(monkeypatch, "2026-11-02")
    monkeypatch.setattr(kap, "api_get", _Gateway())
    kap.run(path)
    assert _frame(path)[0]["status"] == "complete"
    assert kap.progress(path)["details_unavailable"] == 0
    assert kap.progress(path)["frame_complete"]


def test_rows_of_an_earlier_frame_are_reused_and_left_as_they_were(monkeypatch, census):
    """A disclosure the anchor sample already holds is a member of the census.
    Its listing row and its detail are not fetched again and not rewritten."""

    path, first = census
    store.init_db(path)
    old = kap.detail_row({"disclosureIndex": str(first + 1), "senderId": "1",
                          "subject": {"tr": "Eski"}, "time": "02.01.2023 09:00:00"}, b"old")
    with store.connect(path) as con:
        con.execute("INSERT INTO sr_raw_kap_listing VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (first + 1, "kap-anchor-sample-v2", 7, "ODA", "ODA", "1", "A", "{}", "t",
                     "2026-10-11T00:00:00+00:00"))
        con.execute(
            "INSERT INTO sr_raw_kap_detail VALUES (:disclosure_index,:sender_id,:sender_title,"
            ":sender_codes,:related_stocks,:disclosure_type,:disclosure_class,:disclosure_reason,"
            ":subject_tr,:subject_en,:summary_tr,:published_raw,:event_type_code,:body_text,"
            ":metadata_json,:payload_sha256,:source,:retrieved_at)", old)
    gateway = _Gateway()
    monkeypatch.setattr(kap, "api_get", gateway)
    kap.run(path)

    assert f"/disclosureDetail/{first + 1}" not in gateway.calls       # not fetched again
    with store.connect(path) as con:
        listing = dict(con.execute("SELECT * FROM sr_raw_kap_listing WHERE disclosure_index=?",
                                   (first + 1,)).fetchone())
        detail = dict(con.execute("SELECT * FROM sr_raw_kap_detail WHERE disclosure_index=?",
                                  (first + 1,)).fetchone())
        member = dict(con.execute("SELECT * FROM sr_kap_frame_items WHERE disclosure_index=?",
                                  (first + 1,)).fetchone())
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("DELETE FROM sr_kap_frame_items")
    assert (listing["frame_version"], listing["block_seq"]) == ("kap-anchor-sample-v2", 7)
    assert detail["subject_tr"] == "Eski" and detail["payload_sha256"] == old["payload_sha256"]
    assert (member["frame_version"], member["block_seq"], member["is_target"]) == (
        kap.KAP_FRAME_VERSION, 0, 1)
    progress = kap.progress(path)
    assert progress["targets"] == 6 and progress["details_complete"]
    assert progress["targets_reused_from_earlier_frames"] == 1


def test_a_placeholder_row_from_the_old_ingester_counts_as_unavailable(monkeypatch, census):
    """Before the hardening a failed detail was stored as a row with no
    timestamp. Such rows may exist; they are gaps."""

    path, first = census
    store.init_db(path)
    row = kap.detail_row({"disclosureIndex": str(first + 2), "unavailable": "404"}, b"x")
    with store.connect(path) as con:
        con.execute(
            "INSERT INTO sr_raw_kap_detail VALUES (:disclosure_index,:sender_id,:sender_title,"
            ":sender_codes,:related_stocks,:disclosure_type,:disclosure_class,:disclosure_reason,"
            ":subject_tr,:subject_en,:summary_tr,:published_raw,:event_type_code,:body_text,"
            ":metadata_json,:payload_sha256,:source,:retrieved_at)", row)
    monkeypatch.setattr(kap, "api_get", _Gateway())
    kap.run(path)
    progress = kap.progress(path)
    assert progress["details_unavailable"] == 1 and not progress["details_complete"]
    assert kap.unavailable_details(path) == [first + 2]


def test_a_stored_detail_that_cannot_be_dated_fails_verification(monkeypatch, census):
    path, first = census

    class Undated(_Gateway):
        def __call__(self, path, params=None, **kw):
            out = super().__call__(path, params, **kw)
            if path.endswith(f"/{first + 2}"):
                out["time"] = "yesterday"
            return out

    monkeypatch.setattr(kap, "api_get", Undated())
    kap.run(path)
    check = kap.verify_details(path)
    assert not check["verified"] and check["timestamp_unparseable"] == [first + 2]


def _gap(path, index, days=()):
    """One more target with no detail, refused on the given days."""

    with store.connect(path) as con:
        con.execute("INSERT INTO sr_kap_frame_items VALUES (?,?,?,?,?)",
                    (kap.KAP_FRAME_VERSION, index, 0, 1, "now"))
        for day in days:
            con.execute(
                "INSERT INTO sr_kap_detail_failures (disclosure_index, frame_version, block_seq, "
                "http_status, error, source, attempted_at) VALUES (?,?,?,?,?,?,?)",
                (index, kap.KAP_FRAME_VERSION, 0, 404, "x", "t", f"{day}T10:00:00+00:00"))


def test_real_run_needs_complete_retrieval_or_three_refusals_within_the_share(tmp_path, monkeypatch):
    # One refusal so far: the run waits for the other attempts.
    waiting = _database(str(tmp_path / "waiting.db"))
    _gap(waiting, 9_000_001, days=("2026-11-01",))
    registry.register(waiting)
    with pytest.raises(pipeline.NotReady, match="unavailable"):
        pipeline.run_real(waiting, index_db=tmp_path / "none.db")

    # Three refusals on three days, but one in 361 is above the registered 0.1%.
    refused = _database(str(tmp_path / "refused.db"))
    _gap(refused, 9_000_001, days=("2026-11-01", "2026-11-02", "2026-11-03"))
    registry.register(refused)
    with pytest.raises(pipeline.NotReady, match="unavailable"):
        pipeline.run_real(refused, index_db=tmp_path / "none.db")
    assert kap.progress(refused)["details_awaiting_another_attempt"] == 0

    # Within the share (raised here), the run proceeds and names the gap.
    monkeypatch.setattr(kap, "KAP_MAX_UNAVAILABLE_SHARE", 0.01)
    outcome = pipeline.run_real(refused, index_db=tmp_path / "none.db")
    sample = outcome["manifest"]["kap_sample"]
    assert (sample["targets"], sample["details_stored"], sample["details_unavailable"]) == (361, 360, 1)
    assert sample["unavailable_indices"] == [9_000_001]
    assert outcome["manifest"]["frame_complete"]


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
    assert one["counts"]["price_bars"] > 0 and one["counts"]["kap_details"] == 360
    assert one["counts"]["second_source_bars"] > 0
    assert len(one["second_source_price_digest"]) == 64
    assert one["second_source_price_digest"] != three["second_source_price_digest"]

    registry.register(first)
    manifest = pipeline.run_real(first, index_db=tmp_path / "none.db")["manifest"]
    assert manifest["provenance"]["kap_sample_digest"] == one["kap_sample_digest"]


def test_each_ingestion_run_is_logged_with_the_ingesters_own_hash(monkeypatch, census):
    path, _ = census
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
