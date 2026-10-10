"""The real-data path end to end, on synthetic inputs shaped like real ones.

The first run on real data is the registered run, so its plumbing cannot be
debugged against real outcomes. This builds a database the way ingestion
would -- frame, listings, details with real form names, price bars under
provider symbols -- and drives the same code the real run uses.
"""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from stock_research import fixtures, pipeline, reporting, store
from stock_research.config import BENCHMARK_TICKER, KAP_FRAME_VERSION
from stock_research.data import kap, prices
from stock_research.hypotheses import registry
from stock_research.hypotheses.common import DATA_INSUFFICIENT

FORMS = [
    ("Yeni İş İlişkisi", "Yeni sözleşme imzalanması", 0.03),
    ("Kar Payı Dağıtım İşlemlerine İlişkin Bildirim", "Kar payı", -0.02),
    ("Genel Kurul İşlemlerine İlişkin Bildirim", "Genel kurul", 0.0),
    ("Özel Durum Açıklaması (Genel)", "Pay geri alım işlemleri hakkında", 0.0),
    ("Özel Durum Açıklaması (Genel)", "Bilgilendirme", 0.0),
]
FLAGS = ("oda_UpdateAnnouncementFlag| Yapılan Açıklama Güncelleme mi? Hayır (No) "
         "oda_CorrectionAnnouncementFlag| Yapılan Açıklama Düzeltme mi? Hayır (No)")


def _database(path, *, complete=True, per_form=60, seed=3):
    rng = np.random.default_rng(seed)
    days = fixtures.weekdays("2022-06-01", 440)
    codes = [f"TK{n:03d}" for n in range(40)]
    effects, details, taken = {}, [], set()
    index = 2_000_000
    for subject, summary, total in FORMS:
        for _ in range(per_form):
            while True:
                key = (str(rng.choice(codes)), int(rng.integers(150, 410)))
                if key not in taken:
                    taken.add(key)
                    break
            code, position = key
            for relative in range(6):
                effects[(f"{code}.IS", position + relative)] = total / 6
            day = days[position - 1]                     # 19:00 the evening before
            details.append({
                "disclosureIndex": str(index), "senderId": code, "senderTitle": f"{code} A.Ş.",
                "senderExchCodes": [code], "relatedStocks": [], "disclosureType": "ODA",
                "disclosureClass": "ODA", "subject": {"tr": subject}, "summary": {"tr": summary},
                "time": f"{day[8:10]}.{day[5:7]}.{day[:4]} 19:00:00", "_body": FLAGS,
            })
            index += 1

    panel = fixtures.make_panel(days, [f"{c}.IS" for c in codes], seed=seed + 1,
                                intraday_effects=effects)
    frames = {t: f[prices.BAR_COLUMNS].dropna(subset=["close"]) for t, f in panel.bars.items()}
    frames[BENCHMARK_TICKER] = frames.pop(fixtures.BENCHMARK)

    store.init_db(path)
    kap.ensure_frame(path)
    provider = prices.FrameProvider(frames, origin=prices.ORIGIN_REAL)
    with store.connect(path) as con:
        for code in codes:
            con.execute("INSERT INTO sr_raw_members VALUES ('m', ?, ?, ?, 'IGS', 't', 'now')",
                        (code, f"{code} A.Ş.", code))
        for detail in details:
            number = int(detail["disclosureIndex"])
            con.execute("INSERT INTO sr_raw_kap_listing VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (number, KAP_FRAME_VERSION, 0, "ODA", "ODA", detail["senderId"],
                         detail["senderTitle"], "{}", "t", "now"))
            row = kap.detail_row(detail, b"raw")
            row["body_text"] = detail["_body"]
            con.execute(
                "INSERT INTO sr_raw_kap_detail VALUES (:disclosure_index,:sender_id,:sender_title,"
                ":sender_codes,:related_stocks,:disclosure_type,:disclosure_class,:disclosure_reason,"
                ":subject_tr,:subject_en,:summary_tr,:published_raw,:event_type_code,:body_text,"
                ":metadata_json,:payload_sha256,:source,:retrieved_at)", row)
        con.execute("UPDATE sr_kap_frame SET status = ?",
                    ("complete" if complete else "listed",))
        if not complete:
            con.execute("UPDATE sr_kap_frame SET status='complete' WHERE block_seq < 10")
    pipeline.ingest_prices(path, provider=provider)
    return path


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    return _database(str(tmp_path_factory.mktemp("sr") / "stock.db"))


def test_real_run_refuses_before_registration_and_on_a_partial_frame(tmp_path, database):
    with pytest.raises(pipeline.NotReady, match="not registered"):
        pipeline.run_real(_database(str(tmp_path / "unregistered.db"), per_form=5),
                          index_db=tmp_path / "none.db")
    partial = _database(str(tmp_path / "partial.db"), complete=False, per_form=5)
    registry.register(partial)
    with pytest.raises(pipeline.NotReady, match="incomplete"):
        pipeline.run_real(partial, index_db=tmp_path / "none.db")


def test_describe_reports_counts_without_any_return(database, tmp_path):
    ctx, meta = pipeline.real_context(database, index_db=tmp_path / "none.db")
    summary = pipeline.describe(ctx, meta)
    assert summary["kap_events"] == 300 and summary["ticker_resolved"] == 300
    assert summary["by_category"] == {
        "new_contract": 60, "dividend": 60, "administrative": 60, "buyback": 60, "general_other": 60}
    assert summary["by_bucket"] == {"post_close": 300}
    text = json.dumps(summary, default=str)
    assert "car_" not in text and "mar_" not in text
    assert not summary["availability"]["news_events"]["available"]
    assert not summary["availability"]["point_in_time_market_cap"]["available"]


def test_registered_run_goes_end_to_end_and_is_stored_once(database, tmp_path):
    registry.register(database)
    outcome = pipeline.run_real(database, index_db=tmp_path / "none.db")
    results, manifest = outcome["results"], outcome["manifest"]

    assert manifest["protocol_hash"] == registry.protocol_hash()
    assert manifest["frame_complete"] and manifest["sample"]["kap_events"] == 300
    h7 = results["H7"]
    assert h7["status"] == "supported" and h7["primary"]["joint_test"]
    assert (h7["primary"]["highest_category"], h7["primary"]["lowest_category"]) == (
        "new_contract", "dividend")
    assert h7["primary"]["p_holm"] == pytest.approx(min(1.0, 9 * h7["primary"]["p"]))
    # No insider forms in this sample, and no news: everything else says why.
    for key in ("H1", "H2", "H3", "H4", "H5", "H6", "H8", "H9"):
        assert results[key]["status"] == DATA_INSUFFICIENT
        assert results[key]["sufficiency"]["reasons"]

    stored = pipeline.latest_results(database)
    assert stored["manifest"]["manifest_hash"] == manifest["manifest_hash"]
    assert set(stored["results"]) == set(results)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT COUNT(*) FROM sr_results").fetchone()[0] == 9
    finally:
        connection.close()


def test_report_marks_origin_and_writes_data_files(tmp_path):
    results = {"H9": registry.RUNNERS["H9"](fixtures.language_context(effect=True)),
               "H1": registry.RUNNERS["H1"](fixtures.kap_context())}
    results["H9"]["primary"]["p_holm"] = min(1.0, 9 * results["H9"]["primary"]["p"])
    results["H9"]["status"] = "supported"
    files = reporting.write_report(results, output_dir=tmp_path, name="t")
    page = open(files["html"], encoding="utf-8").read()
    assert "results computed on synthetic data" in page
    assert "supported · synthetic" in page and "This is not a finding" in page
    assert "data insufficient" in page and "data:image/png;base64" in page
    summary = open(files["summary_csv"], encoding="utf-8").read().splitlines()
    assert summary[0].startswith("hypothesis,title,status,data_origin,claim_allowed")
    assert json.load(open(files["results_json"], encoding="utf-8"))["results"]["H1"]["status"] == DATA_INSUFFICIENT
