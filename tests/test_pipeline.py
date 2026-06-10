"""
pytest test suite for the BIST100 sentiment pipeline.

Covers each pipeline layer independently, without loading torch/transformers.

Run:  C:\\fin\\Scripts\\pytest tests/ -v
"""

import os
import sys
from datetime import date

import pandas as pd
import pytest

# Force non-interactive backend before any matplotlib/visualize import.
# Without this, the test runner may pick the Tk backend (partially installed
# on some Windows setups) and fail with TclError when tests run in a full suite.
import matplotlib
matplotlib.use("Agg")

# Make sure project root is importable regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import database as db
from pipeline import aggregate_step
from scraper import _is_relevant, _normalise, _parse_date, classify_headline
from sentiment import _extract_score
from visualize import plot_sentiment_vs_price


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Fresh in-memory-equivalent SQLite DB for each test."""
    path = str(tmp_path / "test.db")
    db.init_db(path)
    return path


def _seed_headlines(tmp_db, rows):
    """Insert headline dicts and return their IDs in insertion order."""
    db.insert_headlines(rows, db_path=tmp_db)
    unscored = db.get_unscored_headlines(db_path=tmp_db)
    return list(unscored["id"].astype(int))


# -----------------------------------------------------------------------------
# L1 - Scraper: date parsing
# -----------------------------------------------------------------------------

class TestDateParsing:
    def test_rss_gmt(self):
        assert _parse_date("Mon, 26 May 2026 10:00:00 GMT") == date(2026, 5, 26)

    def test_rss_offset(self):
        assert _parse_date("Mon, 26 May 2026 10:00:00 +0000") == date(2026, 5, 26)

    def test_iso_z(self):
        assert _parse_date("2026-05-26T10:00:00Z") == date(2026, 5, 26)

    def test_iso_offset(self):
        assert _parse_date("2026-05-26T13:00:00+03:00") == date(2026, 5, 26)

    def test_turkish_dotted(self):
        assert _parse_date("26.05.2026 10:00") == date(2026, 5, 26)

    def test_plain_date(self):
        assert _parse_date("2026-05-26") == date(2026, 5, 26)

    def test_empty_returns_none(self):
        assert _parse_date("") is None

    def test_garbage_returns_none(self):
        assert _parse_date("not a date at all") is None


# -----------------------------------------------------------------------------
# L1 - Scraper: relevance filter
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# L1 - Scraper: Turkish normalisation (_normalise)
# -----------------------------------------------------------------------------

class TestNormalise:
    """Each Turkish special character must fold to ASCII before lowercasing."""

    # Uppercase Turkish specials
    def test_capital_dotted_i(self):        # U+0130 İ -> i
        assert _normalise("İstanbul") == "istanbul"

    def test_capital_s_cedilla(self):       # U+015E S with cedilla -> s
        assert _normalise("Şehir") == "sehir"

    def test_capital_c_cedilla(self):       # U+00C7 C with cedilla -> c
        assert _normalise("Çek") == "cek"

    def test_capital_g_breve(self):         # U+011E G with breve -> g
        assert _normalise("Ğzel") == "gzel"

    def test_capital_o_diaeresis(self):     # U+00D6 O with diaeresis -> o
        assert _normalise("Özel") == "ozel"

    def test_capital_u_diaeresis(self):     # U+00DC U with diaeresis -> u
        assert _normalise("Üst") == "ust"

    # Lowercase Turkish specials
    def test_lower_s_cedilla(self):         # U+015F s with cedilla -> s
        assert _normalise("başkan") == "baskan"

    def test_lower_c_cedilla(self):         # U+00E7 c with cedilla -> c
        assert _normalise("gerçek") == "gercek"

    def test_lower_g_breve(self):           # U+011F g with breve -> g
        assert _normalise("büyüme") == "buyume"    # buyuMe -- note u-umlaut too

    def test_lower_o_diaeresis(self):       # U+00F6 o with diaeresis -> o
        assert _normalise("döviz") == "doviz"

    def test_lower_u_diaeresis(self):       # U+00FC u with diaeresis -> u
        assert _normalise("türk") == "turk"

    def test_dotless_i(self):               # U+0131 dotless lowercase i -> i
        assert _normalise("kır") == "kir"

    # Compound: full word roundtrips
    def test_turkiye_full(self):
        assert _normalise("Türkıye") == "turkiye"

    def test_doviz(self):
        assert _normalise("Döviz") == "doviz"

    def test_buyume(self):
        assert _normalise("Büyüme") == "buyume"

    def test_bankacilik(self):
        assert _normalise("bankacılık") == "bankacilik"


class TestRelevanceFilter:
    # --- should PASS (keep) ---
    def test_borsa_istanbul(self):
        assert _is_relevant("Borsa Istanbul gun sonu verileri") is True

    def test_bist_direct(self):
        assert _is_relevant("BIST 100 endeksi 13.500 puanda kapandi") is True

    def test_tcmb_decision(self):
        assert _is_relevant("TCMB faiz kararini acikladi") is True

    def test_turkiye_enflasyon(self):
        assert _is_relevant("Turkiye enflasyon verileri beklentinin uzerinde geldi") is True

    def test_dolar_kuru(self):
        assert _is_relevant("Dolar kuru bugun ne kadar?") is True

    def test_petrol_fiyatlari(self):
        assert _is_relevant("Petrol fiyatlari dustu, Turkiye ithalati etkileniyor") is True

    # --- Turkish I fix: dotted capital i ---
    def test_istanbul_capital_i(self):
        # "Istanbul" has ASCII I, should match "istanbul"
        assert _is_relevant("Istanbul Borsasi yukseldi") is True

    def test_dotted_capital_i(self):
        # Dotted capital İ normalised to i before matching
        assert _normalise("İstanbul") == "istanbul"

    # --- should DROP ---
    def test_bitcoin_dropped(self):
        assert _is_relevant("Bitcoin 74.500 dolara geriledi, jeopolitik...") is False

    def test_ethereum_dropped(self):
        assert _is_relevant("Ethereum ETF cikislarinda rekor kirdi") is False

    def test_kripto_dropped(self):
        assert _is_relevant("Kripto piyasalari karisik seyretti") is False

    def test_sterling_dropped(self):
        assert _is_relevant("Sterlin sakin, Ingiltere perakende satislari dustu") is False

    def test_nikkei_dropped(self):
        assert _is_relevant("Nikkei 225 rekor seviyeye yukseldi") is False

    def test_indian_rupee_dropped(self):
        assert _is_relevant("Hindistan rupisi RBI mudahalesiyle guclendi") is False

    def test_nyse_dropped(self):
        assert _is_relevant("New York borsasi yukselisle acildi") is False

    def test_nasdaq_dropped(self):
        assert _is_relevant("Nasdaq teknoloji hisselerindeki dususle kapandi") is False

    # --- blocklist overridden by strong Turkey marker ---
    def test_bitcoin_with_turkiye_kept(self):
        # Crypto + Turkey context = relevant (Turkey's crypto stance)
        assert _is_relevant("Turkiye'de bitcoin islem hacmi rekor kirdi") is True

    def test_nasdaq_with_bist_kept(self):
        assert _is_relevant("BIST 100 Nasdaq'taki yukselisi takip etti") is True

    # --- real Turkish diacritics (exercises _normalise end-to-end) ---
    def test_turkiye_with_diacritics_kept(self):
        # "Türkiye" -> "turkiye" via fold; "enflasyon" keyword -> keep
        assert _is_relevant("Türkiye enflasyon verileri beklentinin üzerinde geldi") is True

    def test_turk_lirasi_with_diacritics_kept(self):
        # "Türk" -> "turk"; "lira" is in keywords via "turk lirasi"
        assert _is_relevant("Türk lirası dolara karşı değer kaybetti") is True

    def test_doviz_with_diacritics_kept(self):
        # "Döviz" -> "doviz" -> keyword match
        assert _is_relevant("Döviz kurları hareketli seyrediyor") is True

    def test_buyume_with_diacritics_kept(self):
        # "Büyüme" -> "buyume" -> keyword match
        assert _is_relevant("Türkiye büyüme rakamları açıklandı") is True

    def test_bankacilik_with_diacritics_kept(self):
        # "bankacılık" -> "bankacilik" -> keyword match
        assert _is_relevant("Türk bankacılık sektörü güçlü kaldı") is True

    def test_bitcoin_with_diacritic_turkiye_kept(self):
        # Critical: "Türkiye" must normalise to "turkiye" -> strong marker
        # overrides bitcoin blocklist
        assert _is_relevant("Türkiye'de bitcoin işlem hacmi rekor kırdı") is True

    def test_bitcoin_no_turkey_diacritics_dropped(self):
        # Bitcoin with no Turkey context -> dropped even with diacritics elsewhere
        assert _is_relevant("Kripto dünyasında bitcoin rekor kırdı") is False


# -----------------------------------------------------------------------------
# L2 - Sentiment: score extraction (no torch loaded)
# -----------------------------------------------------------------------------

class TestScoreExtraction:
    def test_positive(self):
        result = [
            {"label": "positive", "score": 0.80},
            {"label": "neutral",  "score": 0.15},
            {"label": "negative", "score": 0.05},
        ]
        score, label, p_pos, p_neu, p_neg = _extract_score(result)
        assert abs(score - 0.75) < 1e-6   # 0.80 - 0.05
        assert label == "positive"
        assert abs(p_pos - 0.80) < 1e-6
        assert abs(p_neu - 0.15) < 1e-6
        assert abs(p_neg - 0.05) < 1e-6

    def test_negative(self):
        result = [
            {"label": "negative", "score": 0.70},
            {"label": "neutral",  "score": 0.20},
            {"label": "positive", "score": 0.10},
        ]
        score, label, p_pos, p_neu, p_neg = _extract_score(result)
        assert abs(score - (-0.60)) < 1e-6  # 0.10 - 0.70
        assert label == "negative"
        assert abs(p_pos - 0.10) < 1e-6
        assert abs(p_neg - 0.70) < 1e-6

    def test_neutral(self):
        result = [
            {"label": "neutral",  "score": 0.60},
            {"label": "positive", "score": 0.20},
            {"label": "negative", "score": 0.20},
        ]
        score, label, p_pos, p_neu, p_neg = _extract_score(result)
        assert abs(score - 0.0) < 1e-6   # 0.20 - 0.20
        assert label == "neutral"
        assert abs(p_neu - 0.60) < 1e-6

    def test_uppercase_labels(self):
        """Model may return uppercase label names."""
        result = [
            {"label": "POSITIVE", "score": 0.90},
            {"label": "NEUTRAL",  "score": 0.05},
            {"label": "NEGATIVE", "score": 0.05},
        ]
        score, label, p_pos, p_neu, p_neg = _extract_score(result)
        assert score > 0
        assert label == "positive"
        assert abs(p_pos - 0.90) < 1e-6

    def test_score_range(self):
        """Continuous score must stay in [-1, 1]."""
        result = [
            {"label": "positive", "score": 1.0},
            {"label": "neutral",  "score": 0.0},
            {"label": "negative", "score": 0.0},
        ]
        score, _, p_pos, p_neu, p_neg = _extract_score(result)
        assert -1.0 <= score <= 1.0
        assert abs(p_pos - 1.0) < 1e-6
        assert abs(p_neg - 0.0) < 1e-6

    def test_unrecognised_labels_raise(self):
        """Unknown/empty label output must fail loudly, not silently score 0.0.

        A model swap or transformers update that changes label names would
        otherwise neutral-label every headline without any error.
        """
        import pytest
        with pytest.raises(ValueError):
            _extract_score([])
        with pytest.raises(ValueError):
            _extract_score([{"label": "bullish", "score": 0.9}])

    def test_probabilities_sum_to_one(self):
        """Softmax probabilities should sum to ~1."""
        result = [
            {"label": "positive", "score": 0.70},
            {"label": "neutral",  "score": 0.20},
            {"label": "negative", "score": 0.10},
        ]
        _, _, p_pos, p_neu, p_neg = _extract_score(result)
        assert abs((p_pos + p_neu + p_neg) - 1.0) < 1e-6


# -----------------------------------------------------------------------------
# L1/DB - Database: insert and deduplication
# -----------------------------------------------------------------------------

class TestDatabase:
    def test_insert_returns_count(self, tmp_db):
        rows = [
            {"source": "t", "title": "h1", "url": "http://x.com/1",
             "published_at": date(2026, 5, 26)},
        ]
        n = db.insert_headlines(rows, db_path=tmp_db)
        assert n == 1

    def test_url_deduplication(self, tmp_db):
        row = {"source": "t", "title": "h1", "url": "http://x.com/1",
               "published_at": date(2026, 5, 26)}
        db.insert_headlines([row], db_path=tmp_db)
        n = db.insert_headlines([row], db_path=tmp_db)
        assert n == 0   # duplicate skipped

    def test_null_url_allows_multiple(self, tmp_db):
        """Headlines without URLs use title-based dedup at scraper level;
        at DB level two NULL-url rows are both inserted (NULL != NULL in SQL)."""
        rows = [
            {"source": "t", "title": "h1", "url": None, "published_at": date(2026, 5, 26)},
            {"source": "t", "title": "h2", "url": None, "published_at": date(2026, 5, 26)},
        ]
        n = db.insert_headlines(rows, db_path=tmp_db)
        assert n == 2

    def test_get_unscored_returns_unscored_only(self, tmp_db):
        rows = [
            {"source": "t", "title": "h1", "url": "http://x.com/1",
             "published_at": date(2026, 5, 26)},
            {"source": "t", "title": "h2", "url": "http://x.com/2",
             "published_at": date(2026, 5, 26)},
        ]
        db.insert_headlines(rows, db_path=tmp_db)
        unscored = db.get_unscored_headlines(db_path=tmp_db)
        assert len(unscored) == 2

        # Score one (7-tuple: score, label, p_pos, p_neu, p_neg, model_name, id)
        hid = int(unscored.iloc[0]["id"])
        db.batch_update_sentiment(
            [(0.5, "positive", 0.80, 0.15, 0.05, "test-model", hid)],
            db_path=tmp_db,
        )

        unscored_after = db.get_unscored_headlines(db_path=tmp_db)
        assert len(unscored_after) == 1

    def test_stats_counts(self, tmp_db):
        rows = [
            {"source": "t", "title": f"h{i}", "url": f"http://x.com/{i}",
             "published_at": date(2026, 5, 26)}
            for i in range(5)
        ]
        db.insert_headlines(rows, db_path=tmp_db)
        stats = db.db_stats(db_path=tmp_db)
        assert stats["total_headlines"] == 5
        assert stats["scored_headlines"] == 0
        assert stats["unscored_headlines"] == 5


# -----------------------------------------------------------------------------
# L3 - Aggregation: math correctness
# -----------------------------------------------------------------------------

class TestAggregation:
    def _insert_scored(self, tmp_db, scored_rows):
        """Insert headlines already scored into DB via two-step (insert + update)."""
        raw = [
            {"source": "t", "title": r["title"], "url": r["url"],
             "published_at": r["date"]}
            for r in scored_rows
        ]
        db.insert_headlines(raw, db_path=tmp_db)
        unscored = db.get_unscored_headlines(db_path=tmp_db)
        updates = [
            # 7-tuple: score, label, p_pos, p_neu, p_neg, model_name, id
            (r["score"], r["label"],
             0.8 if r["label"] == "positive" else 0.1,   # p_pos
             0.1,                                          # p_neu
             0.8 if r["label"] == "negative" else 0.1,   # p_neg
             "test-model",
             int(unscored.iloc[i]["id"]))
            for i, r in enumerate(scored_rows)
        ]
        db.batch_update_sentiment(updates, db_path=tmp_db)

    def test_mean_is_correct(self, tmp_db):
        # avg_score is now confidence-weighted (weight = |score|) with time-of-day
        # weighting (no published_hour -> time_weight = 1.0 for all).
        # Expected: np.average([0.5, -0.3, 0.1], weights=[0.5, 0.3, 0.1])
        #         = (0.25 - 0.09 + 0.01) / 0.9 ≈ 0.1889
        scores = [0.5, -0.3, 0.1]
        weights = [abs(s) for s in scores]
        expected = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        self._insert_scored(tmp_db, [
            {"title": "h1", "url": "u1", "date": date(2026, 5, 26),
             "score":  0.5, "label": "positive"},
            {"title": "h2", "url": "u2", "date": date(2026, 5, 26),
             "score": -0.3, "label": "negative"},
            {"title": "h3", "url": "u3", "date": date(2026, 5, 26),
             "score":  0.1, "label": "neutral"},
        ])
        n = aggregate_step(db_path=tmp_db)
        assert n == 1

        sent = db.get_daily_sentiment(db_path=tmp_db)
        assert abs(sent.iloc[0]["avg_score"] - expected) < 1e-6

    def test_label_counts(self, tmp_db):
        self._insert_scored(tmp_db, [
            {"title": "h1", "url": "u1", "date": date(2026, 5, 26),
             "score":  0.5, "label": "positive"},
            {"title": "h2", "url": "u2", "date": date(2026, 5, 26),
             "score": -0.3, "label": "negative"},
            {"title": "h3", "url": "u3", "date": date(2026, 5, 26),
             "score":  0.1, "label": "neutral"},
        ])
        aggregate_step(db_path=tmp_db)
        sent = db.get_daily_sentiment(db_path=tmp_db)
        assert sent.iloc[0]["positive_count"] == 1
        assert sent.iloc[0]["negative_count"] == 1
        assert sent.iloc[0]["neutral_count"]  == 1
        assert sent.iloc[0]["headline_count"] == 3

    def test_bull_bear_ratio(self, tmp_db):
        self._insert_scored(tmp_db, [
            {"title": "h1", "url": "u1", "date": date(2026, 5, 26),
             "score":  0.6, "label": "positive"},
            {"title": "h2", "url": "u2", "date": date(2026, 5, 26),
             "score":  0.4, "label": "positive"},
            {"title": "h3", "url": "u3", "date": date(2026, 5, 26),
             "score": -0.5, "label": "negative"},
        ])
        aggregate_step(db_path=tmp_db)
        sent = db.get_daily_sentiment(db_path=tmp_db)
        # 2 positive / (2 positive + 1 negative) = 2/3
        assert abs(sent.iloc[0]["bull_bear_ratio"] - 2/3) < 1e-6

    def test_multi_day_grouping(self, tmp_db):
        self._insert_scored(tmp_db, [
            {"title": "h1", "url": "u1", "date": date(2026, 5, 25), "score":  0.3, "label": "positive"},
            {"title": "h2", "url": "u2", "date": date(2026, 5, 26), "score": -0.2, "label": "negative"},
        ])
        n = aggregate_step(db_path=tmp_db)
        assert n == 2   # two distinct days

    def test_aggregate_skips_unscored(self, tmp_db):
        rows = [{"source": "t", "title": "h1", "url": "u1",
                 "published_at": date(2026, 5, 26)}]
        db.insert_headlines(rows, db_path=tmp_db)
        n = aggregate_step(db_path=tmp_db)
        assert n == 0   # nothing scored yet

    def test_aggregate_clears_stale_rows(self, tmp_db):
        """
        After cleaning headlines, re-running aggregate must remove the
        now-orphaned daily_sentiment rows -- derived tables must never lie.
        """
        # Seed two days
        self._insert_scored(tmp_db, [
            {"title": "h1", "url": "u1", "date": date(2026, 5, 25),
             "score":  0.5, "label": "positive"},
            {"title": "h2", "url": "u2", "date": date(2026, 5, 26),
             "score": -0.3, "label": "negative"},
        ])
        n = aggregate_step(db_path=tmp_db)
        assert n == 2

        # Manually delete the headline for day 2026-05-25 from the DB
        with db._conn(db_path=tmp_db) as con:
            con.execute("DELETE FROM headlines WHERE published_at = '2026-05-25'")

        # Re-aggregate: stale row for 2026-05-25 must be gone
        n2 = aggregate_step(db_path=tmp_db)
        assert n2 == 1   # only one day remains

        sent = db.get_daily_sentiment(db_path=tmp_db)
        assert len(sent) == 1
        assert sent.iloc[0]["date"] == "2026-05-26"

    def test_aggregate_stores_category_sentiment(self, tmp_db):
        """Per-category aggregation rows are created alongside daily_sentiment."""
        self._insert_scored(tmp_db, [
            {"title": "BIST 100 hisse endeksi yukseldi", "url": "u1",
             "date": date(2026, 5, 26), "score": 0.6, "label": "positive"},
            {"title": "Dolar kuru guclendi", "url": "u2",
             "date": date(2026, 5, 26), "score": -0.2, "label": "negative"},
        ])
        aggregate_step(db_path=tmp_db)
        cat_df = db.get_category_daily_sentiment(db_path=tmp_db)
        assert len(cat_df) >= 1
        assert "category" in cat_df.columns
        assert "avg_score" in cat_df.columns

    def test_probabilities_stored_in_db(self, tmp_db):
        """batch_update_sentiment must persist p_positive, p_neutral, p_negative."""
        raw = [{"source": "t", "title": "h1", "url": "http://x.com/1",
                "published_at": date(2026, 5, 26)}]
        db.insert_headlines(raw, db_path=tmp_db)
        unscored = db.get_unscored_headlines(db_path=tmp_db)
        hid = int(unscored.iloc[0]["id"])

        db.batch_update_sentiment(
            [(0.65, "positive", 0.80, 0.15, 0.05, "test-model", hid)],
            db_path=tmp_db,
        )

        with db._conn(db_path=tmp_db) as con:
            row = con.execute(
                "SELECT p_positive, p_neutral, p_negative, model_name FROM headlines WHERE id=?",
                (hid,),
            ).fetchone()

        assert abs(row["p_positive"] - 0.80) < 1e-6
        assert abs(row["p_neutral"]  - 0.15) < 1e-6
        assert abs(row["p_negative"] - 0.05) < 1e-6
        assert row["model_name"] == "test-model"

    def test_run_logged(self, tmp_db):
        """log_run_start / log_run_end round-trip."""
        run_id = db.log_run_start(model_name="m", db_path=tmp_db)
        assert isinstance(run_id, int) and run_id > 0

        db.log_run_end(run_id, status="ok",
                       headlines_scraped=5, headlines_scored=3,
                       prices_added=60, sentiment_days=3,
                       db_path=tmp_db)

        with db._conn(db_path=tmp_db) as con:
            row = con.execute(
                "SELECT status, headlines_scraped FROM pipeline_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()

        assert row["status"] == "ok"
        assert row["headlines_scraped"] == 5


# -----------------------------------------------------------------------------
# L1 - Scraper: headline category classifier
# -----------------------------------------------------------------------------

class TestCategoryClassifier:
    """classify_headline() must assign the correct bucket (first-match priority)."""

    def test_bist_company(self):
        assert classify_headline("BIST 100 endeksi rekor kırdı") == "bist_company"

    def test_bist_company_hisse(self):
        assert classify_headline("Hisse senetleri yükseliş kaydetti") == "bist_company"

    def test_rates_tcmb(self):
        assert classify_headline("TCMB faiz kararını açıkladı") == "rates_tcmb"

    def test_rates_faiz(self):
        assert classify_headline("Faiz oranları değişmedi") == "rates_tcmb"

    def test_turkey_macro(self):
        assert classify_headline("Türkiye enflasyon verileri açıklandı") == "turkey_macro"

    def test_turkey_macro_buyume(self):
        assert classify_headline("Büyüme rakamları beklentinin altında") == "turkey_macro"

    def test_fx_lira(self):
        assert classify_headline("Dolar kuru 32 liraya geriledi") == "fx_lira"

    def test_fx_doviz(self):
        assert classify_headline("Döviz piyasaları hareketli") == "fx_lira"

    def test_banks(self):
        assert classify_headline("Bankacılık sektörü güçlü kaldı") == "banks"

    def test_energy_commodities(self):
        assert classify_headline("Petrol fiyatları düştü") == "energy_commodities"

    def test_energy_altin(self):
        assert classify_headline("Altın rekor fiyata ulaştı") == "energy_commodities"

    def test_global_risk(self):
        assert classify_headline("Jeopolitik riskler piyasaları etkiledi") == "global_risk"

    def test_crypto(self):
        # No FX / BIST / macro keyword -> crypto wins
        assert classify_headline("Bitcoin kripto piyasasinda yeni rekor kirdi") == "crypto"

    def test_crypto_wins_over_fx(self):
        # crypto now has higher priority than fx_lira; proper nouns beat generic "dolar"
        assert classify_headline("Bitcoin 70.000 dolara geriledi") == "crypto"

    def test_pure_fx_no_crypto(self):
        # headline with only FX keywords and no crypto -> fx_lira
        assert classify_headline("Dolar kuru 32 liraya geriledi") == "fx_lira"

    def test_other_catch_all(self):
        # A headline with no matching keyword -> "other"
        assert classify_headline("Hava durumu sıcaklıkları normalin üzerinde") == "other"

    def test_priority_bist_beats_rates(self):
        # "bist" appears -> bist_company wins even though "faiz" also present
        assert classify_headline("BIST faiz baskısına rağmen yükseldi") == "bist_company"

    def test_diacritic_normalisation_in_classifier(self):
        # Turkish diacritics must be folded before matching keywords
        assert classify_headline("Türkiye büyüme rakamları") == "turkey_macro"

    # -- Expanded taxonomy tests (new keywords) --------------------------------
    def test_hazine_goes_to_rates(self):
        assert classify_headline("Hazine'den yeni borçlanma hamlesi") == "rates_tcmb"

    def test_tahvil_goes_to_rates(self):
        assert classify_headline("Tahvil faizleri geriledi") == "rates_tcmb"

    def test_bakir_goes_to_energy(self):
        assert classify_headline("Bakır fiyatları rekor seviyelere ulaştı") == "energy_commodities"

    def test_bugday_goes_to_energy(self):
        assert classify_headline("Buğday fiyatları CBOT'ta düşüşte") == "energy_commodities"

    def test_ekonomi_goes_to_macro(self):
        assert classify_headline("Türkiye ekonomisi güçlü büyüme kaydetti") == "turkey_macro"

    def test_askeri_goes_to_global_risk(self):
        assert classify_headline("ABD askeri gemileri Hürmüz'de konuşlandı") == "global_risk"

    def test_category_stored_on_insert(self, tmp_db):
        """Headlines inserted with a category field persist it to the DB."""
        rows = [
            {"source": "t", "title": "BIST 100 yukseldi", "url": "http://x.com/1",
             "published_at": date(2026, 5, 26), "category": "bist_company"},
        ]
        db.insert_headlines(rows, db_path=tmp_db)
        with db._conn(db_path=tmp_db) as con:
            row = con.execute("SELECT category FROM headlines WHERE url='http://x.com/1'").fetchone()
        assert row["category"] == "bist_company"


# -----------------------------------------------------------------------------
# L5 - Visualisation: graceful handling of thin / empty data
# -----------------------------------------------------------------------------

class TestVisualize:
    def test_returns_none_when_no_data(self, tmp_db, tmp_path):
        out = str(tmp_path / "plot.png")
        result = plot_sentiment_vs_price(db_path=tmp_db, days=30,
                                         output_path=out, show=False)
        assert result is None
        assert not os.path.exists(out)

    def test_returns_none_when_overlap_too_small(self, tmp_db, tmp_path):
        """Sentiment exists but no price data -> no overlap -> None."""
        db.upsert_daily_sentiment([{
            "date": "2026-05-26", "avg_score": 0.2, "std_score": 0.1,
            "headline_count": 5, "positive_count": 3,
            "negative_count": 1, "neutral_count": 1, "bull_bear_ratio": 0.75,
        }], db_path=tmp_db)
        out = str(tmp_path / "plot.png")
        result = plot_sentiment_vs_price(db_path=tmp_db, days=30,
                                         output_path=out, show=False)
        assert result is None

    def test_saves_png_with_sufficient_data(self, tmp_db, tmp_path):
        """With 5+ overlapping days the plot should render and save."""
        days = [
            ("2026-05-20", 14000.0, 0.50),
            ("2026-05-21", 13900.0, -0.71),
            ("2026-05-22", 14100.0, 0.72),
            ("2026-05-23", 14050.0, -0.35),
            ("2026-05-24", 14200.0, 0.71),
        ]
        prices = pd.DataFrame({
            "date":         [d[0] for d in days],
            "open":         [d[1] * 0.99 for d in days],
            "high":         [d[1] * 1.01 for d in days],
            "low":          [d[1] * 0.98 for d in days],
            "close":        [d[1] for d in days],
            "volume":       [1e9] * 5,
            "daily_return": [None, -0.71, 1.44, -0.35, 1.07],
        })
        db.upsert_prices(prices, db_path=tmp_db)

        sent_rows = [
            {"date": d[0], "avg_score": d[2], "std_score": 0.1,
             "headline_count": 5, "positive_count": 3 if d[2] > 0 else 1,
             "negative_count": 1 if d[2] > 0 else 3, "neutral_count": 1,
             "bull_bear_ratio": 0.75 if d[2] > 0 else 0.25}
            for d in days
        ]
        db.upsert_daily_sentiment(sent_rows, db_path=tmp_db)

        out = str(tmp_path / "plot.png")
        result = plot_sentiment_vs_price(db_path=tmp_db, days=30,
                                         output_path=out, show=False)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 10_000   # a real PNG, not an empty file

    def test_renders_with_preliminary_watermark(self, tmp_db, tmp_path):
        """
        5–29 overlapping days: plot renders (not None) but is in PRELIMINARY mode.
        The function should still save the PNG -- callers inspect the log for warnings.
        """
        # 5 days: above the hard floor (5) but below MINIMUM_OVERLAP_DAYS (30)
        days = [
            ("2026-05-20", 14000.0, 0.50),
            ("2026-05-21", 13900.0, -0.71),
            ("2026-05-22", 14100.0, 0.72),
            ("2026-05-23", 14050.0, -0.35),
            ("2026-05-24", 14200.0, 0.71),
        ]
        prices = pd.DataFrame({
            "date":         [d[0] for d in days],
            "open":         [d[1] * 0.99 for d in days],
            "high":         [d[1] * 1.01 for d in days],
            "low":          [d[1] * 0.98 for d in days],
            "close":        [d[1] for d in days],
            "volume":       [1e9] * 5,
            "daily_return": [None, -0.71, 1.44, -0.35, 1.07],
        })
        db.upsert_prices(prices, db_path=tmp_db)
        sent_rows = [
            {"date": d[0], "avg_score": d[2], "std_score": 0.1,
             "headline_count": 2,   # intentionally < MINIMUM_HEADLINES_PER_DAY (3)
             "positive_count": 1 if d[2] > 0 else 0,
             "negative_count": 0 if d[2] > 0 else 1, "neutral_count": 1,
             "bull_bear_ratio": None}
            for d in days
        ]
        db.upsert_daily_sentiment(sent_rows, db_path=tmp_db)

        out = str(tmp_path / "plot_prelim.png")
        result = plot_sentiment_vs_price(db_path=tmp_db, days=30,
                                         output_path=out, show=False)
        # Must still render (not None) even in preliminary mode
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 5_000


# ─────────────────────────────────────────────────────────────────────────────
# LLM sentiment scorer (sentiment_llm.py) — pure-function contract tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMScorerContract:
    """_to_tuple must keep label, score, and pseudo-probs mutually consistent."""

    def test_positive_label_clears_threshold(self):
        from sentiment_llm import _to_tuple
        from config import SENTIMENT_POSITIVE_THRESHOLD
        score, label, p_pos, p_neu, p_neg = _to_tuple("positive", 0.8)
        assert label == "positive"
        assert score > SENTIMENT_POSITIVE_THRESHOLD
        assert abs((p_pos - p_neg) - score) < 1e-9

    def test_weak_positive_still_clears_threshold(self):
        """A 0.0-strength positive must not fall into the neutral band."""
        from sentiment_llm import _to_tuple
        from config import SENTIMENT_POSITIVE_THRESHOLD
        score, label, *_ = _to_tuple("positive", 0.0)
        assert label == "positive"
        assert score > SENTIMENT_POSITIVE_THRESHOLD

    def test_negative_mirrors_positive(self):
        from sentiment_llm import _to_tuple
        from config import SENTIMENT_NEGATIVE_THRESHOLD
        score, label, p_pos, p_neu, p_neg = _to_tuple("negative", 0.6)
        assert label == "negative"
        assert score < SENTIMENT_NEGATIVE_THRESHOLD
        assert abs((p_pos - p_neg) - score) < 1e-9

    def test_neutral_is_zero(self):
        from sentiment_llm import _to_tuple
        score, label, p_pos, p_neu, p_neg = _to_tuple("neutral", 0.9)
        assert score == 0.0 and label == "neutral"
        assert p_neu == 1.0

    def test_strength_clamped(self):
        from sentiment_llm import _to_tuple
        score, *_ = _to_tuple("positive", 5.0)   # out-of-range strength
        assert -1.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Trading calendar / signal_date alignment
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalDate:
    """signal_date = first trading session that can react to the headline."""

    def test_premarket_weekday_stays_same_day(self):
        from trading_calendar import signal_date
        # 2026-06-10 is a Wednesday
        assert signal_date("2026-06-10", 8) == "2026-06-10"

    def test_intraday_stays_same_day(self):
        from trading_calendar import signal_date
        assert signal_date("2026-06-10", 14) == "2026-06-10"

    def test_postclose_rolls_to_next_day(self):
        from trading_calendar import signal_date
        assert signal_date("2026-06-10", 22) == "2026-06-11"

    def test_friday_postclose_rolls_to_monday(self):
        from trading_calendar import signal_date
        # 2026-06-12 is a Friday -> next session Monday 2026-06-15
        assert signal_date("2026-06-12", 21) == "2026-06-15"

    def test_weekend_rolls_to_monday(self):
        from trading_calendar import signal_date
        # Saturday, any hour
        assert signal_date("2026-06-13", 11) == "2026-06-15"

    def test_null_hour_is_conservative_next_day(self):
        from trading_calendar import signal_date
        # Unknown publish time could be post-close: avoid lookahead
        assert signal_date("2026-06-10", None) == "2026-06-11"

    def test_holiday_skipped(self):
        from trading_calendar import signal_date
        # 2026-07-14 (Tue) post-close; 2026-07-15 is a BIST holiday -> Jul 16
        assert signal_date("2026-07-14", 20) == "2026-07-16"
