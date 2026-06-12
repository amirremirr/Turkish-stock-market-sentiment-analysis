"""
Central configuration for the BIST100 sentiment pipeline.
Edit these values to customise behaviour without touching pipeline code.
"""

import os

# Load .env if python-dotenv is installed (optional convenience for local dev).
# Install: pip install python-dotenv
# Create:  copy .env.example .env  and fill in your key.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -- Storage ------------------------------------------------------------------
DB_PATH = "finance_sentiment.db"

# -- Market data --------------------------------------------------------------
BIST100_TICKER = "XU100.IS"          # Yahoo Finance ticker for BIST 100 index

# -- Alpha Vantage (second data source) ---------------------------------------
# Free tier: 25 requests/day, 5 requests/minute.
# Used for: USD/TRY daily FX rates (BIST 100 index is not available on AV).
# Get a free key: https://www.alphavantage.co/support/#api-key
# Set via environment variable or .env file (see .env.example).
# The pipeline works without this key; the FX rates step is simply skipped.
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")

# -- Scraping ------------------------------------------------------------------
DEFAULT_LOOKBACK_DAYS = 90           # How far back to seed the DB on first run

# RSS feeds - add or remove sources here
# Evaluated with discover_sources.py on 2026-05-27.
# Metrics shown: (items / relevance% / Turkish-chars%)
#
# REMOVED: investing_tr_stocks (news_301.rss) — during holidays/weekends it
#   fills with Bitcoin/crypto content (0% relevance observed). The relevance
#   filter catches most of it, but it adds noise and wastes scoring budget.
#
# To test a new feed before adding it:
#   python discover_sources.py --url <RSS_URL>
RSS_FEEDS = {
    # Core Turkish financial press
    "investing_tr_economy": "https://tr.investing.com/rss/news_1.rss",     # 10 items / 50% / 100%
    "bloomberght":          "https://www.bloomberght.com/rss",              # 20 items / 60% / 95%
    "dunya":                "https://www.dunya.com/rss",                    # 25 items / 44% / 100%
    "sabah_ekonomi":        "https://www.sabah.com.tr/rss/ekonomi.xml",    # 10 items / 60% / 90%
    # New additions (2026-05-27)
    "ntv_ekonomi":          "https://www.ntv.com.tr/ekonomi.rss",          # 20 items / 90% / 100% ← best ratio
    "aa_ekonomi":           "https://www.aa.com.tr/tr/rss/default?cat=ekonomi",  # 30 items / 57% / 100%
    "hurriyet_ekonomi":     "https://www.hurriyet.com.tr/rss/ekonomi",     # 100 items / 47% / 100%
    "haberturk_ekonomi":    "https://www.haberturk.com/rss/ekonomi.xml",   # 30 items / 37% / 97%
    # Political risk sources (added 2026-06-08)
    # These are general/political feeds filtered by the political_risk relevance
    # keywords below. Only high-signal political events (arrests, resignations,
    # elections, crises) will pass the relevance gate.
    "aa_politika":          "https://www.aa.com.tr/tr/rss/default?cat=politika",
    "sozcu_gundem":         "https://www.sozcu.com.tr/rss/gundem.xml",
    # Note: sozcu_siyaset and sozcu_ekonomi return identical content to gundem.
    #
    # REMOVED (2026-06-11): kap_bildirimler — the URL 404s; KAP's redesigned
    # site (Next.js) no longer serves RSS at any discoverable path. The feed
    # contributed 0 headlines since being added on 2026-06-08. The "ozel durum"
    # / "finansal sonuc" keywords stay in the relevance filter so KAP-style
    # disclosure stories carried by the news feeds above are still captured.
    # Company-level KAP data would need their REST API (see ROADMAP).
}

# HTML fallback when RSS is blocked
HTML_SOURCES = {
    "investing_tr": "https://tr.investing.com/news/stock-market-news",
}

REQUEST_TIMEOUT = 15        # seconds
CRAWL_DELAY    = 1.5        # seconds between requests (be polite)

# -- Relevance filtering -------------------------------------------------------
# Two-tier filter applied to every scraped headline:
#
#   Tier 1 - BLOCKLIST: if ANY blocklist term is present AND NO strong
#             Turkey marker is present, the headline is dropped.
#             Catches crypto, foreign equity markets, non-Turkish FX.
#
#   Tier 2 - KEYWORDS: headline must contain at least one keyword.
#             Guards against generic news that snuck through tier 1.
#
# Set RELEVANCE_FILTER_ENABLED = False to disable both tiers.
RELEVANCE_FILTER_ENABLED = True

# Strong Turkey-market anchors - presence of any one overrides the blocklist.
# Political figures are included: their arrest/resignation/statement is always
# BIST-relevant regardless of any incidental blocklist hit in the headline.
RELEVANCE_STRONG = [
    "turkiye", "turk",          # normalised (i -> i) forms used at runtime
    "bist", "borsa istanbul",
    "tcmb",
    "turk lirasi", "tl kuru",
    # Key political figures whose headlines are always market-relevant
    "imamoglu",                 # Istanbul mayor — arrest in 2021 caused ~10% BIST drop
    "kilicdaroglu",             # Former CHP leader, still influential
]

# Topics so far off-topic that they should be dropped even when "Türkiye" appears.
# These bypass the strong-marker override — they are ALWAYS irrelevant for
# financial sentiment regardless of whether the headline mentions Turkey.
# Keep this list short and low-risk: only add terms with near-zero false-positive risk.
RELEVANCE_HARD_BLOCKLIST = [
    "piyango",          # lottery — "milyar TL kazandı" headlines are not market news
    "baraj doluluk",    # water reservoir fill levels — utility operations, not financial
    "dunya kupas",      # World Cup / FIFA tournament news
    "futbol transfer",  # football player transfers (not financial transfers)
    "konser",           # music concerts and entertainment
]

# Topics that are almost never BIST-relevant on their own.
RELEVANCE_BLOCKLIST = [
    "bitcoin", "ethereum", "kripto", "btc", " nft",
    "new york borsasi",         # NYSE mentions
    "nasdaq",
    "wall street",
    "sterling", "pound",
    "nikkei", "dax ", "ftse", "hang seng",
    "hindistan ", " cin",   # leading-space guard: "cin " (China) would match "için" (for/because)
    "japonya",              # other EM country-specific news
]

# General financial keywords - at least one must be present after blocklist check.
RELEVANCE_KEYWORDS = [
    # Turkey / Turkish identifiers (normalised)
    "turkiye", "turk", "ankara", "istanbul",
    # Stock market
    "borsa", "bist", "hisse", "endeks",
    # Monetary policy
    "faiz", "tcmb", "merkez bankasi",
    # Macroeconomic indicators
    "enflasyon", "buyume", "butce", "cari", "gsyh", "ihracat", "ithalat",
    # Currency
    "dolar", "euro", "doviz", "kur", "lira",
    # Commodities
    "petrol", "dogalgaz", "altin", "emtia",
    # Fixed income
    "tahvil", "hazine", "bono",
    # Sectors
    "banka", "bankacilik", "enerji", "sanayi", "holding",
    # Market structure
    "piyasa", "yatirim", " fon",  # " fon" guards against "telefon" substring match
    # Political risk — market-moving political events
    # These terms alone are enough to make a headline BIST-relevant.
    # Routine politics (party meetings, municipal news) won't contain these.
    "imamoglu", "kilicdaroglu",  # opposition leaders — any headline is market-relevant
    "gozalti", "tutuklama",      # arrest / detention of a public figure
    "istifa",                    # resignation (minister, CB governor, CEO)
    "erken secim",               # early election announcement
    "siyasi kriz",               # explicit political crisis framing
    "grev",                      # strike / labor action affecting economy
    # Credit rating actions — sovereign/bank rating changes move BIST instantly
    "kredi notu",                # credit rating (generic)
    "not artirimi",              # rating upgrade
    "not indirimi",              # rating downgrade
    "moodys",                    # Moody's rating action
    "fitch",                     # Fitch rating action
    "s&p",                       # S&P Global rating action
    # KAP / company disclosures — material events from Borsa Istanbul listed firms
    "ozel durum",                # material event disclosure (KAP format)
    "finansal sonuc",            # financial results announcement
]

# -- Event-pipeline migration (see MIGRATION plan in ROADMAP.md) -----------------
# Feature flags: legacy path stays primary until the new path beats baselines
# out-of-sample. Never break `run.bat run`.
USE_EVENT_PIPELINE = False      # flip only after Phase 8 gate passes
EVENTS_DUAL_WRITE  = True       # mirror scored headlines into the events table
EXPERIMENT_ID      = "v1-legacy"  # stamped on every pipeline run for provenance

# Source tiers (Phase 3 will add Tier A ingestion — KAP, TCMB, TUIK).
# A = structured/auditable primary sources, B = wires/official statements,
# C = general press RSS (sentiment-heavy, noisy).
SOURCE_TIERS = {
    "aa_ekonomi":           "B",
    "aa_politika":          "B",
    "bloomberght":          "B",
    "dunya":                "C",
    "haberturk_ekonomi":    "C",
    "hurriyet_ekonomi":     "C",
    "investing_tr_economy": "C",
    "ntv_ekonomi":          "C",
    "sabah_ekonomi":        "C",
    "sozcu_gundem":         "C",
}
DEFAULT_SOURCE_TIER = "C"

# -- KAP Tier-A ingestion (migration Phase 3) ------------------------------------
# MKK API Portal credentials in .env: MKK_API_KEY / MKK_API_SECRET (HTTP Basic).
# IMPORTANT: the dev gateway serves a HISTORICAL SAMPLE dataset (late 2023) —
# leave KAP_ENABLED=False until production access is granted, or sample-era
# events would pollute the research store. Validate with:
#   python main.py kap-ingest --dry-run
KAP_ENABLED               = False
KAP_BASE_URL              = "https://apigwdev.mkk.com.tr/api/vyk"
KAP_DISCLOSURE_TYPES      = ["ODA", "FR"]   # material events + financial reports
KAP_MAX_DETAILS_PER_RUN   = 30              # detail calls per run (throttle: 6/min)
KAP_THROTTLE_SECONDS      = 11

# -- Sentiment scoring backend --------------------------------------------------
# "llm"  : OpenAI gpt-5-mini via API (sentiment_llm.py). Benchmarked 2026-06-12:
#          84.5% accuracy on held-out human labels vs 76.8% for tuned XLM-R.
#          Needs OPENAI_API_KEY in .env. Cost: ~half a cent per daily run.
# "xlmr" : local XLM-RoBERTa (sentiment.py). Free, offline, no API dependency —
#          kept as the fallback backend.
# IMPORTANT: don't flip back and forth casually — mixing backends across the
# history corrupts the signal analysis. If you switch, re-score everything:
#   UPDATE headlines SET sentiment_score=NULL; then python main.py score
SENTIMENT_BACKEND = "llm"

LLM_SENTIMENT_MODEL      = "gpt-5-mini"
LLM_SENTIMENT_BATCH_SIZE = 50     # headlines per API call

# -- Sentiment model (XLM-R fallback backend) ------------------------------------
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# The model labels - listed in the order the logits are emitted.
# Verified against the model card; change only if you swap the model.
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_BATCH_SIZE = 1    # 1 headline at a time = minimal memory, avoids OOM crashes on wake
SENTIMENT_MAX_LENGTH = 128   # tokens; headlines are short, 128 is plenty
# Tuned thresholds from 198-headline human validation (2026-06-08).
# Model argmax was 69.2% accurate; these thresholds give 76.8%.
# Rule: score > HIGH -> positive | score < LOW -> negative | else -> neutral
# The wide neutral band fixes the model's tendency to over-call negative on
# routine financial language ("reserves fell", "forecast lowered").
SENTIMENT_POSITIVE_THRESHOLD =  0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05
# Floor for the confidence weight |score| used in daily aggregation.
# Without a floor, a perfectly neutral headline (score ~ 0) has ~zero weight,
# so a day with ten neutrals and one +0.6 headline would aggregate to ~+0.6.
# A neutral headline IS information ("nothing happened") and should pull the
# daily average toward 0. Floor 0.10 = a neutral counts 1/10 of a max-conviction
# headline.
SENTIMENT_CONFIDENCE_FLOOR = 0.10
# LLM relevance grade (0.0-1.0, stored in headlines.relevance) multiplies the
# aggregation weight, so barely-relevant headlines barely move the daily mood.
# Rows BELOW this threshold are excluded from daily aggregates entirely (but
# never deleted — the grade is auditable and the threshold is tunable).
# NULL relevance (ungraded rows) is treated as 1.0.
# PROVISIONAL: the relevance grade and this cutoff are NOT yet validated
# against human judgment (sentiment is; relevance isn't). The label-export CSV
# now carries a human_relevant column — once ~300 labels exist, measure
# agreement before trusting this boundary. See METHODOLOGY §13.
RELEVANCE_MIN_FOR_AGGREGATION = 0.25

# -- Visualisation -------------------------------------------------------------
PLOT_DAYS    = 90            # default window for the chart
PLOT_DPI     = 150
PLOT_OUTPUT  = "sentiment_vs_bist100.png"

# -- BIST official holiday calendar -------------------------------------------
# Days when Borsa Istanbul is closed (no price data expected).
# Used by evaluate.py L4 to distinguish "known holiday gap" from "unexpected gap".
# Sources: KAP (Kamuyu Aydınlatma Platformu) annual market calendar.
#
# Format: "YYYY-MM-DD"  — add new years at the start of each year.
# Recurring fixed-date holidays (approximate — exact dates vary for lunar ones):
#   Jan 1   Yılbaşı (New Year)
#   Apr 23  Ulusal Egemenlik ve Çocuk Bayramı
#   May 1   İşçi Bayramı
#   May 19  Atatürk'ü Anma, Gençlik ve Spor Bayramı
#   Jul 15  Demokrasi ve Milli Birlik Günü
#   Aug 30  Zafer Bayramı
#   Oct 28  (afternoon) Cumhuriyet Bayramı eve  ← half-day, often omitted
#   Oct 29  Cumhuriyet Bayramı
# Lunar holidays (dates shift ~11 days earlier each year):
#   Ramazan Bayramı (Eid al-Fitr): 3.5 days (eve + 3 days)
#   Kurban Bayramı (Eid al-Adha): 4.5 days (eve + 4 days)
BIST_HOLIDAYS = [
    # 2026 -----------------------------------------------------------------------
    "2026-01-01",   # Yılbaşı
    "2026-03-19",   # Ramazan Bayramı arife (half-day / full closure)
    "2026-03-20",   # Ramazan Bayramı 1. günü
    "2026-03-21",   # Ramazan Bayramı 2. günü  (weekend in 2026 — not a gap)
    "2026-03-22",   # Ramazan Bayramı 3. günü  (weekend in 2026 — not a gap)
    "2026-04-23",   # Ulusal Egemenlik ve Çocuk Bayramı
    "2026-05-01",   # İşçi Bayramı
    "2026-05-19",   # Atatürk'ü Anma, Gençlik ve Spor Bayramı
    "2026-05-26",   # Kurban Bayramı arife
    "2026-05-27",   # Kurban Bayramı 1. günü
    "2026-05-28",   # Kurban Bayramı 2. günü
    "2026-05-29",   # Kurban Bayramı 3. günü
    "2026-06-01",   # Kurban Bayramı 4. günü
    "2026-07-15",   # Demokrasi ve Milli Birlik Günü
    "2026-08-30",   # Zafer Bayramı  (Sunday — may be observed Mon Aug 31)
    "2026-10-28",   # Cumhuriyet Bayramı arife (half-day)
    "2026-10-29",   # Cumhuriyet Bayramı
    # 2025 -----------------------------------------------------------------------
    "2025-01-01",   # Yılbaşı
    "2025-03-30",   # Ramazan Bayramı arife
    "2025-03-31",   # Ramazan Bayramı 1. günü
    "2025-04-01",   # Ramazan Bayramı 2. günü
    "2025-04-02",   # Ramazan Bayramı 3. günü
    "2025-04-23",   # Ulusal Egemenlik ve Çocuk Bayramı
    "2025-05-01",   # İşçi Bayramı
    "2025-05-19",   # Atatürk'ü Anma, Gençlik ve Spor Bayramı
    "2025-06-05",   # Kurban Bayramı arife
    "2025-06-06",   # Kurban Bayramı 1. günü
    "2025-06-07",   # Kurban Bayramı 2. günü
    "2025-06-08",   # Kurban Bayramı 3. günü
    "2025-06-09",   # Kurban Bayramı 4. günü
    "2025-07-15",   # Demokrasi ve Milli Birlik Günü
    "2025-08-30",   # Zafer Bayramı
    "2025-10-28",   # Cumhuriyet Bayramı arife (half-day)
    "2025-10-29",   # Cumhuriyet Bayramı
]

# -- Quality gates -------------------------------------------------------------
# Minimum trading-day overlap (sentiment rows that have a matching price row)
# required before signal statistics are shown as reliable.
# Below this threshold, scatter/rolling panels carry a PRELIMINARY watermark.
MINIMUM_OVERLAP_DAYS      = 30
# Days whose scored headline count is below this are marked unreliable in the
# sentiment bar chart (hatched bars) and excluded from signal stats.
MINIMUM_HEADLINES_PER_DAY = 3

# -- News categories -----------------------------------------------------------
# Ordered list of (category_slug, [keywords...]).
# classify_headline() returns the FIRST matching category (priority order).
# Keywords must use ASCII-folded forms (same convention as RELEVANCE_KEYWORDS).
#
# Priority rationale:
#   1. bist_company    - most specific; BIST/hisse proper nouns
#   2. rates_tcmb      - TCMB / hazine / tahvil - high-impact catalyst
#   3. political_risk  - opposition arrests, elections, crises — sudden market shocks
#   4. turkey_macro    - broad macro (inflation, growth, budget)
#   5. crypto          - BEFORE fx_lira: "bitcoin" + "dolar" in one headline
#                        should classify as crypto, not FX.  Crypto keywords
#                        are proper nouns and unambiguous.
#   6. global_risk     - BEFORE fx_lira: geopolitical or central-bank news
#                        that mentions foreign currencies incidentally.
#   7. fx_lira         - generic FX/TL terms (comes after more specific buckets)
#   8. banks           - sector
#   9. energy_commodities - commodities (oil, gas, metals, grains)
#   "other"            - catch-all if no rule matches
#
# Expanding keywords: any term also in RELEVANCE_KEYWORDS that has no category
# bucket creates an "other" headline that passed the relevance gate.  Keep the
# two lists in sync by reviewing  python evaluate.py --layer 1  periodically.
NEWS_CATEGORIES = [
    ("bist_company",       ["bist", "borsa istanbul", "hisse", "endeks",
                            "xu100", "xu030",
                            "borsa gune", "borsa gunu",   # daily open/close summaries
                            "borsa", "borsada",            # bare "borsa" = BIST in Turkish context
                            "halka arz",                   # IPO
                            "spk",                         # Capital Markets Board
                            "hisse senedi",
                            "sermaye piyasas",             # capital markets
                            "piyasa degeri",               # market cap stories
                            # KAP (material disclosures from listed companies)
                            "ozel durum",                  # "Özel Durum Açıklaması" — KAP format
                            "finansal sonuc",              # earnings/financial results release
                            # Major BIST constituents by name (too prominent to miss)
                            "thy", "turk hava yollari", "thyao",  # BIST's largest company by market cap
                            "holding",                     # corporate groups (Yıldız, Sabancı, Koç etc.)
                            "iflas",                       # company bankruptcy — always market-relevant
                            ]),
    ("rates_tcmb",         ["tcmb", "merkez bankasi", "faiz", "repo",
                            "para politikasi",
                            "hazine", "tahvil", "bono",
                            "sukuk",                       # Islamic bonds
                            ]),
    # Political risk: Turkey-specific political events that move BIST.
    # Placed AFTER rates_tcmb so "merkez bankasi baskani istifa" stays in rates.
    # Placed BEFORE turkey_macro so political crises don't dilute macro signals.
    #
    # Design principle: keywords here are SPECIFIC — party names, leader names,
    # event-type words (arrest, election, strike). Generic political adjectives
    # ("siyasi") are only included in compound phrases to avoid noise.
    # political_risk: ONLY event-driven headlines that can cause a BIST shock.
    # Routine party/parliament news (meeting minutes, local wins) is excluded.
    # Bare party acronyms (chp, akp, mhp) are intentionally NOT here — a routine
    # "AKP holds regional meeting" headline would bloat this category without
    # adding signal.  Those headlines still pass the relevance filter and land in
    # turkey_macro or "other", which is correct.
    ("political_risk",     [
                            # Named leaders — ANY mention is market-relevant
                            "imamoglu", "kilicdaroglu", "ozgur ozel",
                            # Event types: arrest/detention of a public figure
                            "gozalti", "tutuklama",
                            # Resignation (PM/minister/CB governor/CEO)
                            "istifa",
                            # Electoral shocks
                            "erken secim",                 # early election call
                            "secim sonuclari",             # election results night
                            # Explicit crisis language
                            "siyasi kriz",                 # political crisis framing
                            "siyasi belirsizlik",          # political uncertainty
                            # Cabinet changes (market-moving, not routine)
                            "kabine degisikligi",          # cabinet reshuffle
                            "cumhurbaskani yardimcisi",    # VP-level appointment
                            # Civil unrest with economy-wide impact
                            "grev",                        # nationwide strike
                            "protesto",                    # large-scale protest
                            # Constitutional/legal shocks
                            "anayasa mahkemesi",           # Constitutional Court
                            "ysk",                         # Supreme Election Council
                            ]),
    ("turkey_macro",       ["enflasyon", "buyume", "gsyh", "butce", "cari acik",
                            "ihracat", "ithalat", "issizlik", "tufe", "ufe",
                            "ekonomi", "istihdam",
                            "savunma sanayi",              # defense industry exports
                            "turizm",                      # tourism receipts
                            "gayrimenkul",                 # real estate market
                            "konut satis",                 # housing sales data
                            "konut yatirim",               # housing investment
                            "yabanci yatirim",             # FDI headlines
                            "serbest ticaret",             # Free Trade Agreement talks
                            "ticaret anlasmas",            # trade agreement (generic)
                            "suriye",                      # Turkey-Syria trade corridor (recurring macro theme)
                            "tuik",                        # TÜİK data releases (CPI, employment, trade stats)
                            ]),
    # crypto before fx_lira: proper nouns win over the generic word "dolar"
    ("crypto",             ["bitcoin", "ethereum", "kripto", "btc", "nft",
                            " coin",                       # catches "Bee coin", "SOL coin" etc.
                            "solana",
                            ]),
    # global_risk before fx_lira: geopolitical headlines that mention foreign
    # currencies should land here, not in fx_lira
    ("global_risk",        ["fed", "ecb", "jeopolitik", "savas", "kriz",
                            "resesyon",
                            "askeri", "yaptirim", "ambarago",
                            "nato",                        # NATO summits affect Turkey
                            "avrupa borsas",               # European stock markets
                            "dunya borsas",                # global markets headline
                            "kuresel piyasalar",           # "Küresel piyasalar" — global markets
                            "abd borsas",                  # US stock market headlines
                            "kuresel borsa",               # global stock markets
                            # Credit rating actions (sovereign or sector)
                            "kredi notu",                  # credit rating
                            "not artirimi",                # rating upgrade
                            "not indirimi",                # rating downgrade
                            "moodys",                      # Moody's
                            "fitch",                       # Fitch Ratings
                            "s&p",                         # S&P Global
                            ]),
    ("fx_lira",            ["dolar", "euro", "doviz",
                            " kur",                        # space-guard: avoids "kurul" (board/council)
                            "lira", "turk lirasi", "tl kuru",
                            "tl analiz",                   # TL/currency analysis pieces
                            ]),
    ("banks",              ["banka", "bankacilik", "kredi", "mevduat"]),
    ("energy_commodities", ["petrol", "dogalgaz", "altin", "emtia",
                            "enerji", "brent", "opec",
                            "lng",                         # liquefied natural gas trade
                            # Metals and agricultural commodities
                            "bakir", "demir", "celik", "aluminyum",
                            "bugday", "misir", "pamuk",
                            "findik",                      # hazelnut — Turkey's #1 ag export
                            "tahil",                       # grain (generic)
                            "elektrik", "yenilenebilir"]),
]

# -- Economic calendar ---------------------------------------------------------
# Known scheduled events that can explain sentiment/price moves.
# NOT YET WIRED IN: no module reads this yet — reserved for the event-calendar
# integration (chart overlay + event-study tables, see ROADMAP.md). Kept
# up to date so the data is ready when that lands.
#
# IMPORTANT: dates below are APPROXIMATE for 2026.
# Verify and update from official sources before each event:
#   TCMB PPK meetings : https://www.tcmb.gov.tr/wps/wcm/connect/tr/tcmb+tr/main+menu/para+politikasi
#   TUIK CPI releases  : https://www.tuik.gov.tr/duyurular/
#
# Format: "YYYY-MM-DD" -> "short description"
ECONOMIC_CALENDAR = {
    # TCMB Para Politikasi Kurulu (Monetary Policy Committee) meetings
    # Typically 8-9 meetings per year, ~6 weeks apart.
    # Decision announced same day at 14:00 Istanbul time.
    "2026-01-22": "TCMB PPK",
    "2026-03-05": "TCMB PPK",
    "2026-04-16": "TCMB PPK",
    "2026-05-21": "TCMB PPK",
    "2026-07-02": "TCMB PPK",
    "2026-08-13": "TCMB PPK",
    "2026-09-24": "TCMB PPK",
    "2026-11-05": "TCMB PPK",
    "2026-12-17": "TCMB PPK",
    # TÜİK TÜFE (CPI) monthly release — typically 3rd business day of month
    # Covers prior month's inflation; always high-impact for TCMB expectations.
    "2026-01-06": "TUIK TUFE",
    "2026-02-04": "TUIK TUFE",
    "2026-03-04": "TUIK TUFE",
    "2026-04-03": "TUIK TUFE",
    "2026-05-05": "TUIK TUFE",
    "2026-06-03": "TUIK TUFE",
    "2026-07-03": "TUIK TUFE",
    "2026-08-04": "TUIK TUFE",
    "2026-09-03": "TUIK TUFE",
    "2026-10-05": "TUIK TUFE",
    "2026-11-04": "TUIK TUFE",
    "2026-12-03": "TUIK TUFE",
}
