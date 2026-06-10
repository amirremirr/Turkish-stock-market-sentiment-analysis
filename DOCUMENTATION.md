# BIST 100 Turkish News Sentiment Pipeline — Full Documentation

**Created:** May 2026  
**Language:** Python 3.10 · SQLite · XLM-RoBERTa

---

## Table of Contents

1. [What this project does](#1-what-this-project-does)
2. [Architecture overview](#2-architecture-overview)
3. [Data flow — step by step](#3-data-flow--step-by-step)
4. [File reference](#4-file-reference)
5. [Database schema](#5-database-schema)
6. [Configuration reference](#6-configuration-reference)
7. [CLI commands](#7-cli-commands)
8. [Automation](#8-automation)
9. [Quality audit system (evaluate.py)](#9-quality-audit-system-evaluatepy)
10. [The visualisation](#10-the-visualisation)
11. [Turkish language handling](#11-turkish-language-handling)
12. [News category classifier](#12-news-category-classifier)
13. [Relevance filter](#13-relevance-filter)
14. [Test suite](#14-test-suite)
15. [Current state and known limitations](#15-current-state-and-known-limitations)
16. [What to do next](#16-what-to-do-next)

---

## 1. What this project does

This pipeline collects Turkish financial news headlines every weekday, scores each headline for market sentiment using a multilingual NLP model, and compares that sentiment signal against BIST 100 index daily returns.

**The core research question:** Does the sentiment of Turkish financial news published on day *t* predict the direction of the BIST 100 index on day *t+1*?

**What it is:**
- A data-accumulation engine for building a sentiment dataset over time
- A quality-audited research tool with explicit uncertainty gates
- A foundation for future backtesting and model validation

**What it is not (yet):**
- A trading signal (not enough data, not validated)
- A real-time system (daily batch, not streaming)
- A production system (single price source, no redundancy)

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DAILY PIPELINE                           │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐  │
│  │  Scraper │──▶│ Relevance│──▶│ Sentiment │──▶│  Aggreg. │  │
│  │  (RSS +  │   │  Filter  │   │  Scorer   │   │  (daily  │  │
│  │   HTML)  │   │ (2-tier) │   │ XLM-RoBERTa   │  averages│  │
│  └──────────┘   └──────────┘   └───────────┘   └──────────┘  │
│       │                              │                 │        │
│       ▼                              ▼                 ▼        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                     SQLite DB                          │    │
│  │  headlines  ·  bist100_prices  ·  daily_sentiment      │    │
│  │  category_daily_sentiment  ·  pipeline_runs            │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│       ┌──────────────────────┼──────────────────┐             │
│       ▼                      ▼                  ▼             │
│  ┌──────────┐          ┌──────────┐       ┌──────────┐       │
│  │  yfinance│          │Visualise │       │ evaluate │       │
│  │  prices  │          │  (PNG)   │       │  (audit) │       │
│  └──────────┘          └──────────┘       └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data flow — step by step

### Step 1 — Scrape (`scrape_step`)

**What happens:**
1. Opens HTTP sessions to 5 configured RSS feeds simultaneously (sequential with 1.5 s crawl delay)
2. Parses RSS 2.0 / Atom XML for each feed — handles BOM bytes, windows-1254 encoding, CDATA wrappers
3. Each headline goes through a **two-tier relevance filter** (see §13)
4. Headlines that pass are classified into one of 8 category buckets (see §12)
5. Surviving headlines are de-duplicated by URL (or first 120 chars of title as fallback)
6. If ALL RSS sources fail, the HTML scraper for `tr.investing.com` is used as fallback
7. New unique headlines are `INSERT OR IGNORE`d into the `headlines` table
8. Per-source status (`ok (N items)` / `failed: error message`) is logged

**RSS sources configured:**
| Key | URL | Typical volume |
|---|---|---|
| `investing_tr_stocks` | `tr.investing.com/rss/news_301.rss` | ~10 items |
| `investing_tr_economy` | `tr.investing.com/rss/news_1.rss` | ~10 items |
| `bloomberght` | `bloomberght.com/rss` | ~20 items |
| `dunya` | `dunya.com/rss` | ~25 items |
| `sabah_ekonomi` | `sabah.com.tr/rss/ekonomi.xml` | ~10 items |

**Output:** N new rows in `headlines` table (title, URL, date, source, category)

---

### Step 2 — Score (`score_step`)

**What happens:**
1. Queries all `headlines` rows where `sentiment_score IS NULL`
2. Loads the XLM-RoBERTa model (lazy-loaded on first call — downloads ~1.1 GB on first run)
3. Sends headlines in batches of 32 (max 128 tokens per headline) through the model
4. Model returns logits for three labels: `negative`, `neutral`, `positive`
5. Softmax converts logits to probabilities: `p_positive`, `p_neutral`, `p_negative`
6. Continuous score = `p_positive − p_negative` ∈ [−1, +1]
7. Dominant label = whichever of the three probabilities is highest
8. All five values + model name + timestamp are written back to the `headlines` row

**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`  
Trained on 198M multilingual tweets. Supports Turkish natively. Not fine-tuned on financial text.

**Score interpretation:**
| Score range | Meaning |
|---|---|
| +1.0 | Maximally bullish (model is 100% confident: positive) |
| +0.3 to +1.0 | Positive sentiment |
| −0.1 to +0.3 | Neutral zone |
| −1.0 to −0.1 | Negative sentiment |
| −1.0 | Maximally bearish |

**Output:** `sentiment_score`, `sentiment_label`, `p_positive`, `p_neutral`, `p_negative`, `model_name`, `scored_at` columns populated in `headlines`

---

### Step 3 — Aggregate (`aggregate_step`)

**What happens:**
1. Backfills `category` for any headlines that have `NULL` (inserted before the classifier existed)
2. **Deletes ALL rows** from `daily_sentiment` and `category_daily_sentiment` — guarantees no stale data ever survives a clean or re-scrape
3. Groups all scored headlines by `published_at` date
4. For each day, computes:
   - `avg_score` — mean of all sentiment scores that day
   - `std_score` — standard deviation
   - `headline_count` — number of headlines
   - `positive_count`, `neutral_count`, `negative_count` — label breakdown
   - `bull_bear_ratio` = `positive / (positive + negative)` — ignores neutrals
5. Also groups by `(date, category)` and writes per-category rows to `category_daily_sentiment`
6. Upserts both result sets

**Why delete-then-recompute?**  
If you run `clean` to remove off-topic headlines and then `aggregate`, the daily averages must reflect the cleaned state. An upsert-only approach would leave stale rows for days that lost all their headlines. Deleting first makes the derived tables always a faithful reflection of what's actually in `headlines`.

**Output:** Rows in `daily_sentiment` and `category_daily_sentiment`

---

### Step 4 — Prices (`prices_step`)

**What happens:**
1. Downloads BIST 100 OHLCV data via `yfinance` (ticker: `XU100.IS`)
2. Handles the MultiIndex columns yfinance ≥0.2.38 returns
3. Computes `daily_return = close.pct_change() × 100` (percentage)
4. Upserts rows into `bist100_prices` (INSERT OR REPLACE — overwrites if already present)

**Output:** Up to 90 days of price rows (configurable via `DEFAULT_LOOKBACK_DAYS`)

---

### Step 5 — Plot (`plot_step`)

Generates a 3-panel PNG. See §10 for details.

---

### Full pipeline (`run_all`)

Runs steps 1–5 in sequence. Before step 1, inserts a row in `pipeline_runs` with `status='running'`. After all steps complete, updates it to `status='ok'`. If any step raises an exception, updates to `status='error'` with the error message, then re-raises.

This means every full run is permanently recorded in the database audit trail.

---

## 4. File reference

```
finance/
│
├── config.py            Central configuration (all tunable parameters)
├── database.py          SQLite layer (schema, migrations, all queries)
├── scraper.py           RSS + HTML headline fetcher, relevance filter, classifier
├── sentiment.py         XLM-RoBERTa wrapper (lazy-loaded, batch inference)
├── pipeline.py          Orchestrates steps 1–5, run_all(), clean_step()
├── visualize.py         3-panel matplotlib figure
├── main.py              CLI entry point (argparse, subcommands)
├── evaluate.py          Quality audit (L0–L5 report)
│
├── run.bat              Convenience launcher (resolves Python: .venv > venv > PATH)
├── run_scheduled.py     Daily automation script (called by Task Scheduler)
├── register_task.bat    One-time Task Scheduler registration (run as Admin)
├── task_modified.xml    Task Scheduler XML (StartWhenAvailable + battery fix)
│
├── finance_sentiment.db SQLite database (all data lives here)
├── sentiment_vs_bist100.png  Latest plot output
│
└── tests/
    └── test_pipeline.py  99-test pytest suite (no torch dependency)
```

### config.py
Single source of truth for every tunable parameter. Edit here; no code changes needed.  
Key sections: DB path, RSS feeds, relevance filter lists, sentiment model name, quality gate thresholds, news category taxonomy.

### database.py
All SQLite interaction. Functions:
- `init_db()` — creates tables + runs migrations
- `insert_headlines()` — bulk insert with dedup
- `get_unscored_headlines()` — returns DataFrame of rows needing scoring
- `batch_update_sentiment()` — writes scores back in one transaction
- `upsert_prices()` / `get_prices()` — BIST 100 price rows
- `upsert_daily_sentiment()` / `get_daily_sentiment()` — aggregated signal
- `upsert_category_sentiment()` / `get_category_daily_sentiment()` — per-category signal
- `log_run_start()` / `log_run_end()` — audit trail
- `clean_off_topic_headlines()` — deletes headlines that fail current filter
- `db_stats()` — summary statistics for `main.py status`

Schema migrations are handled via `_MIGRATIONS` list + `_apply_migrations()`, which uses `PRAGMA table_info()` to safely add columns to live databases without data loss.

### scraper.py
- `_normalise(text)` — ASCII-folds all 12 Turkish diacritics, then lowercases (see §11)
- `_is_relevant(title)` — two-tier relevance filter (see §13)
- `classify_headline(title)` — maps headline to one of 8 category buckets (see §12)
- `RSSFeedScraper` — fetches and parses all configured RSS feeds; records `source_status` dict
- `InvestingTRScraper` — HTML fallback for `tr.investing.com`

### sentiment.py
- `SentimentScorer` — wraps HuggingFace `pipeline("sentiment-analysis")`; lazy-loads on first call
- `_extract_score(result)` — converts raw model output to `(score, label, p_pos, p_neu, p_neg)`
- `get_scorer()` — singleton; returns cached scorer after first load

### pipeline.py
Pure orchestration. Each `*_step()` function does one thing and returns a count. `run_all()` chains them and manages the audit trail. `clean_step()` calls `db.clean_off_topic_headlines()` and re-runs `aggregate_step()`.

### visualize.py
Three-panel figure:
- **Panel 1:** BIST 100 closing price (line + fill)
- **Panel 2:** Daily sentiment bars (green positive / red negative; hatched if `< MINIMUM_HEADLINES_PER_DAY`)
- **Panel 3a:** Scatter plot of today's sentiment vs next-day return + OLS line + Pearson r
- **Panel 3b:** 30-day rolling correlation between sentiment and next-day return

Panels 3a and 3b are watermarked **PRELIMINARY** until `MINIMUM_OVERLAP_DAYS` = 30 overlapping days exist.

### main.py
CLI entry point. Parses arguments and routes to the correct `cmd_*` function. All subcommands share `--db`, `--days`, `--output`, `--no-show`, `--log-level` flags via a parent parser.

### evaluate.py
Independent read-only audit tool. Six layers (see §9). Imports from `config.py` so thresholds stay in sync automatically.

---

## 5. Database schema

### `headlines`
The primary data table.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `source` | TEXT | RSS source key (e.g. `bloomberght`) |
| `title` | TEXT | Original Turkish headline |
| `url` | TEXT UNIQUE | Article URL (dedup key) |
| `published_at` | TEXT | ISO date of publication |
| `scraped_at` | TEXT | UTC timestamp when scraped |
| `category` | TEXT | Classifier bucket (e.g. `fx_lira`) |
| `sentiment_score` | REAL | P(positive) − P(negative) ∈ [−1, +1] |
| `sentiment_label` | TEXT | `positive` / `neutral` / `negative` |
| `p_positive` | REAL | Raw model probability for positive |
| `p_neutral` | REAL | Raw model probability for neutral |
| `p_negative` | REAL | Raw model probability for negative |
| `model_name` | TEXT | Model identifier (for auditability) |
| `scored_at` | TEXT | UTC timestamp when scored |

### `bist100_prices`
| Column | Type | Description |
|---|---|---|
| `date` | TEXT PK | ISO date (YYYY-MM-DD) |
| `open` / `high` / `low` / `close` | REAL | OHLC prices |
| `volume` | REAL | Trading volume |
| `daily_return` | REAL | `(close_t / close_{t-1} − 1) × 100` percent |

### `daily_sentiment`
Derived table. Rebuilt from scratch on every `aggregate` run.

| Column | Type | Description |
|---|---|---|
| `date` | TEXT PK | ISO date |
| `avg_score` | REAL | Mean sentiment score for the day |
| `std_score` | REAL | Standard deviation |
| `headline_count` | INTEGER | Number of scored headlines that day |
| `positive_count` / `neutral_count` / `negative_count` | INTEGER | Label breakdown |
| `bull_bear_ratio` | REAL | `positive / (positive + negative)` |
| `updated_at` | TEXT | When this row was last computed |

### `category_daily_sentiment`
Also rebuilt from scratch on every `aggregate` run.

| Column | Type | Description |
|---|---|---|
| `date` + `category` | TEXT PK (composite) | One row per (day, category) pair |
| `avg_score` | REAL | Mean sentiment for that category that day |
| `headline_count` | INTEGER | Headlines in that category that day |

### `pipeline_runs`
Audit trail. One row per `run_all()` call.

| Column | Type | Description |
|---|---|---|
| `run_id` | INTEGER PK | Auto-increment |
| `started_at` | TEXT | UTC start timestamp |
| `finished_at` | TEXT | UTC end timestamp |
| `headlines_scraped` | INTEGER | New headlines added this run |
| `headlines_scored` | INTEGER | Headlines scored this run |
| `prices_added` | INTEGER | Price rows stored this run |
| `sentiment_days` | INTEGER | Daily sentiment rows computed |
| `model_name` | TEXT | Sentiment model used |
| `status` | TEXT | `running` / `ok` / `error` |
| `error_msg` | TEXT | Exception message if `status='error'` |

---

## 6. Configuration reference

All settings are in [config.py](config.py). Key parameters:

| Parameter | Default | What it controls |
|---|---|---|
| `DB_PATH` | `finance_sentiment.db` | SQLite file location |
| `BIST100_TICKER` | `XU100.IS` | Yahoo Finance ticker |
| `DEFAULT_LOOKBACK_DAYS` | `90` | How far back to seed on first run |
| `RSS_FEEDS` | 5 feeds | Sources to scrape (add/remove freely) |
| `RELEVANCE_FILTER_ENABLED` | `True` | Toggle the two-tier filter |
| `RELEVANCE_STRONG` | 6 terms | Turkey anchors that override blocklist |
| `RELEVANCE_BLOCKLIST` | 13 terms | Terms that trigger rejection |
| `RELEVANCE_KEYWORDS` | ~30 terms | At least one must be present to pass |
| `SENTIMENT_MODEL` | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | HuggingFace model ID |
| `SENTIMENT_BATCH_SIZE` | `32` | Headlines per model batch |
| `SENTIMENT_MAX_LENGTH` | `128` | Token limit per headline |
| `MINIMUM_OVERLAP_DAYS` | `30` | Overlap days before signal stats are shown as reliable |
| `MINIMUM_HEADLINES_PER_DAY` | `3` | Headlines/day below this → hatched bars, excluded from stats |
| `PLOT_DAYS` | `90` | Chart window |
| `PLOT_DPI` | `150` | Output resolution |
| `NEWS_CATEGORIES` | 8 buckets | Category taxonomy (priority-ordered) |

---

## 7. CLI commands

All commands go through `main.py` or the `run.bat` wrapper.

```
run.bat run                     # Full pipeline: scrape → score → aggregate → prices → plot
run.bat scrape                  # Only fetch new headlines
run.bat score                   # Only score unscored headlines (loads model)
run.bat aggregate               # Recompute daily_sentiment from current headlines
run.bat recategorize            # Re-classify all headlines with current rules + re-aggregate
run.bat prices                  # Download latest BIST 100 prices
run.bat fx-rates                # Download USD/TRY FX rates (requires ALPHA_VANTAGE_KEY in .env)
run.bat plot                    # Regenerate the chart
run.bat status                  # Print DB statistics
run.bat clean                   # Delete headlines that fail current relevance filter
run.bat clean --dry-run         # Preview how many would be deleted
run.bat export-labels           # Export 150 headlines as CSV for manual labelling
run.bat export-labels --n 300   # Export 300 headlines
run.bat test                    # Run the pytest test suite
```

Global flags (work with any subcommand):
```
--db PATH        Use a different SQLite file
--days N         Override the lookback window (default 90)
--no-show        Don't open the interactive plot window (essential for automation)
--log-level      DEBUG | INFO | WARNING
```

**Direct Python (same result, no wrapper):**
```
python main.py run --no-show
python evaluate.py
python evaluate.py --layer 1
python evaluate.py --layer 2 --sample 20
```

---

## 8. Automation

### How it works

Two files handle automation:

**`run_scheduled.py`** — the actual daily job:
1. Checks if today is a weekday (skips Sat/Sun via Python `datetime.weekday()`)
2. Creates `logs\` directory if missing
3. Writes all output to `logs\YYYY-MM-DD.log`
4. Runs `main.py run --no-show` (full pipeline, no GUI window)
5. Runs `evaluate.py` immediately after (quality audit appended to same log)
6. Exits with pipeline's exit code (Task Scheduler can detect failures)

**`register_task.bat`** — one-time setup (run as Administrator):
- Imports `task_modified.xml` into Windows Task Scheduler
- Creates task **BIST100-Sentiment**: Mon–Fri at 07:30
- With `StartWhenAvailable=true`: if laptop was closed at 07:30, task runs the moment it wakes up
- With battery restriction removed: runs on battery power too

### Task settings summary

| Setting | Value |
|---|---|
| Task name | `BIST100-Sentiment` |
| Trigger | Weekly: Mon, Tue, Wed, Thu, Fri at 07:30 |
| Run if missed | ✅ Yes — fires on wake if laptop was closed |
| Run on battery | ✅ Yes |
| Run as | Current user (RasaComputer) |

### Useful Task Scheduler commands

```
# Check task status
schtasks /query /tn "BIST100-Sentiment" /fo LIST /v

# Run immediately (test it)
schtasks /run /tn "BIST100-Sentiment"

# Pause (e.g. travelling)
schtasks /change /tn "BIST100-Sentiment" /disable

# Resume
schtasks /change /tn "BIST100-Sentiment" /enable

# Remove entirely
schtasks /delete /tn "BIST100-Sentiment" /f
```

### Reading the logs

Log files appear in `finance\logs\YYYY-MM-DD.log`.
Each log contains:
- Pipeline step outputs (`[OK] Scrape - N new headlines`)
- Any per-source failures (`[source] bloomberght  failed: ...`)
- Full evaluate.py audit (L0 through L5)
- Exit code summary at bottom

---

## 9. Quality audit system (evaluate.py)

Run `python evaluate.py` for a full report, or `--layer N` for one section.

### L0 — System health
- Pipeline run log (how many runs, last run status, error if any)
- Derived-table freshness: is `daily_sentiment` up to date with the latest scored headlines?
- Schema completeness: what % of scored headlines have raw probability fields?
- Category backfill status

### L1 — Scraper quality
- Total headlines, date coverage, span, gaps
- Headlines per day (min / mean / max)
- Source breakdown (how many from each RSS feed)
- URL deduplication check
- Turkish character encoding health (should be >30% with Turkish chars)
- Category breakdown with warning if `other` > 20%
- **Blocklist-override edge cases**: headlines that passed the filter only because a strong Turkey marker (`turk`, `bist`, etc.) overrode a blocklist hit (`bitcoin`, `nasdaq`, etc.). These are the highest misclassification risk — displayed for manual review.

### L2 — Model quality
- Raw probability completeness (% of scored headlines with p_positive populated)
- Model version (warns if multiple model versions in DB)
- Score distribution (mean, median, std dev, percentiles)
- Confidence check: how many scores are near-zero (under-confident) vs decisive (|s| > 0.4)
- Label breakdown (warns if >70% neutral)
- Spot-check: N random headlines with their scores printed for manual review

### L3 — Aggregation quality
- Signal thickness per day vs `MINIMUM_HEADLINES_PER_DAY` gate
- Signal volatility (warns if daily signal variance is too low)
- Bull/bear ratio vs avg_score consistency
- Per-category mean sentiment table

### L4 — Price data quality
- Coverage and staleness (warns if >3 days stale)
- Missing weekday detection (finds gaps — likely public holidays)
- Return distribution sanity (warns if any |return| > 5%)
- Price level sanity (latest close, 52-week high/low)

### L5 — Signal quality (gated)
**Only shown when `MINIMUM_OVERLAP_DAYS` = 30 overlapping days exist.**

- Pearson r: sentiment(t) vs return(t+1) — the predictive direction
- Pearson r: sentiment(t) vs return(t) — same-day correlation (reaction vs lead)
- Hit rate: does positive sentiment predict market up next day?
- Binomial test: is the hit rate statistically different from random?
- Naive strategy cumulative return vs buy-and-hold
- Granger causality test (if ≥20 overlap days)

---

## 10. The visualisation

File: `sentiment_vs_bist100.png` (regenerated on every `run` or `plot` command)

**Panel 1 (top, full width): BIST 100 closing price**
- Blue line + light fill
- Turkish labels (kapanış = closing)

**Panel 2 (middle, full width): Daily sentiment bars**
- Green bars = positive average sentiment, red bars = negative
- Hatched / faded bars = days with fewer than `MINIMUM_HEADLINES_PER_DAY` headlines (unreliable)
- Orange dashed line = 5-day rolling average
- Number above each bar = headline count for that day

**Panel 3a (bottom-left): Scatter plot**
- X axis: sentiment score on day t
- Y axis: BIST 100 return on day t+1
- Points coloured by return (red=down, green=up)
- OLS regression line in purple
- Pearson r and p-value annotated

**Panel 3b (bottom-right): 30-day rolling correlation**
- Correlation between sentiment(t) and return(t+1) computed in a rolling window
- Green fill = positive correlation period, red fill = negative

**PRELIMINARY watermark:** Panels 3a and 3b show a red diagonal watermark until 30 overlapping days exist. This is intentional — the scatter and correlation are statistically meaningless with fewer than 30 points.

---

## 11. Turkish language handling

Turkish has 6 special characters not in ASCII: `ş ç ğ ı ö ü` (plus uppercase `Ş Ç Ğ İ Ö Ü`). Python's `.lower()` does not map these to their ASCII equivalents — `"Türkiye".lower()` returns `"türkiye"`, not `"turkiye"`. Since all keywords in `config.py` use ASCII-folded forms, headlines must be normalised the same way before matching.

The `_normalise()` function in `scraper.py` uses a `str.maketrans` table mapping all 12 Unicode codepoints to their ASCII equivalents:

| Turkish char | Unicode | Maps to |
|---|---|---|
| İ (capital dotted I) | U+0130 | i |
| ı (dotless lowercase i) | U+0131 | i |
| Ş / ş | U+015E / U+015F | s |
| Ç / ç | U+00C7 / U+00E7 | c |
| Ğ / ğ | U+011E / U+011F | g |
| Ö / ö | U+00D6 / U+00F6 | o |
| Ü / ü | U+00DC / U+00FC | u |

After this mapping, `.lower()` is applied. Both `_is_relevant()` and `classify_headline()` call `_normalise()` first, so "Türkiye", "turkiye", "TÜRKIYE", and "Türkiye" all match the keyword `"turkiye"`.

---

## 12. News category classifier

`classify_headline(title)` in `scraper.py` normalises the headline and tries each category in priority order, returning the first match. Returns `"other"` if nothing matches.

**Priority order (highest to lowest):**

| Priority | Category | Why this priority |
|---|---|---|
| 1 | `bist_company` | Most specific — BIST/hisse are proper nouns |
| 2 | `rates_tcmb` | High-impact policy catalyst |
| 3 | `turkey_macro` | Broad macro — comes before FX to capture growth/inflation stories |
| 4 | `crypto` | **Before fx_lira**: "Bitcoin 70.000 dolara" → crypto, not FX |
| 5 | `global_risk` | **Before fx_lira**: geopolitical stories that mention foreign currencies land here |
| 6 | `fx_lira` | Generic FX/TL terms — catches what wasn't caught above |
| 7 | `banks` | Sector-specific |
| 8 | `energy_commodities` | Oil, gas, metals, agricultural |
| — | `other` | Catch-all if nothing matches |

The `other` category is a health metric. If it exceeds 20% of headlines, the taxonomy needs expanding. The evaluate.py L1 audit warns about this automatically.

---

## 13. Relevance filter

`_is_relevant(title)` applies two tiers:

**Tier 1 — Blocklist:**  
If the headline contains any blocklist term AND does not contain any strong Turkey marker, it is **dropped**.  
Example: "Bitcoin rallied to $70,000" → blocklist hit (`bitcoin`), no strong marker → dropped.  
Example: "Türk yatırımcılar Bitcoin'e ilgi gösteriyor" → blocklist hit (`bitcoin`) BUT strong marker present (`turk`) → passes to Tier 2.

**Tier 2 — Keyword gate:**  
The headline must contain at least one keyword from `RELEVANCE_KEYWORDS`. This catches generic news that slipped through Tier 1.

**Strong Turkey markers** (`RELEVANCE_STRONG`):
`turkiye`, `turk`, `bist`, `borsa istanbul`, `tcmb`, `turk lirasi`, `tl kuru`

**Blocklist** (`RELEVANCE_BLOCKLIST`):
`bitcoin`, `ethereum`, `kripto`, `btc`, `nft`, `new york borsasi`, `nasdaq`, `wall street`, `sterling`, `pound`, `nikkei`, `dax`, `ftse`, `hang seng`, and other country/exchange names

**Keywords** (`RELEVANCE_KEYWORDS`): ~30 terms covering stock market (`borsa`, `hisse`), monetary policy (`faiz`, `tcmb`), macro (`enflasyon`, `büyüme`), FX (`dolar`, `euro`, `kur`), commodities (`petrol`, `altin`), etc.

The filter can be disabled entirely with `RELEVANCE_FILTER_ENABLED = False` in `config.py`.

---

## 14. Test suite

**Location:** `tests/test_pipeline.py`  
**Run:** `run.bat test`  
**Count:** 99 tests, all passing  
**Dependency:** No torch — model is mocked via `_extract_score` unit tests

**Test classes:**

| Class | Tests | What it covers |
|---|---|---|
| `TestNormalise` | 17 | Each Turkish character individually + compound words |
| `TestRelevanceFilter` | ~15 | Blocklist, keyword matching, diacritic forms |
| `TestDateParsing` | ~8 | All `_DATE_FORMATS` patterns |
| `TestScoreExtraction` | ~8 | 5-tuple extraction, probabilities sum to 1, label detection |
| `TestCategoryClassifier` | 18 | All 8 buckets, priority ordering, diacritic normalisation |
| `TestAggregation` | ~12 | Math correctness, delete-then-recompute, stale row removal, category rows, p_positive storage, pipeline_runs audit |
| `TestVisualize` | 2 | PNG renders with full data, PRELIMINARY watermark present with thin data |

**Important implementation note:** The test file forces `matplotlib.use("Agg")` before importing `visualize`. Without this, the Windows App Store Python's partially-installed Tcl/Tk can cause the test suite to fail when run as a whole (though individual tests pass in isolation).

---

## 15. Current state and known limitations

As of June 2026:

| Metric | Value |
|---|---|
| Headlines in DB | ~492 (post-dedup) |
| Headlines scored | ~492 |
| Validation accuracy | 76.8% on 198 human-labeled headlines |
| Overlap days (sentiment ∩ prices, reliable) | ~10 of 30 needed for L5 gate |
| `published_hour` coverage | NULL for pre-June-2026 rows; populates going forward |

**Known limitations:**

1. **Signal too young.** L5 signal stats need 30 overlapping reliable trading days. Currently ~10. Resolves naturally as the daily pipeline accumulates data (~4 more weeks).

2. **`published_hour` missing for older rows.** All headlines scraped before June 2026 have `NULL` for the publication hour column. The time-of-day weight defaults to 1.0× (neutral) for these. Will improve going forward.

3. **Validation is in-sample.** The 76.8% accuracy was measured on 198 labels using the full dataset for threshold tuning (no holdout). A proper tune/test split on 300–500 labels is needed for reliable accuracy claims.

4. **Model not fine-tuned for Turkish finance.** The XLM-RoBERTa model was trained on tweets, not financial text. It may mishandle financial negation ("düşmedi" = "did not fall"), sarcasm, or domain-specific terms.

5. **Single price source.** Only Yahoo Finance (yfinance). No reconciliation or failover. If yfinance is down or changes the ticker format, prices silently stop updating.

See [METHODOLOGY.md](METHODOLOGY.md) for a full list of known limitations with mitigations.

---

## 16. What to do next

### Immediate (before the next session)

**1. Rescore existing headlines** to populate raw probability fields:
```python
# Clear old scores (keeps headlines, clears only the scoring columns)
import sqlite3
c = sqlite3.connect('finance_sentiment.db')
c.execute('''UPDATE headlines SET
    sentiment_score=NULL, sentiment_label=NULL,
    p_positive=NULL, p_neutral=NULL, p_negative=NULL,
    model_name=NULL, scored_at=NULL''')
c.commit()
c.close()

# Re-score and re-aggregate
python main.py score
python main.py aggregate
```

**2. Let the daily pipeline run.** The Task Scheduler is active. Every weekday at 07:30 (or on wake), it runs automatically. After 30 trading days (~6 weeks), the L5 signal stats will unlock.

---

### Short term (weeks)

**3. Model validation** — the most important step for research credibility:
```
run.bat export-labels --n 300
```
Open `labels_to_validate.csv`, fill the `human_label` column (`positive` / `neutral` / `negative`) for each headline, then compute:
```python
import pandas as pd
df = pd.read_csv('labels_to_validate.csv')
accuracy = (df['model_label'] == df['human_label']).mean()
by_category = df.groupby('category').apply(
    lambda g: (g['model_label'] == g['human_label']).mean()
)
```
Target: ≥70% accuracy per category. Anything below reveals where the model fails (likely negation-heavy headlines and macro announcements).

**4. Price source verification.** Add a second price source and cross-check for consistency. The current single source (Yahoo Finance) has no validation.

---

### Medium term (months)

**5. Backtesting framework.** Once you have 60–120 trading days of data:
- Walk-forward backtest (train on first 60 days, test on next 30, roll forward)
- Add transaction costs (bid-ask spread, commission)
- Compute Sharpe ratio, maximum drawdown
- Compare vs buy-and-hold baseline

**6. Relevance hardening.** Add a `blocked_headlines` table where you can manually mark headlines as noise. The relevance filter consults this table before accepting. This is better than adding more hardcoded rules because it lets you curate exceptions case by case.

**7. Model improvement.** If validation shows accuracy below 70% on a specific category, consider:
- Fine-tuning the XLM-RoBERTa model on your labeled dataset
- Adding rule overrides for systematic failures (e.g. always mark "enflasyon düştü" as positive even if the model disagrees)
- Switching to a larger model (Turkish BERT variants)

---

*For design decisions behind every pipeline choice, see [METHODOLOGY.md](METHODOLOGY.md).*
