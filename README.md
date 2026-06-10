# BIST 100 Turkish News Sentiment Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-104%20passing-brightgreen.svg)
![Model](https://img.shields.io/badge/sentiment-gpt--5--mini-orange.svg)
![Status](https://img.shields.io/badge/status-active%20research-yellow.svg)

A personal research project that collects Turkish financial news headlines every weekday, scores each headline for market sentiment using a multilingual NLP model, and tracks whether that signal correlates with BIST 100 daily returns.

> **Research question:** Does the sentiment of Turkish financial news on day *t* predict the direction of BIST 100 on day *t+1*?

> ⚠️ **Honest status:** ~750 headlines collected. Sentiment scored by gpt-5-mini with few-shot examples — **84.5% accuracy on held-out human-labeled headlines** (vs 76.8% for the previous XLM-RoBERTa scorer, which was tuned in-sample). ~15 of the 30 reliable overlap days needed before signal statistics are meaningful. This is a research tool, not a trading signal.
>
> **Read the 84.5% correctly:** it measures *agreement with a human reading headlines* — not predictive power. A scorer can label sentiment perfectly and still produce a useless market signal (news may react to prices, aggregation may destroy information, daily noise may dominate). Whether sentiment predicts returns is a separate, unanswered question that needs the 30-day gate and out-of-sample testing.

---

## Sample Output

![Sentiment vs BIST 100](docs/sample_output.png)

*Top: BIST 100 closing price. Middle: daily sentiment bars (green = bullish, red = bearish; hatched = thin data days). Bottom: lead-lag scatter and 30-day rolling correlation — watermarked PRELIMINARY until statistically meaningful.*

---

## How it works

```
Turkish RSS feeds  →  Two-tier relevance filter  →  Category classifier
                                                            │
                                                            ▼
                                          LLM sentiment score (gpt-5-mini, few-shot)
                                          score ∈ [−1, +1]  ·  XLM-RoBERTa fallback
                                                            │
                                                            ▼
                                        Confidence-weighted daily aggregate
                                        + time-of-day weighting (Istanbul UTC+3)
                                                            │
                                                   ┌────────┴────────┐
                                                   ▼                 ▼
                                              SQLite DB         BIST 100 prices
                                                   └────────┬────────┘
                                                            ▼
                                               3-panel chart  +  quality audit
```

---

## Features

- **LLM sentiment scoring** — gpt-5-mini with 30 few-shot examples from the human-labeled set; benchmarked at 84.5% on held-out labels. Local XLM-RoBERTa kept as an offline fallback backend (`SENTIMENT_BACKEND` in config.py)
- **Smart relevance filter** — Two-tier system drops off-topic stories (crypto, foreign indices) before scoring
- **9-category taxonomy** — `bist_company`, `rates_tcmb`, `fx_lira`, `banks`, `global_risk`, `political_risk`, and more
- **Confidence weighting** — High-conviction headlines dominate the daily average, with a weight floor so a quiet news day reads as neutral instead of echoing its single loudest headline
- **Time-of-day weighting** — Pre-market news (before 10:00 Istanbul) weighted 1.5×, post-market 0.8×
- **Title-based deduplication** — Same story from multiple RSS sources or repeated across daily runs counted only once
- **6-layer quality audit** — Scraper health · source dashboard · model scores · aggregation · price data · signal statistics
- **Validation workflow** — Threshold sweep, confusion matrix, per-category accuracy, holdout split, label progress tracker
- **Daily automation** — Windows Task Scheduler with crash recovery, step-by-step retry, and daily logs
- **99 pytest tests** — Full suite, no GPU or model download required

---

## Quick Start

```bash
git clone https://github.com/amirremirr/Turkish-stock-market-sentiment-analysis.git
cd Turkish-stock-market-sentiment-analysis

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Optional: set your free Alpha Vantage key for USD/TRY FX rates
cp .env.example .env   # then edit .env

# First run — downloads ~1.1 GB XLM-RoBERTa model once, then runs offline
run.bat run
```

---

## Commands

| Command | Description |
|---------|-------------|
| `run.bat run` | Full pipeline: scrape → score → aggregate → prices → plot |
| `run.bat scrape` | Fetch latest headlines only |
| `run.bat score` | Score unscored headlines with the sentiment model |
| `run.bat aggregate` | Recompute daily sentiment averages |
| `run.bat recategorize` | Re-classify all headlines with keyword rules |
| `run.bat recategorize --llm` | Re-classify with the LLM and purge irrelevant headlines |
| `run.bat relabel` | Recompute sentiment labels from stored probabilities after a threshold change |
| `run.bat prices` | Download BIST 100 price history (Yahoo Finance) |
| `run.bat fx-rates` | Download USD/TRY FX rates (Alpha Vantage) |
| `run.bat plot` | Regenerate the visualisation |
| `run.bat status` | Show database statistics |
| `run.bat clean` | Remove off-topic headlines |
| `run.bat export-labels --n 300` | Export CSV for manual sentiment validation |
| `run.bat validate-labels <csv>` | Validate model against a labeled CSV (accuracy, confusion matrix, holdout) |
| `run.bat validate-labels --tracker` | Show label collection progress toward 300 / 500 targets |
| `run.bat test` | Run the full test suite |

---

## Quality Audit

```bash
python evaluate.py                        # Full 6-layer audit report
python evaluate.py --layer 2 --sample 20  # Model quality + 20 random spot-checks
python evaluate.py --save                 # Also save metrics snapshot to reports/
```

| Layer | What it checks |
|-------|---------------|
| L0 | System health, pipeline run log, schema completeness |
| L1 | Scraper quality, source quality dashboard, dedup audit, encoding health |
| L2 | Model score distribution, confidence, label balance, spot-check |
| L3 | Daily aggregation quality, bull/bear consistency |
| L4 | Price data coverage, staleness, return distribution |
| L5 | Pearson r, hit rate, Granger causality — **gated until 30 reliable overlap days** |

**Model validation** (once you have labeled headlines):

```bash
python main.py export-labels --n 300                  # export CSV, fill human_label column
python validate_labels.py labels_to_validate.csv       # accuracy, confusion matrix, holdout split
python validate_labels.py labels_to_validate.csv --save  # also save to reports/
python validate_labels.py --tracker                    # progress toward 300 / 500 label targets
```

---

## Project Structure

```
├── config.py            All tunable parameters (feeds, keywords, thresholds)
├── database.py          SQLite layer (schema, migrations, queries)
├── scraper.py           RSS fetcher, relevance filter, category classifier
├── sentiment.py         XLM-RoBERTa wrapper (lazy-loaded, batch inference)
├── pipeline.py          Step orchestration and aggregation math
├── visualize.py         3-panel matplotlib figure
├── main.py              CLI entry point
├── evaluate.py          Read-only quality audit (L0–L5), --save for trend reports
├── validate_labels.py   Model validation against human-labeled headlines
├── discover_sources.py  Evaluate new RSS feeds before adding them
│
├── run.bat              Windows convenience launcher
├── run_scheduled.py     Daily automation script (Task Scheduler)
│
├── METHODOLOGY.md       Every design decision with rationale
├── DOCUMENTATION.md     Full technical reference
│
└── tests/
    └── test_pipeline.py  99 tests, no torch dependency
```

---

## Design Decisions

Full rationale for every choice is in [METHODOLOGY.md](METHODOLOGY.md). Summary:

| Choice | Why |
|--------|-----|
| XLM-RoBERTa over translation | Lower latency, no API cost, handles Turkish natively |
| Threshold ±0.05 neutral band | Corrects model over-calling negative on routine financial language (+7.6 pp accuracy) |
| Confidence weighting | High-conviction scores carry more information than near-zero scores |
| Title dedup over URL dedup | Same story appears across multiple RSS sources with different URLs |
| 30-day signal gate | Fewer points produce statistically meaningless Pearson r values |

---

## Known Limitations

- Sentiment accuracy is ~84.5% on held-out human labels — the remaining errors are mostly genuinely ambiguous headlines where human annotators would also disagree
- LLM scoring depends on an external API; the XLM-RoBERTa fallback runs offline but scores lower (76.8% in-sample)
- Signal statistics gated at 30 reliable overlap days — currently ~10
- `published_hour` is NULL for pre-June-2026 headlines; time weighting active only for new scrapes

---

## Tech Stack

`Python 3.10` · `SQLite` · `OpenAI API (gpt-5-mini)` · `HuggingFace Transformers / XLM-RoBERTa (fallback)` · `pandas` · `yfinance` · `matplotlib` · `pytest`

---

## License

MIT — see [LICENSE](LICENSE)
