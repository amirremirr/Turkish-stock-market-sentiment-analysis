# BIST 100 Turkish News Sentiment Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen.svg)
![Sentiment](https://img.shields.io/badge/sentiment-gpt--5--mini-orange.svg)
![Status](https://img.shields.io/badge/status-active%20research-yellow.svg)

This project investigates whether sentiment extracted from Turkish financial news contains predictive information about subsequent **BIST 100** returns. It combines automated news collection, multilingual transformer models, human-validated evaluation, and time-series analysis in a reproducible research pipeline.

> **Research question:** Does the sentiment of Turkish financial news on day *t* predict the direction of BIST 100 on day *t+1*?

The implementation was developed with AI-assisted coding tools, while the research design, data collection, validation methodology, evaluation framework, and experimental decisions were designed and reviewed by me. The repository documents the reasoning behind important choices so that the work can be inspected and reproduced.

> **Current scope (late June 2026):** The corpus contains ~1,200 headlines collected since March. gpt-5-mini reaches **~83% agreement with held-out human labels**, and the relevance filter reaches **~91%**. The predictive analysis remains preliminary: 30 reliable overlapping news-and-market days are required, and the current dataset has 22. These validation figures measure agreement with human annotations, not predictive power.

![Sentiment vs BIST 100](docs/sample_output.png)

*Top: BIST 100 closing price. Middle: daily sentiment (green = bullish, red = bearish; hatched = thin-data days). Bottom: lead–lag scatter and rolling correlation — watermarked PRELIMINARY until there is enough data to mean anything.*

---

## The research question, and why it's hard

The hypothesis sounds simple: positive news today → market up tomorrow. It is genuinely difficult to test honestly, for reasons that are themselves the interesting part:

- **Markets are roughly efficient.** By the time news is public, prices may already reflect it. Finding *un-priced* information is the whole game.
- **Causality runs both ways.** Sentiment may *react* to prices rather than lead them. A model that scores headlines beautifully can still just be measuring yesterday's move.
- **Daily frequency is noisy.** With one data point per trading day, any real effect is buried under everything else that moves a market.

The null hypothesis is **"no signal."** The pipeline is designed to test it fairly, with safeguards against overfitting and misleading alignment.

---

## What I decided, and why

**1. Benchmark before you believe a model.** The first scorer (XLM-RoBERTa, a Twitter-trained multilingual model) reported 76.8% accuracy — but that number was measured on the very labels used to tune it. In-sample numbers flatter you. I required a *held-out* benchmark before trusting any scorer, then ran a bake-off — XLM-RoBERTa vs Google's Gemini vs OpenAI's gpt-5-mini — on human-labeled headlines the models had never seen. gpt-5-mini won at **83% held-out** and became the production scorer. The rule "no accuracy claim without a held-out number" is the single most important habit in this repo.

**2. Grade relevance; never delete data.** To cut noise (celebrity, sports, lottery stories that slip through the filter), the AI's first implementation simply *deleted* headlines it judged irrelevant. I overruled it: an **unvalidated judgment may down-weight data, but must never destroy it.** We rebuilt it as a 0–1 relevance grade that shrinks a headline's weight in the daily average toward zero — reversible, auditable, and tunable from one config value. It was later validated at **91% agreement** with my own keep/drop calls. (The 60 headlines the first version had deleted were restored from a backup.)

**3. A "mood" is more than an average.** The daily score is *confidence-weighted* — a decisive headline counts more than a wishy-washy one — but with a floor, so a single loud headline can't hijack an otherwise-quiet day. News is also weighted by time of day (pre-market headlines set the tone; post-close ones can't move that day's price).

**4. Align news to when the market can actually react.** A headline published after the close belongs to the *next* trading session, not the calendar date it was printed. An early version ignored this and silently mismatched ~500 of 750 headlines. The fix — a `signal_date` that rolls post-close and weekend news forward to the next session — is small, but it's the difference between testing the real hypothesis and testing noise.

**5. Refuse to over-interpret.** Signal statistics stay hidden behind a 30-day "reliable data" gate; the aggregation weights are **frozen** until there's enough data to tune them honestly; and every accuracy figure is scoped to exactly what it measures. With ~25 data points and a dozen plausible metrics, you can *always* find something that looks significant — so the project pre-commits to not going looking.

**6. Trust the source, not just the words.** The project is mid-migration from treating the *headline* as the unit of analysis to treating the *event* as the unit, with sources tiered by quality — official **KAP** company disclosures (Tier A) ranked above general-press RSS (Tier C). The bet, borrowed from the market-microstructure literature, is that structured disclosures carry the signal that lifestyle-heavy news drowns out.

**7. Don't run a daily job on a laptop.** It began on Windows Task Scheduler and kept dying whenever the machine slept. It now runs itself in the cloud (GitHub Actions), with its database living on a dedicated git branch — no personal hardware in the loop.

---

## Quality checks and corrections

These are real mistakes the process surfaced before they could affect a conclusion.

- **The signal was computed on the wrong days.** "Next-day return" was calculated *after* the data was filtered, so ~40% of the (sentiment, next-day-return) pairs were actually **2–15 days apart** — quietly corrupting every correlation. Caught in a code review; fixed by computing returns on the full, gap-free trading-day series *before* matching them to sentiment.
- **One news source was triple-counted.** Headlines without a URL slipped past the database's duplicate check on every run (SQLite treats every `NULL` as unique), so one outlet's stories kept inflating the daily average. Caught by the de-duplication audit; fixed with content-based dedup.
- **The AI deleted data; I caught it.** (See decision #2.) A backup and a skeptical human turned a data-loss bug into a better design.
- **A crash that was actually a good sign.** A full re-scoring run died on an API error — because the model's *version tag* was accidentally being sent as the model *name*. It failed **loudly and immediately** instead of silently mis-scoring 1,000 headlines. Separating the request field from the provenance field fixed it. (Failing loud beats failing quiet.)
- **My own labels drifted.** Labeling headlines weeks apart, my share of "neutral" calls jumped from 26% to 59% — hard proof that even the *ground truth* isn't perfectly stable, and that an 83% model must be read against the human ceiling. This led to tooling that measures my own self-consistency, so I know whether the model is the bottleneck or I am.
- **Tests with an expiry date.** Two tests used hard-coded dates that quietly aged out of the analysis window — they would have started failing on their own as the calendar advanced. Fixed to use dates relative to "today."

The recurring theme: **backups, loud failures, held-out validation, and an audit layer** are what let a one-person project move fast without lying to itself.

---

## Current status

- **~1,200 headlines** collected daily since March 2026; sentiment scored at **~83% held-out** agreement, relevance at **~91%**.
- The research question needs 30 reliable overlapping days; the current analysis has **22** (~early July). A genuine out-of-sample answer (walk-forward testing, net of transaction costs) requires ~60 days — roughly mid-August.
- A null result would still be informative: Turkish daily news sentiment may not predict next-day BIST returns. The evaluation is designed to avoid tuning parameters until a spurious signal appears.

The central contribution is the evaluation discipline around the model: validation, alignment, and safeguards against premature conclusions.

---

## What the news itself looks like

The predictive question needs more data, but the ~1,300-headline corpus already tells a story *now* — pure description, no overfitting risk (`analyze_corpus.py`):

![corpus overview](docs/corpus_overview.png)

- **Currency/lira news skews most bearish** (average −0.19) while Turkish-economy news skews most bullish (+0.14) — consistent with a chronically depreciating lira and upbeat official macro framing.
- **Outlets differ systematically, not randomly.** Pro-government *Sabah* is both the most on-topic and the most bullish; opposition *Sözcü* is the most bearish — a measurable media-slant effect.
- An **emerging-markets index, oil, and USD/TRY** are now collected daily alongside BIST, so any eventual signal can be tested *net of* global moves — rather than crediting "all of EM rose today" to Turkish news.

### Headline finding — a political slant in financial sentiment

Chasing something *non-obvious*, the strongest result came not from the market series (too little data) but from the news ecosystem itself, where there are thousands of observations:

![media polarization](docs/polarization.png)

Turkish financial-news sentiment carries a **large, highly significant political slant** — pro-government/state outlets average **+0.11**, opposition **−0.09**, a gap of **+0.20** (*p ≈ 4×10⁻²⁴*, Cohen's d = 0.74). It replicates on a second independent model (Gemini), so it's in the text, not one scorer's artifact. And it's *political*, not just tonal: the divergence is concentrated in domestic-economic coverage (**+0.21** on macro) and nearly vanishes on externally-set topics (**+0.04** on energy/commodities) — outlets split on how the Turkish economy is doing but agree about oil prices. *(How much is spin vs which stories each camp covers — a genuine open question I tested and only partly resolved — is in the write-up.)* Full detail and caveats: [docs/polarization_findings.md](docs/polarization_findings.md).

---

## Run it

```bash
git clone https://github.com/amirremirr/Turkish-stock-market-sentiment-analysis.git
cd Turkish-stock-market-sentiment-analysis

pip install -r requirements.txt          # full set; requirements-cloud.txt is the slim, no-torch set
echo "OPENAI_API_KEY=sk-..." > .env       # the LLM scorer; or set SENTIMENT_BACKEND="xlmr" for the offline model

run.bat run                               # scrape -> score -> aggregate -> prices -> plot
```

The pipeline also runs **unattended every weekday in the cloud** (GitHub Actions); its SQLite database lives on a `data` branch, and `pull-cloud-db.bat` fetches the latest to inspect locally.

**Useful commands** (`run.bat <cmd>` or `python main.py <cmd>`):

| Command | What it does |
|---|---|
| `run` | Full pipeline end to end |
| `status` / `dashboard` | DB statistics / self-contained HTML dashboard |
| `score` · `aggregate` · `relabel` | Re-score, recompute daily aggregates, relabel from stored probabilities |
| `recategorize --llm` | Re-classify category + relevance with the LLM |
| `export-labels --n 300 [--uncertain]` | Export headlines for human labeling (random, or active-learning) |
| `validate-labels <csv>` | Accuracy, confusion matrix, holdout split |
| `kap-ingest --dry-run` | KAP Tier-A disclosure ingestion (migration, dev-validated) |
| `run.bat test` | 111-test suite (no GPU or model download) |

Quality audit: `python evaluate.py` runs a read-only 6-layer report (L0 system health → L5 signal statistics, the last gated until 30 reliable days).

---

## Architecture

```mermaid
flowchart LR
    A[Turkish financial news] --> B[RSS scraper]
    B --> C[(SQLite database)]
    C --> D[Sentiment model]
    D --> E[Daily aggregation]
    E --> F[Market-session alignment]
    G[BIST 100 market data] --> F
    F --> H[Evaluation and reporting]
```

The pipeline preserves raw headlines and intermediate metadata, making scores, weights, market-session assignment, and evaluation outputs auditable.

```
config.py          Every tunable parameter (feeds, keywords, thresholds, weights)
scraper.py         RSS fetch, relevance filter, keyword classifier
sentiment_llm.py   gpt-5-mini scorer — sentiment + category + graded relevance
sentiment.py       XLM-RoBERTa offline fallback backend
pipeline.py        Step orchestration + the aggregation math
trading_calendar.py  signal_date: news → first session that can react
events_bridge.py   Headline → event store (event-centric migration)
kap_ingest.py      KAP Tier-A disclosure ingestion (MKK API)
database.py        SQLite layer (schema, migrations, queries)
visualize.py       3-panel matplotlib figure        dashboard.py   HTML dashboard
evaluate.py        Read-only 6-layer quality audit
benchmark_llm.py   Held-out model bake-off (OpenAI / Gemini)
label_audit.py     Adjudication + intra-annotator consistency tools
.github/workflows/daily.yml   Cloud daily run

METHODOLOGY.md  every design decision    MIGRATION.md  event-pipeline migration plan
LABELING.md     the labeling rubric      ROADMAP.md    what's done / next / frozen
CLOUD.md        the cloud setup          DOCUMENTATION.md  full technical reference
```

**Tech stack:** `Python 3.10` · `SQLite` · `OpenAI API (gpt-5-mini)` · `XLM-RoBERTa (offline fallback)` · `pandas` · `yfinance` · `matplotlib` · `pytest` · `GitHub Actions`

---

## Future Work

- Expand annotation with multiple independent human annotators and formal inter-annotator agreement.
- Move from headline-level scores to event-level sentiment, reducing duplicate coverage of the same news event.
- Fine-tune and benchmark a Turkish finance-specific sentiment model.
- Use time-aware validation and walk-forward evaluation as the market-history sample grows.
- Compare the current aggregate signal with predictive models, including transformer-based approaches, while accounting for transaction costs and global-market controls.

---

## License

MIT — see [LICENSE](LICENSE).
