# Stock-level study: implementation audit (Phase 0)

*Audited 2026-10-11 against the code at commit `f2948f9`, the production
database snapshot of 2026-10-08, live calls to the MKK API Portal dev gateway,
and live Yahoo Finance requests. Claims in the existing documentation were
checked against code and data; where they disagree, the code and data win and
the disagreement is recorded here.*

## 1. What exists and can be reused

| Component | File | Verdict |
|---|---|---|
| Session assignment (pre-open / during / post-close / weekend) | `trading_calendar.py` | **Reuse**, with a limit: see 3.1 |
| Proven timing convention (`signal_date` = first session able to react) | `research/timing.py`, `docs/TIMING.md` | **Reuse the rule**, not the module: it is written for headline groups |
| Daily-bar completeness rules | `price_bars.py` | **Reuse the idea** (provisional vs settled); the module is index-specific |
| KAP API client (auth, throttle) | `kap_ingest.py` | **Reuse auth and endpoint knowledge**; the ingester itself is unsuitable, see 3.2 |
| Sentiment scorer (`gpt-5-mini`, prompt `p3`) | `sentiment_llm.py` | **Reuse scores already stored**; not validated for KAP disclosure text |
| Headline corpus with timestamps and scores | `finance_sentiment.db` (`headlines`, `raw_headline_observations`) | **Read-only input** |
| Headline entity dictionary (19 issuers) | `events/entities.py` | **Too small**; a starting alias list only |
| Frozen-protocol and append-only machinery | `research/protocol.py`, `database.py` triggers | **Reuse the pattern** in a separate database |
| Cluster bootstrap, walk-forward folds | `research/walkforward.py` | **Reuse the pattern**; stock-level needs two-dimensional dependence handling |

## 2. What is missing entirely

- Any stock-level price data. The project stores one series: the BIST 100.
- Any issuer master, ticker history, sector, market capitalisation, or
  corporate-action table.
- Any news-to-ticker mapping beyond 19 hard-coded large caps.
- Any event-study estimator (expected returns, abnormal returns, CAR, BHAR).
- Any stock-level hypothesis, protocol, or report.
- English-language news. All 11 sources are Turkish.
- Intraday prices.

`docs/STOCK_LEVEL_STUDY_PLAN.md` describes the intent and is accurate about
this: it says no code exists.

## 3. Findings that change the design

### 3.1 The trading calendar only knows 2025 and 2026

`config.BIST_HOLIDAYS` lists holidays for 2025–2026 only, and
`trading_calendar.is_trading_day` treats every other weekday as a session. The
only real KAP data available is from 2023 (see 3.3). Used as-is, the calendar
would assign 2023 holiday publications to sessions that never traded.

**Change:** the stock study derives its session calendar from the benchmark's
own price bars (a date is a session if the index traded), and treats a
publication after 12:30 on the weekday before a multi-day closure as
after-close, because half-day closes for 2023 are not recorded anywhere in the
repository. That is conservative: it can delay an event's day 0, never advance
it.

### 3.2 The existing KAP ingester would lose most tickers and the minute

`kap_ingest.py` takes tickers from `relatedStocks`, which the API leaves empty
for ordinary company disclosures (verified on live payloads); the sender's
codes are in `senderExchCodes`. It also keeps only the publication **hour**. A
disclosure at 18:05 and one at 18:55 are both "hour 18", but the close is at
18:10, so the hour cannot separate during-session from after-close.

**Change:** the stock study has its own ingester that stores the full timestamp
and both code lists. `kap_ingest.py` is left untouched; it is disabled in
production (`KAP_ENABLED = False`) and feeds the index pipeline's tables.

### 3.3 KAP data: a 2023 sample, one detail call per event

Measured on the dev gateway:

- Available index range ≈ 1,091,673 to 1,231,017: about 139,000 disclosures,
  ending 2023-12-31.
- `/disclosures` returns 50 rows per call **without a timestamp or subject**.
  Time, subject and body need `/disclosureDetail`, one call each.
- The plan allows 6 calls a minute. A census of material-event disclosures
  would take weeks.
- `/members` returns 1,037 members (918 with stock codes). It is a **current
  roster**, not a point-in-time history: companies delisted before the snapshot
  are absent, and tickers are today's.

**Change:** a pre-specified systematic sample of 120 listing blocks spread
evenly over the index range (`kap-block-sample-v1`), fixed before any price was
read. Production access (requested from MKK) would lift the rate problem only
partly; the sampling design stays valid either way.

### 3.4 Prices: available, with three known defects

Yahoo Finance serves daily OHLCV, dividends and splits for `.IS` tickers,
including small caps, back to 2000 for older listings. Verified live for
THYAO, SDTTR, DYOBY, ISCTR, FRIGO and XU100.

- **Survivorship:** delisted tickers are generally not served. The member
  roster has the same gap. Results are conditional on surviving to the
  snapshot, and are labelled so.
- **Adjustment opacity:** `Close` is split-adjusted retroactively; rights
  issues (bedelli) may not be reflected. Unexplained jumps must be detected,
  not assumed away.
- **No trading status.** A halted day and a missing day look the same.

### 3.5 News cannot support stock-level tests yet

- Scored headlines: 7,158. Only 97 event-group mentions resolve to a listed
  issuer, across 17 issuers.
- 3,727 scored headlines have a session before 2026-08-10; the rest fall
  inside the sealed index window (see 4).
- 3,693 of 7,158 scored headlines carry a full timestamp; the others have an
  hour or nothing.

Stock-specific sentiment baselines (H3) and coverage baselines (H4) need dense
per-stock news. This corpus does not have it.

### 3.6 No point-in-time market capitalisation

No free source of historical shares outstanding or free float for BIST was
found. Multiplying a historical price by today's share count is look-ahead and
is not done. Size-dependent hypotheses are blocked on real data until a
licensed or verifiable source exists; the provider interface is in place.

## 4. Protection of the sealed index test

`untouched_future_v1` forbids reading index outcomes on sessions from
2026-08-10 until its single result is recorded. Stock returns around news are
close to that target. So the stock module refuses to compute any outcome on or
after 2026-08-10 while `future_validation_results` is empty
(`stock_research/guards.py`). The 2023 KAP sample is unaffected.

The stock module also:

- writes only to `stock_research.db`, a separate file, and opens
  `finance_sentiment.db` read-only;
- never imports from, or is imported by, the daily pipeline;
- leaves `research/`, `docs/frozen/` and the index workflows unchanged.

## 5. Risks of modifying existing code, and the decision

| Risk | Decision |
|---|---|
| Extending `config.BIST_HOLIDAYS` backwards changes `TRADING_CALENDAR_RULE_VERSION` inputs to the index study | Not done; calendar derived from data inside the stock module |
| Fixing `kap_ingest.py` changes rows the index pipeline would write | Not done; separate ingester |
| Adding tables to `finance_sentiment.db` changes the published snapshot and the guard's markers | Not done; separate database |
| New dependencies | None added. Uses numpy, pandas, scipy, statsmodels, matplotlib, bs4, yfinance, all already required |

## 6. Feasibility by hypothesis

"Real" means historical observations. "Fixture" means the estimator is
implemented and tested on synthetic data with known answers, and reports
`data_insufficient` on real data.

| | Needs | Real data now | Status expected |
|---|---|---|---|
| H1 Small-cap underreaction | stock sentiment events, point-in-time market cap | neither | **Blocked** (market cap), **insufficient** (news) |
| H2 Overnight gap | after-close company news with sentiment, open and close | timestamps yes, sentiment no | **Insufficient**: KAP disclosures are unscored |
| H3 Sentiment surprise | dense per-stock sentiment history | no | **Insufficient** |
| H4 Abnormal coverage | per-stock article counts over time | no | **Insufficient** |
| H5 Limit-down reversal by news | limit days, complete news around them | prices yes; news is a 4% sample, so "no news" cannot be established | **Blocked** for the conditional claim; unconditional description possible |
| H6 Limit-up streak ending | streaks, news intensity | prices yes; news no | **Insufficient** for the news terms; baseline model runnable |
| H7 KAP category drift | categorised disclosures, prices | **yes** (block sample) | **Runnable** |
| H8 Insider buying | insider purchase disclosures, prices; size split needs market cap | **partly** (count unknown until ingested) | **Runnable if enough events**; size split blocked |
| H9 English vs Turkish lead-lag | English stream, intraday prices | neither | **Blocked** |

The honest expectation before running anything: two hypotheses can be tested
on real data, and seven will report why they cannot.

## 7. Planned changes

1. `stock_research/` package: store, calendar, providers, prices, KAP ingester,
   taxonomy, entity linker, event-study engine, inference, costs, nine
   hypothesis modules, registry, report, CLI.
2. `stock_research.db` (gitignored).
3. `tests/stock_research/`, including estimators checked against synthetic data
   with a known effect and with no effect.
4. `docs/stock_research/` (twelve documents).
5. One paragraph in the main README.

No existing file's behaviour changes.
