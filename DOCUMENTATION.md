# BIST 100 Turkish News Research Pipeline - Technical Reference

**Updated:** 2026-08-01

**Runtime:** Python 3.10, SQLite, OpenAI LLM production scorer, XLM-RoBERTa fallback

**Status:** active research; prediction is exploratory

This reference describes current code behavior. The former May 2026 manual documented the original XLM-RoBERTa/Windows-only system and is not mixed into the active specification.

## 1. Scope and claims

The code supports three distinct research areas:

1. scorer agreement with a written sentiment/relevance rubric;
2. descriptive media framing and polarization analysis;
3. exploratory analysis of subsequent market returns.

The pipeline is not a validated trading system and has not demonstrated alpha. Classification agreement is not predictive evidence.

## 2. Current status

| Component | Implemented state |
|---|---|
| Scoring | `gpt-5-mini`, prompt `p3`; XLM-RoBERTa fallback available; partial results are omission-aware |
| Observation state | `pending`, `scored`, `retry_pending`, `failed`, with attempt/error metadata |
| Preservation | Source-distinct raw-observation audit plus reversible, versioned exclusion history |
| Aggregation | Session-aligned `simple_mean` primary baseline plus three sensitivity variants |
| Timing | Istanbul-normalized timestamp, explicit timing bucket, versioned first-reactable-session assignment |
| Prediction | All market-linked consumers use `daily_signal_variants.simple_mean`; no calendar-table fallback |
| Run health | `success`, `degraded`, or `failed` final outcome plus five component states and structured diagnostics |
| Polarization | Dependence-aware observational report with separate selection/framing mechanisms and guarded event identity |
| Public demo | Deterministic committed sample, cached sentiment, audit JSON, table, and chart; no private services |
| Automation | GitHub Actions weekday run; database snapshot on the `data` branch |
| Data snapshots | Checked-in local DB through 2026-07-07; latest known `origin/data` snapshot dated 2026-07-28 |
| Method version | Credibility methodology updated 2026-08-01; scorer prompt `p3` last changed 2026-06-13 |
| Events | Headline aggregation primary, event dual-write enabled, KAP production ingest disabled |

Counts are deliberately omitted because the cloud snapshot changes independently of the checked-in local database.

## 3. End-to-end data flow

```text
RSS / HTML item within lookback
    -> source-scoped raw observation + filter metadata
    -> canonical headline + optional active exclusion
    -> pending / retry-aware scoring state
    -> complete scored row or explicit failed state
    -> versioned Istanbul session assignment
    -> simple baseline + three session sensitivity variants
    -> complete-price-series next-session target
    -> charts, dashboard, audit, and sensitivity report
```

### 3.1 Scraping and raw observations

`pipeline.scrape_step()` calls `RSSFeedScraper.scrape_all()` for `config.RSS_FEEDS`; an Investing.com HTML scraper is the empty-RSS fallback. Each source records counts for fetched, returned, included, excluded, too-old, and same-source duplicate items, or a failure message.

The scraper does not discard an otherwise age-eligible item merely because it fails the keyword rules. It returns the item with flat exclusion metadata:

```text
is_excluded
exclusion_reason
exclusion_rule
exclusion_version
```

In-run deduplication is source scoped. `database.insert_headlines()` first records stable source-distinct items in `raw_headline_observations`, including source payload JSON, and only then applies canonical deduplication. A native source item ID is the preferred observation key, followed by source-scoped URL, then source/title/date metadata. Replaying the same fetch is idempotent.

The canonical `headlines.url` constraint remains globally unique for backward compatibility. URL-less canonical rows are deduplicated by source, normalized title prefix, and publication date. Consequently, two outlets with the same URL can share a canonical row, but their two source observations remain auditable and contribute to source-breadth metadata.

### 3.2 Exclusions and restoration

`headline_exclusions` is append-oriented history. A partial unique index permits at most one active exclusion per canonical headline; restoring it writes `restored_at`. Scrape keyword decisions, `clean`, and low-LLM-relevance decisions all use this mechanism.

`clean [--dry-run]` now previews or stores reversible keyword-rule exclusions. It never deletes raw or canonical rows. `restore-exclusion ID` restores one active decision. Both commands rebuild derived aggregates only when a decision changed.

Permanent deletion is intentionally absent from the ordinary CLI. The low-level `database.delete_headlines(ids, confirm=True)` / `permanently_delete_headlines(...)` API requires the literal confirmation flag. Foreign-key enforcement is enabled on every database connection; a purged canonical row nulls the raw observation link rather than deleting the raw observation itself.

### 3.3 Scoring

`pipeline.score_step()` selects only non-excluded `pending` and `retry_pending` rows. Rows whose attempt count already meets the configured cap are not retried.

The LLM scorer exposes keyed partial results. Only a unique, in-range, valid returned ID is accepted. Omitted, duplicated, out-of-range, or invalidated items remain absent instead of receiving a placeholder. The pipeline:

1. writes complete returned items as `scored`;
2. increments attempt/error metadata on missing items;
3. retries only that missing subset;
4. marks exhausted items `failed` after `LLM_SCORING_MAX_ATTEMPTS`;
5. preserves NULL sentiment fields for items with no valid result.

An explicitly returned neutral result is valid and stored. Malformed structured output or a transport exception is recorded as a failed attempt for the active candidates and then raised. HTTP request retry count and exponential backoff are separately configurable.

For LLM rows, `p_positive`, `p_neutral`, and `p_negative` are synthetic compatibility components. Only XLM-RoBERTa rows contain softmax probabilities. `score_components_kind` records `synthetic_compatibility`, `softmax_probability`, a caller-supplied kind, or conservative legacy provenance.

### 3.4 Aggregation

`pipeline.aggregate_step()` backfills missing categories, refreshes outdated session assignments, reconciles LLM relevance exclusions, selects complete eligible `scored` rows, clears all derived aggregate tables, and rebuilds them in one explicit run.

The canonical table is `daily_signal_variants`:

| Column | Definition |
|---|---|
| `simple_mean` | unweighted arithmetic mean; primary market-linked baseline |
| `relevance_weighted` | relevance-only weighted mean |
| `intensity_relevance_weighted` | relevance times `max(abs(score), floor)` |
| `full_weighted` | relevance x intensity x source x time x category; current neutral source/category defaults reproduce the historical relevance/intensity/time formula |
| counts and shares | headline, positive, negative, neutral, unclassified, source, and event counts; label shares |
| audit statistics | population sentiment dispersion and each weighted denominator |

The pure formula lives in `aggregation/signals.py`. Missing/non-finite scores are not observations; an explicit `0.0` score is. A zero weighted denominator returns NULL rather than falling back to the baseline.

Legacy/description tables are still rebuilt for compatibility:

- `daily_sentiment`: publication-date full-weighted aggregate;
- `daily_sentiment_by_signal`: session-date full-weighted legacy shape;
- `category_daily_sentiment`: publication-date full-weighted category description;
- `category_sentiment_by_signal`: session-date unweighted category description.

They are clearly separate from the canonical predictive table. `database.init_db()` applies schema migrations but does not invoke aggregation. Existing aggregates change only through an explicit aggregate-calling command.

### 3.5 Market data and outputs

The pipeline downloads BIST 100 OHLCV from Yahoo Finance, optional USD/TRY from Alpha Vantage, and contextual market factors from Yahoo Finance. If the primary market fetch fails, cached data no older than `MARKET_DATA_STALE_AFTER_DAYS` produces a degraded outcome; absent or older data produces a failed outcome. Optional factor failures degrade rather than erase otherwise usable output.

`visualize.py` writes the PNG chart, `dashboard.py` generates a standalone HTML view, and `evaluate.py` prints a read-only quality audit. A checked-in HTML or figure is a dated generated artifact and can lag current code/data.

## 4. Database schema and additive migration

`database.init_db()` runs idempotent `CREATE TABLE IF NOT EXISTS` statements, adds missing columns with `ALTER TABLE`, creates dependent indexes, enables foreign keys, and uses WAL mode. It never drops a table or column.

### 4.1 Core and derived tables

| Table | Role |
|---|---|
| `headlines` | canonical text, publication/session metadata, score fields, relevance, provenance, and processing state |
| `raw_headline_observations` | source-distinct fetch audit before canonical deduplication |
| `headline_exclusions` | versioned active/restored filtering history |
| `daily_signal_variants` | canonical session baseline and three sensitivity variants |
| `category_sentiment_by_signal` | session-aligned simple category description |
| `daily_sentiment` | legacy publication-date full-weighted aggregate |
| `daily_sentiment_by_signal` | legacy session-date full-weighted shape |
| `category_daily_sentiment` | legacy publication-date category aggregate |
| `bist100_prices` | BIST 100 OHLCV and return fields |
| `usdtry_rates` | optional Alpha Vantage USD/TRY series |
| `market_factors` | EEM, Brent, and Yahoo USD/TRY context series |

### 4.2 Research and audit tables

| Table | Role |
|---|---|
| `pipeline_runs` | final/component outcomes, counts, provenance, warnings, and errors |
| `experiments` | experiment/configuration provenance |
| `events` | event migration store populated mainly by headline dual-write |
| `event_entities` | event-to-entity links |
| `kv_state` | ingestion cursors such as KAP state |
| `external_series` | Google Trends/GDELT and similar dated external series |

### 4.3 Added headline columns

The additive migration includes:

```text
processing_status
scoring_attempts
last_scoring_attempt_at
scoring_last_error
score_components_kind
published_timestamp
timing_bucket
session_rule_version
published_hour
signal_date
relevance
```

When `processing_status` is first added to a legacy database, complete historical score rows are classified `scored`, fully empty rows `pending`, and partial rows `retry_pending`. That classification runs only when the column is introduced, so later explicit state transitions are not overwritten. Component provenance is inferred conservatively from the stored model name. Session assignment is versioned and refreshed by aggregation; old rows without a recoverable time remain conservatively assigned as unknown-time observations.

### 4.4 Added pipeline-run columns

```text
scrape_status
scoring_status
aggregation_status
market_data_status
audit_status
warnings_json
errors_json
experiment_id
```

New full runs finish as `success`, `degraded`, or `failed`. Component values also permit `pending`, `running`, and `skipped`. Legacy `ok`, `error`, `recovered`, and `crashed` rows remain readable and expose a mapped `canonical_status`.

Schema migration does not re-score data, rerun research scripts, or regenerate dated output. Those mutations remain explicit operations.

## 5. Session assignment

`trading_calendar.assign_trading_session()` returns a normalized Istanbul timestamp, timing bucket, and first-reactable `signal_date`. A naive time is assumed to be Istanbul-local; an aware timestamp is converted.

| Condition | Bucket | Assigned session |
|---|---|---|
| before 10:00 on trading day | `pre_open` | same day |
| 10:00 through scheduled close | `during_session` | same day |
| after scheduled close | `post_close` | next session |
| weekend or full holiday | `weekend_or_holiday` | next available session |
| time missing/invalid on trading day | `unknown` | next session |

Regular close is 18:10; configured 2025-2026 half-days close at 13:00. The rule version is stored on each canonical headline so calendar corrections can be backfilled explicitly.

### Predictive consumers

| Consumer | Current convention |
|---|---|
| `visualize.py` | `daily_signal_variants.simple_mean`; price lead formed before join |
| `dashboard.py` | `daily_signal_variants.simple_mean` |
| `explore_signal.py` | session variants plus targets formed on complete price/factor series |
| `analyze_external.py` | session variants; no calendar fallback; leads formed before merge |
| `evaluate.py` | session baseline is primary; legacy calendar/category material labeled descriptive |
| `analysis.prediction.sensitivity` | all four session variants, no preferred specification |

## 6. Run-status contract

`pipeline.run_all()` persists a run even when a component raises. It records the active component as failed, skips components that were never attempted, saves structured issues, and re-raises the exception to the caller. A successful completion derives its final status from the component states:

- any failed component -> `failed`;
- otherwise any degraded component -> `degraded`;
- otherwise -> `success`.

The post-processing audit fails on a `scored` row with incomplete required output and degrades when pending/retry/failed observations remain. The CLI `status` command exposes final status, component states, processing-state counts, active exclusions, and structured warnings/errors.

## 7. CLI reference

Commands run as `python main.py <command>` or, on Windows, `run.bat <command>`.

| Command | Current behavior |
|---|---|
| `run` | scrape, score, aggregate, prices/context, plot, and persist component outcomes |
| `scrape` | fetch observations, including exclusion metadata, and insert raw/canonical records |
| `score` | process eligible pending/retry rows with missing-only retries |
| `aggregate` | explicitly rebuild all legacy and canonical derived signal tables |
| `recategorize [--llm]` | refresh topics; LLM mode retries missing analyses, updates relevance/exclusions, and aggregates |
| `relabel` | derive labels from stored backend-specific compatibility components |
| `prices` | fetch BIST 100 prices with cache-freshness outcome logic |
| `fx-rates` | fetch optional Alpha Vantage USD/TRY data |
| `fetch-factors` | fetch EEM, Brent, and Yahoo USD/TRY context |
| `plot` | regenerate the current session-baseline chart |
| `status` | print state counts, exclusions, and latest final/component outcomes |
| `dashboard` | generate and open the standalone dashboard |
| `clean [--dry-run]` | preview/store reversible off-topic exclusions; no deletion |
| `restore-exclusion ID` | restore one active exclusion and aggregate |
| `migrate-events` | idempotently sync scored headlines to events |
| `kap-ingest --dry-run` | validate KAP integration without storing sample-era data |
| `export-labels` | export a human-labeling CSV |
| `validate-labels` | report agreement with the human-label rubric |

The sensitivity command is a package entry point rather than a `main.py` subcommand:

```bash
python -m analysis.prediction.sensitivity \
  --db finance_sentiment.db \
  --output outputs/signal_sensitivity.json
```

It reports variant correlations, directional agreement, distributions, alignment audit rows, and next-session exploratory metrics as strict JSON.

Two Stage 4 package commands are also independent of `main.py`:

```bash
python -m analysis.polarization.inference --db finance_sentiment.db
python -m scripts.demo --output-dir demo_output
```

The first is read-only unless `--json-output` is supplied. The second is fully
offline and writes a result table, audit JSON, and chart from committed samples.
See [docs/POLARIZATION_METHODS.md](docs/POLARIZATION_METHODS.md) for estimands,
dependence diagnostics, event-ID guards, and interpretation limits.

## 8. Automation

The primary automation is `.github/workflows/daily.yml`:

1. runs Monday-Friday at 06:30 UTC;
2. restores `finance_sentiment.db` from `origin/data`;
3. installs the no-Torch cloud dependencies;
4. runs `python main.py run --no-show` with the OpenAI key;
5. publishes an updated current chart when it changes;
6. force-updates an orphan data branch with the current DB and chart snapshot.

`run_scheduled.py`, Task Scheduler XML, and registration scripts remain a legacy local route with rolling backups. They are not the cloud workflow.

## 9. Evaluation and test communication

`evaluate.py` reports system health, collection quality, scoring/provenance, aggregate sensitivity, market coverage, and exploratory session-return relationships. It labels legacy calendar/category output descriptive, treats `simple_mean` as primary, and treats the 30-overlap threshold as a reporting gate only.

Test credibility is reported by risk rather than by a raw count. The exact regression modules, behaviors covered, and residual gaps are maintained in [docs/TEST_RISK_MAP.md](docs/TEST_RISK_MAP.md).

## 10. Research artifacts

Files under `docs/*_findings.md`, generated HTML, JSON reports, and figures are dated analysis snapshots. Their sample descriptions define their scope. Neither `init_db()` nor an additive migration silently changes them. If an explicit rerun changes a result, the updated artifact must document the reason and changed interpretation.

See [METHODOLOGY.md](METHODOLOGY.md) for research rationale, [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) for the incremental package/wrapper layout, and [AI_ASSISTANCE.md](AI_ASSISTANCE.md) for the AI-use disclosure.
