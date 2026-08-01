# Migration: Daily-Sentiment Pipeline to Event-Centric Research System

**Status reviewed:** 2026-08-01

This is a phased migration, not a rewrite. The headline path, CLI, and legacy tables remain available while event-centric work is developed behind explicit flags. Cutover is conditional on pre-specified out-of-sample evidence; a documented null result is an acceptable endpoint. Pre-migration snapshot: git commit `acf47c4`.

The target reframing is:

- unit of analysis: event rather than headline;
- target: abnormal return rather than raw BIST direction;
- validation: chronological walk-forward evaluation rather than a post-hoc audit alone.

These remain migration goals, not descriptions of the current production aggregation unit.

## Phase status

| Phase | Theme | Status |
|---|---|---|
| 0 | Hardening: experiment registry, feature flags, rolling backups | completed 2026-06-12 |
| 1 | Temporal primitives and predictive cutover | completed end to end 2026-08-01: versioned session assignment, four variants, and all market-linked consumers migrated |
| 2 | Event schema and dual-write bridge (`events`, `event_entities`, `events_bridge.py`, `migrate-events`) | schema/bridge completed 2026-06-12; headline aggregation remains primary |
| 3 | KAP Tier-A ingestion (`kap_ingest.py` via MKK API Portal) | built and dry-run validated 2026-06-13; awaiting production access, `KAP_ENABLED=False` |
| 4 | Structured extraction: direction, magnitude, event type, entities | pending |
| 5 | Session windows W1/W2/W3 (`session_features`) | pending |
| 6 | Entity linking and free-float cap weights | pending |
| 7 | Feature store and abnormal-return target | pending; context series already collected |
| 8 | Walk-forward evaluator | pending; observation count alone is not a reliability threshold |
| 9 | Event-path cutover or documented null result | pending |

The Stage 2/3 labels in [ROADMAP.md](ROADMAP.md) refer to the 2026-08 credibility-hardening program, not to the numbered event-migration phases above.

## Current compatibility architecture

### Headline path

- `headlines` remains the canonical scoring unit.
- `raw_headline_observations` preserves source-distinct fetched observations before canonical URL/title deduplication.
- `headline_exclusions` preserves versioned active/restored filtering history.
- scoring uses explicit pending/scored/retry/failed states and omission-aware retries.
- `daily_signal_variants.simple_mean` is the canonical market-linked headline baseline.

### Event bridge

- `EVENTS_DUAL_WRITE=True` runs `events_bridge.sync()` after scoring.
- headline bridge semantics remain `direction = sentiment_score` with source-tier credibility defaults.
- `events.headline_id` is unique when present; non-headline events can use a unique `external_id`.
- event and entity tables are additive. They do not replace or delete headline records.
- an event-bridge exception is logged without corrupting the completed headline-scoring path.

### Legacy aggregates

The following tables remain intentionally available:

- `daily_sentiment`: publication-date full-weighted compatibility series;
- `daily_sentiment_by_signal`: session-date full-weighted compatibility shape;
- `category_daily_sentiment`: publication-date category description;
- `category_sentiment_by_signal`: session-date simple category description.

Predictive code no longer uses the calendar aggregate as a fallback. `daily_signal_variants` is the session-keyed source for market-linked work, with `simple_mean` primary and three weighted sensitivities.

## Additive schema changes through 2026-08-01

The migration discipline remains add-only. `database.init_db()` uses idempotent table creation and `ALTER TABLE` for missing columns; no current migration drops or rewrites a table.

### Processing and provenance columns

Added to `headlines`:

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

The one-time state backfill classifies complete legacy score rows as `scored`, empty rows as `pending`, and partial rows as `retry_pending`; it does not alter the stored score values. Component provenance is inferred conservatively from the stored model name. Session metadata is versioned and refreshed explicitly during aggregation.

### New tables

```text
raw_headline_observations
headline_exclusions
daily_signal_variants
category_sentiment_by_signal
```

Existing event, experiment, market, external-series, and legacy aggregate tables remain in place.

### Pipeline-run audit additions

`pipeline_runs` now has scrape, scoring, aggregation, market-data, and audit status columns plus structured warnings/errors. New full runs use final `success`, `degraded`, or `failed`; old `ok`, `error`, `recovered`, and `crashed` rows remain readable through a canonical-status mapping.

## Temporal convention completed in Phase 1

The current temporal primitive is a versioned `SessionAssignment` containing:

- Istanbul-normalized publication timestamp when recoverable;
- timing bucket (`pre_open`, `during_session`, `post_close`, `weekend_or_holiday`, `unknown`);
- first BIST session able to react.

Pre-open and in-session publications map to the same session. Post-close, non-trading-day, and unknown-time publications roll forward. The calendar includes configured full closures and half-day closes. Unknown time is conservative to avoid look-ahead.

All market-linked consumers now use the session baseline. Next-session targets are constructed on the complete ordered price sequence before joining sparse signals. The old calendar/session dual display is retained only where it is explicitly labeled descriptive or historical.

## Signal-specification convention completed in Phase 1

`daily_signal_variants` stores four pre-specified headline aggregates for each session:

1. unweighted `simple_mean` baseline;
2. `relevance_weighted`;
3. `intensity_relevance_weighted`;
4. `full_weighted`, retaining historical time weighting as sensitivity.

The sensitivity command compares them without selecting a winner on the evaluation sample:

```bash
python -m analysis.prediction.sensitivity --db finance_sentiment.db --output outputs/signal_sensitivity.json
```

Any eventual event feature must beat the simple headline baseline and relevant controls under the same chronological evaluation protocol. Complexity is not evidence.

## Phase gates

| End of | Gate | If no |
|---|---|---|
| Phase 3 | KAP ingest at least one usable event per trading day over two weeks | remain RSS-heavy and delay cap weighting |
| Phase 4 | Direction MAE at most 0.35 versus held-out human labels | retain `sentiment_score` as the direction bridge |
| Phase 8 | News features beat momentum/FX baselines out of sample, net of 10 bps costs, across at least 20 windows | Phase 9 null-result path: stop tuning and keep archiving |

These gates are decision rules, not current achievements.

## Standing rules during migration

1. Migrations are additive: do not drop a column/table or erase raw/exclusion history during development.
2. Every bulk re-score gets explicit experiment/model/prompt provenance; do not silently mix scorer versions as one measurement series.
3. `simple_mean` remains the primary headline baseline. Weighted variants are sensitivities and are not tuned on the sample used to assess them.
4. Event features do not replace the headline baseline until they pass a pre-specified chronological comparison.
5. The current corpus still includes general-press RSS. A proposed Tier A/B event core must define and test its inclusion rule before cutover.
6. The migration may create/backfill schema metadata, but it does not silently re-score observations, rebuild aggregates, or regenerate dated research artifacts.
7. Explicit reruns that change a research result must record the reason, affected sample/specification, and before/after interpretation.

## Reproducibility and snapshots

The checked-in local database extends through 2026-07-07. The latest known `origin/data` snapshot is dated 2026-07-28. Counts are omitted because the cloud snapshot changes.

`run_scheduled.py` keeps rolling local backups; GitHub Actions persists the active cloud snapshot on the data branch. Files under `docs/*_findings.md` are dated research outputs, not live views of the latest migrated database.

## Parallel workstreams

- Canonical 300-label rubric, including `human_relevant`: completed; future labeling must not silently merge the older 198-label convention.
- Daily collection: automated and continuing.
- KAP production access: pending.
- BIST ticker/entity master: supports Phase 6.
- EM, oil, FX context: collected; supports Phase 7.
- Dependence-aware polarization inference and the API-key-free demo are complete; verified shared-event coverage still depends on later event-resolution phases.
