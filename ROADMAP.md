# Roadmap

**Rebased:** 2026-08-01

**Principle:** methodological transparency and simple baselines before added complexity.

This roadmap separates the credibility-hardening stages from the longer event-centric migration. Nothing here implies a validated trading strategy.

## Current status

| Area | Current implementation |
|---|---|
| Production scorer | `gpt-5-mini`, prompt `p3`; 83.3% held-out categorical agreement with the project rubric |
| Missing output | Explicit pending/scored/retry/failed states; missing-only configurable retries; no missing-to-neutral substitution |
| Preservation | Source-level raw audit, reversible versioned exclusions/restoration, guarded permanent purge API |
| Aggregation | Session-aligned `simple_mean` baseline plus relevance, intensity/relevance, and full-weighted sensitivities |
| Timing | Normalized Istanbul timestamp, explicit timing bucket, versioned holiday/half-day session assignment |
| Prediction | Every market-linked consumer uses the session baseline and complete-price-series return alignment |
| Run health | Final success/degraded/failed state, five component states, structured warnings/errors, market-cache freshness policy |
| Automation | GitHub Actions weekday run with database snapshot on the data branch |
| Prediction claim | Exploratory; no validated alpha, strategy, or walk-forward result |
| Event migration | Headline aggregation primary; event dual-write enabled; KAP production access pending |

Data reference points: the checked-in local DB extends through 2026-07-07 and the latest known `origin/data` snapshot is dated 2026-07-28. Counts are omitted because the automated snapshot changes.

Additive schema initialization does not silently re-score data, rebuild aggregate tables, or regenerate historic findings/figures. Those changes require an explicit command, and result-changing reruns must be documented.

## Stage 1 - Terminology and documentation - completed

Completed 2026-08-01:

- separated measurement quality, media framing/polarization, and exploratory return prediction;
- replaced confidence language for LLM intensity with model-reported intensity and documented synthetic compatibility components;
- scoped headline metrics as categorical agreement with the project's human-label rubric;
- defined 30 observations as an exploratory reporting gate, not a reliability threshold;
- published current technical/methodological references, AI assistance disclosure, dated-artifact policy, and test-to-risk map;
- kept unresolved behavior visible instead of describing intentions as implementation.

## Stage 2 - Processing integrity and raw-data preservation - completed

Completed 2026-08-01:

### Missing scorer output

- added `pending`, `scored`, `retry_pending`, and `failed` headline states;
- preserved NULL sentiment fields when no valid result exists;
- made partial-result IDs explicit and invalidated duplicate/out-of-range results;
- added configurable missing-only retries and separate configurable HTTP retry/backoff;
- stored attempt count, attempt timestamp, last error, and backend component kind;
- admitted neutral only when the scorer explicitly returned neutral;
- restricted aggregation to complete eligible `scored` rows.

### Reversible exclusions

- added source-distinct `raw_headline_observations` before canonical deduplication;
- changed scraper filter failures into persisted exclusion metadata;
- added append-oriented `headline_exclusions` history with idempotent active exclusion and timestamped restoration;
- reconciled low-LLM-relevance decisions by rule/version without restoring unrelated exclusions;
- changed `clean` from deletion to reversible exclusion and added `restore-exclusion`;
- guarded low-level permanent deletion behind `confirm=True`, with foreign-key protection for raw observations.

### Pipeline health

- added final `success`, `degraded`, and `failed` outcomes;
- added scrape, scoring, aggregation, market-data, and audit component states;
- stored structured warnings and errors;
- distinguished partial source/factor failures from total ingestion failure;
- distinguished fresh-cache degradation from stale/absent market-data failure.

Regression evidence and still-open branches are listed in [docs/TEST_RISK_MAP.md](docs/TEST_RISK_MAP.md). Completion means the core behavior is implemented and protected by focused tests; it is not a claim that every network/provider or migration-failure path has been simulated.

## Stage 3 - Baselines, timing, and predictive alignment - completed

Completed 2026-08-01:

### Signal variants

`daily_signal_variants` now stores, per first-reactable market session:

1. `simple_mean` - primary unweighted baseline;
2. `relevance_weighted`;
3. `intensity_relevance_weighted`;
4. `full_weighted` - legacy intensity/relevance/time behavior under current neutral source/category defaults.

It also stores label counts/shares, unclassified count, population dispersion, source count, event count, and weighted denominators. Zero-weight variants remain NULL instead of falling back to the baseline.

`python -m analysis.prediction.sensitivity` compares correlations, directional agreement, distributions, and exploratory next-session metrics for all variants. It reports no preferred variant.

### Timing separation

- added `published_timestamp`, `timing_bucket`, and `session_rule_version`;
- normalized aware timestamps to Europe/Istanbul and treated naive timestamps as Istanbul-local;
- separated pre-open, during-session, post-close, weekend/holiday, and unknown-time buckets;
- encoded regular and half-day closes plus consecutive closures;
- made the unweighted mean independent of time weighting;
- retained the historical time multiplier only in the full-weighted sensitivity;
- moved `visualize.py`, `dashboard.py`, `evaluate.py`, `explore_signal.py`, and `analyze_external.py` to the session baseline;
- formed next-session targets on the complete ordered market series before joining sparse signals;
- labeled calendar-date and legacy category tables descriptive/compatibility only.

Focused tests cover formulas, timestamp boundaries and conversion, holidays and half-days, storage/versioning, consumer table use, and exact sparse-signal return alignment. Annual calendar maintenance and additional provider-zone fixtures remain ongoing operational risks, not reasons to revert the completed convention.

## Stage 4 - Inference, structure, and reproducibility - completed

Completed 2026-08-01. Implementation adds methods and reproducibility tooling;
it does not retroactively replace or regenerate dated research findings.

### Polarization inference

- reports raw camp/outlet means, the raw gap, and pooled Cohen's *d*;
- resamples publication dates and controls for topic and date;
- reports outlet/date clustered sensitivities and attempts event clustering only with defensible repeated event identity;
- separates coverage selection from event-held-fixed framing and labels lexical pairs unverified;
- retains descriptive, non-causal language and explicit rank/few-cluster diagnostics.

### Repository structure

- moved corpus/prediction/polarization analysis and optional research fetchers into documented packages;
- retained import-compatible root wrappers for established commands;
- left interdependent production modules in place to avoid an aesthetic import rewrite.

The current incremental boundaries and compatibility-entry-point policy are documented in [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md).

### Reproducible demonstration

- `python -m scripts.demo` uses committed sample headlines, cached sentiment, prices, session assignment, and all four variants;
- emits `signal_results.csv`, `audit.json`, and `signal_variants.png`;
- is regression-tested without a key, network, model load, or private database.

## Predictive decision rule

No result becomes a finding merely because 30 observations are available. A predictive conclusion requires a pre-specified variant and target, chronological out-of-sample evaluation, controls, transaction costs, multiple-testing discipline, and enough independent evaluation windows to characterize uncertainty. A null result is acceptable. No parameter should be tuned after observing an appealing in-sample result.

## Event-centric migration status

The event workstream remains separate from the completed credibility stages:

- experiment registry, feature flags, rolling backups, event schema, and headline-to-event dual-write exist;
- KAP integration is built and dry-run validated against a historical development sample but remains disabled pending production access;
- structured extraction, session windows, entity/cap weighting, abnormal-return feature store, walk-forward evaluation, and a final event-path cutover/null decision remain future work.

See [MIGRATION.md](MIGRATION.md) for the phase gates and compatibility rules.

## Historical record (June-July 2026)

The following work is retained for provenance and is not the current to-do list:

- **2026-06-10/11:** parsing, relevance rules, category coverage, duplicate cleanup, aggregate freshness, return-shift, and Granger double-shift fixes;
- **2026-06-12:** production scorer switched from XLM-RoBERTa to gpt-5-mini; historical headlines re-scored; intensity floor, relevance grading, initial `signal_date`, experiment registry, event schema, and dual-write added;
- **2026-06-13:** canonical 300-label rubric and prompt `p3`; KAP integration built against the historical development sample but left disabled;
- **2026-06-16:** GitHub Actions became the primary daily automation path;
- **2026-06-19/24:** disagreement adjudication, intra-annotator consistency, active-learning export, and public methodology narrative;
- **2026-06-25:** corpus description and market-factor context;
- **2026-07-07:** exploratory target/aggregation sweep with FDR correction, polarization robustness work, and external-series analysis; the predictive sweep remained null and exploratory;
- **2026-08-01:** Stages 1-4 completed: terminology, omission-aware state/retries, reversible observation preservation, component outcomes, session variants/alignment, dependence-aware polarization inference, incremental package cleanup, and the offline demo.

## Maintenance

- update full-holiday and half-day data from an authoritative Borsa Istanbul calendar before each year;
- review RSS health and source definitions when collection volume changes;
- preserve scorer/prompt/experiment provenance across any re-score;
- run aggregation explicitly after intended eligibility/calendar changes;
- keep research snapshots dated and never silently overwrite their sample interpretation;
- update [docs/TEST_RISK_MAP.md](docs/TEST_RISK_MAP.md) whenever a behavioral risk gains or loses coverage.
