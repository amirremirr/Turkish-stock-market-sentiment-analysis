# BIST 100 Sentiment Pipeline - Methodology

**Current scorer-method version:** 2026-06-13 (`p3`)

**Documentation audit:** 2026-08-01

**Research status:** active and exploratory

This document describes the method implemented in the current code. It separates current behavior from historical compatibility paths and does not describe a validated trading strategy.

## 1. Three separate research areas

The repository contains three related but distinct lines of work:

1. **Sentiment and relevance measurement quality.** Does the scorer reproduce the project's written human-label rubric consistently?
2. **Media framing and polarization.** Do outlets or outlet groups show systematic descriptive differences in story selection and tone?
3. **Exploratory return prediction.** Do session-aligned news measurements contain information about subsequent BIST 100 returns?

Evidence in one area does not validate another. Agreement with a human-label rubric is not predictive power, and an outlet-associated tone difference is not evidence of profitable alpha or causal political bias.

## 2. Observation capture, canonical headlines, and exclusions

The production unit is still a headline. RSS and fallback HTML items within the configured collection window are parsed into source, title, URL, publication metadata, provisional category, and source payload metadata.

The ingestion path deliberately separates two layers:

- `raw_headline_observations` records each stable, source-distinct fetched item before canonical deduplication. Its key prefers source-native item ID, then source-scoped URL, then source/title/date metadata, so replaying one feed item is idempotent while the same item observed by two outlets remains two audit observations.
- `headlines` is the canonical processing table. Its historical global URL uniqueness remains for compatibility; URL-less deduplication is source/title/date scoped. A same-URL cross-source observation can therefore share one canonical row while remaining visible in the raw audit table.

The keyword relevance rules no longer drop an otherwise age-eligible fetched observation. The scraper annotates it with `is_excluded`, reason, rule, and rule version. Insertion links that decision to a versioned row in `headline_exclusions`. An active exclusion keeps the canonical row out of scoring and aggregation without deleting its text or audit record.

`python main.py clean` applies the current keyword rule as another reversible exclusion and then explicitly rebuilds aggregates when a decision changed. `restore-exclusion ID` timestamps the active exclusion's `restored_at` value and rebuilds aggregates. A manual restoration of the current LLM-relevance rule is honored during that rebuild; a newly stored relevance judgment supersedes the override and permits a fresh versioned decision. Repeated exclusion is idempotent, and the history is retained. Permanent canonical-row deletion exists only through `database.delete_headlines(..., confirm=True)` (or its explicitly named alias); it is not the normal filtering workflow. The source-level raw observation survives such a canonical purge with `headline_id` set to NULL.

This preservation guarantee begins with observations fetched by the current implementation. It cannot reconstruct items rejected or deleted by older code before the audit table existed, and the configured lookback still determines which feed items are considered for ingestion.

## 3. Production scorer and labeling target

The production backend is OpenAI `gpt-5-mini` using prompt version `p3` in `sentiment_llm.py`. The prompt requests:

- direction: `positive`, `neutral`, or `negative`;
- model-reported sentiment intensity in `[0, 1]` for directional labels;
- topic category;
- graded relevance in `[0, 1]`.

The target is the convention in [LABELING.md](LABELING.md): the direction a Turkish-equity investor would read from the headline alone. Neutral is the default in the labeling rubric for routine or genuinely ambiguous reporting. It is not a fallback for processing failure.

Prompt `p3` showed 83.3% categorical agreement on the 270 canonical human labels held out from its 30 few-shot examples. Relevance at the 0.25 cutoff showed 90.7% agreement with 300 human keep/drop judgments. These figures measure agreement with one project's annotation convention, not calibrated correctness or market prediction.

### LLM score and compatibility components

For the LLM backend, the continuous score is derived from the returned direction and intensity:

```text
positive -> +intensity
negative -> -intensity
neutral  -> 0
```

Positive, neutral, and negative components are then derived mechanically so legacy database and relabeling interfaces can operate.

> For the LLM backend, sentiment direction and intensity are returned by the model. Any positive, neutral, and negative components derived from those values are synthetic compatibility fields, not calibrated probabilities of correctness or class membership.

`score_components_kind` records this distinction. LLM rows use `synthetic_compatibility`; XLM-RoBERTa rows use `softmax_probability`; migrated rows that cannot be identified conservatively use `legacy_unknown`.

### Missing output and retry semantics

Scoring is stateful and omission-aware:

1. eligible rows begin as `pending`;
2. only explicit, valid response IDs are accepted;
3. an omitted, duplicate-invalidated, out-of-range, or otherwise absent item keeps all sentiment fields NULL, increments `scoring_attempts`, and becomes `retry_pending`;
4. only the missing subset is retried, up to configurable `LLM_SCORING_MAX_ATTEMPTS`;
5. an exhausted item becomes `failed` with attempt timestamp and last error;
6. an explicit returned neutral is stored as a valid `scored` row.

Malformed response envelopes and request failures raise rather than fabricate values. HTTP transport retries have their own configurable limit and backoff. A successful write is atomic at the row/batch boundary and records model and component provenance. Rows already marked `scored` are not downgraded by a replayed failure.

## 4. Historical XLM-RoBERTa fallback

The original scorer was `cardiffnlp/twitter-xlm-roberta-base-sentiment`. It emits three softmax probabilities, with score `P(positive) - P(negative)`. Thresholds of `+0.05` and `-0.05` produced 76.8% agreement on the same 198 labels used to tune them, so that figure is in-sample.

XLM-RoBERTa remains available through `SENTIMENT_BACKEND="xlmr"` but is not the active production scorer. Mixing scorers within one research series is not methodologically acceptable; a backend change requires explicit provenance and a complete, documented re-score.

## 5. Relevance measurement and aggregate eligibility

The LLM stores `headlines.relevance` on a continuous 0-1 scale:

| Grade | Prompt interpretation |
|---|---|
| 1.0 | directly about Turkish markets, economy, or policy |
| 0.7 | global financial/geopolitical news with clear Turkish implications |
| 0.4 | business news with an indirect or weak connection |
| 0.1 | barely related |
| 0.0 | unrelated material |

Rows below `RELEVANCE_MIN_FOR_AGGREGATION = 0.25` receive an active, versioned `llm_relevance` exclusion. If a later relevance result meets the cutoff, reconciliation restores only the active exclusion created by that LLM rule; it never removes an unrelated editorial or keyword exclusion. A NULL relevance value has weight 1.0 for legacy scored rows, but a pending or failed row is never admitted merely because relevance is NULL.

Aggregate eligibility requires all of the following:

- `processing_status = 'scored'`;
- complete score, label, component, model, and score-timestamp fields;
- a publication date and, for the canonical predictive table, an assigned session;
- no active exclusion;
- relevance at or above the configured cutoff after NULL legacy values are mapped to 1.0.

## 6. Session-aligned signal variants

`daily_signal_variants` is the canonical market-linked aggregate. For each `signal_date`, it stores four pre-specified variants:

```text
simple_mean
    = sum(score_i) / n

relevance_weighted
    = sum(score_i * relevance_i) / sum(relevance_i)

intensity_relevance_weighted
    = sum(score_i * relevance_i * max(abs(score_i), 0.10))
      / sum(relevance_i * max(abs(score_i), 0.10))

full_weighted
    = sum(score_i * relevance_i * intensity_i * source_i * time_i * category_i)
      / sum(relevance_i * intensity_i * source_i * time_i * category_i)
```

`simple_mean` is the primary baseline. In current pipeline calls, source and category multipliers take their neutral default of 1.0, so `full_weighted` reproduces the historical intensity x relevance x time formula. The intensity floor keeps an explicit zero-score neutral observation from automatically receiving zero weight. It is an engineering convention, not confidence.

The same row stores headline and label counts, positive/negative/neutral shares, unclassified count, population dispersion, source count, `event_count`, and the three weighted denominators. Here `event_count` means distinct bridge-linked event records attached to the eligible headlines. It is not a count of independently resolved real-world events and must not be interpreted as independent event coverage. A weighted zero denominator returns NULL rather than silently falling back to the simple mean.

Eligible scores must also share one experiment identity. New score writes store the configured experiment ID. If more than one eligible identity is present, aggregation stops before changing session assignments, exclusions, or any derived table. Mixing is possible only with the explicit `allow_mixed_experiments`/`--allow-mixed-experiments` override; a full pipeline run then persists a degraded aggregation state and the identities that were mixed.

Experiment identity is never guessed. Legacy identity may be reconstructed only when stored evidence uniquely establishes it, and the reconstruction is recorded auditably. Scores written before the `experiment_id` column existed retain NULL and are represented by a clearly marked, model-scoped legacy identity until a reviewed provenance migration resolves them. That migration assigns an identity only to rows with an exact model/prompt match, complete and consistent score components, and no existing assignment; it never overwrites a non-NULL identity and never modifies a score, label, timestamp, or model name. Every assignment appends a row to `experiment_assignment_audit` recording the evidence relied on, so a reconstructed identity stays distinguishable from one recorded at scoring time, and a row whose evidence is ambiguous keeps NULL and keeps blocking aggregation.

`analysis.prediction.sensitivity` compares all four variants through pairwise correlations, directional agreement, distributions, and next-session exploratory metrics. It sets `preferred_variant` to NULL and does not tune or choose a specification on the evaluation sample.

### Legacy and category tables

Compatibility tables remain deliberately separate:

- `daily_sentiment`: publication-date, legacy full-weighted aggregate;
- `daily_sentiment_by_signal`: session-date, legacy full-weighted shape;
- `category_daily_sentiment`: publication-date, legacy full-weighted category aggregate;
- `category_sentiment_by_signal`: session-date, unweighted category description.

They support historical and descriptive consumers. They are not the primary market-return signal. Derived tables are cleared and rebuilt only when aggregation is explicitly invoked; schema initialization alone does not change them.

## 7. Publication timing and market-session assignment

The predictive date is the first Borsa Istanbul session able to react. When a timestamp is available, it is normalized to `Europe/Istanbul`; a naive timestamp is interpreted as Istanbul-local. The normalized ISO timestamp is stored as `published_timestamp` alongside:

- `timing_bucket`: `pre_open`, `during_session`, `post_close`, `weekend_or_holiday`, or `unknown`;
- `signal_date`: the assigned first-reactable session;
- `session_rule_version`: the calendar/assignment version used.

The implemented policy is:

- before 10:00 on a trading day -> `pre_open`, same session;
- 10:00 through the scheduled close -> `during_session`, same session;
- after close -> `post_close`, next trading session;
- weekend or configured full holiday -> next available session;
- missing/invalid time on a trading day -> `unknown`, next session conservatively.

The regular configured close is 18:10. Configured official half-days use a 13:00 close, and consecutive closures are skipped. Timing classification and time weighting are independent: `simple_mean` has no time multiplier, while the historical time convention is retained only inside `full_weighted` (1.5 before 10:00, 1.0 through hour 18, 0.8 later, 1.0 when unknown).

This policy was **verified against production records rather than assumed**. `scripts/timing_audit.py` states the competing reading — that `signal_date` is the publication session — and refutes it on the 607 rows where the two readings differ, confirming the first-reactable reading on all 3 893. Doing so exposed a defect: the return-window builder had been treating the already-shifted `signal_date` as a publication date, so every `post_close` and `weekend_or_holiday` window was built one session late and measured the session *after* the one the news could move. That is not look-ahead — a late window uses less information, not more — but it would have surfaced a real relationship as a null. The windows were rebuilt; no score, label, detailed category or experiment identity was touched. `research/timing.py` is now the single definition, and [docs/TIMING.md](docs/TIMING.md) records the proof.

Every market-linked consumer (`visualize.py`, `dashboard.py`, `evaluate.py`, `explore_signal.py`, and `analyze_external.py`) now reads `daily_signal_variants` and uses `simple_mean`. No predictive consumer falls back to publication-date `daily_sentiment`. Next-session price targets are computed on the complete ordered price series before sparse signal rows are joined, preventing multi-session gaps from being mislabeled as one-session returns.

## 8. Pipeline and component outcomes

Each full pipeline run has a canonical final state: `success`, `degraded`, or `failed`. `pipeline_runs` also records separate scrape, scoring, aggregation, market-data, and audit states, plus structured `warnings_json` and `errors_json`.

Examples of the implemented policy include:

- one or more failed RSS sources with usable observations -> degraded scrape;
- all configured ingestion paths failing -> failed run;
- scorer omissions exhausted for some items -> degraded scoring/audit, with those rows excluded;
- aggregation exception or structurally incomplete `scored` rows -> failed component/run;
- market download failure with sufficiently fresh cached prices -> degraded; absent or stale cache beyond `MARKET_DATA_STALE_AFTER_DAYS` -> failed;
- optional market-factor failure -> degraded.

Legacy `ok`, `error`, `recovered`, and `crashed` rows remain readable and are mapped to a canonical status for reporting; new full runs use the new vocabulary.

## 9. Evaluation gates and interpretation

Days with fewer than three included headlines are treated as thin for the existing report. `MINIMUM_OVERLAP_DAYS = 30` is a presentation gate for exploratory statistics.

Thirty observations do not make a result reliable, establish adequate power, or validate a trading rule. A predictive claim still requires a pre-specified signal and target, chronological out-of-sample evaluation, market controls, multiple-testing discipline, transaction costs, and enough independent windows to characterize uncertainty. The project has no validated alpha or strategy.

## 10. Media framing and polarization

Outlet comparisons remain descriptive. The maintained inference module loads
source-distinct observations, reports raw camp/outlet means and pooled Cohen's
*d*, resamples whole publication dates for a deterministic bootstrap interval,
and fits `sentiment ~ camp + C(category) + C(date)`. Conventional uncertainty is
shown alongside separate outlet- and date-cluster sensitivities. Event-cluster
inference is attempted only when an explicit repeated cross-camp canonical event
identifier is available and diagnostically adequate.

Story selection and framing are separate outputs. Topic and event coverage
describe selection. Framing holds a defensible event fixed; the current 1:1
headline-event bridge is not accepted as shared-event evidence. In its absence,
deterministic one-to-one lexical/date pairs are labeled unverified sensitivity
candidates and never known same stories. These controls address selected forms
of dependence but do not establish intent or causal political bias. Full method
and limitations: [docs/POLARIZATION_METHODS.md](docs/POLARIZATION_METHODS.md).

## 11. Automation, provenance, and artifacts

The primary automation is `.github/workflows/daily.yml`, scheduled weekdays at 06:30 UTC (09:30 Istanbul). It restores the SQLite snapshot from the data branch, runs the pipeline, publishes current output, and updates that branch. `run_scheduled.py` and Windows Task Scheduler files are legacy/local routes.

Scored rows retain model provenance including the API snapshot and prompt version. `EXPERIMENT_ID = "v1-p3"` identifies the current research configuration. The checked-in local database covers through 2026-07-07; the latest known `origin/data` snapshot is dated 2026-07-28. Counts are omitted because the automated snapshot changes.

Files under `docs/*_findings.md` and checked-in figures are dated research artifacts. Additive schema migration does not silently regenerate them. A result-changing re-score, re-aggregation, or analysis rerun must be explicit and its changed interpretation documented.

## 13. Descriptive signal families and indicators

The detailed `category` assigned at scoring time is a measurement input and is frozen. A `signal_family` layer is derived from it plus transparent Turkish headline rules and carries its own version, so the economic grouping can be revised without redefining the historical record. Families separate channels the topical categories do not: monetary policy, inflation/macro, political-regulatory risk, FX, banking sector, named-issuer events, global risk, market recap, media narrative, and other.

The banking boundary is entity specificity, not industry. Sector- and system-level banking news is `banking_financial_sector`; a named listed bank's own earnings, dividend or disclosure is `company_kap`, exactly as for any other issuer. A named bank without an issuer-level event is assigned to the sector and flagged ambiguous rather than forced. Unresolvable assignments are reported in a coverage report; they are never silently coerced.

Market recap is a separate rules-based classification, not a scoring category. Extending the scoring prompt would change the stored model identity, split the experiment and invalidate the held-out validation, so recap detection lives beside the score instead. A recap reports a price move that already occurred; including such headlines in a directional signal creates a reverse-causality trap in which the tone follows the return by construction. Recaps are preserved for attention and reverse-causality analysis and excluded by default only from directional research outputs.

A domestic-only composite excludes global risk and market recap. The pre-existing overall session aggregate is unchanged; the composite is an addition, not a replacement.

Time-series normalization uses observations strictly before the date being described. Abnormal tone compares each outlet, outlet-family and family against its own prior rolling window; a full-sample mean would leak future information into every historical value and make later evaluation meaningless. Below a minimum history the value is NULL, and a zero-variance prior yields NULL rather than an unbounded z-score. This is time-series normalization against a key's own past, not a cross-sectional ranking against other keys on the same date.

Disagreement indicators measure variation among observed news sources. They are not market uncertainty and are not described as such. Cross-outlet statistics require a minimum number of independently represented outlets and report NULL otherwise, because zero would assert a consensus that was never observed. Volume indicators separate headline count, distinct-event count and outlet breadth, since syndicated copies of one wire story are one event covered widely rather than several independent observations.

Wherever a value cannot be defensibly computed the indicator reports NULL. A zero would be read as a substantive neutral finding when the truth is an insufficient sample.

## 12. Principal limitations

1. One primary human annotator defines the current rubric; multi-annotator generalizability is not established.
2. LLM intensity and synthetic compatibility components are uncalibrated.
3. Historical observations rejected or deleted before raw-audit capture cannot be reconstructed by the new schema.
4. Mixed model versions are detectable in the audit but not blocked automatically at aggregation time.
5. The holiday/half-day configuration requires authoritative maintenance as new exchange calendars are published.
6. Headline deduplication is not event resolution; event dual-write does not make the event the production aggregation unit.
7. Polarization inference remains sensitive to few clusters, researcher-defined camps, source coverage, scorer error, and the lack of verified repeated canonical events in the current bridge.
8. Predictive results are exploratory, underpowered for modest effects, and not evidence of a trading strategy.

See [AI_ASSISTANCE.md](AI_ASSISTANCE.md) for the AI-use disclosure and [docs/TEST_RISK_MAP.md](docs/TEST_RISK_MAP.md) for exact behavioral coverage and remaining gaps.
