# Sentiment-table consumer audit

Audit date: 2026-08-01. This report distinguishes live analytical consumers
from table producers, compatibility storage, tests, and committed output
snapshots. “Predictive” below means that sentiment is compared with a later
market observation; it does not imply that a validated predictive model exists.

## Table roles

| Table | Time key and aggregation | Supported role |
|---|---|---|
| `daily_signal_variants` | First exchange session able to react; `simple_mean`, `relevance_weighted`, `intensity_relevance_weighted`, and `full_weighted` | Canonical market-linked table. `simple_mean` is the pre-specified default; weighted columns are sensitivity variants only. |
| `category_sentiment_by_signal` | Reaction session and category; unweighted mean | Canonical session/category descriptive table. It is stored for future category reporting but has no live analytical reader at this audit. |
| `daily_sentiment` | Headline publication calendar date; legacy full-weighted aggregate | Legacy descriptive and migration-compatibility table. It must not drive a return test. |
| `daily_sentiment_by_signal` | Reaction session; legacy full-weighted aggregate | Transitional compatibility table. It has no live analytical reader after the variant-table migration. |
| `category_daily_sentiment` | Publication calendar date and category; legacy aggregate | Legacy descriptive table. |

## Live Python consumers

| Consumer | Classification | Table(s) read | Current behavior |
|---|---|---|---|
| `visualize.py` | Market-linked chart plus exploratory predictive panels | `daily_signal_variants` through `database.get_signal_variants()` | Bars and rolling series use session-aligned `simple_mean`. Scatter and rolling correlation compare that baseline with the subsequent session's close-to-close return. The return is formed on the complete ordered price table before the signal join. |
| `dashboard.py` | Descriptive market overlay | `daily_signal_variants` through the adapter | Displays session-aligned `simple_mean` against the matching BIST 100 session close. It does not estimate a return relationship. |
| `evaluate.py` | Quality audit and exploratory predictive audit | Primarily `daily_signal_variants`; also `category_daily_sentiment` and a freshness-only read of `daily_sentiment` | Aggregate quality and market tests use session-aligned `simple_mean`; weighted variants are side-by-side sensitivity diagnostics. The calendar/category reads are explicitly labelled legacy/descriptive. |
| `explore_signal.py` | Exploratory predictive analysis | `daily_signal_variants` through the adapter | Uses `simple_mean` for the target screen. The three weighted variants appear only in the aggregation-sensitivity block. Subsequent-session BIST return is computed before joining signal dates. |
| `analyze_external.py` | Mixed descriptive and exploratory predictive analysis | `daily_signal_variants` through the adapter; raw publication-date headlines for polarization | Domestic-press sentiment is session-aligned `simple_mean`. BIST and FX leads are constructed on their complete ordered market tables before sparse external/sentiment joins. The headline-derived polarization series is a separate descriptive publication-date measure. |
| `analysis/prediction/sensitivity.py` | Exploratory predictive sensitivity report | `daily_signal_variants` through the adapter | Reports all four variants without selecting a winner. Its subsequent-session return is constructed before the join. |
| `polarization_dynamics.py` | User research / descriptive legacy artifact | `daily_sentiment` | Loads the legacy calendar aggregate but does not use the returned frame in its current calculations or chart. Its active calculations use raw headline dates, FX, and external attention data. The file is user-owned and was intentionally not rewritten by this migration. |

There are no remaining live predictive reads of `daily_sentiment`,
`daily_sentiment_by_signal`, or either category table. The only direct
`daily_sentiment` analytical dependency is the unused load in the user-owned
`polarization_dynamics.py` artifact.

## Producers and non-analytical references

| File | Role |
|---|---|
| `pipeline.py` | Rebuilds derived sentiment tables. Legacy tables remain additive compatibility outputs; market-linked consumers do not use them. |
| `database.py` | Owns schemas and read/write adapters. It is infrastructure, not an analytical consumer. |
| `main.py` | Triggers aggregation and reports table-refresh messages; it does not read a sentiment series for analysis. |
| `tests/test_pipeline.py` and `tests/test_stage2_pipeline.py` | Exercise legacy derived-table compatibility and integrity. Test reads are validation fixtures, not research consumers. |
| `tests/test_signal_variants.py` and predictive-consumer tests | Exercise the canonical variant calculations and alignment rules. |

## Charts and committed snapshots

| Artifact | Producer | Status |
|---|---|---|
| `docs/sample_output.png` | `visualize.py` | Generated snapshot. New generations use the session-aligned unweighted baseline; the committed bitmap should be treated as historical until explicitly regenerated. |
| `dashboard.html` | `dashboard.py` | Generated self-contained snapshot, not a live database reader. New generations use the canonical baseline; the committed file may lag. |
| `docs/external_overview.png` | `analyze_external.py` | Dated research output. New generations use session-aligned unweighted domestic-press sentiment, while the committed findings document remains a historical snapshot. |
| `docs/polarization.png` | polarization research scripts | Descriptive outlet-associated snapshot derived from headline-level observations, not a daily market signal table. |
| `docs/polarization_dynamics.png` | user-owned `polarization_dynamics.py` | User research/descriptive snapshot. It is not evidence that the legacy daily aggregate remains an approved predictive input. |

## Migration rule

Any new market-linked analysis must obtain `daily_signal_variants` through the
database adapter, use `simple_mean` as its default sentiment input, and form
market leads on the complete ordered market table before joining or filtering
signal sessions. A weighted variant may be displayed only as a named
sensitivity analysis. Publication-date and category aggregates remain valid
for explicitly descriptive questions, but not as a fallback when the canonical
session table is empty.
