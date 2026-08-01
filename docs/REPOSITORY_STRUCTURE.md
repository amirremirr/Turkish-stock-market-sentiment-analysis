# Repository structure

The repository is being reorganized incrementally so existing commands and
imports continue to work. Root-level compatibility entry points remain where a
script was already part of the documented workflow.

## Current layout

| Path | Responsibility |
|---|---|
| `aggregation/` | Pure signal construction, including the unweighted baseline and weighting sensitivities. |
| `analysis/corpus/` | Corpus-quality and descriptive corpus reporting. |
| `analysis/polarization/` | Outlet-associated tone and same-event inference. |
| `analysis/prediction/` | Exploratory, session-aligned prediction sensitivity analysis. |
| `scripts/demo.py` | Deterministic offline demonstration using committed sample data. |
| `scripts/research/` | Optional research-data collection utilities outside the production loop. |
| `sample_data/` | Small public fixtures used by the offline demo. |
| `tests/` | Behavioral and methodological regression tests. |
| `docs/` | Methods, audits, risk-to-test mapping, and historical findings. |

The production orchestration and persistence modules remain at the repository
root for now (`pipeline.py`, `database.py`, `scraper.py`, `sentiment_llm.py`, and
`trading_calendar.py`). Moving those interdependent modules would create a broad
import rewrite with little methodological benefit.

## Compatibility entry points

The following root commands are thin wrappers around their organized modules:

- `python analyze_corpus.py` calls `analysis.corpus.report`.
- `python fetch_gdelt.py` calls `scripts.research.fetch_gdelt`.
- `python fetch_gtrends.py` calls `scripts.research.fetch_gtrends`.
- `python polarization_analysis.py` calls `analysis.polarization.inference` while retaining historical loader/filter helpers.
- `python same_story_analysis.py` calls the same maintained inference path and exposes the audited one-to-one fallback matcher.

New code should import the organized module rather than a root wrapper. The
wrappers may be removed only in a future breaking release with a documented
migration path.

## Boundaries

- `scripts/research/` utilities are optional and do not run in the scheduled
  production pipeline.
- `analysis/` code reports descriptive or exploratory evidence; it does not
  produce a trading signal for deployment.
- Generated outputs, private databases, credentials, and local cache files are
  not source modules and must not be imported by tests or the offline demo.
- Raw observations are persistence-layer records. Repository reorganization
  never authorizes deleting or rewriting them.
