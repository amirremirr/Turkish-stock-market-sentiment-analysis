# Architecture

The stock-level study is a separate package, `stock_research/`, inside the
same repository as the BIST 100 index study. It shares nothing with the index
pipeline at run time except four read-only constants.

## Isolation from the index study

| | Index study | Stock study |
|---|---|---|
| Database | `finance_sentiment.db` (canonical copy on the `data` branch) | `stock_research.db` (local, gitignored: it holds licensed raw data) |
| Protocol | `research/protocol.py`, sealed | `stock_research/hypotheses/registry.py`, its own hash |
| Scheduled runs | daily and after-close workflows | none |
| Imports the other | no | `config.BIST_HOLIDAYS`, `config.BIST_HALF_DAYS`, `config.ECONOMIC_CALENDAR`, `config.KAP_BASE_URL` |

The stock module opens `finance_sentiment.db` read-only, and only to count
headlines and to check whether the sealed index test has a result yet. No
existing file's behaviour was changed to build it.

## Modules

```
stock_research/
  config.py          constants that affect a result
  store.py           the database: schema, append-only triggers
  calendar.py        sessions derived from the benchmark; first session able to act
  guards.py          the sealed-window guard
  data/
    kap.py           resumable KAP sample ingestion (network)
    prices.py        providers, immutable bar snapshots, the analysis panel
  events.py          KAP taxonomy, flags, ticker resolution, event table
  insider.py         share-transaction form parser
  entities.py        news-to-ticker linker
  limits.py          price-limit rule, detection, data consistency check
  eventstudy.py      expected returns, AR, CAR, BHAR, overlap handling
  stats.py           clustered inference, Wald test, Holm, Benjamini-Hochberg
  costs.py           cost scenarios and fill rules
  hypotheses/
    common.py        Spec, Context, the verdict rule, shared helpers
    prep.py          admissibility filters, counted
    h1.py … h9.py    one registered hypothesis each
    registry.py      protocol document and hash, run_all, result storage
  pipeline.py        real context, the registered run, synthetic demo
  reporting.py       HTML report, CSV and JSON
  fixtures.py        synthetic panels and contexts with planted effects
  registry_doc.py    generates HYPOTHESIS_REGISTRY.md
  cli.py             command line
```

## Data flow

```
MKK API ──▶ sr_raw_kap_listing / sr_raw_kap_detail ─┐
Yahoo   ──▶ sr_raw_price_bars ──▶ Panel ────────────┼──▶ Context ──▶ H1…H9 ──▶ sr_results
finance_sentiment.db (read-only) ──▶ counts ────────┘                 │
                                                                      └──▶ report
```

Raw tables are append-only by trigger. Everything derived (the panel, the
event table) is rebuilt from them on each run and carries the version of the
rule that built it.

## Two rules enforced in code

1. **Rules before outcomes.** `pipeline.run_real` refuses unless the current
   protocol hash is in `sr_protocols`. The hash covers the nine
   specifications, every rule version and constant, and a SHA-256 of each
   result-bearing source file. Change a line of an estimator and the run
   refuses until the new protocol is registered.
2. **No partial sample.** `run_real` refuses while the KAP sample frame is
   incomplete. There is no flag for an interim look.

Results are stored under a manifest hash and never overwritten. A second run
of the same manifest is a no-op; a run under a different protocol or snapshot
is a new row beside the old one.

## The boundary between statistics and trading

`eventstudy.py` and `stats.py` produce abnormal returns and tests.
`costs.py` decides whether an order could plausibly have filled and what it
would have cost under stated scenarios. A hypothesis reports the two
separately: `primary` is the statistical test; `execution` is a long-only,
hedged, cost-charged version with its own coverage count. A verdict never
depends on the execution block, except that H5, whose claim is about buying
at a locked limit price, can only reach `execution_not_verifiable`.

## What was deliberately not built

- No changes to `kap_ingest.py`, `trading_calendar.py` or
  `events/entities.py`, though the audit found defects in each. They belong to
  the index study.
- No market-capitalisation estimate. The provider interface exists; the data
  do not.
- No scheduled job. Ingestion is started by hand and can be resumed.
