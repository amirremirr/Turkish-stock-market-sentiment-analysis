# Reproducibility

## Setup

Python 3.10. No dependency was added for the stock module; it uses what
`requirements.txt` already lists (numpy, pandas, scipy, matplotlib,
beautifulsoup4, lxml, requests, yfinance).

```bash
pip install -r requirements-cloud.txt
```

### Environment variables

Only KAP ingestion needs credentials. Everything else, including every test
and the synthetic demo, runs without any.

| Variable | Needed for | Where to get it |
|---|---|---|
| `MKK_API_KEY`, `MKK_API_SECRET` | `ingest-kap` | An application on apiportal.mkk.com.tr |

Put them in `.env` (gitignored). Never commit them. `stock_research.db` holds
licensed raw data and is gitignored for the same reason.

## Commands

```bash
python -m stock_research.cli status           # protocol hash, frame progress, table counts
python -m stock_research.cli ingest-kap       # resume the KAP sample (about 11 hours in total)
python -m stock_research.cli ingest-prices    # bars for every sampled issuer; skips what is stored
python -m stock_research.cli describe         # sample counts and data diagnostics; reads no return
python -m stock_research.cli register         # register the current protocol
python -m stock_research.cli run              # the registered real-data run
python -m stock_research.cli report           # HTML report from the latest stored run
python -m stock_research.cli demo             # every hypothesis on synthetic data
python -m stock_research.cli evaluate-linker  # entity linker on the labelled set
python -m stock_research.registry_doc         # regenerate HYPOTHESIS_REGISTRY.md
python -m pytest tests/stock_research         # 100 tests, no network, about two minutes
```

`run` exits with status 3 and prints why if the protocol is not registered
for the current code or the sample frame is incomplete.

Reports are written to `outputs/stock_research/` (gitignored): an HTML page, a
summary CSV and the full results as JSON.

## To reproduce a real-data run from nothing

```bash
python -m stock_research.cli ingest-kap       # until status shows frame_complete: true
python -m stock_research.cli ingest-prices
python -m stock_research.cli describe
python -m stock_research.cli register
python -m stock_research.cli run
```

Ingestion is resumable: every listing row and detail is committed as it
arrives, and a rerun skips what is stored.

**What will differ between two reproductions, and what will not.** The KAP
sample is deterministic: same anchors, same disclosures. Prices are fetched
from a provider that revises its history (adjustments, corrections), so two
snapshots taken on different days can differ. Each run records its snapshot
id and the retrieval time of every bar, and a result is tied to its snapshot
through the manifest.

## What a result is tied to

Every stored run has a manifest:

| Field | Meaning |
|---|---|
| `protocol_hash` | Specifications, rule versions, constants and source-file hashes |
| `dataset_snapshot` | Price snapshot id |
| `kap_frame_version` | Sample frame |
| `code_commit` | Git commit at run time |
| `data_origin`, `frame_complete` | Whether a claim is allowed |
| `analysis_date` | Run date |
| `sentiment_model` | Source of sentiment scores, where used |
| `benchmark`, `return_adjustment` | XU100; provider-adjusted closes |
| `bootstrap_seed` | Fixed seed for resampling |
| `sealed_from` | First session the sealed index test forbids, or null |
| `availability` | What each dataset could support, with reasons |
| `sample` | Counts and data diagnostics at run time |
| `manifest_hash` | SHA-256 of all the above |

`sr_manifests` and `sr_results` are append-only. Running the same manifest
twice stores nothing new.

## Determinism

- Clustered standard errors are closed-form.
- The bootstrap helper takes an explicit seed (`BOOTSTRAP_SEED`).
- Synthetic fixtures are seeded; the demo gives the same output every time.
- The logistic fit in H6 is a deterministic Newton iteration.

## Protecting the index study

The stock module writes only to `stock_research.db` and
`outputs/stock_research/`. To confirm the index study is untouched:

```bash
python -m scripts.freeze_result --db finance_sentiment.db --verify-only
python -m scripts.run_future_validation --check --db finance_sentiment.db
```

No stock-level outcome on or after 2026-08-10 is computed until the index
study's sealed test has recorded its result.
