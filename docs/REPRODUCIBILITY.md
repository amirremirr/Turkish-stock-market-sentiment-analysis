# Reproducibility and credentials

Two commands. Neither needs an API key.

## Run the demo

```bash
python -m scripts.demo_phase_a
```

Produces the full descriptive layer from a small committed fixture: signal
families, market-recap classification, prior-only abnormal tone, disagreement,
attention, candidate event groups, a worked event brief, timing-safe market
windows, the timing convention across buckets, and the session-vs-event
duplication factor.

**No API key, no network, no model download, no private database.** The test
suite asserts this by replacing `socket.socket` with a function that raises, so
the demo cannot open a connection even by accident. Runs in under a second and
is byte-identical across machines.

## Verify everything

```bash
python -m scripts.verify_all --db finance_sentiment.db
```

Six independent checks, each reported separately rather than stopping at the
first failure:

| Check | What it proves | Needs a DB? |
|---|---|---|
| `schema` | every table and column exists in a freshly built database, and the append-only triggers actually refuse a write | no |
| `artifacts` | the frozen study re-hashes to its stored value, its conclusion is verbatim, and the committed JSON matches the database | yes |
| `integrity` | historical scores, categories, provenance and reported findings are unchanged | yes |
| `timing` | the `signal_date` convention still holds against real rows, and every sampled window is aligned | yes |
| `tests` | the full pytest suite | no |
| `demo` | the offline demo produces its artifacts and makes no predictive claim | no |

Omit `--db` to run only the four checks that need no database. Exit code 0 means
everything that ran, passed.

## What requires credentials

Only three production components, all in the collection path:

| Component | Credential | Where | Consequence if absent |
|---|---|---|---|
| Sentiment scoring | `OPENAI_API_KEY` | `.env` locally; GitHub repo secret in CI | headlines are collected and stored but stay `pending`; nothing else is affected |
| KAP structured disclosures | MKK API Portal key + secret | `.env` | `KAP_ENABLED=False` by default; issuer events remain headline-derived |
| USD/TRY intraday rates | `ALPHA_VANTAGE_KEY` | `.env` | the FX-rate step logs a warning and skips; daily USD/TRY still arrives via the free factor feed |

Everything else — news scraping, price bars, market factors, session
assignment, taxonomy, indicators, event grouping, return windows, controls, the
research dataset, the frozen protocol, walk-forward evaluation, readiness
reporting, the dashboard and the demo — runs with **no credentials at all**.

The two scheduled workflows differ deliberately:

- `daily-pipeline` receives `OPENAI_API_KEY` because it scores headlines.
- `after-close-prices` receives **no key**. It settles price bars only, so it
  cannot alter a score, a label, an experiment identity or an event even if a
  future edit tried to. A test asserts the key is absent from its parsed
  environment and from every command it runs.

## Determinism

| Component | How determinism is achieved |
|---|---|
| Event grouping | ascending headline id, anchored window; a test compares forward and reversed input |
| Review sample | ordered by SHA-256 of the group key, never `random` |
| Bootstrap | fixed seed (`20260806`), asserted stable across calls |
| Logistic regression | fixed iteration count, fixed step, zero initialisation, no early stopping |
| Protocol hash | canonical JSON with sorted keys and no incidental whitespace |
| Frozen artifact hash | excludes the freeze timestamp, so re-checking proves identity |

## Environment

System Python 3.10, no virtualenv required. `pip install -r requirements.txt`
for the full local stack; `requirements-cloud.txt` is the slim CI set (no
torch — the scorer is an API backend). `requirements-lock.txt` pins the
versions the current results were produced under.

## The data branch

The production SQLite snapshot lives on an orphan `data` branch, not in `main`.
The cloud pipeline restores it, runs, and force-pushes a single snapshot back.
`scripts/guard_db_snapshot.py` refuses to publish a candidate that is behind the
canonical snapshot on any of five monotonic markers, so a stale local database
cannot overwrite production.

```bash
git fetch origin data
git show origin/data:finance_sentiment.db > finance_sentiment.db
```
