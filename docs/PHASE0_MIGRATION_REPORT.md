# Phase 0 — Production migration verification and deployment report

This report has two parts, recorded in the order they happened. Do not read the
first as describing the second.

| Part | Date | Scope |
|---|---|---|
| **§§1–12 — verification** | 2026-08-06 | Copy-only. Nothing was migrated in place and nothing was pushed. Every figure came from throwaway copies. |
| **§13 — deployment** | 2026-08-06, later same day | The canonical production database was migrated and published to the `data` branch after the verification passed. |

Sections 1–12 are preserved as written at verification time. Statements there
such as "nothing was pushed" describe that stage, not the current state; §13
records what changed at deployment.

Reproduce the verification with:

```
python -m scripts.verify_migration backups/phase0_canonical_data_2026-07-31.db
```

---

## 1. Production freeze confirmed

`gh workflow list` reports `daily-pipeline` as **`disabled_manually`**
(workflow id 296979662), so `origin/data` is frozen and safe to treat as a
fixed target.

It did not stop cleanly. The three scheduled runs before it was disabled all
**failed**, at the same place:

| Run | Date | Result |
|---|---|---|
| 30619653608 | 2026-07-31 | success — produced the canonical snapshot |
| 30804506972 | 2026-08-03 | failure |
| 30895676162 | 2026-08-04 | failure |
| 30992589067 | 2026-08-05 | failure |

```
pipeline.MixedExperimentAggregationError: aggregation blocked because eligible
scores span multiple experiment identities:
[legacy-unassigned] model=gpt-5-mini-2025-08-07/p3, v1-p3
```

Newly scored headlines receive `experiment_id='v1-p3'` from
`_resolve_experiment_id`, while the 3 465 legacy rows keep `experiment_id IS
NULL` and resolve to `[legacy-unassigned] model=…`. Two identities trip the
mixed-experiment safeguard in `aggregate_step`.

**The safeguard worked correctly and the data is intact.** Each run failed *at*
the aggregation step, so the later "Persist DB to data branch" step never ran.
`origin/data` is still commit `b1ffde7` (2026-07-31T09:23Z), pre-migration and
uncontaminated. This is confirmed by the snapshot hash in §2.

This is an open blocker for resuming production. See §8.

---

## 2. Snapshot inventory

Both files are byte-preserved backups taken before any work began.

| | Canonical (`origin/data`) | Stale local (working tree) |
|---|---|---|
| Backup path | `backups/phase0_canonical_data_2026-07-31.db` | `backups/phase0_local_stale_2026-07-07.db` |
| Origin | `git show b1ffde7:finance_sentiment.db` | working-tree copy |
| SHA-256 | `90e3ec76d5e351d1d9ab31928c50e9d6c7e44d5a6b31172e6ec660ff3d8649ac` | `73a2206588575d1a578ab8ad8ba9099bdb0bd576f48b1c00eced421ca359b1ba` |
| Size | 3 006 464 bytes | 1 851 392 bytes |
| `PRAGMA user_version` | 0 | 0 |
| Headlines | 3 465 | 1 991 |
| Headline dates | 2026-03-12 → 2026-07-31 | 2026-03-12 → 2026-07-07 |
| BIST price rows | 108 (2026-02-20 → 2026-07-31) | 91 (2026-02-20 → 2026-07-07) |
| `market_factors` | 267 | 213 |
| `external_series` | **0** | 890 |
| `pipeline_runs` | 52 | — |
| Scorer identity | `gpt-5-mini-2025-08-07/p3` (single) | same |

The project does not use `PRAGMA user_version`; schema state is identified by
column presence, which is what `scripts/verify_migration.py` compares.

Note the inversion on `external_series`: the GDELT and Google Trends series
exist **only** in the stale local copy. Production has never held them, because
those fetchers are not part of the cloud run.

The canonical file is also recoverable at any time from git history
(`git show b1ffde7:finance_sentiment.db`), which is a stronger guarantee than
the on-disk backup alone. The two backup binaries are intentionally **not
committed** — they are 4.8 MB of redundant blob, and `.gitignore` already
excludes database files.

---

## 3. Schema: before and after

Applying `init_db()` is purely additive. Nothing was dropped or renamed.

**4 tables added** (all empty on arrival — they are populated by
`aggregate_step`, not by the migration):

`daily_signal_variants` · `category_sentiment_by_signal` ·
`raw_headline_observations` · `headline_exclusions`

**Columns added to existing tables:**

| Table | Added |
|---|---|
| `headlines` | `experiment_id`, `processing_status`, `scoring_attempts`, `last_scoring_attempt_at`, `scoring_last_error`, `score_components_kind`, `published_timestamp`, `timing_bucket`, `session_rule_version` |
| `pipeline_runs` | `scrape_status`, `scoring_status`, `aggregation_status`, `market_data_status`, `audit_status`, `warnings_json`, `errors_json` |

---

## 4. Stage-by-stage result

The historical-score digest covers `sentiment_score`, `sentiment_label`,
`scored_at` and `model_name` across all 3 465 rows, ordered by id.

| Stage | File size | Content digest | Score digest |
|---|---|---|---|
| baseline | 3 006 464 | `6a6fb6b805a3e5a4` | `4507c065f4a87439` |
| `init_db` | 3 321 856 | `65d02084aa1fd6aa` | `4507c065f4a87439` |
| `init_db` again | 3 321 856 | `65d02084aa1fd6aa` | `4507c065f4a87439` |
| `backfill_session_assignments` | 3 538 944 | `8eef21825ae5f514` | `4507c065f4a87439` |
| `reconcile_relevance_exclusions` | 3 579 904 | `a8036f85adc387a4` | `4507c065f4a87439` |

**The score digest is constant across every stage.** No historical model output
was rewritten at any point.

The second `init_db()` produced an identical content digest — a proven no-op,
not merely an unchanged file size.

### Row counts

Every pre-existing table holds exactly its original count:

| Table | Before | After |
|---|---|---|
| `headlines` | 3 465 | 3 465 |
| `events` | 3 465 | 3 465 |
| `bist100_prices` | 108 | 108 |
| `market_factors` | 267 | 267 |
| `usdtry_rates` | 72 | 72 |
| `pipeline_runs` | 52 | 52 |
| `daily_sentiment` | 94 | 94 |
| `daily_sentiment_by_signal` | 68 | 68 |
| `category_daily_sentiment` | 495 | 495 |
| `headline_exclusions` | *(new)* | 272 |
| `daily_signal_variants` | *(new)* | 0 |
| `category_sentiment_by_signal` | *(new)* | 0 |
| `raw_headline_observations` | *(new)* | 0 |

The three historical derived tables survive `init_db()` untouched. They are
cleared and rebuilt only by `aggregate_step`, which Phase 0 did not run.

---

## 5. Backfilled metadata

**Processing state** — all 3 465 rows classify as `scored`. None landed in
`pending`, `retry_pending`, or `failed`:

```
processing_status      {'scored': 3465}
score_components_kind  {'synthetic_compatibility': 3465}
experiment_id          {'NULL': 3465}
```

`synthetic_compatibility` is correct: the `p_*` fields from the LLM scorer are
derived compatibility values, not calibrated probabilities.

**Timing buckets** — 3 465 rows assigned, none left NULL:

| Bucket | Rows | Share |
|---|---|---|
| `during_session` | 2 019 | 58.3% |
| `pre_open` | 692 | 20.0% |
| `unknown` | 435 | 12.6% |
| `weekend_or_holiday` | 213 | 6.1% |
| `post_close` | 106 | 3.1% |

The 487 rows with `published_hour IS NULL` all received a conservative bucket
(`unknown` on a trading day, `weekend_or_holiday` otherwise) and were pushed to
the next available session rather than assumed tradable — the documented
behaviour.

---

## 6. Verification gates — 10 of 10 pass

| Gate | Result |
|---|---|
| Stable table row counts unchanged | PASS |
| Scored-sentiment digest unchanged across all stages | PASS |
| Second `init_db()` is a content no-op | PASS |
| Second `init_db()` adds no columns | PASS |
| No tables removed | PASS |
| No columns removed | PASS |
| Every previously scored row marked `scored` | PASS |
| Single eligible experiment identity | PASS |
| Session backfill targets real trading sessions | PASS |
| Missing-hour rows receive a conservative bucket | PASS |

---

## 7. Deviations requiring sign-off

Both are permitted by design. Neither is silent.

### 7.1 — 41 `signal_date` values re-derived (0.9% of rows)

`backfill_session_assignments` touched all 3 465 rows because
`session_rule_version` was NULL. 3 424 assignments were confirmed unchanged.
**41 changed, every one moving earlier**, all with `published_hour IS NULL`,
all clustered on 2026-05-25 → 2026-05-31.

The superseded calendar rule treated the whole Kurban Bayramı stretch as closed
and pushed these headlines to 2026-06-02. It was wrong, and the price table
proves it:

| Date | Weekday | Price row | Volume |
|---|---|---|---|
| 2026-05-25 | Mon | yes | 8.45 bn |
| 2026-05-26 | Tue | **yes** | 3.43 bn (half day) |
| 2026-05-27–29 | Wed–Fri | no | — (Kurban Bayramı) |
| 2026-06-01 | Mon | **yes** | 8.55 bn |
| 2026-06-02 | Tue | yes | 8.53 bn |

2026-05-26 traded as a documented half day and 2026-06-01 traded normally. The
old assignment skipped two real sessions; the new one does not. Every corrected
target was verified to be both a trading day under the current calendar and a
date with an actual price row.

This is the intended purpose of `TRADING_CALENDAR_RULE_VERSION`
(`bist-official-calendar-2025-2026-v2`) — a versioned rule exists precisely so a
corrected calendar can re-derive stale assignments.

**Consequence to accept:** these 41 headlines will land on different sessions
than in any analysis run before the migration. No published finding depends on
them (`daily_signal_variants` has never existed in production), but the change
is real and is recorded here rather than absorbed silently.

### 7.2 — 272 reversible low-relevance exclusions created (7.9% of rows)

`reconcile_relevance_exclusions` inserted 272 rows into `headline_exclusions`
for headlines graded below `RELEVANCE_MIN_FOR_AGGREGATION = 0.25` under rule
`llm-relevance-p3-cutoff-0.25`.

- No headline was deleted — `headlines` still holds 3 465 rows.
- Every exclusion is active with `restored_at IS NULL`, i.e. fully reversible.
- Raw observations and scores are untouched.

These headlines become ineligible for aggregation. Since production has no
`daily_signal_variants` history, nothing published changes — but the first
aggregate built after migration will rest on 3 193 eligible rows, not 3 465.

---

## 8. Resolved — reviewed legacy provenance migration

Migrating the schema alone does not fix the failing pipeline: the moment the
next run scores a headline, the mixed-experiment error from §1 returns — 3 465
legacy rows at `NULL` against new rows at `v1-p3`.

**Approved resolution:** reconstruct the legacy identity as `v1-p3` rather than
minting a separate legacy identity, which would have kept the blocker alive.
The stored evidence uniquely establishes it: all 3 465 rows carry exactly
`gpt-5-mini-2025-08-07/p3`, which *is* the configuration `v1-p3` names.

### Eligibility — all five clauses must hold

1. `sentiment_score IS NOT NULL`
2. `processing_status = 'scored'`
3. `model_name` exactly equals the reviewed identity
4. no conflicting evidence — label, `scored_at` and all three score components
   present, and `score_components_kind` either NULL or `synthetic_compatibility`
5. `experiment_id` is NULL or blank

A non-NULL `experiment_id` is never overwritten. No score, label, timestamp or
model name is modified. Anything ambiguous keeps NULL and keeps blocking
aggregation — an unassigned row is visible, a wrongly assigned one is not.

### Audit trail

Every assignment appends to `experiment_assignment_audit`
(`headline_id`, `assigned_experiment_id`, `assignment_method`, `evidence`,
`reviewed_at`, `migration_version`), enforced append-only by SQLite triggers
that abort any `UPDATE` or `DELETE`. `assignment_method` is
`reviewed_legacy_backfill`; a rollback appends `reviewed_legacy_rollback` rows
rather than erasing anything. A reconstructed identity is therefore always
distinguishable from one recorded at scoring time.

Sample row:

```json
{"headline_id": 11, "assigned_experiment_id": "v1-p3",
 "assignment_method": "reviewed_legacy_backfill",
 "migration_version": "legacy-experiment-provenance-v1",
 "reviewed_at": "2026-08-05T23:10:38Z",
 "evidence": {"model_name": "gpt-5-mini-2025-08-07/p3",
              "score_components_kind": "synthetic_compatibility",
              "scored_at": "2026-06-12T23:50:10Z",
              "rule": "exact model/prompt identity with complete score components and no prior experiment_id"}}
```

### Result on a copy of the canonical database

| | Before | After |
|---|---|---|
| File SHA-256 | `82ce263303902b392404da99b8ef45a2517f9bee3c866d9c17ea32bf8237629b` | `b4123e7dddef5053cb16f1ef37b978e0821a1b23a88a49808bd6f18ef184f1ea` |
| Score digest | `fd3a7516a47f695a95a47966f603530a31690d94b3754409fe34a9a2c3d4eed8` | **identical** |
| `experiment_id` NULL | 3 465 | 0 |
| `experiment_id = v1-p3` | 0 | 3 465 |
| Eligible identities | `['[legacy-unassigned] model=gpt-5-mini-2025-08-07/p3']` | `['v1-p3']` |
| Audit rows | — | 3 465 (`reviewed_legacy_backfill`) |
| `headlines` / `events` / `bist100_prices` | 3 465 / 3 465 / 108 | unchanged |
| Blocked rows | — | 0 |

*(The before-hash differs from §2 because the schema migration of §4 was applied
first; the provenance step starts from that state.)*

### End-to-end proof

Simulating one freshly scraped headline scored as `v1-p3` — the exact scenario
that broke production:

```
new headline id: 4796      its experiment_id: v1-p3
eligible identities: ['v1-p3']
aggregate_step WITHOUT override -> SUCCEEDED, 71 signal sessions
daily_signal_variants: 71 rows    category_sentiment_by_signal: 417 rows
new headline in audit trail: 0 rows   (scored normally, not reconstructed)
```

Aggregation succeeds with no `--allow-mixed-experiments` override. A companion
test asserts the same scenario *fails* without the backfill, so this proves a
real fix rather than a hypothetical one.

### Rollback verified

```
reverted 3465 row(s)   skipped diverged 0 row(s)
experiment_id NULL: 3465     still v1-p3: 1   <- the scoring-time assignment survives
audit methods: {reviewed_legacy_backfill: 3465, reviewed_legacy_rollback: 3465}
headlines total: 3466        (nothing deleted)
```

Rollback reverts only what this migration wrote. A row reassigned since then is
reported as diverged and left alone.

---

## 9. Artifacts confirmed unchanged

- **Offline demo: byte-identical.** `python -m scripts.demo` was run from a
  clean `git worktree` at `HEAD` and from the Phase 0 tree; all three artifacts
  match exactly, PNG included:

  | Artifact | SHA-256 (first 16) |
  |---|---|
  | `audit.json` | `56a4b8f049250596` |
  | `signal_results.csv` | `941d1d93a82dfd40` |
  | `signal_variants.png` | `da3154cae4ce3e56` |

- **No findings, PNGs or CSVs regenerated.** `git status` shows no change to
  `docs/sample_output.png`, `docs/corpus_overview.png`,
  `docs/external_overview.png`, `docs/polarization.png`, `labels_validated*.csv`
  or any dated findings document.
- **The live working-tree database was never opened for writing.** Its SHA-256
  is `73a2206…`, identical before and after Phase 0.
- `docs/polarization_findings.md`, `docs/polarization_dynamics.png` and
  `polarization_dynamics.py` appear in `git status` as **pre-existing
  uncommitted user work from before Phase 0**. They were not touched.

---

## 10. Test suite

`201 → 250 passed` (+49), full suite green in 40.7 s.

| File | Tests | Covers |
|---|---|---|
| `tests/test_production_migration.py` | 15 | Additive-only migration, score immutability, `init_db` idempotency, processing state, experiment identity, conservative timing buckets, calendar-correction validity, reversible exclusions, derived-table survival, source-never-written, fixture determinism, fixture/schema agreement |
| `tests/test_snapshot_guard.py` | 9 | Stale-snapshot refusal, single-marker regression, override recording, absent-table tolerance, CLI exit codes |
| `tests/test_legacy_experiment_provenance.py` | 25 | Exact eligible set assigned, no score/label/timestamp/model change, existing IDs never overwritten, six conflicting-evidence cases left unassigned, conflicting scorer still blocks aggregation, additive + idempotent, audit completeness, append-only trigger enforcement, reconstruction distinguishable from scoring-time assignment, single identity after backfill, new `v1-p3` headline aggregates, blocked without the backfill, rollback scope/idempotency/divergence handling |

---

## 11. Proposed production migration commands

**Not executed. Shown for approval.** Run only after §8 is decided.

```bash
# 0. Confirm the workflow is still disabled
gh workflow list --all | grep daily-pipeline        # expect: disabled_manually

# 1. Re-take a backup at the moment of migration
git fetch origin data
git show origin/data:finance_sentiment.db > backups/pre_stage_a_$(date -u +%Y-%m-%d).db
sha256sum backups/pre_stage_a_$(date -u +%Y-%m-%d).db

# 2. Re-verify on a copy (must print VERDICT: PASS)
python -m scripts.verify_migration \
    backups/pre_stage_a_$(date -u +%Y-%m-%d).db \
    --json-out reports/phase0_verification.json

# 3. Install the canonical snapshot locally, replacing the stale copy
python -m scripts.guard_db_snapshot \
    backups/pre_stage_a_$(date -u +%Y-%m-%d).db \
    --reference finance_sentiment.db          # expect: safe to publish
cp finance_sentiment.db backups/superseded_local_$(date -u +%Y-%m-%d).db
cp backups/pre_stage_a_$(date -u +%Y-%m-%d).db finance_sentiment.db

# 4. Apply the schema migration in place
python -c "import database; database.init_db()"
python -c "import database; print(database.backfill_session_assignments())"
python -c "import database; print(database.reconcile_relevance_exclusions())"

# 5. Survey, then apply the reviewed provenance migration
python -m scripts.migrate_legacy_experiment_id --survey     # expect eligible 3465, blocked 0
python -m scripts.migrate_legacy_experiment_id --apply \
    --json-out reports/phase0_provenance.json

# 6. Confirm the result
python -c "import database; print(database.get_eligible_experiment_ids())"   # expect ['v1-p3']
python -m pytest -q
python -m scripts.demo --output-dir /tmp/demo_check
```

### Rollback

**Provenance migration only** — reverts `experiment_id` to NULL for exactly the
rows it assigned, leaving any identity written since then untouched:

```bash
python -m scripts.migrate_legacy_experiment_id --rollback
```

**Whole migration** — the schema step adds columns and tables but rewrites no
historical value, so rollback is a file replace:

```bash
# Local rollback
cp backups/pre_stage_a_<DATE>.db finance_sentiment.db

# Or restore straight from immutable git history
git show b1ffde7:finance_sentiment.db > finance_sentiment.db

# Verify the restore
python -c "
import hashlib,pathlib
print(hashlib.sha256(pathlib.Path('finance_sentiment.db').read_bytes()).hexdigest())"
# expect 90e3ec76d5e351d1d9ab31928c50e9d6c7e44d5a6b31172e6ec660ff3d8649ac
```

Nothing has been pushed, so no remote rollback is required. `origin/data`
remains at `b1ffde7`.

---

## 12. Not done at verification time

**State as of the verification stage only.** §13 supersedes the first three
entries; the last two still hold.

- The canonical database was **not** migrated in place. Both the schema and
  provenance migrations were exercised only on copies. *(Superseded by §13.)*
- Nothing was pushed to `origin/data` or `main`. *(Superseded by §13.)*
- `aggregate_step` was run only on a throwaway copy, to prove the mixed-identity
  blocker is resolved. No production derived table was built. *(Superseded by
  §13.)*
- The workflow remains disabled; the guard step is added but untested against a
  live run. **Still true.**
- No Phase A feature work was started. The `signal_family` taxonomy — including
  the approved `banking_financial_sector` family — is **recorded** in
  [ROADMAP.md](../ROADMAP.md) but not implemented. **Still true.**

---

## 13. Deployment completed — 2026-08-06

The verification in §§1–12 passed, and the canonical production database was
subsequently migrated and published. This section records that deployment; it
did not happen during the verification stage above.

### Source and result

| | Before | After |
|---|---|---|
| `origin/data` commit | `b1ffde7cdd33ef3ffafbd7976af1777f7113b9b2` (2026-07-31T09:23Z) | `da703ba1e54e75ad01dd00eea354e444d97b054a` (2026-08-06) |
| Database SHA-256 | `90e3ec76d5e351d1d9ab31928c50e9d6c7e44d5a6b31172e6ec660ff3d8649ac` | `01c271b71bed70a7be96c715701b506ab6fa93e52bc7e95655918f5719bb1e81` |
| Size | 3 006 464 bytes | 4 911 104 bytes |

`origin/data` was confirmed still at `b1ffde7` immediately before the operation,
and that fetched copy — not the stale working-tree database — was the sole
production source. The pre-migration file was backed up to
`backups/production_pre_migration_2026-08-06.db` (hash as above, gitignored and
uncommitted), and all migration work ran on a separate copy.

**The published snapshot was fetched back and re-hashed: the remote hash matched
the local migrated hash exactly.**

### What the migration did

| Change | Count |
|---|---|
| Historical rows given reviewed experiment identity `v1-p3` | **3 465** |
| Append-only provenance audit records created | **3 465** |
| Reversible low-relevance exclusions created | **272** |
| Session assignments corrected | **41** |

Every corrected session date was verified to be a real trading day with an
actual price row. No headline was deleted; every exclusion is active and
reversible.

### Verified invariants

- 3 465 historical headlines preserved.
- Score digest `fd3a7516a47f695a95a47966f603530a31690d94b3754409fe34a9a2c3d4eed8`
  identical before the migration, after every stage, and on the fetched-back
  remote copy — no `sentiment_score`, `sentiment_label`, `scored_at` or
  `model_name` was rewritten.
- `experiment_id = 'v1-p3'` on 3 465 rows, 0 remaining NULL, 0 blocked.
- A second `init_db()` was a content no-op.

### Aggregation

`aggregate_step` ran with **no** `--allow-mixed-experiments` override:

```
status=success   sessions=70   warnings=[]
eligible_experiment_ids=['v1-p3']
mixed_experiments=False   mixed_experiments_override=False
```

`daily_signal_variants` 70 rows, `category_sentiment_by_signal` 416 rows.
Headline, event, price, factor, FX and run counts unchanged.

### Publication method

The snapshot is an orphan commit holding `finance_sentiment.db` plus the
pre-existing `sentiment_vs_bist100.png` blob carried over byte-identical from
`b1ffde7`. The commit was assembled with git plumbing against a temporary index
rather than `git checkout --orphan`, so the working tree and `HEAD` were never
touched — the documented procedure's result without disturbing unrelated
uncommitted work. `sentiment_vs_bist100.png` was preserved rather than dropped:
deleting it on a force-pushed branch would destroy it irrecoverably.

### Still true after deployment

- **The scheduled workflow remains disabled** (`disabled_manually`).
- **No live scrape or scoring run has occurred.** The published database
  contains no data collected after 2026-07-31.
- The pre-migration backup remains available locally and uncommitted, and the
  pre-migration commit `b1ffde7` remains available for rollback.

---

## 14. Known unresolved issue — incomplete 2026-07-31 price bar

**This predates the migration.** It was present in the canonical `b1ffde7`
snapshot and was neither introduced nor corrected by the deployment.

The `bist100_prices` row for **2026-07-31** was captured during the trading
session, not after it:

| Symptom | Detail |
|---|---|
| Capture time | pipeline run 52 started 2026-07-31T09:22Z = 12:22 Istanbul, mid-session (BIST closes 18:10) |
| Volume | **0.0** — the only such row in the table |
| Stored close | 13 251.72 |
| Completed daily close | 13 458.10 (live provider), a **1.53%** difference |

The stored value is an intraday snapshot recorded as if it were a daily close,
so the 2026-07-31 close, its daily return, and any return computed into or out
of that session are wrong.

Sentiment aggregates are unaffected: `aggregate_step` reads only headlines, so
`daily_signal_variants` and the category tables do not depend on this row.

**This row must be refreshed before any price-based analysis uses that session**,
and daily-bar completeness must be enforced before the scheduled workflow is
re-enabled — otherwise every run started at 06:30 UTC will write another
mid-session bar for the current day.
