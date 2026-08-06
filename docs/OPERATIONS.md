# Operations

How the production pipeline runs, what each scheduled job may touch, and how to
undo any of it.

## Two scheduled workflows

| | `daily-pipeline` | `after-close-prices` |
|---|---|---|
| File | `.github/workflows/daily.yml` | `.github/workflows/after_close_prices.yml` |
| Cron | `30 6 * * 1-5` (06:30 UTC ≈ 09:30 Istanbul, pre-open) | `10 16 * * 1-5` (16:10 UTC ≈ 19:10 Istanbul, post-close) |
| Scrapes headlines | yes | **no** |
| Calls the LLM scorer | yes | **no** — no credential is exposed to the job |
| Writes scores, labels, experiment IDs, exclusions, events | yes | **no** |
| Fetches prices and market factors | yes | yes |
| Rebuilds derived signal tables | yes | no |
| Publishes to the `data` branch | yes | yes |
| Commits a README figure to `main` | yes (skippable) | **never** |

Both jobs restore the database from `origin/data`, run, and force-push a single
orphan snapshot back. `origin/data` has no history: **the branch tip is the only
copy**, which is why the publication guard exists.

### Why two jobs

The morning job runs before the open, so the bar it stores for the current
session is an intraday snapshot by construction. It is written as `provisional`
and withheld from analysis, which means every analysis trails a session. The
after-close job returns once the market has settled and promotes that bar to
`complete`.

Splitting them also bounds the blast radius: the after-close job cannot reach
the research record at all, so a failure there costs price freshness and nothing
else.

### Concurrency

Both declare `concurrency.group: bist-database-writer` with
`cancel-in-progress: false`. They write the same SQLite file and force-push the
same branch, so they must queue rather than overlap, and neither may cancel the
other mid-write.

### The runtime guard beats the cron

The 16:10 UTC cron is a coarse trigger, not the decision. GitHub fires schedules
late and unpredictably — the 2026-07-31 incident came from a 06:30 UTC cron
that actually fired at 09:22 UTC, landing mid-session — and a UTC cron drifts an
hour against Istanbul across daylight-saving changes.

So `scripts/after_close_refresh.py` re-checks the real session close at runtime,
in Europe/Istanbul, including official half-day early closes. Before settlement,
on a weekend, or on a holiday, the job no-ops and every later step is skipped.

```bash
python -m scripts.after_close_refresh --check-only    # report the decision
python -m scripts.after_close_refresh                 # refresh if settled
python -m scripts.after_close_refresh --force         # ignore the guard
```

### Manual dispatch

```bash
gh workflow run daily.yml -f publish_chart=false      # skip the README figure
gh workflow run after_close_prices.yml
```

`publish_chart` defaults to `true`, so scheduled runs keep refreshing the README
chart. Only a manual dispatch can opt out.

Dispatching requires the workflow to be enabled — GitHub has no dispatch-only
enable. To rehearse while keeping the schedule off: `gh workflow enable` →
dispatch → `gh workflow disable`, and check the next cron time first.

## Price-bar completeness

Every daily bar carries a status (`price_bars.py`):

| Status | Meaning |
|---|---|
| `provisional` | observed before the session settled; withheld from analysis |
| `complete` | observed after the close plus the safety delay |
| `corrected` | a settled bar that replaced a provisional or invalid one |
| `provider_invalid` | a bar dated to a day the exchange did not trade |

Settlement is the scheduled close plus `PRICE_BAR_SETTLEMENT_MINUTES` (30).
Half-days settle from the official early close, not an assumed 18:10.

**`get_prices()` returns `complete` and `corrected` bars only by default.** An
unclassified (NULL-status) row is withheld too, because it is not a verified
one; `backfill_price_bar_status()` resolves those from recorded run times.

### Status transitions

Completion never runs backwards, and `corrected` is provenance rather than a
quality tier:

- `provisional → complete` / `corrected` — allowed
- `provider_invalid → corrected` — allowed
- `corrected → corrected` — a later settled refresh keeps the status and updates
  only values and the observation timestamp
- `corrected → provisional` / `provider_invalid` — **never**
- `complete → corrected` — only through the explicit repair path
  (`scripts/refresh_price_bar.py`, or `upsert_prices(mark_corrected=True)`)

Without stickiness, an ordinary refresh returning the same settled values erased
the record that a bar had needed repair — which is exactly what happened to
2026-07-31 during run 53.

A zero or missing volume on a session that did trade sets `bar_review_reason`
rather than changing the status: it is a data-quality signal, not proof the bar
is wrong. `list_price_bars_for_review()` surfaces those and anything withheld.

### Return recomputation

`daily_return` is rebuilt from the **full ordered stored series** after every
upsert, not from the downloaded window. A window's first row has no predecessor
inside it, so the provider returns NULL there and would overwrite a valid stored
value — that is how 2026-05-04 lost its return, along with 23 other sessions at
past window boundaries.

Only settled bars form the series: chaining a return through an intraday
snapshot would corrupt both of its neighbours. Provisional and invalid rows hold
NULL until they settle. The earliest stored session keeps NULL because its
predecessor is genuinely unavailable.

Correcting one close therefore updates that session's return and the following
session's automatically.

## Run status and the processing audit

A run reports `success`, `degraded`, or `failed`. The processing audit
distinguishes headlines by **eligibility**, not just processing status:

| Key | Meaning |
|---|---|
| `pending_eligible`, `retry_pending_eligible`, `failed_eligible` | genuinely unresolved — **these degrade the run** |
| `pending_excluded`, `retry_pending_excluded`, `failed_excluded` | carry an active exclusion; deliberately never scored |
| `scored`, `scored_excluded`, `active_exclusions` | context |

**Excluded headlines do not degrade a run.** The relevance filter withholds them
at ingest and the scorer skips them by design, so counting them as unresolved
marked every healthy run degraded and drained the meaning from that signal — run
53 reported degraded on 198 rows that were working exactly as intended.

They stay visible under an informational `excluded_items_not_scored` warning, so
a filter regression is still noticeable. Exclusions remain reversible: a restored
headline simply becomes eligible again and the next scoring pass picks it up. The
`processing_status` of an excluded row is never rewritten to make the audit pass.

## Phase A descriptive indicators

The morning ingestion run computes them after aggregation; the after-close price
job does not — it is prices only and never runs headline analytics.

| Table | Contents |
|---|---|
| `daily_family_signals` | per-family daily descriptive signals, plus the `__domestic__` composite |
| `abnormal_tone_daily` | prior-only normalization by outlet, outlet×family and family |
| `news_disagreement_daily` | dispersion and camp gaps among observed sources |
| `news_volume_daily` | attention shocks per family and an `__all__` series |

The step is fail-soft: a failure degrades the run and leaves every pre-existing
aggregate intact. Component status appears as `indicators` in the run record.

```bash
python -c "import pipeline; print(pipeline.indicators_step(return_outcome=True))"
python -m scripts.demo_phase_a          # offline, credential-free
```

See [FINANCIAL_INDICATORS.md](FINANCIAL_INDICATORS.md).

## Candidate events and research dataset

Computed by the morning run after the descriptive indicators; the after-close
price job never runs them. Component status appears as `events`.

| Table | Contents |
|---|---|
| `event_groups` / `event_headline_map` / `event_group_entities` | candidate groups and their evidence |
| `event_group_audit` | append-only manual review (split, merge, confirm, reject) |
| `event_return_windows` | timing-matched windows over settled bars only |
| `control_residual_returns` | residuals per session, window and control set |
| `event_research_dataset` | one row per event and window |

```bash
python -c "import pipeline; print(pipeline.events_step(return_outcome=True))"
```

Regrouping replaces rows for the current algorithm version only and never
touches the review audit. See [EVENT_MODEL.md](EVENT_MODEL.md).

## Rollback

### Data branch

The tip is the only copy, so roll back by pushing a known commit:

```bash
git push -f origin <commit>:refs/heads/data
git fetch origin data:refs/remotes/origin/data --force
git show origin/data:finance_sentiment.db | \
  python -c "import sys,hashlib;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())"
```

| Snapshot | Commit | DB SHA-256 |
|---|---|---|
| Pre-migration production | `b1ffde7` | `90e3ec76…3d8649ac` |
| Phase 0 migrated | `da703ba` | `01c271b7…f184f1ea` |
| 2026-07-31 price corrected | `c9ba964` | `3436939d…17fad392` |
| Controlled run 53 | `f32bdfd` | `a161b71e…0af1837` |
| Hardened run 54 | `7d26571` | `44354b6e…099add03` |
| Corrected provenance restored | `0f0ade4` | `0d8418a9…7afad93` |

A local backup of the pre-migration database is at
`backups/production_pre_migration_2026-08-06.db` (gitignored, uncommitted).

### Stop the schedules

```bash
gh workflow disable daily.yml
gh workflow disable after_close_prices.yml
```

### Revert code

```bash
git revert <commit> --no-edit && git push origin main
```

## Publication guard

`scripts/guard_db_snapshot.py` refuses to publish a database that is behind the
canonical snapshot on any monotonic marker (headline count, latest `scraped_at`,
latest `published_at`, latest price date, latest run start). Both workflows run
it immediately before the force-push.

```bash
python -m scripts.guard_db_snapshot finance_sentiment.db --reference-git origin/data
```

Exit `0` safe, `1` refused, `2` bad input. An override exists but requires a
written reason, which is echoed into the output for the record.
