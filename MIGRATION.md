# Migration: Daily-Sentiment Pipeline → Event-Centric Research System

Phased migration, not a rewrite. **The legacy path keeps running and `run.bat
run` never breaks.** Each phase adds modules behind flags; deprecation comes
only after the new path beats baselines out-of-sample. Pre-migration snapshot:
git commit `acf47c4`.

Core reframing: the unit of analysis becomes the **event** (not the headline),
the target becomes **abnormal return** (not raw BIST direction), and validation
becomes **walk-forward from day one** (not a post-hoc audit layer).

## Phase status

| Phase | Theme | Status |
|---|---|---|
| 0 | Hardening: experiment registry, feature flags, rolling backups | ✅ done 2026-06-12 |
| 1 | Temporal alignment: `signal_date`, dual aggregation, dual L5 | ✅ done 2026-06-12 |
| 2 | Event schema + dual-write bridge (`events`, `event_entities`, `events_bridge.py`, `migrate-events`) | ✅ done 2026-06-12 — 755 events |
| 3 | KAP Tier-A ingestion (REST API spike → `ingest/kap.py`) | ⏳ next — needs API research spike |
| 4 | Structured extraction (direction/magnitude/event_type/entities replaces 3-class sentiment) | pending |
| 5 | Session windows W1/W2/W3 (`session_features` table) | pending |
| 6 | Entity linking + free-float cap weights | pending |
| 7 | Feature store + abnormal-return target (needs EM index data) | pending |
| 8 | Walk-forward evaluator (replaces L5 as the decision metric) | pending — needs ≥60 reliable days |
| 9 | Cutover or documented null result | Month 6+ |

## What exists after Phases 0–2

- `config.py`: `USE_EVENT_PIPELINE` (False), `EVENTS_DUAL_WRITE` (True),
  `EXPERIMENT_ID`, `SOURCE_TIERS` (A/B/C; AA + BloombergHT = B, rest = C)
- `database.py`: `experiments`, `events`, `event_entities` tables;
  `pipeline_runs.experiment_id`; `daily_sentiment_by_signal`
- `trading_calendar.py`: `signal_date()` — first session that can react
  (post-close / weekend / NULL-hour news rolls to the next trading day;
  NULL-hour is conservative to avoid lookahead)
- `events_bridge.py`: idempotent headline→event sync, runs after every score
  step; bridge semantics `direction = sentiment_score`,
  `credibility = tier default (A 1.0 / B 0.75 / C 0.5)`
- `run_scheduled.py`: rolling 7-day DB backups in `backups/daily_*.db`
- `evaluate.py` L5: calendar AND signal alignments side by side + Spearman +
  fixed-band hit rates

## Phase gates (decision points)

| End of | Gate | If NO |
|---|---|---|
| Phase 3 | KAP ingest ≥1 event/trading day over 2 weeks | stay RSS-heavy; delay cap-weighting |
| Phase 4 | Direction MAE ≤ 0.35 vs human labels on holdout | keep sentiment_score as direction bridge |
| Phase 8 | News features beat momentum/FX baselines OOS (net of 10 bps costs, ≥20 windows) | Phase 9 null-result path: stop tuning, keep archiving |

## Standing rules during migration

1. Migrations are additive only — never DROP a column or table mid-migration.
2. Every bulk re-score gets a new `EXPERIMENT_ID`; never silently mix scorer
   versions in one experiment.
3. Aggregation weights are frozen (see ROADMAP) — tuning happens only inside
   the walk-forward harness once it exists.
4. Tier C (general press RSS) never enters the core model until Tier A+B are
   wired (Phase 3).

## Parallel workstreams (don't block phases)

- Label sprint S1 (≥300 labels, now incl. `human_relevant` column) — manual
- Daily `run.bat run` keeps accumulating data — automated
- KAP API research spike (1 session) — feeds Phase 3
- BIST ticker master CSV (1 afternoon) — feeds Phase 6
- EM index price series (1 hour) — feeds Phase 7
