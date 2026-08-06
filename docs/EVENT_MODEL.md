# Event model and market targets

A versioned event-level research dataset connecting auditable news groups to
timing-safe subsequent market outcomes.

**No predictive model, trading strategy or frozen research protocol exists yet.**
This package builds the dataset those would later consume.

## Candidate events, not verified events

Groups are formed by transparent rules. They are called **candidate event
groups** or **algorithmic event groups** throughout, and the vocabulary does not
change until a human reviews one. Calling them verified would claim a resolution
the method does not have.

A headline joins a group when **all** of these hold:

| Criterion | Rule |
|---|---|
| Entity | a shared normalized entity, or none on either side |
| Family | identical `signal_family` |
| Time | within 48h of the group's **first** member |
| Similarity | normalized-title Jaccard ≥ 0.30 with a shared entity, ≥ 0.60 without |

Every mapping retains its similarity score, match rule, entity overlap and
algorithm version in `event_headline_map`.

### The window is anchored, not sliding

The 48h window is measured from the group's first member and never advances. An
advancing window chains: a recap headline printed most mornings is always within
24h of the previous one, so months of separate daily stories collapse into a
single pseudo-event. Anchoring bounds a group's total duration to the window,
which is what the window is meant to mean.

Where a publication timestamp is missing there is no proximity evidence at all,
so grouping falls back to requiring the **same reaction session**.

### Source-independent duplication

A group drawn entirely from one outlet is one voice repeating itself. It is
flagged `is_single_source` rather than counted as corroborated coverage, and
`is_singleton` marks a group of one.

## Entities and event types

Entity extraction is a curated dictionary lookup over institutions, listed
issuers, macro concepts and instruments — not named-entity recognition. A
curated list is auditable and deterministic; an entity it misses produces a
*smaller* group rather than a wrong one. Matching is word-bounded, so `bist`
does not match `bistro`.

Event types (`rate_decision`, `data_release`, `rating_action`, `earnings`,
`corporate_action`, `m_and_a`, `regulatory_action`, `appointment`, `guidance`,
`market_move`, `geopolitical`) come from the same transparent vocabulary. An
unmatched headline gets **no type** rather than a catch-all label.

## Manual review

`event_group_audit` is append-only, enforced by trigger. A `split`, `merge`,
`confirm`, `reject` or `annotate` action **appends** a record; it does not
rewrite the automatic grouping. The algorithm's output and the human judgement
therefore always stay distinguishable, and regrouping never erases review
history.

```python
db.record_event_group_action(group_key, "split", "analyst",
                             algorithm_version=..., headline_ids=[...],
                             rationale="different decision")
```

## Market windows

A return is only a valid target if someone could have earned it.

| Timing | Window | Entry → exit | Cutoff |
|---|---|---|---|
| `pre_open` | `same_session_open_to_close` | open → close, same session | that session's open |
| `post_close` | `close_to_next_open` | close → next open | that session's close |
| `post_close` | `next_open_to_next_close` | next open → next close | that session's close |
| `post_close` | `close_to_next_close` | close → next close | that session's close |
| `during_session` | — | **blocked** | no intraday data |
| unknown | — | **blocked** | publication time unknown |

Each row records its `information_cutoff`, `assumed_execution`, and the exact
`entry_price_field` / `exit_price_field` used, so the assumption behind any
number is readable rather than implied.

**Only `complete` and `corrected` price bars are visible to the window builder.**
A provisional bar is an intraday snapshot; using one would reintroduce exactly
the fault that corrupted 2026-07-31.

### Eligibility

| Status | Reason |
|---|---|
| `eligible` | pre-open or post-close, with settled prices |
| `blocked` | `intraday_prices_unavailable` |
| `blocked` | `publication_time_unknown` |
| `blocked` | `market_recap_excluded_by_default` |
| `blocked` | `no_complete_price_bar` / `no_following_complete_session` |

## Controls

The distinction the schema enforces: a control is **tradable** only if its value
was observable at the assumed execution moment.

| Set | Kind | Controls |
|---|---|---|
| `none` | tradable | — (raw return baseline) |
| `em_lagged` | tradable | `EEM_lag1` |
| `em_oil_fx_lagged` | tradable | `EEM_lag1`, `BZ=F_lag1`, `USDTRY=X_lag1` |
| `em_contemporaneous` | **contemporaneous_descriptive** | `EEM` |
| `em_oil_fx_contemporaneous` | **contemporaneous_descriptive** | `EEM`, `BZ=F`, `USDTRY=X` |

EEM and Brent close hours after Borsa Istanbul, so their same-day close is *not*
observable when a Turkish position opens the next morning. Same-session sets are
therefore labelled contemporaneous and are for describing how much of a move was
market-wide — never for execution-sensitive claims.

Betas come from a **rolling prior window** (60 sessions, minimum 30
observations). A full-sample beta applied to its own estimation period leaks the
future into every residual. Below the minimum the residual is NULL.

Control sets are pre-specified and versioned so a later stage cannot choose
whichever flatters a result. **No model selection or random splitting happens
here.**

## Tables

| Table | Contents |
|---|---|
| `event_groups` | candidate groups with tone, dispersion, novelty, review state |
| `event_headline_map` | every mapping with similarity and match rule |
| `event_group_entities` | normalized entities per group |
| `event_group_audit` | append-only manual review actions |
| `event_return_windows` | per-group windows with full timing provenance |
| `control_residual_returns` | residuals per session, window and control set |
| `event_research_dataset` | one row per (event, window), ready for walk-forward |

## Blocked data

Recorded on every dataset row rather than omitted, because a missing column
reads as an oversight while a stated reason reads as a limitation:

- `intraday_prices` — during-session events are descriptive only
- `consensus_expectations` — licensed; macro surprise cannot be computed
- `kap_structured_events` — KAP production access pending

## Limitations

- Grouping is lexical and dictionary-based; it does not understand meaning.
- The corpus is dominated by singletons — most headlines have no near-duplicate.
- Novelty measures how often an entity has already produced a candidate group,
  not whether the underlying news is new.
- Dispersion measures disagreement among covering outlets, not market
  uncertainty.
- No grouping has been human-reviewed; all are `unreviewed` by default.
- Nothing here has been evaluated out-of-sample, and no predictive relationship
  is claimed.
