# Turkish financial news and the BIST 100: a null result, honestly obtained

**Status of this document.** It reports three kinds of claim, and never lets
them blur:

| Kind | What it means | Where it appears |
|---|---|---|
| **Descriptive** | measured properties of the corpus and the market | §2, §3, §4, §6, §7 |
| **Retrospective exploratory** | out-of-sample under a frozen protocol, on data already collected and inspected | §11, §12, §13 |
| **Frozen future validation** | a test on data that did not exist when the rules were written | §15 |

Nothing here claims alpha. The headline result is a null, and the reason it is
worth reading is the discipline that produced it rather than the finding
itself.

---

## 1. Research question

*Does the tone of Turkish-language financial news carry information about the
next tradable move in the BIST 100 index, beyond what lagged market factors
already explain?*

Three commitments follow from taking that question literally.

**"Next tradable move" is a timing claim.** A return is only a valid target if
someone could have earned it, which requires knowing exactly when a position
could first have been opened. §5 and §9 are about getting that right, and about
what went wrong when it was merely assumed.

**"Beyond what market factors explain" requires controls that were observable.**
EEM and Brent close hours after Borsa Istanbul, so their same-day close is not
available to a Turkish position opening the next morning. §8 separates tradable
from descriptive controls.

**"Carries information" is a claim that has to be falsifiable.** §11 fixes what
would count as success *before* looking, and §13 reports what happened.

---

## 2. The Turkish financial-news dataset

Daily automated collection from Turkish financial news outlets, weekdays, since
March 2026. Every fetched item is preserved in `raw_headline_observations`
before any de-duplication or filtering, so the canonical table's rules can be
argued with after the fact rather than trusted.

| Property | Value |
|---|---|
| Headlines | ~4 200 |
| Sources | 11 Turkish outlets |
| Corpus span | 2026-03-13 → present (reaction sessions) |
| Price history | BIST 100 daily bars from 2025-01-02 (backfilled), live since 2026-02-20 |
| Market factors | EEM, Brent (BZ=F), USD/TRY from 2025-01-02 |

**Known coverage limitations.** The corpus is short — under six months of live
collection. Outlet mix is not a random sample of Turkish financial media. There
is no intraday data, and no licensed consensus-forecast data, so a macro
release cannot be turned into a surprise. KAP structured disclosures await
production API access; issuer events are headline-derived meanwhile.

---

## 3. Sentiment methodology

Headlines are scored by an LLM (`gpt-5-mini`, prompt `p3`) against a labelling
rubric developed for this project, not a generic sentiment scale: the question
asked is whether the item is *bullish or bearish for Turkish equities*, which
is often the opposite of whether it is good news.

| Property | Value |
|---|---|
| Held-out categorical agreement | **83.3%** (vs 68.5% for the prior prompt, 61.5% for majority-neutral) |
| Label set | 300 human-labelled headlines under `LABELING.md` |
| Positive↔negative flips on held-out data | 0 |
| Relevance cutoff | 0.25, validated at 90.7% agreement (1 false exclusion in 300) |

Every score carries `experiment_id`, model name and scoring timestamp.
Aggregation across mixed experiment identities is **refused at runtime**
(`MixedExperimentAggregationError`) rather than silently averaged — a rule that
caught a real production defect and stopped three days of runs rather than
letting them produce a quietly wrong number.

*Descriptive finding.* The corpus skews neutral (~59% of labels), the currency
family is the most bearish, and outlet tone differs systematically — pro-
government outlets score more bullish on the same day's news than opposition
outlets. That is a measurement of media slant, not of the market.

---

## 4. Taxonomy

Two versioned classifications sit *beside* the frozen LLM category, never
replacing it:

**Signal families** (`signal-family-v1`) — ten families including
`monetary_policy`, `inflation_macro`, `fx_lira`, `banking_financial_sector`,
`company_kap`, `global_risk`, `market_recap`. The banking/company boundary is
**entity specificity, not industry**: a named bank's earnings is `company_kap`;
"banking sector loan growth slows" is `banking_financial_sector`. Because the
rule keys on whether a specific listed entity is named, it applies uniformly
across sectors rather than special-casing banks.

**Market recap** (`market-recap-rules-v1`) — rules-based detection of "the index
closed up 1.2%" reporting. Recaps are excluded from directional research by
default because their tone *follows* the return by construction, and retained
for attention analysis and reverse-causality work.

Ambiguous classifications are flagged, counted and reported rather than forced
into a family.

---

## 5. Timing alignment

The single most consequential definition in the project.

**`signal_date` is the first trading session able to react to a publication.**
Not the session it was published in.

This was *proven*, not assumed. `scripts/timing_audit.py` states the competing
reading and lets production records refute it:

| bucket | rows | matches "publication session" | matches "first reactable" |
|---|---|---|---|
| `pre_open` | 899 | 899 | 899 |
| `during_session` | 2 375 | 2 375 | 2 375 |
| **`post_close`** | **235** | **0** | **235** |
| **`weekend_or_holiday`** | **259** | **0** | **259** |
| **`unknown`** | **435** | **0** | **435** |

`pre_open` and `during_session` cannot discriminate — for those the two
readings coincide. The verdict rests on the remaining 929 rows, and it is
unanimous.

Full definitions in [TIMING.md](TIMING.md).

---

## 6. Descriptive indicators

*All descriptive. None is a validated predictive relationship.*

- **Family signals** — per-family daily tone, plus a domestic-only composite
  that excludes global risk and market recap.
- **Abnormal tone** — standardised against a **prior-only** rolling window. A
  full-sample mean would leak the future into every historical reading.
- **Disagreement** — cross-outlet dispersion on the same day and family,
  requiring at least three independent sources; below that it is NULL, never a
  fabricated zero.
- **Attention** — headline count, observation count and source breadth reported
  separately, because four outlets carrying one decision is one event and four
  sources.

The recurring discipline: **NULL where a value could not be defensibly
computed.** A zero would assert neutrality that was never measured.

---

## 7. Candidate-event methodology

Headlines are grouped into **candidate event groups** by transparent rules, not
learned similarity. A headline joins a group when all of: a shared normalised
entity (or none on either side), identical signal family, publication within 48h
of the group's **first** member, and normalised-title Jaccard ≥ 0.30 with a
shared entity or ≥ 0.60 without.

**The window is anchored, not sliding.** An advancing window chains: a recap
headline printed most mornings is always within 24h of the previous one, so
months of separate daily stories collapse into a single pseudo-event. This was
an actual defect, found by inspecting the largest groups.

**Vocabulary.** These are *candidate* or *algorithmic* groups. Nothing calls
them verified events, because no human has reviewed one. A stratified review
sample (`scripts/event_review_sample.py`, 120 groups across 8 strata including
near-neighbour pairs that did *not* merge) is drawn deterministically and
**without reference to market returns**, so a reviewer cannot be nudged by
outcome.

*Descriptive finding.* 94% of groups are singletons. Most Turkish financial
headlines have no near-duplicate, so event-level statistics rest on a small
multi-headline subset.

---

## 8. Market targets

With `D` the first reactable session and `P` the session before it:

| Window | Entry → exit | Tradable? |
|---|---|---|
| `reactable_open_to_close` | open(D) → close(D) | **yes** |
| `prior_close_to_reactable_open` | close(P) → open(D) | no |
| `prior_close_to_reactable_close` | close(P) → close(D) | no |

The gap windows are **never tradable in any bucket**: entering at close(P)
requires holding a position before the news was public. They are kept because
that gap is where a pre-open story's reaction actually lands, and labelled for
what they are — a measurement of reaction, not an achievable return.

**Controls.** Tradable sets use only lagged factors (`EEM_lag1`, `BZ=F_lag1`,
`USDTRY=X_lag1`). Same-session sets are labelled
`contemporaneous_descriptive` in the schema and are barred from tradable
specifications. Betas come from a rolling 60-session prior window with a
30-observation minimum, estimated on the **full settled price history** rather
than only on sessions that carried an event — the window return is a property of
the index, defined on every session. Below the minimum the residual is NULL.

Only `complete` and `corrected` price bars are visible. A provisional bar is an
intraday snapshot.

---

## 9. Timing defects discovered

Three, all found by building the audit rather than by a test failing.

**1. Post-close and weekend windows were one session late.** The window builder
read the already-shifted `signal_date` as a publication date and stepped forward
again. A headline published Monday 21:00 was scored Tuesday-close to
Wednesday-open.

| Headline 1647, published 2026-06-08 21:00 | v1 (wrong) | v2 (correct) |
|---|---|---|
| information cutoff | 2026-06-09 18:10 | 2026-06-08 18:10 |
| assumed execution | 2026-06-10 10:00 | 2026-06-09 10:00 |
| entry → exit | close 06-09 → open 06-10 | open 06-09 → close 06-09 |

This is **not a look-ahead leak** — a late window uses *less* information than
it could. It is worse in a quieter way: it measures the session *after* the one
the news could move, so a genuine relationship would surface as a null and be
written up as "no predictive content found". A leak makes you wrong loudly;
this makes you wrong invisibly.

**2. An event's timing was assembled from two different headlines.** The group
took the *earliest* member's `signal_date` and the *most restrictive* member's
timing bucket. An event is not fully known until its last member is published,
so the governing member is now the last one, and it supplies both fields.

**3. A fold geometry applied where it could not fit.** The protocol declared its
primary geometry applicable at ≥50 sessions while needing 51 (40 train + 1
embargo + 10 test). At exactly 50 it selected that geometry and then blocked all
72 specifications. Thresholds are now derived from the parameters.

---

## 10. Dependence: the 14.9× pseudo-replication problem

The target is the **BIST 100 index return**. Every candidate event whose first
reactable session is 2026-06-09 is scored against *the same number*.

| | Frozen study |
|---|---|
| Event rows | 773 |
| Distinct events | 773 |
| Distinct sessions | **50** |
| Distinct outcomes | **50** |
| Duplication factor | **15.5×** |

Treating event rows as independent would shrink every standard error by roughly
√15.5 ≈ 3.9× — enough to turn a null into a publishable discovery, with no step
along the way that looks like cheating.

So the primary statistical unit is the **session**: one row per (first reactable
session, target window), with event features aggregated by rules fixed in
advance. `build_session_units` raises if two events on one session carry
different targets, because that could only mean two windows were mixed.

Residual dependence remains: adjacent sessions are not independent. The
one-session embargo bounds story overlap; the session-cluster bootstrap handles
cross-sectional dependence but not autocorrelation. Both are stated as
limitations rather than assumed away.

---

## 11. The walk-forward protocol

`walk-forward-protocol-v1`, hashed before comparative results were read.
Full specification in [PREDICTIVE_PROTOCOL.md](PREDICTIVE_PROTOCOL.md).

| Element | Value |
|---|---|
| Primary target | `reactable_open_to_close`, `raw_return` |
| Eligible buckets | `pre_open`, `post_close`, `weekend_or_holiday` |
| Excluded | `during_session`, `unknown`, market recaps, timing conflicts |
| Feature sets | 6 baselines, 6 news sets, enumerated in advance |
| Models | training mean, majority direction, ridge (α = 1), logistic (L2 = 1) |
| Folds | expanding-window, chronological, 1-session embargo |
| Missing values | drop the row; **never impute zero** |
| Uncertainty | session-cluster bootstrap, 2 000 resamples, fixed seed |
| Margins | MAE +0.05 and directional +0.05 over the best baseline |

Enforced structurally, not by convention: folds are prefixes of an ordered
session list; the standardiser learns from training folds only; a specification
below the sample gate is **not fitted** but recorded with its binding
requirement. Baselines are re-scored on exactly the sessions each news
specification predicted, so a coverage difference is never read as a model
difference.

---

## 12. Results

*Retrospective exploratory. Frozen artifact `bfdbadb0…`, protocol
`d987de7b…`.*

| | |
|---|---|
| Independent sessions | 50 |
| Specifications fitted | 22 |
| Specifications refused by the sample gate | 50 |
| **News specifications meeting all criteria** | **0** |

Best news specification against the session-matched best baseline:

| | MAE | Directional accuracy | Hit-rate 95% CI |
|---|---|---|---|
| `abnormal_tone` / ridge | 0.856 | 0.583 | [0.333, 0.833] |
| `market_controls_only` / logistic (baseline) | 0.912 | 0.667 | — |
| **Difference** | **+0.057** ✓ | **−0.083** ✗ | spans 0.5 ✗ |

It clears the MAE margin, loses on direction, and its interval includes chance.
One of three criteria is not two out of three; the pre-specified rule required
all three.

Fold stability: MAE 0.86–1.13, directional accuracy 0.39–0.67 across 2–3 fitted
folds per specification. Sensitivity analyses on multi-source-only (10 sessions)
and singletons-removed (12 sessions) samples were both **refused by the sample
gate** rather than fitted.

### Frozen conclusion

> No evaluated news specification demonstrated reliable incremental
> out-of-sample predictive value under the pre-specified criteria in the current
> sample.

This artifact is immutable. A later version that performs differently does not
revise it; it is a different study with its own artifact.

---

## 13. Interpreting the null

**What this result does and does not license.**

It does **not** show that Turkish financial news is uninformative about the BIST
100. With 50 independent sessions, the study had little power to detect a
relationship of the size one would plausibly expect. Most specifications were
never fitted at all. A null under these conditions is close to what an
absence-of-evidence result looks like when the evidence is simply thin.

It does show that **no easily-available news feature produced a detectable edge
under honest accounting** — and the qualifier matters, because the same data
handled less carefully would have produced something publishable. Three things
each would have manufactured a positive result on this corpus:

1. treating 773 event rows as independent (√15.5 ≈ 3.9× narrower intervals);
2. leaving the one-session timing shift in place and then "fixing" it once the
   result looked wrong;
3. reporting the MAE improvement of the best news model without the directional
   loss or the interval.

The honest version of this project is the null. That is the finding.

**What would change the answer:** more sessions. The binding constraint is data,
not method — the protocol is fixed and rerunning it on a longer corpus requires
no new decisions.

---

## 14. Limitations

**Data.** Under six months of live collection; 50 independent sessions in the
frozen study. No intraday prices (42% of dataset rows blocked). No licensed
consensus data, so macro surprise cannot be computed. KAP structured
disclosures pending production access.

**Method.** Grouping is lexical and dictionary-based; it cannot recognise a
shared topic expressed in different words. 94% of groups are singletons. No
group has been human-reviewed — the review sheet exists but is unfilled.
Novelty measures entity repetition among candidate groups, not whether the news
is new. Serial dependence between adjacent sessions is bounded by the embargo
but not modelled.

**Scope.** The target is the index, not individual equities. No transaction
costs, slippage, market impact, position sizing or capacity analysis has been
performed, and no trading strategy exists.

**Epistemic.** The retrospective study is exploratory by construction: the
corpus was collected and inspected before the protocol was written. That is
what §15 exists to fix.

---

## 15. Untouched future validation

`untouched_future_v1` — a test on data that did not exist when the rules were
written.

| Element | Value |
|---|---|
| Validation start | 2026-08-08 |
| First eligible reaction session | **2026-08-10** |
| Minimum sessions | 51 (40 train + 1 embargo + 10 test) |
| Minimum distinct outcomes | 51 |
| Minimum horizon | 120 days |
| Sealed | feature design, feature selection, model selection, hyperparameters, target, thresholds, success criteria |

Every observation is stamped `corpus_epoch`: `retrospective` before the
boundary, `untouched_future` after. The two are never pooled.

**The outcome side is sealed.** No accuracy, error or correlation is computed or
displayed for the untouched sample until both the sample-size and horizon
requirements are met. `database.record_future_readiness` *rejects* any report
carrying an outcome statistic. Until then the project reports readiness and data
quality only: sessions accumulated, eligible events, family coverage,
missingness, control availability.

This is not fastidiousness. Watching out-of-sample accuracy accumulate and
running the evaluation when it looks favourable is optional-stopping: it
inflates the false-positive rate and leaves **no trace** in the resulting
interval. The number looks exactly like a number obtained honestly. The only
defence is not to look, and the only reliable way not to look is to build
something that cannot show you.

**A failed future validation will be reported as a failed future validation.**
It does not license a revised protocol presented as the same test.

---

## Reproducing this

```bash
python -m scripts.demo_phase_a      # offline demo, no credentials
python -m scripts.verify_all --db finance_sentiment.db
```

The second command checks schema, frozen artifacts, historical integrity,
timing alignment, the full test suite and the demo outputs, and reports each
independently. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exactly which
components require credentials.
