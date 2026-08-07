# The frozen walk-forward protocol

A predictive protocol fixed and hashed before comparative results were read.
`research/protocol.py` is the specification; the hash is the freeze.

```
protocol_version  walk-forward-protocol-v1
status            retrospective_walk_forward_exploration
```

## What this is, and what it is not

**It is** a chronological walk-forward evaluation. Folds run forward in time,
no training fold postdates its test fold, preprocessing is fitted on training
folds only, and no feature was selected using test results.

**It is not** an untouched future test. This corpus was already collected,
already inspected, and already used to build the features being evaluated. A
genuinely untouched test needs data that did not exist when the protocol was
written. Every result carries the `retrospective_walk_forward_exploration`
label for that reason, in the database, the report and the dashboard.

**It is not a trading strategy.** No transaction costs, no position sizing, no
execution model, no recommendation.

## The statistical unit

The target is the BIST 100 index return. Every candidate event whose first
reactable session is the same day is scored against **the same number**.

On the current corpus: **731 event rows, 49 distinct sessions, 49 distinct
outcomes** — a duplication factor of 14.9. Treating event rows as independent
would shrink every standard error by roughly √14.9 ≈ 3.9× and turn a null into
a discovery.

So the primary unit is **one row per (first reactable session, target window)**.
Event features are aggregated onto the session by rules fixed in advance in
`research/modelling_unit.py` (`AGGREGATION_RULES`), not chosen after seeing
which aggregation helps. `build_session_units` raises if two events on one
session carry different targets — that would mean two windows were mixed.

Event-level rows are retained as a declared sensitivity, to be read with
session-cluster-aware inference.

## Frozen choices

| Element | Value |
|---|---|
| Primary target | `reactable_open_to_close`, `raw_return` |
| Secondary targets | `residual_em_lagged`, `residual_em_oil_fx_lagged`, and the two descriptive gap windows |
| Eligible buckets | `pre_open`, `post_close`, `weekend_or_holiday` |
| Excluded buckets | `during_session`, `unknown` |
| Market recap | excluded (its tone follows the return by construction) |
| Timing conflicts | excluded from primary, retained descriptively |
| Control history | rolling 60 prior sessions, minimum 30 observations |
| Feature sets | 6 baselines, 6 news sets, enumerated in advance |
| Models | training mean, majority direction, ridge (α = 1, standardised), logistic (L2 = 1, 200 iterations, η = 0.1) |
| Fold design | expanding-window, chronological, 1-session embargo |
| Missing values | drop the row for that specification; **never impute zero** |
| Metrics | MAE, RMSE, Pearson r, R² vs reference, directional and balanced accuracy, Brier, calibration |
| Uncertainty | session-cluster bootstrap, 2 000 resamples, fixed seed |
| Direction threshold | 0.0 |
| MAE margin over best baseline | 0.05 |
| Directional margin over best baseline | 0.05 |
| α | 0.05, with explicit multiplicity accounting |

`residual_none` is deliberately **not** a secondary target: for the empty
control set the residual *is* the raw return, so running it would add a
duplicate to the multiplicity count and no information.

### Fold geometry

Two geometries, both frozen, selected by **session count alone**:

| Geometry | Applies when | Train | Test | Step | Can declare success |
|---|---|---|---|---|---|
| `primary` | ≥ 51 sessions | 40 | 10 | 10 | yes |
| `reduced` | 32–50 sessions | 25 | 6 | 6 | **no** |
| none | < 32 sessions | — | — | — | everything `insufficient_sample` |

Each threshold is **derived** from its own geometry —
`initial_train + embargo + test` — never written down beside it. The first
production run selected `primary` at exactly 50 sessions against a hand-written
threshold of 50, then blocked all 72 specifications because that geometry needs
51. A geometry declared to apply where it cannot produce a single fold is
internally inconsistent regardless of any result, so the threshold is now
computed and a test fits every geometry at its own boundary.

Sample size is a property of data collection, knowable and known before any
target was read, so indexing the geometry on it is not selecting on a result.
The `reduced` geometry gives 25 training sessions against up to six features —
roughly four observations per parameter. It exists so a small sample produces a
stated null rather than an empty report, and it is **barred in advance from
declaring success**; its verdict is capped at `inconclusive`.

*Disclosure:* the geometry rule was written after the session count (49) was
measured and before any target, metric or model output was inspected.

## The sample-size gate

Runs **before** any model touches the data. A specification with too few
complete observations is not fitted; it is recorded as `insufficient_sample`
with the exact binding requirement.

Loosening a rule to raise coverage is not available. In particular the
30-observation minimum on rolling control estimation stays: it protects the
residuals themselves, and weakening it to fit more models would corrupt the
targets in order to have more of them.

## Prohibited

Recorded in the specification, and asserted by tests:

- random train/test splitting
- full-sample normalisation
- fitting preprocessing on a test fold
- feature selection using test results
- contemporaneous controls in a tradable specification
- reporting significance without accounting for repeated sessions
- transaction-cost or trading-strategy evaluation

## Success, failure, inconclusive

- **Success** — a news feature set beats every baseline on MAE *and* directional
  accuracy by the stated margins, in a majority of folds, with session-cluster
  intervals excluding the baseline.
- **Failure** — no news feature set clears the margins, or the gate blocks the
  comparison.
- **Inconclusive** — margins cleared in fewer than a majority of folds, fewer
  than three fittable folds, or the `reduced` geometry is in force.

Failure and inconclusive results are reported in full. The protocol is not
re-run with different settings to obtain a different answer.

## Comparisons are session-matched

Specifications differ in coverage: a feature with more missing values predicts
fewer sessions. Comparing its MAE against a baseline scored on a larger,
different set compares samples rather than models. Every baseline is therefore
**re-scored on exactly the sessions the news specification predicted** before
any margin is computed.

MAE is also not comparable *across windows*: an overnight gap is a smaller
number than a full session's range, so lower error there reflects a narrower
target, not a better model.

## Provenance

Stored per run in `validation_protocols`, `validation_runs`,
`validation_results` and `validation_predictions`:

protocol JSON · protocol hash · code commit · database snapshot SHA-256 ·
dataset version · feature version · target version · modelling-unit version ·
timing-rule version · return-window version.

Every out-of-sample prediction is stored, so a reported metric can be recomputed
rather than trusted.

## Running it

```bash
python -m scripts.run_validation --db finance_sentiment.db --report validation.md
```

The hash is recomputed and compared on every run. A changed hash means the
protocol moved, and results across the two are results from two different
studies.
