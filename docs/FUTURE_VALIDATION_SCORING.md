# How untouched_future_v1 is run and scored

*Addendum written 2026-10-08, while the untouched sample was accumulating
(42 of 51 sessions, 61 of 120 days) and before any outcome on it had been
computed. Nothing here was chosen by looking at untouched data; all development
used synthetic sessions. The commit that adds this file is the timestamp.*

## Why an addendum was needed

Two gaps surfaced in a review on 2026-10-07:

1. **No runner existed.** `research/future_validation.py` could partition the
   data, but nothing fitted the sealed specifications on untouched sessions
   only. `scripts/run_validation.py` runs the retrospective study on the full
   dataset. Writing the runner after the sample became eligible would have meant
   making implementation choices with the outcome one command away.
2. **The scoring code was looser than the sealed text.** The retrospective
   comparison (`research.walkforward.compare_to_baselines`) pools all folds and
   declares success if any specification clears the margins on the pooled
   predictions. The sealed success criteria require a majority of folds, at
   least three fittable folds, intervals excluding the baseline, and a
   multiplicity correction.

The sealed text wins: it was frozen first. The retrospective code is left as it
is because its result is frozen. That result does not depend on the reading
either way: no retrospective news specification cleared both margins, so it is
"failure" under both.

## What runs

```bash
python -m scripts.run_future_validation --check   # sealed inputs + readiness, never reads an outcome
python -m scripts.run_future_validation           # the one real run
```

The runner refuses, in this order, before anything after the refusal can run:

1. **Sealed inputs.** The stored definition re-hashes to its key. The frozen
   retrospective artifact verifies. Its stored protocol specification hashes to
   the definition's `protocol_hash`. Today's code reproduces that specification
   and the definition hash.
2. **Already run.** One result per definition, in the append-only
   `future_validation_results` table. A second run is refused, not ignored.
3. **Readiness.** The same report the daily workflow records. Until every gate
   passes, no untouched target is read.

Then it fits the frozen specifications on untouched sessions only and scores
them with `research/future_scoring.py`. Specifications follow the
retrospective run exactly: same feature sets, same model per set, primary target
only, as the definition names. The result goes to the database and to
`docs/frozen/untouched_future_v1_result.json`.

**On the protocol hash.** Today's protocol specification hashes to `1f94f755…`,
not the sealed `d987de7b…`. The only difference is the dataset version label:
the retrospective run recorded `event-research-dataset-v2`, and the sealed
definition names `-v3` in `allowed_feature_versions`. Every rule is identical.
The runner checks exactly this: it swaps that one label back and requires the
sealed hash.

## How the criteria are read

Where the sealed wording admits more than one reading, the reading that makes
success harder was chosen. The same rules sit beside the code in
`research/future_scoring.py`.

| | Sealed phrase | Reading |
|---|---|---|
| R1 | "the margins" | Both pooled gains, against the best baseline on exactly the sessions the news specification predicted: MAE improvement ≥ 0.05 and directional-accuracy improvement ≥ 0.05. |
| R2 | "beats every baseline" | Beating the best baseline on each metric, chosen per metric on the matched sessions. |
| R3 | "in a majority of folds" | The margins are re-measured inside each fold and must be cleared in strictly more than half of the **fittable** folds: folds where at least one baseline produced predictions. A fittable fold the news specification missed counts against it. |
| R4 | "fewer than three folds are fittable" | Fewer than three fittable folds → inconclusive, whatever the margins show. |
| R5 | "session-cluster-aware intervals excluding the baseline" | Paired bootstrap intervals (2,000 resamples, sessions clustered on exit date) for the MAE gain and the hit-rate gain over the best baseline. Both lower bounds must be above zero. |
| R6 | multiplicity rule | Bonferroni: the intervals are two-sided at 1 − 0.05/m, where m is the number of news specifications fitted. |
| R7 | overlap of "failure" and "inconclusive" | See precedence below. |

**Precedence (R7):**

1. No fitted news specification, or no fitted baseline → **failure** (the
   sample gate blocks the comparison).
2. Fewer than three fittable folds → **inconclusive**.
3. Any news specification meets R1, R3, R4, R5 and R6 → **success**.
4. Any news specification clears the pooled margins → **inconclusive**: cleared
   overall, but not consistently or not beyond the corrected interval.
5. Otherwise → **failure**.

Step 4 is the reading of "margins cleared in fewer than a majority of folds".
Read literally, "fewer than a majority" includes zero folds, which would make
failure impossible. That cannot be the intent.

## A consequence to know before December

The first primary fold is never fittable. The one-session embargo leaves 39
training sessions against the 40-session minimum, which is also true in the
retrospective code. Three fittable folds therefore need **80 untouched sessions**
(folds 2 to 4: 50 + 10, 60 + 10, 70 + 10), not 51.

Sessions have been arriving one per trading day. That projects about 84 by the
120-day gate (around 2026-12-06), which is enough, but only just. If fewer than
80 sessions exist when the run happens, the verdict is inconclusive by
construction. Waiting for 80 sessions is not a protocol change: R4 makes the
verdict depend on it either way, and this note records it before any outcome.

## Reporting rule (sealed)

Failure and inconclusive results are reported in full. The protocol is not
re-run with different settings to obtain a different answer, and a failed
future validation does not license a revised protocol presented as the same
test.
