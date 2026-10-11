# Protocol v3: a validated candidate

Branch `research/stock-protocol-v3`. Status: **candidate, not registered.**
No real-data hypothesis has been run under it, no event-window CAR has been
computed, the census has not been started, and nothing is merged.

It exists because the readiness audit returned BLOCKED for v2
(`../readiness/READINESS_DECISION.md`). Each correction that audit proposed
was checked again here before being built; one was changed (the reference
group, item 4 of the approval list) and one proposal was refuted (a
date-level bootstrap alone, which the audit had found adequate on a design
without overlapping issuer windows).

## Read in this order

| File | Content |
| --- | --- |
| [APPROVALS_REQUIRED.md](APPROVALS_REQUIRED.md) | every decision that needs approval before registration, and the steps after it |
| [CHANGES_FROM_V2.md](CHANGES_FROM_V2.md) | all 28 rule changes: v2, v3, reason, code location |
| [VALIDATION.md](VALIDATION.md) | simulation evidence for the test, power, the verdict rule; two-source price validation; what was not validated |
| [CENSUS_PLAN.md](CENSUS_PLAN.md) | the census frame, what it reuses, its cost, how to start it |
| `validation/*.json` | the numbers behind the tables |
| `../HYPOTHESIS_REGISTRY.md` | generated specifications of the candidate |

## In brief

* **Test.** H7's joint p-value comes from a wild cluster bootstrap of the
  two-way clustered Wald statistic. Across eleven synthetic designs it rejects
  a true null 4.4–5.3% at the 5% level and 0.43–0.70% at the Holm level
  (intended 0.56%). The v2 test: 6.9–11.2% and 1.0–2.8%. A bootstrap or CR2
  correction that looks at dates only rejects 22% and 9% when issuers file on
  consecutive sessions.
* **Hypothesis.** Eight named classes of filing, each contrasted with
  procedural filings; an issuer's overlapping events count once; a class needs
  30 events on 20 dates from 20 issuers.
* **Prices.** A return is used only if two providers agree on it within one
  point. On the current data that removes 1,611 returns, 1,603 of them on the
  sixteen dates where the first provider's close is not a close, and six wrong
  corporate-action factors.
* **Calendar.** The 2023 schedule is recorded, not inferred, and checked
  against the data.
* **Sample.** A census of 2023, reusing every stored row untouched.
* **Gates.** No hypothesis computes an outcome on real data until
  registration, completeness, verification, price coverage from both sources
  and the calendar check have all passed.
* **Power, honestly.** The census detects every class differing by 2% (91%)
  or buyback alone by 3% (68%); a single small class at 3% only one time in
  four.

## Tests

`tests/stock_research/test_protocol_v3.py` (36 tests) plus the updated
existing files. Results of the last full run are in the commit message and
the final report.

## What was left untouched

The research database (`stock_research.db`) was opened read-only to be copied;
all v3 tables, second-source data and simulations live in a copy inside this
worktree's ignored output directory. The v1 and v2 rows of `sr_protocols`,
the empty result tables, the frozen index artifact and the running anchor
ingestion are as they were.
