# Readiness decision

Audit of 2026-10-11, branch `audit/stock-research-readiness`, on the hardened
implementation (protocol v2,
`09e9382e77ec44bc06aaaf5c4bde0d64e06a6584d2aeb4b65ed35da0b2e7d99a`).

## Decision

**BLOCKED**

The first sealed real-data evaluation should not be run under protocol v2.
The registered H7 test does not hold its significance level on the sample it
will be applied to, and the price data contain errors the registered cleaning
rules do not catch. Both need a new protocol version. Separately, the
registered sample is too small to detect effects of plausible size, so the
new version should also expand it.

No real outcome was computed to reach this decision, and none exists in the
database.

## Why, in numbers

**1. The test (INFERENCE_CALIBRATION.md, I-1).** On the projected sample with
no effect in the data, the registered joint test rejects 10.5% of the time at
the 5% level (95% interval 9.7 to 11.4; 4,997 simulated samples) and 3.06% at
the 0.56% level a lone test must meet under Holm (2.58 to 3.54). The verdict
`supported` comes out for pure noise in 2.9% of samples, against an intended
0.56%. With heavier issuer recurrence the figure is 7 to 10%; with a tripled
sample 4 to 5%, because more small categories qualify. The implementation is
correct (it matches statsmodels to ten digits); the fault is an F reference
distribution with degrees of freedom from all 75 date clusters when the
smallest categories sit on about 24.

**2. The prices (PRICE_ADJUSTMENT_AUDIT.md, A-1, A-2).** On 16 dates the
provider's "close" is not the closing price for a large share of stocks: on
2023-01-05, inside the sample year, for 210 of 234. A CAR(0,+5) starting the
next day is wrong by a median of about 2 points for almost every stock at
once. Four capital increases carry a wrong adjustment factor and produce
returns of +13% to +43% on a market with a 10% limit; the cleaning rules
exempt them. Both were found by comparison with a second source across
104,535 returns.

**3. The power (POWER_ANALYSIS.md, W-1, W-2).** About 1,000 usable events on
74 dates in five categories. With a test that holds its level, 80% power
needs a spread of about 5% if spread evenly, a little over 3% if one large
category differs. At the registered material effect of 1% power is 1 to 3%.
Four of the five tested categories are residual or administrative.

**4. The sample is not there yet.** 19 of 80 anchors were ingested when this
was written, and the price snapshot covered 237 of the 404 issuer codes the
sample named.

Any one of points 1 and 2 could produce a false finding or corrupt a true
one. Point 3 means that even a correct analysis would most likely return
`inconclusive` and spend the sample doing so.

### Why not "expand data first"

Because expansion alone does not repair the test: at 240 anchors it rejects a
true null 14.8% of the time at 5% and 4.84% at the Holm level. And why not
"ready with limitations": a limitation narrows what a result means; a test
that rejects five times too often changes whether there is a result.

### What held up

| Checked | Result |
| --- | --- |
| Two-way clustered covariance and Wald test | match statsmodels and a brute-force reference |
| The earlier 6.7% null rejection | Monte Carlo noise: 4.84% in 5,000 runs, interval coverage 95.2% |
| Market model: window, minimum history, strictly pre-event | correct |
| Benchmark series | agrees with a second source on 439 of 439 sessions |
| Dividend adjustment | internally exact on 204 of 205 dates; 198 of 204 within 1% of the second source |
| Plain bonus issues | 47 of 51 within 1% |
| 2023 sessions, holidays, cancelled session | match the official calendar |
| Day-0 assignment at every boundary | correct; errors are on the conservative side |
| KAP retrieval so far | 826 of 826 target details, no refusal |
| v1 and v2 protocol hashes | reproduce from their commits |
| Append-only tables, frozen index artifact | intact |
| Synthetic results | byte-identical across runs and processes |
| Results stored under any protocol | none |

## All findings

| Id | Severity | Finding | Amendment needed |
| --- | --- | --- | --- |
| I-1 | critical | Registered H7 test over-rejects under the null | yes |
| A-1 | high | Provider closes are not closes on 16 dates, one pair in 2023 | yes |
| W-1 | high | Sample cannot detect spreads below about 3 to 5% | yes (frame) |
| W-2 | high | Tested categories are mostly residual ones | yes, if redefined |
| A-2 | medium | Four wrong capital-increase factors escape cleaning | yes |
| I-2 | medium | A category can qualify on a handful of issuers | yes |
| K-1 | medium | 20% of sampled disclosures have no priced ticker, unevenly by category | no |
| K-2 | medium | Nothing checks price coverage of sampled issuers before a run | yes (gate) |
| C-1 | medium | Filings during the February 2023 closure are dated to the reopening day | yes, if excluded |
| P-1 | medium | The run gate can be walked around through the hypothesis runners | yes |
| M-1 | medium | Benchmark explains about 8% of a typical stock's variance | only if changed |
| M-2 | medium | A category's mean CAR is not a clean abnormal return | only if changed |
| W-3 | medium | The 1% materiality condition does not filter | yes |
| K-3 | low | A refused detail is asked for once per ingestion run | no |
| K-4 | low | Roster of listed companies is today's | no |
| C-2, C-3, C-4 | low | Half days of 2023 are inferred, not recorded | yes, if changed |
| P-2 | low | Data were in the database before registration | no |
| P-3 | low | Three modules read constants outside the fingerprint | yes |
| P-4, P-5 | low | Library versions not in the manifest; snapshot contents grow | partly |
| A-3, A-4 | low | Rights issues unflagged; six dividend amounts differ | no |
| M-3, M-4, M-5 | low | Price index against total-return stocks; beta noise; issuer's earlier events in the estimation window | no |
| I-4, P-6, P-7, C-5 | informational | | no |

## Proposed protocol v3 (not implemented, not registered)

This is a proposal for approval. Nothing below exists in code. v2 stays
registered and unrun; v3 would supersede it the way v2 superseded v1, and
would say that it was written without sight of any outcome and on the
evidence of simulations.

1. **Primary test for H7.** Replace the F reference with a wild cluster
   bootstrap by day 0, null imposed, Rademacher weights, a fixed number of
   draws (9,999) and a fixed seed; p = (1 + draws at least as extreme) /
   (1 + draws). Keep the two-way F test as a reported sensitivity. CR2 with
   Satterthwaite degrees of freedom is the alternative to evaluate beside it.
2. **Calibrate before unsealing.** On the final sample structure, with no
   return read: null rejection at 5% and at the Holm level within Monte Carlo
   error of nominal, including under the issuer-recurrence stress. Registered
   as a precondition of the run.
3. **Category rule.** Add a minimum number of distinct issuers per category
   and raise the minimum number of dates, at values fixed by step 2. Name the
   categories of interest in advance and pool residual forms as the
   reference.
4. **Sample.** A census of 2023 material-event disclosures by listed
   companies (about 15,100 targets, about 4,500 usable events on about 220
   dates, about 42 more hours of ingestion), under a new frame version.
   Complete retrieval required; any exception listed by ID with three
   attempts on three days.
5. **Price rules.** A second source for every stock; a return is missing
   where the two sources differ by more than one percentage point; the
   price-limit test applies to the adjusted return on corporate-action dates.
6. **Eligibility.** Exclude events whose day 0 is the reopening of 15
   February 2023 for filings made before it; record the 2023 half days
   explicitly.
7. **Gates and provenance.** The registration, completeness and
   price-coverage checks move into the construction of a real-data context;
   constants read from the repository-root `config.py` enter the protocol
   document; library versions enter the manifest.
8. **Estimand.** State the size of an effect as named contrasts or a
   bias-corrected spread, not as a raw maximum minus minimum.

## Next steps before any real-data run

1. Do not run `python -m stock_research.cli run`. Under v2 it would refuse
   today (the frame is incomplete), and once the frame completes it would
   execute a test this audit found to be miscalibrated.
2. Let the ingestion finish; it is healthy and was not touched. If any
   refusal is logged, run `ingest-kap` again on two later days.
3. Run `ingest-prices` after ingestion ends, and confirm that every issuer
   code in the sample has an availability row.
4. Count, from metadata only, sampled events with day 0 on 2023-01-05 or
   2023-01-06, and events of DMSAS, HEKTS, MAGEN and NATEN.
5. Copy the database and record the copy's SHA-256.
6. Decide on v3: the estimator, the price rules and the expansion each need
   approval, the last because it changes the registered sampling frame.
7. If approved: implement on a new branch, rerun this audit's calibration and
   power simulations on the final structure, register v3, push the
   registration, and only then run.
