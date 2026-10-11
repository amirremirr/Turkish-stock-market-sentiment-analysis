# Statistical power of H7

Audit output of 2026-10-11. **Every probability in this document comes from
synthetic returns.** What is taken from the real sample is its structure only:
which issuer filed in which category on which session. No return of a sampled
event was read, and the simulation's noise parameters come from 2022, before
the first sampled disclosure.

Reproduce: `python -m readiness_audit.make_structure` (copy of the database,
structure, 2022 calibration), `python -m readiness_audit.run_simulations`
(all simulations), `python -m readiness_audit.figures`,
`python -m readiness_audit.tables`. Full results: `POWER_RESULTS.csv` (364
rows), `power_designs.json`, `calibration_power_comparison*.csv`.

## Answer

* The registered 80-anchor frame will give H7 about **1,000 usable events on
  about 74 dates from about 470 issuers, in five categories**, four of which
  are residual or administrative.
* With a test that holds its level, **80% power needs a true spread of about
  5%** in CAR(0,+5) if the differences are spread evenly over categories, and
  **a little over 3%** if one large category alone differs. At the registered
  material effect of 1% the power is 1 to 3%.
* The registered test shows more power than that (80% at 4.5% and 2.7%), but
  about half of it is its inflated level (INFERENCE_CALIBRATION.md).
* The categories with the clearest economic content (dividend, bonus issue,
  rights issue, insider transaction, asset purchase) will not reach the
  30-event minimum and will not be tested at all.
* **The sample is not adequate** for effects of the size such events usually
  show (one or two percent over a week). A census of the year's
  material-event disclosures, about 4.3 times the sample and about 42 more
  hours of ingestion, brings the detectable spread to about 1.7% for a large
  category and puts ten categories in the test.

## 1. The sample H7 will have

From the copy taken at 11:59 with 14 anchors complete and one partly ingested
(646 disclosures):

| Step of the registered eligibility rule | Remaining |
| --- | --- |
| Sampled disclosures | 646 |
| Issuer resolved to a priced ticker | 516 |
| Not an update or correction, timing unambiguous | 461 |
| One row per issuer and day 0 | 386 |
| One category on that row | 358 |
| Not a price-triggered category | 349 |
| Published before the open, after the close or on a non-session day | 202 |
| Prices and a 60-session estimation window available | **198** |

Three in ten sampled disclosures are usable. The largest loss is timing:
filings made during the session are outside the primary sample, and an anchor
is fifty consecutive filings, so an anchor that falls in trading hours
contributes nothing. Two of the 14 complete anchors, and the partial one so
far, yield no usable event.

Usable events by category: general_other 69, buyback 56, other_form 30,
administrative 18, debt_issuance 6, new_contract 5, asset_acquisition 4,
rights_issue 3, insider_trade 3, dividend 2, capital_change_other 1,
bonus_issue 1. They fall on 13 dates and come from 140 issuers; the busiest
issuer has 8.

### Projection to the full frame

Each anchor not yet ingested is given a session by interpolating its number
against the sessions of the anchors already seen, and a set of events by
drawing one of the 14 complete anchors, empty ones included. Issuers of new
anchors are drawn so that the share of first-time filers in each category
matches what has been seen (a Good–Turing estimate) and frequent filers stay
frequent. Twenty projections are drawn per analysis and averaged over.

| | 80 anchors (registered) | 160 | 240 | 345 (about a census) |
| --- | --- | --- | --- | --- |
| Usable events | 1,013 | 2,086 | 3,252 | 4,502 |
| Dates | 74 | 138 | 192 | 221 |
| Issuers | 466 | 800 | 1,111 | 1,393 |
| Categories meeting 30 events on 15 dates | 5.2 | 8.8 | 9.9 | 10.4 |
| general_other | 382 | 737 | 1,140 | 1,567 |
| buyback | 315 | 590 | 899 | 1,221 |
| other_form | 170 | 326 | 503 | 707 |
| administrative | 105 | 205 | 309 | 429 |
| debt_issuance | 38 (in 65% of projections) | 69 | 107 | 144 |
| new_contract | 33 (in 45%) | 53 | 84 | 118 |
| asset_acquisition | 30 (in 5%) | 44 | 69 | 102 |
| rights_issue | not tested | 35 | 55 | 75 |
| insider_trade | not tested | 37 | 54 | 75 |
| dividend | not tested | 33 | 38 | 51 |
| bonus_issue | not tested | not tested | not tested | 32 |

A category's size is its mean over the projections in which it qualifies. The
projection rests on 14 anchors; its own uncertainty is visible in the range of
the 80-anchor total across projections, 865 to 1,154.

**Events are not independent observations.** About 14 events share each date,
and 45% of the usable events seen so far come from issuers with more than one. The simulation carries
both: every event of one issuer is a sum over the same simulated days, so
overlapping windows share returns, and all stocks share a daily market-wide
shock.

## 2. The noise

Calibrated on market-model residuals of 229 stocks, 2022-06-01 to 2022-12-30:

| Quantity | Value | Use |
| --- | --- | --- |
| Daily residual standard deviation, 10th / 50th / 90th percentile | 2.35% / 3.03% / 4.02% | each issuer keeps its own; new issuers draw from the pool |
| Share of daily residual variance common to all stocks | 4.7% | market-wide shock |
| First-order serial correlation of residuals | 0.08 | AR(1) |
| Excess kurtosis | 1.17 | Student t with 9 degrees of freedom |

A CAR(0,+5) of the median stock therefore has a standard deviation of about
7.6%. The intercept is estimated on the registered window in every simulated
event, so its error is included. Beta is not re-estimated; that omission
makes the figures slightly optimistic.

## 3. The test and the scenarios

The test is the registered one, called directly: OLS on category indicators,
two-way clustered covariance, joint F test, Holm with a family of nine (a
lone p-value must be below 0.05/9 = 0.0056), and the verdict `supported` only
if the observed spread is also at least 1%. Two shapes of alternative, each
with a true spread between the highest and lowest category mean of 0, 0.5, 1,
1.5, 2, 3 and 5%:

* **evenly spaced:** category means spread evenly from −half to +half of the
  spread;
* **one category differs:** one category's mean is shifted by the spread, the
  rest are equal.

2,000 simulated samples per row for the registered frame (1,000 for the census
size), with the same noise reused across effects and scenarios. The standard
error of a cell is at most 1.1 points (1.6 for 1,000 runs); it is in
`POWER_RESULTS.csv` for every cell. Variation between projections of the
sample is not in that figure and is of similar size.

## 4. Results on the registered frame

### Under the null

| | Registered test | Intended |
| --- | --- | --- |
| Rejects at 5% | 10.4% | 5% |
| Rejects at the Holm level | 2.9% | 0.56% |
| Verdict `supported` | 2.9% | at most 0.56% |
| Observed spread at least 1% | 85% | |

The registered test does not hold its level, and the 1% materiality condition
is met by noise alone in most samples. INFERENCE_CALIBRATION.md has the
analysis. The power figures for the registered test below must be read with
that in mind.

### Registered test: probability of the verdict `supported`

| Where the effect sits | 0.0% | 0.5% | 1.0% | 1.5% | 2.0% | 3.0% | 5.0% | 80% power at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| evenly spaced over all categories | 2.9% | 3.8% | 6.0% | 11.2% | 19.5% | 46.1% | 92.5% | 4.5% |
| only buyback differs | 2.9% | 4.3% | 11.0% | 27.1% | 51.5% | 90.8% | 99.5% | 2.7% |
| only general_other differs | 2.9% | 3.9% | 10.2% | 26.8% | 52.8% | 93.5% | 99.5% | 2.7% |
| only other_form differs | 2.9% | 3.9% | 7.0% | 16.1% | 32.9% | 73.6% | 99.4% | 3.5% |
| only administrative differs | 2.9% | 3.4% | 5.9% | 11.3% | 21.1% | 52.4% | 96.5% | 4.2% |
| only debt_issuance differs | 3.5% | 4.0% | 4.8% | 6.3% | 9.2% | 20.2% | 55.2% | above 5% |
| only new_contract differs (900 runs) | 4.4% | 6.0% | 6.9% | 9.0% | 13.0% | 24.0% | 58.1% | above 5% |

### Nominal against Holm, registered test

| Where the effect sits | Level | 1.0% | 2.0% | 3.0% | 5.0% | 80% power at |
| --- | --- | --- | --- | --- | --- | --- |
| evenly spaced | p < 0.05 | 18.8% | 42.4% | 73.4% | 98.0% | 3.5% |
| evenly spaced | Holm over nine | 6.0% | 19.5% | 46.1% | 92.5% | 4.5% |
| only buyback | p < 0.05 | 31.3% | 77.5% | 97.5% | 99.5% | 2.1% |
| only buyback | Holm over nine | 11.0% | 51.5% | 90.8% | 99.5% | 2.7% |
| only administrative | p < 0.05 | 19.8% | 46.9% | 79.2% | 99.4% | 3.1% |
| only administrative | Holm over nine | 5.9% | 21.1% | 52.4% | 96.5% | 4.2% |

Correcting for the family of nine raises the detectable spread by about one
percentage point.

### With a test that holds its level

The wild cluster bootstrap by date rejects a true null 4.3% of the time at 5%
and 0.40% at the Holm level on this design. On the same simulated samples
(600 runs, 499 draws, standard errors 1 to 2 points):

| Planted effect | Registered, Holm level | Wild bootstrap, Holm level |
| --- | --- | --- |
| Evenly spaced, 2% | 18.8% | 9.0% |
| Evenly spaced, 3% | 49.3% | 22.2% |
| Evenly spaced, 5% | 91.5% | 80.2% |
| Only buyback, 1.5% | 23.3% | 10.5% |
| Only buyback, 2% | 45.8% | 24.7% |
| Only buyback, 3% | 87.8% | 72.3% |

**Minimum detectable spread at 80% power and 5% family-wise error, registered
sample structure: about 5% if evenly spaced; a little over 3% if it sits in
buyback or general_other alone; larger for every other category.**

The audit's degrees-of-freedom correction of the registered statistic
(columns `power_corrected_*` in the results file) agrees at low and moderate
effects. It yields no p-value in about 12% of projected samples, where some
category is carried by fewer issuers than the test has restrictions, so its
power cannot exceed 88%; those columns are a lower bound.

### A joint difference is not a particular category's drift

The joint test answers "do categories differ". Whether a *given* category's
mean CAR differs from zero is the exploratory per-category test (two-way
clustered mean, 5% level, reported among about 35 category-by-window cells
with a false-discovery adjustment). Its power at the unadjusted 5% level:

| Category | Null | 1.0% | 1.5% | 2.0% | 3.0% | 5.0% | 80% power at |
| --- | --- | --- | --- | --- | --- | --- | --- |
| general_other | 5.8% | 48.9% | 81.8% | 97.1% | 100% | 100% | 1.5% |
| buyback | 7.1% | 44.1% | 75.6% | 91.8% | 99.8% | 100% | 1.6% |
| other_form | 5.6% | 30.3% | 55.8% | 79.4% | 98.1% | 100% | 2.0% |
| administrative | 6.1% | 20.6% | 39.0% | 60.7% | 91.2% | 100% | 2.6% |
| debt_issuance | 6.2% | 11.5% | 19.1% | 28.4% | 49.6% | 88.9% | 4.5% |
| new_contract (900 runs) | 4.7% | 9.0% | 13.9% | 21.0% | 35.7% | 58.3% | above 5% |

Three things separate this table from the joint one. It is not corrected for
the number of cells it sits among; with the adjustment a single real effect
needs a p-value near 0.001, not 0.05. It is exploratory by registration and
cannot support a claim. And a category mean is a level, which the benchmark
does not pin down (MARKET_MODEL_AUDIT.md, M-2). A 2% drift after buybacks
would very probably show in its own row (92%) and only even odds in the
registered verdict (52%, of which half is the inflated level).

## 5. Sensitivity to dependence

Registered test, verdict `supported`, 80 anchors, 2,000 runs each.

| Variant | Null | Evenly spaced: 80% power at | Only buyback: 80% power at | Only administrative: 80% power at |
| --- | --- | --- | --- | --- |
| Calibrated (common share 4.7%, serial correlation 0.08) | 2.9% | 4.5% | 2.7% | 4.2% |
| Common daily share 0% | 2.9% | 4.4% | 2.6% | 4.2% |
| Common daily share 10% | 3.0% | 4.5% | 2.8% | 4.2% |
| Common daily share 20% | 2.6% | 4.6% | 2.9% | 4.2% |
| Serial correlation 0.25 | 2.9% | 4.9% | 3.2% | 4.6% |
| No issuer files twice | 1.8% | 4.5% | 2.8% | 4.2% |
| Issuers recur as in the observed anchors | 10.4% | 4.5% | 2.8% | 4.3% |
| Event-window volatility 1.5 times ordinary | 2.7% | above 5% | 4.3% | above 5% |
| Event-window volatility 2 times ordinary | 2.7% | above 5% | above 5% | above 5% |

The common daily share is the intracluster correlation of CARs within a date;
across the range 0 to 20% it moves the detectable spread by 0.3 points,
because the test compares categories within the same dates and a shock common
to a date largely cancels. Issuer recurrence does not change power much but
decides whether the test holds its level: with heavy recurrence it rejects a
true null 10.4% of the time at the Holm level. What moves power is the
volatility of the event window. Filing days are more volatile than ordinary
days; if by half, the detectable spread for buyback is 4.3% even with the
registered test.

## 6. Larger samples

Registered test, verdict `supported` (2,000 runs; 1,000 for 345 anchors):

| Sample | Null | Evenly spaced | Only buyback | Only administrative | Only new_contract |
| --- | --- | --- | --- | --- | --- |
| 80 anchors (registered) | 2.9% | 4.5% | 2.7% | 4.2% | above 5% |
| 160 anchors (double) | 5.1% | 4.2% | 2.1% | 3.0% | above 5% |
| 240 anchors (triple) | 4.2% | 3.6% | 1.7% | 2.6% | 4.7% |
| 345 anchors (about a census) | 1.8% | 2.9% | 1.5% | 2.2% | 4.2% |

Cells are the spread at which power reaches 80%. "Evenly spaced" improves
slowly because more categories qualify as the sample grows, so the same
spread is divided among ten small differences instead of five.

Wild bootstrap at the Holm level, 300 runs and 299 draws per cell (standard
errors up to 3 points; with 299 draws the bootstrap can only reject at an
effective level of 0.33%, so these are slightly low):

| Planted effect | 80 anchors | 240 anchors | 345 anchors |
| --- | --- | --- | --- |
| Only buyback, 1% | | 6.7% | 23.3% |
| Only buyback, 1.5% | 10.5% | 31.0% | 65.7% |
| Only buyback, 2% | 24.7% | 73.3% | 94.7% |
| Only buyback, 3% | 72.3% | | |
| Only administrative, 2% | | 27.0% | 58.7% |
| Only administrative, 3% | | 78.3% | 97.7% |
| Evenly spaced, 3% | 22.2% | 42.7% | 67.3% |

**Benefit of expansion, with a test that holds its level:** the spread
detectable with 80% power in a large category falls from a little over 3%
(80 anchors) to about 2.1% (triple) and about 1.7% (census); for a mid-sized
category such as administrative, from above 5% to about 3% and about 2.5%.
Doubling buys about a third of the way. Even a census does not reach the
registered material effect of 1%: its power there is about one in four for
the largest category.

Cost: the gateway allows six calls a minute, about 9.4 minutes per anchor.
Tripling is 160 more anchors, about 25 hours; a census is about 265 more,
about 42 hours. The whole range holds an estimated 15,100 target disclosures
(0.108 per disclosure number over 139,317 numbers), of which the registered
frame takes 23%.

Beyond 2023 there is nothing to expand into on the current access: the
development gateway serves 2023 only.

## Findings

### W-1. The registered sample cannot detect effects of plausible size

* **Severity:** high.
* **Evidence:** sections 4 and 6.
* **Source:** the frame, `stock_research/config.py:34-36` (80 anchors of 50),
  and the timing rule that removes during-session filings
  (`stock_research/hypotheses/h7.py:93-94`).
* **How to verify:** `python -m readiness_audit.run_simulations power_main`.
* **Impact:** an `inconclusive` verdict would say almost nothing about
  spreads below 3%, and the sample is spent once it is looked at.
* **Recommended action:** expand to a census of 2023 before any outcome is
  read.
* **Protocol amendment:** yes (the sampling frame is registered; changing it
  needs approval and a new frame version).

### W-2. The categories that will be tested are mostly the uninformative ones

* **Severity:** high for the meaning of a result.
* **Evidence:** section 1. Of about 1,000 usable events, about 550 are in
  `general_other` (a general form in which no phrase matched) and
  `other_form` (a form name outside the taxonomy), and 105 are
  administrative. Buyback is the one large category with an economic
  reading. Dividend, bonus issue, rights issue, insider transaction and asset
  purchase stay below 30 events.
* **Impact:** a significant joint test would most likely mean "buybacks
  differ from residual filings", or a difference between two residual
  classes that has no interpretation.
* **Recommended action:** with the census, ten or more categories qualify. In
  a future protocol, name in advance the categories of interest and pool the
  residual ones as the reference.
* **Protocol amendment:** yes, if the test's categories are redefined.

### W-3. The materiality condition does not filter

* **Severity:** medium.
* **Evidence:** under the null the observed spread is at least 1% in 85% of
  samples (92 to 95% when a small category is present).
* **Source:** `stock_research/hypotheses/common.py:170-171`; the registry
  notes the upward bias of a maximum minus a minimum in H7's limitations.
* **Impact:** the verdict rests on the p-value alone, and an observed spread
  must not be reported as an effect size without that caveat.
* **Recommended action:** in a future protocol, state the estimand as a
  bias-corrected spread or as named contrasts.
* **Protocol amendment:** yes.

### W-4. Limits of this power analysis

* **Severity:** informational.
* The projection is from 14 of 80 anchors; the final counts will differ, and
  the calibration and power runs should be repeated on the final structure
  (no return is needed for that).
* Noise is calibrated on 2022. If 2023 residuals were more volatile, power is
  lower than shown.
* No beta estimation error, no wrong prices, no missing returns inside
  windows: each would lower power a little.
* Event-window volatility is taken as ordinary-day volatility in the main
  runs; section 5 shows what a higher level does.

## Figures

`figures/power_curves.png`, `figures/nominal_vs_holm.png`,
`figures/power_by_sample_size.png`, `figures/null_rejection_rates.png`.
