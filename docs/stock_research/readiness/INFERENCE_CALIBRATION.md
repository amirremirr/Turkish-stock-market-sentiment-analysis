# Inference calibration

Audit output of 2026-10-11. Every number here is a rejection rate on
**synthetic returns with no effect in them**, or with an effect planted by
hand. No real return was used. The registered estimator was called as it is;
the alternatives live in `readiness_audit/calibration.py`, outside the
registered package, and were not swapped in.

Reproduce: `python -m readiness_audit.run_simulations mean calibration
small_category wild_null wild_power`. Results:
`calibration_results.json`, `calibration_wild_bootstrap.json`,
`calibration_small_category.json`, `calibration_power_comparison.csv`,
`mean_test_calibration.json` in this directory.

## Summary

1. **The covariance is implemented correctly.** It equals a brute-force
   reference and statsmodels' two-way clustered covariance to ten digits, and
   the Wald statistic, degrees of freedom and p-value match as well.
2. **The 6.7% reported by the development audit was Monte Carlo noise.** The
   same design run 5,000 times rejects 4.84% of the time.
3. **The registered H7 joint test does not hold its level on the sample H7
   will actually have.** With no effect in the data it rejects about 10% of
   the time at the 5% level and about 3% of the time at the level a lone test
   must meet under Holm (0.56%), five to six times too often. A verdict of
   `supported` under the null comes out in 2.9% of runs.
4. The cause is the reference distribution, not the variance: F with
   degrees of freedom from *all* date clusters, on a design where the
   smallest categories sit on a quarter of them.
5. A wild cluster bootstrap by date holds the level (4.3% and 0.40%). Its
   power is well below what the registered test appears to have.

## 1. Is the implementation right?

`stats.ols_cluster` (`stock_research/stats.py:39-106`) against two references
on a random design with 40 date clusters, 70 issuer clusters and three
category indicators:

| Reference | Largest relative difference in the covariance |
| --- | --- |
| Textbook sum over clusters, written out in the test | below 1e-10 |
| statsmodels 0.14.6, `cov_type="cluster"` with two groups | 1.9e-14 |

`stats.wald` (`stats.py:109-126`) against statsmodels' `wald_test` with an F
reference: same statistic (0.88336), same denominator degrees of freedom
(39), same p-value. Tests:
`test_two_way_covariance_matches_a_brute_force_reference`,
`test_two_way_covariance_and_wald_match_statsmodels`.

## 2. The earlier 6.7%

The development audit's null simulation (twelve issuers, ten filings each on
nearby sessions, CAR(+1,+20), 300 runs) gave 6.7%. With 300 runs the standard
error of that figure is 1.4 points, so 6.7% is 1.2 standard errors from 5%.
The same design run 5,000 times:

| Procedure | Rejection at 5% | 95% interval |
| --- | --- | --- |
| Date clusters only | 49.0% | 47.6% to 50.3% |
| Two-way (date and issuer), as registered | 4.84% | 4.25% to 5.43% |
| Issuer clusters only | 4.70% | 4.11% to 5.29% |

Coverage of the two-way 95% interval for the mean: 95.2%. Independent
clusters: 12 issuers, about 90 dates; the binding count is 12 and the t
reference uses 11 degrees of freedom. **Monte Carlo noise; the mean test is
calibrated on that design.** Date-only clustering fails as badly as the
development audit said.

## 3. The joint test on the projected H7 sample

Designs are the full frame as projected from the ingested anchors (see
POWER_ANALYSIS.md for how), returns are synthetic with volatility, common
shocks, serial correlation and tails calibrated on 2022. "Registered" is
`stats.ols_cluster` with two-way clustering followed by `stats.wald`, exactly
the calls `h7._joint` makes (`stock_research/hypotheses/h7.py:99-112`; test
`test_audit_joint_test_is_the_registered_h7_test`).

| Design | Categories | Smallest category | Fewest dates in a category | Runs | Registered, at 5% | Registered, at 0.56% |
| --- | --- | --- | --- | --- | --- | --- |
| Projected sample, calibrated dependence | 5.3 | 45 | 24 | 4,997 | 10.5% (9.7 to 11.4) | 3.06% (2.58 to 3.54) |
| Same, four largest categories only | 4 | 102 | 45 | 5,000 | 7.7% (7.0 to 8.4) | 1.56% (1.22 to 1.90) |
| Same, three largest | 3 | 171 | 58 | 5,000 | 6.9% (6.2 to 7.6) | 1.20% (0.90 to 1.50) |
| Same, two largest | 2 | 306 | 58 | 5,000 | 5.9% (5.3 to 6.6) | 0.84% (0.59 to 1.09) |
| Independent data: no issuer twice, no common shock, no serial correlation | 5.0 | 53 | 27 | 5,000 | 7.3% (6.6 to 8.0) | 1.44% (1.11 to 1.77) |
| Common daily share 20% | 5.3 | 46 | 25 | 3,000 | 10.5% (9.4 to 11.6) | 3.07% (2.45 to 3.68) |
| Serial correlation 0.25 | 5.3 | 46 | 25 | 3,000 | 10.6% (9.5 to 11.7) | 3.20% (2.57 to 3.83) |
| Event-window volatility doubled | 5.3 | 46 | 25 | 3,000 | 10.6% (9.5 to 11.7) | 3.13% (2.51 to 3.76) |
| Issuers recur as often as in the observed anchors | 5.1 | 55 | 27 | 2,910 | 17.4% (16.0 to 18.7) | 6.84% (5.92 to 7.76) |
| 240 anchors | 9.8 | 38 | 21 | 2,996 | 14.8% (13.5 to 16.0) | 4.84% (4.07 to 5.61) |
| 345 anchors (about a census) | 10.1 | 46 | 23 | 1,997 | 12.4% (10.9 to 13.8) | 3.36% (2.57 to 4.14) |

Intervals are 95%, from the binomial standard error; they do not include the
variation between projections of the sample, which is of similar size. A
separate run of 2,000 for the power analysis gave 10.4% and 2.9% for the first
row.

Independent clusters behind the first row: about 75 dates and 480 issuers.
The reference distribution is F(q, 74) with q of 4 or 5.

Reading the table:

* **It is not dependence the clustering fails to capture.** With fully
  independent data the test still rejects 7.3% and 1.44%.
* **It is the small categories.** Dropping to the four largest categories
  halves the excess; with two it is nearly gone. In a controlled design (four
  categories of 300 independent events on 80 dates, plus a fifth):

  | Fifth category | Registered, at 5% | Registered, at 0.56% |
  | --- | --- | --- |
  | 30 events on 15 dates (the registered minimum) | 7.0% (6.1 to 8.0) | 1.23% (0.84 to 1.63) |
  | 40 on 22 | 6.9% (6.0 to 7.8) | 1.33% (0.92 to 1.74) |
  | 60 on 30 | 6.3% (5.5 to 7.2) | 0.60% (0.32 to 0.88) |
  | 90 on 45 | 6.3% (5.4 to 7.1) | 0.70% (0.40 to 1.00) |
  | 150 on 60 | 5.8% (5.0 to 6.7) | 0.90% (0.56 to 1.24) |
  | 300 on 80 (balanced) | 5.4% (4.6 to 6.2) | 0.53% (0.27 to 0.79) |

  3,000 runs each. One category at the registered minimum is enough to
  double the error rate at the Holm level.
* **Dependence adds to it.** Overlapping windows of recurring issuers, common
  shocks and heavy tails take 7.3% to 10.5%. If issuers recur as often across
  the year as they do inside one anchor, it is 17%.
* **A larger sample does not cure it.** More anchors let more small
  categories pass the 30-event rule (ten instead of five), each on about 20
  dates, and the rate rises to 12–15%.

Why: the variance of a small category's mean is estimated from the 20-odd
date clusters that contain it, but the F reference takes its denominator
degrees of freedom from all 75. The statistic is more variable than the
reference allows, in the tail most of all. This is the known failure of CR1
Wald tests with unbalanced cluster-level regressors (Imbens and Kolesár 2016;
Pustejovsky and Tipton 2018; MacKinnon and Webb 2017).

### What the verdict rule adds

The rule needs the spread between the highest and lowest category mean to be
at least 1% as well (`stock_research/hypotheses/common.py:170-171`). Under the
null that condition is met in 85% of simulated samples, because the range of
five or six noisy means is rarely below 1%. It filters nothing: the rate of a
false `supported` is the rate of rejection at the Holm level, 2.9%.

### Does the family-wise error rate still hold?

If H7 is the only hypothesis that produces a p-value, the chance that the
study reports a false finding is that 2.9–3.1%, below 5%. That is an accident
of Holm's slack when eight of nine tests cannot run, not a property of the
test: with the recurrence stress it is 6.8%, with 240 anchors 4.8%, and each
further hypothesis that does run adds its own share. The p-value itself is
wrong by a factor of five, and every figure derived from it (the nominal
sensitivity tests, the day-0-only variant) with it.

## 4. Alternatives, on the same simulated samples

| Procedure | At 5% | At 0.56% | Note |
| --- | --- | --- | --- |
| Registered: two-way CR1, F(q, G−1) | 10.5% | 3.06% | |
| Date clusters only, F(q, G−1) | 8.9% | 1.64% | registered as a sensitivity |
| Issuer clusters only | 8.3% | 1.66% | |
| Delete-one-date jackknife (CV3) | 7.5% | 1.30% | larger variance, same reference; not enough |
| Two-way without subtracting the intersection | 0.1% | 0.00% | far too conservative |
| Registered statistic, degrees of freedom from the scarcest category | 5.2% | 0.50% | no p-value in 4% of samples |
| Date clusters, degrees of freedom from the scarcest category | 5.0% | 0.34% | |
| **Wild cluster bootstrap by date, null imposed, 999 draws** | **4.3% (3.6 to 5.1)** | **0.40% (0.17 to 0.63)** | 3,000 runs; registered test on the same samples 9.1% and 2.21% |

The degrees-of-freedom rule is a heuristic of this audit: denominator degrees
of freedom equal to the fewest clusters carrying any one category, less the
number of restrictions. It holds the level on the 80-anchor design and
becomes far too conservative when many small categories are present (1.3% and
0.1% at 240 anchors). It is useful as a diagnosis, not as a proposal.

A pairs (block) bootstrap by date was not run. With categories on 20 dates a
resample often loses most of a category, and the jackknife, which is its
first-order version, already fails. A CR2 covariance with Satterthwaite
degrees of freedom (Pustejovsky and Tipton) is the other standard remedy and
was not implemented.

### Power of the procedures that hold their level

Same simulated samples for all three; 600 runs per row; rejection at the Holm
level.

| Planted effect | Registered | Scarcest-category degrees of freedom | Wild bootstrap |
| --- | --- | --- | --- |
| Evenly spaced, spread 2% | 18.8% | 9.5% | 9.0% |
| Evenly spaced, spread 3% | 49.3% | 25.3% | 22.2% |
| Evenly spaced, spread 5% | 91.5% | 78.0% | 80.2% |
| Only buyback differs, 1.5% | 23.3% | 12.0% | 10.5% |
| Only buyback differs, 2% | 45.8% | 26.3% | 24.7% |
| Only buyback differs, 3% | 87.8% | 72.7% | 72.3% |

Standard errors are 1 to 2 points. Roughly half of the registered test's
apparent power at moderate effects is its inflated level.

## Findings

### I-1. The registered H7 test rejects a true null five to six times too often at the level it is judged at

* **Severity:** critical for H7.
* **Evidence:** section 3.
* **Source:** `stock_research/stats.py:99` (degrees of freedom from the
  smaller total cluster count) and `:120-126` (F reference), applied by
  `stock_research/hypotheses/h7.py:99-112` to categories admitted at 30
  events on 15 dates (`h7.py:136`, `stock_research/config.py:62-63`).
* **How to verify:** `python -m readiness_audit.run_simulations calibration`,
  or in under two minutes
  `pytest tests/stock_research/test_readiness.py -k overrejects`.
* **Impact:** a `supported` verdict for H7 would carry a stated family-wise
  level of 5% and a real per-test level several times the intended one. An
  `inconclusive` verdict is unaffected in direction but, with the power
  available, says little.
* **Recommended action:** do not run H7 under protocol v2. Register a v3
  whose primary test is a wild cluster bootstrap by day 0 with the null
  imposed (or CR2 with Satterthwaite degrees of freedom), with the number of
  draws and the seed fixed, and with the calibration above repeated on the
  final sample structure before unsealing.
* **Protocol amendment:** yes. Proposed in READINESS_DECISION.md; not
  implemented.

### I-2. A category can qualify on a handful of issuers

* **Severity:** medium.
* **Evidence:** the sufficiency rule counts events and dates
  (`stock_research/hypotheses/common.py:145-146`), not issuers. In the
  recurrence stress design a category is carried by 9 issuers on average, and
  in 3% of those samples the two-way variance is not positive and no p-value
  is produced at all.
* **Impact:** a category's mean then rests on a few companies, and its
  variance on fewer independent units than either count suggests.
* **Recommended action:** in v3, require a minimum number of issuers per
  category and a higher minimum number of dates, chosen by repeating the
  controlled experiment of section 3 with the test v3 registers.
* **Protocol amendment:** yes.

### I-3. The mean test with two-way clustering is calibrated

* **Severity:** informational. Section 2. The category's own mean test on the
  projected sample rejects 4.7% to 7.1% at the 5% level under the null,
  depending on the category (POWER_RESULTS.csv, `power_category_own_test` at
  zero effect), so its intervals cover between 93% and 95%.

### I-4. No inference in a fraction of a percent of samples

* **Severity:** low. In 0.1 to 0.2% of projected samples the two-way variance has a
  non-positive diagonal and the code reports no p-value
  (`stock_research/stats.py:90-94`). That is the documented behaviour and the
  right one. It would turn H7 into `inconclusive` for a reason unrelated to
  the data's content.

## References

Cameron, Gelbach and Miller (2008), "Bootstrap-based improvements for
inference with clustered errors", *Review of Economics and Statistics*.
Cameron, Gelbach and Miller (2011), "Robust inference with multiway
clustering", *Journal of Business & Economic Statistics*.
Imbens and Kolesár (2016), "Robust standard errors in small samples",
*Review of Economics and Statistics*.
MacKinnon and Webb (2017), "Wild bootstrap inference for wildly different
cluster sizes", *Journal of Applied Econometrics*.
Pustejovsky and Tipton (2018), "Small-sample methods for cluster-robust
variance estimation and hypothesis testing in fixed effects models",
*Journal of Business & Economic Statistics*.
