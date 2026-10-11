# Validation of the candidate protocol v3

2026-10-11. **Every rejection rate and power figure below comes from synthetic
returns.** The price section compares single-session returns of two providers
for every stock on every session; it builds no event window. No real
hypothesis was run and no event-window CAR was computed.

Results files: `validation/null.json`, `validation/power.json`,
`validation/verdict.json`, `validation/price_reconciliation.json`.

Reproduce (each simulation piece is independent and resumable; on this
machine, 7.4 GB of memory, no more than six at once):

```
python -m v3_validation.snapshot                   # read-only copy of the database, structure, 2022 calibration
python -m v3_validation.calibrate run <design> <seed> <simulations> 999 null|power|verdict
python -m v3_validation.calibrate merge
python -m v3_validation.price_quality              # second source into the copy, then the comparison
```

## 1. Which joint test

### Candidates

All use the same estimator: OLS of CAR(0,+5) on class indicators with the
reference group as baseline, and a Wald statistic for "all contrasts zero".

| Name | Covariance in the statistic | Reference distribution |
| --- | --- | --- |
| v2 F | two-way (day 0, issuer), CR1 | F(q, G−1) |
| Wild, date statistic | by day 0 | wild cluster bootstrap, weights by day 0 |
| **Wild, two-way statistic** | two-way (day 0, issuer) | wild cluster bootstrap, weights by day 0 |
| CR2 / Satterthwaite | by day 0, CR2 | F with Satterthwaite degrees of freedom (Pustejovsky–Tipton AHT) |

Implementation checks (tests): the bootstrap equals refitting the registered
estimator on every bootstrap sample, one-way and two-way; CR2 gives G−1
degrees of freedom and the t-test's p-value for a mean with equal clusters.

### Designs

Structure comes from the 29 anchors complete in the copy (1,279 disclosures;
156 usable under v3 eligibility). Returns are simulated by the readiness
audit's simulator with its 2022 calibration: issuer volatility, a market-wide
daily shock, serial correlation, heavy tails, an estimated intercept, and
shared returns wherever one issuer's windows overlap.

| Design | What it stresses | Events | Dates | Issuers | Tested classes |
| --- | --- | --- | --- | --- | --- |
| Audit, v2 rules | the audit's own benchmark | 999 | 73 | 457 | 4.0 |
| 80 anchors, v3 rules | the frame as registered | 320 | 53 | 181 | 2.1 |
| Census | full year | 1,456 | 192 | 796 | 6.9 |
| Census, persistent | issuers return within ten sessions at twice the measured rate | 1,298 | 188 | 695 | 7.0 |
| Census, programmes | buyback filers return with probability 0.9; thinned | 951 | 177 | 636 | 6.8 |
| Census, programmes, unthinned | the same without thinning; one issuer files up to 99 times | 1,219 | 195 | 641 | 6.8 |
| Census, recurrence | half of all events keep a copied anchor's issuer | 1,276 | 186 | 439 | 7.0 |
| Census, common 20% | market-wide share of variance 20% | 973 | 183 | 638 | 6.7 |
| Census, serial 0.25 | serial correlation 0.25 | 970 | 177 | 645 | 6.7 |
| Controlled, minimum | one class at 30 events, 16 dates, 20 issuers | 1,125 | 200 | 770 | 4 |
| Controlled, programme | one class of 20 issuers each filing on six consecutive sessions | 1,215 | 225 | 770 | 4 |

A design in which every unseen anchor repeats a seen anchor's issuers could
not be run: it leaves fewer distinct issuers than the registered minimum, so
the study would return `data_insufficient`. The issuer rule is what stops it.

### Rejection of a true null

Percent; intended 5 and 0.56 (the level a lone test must meet under Holm over
nine). 999 bootstrap draws. 95% intervals are in `null.json`; at 3,000 runs
they are about ±0.8 and ±0.3.

| Design | Runs | v2 F | Wild, date statistic | Wild, two-way statistic | CR2 |
| --- | --- | --- | --- | --- | --- |
| Audit, v2 rules | 3,000 | 9.6 / 2.43 | 5.0 / 0.47 | 4.7 / 0.57 | 5.0 / 0.57 |
| 80 anchors, v3 rules | 3,000 | 6.9 / 1.13 | 4.9 / 0.43 | 5.1 / 0.47 | 4.5 / 0.43 |
| Census | 2,000 | 7.5 / 1.35 | 4.2 / 0.55 | 4.5 / 0.70 | 4.2 / 0.70 |
| Census, persistent | 2,000 | 8.7 / 1.75 | 4.9 / 0.30 | 5.3 / 0.50 | 4.9 / 0.35 |
| Census, programmes | 3,000 | 8.8 / 1.67 | 4.4 / 0.47 | 5.2 / 0.50 | 4.4 / 0.43 |
| **Census, programmes, unthinned** | 3,000 | 9.7 / 1.50 | **7.2 / 0.93** | 4.7 / 0.60 | **7.2 / 0.97** |
| Census, recurrence | 3,000 | 11.2 / 2.83 | 3.9 / 0.33 | 4.7 / 0.57 | 3.6 / 0.50 |
| Census, common 20% | 1,500 | 8.1 / 1.80 | 4.3 / 0.60 | 4.7 / 0.47 | 4.1 / 0.53 |
| Census, serial 0.25 | 2,500 | 8.0 / 1.56 | 4.2 / 0.40 | 4.4 / 0.48 | 4.0 / 0.48 |
| Controlled, minimum | 3,000 | 8.6 / 1.83 | 5.3 / 0.37 | 5.0 / 0.43 | 5.3 / 0.50 |
| **Controlled, programme** | 3,000 | 6.9 / 1.00 | **22.1 / 9.17** | 5.1 / 0.60 | **22.4 / 9.67** |

Regression benchmark: on the audit's design the readiness audit measured
10.5 / 3.06 for the v2 test (9.1 / 2.21 in its bootstrap run) and 4.3 / 0.40
for the date-level bootstrap. Reproduced here with new code: 9.6 / 2.43 and
5.0 / 0.47.

### Reading

* **The v2 F test fails everywhere**, by a factor of two to five at the Holm
  level. Confirmed.
* **Anything that sees dates only fails when issuers recur with overlapping
  windows.** With one class filed on six consecutive sessions by each of
  twenty issuers, the date-statistic bootstrap and CR2 by date both reject a
  true null 22% of the time at 5% and over 9% at the Holm level. In the
  unthinned census they reject 7.2% and about 0.95%. A date-only procedure
  does not resolve the dependence; the instruction not to assume so was right.
* **The bootstrap of the two-way statistic holds its level in every design**:
  4.4 to 5.3% at 5%, 0.43 to 0.70% at the Holm level, including both designs
  where the others fail. The highest Holm-level figure (0.70, census, 2,000
  runs) has an interval of 0.33 to 1.07.
* CR2 by date behaves like the date bootstrap: fine without issuer overlap,
  wrong with it. It has no two-way form here.

### Power, at the Holm level

Percent; 999 draws; standard errors 2 to 3 points.

| Planted effect | Design | Runs | v2 F (level not held) | Wild, date | Wild, two-way | CR2 |
| --- | --- | --- | --- | --- | --- | --- |
| buyback +1% | census, programmes | 250 | 6.8 | 1.6 | 3.2 | 1.6 |
| buyback +1.5% | census, programmes | 250 | 20.8 | 5.6 | 10.8 | 5.2 |
| buyback +2% | census, programmes | 250 | 38.8 | 17.6 | 28.0 | 18.4 |
| buyback +3% | census, programmes | 250 | 80.8 | 64.4 | 67.6 | 66.8 |
| new_contract +2% | census, programmes | 500 | 14.2 | 7.8 | 8.2 | 7.6 |
| new_contract +3% | census, programmes | 500 | 39.6 | 25.6 | 28.0 | 27.4 |
| dividend +3% | census, programmes | 500 | 25.4 | 16.0 | 14.4 | 17.4 |
| every class ±2%, alternating | census, programmes | 500 | 95.8 | 91.2 | 91.0 | 92.6 |
| buyback +2% | 80 anchors | 300 | 17.3 | 8.0 | 9.3 | 7.7 |
| buyback +3% | 80 anchors | 300 | 47.7 | 30.0 | 31.7 | 30.0 |
| every class ±2% | 80 anchors | 300 | 38.3 | 25.3 | 23.7 | 25.7 |

The two-way bootstrap gives up nothing against the date-only procedures and
is ahead of them where the issuer dimension carries information (buyback).

### Selection

**Primary: wild cluster bootstrap of the two-way clustered Wald statistic,
null imposed, Rademacher weights by day 0, 9,999 draws, fixed seed.** It is
the only candidate that held its level under date clustering, issuer
recurrence and unbalanced classes together, and it has at least the power of
the others. Thinning an issuer's overlapping events is kept as well: it is
what makes an event an observation, and the unthinned analysis is a
registered sensitivity. CR2 by date, the date-statistic bootstrap and the v2
F test are reported as sensitivities, the last marked as known to over-reject.

### The verdict rule

`supported` needs the joint p-value to pass Holm over nine and a contrast that
is itself significant (Holm across contrasts, 5%) with absolute size of at
least 1%. With no effect in the data:

| Design | Runs | Joint test passes Holm | Verdict `supported` | Any contrast significant |
| --- | --- | --- | --- | --- |
| Census, programmes | 600 | 0.67% (0.02 to 1.32) | 0.50% (0.00 to 1.06) | 4.3% |
| 80 anchors | 300 | 0.33% | 0.33% | 4.0% |

Under v2 the corresponding figure was 2.9%. The contrast tests hold their own
5% level.

## 2. What the census can and cannot detect

With the selected test, 80% power at the Holm level needs roughly: every
tested class differing by 2% in alternating directions (91%); or buyback alone
by a little over 3% (68% at 3%). A single small class is hard: +3% in
new_contract is detected 28% of the time, in dividend 14%. The census under
v3 rules tests about 950 thinned events, not the 4,500 of the audit's census
projection, because unclassified filings (about four in ten) are outside the
test and the reference group is procedural filings only.

This is a limit of one year of data, stated before any outcome is read. An
`inconclusive` verdict will not rule out effects of 1 to 2% in a single class.

## 3. Two-source price validation

On the copy, for all 477 issuer codes in the sample at the time (29 anchors):

| | |
| --- | --- |
| Codes served by the first provider / by the second | 346 / 356 of 477 |
| Stocks with first-provider bars and no second-source series | 2 (CVKMD, LRSHO) |
| Returns compared | 143,654 |
| Median absolute difference | 0.0001% |
| Differing by more than 0.2% / 0.5% / 1% / 2% | 2.6% / 1.8% / 1.1% / 0.6% |
| Over the 1-point tolerance, set to missing | 1,611 |
| of which on 16 dates | 1,603 |
| elsewhere | 8 |
| Benchmark returns that disagree | 0 |
| Set to missing for want of a second-source return | 653 |
| Set to missing by the limit test on action dates | 6 |
| Returns kept: v2 rules / v3 rules | 144,307 / 142,043 (1.6% fewer) |
| Sessions against the recorded 2023 schedule | no mismatch |

The sixteen dates are the ones the readiness audit found, now on a larger
panel: 2022-06-27/28, 11-09/10, 11-17/18, 11-29/30, 12-08/09, 12-12/13,
12-16/19 and 2023-01-05/06 (283 of 323 stocks). v2's cleaning removed 26 of
these returns; v3 removes all of them.

Corporate actions (second source's list against the first's records):

| | Listed | Recorded by the first provider | Return verified | Not verified |
| --- | --- | --- | --- | --- |
| Capital increases | 103 | 72 | 90 | 13 |
| Cash dividends | 253 | 252 | 242 | 11 |

The first provider records no action for 31 of the capital increases. The
readiness audit found these to be rights issues it adjusts for without an
action record (finding A-3); 90 of the 103 returns verify either way. The 13
unverified capital increases, all set to missing:

* adjusted move beyond the price limit (wrong factor): DMSAS 2022-06-21,
  HEKTS 2022-10-12, MAGEN and NATEN 2023-11-29, EGSER 2023-05-04. The last was
  not in the audit's panel. SEYKM 2022-09-22 is flagged by the same rule on a
  date the second source lists no capital increase.
* sources disagree: GEDIK 2022-12-14, HEDEF 2022-08-03, SASA 2023-05-23,
  VKGYO 2023-09-26.
* no return to check (no trade, or no second-source series): ISATR twice,
  ISBTR, LRSHO.

**Unresolved.** Which provider is right is settled only where the price limit
or the paired-error pattern settles it. For the four plain disagreements and
the 11 dividend dates it is not; those returns are missing, not corrected. The
second source is a broker's service, not the exchange.

**Effect on the sample so far** (existence of data only): of 154 eligible
thinned events with complete data under v2 rules, 153 keep it under v3. No
sampled disclosure has day 0 on a mass-disagreement date. The anchors near
5–6 January 2023 were not yet in the copy.

**Coverage limits.** 131 of 477 codes are not served by the first provider
and 121 not by the second (bond-only issuers, delisted or renamed tickers).
Their events have no price and are counted as unresolved. This is the
survivorship limitation of the readiness audit (K-1), unchanged.

## 4. What was not validated

* The census structure is projected from 29 of 80 anchors. Calibration and
  power must be repeated on the real census structure before unsealing; that
  needs no return (approval item 7).
* Buyback recurrence is measured between anchors days apart (21%, a lower
  bound) and stressed to 0.9; the true daily pattern is unknown until the
  census exists.
* Noise is calibrated on 2022.
* Power runs are 250 to 500 simulations per cell.
* H1–H6, H8 and H9 were not re-examined. Their specifications are unchanged;
  they inherit the price rules, the calendar and the gates.
