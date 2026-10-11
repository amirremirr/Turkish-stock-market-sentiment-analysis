# Market model and benchmark

Audit output of 2026-10-11. The numbers below come from sessions up to
2022-12-30 only, before the first sampled disclosure, and involve no event.
Reproduce with `python -m readiness_audit.market_model`.

## The registered model, as implemented

| Item | Implementation | Verdict |
| --- | --- | --- |
| Estimation window | sessions −130 to −11 relative to day 0 (`stock_research/config.py`, used at `stock_research/eventstudy.py:129`) | as registered |
| Strictly before the event | the window ends 11 sessions before day 0, and `_check_windows` refuses any window that reaches the earliest event day (`eventstudy.py:62-71`) | correct |
| Short history | the slice is clipped at the start of the data, never moved toward the event (`eventstudy.py:127-129`) | correct |
| Minimum observations | 60 sessions with both a stock and a market return, else no model and no CAR (`eventstudy.py:85-88`, `:138-139`) | as registered |
| Alpha and beta | OLS: beta = cov/var, alpha = mean difference (`eventstudy.py:93-94`) | correct; reproduced by an independent fit in the audit code |
| Abnormal return | `R − (alpha + beta · Rm)`, summed over the window (`eventstudy.py:147-150`) | correct |
| Missing returns | a window with any missing stock or market return has no CAR (`eventstudy.py:144`) | correct; nothing is filled |
| Calendar | stock frames are reindexed to the benchmark's sessions (`data/prices.py:257`), so both series are on one calendar | correct |
| Missing benchmark session | 2023-12-25 is folded into 26 December for stock and market alike (`data/prices.py:320-332`) | correct; the second source confirms it was a session |
| Benchmark | `XU100.IS`; its closes agree with a second source on all 439 sessions | verified |

No look-ahead was found in the expected-return model.

## Findings

### M-1. The benchmark explains little of a typical sampled stock's return

* **Severity:** medium, as a limit on interpretation; it does not bias the
  joint test.
* **Evidence (229 stocks, 2022-06-01 to 2022-12-30):** median beta against
  XU100 0.51 (10th to 90th percentile 0.19 to 0.87); median R² 7.7% (1.3% to
  27%). Out of sample (fitted on the first 74 sessions, judged on the next
  75, 225 stocks) the median daily residual standard deviation is 3.34% under
  the market model against 3.47% for the raw return: the model removes about
  7% of the variance. A market-adjusted return (beta of one) is worse than no
  adjustment, 3.51%. A model on the equal-weighted return of the panel's own
  stocks does slightly better, 3.21%.
* **Impact.** XU100 is weighted by float-adjusted capitalisation and
  dominated by a few large companies; most sampled issuers are small. The
  abnormal return is therefore close to the raw return less a small market
  term, and its noise is close to the raw return's. That is what limits power
  (POWER_ANALYSIS.md), and no choice of single index changes it much.
* **Recommended action.** None for the registered study. For a later
  protocol: an equal-weighted index of the listed universe, or a size- or
  sector-matched portfolio, as a registered alternative; the pre-sample
  evidence suggests a gain of a few percent in precision, not a
  transformation.
* **Protocol amendment:** only if adopted.

### M-2. The level of a category's mean CAR is not a clean abnormal return

* **Severity:** medium, for the per-category tables; low for the primary
  joint test.
* **Evidence.** In the out-of-sample half of the pre-sample period the
  average stock's market-model abnormal return was +0.15% a day (+0.11% under
  market adjustment, −0.01% under the equal-weighted model). Over a six-session
  window that is about +0.9% for an event that never happened. Its sign and
  size in 2023 are unknown and were not measured.
* **Why.** An intercept fitted in one half-year does not carry into the next,
  and small stocks as a group move away from a large-cap index for months at
  a time. The model has no term for that.
* **Impact.** The primary H7 test compares categories, and a shift common to
  all events cancels in a comparison. It does not cancel when categories sit
  on different dates or on different kinds of stock. Category means, and the
  exploratory per-category tests in every window
  (`stock_research/hypotheses/h7.py`, the `exploratory` block), should not be
  read as "this kind of filing is followed by an abnormal return of x": a
  non-zero mean for *all* categories is what benchmark misfit looks like.
* **Recommended action.** State this next to any per-category figure. In a
  later protocol, test category differences with day-0 fixed effects or
  against matched non-event stock-days.
* **Protocol amendment:** only if adopted.

### M-3. Price index against dividend-adjusted stock returns

* **Severity:** low.
* **Evidence.** `XU100.IS` is the price index; stock returns come from
  `adj_close`, which adds dividends back (`data/prices.py:260-261`). On an
  ex-dividend date the stock's adjusted return includes its dividend and the
  index return does not include the index's.
* **Impact.** The index's dividend yield is a few percent a year, spread over
  the spring. The mismatch on any one session is a few basis points and is
  partly absorbed by alpha. It is negligible next to a 1% material effect.
* **Recommended action:** document; use the total-return index if a later
  protocol changes the benchmark anyway.
* **Protocol amendment:** no.

### M-4. Beta is unstable from one half-year to the next

* **Severity:** low.
* **Evidence.** Correlation of a stock's beta between the two halves of the
  pre-sample period: 0.56. Median absolute change: 0.18. With a median R² of
  8%, a 120-session beta has a standard error of about 0.15, so most of the
  instability is estimation noise.
* **Impact.** Small: an error of 0.2 in beta times a six-session market move
  of 3% is 0.6 points on one event, with no systematic sign. It adds noise the
  simulation already has through the estimated intercept, and none through
  beta; the power figures are to that extent slightly optimistic.
* **Protocol amendment:** no.

### M-5. Earlier events of the same issuer sit inside the estimation window

* **Severity:** low.
* **Evidence.** The estimation window is not cleaned of the issuer's own
  earlier disclosures. In the snapshot an issuer has 1.4 usable events on
  average and at most 8.
* **Impact.** If filings are followed by abnormal returns, frequent filers'
  intercepts absorb part of it, pulling their CARs toward zero. This works
  against finding an effect, not for it.
* **Protocol amendment:** no.

### M-6. Wrong returns inside estimation windows

Fourteen of the sixteen dates with wrong closes, and two of the four wrong
capital-increase factors, fall in 2022, inside the estimation windows of 2023
events. Paired errors cancel in an intercept; the two wrong factors do not
(+24 and +10 points on one day each). See PRICE_ADJUSTMENT_AUDIT.md, A-1 and
A-2.

### M-7. Thin trading

One of 225 stocks had no trade on more than 5% of pre-sample sessions.
Non-trading is not a material source of beta bias in this panel.

## What may not be changed

The registered primary benchmark stays XU100 under the market model. The
alternatives above are proposals for a later protocol and were not introduced
into any registered code.
