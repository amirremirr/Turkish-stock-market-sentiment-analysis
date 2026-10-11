# Corporate actions and price adjustment

Audit output of 2026-10-11. This document compares single-session price
ratios between two sources, for every stock and every session alike. It
builds no event window, fits no market model and looks up no sampled
disclosure's outcome. Where it counts sampled events near a suspect date, it
counts them by ticker and date only.

## Method

* **Study source.** Yahoo Finance daily bars, as stored
  (`sr_raw_price_bars`, snapshot `yahoo-2022-06-01-2024-03-01`): 253 stocks
  and XU100, 2022-06-01 to 2024-02-29, read from a copy of the database.
* **Second source.** İş Yatırım's public data service: for each stock its
  daily closes, raw and adjusted, and its list of capital increases and
  dividends with dates and price-adjustment ratios; and the XU100 closing
  value by date. Fetched for all 253 stocks; 252 returned a series (LRSHO did
  not). The series are kept in `outputs/stock_research/readiness/isyatirim/`
  and are not committed.
* **Selection.** Every corporate action either source records inside the
  price range, not a chosen subset: 70 capital increases, 208 cash dividends.
  Nothing was selected by a return.
* **Comparison.** The return the study would use (`ret`, after the cleaning
  rules of `stock_research/data/prices.py:245-296`) against the second
  source's adjusted close-to-close return, session by session: 104,535 pairs.

Reproduce: `python -m readiness_audit.price_checks`, then
`python -m readiness_audit.independent_prices fetch` and `... compare`.

## Answers to the five questions

1. **Are closing prices adjusted consistently?** For dividends and for plain
   bonus issues, yes. For four capital increases, no (finding A-2). And on
   sixteen dates the stored "close" is not the closing price at all for a
   large share of stocks (finding A-1), which is a larger problem than any
   adjustment.
2. **Are opens and closes comparable across ex-dates?** The study does not
   compare them there: same-day quantities are withheld on any date with a
   recorded action (`prices.py:274-276`). Rights issues carry no action record
   in Yahoo (finding A-3), but its open and close are both already adjusted
   on those dates, so the pair is consistent.
3. **Do adjustments land on the right date?** Yes. 51 of the 52 capital
   increases Yahoo records are on the second source's date and the other is
   within one session; 206 of 208 dividends are on the same date; no Yahoo
   action lacks a counterpart.
4. **Do benchmark and stock returns use compatible conventions?** Nearly.
   XU100 is a price index and the stock series is dividend-adjusted; see
   MARKET_MODEL_AUDIT.md, M-3. Yahoo's XU100 closes agree with the second
   source on all 439 sessions (no return differs by more than 0.2%).
5. **Can Yahoo's values be validated independently?** Yes, and they were.

## Results

| Check | Result |
| --- | --- |
| Ordinary sessions compared | 104,208 returns; median absolute difference 0.0001% |
| Ordinary sessions differing by more than 1% | 1,154 (1.1%), every one of them on 16 dates |
| All retained returns differing by more than 1% | 1,166, of which 1,157 on those 16 dates |
| Dividend days: Yahoo's adjusted ratio against its own dividend and close | 204 of 205 exact to 0.00003%; one (DOCO, 2023-07-24) off by 1.5% |
| Dividend days: Yahoo against the second source | 198 of 204 within 1% |
| Bonus issues recorded by both | 47 of 51 within 1% |
| Rights issues (15, second source) | none recorded as an action by Yahoo; 14 of 15 returns agree to 0.01%, 1 differs (GEDIK, 2022-12-14, before the sample) |
| Rights and bonus together (1) | wrong in Yahoo (HEKTS) |
| Returns the cleaning rules already remove | 26: 24 beyond the 10% limit, 2 jumps over 25% |
| Benchmark | 439 of 439 sessions agree |

## Findings

### A-1. On sixteen dates Yahoo's close is not the closing price for many stocks

* **Severity:** high for data validity; the effect on H7 depends on one date.
* **Evidence:** disagreements of more than 1% are not scattered. They sit on
  eight pairs of consecutive sessions, and within a pair the two differences
  have opposite signs (1,136 of 1,166), which is what one wrong price between
  two right ones produces.

  | First date of pair | Stocks off by >1% | Second date | Stocks off by >1% |
  | --- | --- | --- | --- |
  | 2022-06-27 | 75 of 213 | 2022-06-28 | 71 of 210 |
  | 2022-11-09 | 40 of 228 | 2022-11-10 | 41 of 224 |
  | 2022-11-17 | 36 of 228 | 2022-11-18 | 41 of 228 |
  | 2022-11-29 | 41 of 228 | 2022-11-30 | 39 of 227 |
  | 2022-12-08 | 28 of 230 | 2022-12-09 | 25 of 228 |
  | 2022-12-12 | 42 of 230 | 2022-12-13 | 41 of 230 |
  | 2022-12-16 | 113 of 231 | 2022-12-19 | 105 of 221 |
  | **2023-01-05** | **210 of 234** | **2023-01-06** | **209 of 233** |

  The median size of a disagreement is 2.3%; the largest ordinary one is 12.6%.
  Yahoo is the source that is wrong: on the second date of several pairs its
  returns exceed the 10% price limit (11 stocks on 2022-12-19 alone, already
  flagged by the cleaning rule), the second source's do not, and the two
  sources agree on the index for every one of these dates. The index fell
  7.4% on 2023-01-05; Yahoo's stock bars for that day show a small fraction of
  that move and give it back the next day.
* **Source:** the provider; the cleaning rules catch only the part that
  breaks the price limit (`prices.py:278-288`), 26 returns out of about 1,150
  wrong ones.
* **How to verify:** `outputs/stock_research/readiness/return_disagreements.csv`
  after `independent_prices compare`; group by `date`.
* **Impact.** Inside a window that contains both dates of a pair the two
  errors nearly cancel. A window that starts or ends between them is wrong by
  the full amount, for almost every stock on that date and in the same
  direction, while the benchmark is right. In the sample year only
  2023-01-05/06 is affected (419 of the 425 disagreements of 2023). For H7's
  CAR(0,+5) the exposed events are those with **day 0 on 2023-01-06**: their
  first return is measured from a wrong close. The other windows are exposed
  for day 0 on 5 January ([0,0]) and neighbouring dates. In the snapshot no
  sampled event has day 0 on either date, but anchors 1 and 2 had not been
  ingested and, by their position in the disclosure numbering, fall on about
  5–9 January. The fourteen earlier dates lie in 2022 and enter estimation
  windows only, where both dates of a pair are always present.
* **Recommended action.** Before any run, count events with day 0 on
  2023-01-05 or 2023-01-06 (metadata only). In a future protocol, replace the
  price source or add a registered second-source check that sets a return to
  missing when the two sources differ by more than a stated tolerance.
* **Protocol amendment:** yes (price rules).

### A-2. Four capital increases are adjusted with the wrong factor and escape cleaning

* **Severity:** medium.
* **Evidence:**

  | Stock | Date | Yahoo split ratio | Second source's factor | Yahoo return | Second-source return |
  | --- | --- | --- | --- | --- | --- |
  | DMSAS | 2022-06-21 | 2.097 | 1.905 | +13.3% | +2.9% |
  | HEKTS | 2022-10-12 | 1.442 | 2.874 | +29.3% | +5.5% |
  | MAGEN | 2023-11-29 | 3.611 | 2.878 | +12.9% | −10.0% |
  | NATEN | 2023-11-29 | 7.854 | 5.000 | +42.7% | −9.2% |

  All four Yahoo returns are beyond the 10% limit, so they cannot have
  happened. They survive because a date with a recorded action is exempt from
  both tests that would catch them: the jump test applies only where no
  action is recorded (`prices.py:262-266`), and the limit test runs on
  `raw_ret`, which is withheld on action dates (`prices.py:275-276`,
  `:283-286`).
* **How to verify:** `python -m readiness_audit.price_checks`
  (`split_days_adjusted_move_beyond_limit: 4`), and
  `capital_actions_compared.csv`.
* **Impact.** Any event of these four issuers whose event window or
  estimation window contains the date is wrong: by 23 to 52 percentage points
  if the date is in the event window, and by about 0.2 to 0.4 points a day in
  the expected return if it is in the estimation window. In the snapshot, two
  usable H7 events (HEKTS, day 0 on 2023-02-15 and 2023-04-13) have the 2022
  date in their estimation window. One such event in a category of 40 moves
  its mean by 0.6 to 1.3 points if the date is in the event window, which is
  the size of the registered material effect.
* **Recommended action.** In a future protocol apply the price-limit test to
  the adjusted return on action dates as well. An adjusted move beyond the
  limit on an ex-date is a wrong factor.
* **Protocol amendment:** yes (price rules).

### A-3. Rights issues are adjusted but not recorded as actions

* **Severity:** low.
* **Evidence:** 15 rights issues in the second source's list; Yahoo's close is
  continuous across all of them and its return matches the second source to
  0.01% in 14, but `split_ratio` is zero on the date. The `corporate_action`
  flag is therefore false there.
* **Impact:** none on returns. Same-day quantities (gap, intraday, limit
  detection) are computed on those dates from prices that are already
  adjusted on both sides, so they are consistent.
* **Recommended action:** note in DATA_AVAILABILITY that the action flag does
  not mark rights issues.
* **Protocol amendment:** no.

### A-4. Dividend amounts differ between sources on six of 204 dates

* **Severity:** low.
* **Evidence:** CCOLA 2023-05-22 (1.0 points), ORCAY 2023-10-26 (1.2), TUPRS
  2023-09-29 (5.1), and KRDMA, KRDMB, KRDMD on 2022-06-28, which is one of the
  dates of finding A-1. Which source has the right amount for TUPRS was not
  established.
* **Impact:** one stock-day in the sample year with a difference above 2
  points.
* **Protocol amendment:** no; covered by the second-source check proposed in A-1.

### A-5. What the second source cannot settle

It is a broker's data service, not the exchange. Where the two sources differ
the argument above rests on the price limit, on the index agreeing, and on the
pattern of paired errors, not on the second source being official. A third,
official source (Borsa İstanbul's own end-of-day files) was not available to
this audit.

## Suspicious adjustments, in one list

`outputs/stock_research/readiness/capital_actions_compared.csv` (every capital
increase, both sources' returns) and `return_disagreements.csv` (every
retained return that differs by more than 1%) are the full lists. Neither is
committed, because both hold provider prices; both are rebuilt by the
commands above.
