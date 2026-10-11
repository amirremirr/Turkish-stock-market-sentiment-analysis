# Event-study method

Implemented in `stock_research/eventstudy.py`, `calendar.py`, `data/prices.py`
and `stats.py`.

## Day 0

Day 0 is the **first session whose prices could reflect the event**. It is
never the publication date as such.

| Published | Day 0 | Bucket | Day-0 return is a clean first reaction |
|---|---|---|---|
| On a non-session day | next session | `weekend_or_holiday` | yes |
| Before 10:00 on a session day | that session | `pre_open` | yes |
| 10:00 to the close | that session | `during_session` | **no**: it includes the hours before publication |
| After the close | next session | `post_close` | yes |
| Time unknown | next session | `unknown` | no |

Times are Istanbul local. A KAP timestamp is taken as local; an aware
timestamp from any other source is converted.

**Sessions** are the dates on which the benchmark has a bar. The repository's
holiday list covers 2025–2026 only, so it is not used for other years.

**The close** is 18:10. Official early closes are known for 2025–2026. For
other years they are not recorded anywhere here, so a publication after 12:30
on a session immediately followed by a weekday closure is treated as after the
close and flagged `timing_ambiguous`. That can delay day 0; it cannot advance
it. Ambiguous events are excluded from every primary sample.

Events from one issuer that share a day 0 are one observation. If they were
published at different kinds of time the merged row is `mixed`, not clean.

## Returns

- **Close-to-close return** on session *t*: `adj_close_t / adj_close_{t-1} − 1`,
  using the provider's dividend-and-split-adjusted close.
- **Overnight gap**: `log(open_t / close_{t-1})`.
- **Intraday return**: `log(close_t / open_t)`.

The gap and the prior-close move use unadjusted prices and are withheld on any
date with a recorded dividend or split, where the two closes are not
comparable.

## What is done to prices, and why

Each rule exists because the provider's bars were found to violate it.

| Rule | Handling | Evidence |
|---|---|---|
| A session whose trades the exchange annulled never happened | Its bars are removed from every series | 2023-02-08: Borsa Istanbul cancelled the session's trades; the provider still serves a bar |
| A bar with no volume is not a trade | For stocks it is a missing session | One ticker had 441 carried-forward bars |
| A missing session is missing | No return that day, and none on the next (which would span two days) | |
| A move beyond a verified price limit cannot have happened | Set to missing, recorded | Partial bars on 2022-12-16 implied +11% to +18% the next day |
| A jump over 25% with no recorded action | Set to missing, recorded | Unadjusted capital changes |
| A session the benchmark lacks | Folded into the next session for stock and index alike; same-day quantities withheld there | The index has no bar for 2023-12-25, though 96 stocks traded |

Nothing is filled, forward or with zero.

## Expected and abnormal returns

- **Market model**: `R_i = α + β·R_m + ε`, fitted by OLS on sessions −130 to
  −11 relative to day 0, at least 60 of which must have both returns. The code
  refuses an estimation window that reaches the earliest event window.
- `AR_t = R_t − (α + β·R_m,t)`.
- `CAR(a, b)` is the sum of `AR_t` for `t` in `[a, b]`.
- `BHAR(a, b) = Π(1 + R_t) − Π(1 + R_m,t)`.
- **Market-adjusted return** (`α = 0, β = 1`) is computed alongside as a
  sensitivity and needs no estimation window.

The benchmark is XU100, a price index. Stock returns include dividends. The
mismatch is about one basis point a day on average and is a stated limitation.

Windows: `[−5,−1]`, `[0,0]`, `[0,+1]`, `[+2,+5]`, `[0,+5]`, `[+1,+10]`,
`[+1,+20]`, plus the windows a hypothesis registers for itself.

## Missing data

A window with any missing return has no CAR. The event stays in the table with
a status, so the loss is countable:

| Status | Meaning |
|---|---|
| `ok` | studied |
| `ticker_not_in_panel` | no price series |
| `day0_outside_calendar` | no session to anchor to |
| `window_beyond_data` | a window runs past the data |
| `estimation_insufficient` | fewer than 60 estimation returns; market-adjusted values still present |
| `sealed_window` | an outcome would fall in the index study's sealed window |

## Overlapping events

Three different things, handled three ways:

1. **Same issuer, same day 0**: merged into one row.
2. **Same issuer, nearby days**: outcome windows overlap and share returns.
   Primary tests cluster by issuer as well as by date. H8 also reports a
   thinned sample, one purchase per issuer per 21 sessions.
3. **Different issuers, same day**: they share market and sector shocks.
   Primary tests cluster by date.

## Inference

`stats.ols_cluster` gives cluster-robust standard errors, one-way or two-way
(Cameron, Gelbach and Miller, 2011), with the CR1 small-sample correction and
a t reference distribution with (smallest number of clusters − 1) degrees of
freedom. Below 10 clusters in any dimension no p-value is produced. If a
two-way variance is not positive, no inference is reported; it is not clipped.

Why two-way is the default for multi-day windows: on simulated data with no
effect, 12 issuers each filing on 10 nearby sessions, a date-clustered test of
mean CAR(+1,+20) rejected 48% of the time at the 5% level. The two-way test
rejected 6.7%.

Measured rejection rates of the full hypothesis tests under a null (100
synthetic contexts each, nominal 5%): H1 3%, H2 4%, H4 2%, H5 6%, H7 8%, H8 4%.

## What this method does not establish

An abnormal return after a disclosure is an association in event time. It is
not evidence that the disclosure caused the return, and it is not a trading
profit. Costs and fills are a separate calculation (EXECUTION_ASSUMPTIONS.md).
