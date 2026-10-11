# Execution assumptions

Implemented in `stock_research/costs.py` and the `execution` block of each
hypothesis.

**A statistically significant abnormal return is not a trading profit.** This
document is about the distance between the two.

## Costs are scenarios

| Scenario | Round-trip cost |
|---|---|
| Low | 0.30% |
| Middle | 0.40% |
| High | 0.50% |

**These are assumptions, not measured fees.** No broker statement or
order-book data was used. They are there to show how a gross number changes
across plausible costs.

`CostModel` splits the middle scenario into components so that one assumption
can be changed and named:

| Component | Default (round trip) |
|---|---|
| Brokerage commission | 0.10% |
| Exchange fees | 0.02% |
| Bid-ask spread | 0.15% |
| Slippage | 0.08% |
| Price impact | 0.05% |

Order size and partial fills are not modelled. Every figure is per unit of
traded value with no size, which is the most favourable case and is worst for
small caps, where the spread and impact of a real order would be far larger
than these defaults.

## A hedged trade is two trades

Each executable strategy is hedged with the index, so that it earns the
abnormal return the test measures. That is two round trips: the stock and the
hedge. The net figures subtract two.

Whether an index hedge is available at these costs (a future, an ETF) is
assumed and not verified.

## Long only

Short sales are not assumed, least of all in small caps. Where a hypothesis
is symmetric, only the long side is costed:

| | Executable version | Not assumed |
|---|---|---|
| H1 | After positive news on a small cap: buy at the close of day +1, sell at the close of day +5 | Shorting after negative news |
| H2 | After an upward gap: buy at the open, sell at the close of day 0 | Shorting after a downward gap |
| H5 | After a no-news limit-down close: buy at the next open, sell at the close two sessions later | Buying at the locked limit-down close |
| H8 | Buy at the close of day 0, hold 20 sessions | |

H3, H4, H6, H7 and H9 have no executable version: they are statements about
prediction or description, and no trading rule is registered for them.

## Fills

Daily bars cannot show the order book. Feasibility is decided by rule, and
every rule errs toward "not filled":

| Condition on the entry session | Result |
|---|---|
| No bar | not filled |
| No opening print | not filled |
| No volume | not filled (in the panel a zero-volume bar is already no bar) |
| One price all day and a move of about a full limit from the prior close | **locked**: not filled |
| Otherwise | assumed filled at the stated price |

Entry at a session's high or low is never assumed. Closing-auction and
opening-auction fills are assumed where stated and are not verified.

Each `execution` block reports coverage: how many signals there were and how
many passed the fill rule. A net return is the mean over filled trades only.

## Price limits

The rule in force comes from Borsa Istanbul's announcement 2020/20: from
2020-03-13 a 10% margin for all equity groups, until further notice. No later
change was found. Before that date the limit depended on a stock's market
group, which is not recorded here, so no limit day is detected there.

A limit day for H5 and H6 is a session that **closed** at the limit with the
close at the day's extreme. A session that touched the limit and came back is
not one.

**A locked price is not an entry price.** At a locked limit one side of the
book is empty. H5's statistic starts at the limit-down close, but its
executable version does not buy there, and the hypothesis is registered as
`execution_dependent`: the most it can reach is `execution_not_verifiable`.

## What is reported

For a strategy with trades: gross mean, net mean under each scenario, round
trips charged, number of trades, and coverage. `costs.max_drawdown` is
available for a compounded sequence of trades.

Not reported, because the data do not support them: turnover and exposure of
a portfolio (no portfolio is formed; each event is a separate trade),
capacity, and the effect of order size.

## What would make any of this verifiable

Intraday trades and quotes around the entry times, the exchange's own record
of limit states and halts, and a broker's actual fee schedule. None is in this
repository.
