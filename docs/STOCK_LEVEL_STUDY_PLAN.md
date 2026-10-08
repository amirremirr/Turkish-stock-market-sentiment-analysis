# Plan: a stock-level KAP study (v2)

*Drafted 2026-10-08. A plan, not a protocol: nothing here is frozen, and no
code for it exists yet. It does not touch `untouched_future_v1`, which keeps
running on the index exactly as sealed.*

## Why the next study should be stock-level

The index study's binding constraint was data, not method, and its target caps
the data it can ever have. Every event that reacts on the same session shares
one BIST 100 return. 731 event rows carried only 49 distinct outcomes. One
index return per trading day is about 250 independent outcomes a year, however
many headlines are collected.

Company disclosures change that arithmetic. A material-event disclosure about
one listed company has its own outcome: that stock's return, net of the market,
on the first session able to react. Many companies report on the same day, so
outcomes accumulate across stocks as well as across days.

## The unit and the target

- **Unit:** one stock on one reactable session. Several disclosures from the same
  company on the same session collapse to one row, the same rule v1 applies to
  the index.
- **Target:** the stock's open-to-close return on its first reactable session,
  minus the BIST 100 return over the same window. This is an abnormal return, so
  a market-wide day does not count as company news.
- **Timing:** v1's proven convention (`docs/TIMING.md`). A disclosure published
  after the close reacts at the next session's open, never the publication
  session.

## The trap this design must not fall into

Stock rows on the same session are not independent. A bad day for Turkish
banks moves every bank. Treating 40 stock rows on one day as 40 observations
would repeat v1's event-counting error at a larger scale. So:

- uncertainty uses two-way clustering, by session and by stock, or a session
  block bootstrap;
- the sample gate counts **sessions** and **distinct stocks** as well as rows;
- sector-level co-movement is reported, so a "signal" that is really one sector
  having a run is visible.

## Data

| Input | Source | Status |
|---|---|---|
| Disclosures (ODA material events; FR for context) | MKK API Portal VYK API, `kap_ingest.py` | Built; dev gateway only |
| Listed-company master and tickers | `/members`, `/memberSecurities` | Endpoint known |
| Stock prices | yfinance `.IS` tickers, the same provider as the index | Not yet collected |
| Index return for the abnormal-return target | existing `bist100_prices` | Live |

**Production access is the long pole.** The dev gateway serves a historical
sample that ends in December 2023, so live data needs the production gateway.
A draft request to kapdestek@mkk.com.tr is in the owner's Gmail drafts.

**The dev sample is still useful.** Real 2023 disclosures plus free historical
stock prices are enough to build and test the whole pipeline, and to run a
retrospective exploration while production access is pending. Its depth is
unknown; check it on the first authenticated call.

## Order of work

1. Request production access (done as a draft; the owner sends it).
2. Measure the dev sample's depth and disclosure volume per session. Counts
   only.
3. Collect `.IS` daily bars for every ticker in `/members`, with v1's
   price-bar completeness rules.
4. Build the stock-session dataset and the abnormal-return target, and test the
   timing rule against disclosure timestamps the way `docs/TIMING.md` did.
5. **Freeze a protocol before any comparative result.** That means feature sets,
   models, folds (by session, with an embargo), clustering, multiplicity and
   success criteria, in the same style as `research/protocol.py`, with a
   versioned hash.
6. Retrospective walk-forward on the dev sample: exploratory, labelled as such.
7. Once production data flows: register an untouched future window for the
   stock-level protocol, as v1 did, before any production outcome is read.

## What carries over from v1 unchanged

The session-level timing convention, the price-bar completeness rules, the
modelling-unit discipline, the frozen-protocol and sealed-artifact machinery,
and the rule that failure and inconclusive results are reported in full.
