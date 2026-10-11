# Data availability

*Measured 2026-10-11. Every number here was observed, by a live call or a
query; none is taken from documentation.*

## KAP disclosures

| | |
|---|---|
| Source | MKK API Portal, VYK API, **dev gateway** (`apigwdev.mkk.com.tr`) |
| What it serves | A historical sample, not live data |
| Index range | about 1,091,673 to 1,231,017 (≈139,000 disclosures) |
| Dates | 2023-01-02 to 2023-12-31 |
| Rate limit | 6 calls a minute |
| Listing call | 50 rows; type, class, company; **no timestamp, no subject** |
| Detail call | one disclosure: timestamp to the second, subject, summary, sender codes, related companies, HTML body |
| Roster | 1,037 members, 918 with exchange codes; current, not historical |

About two thirds of all disclosures are daily fund bulletins. Listing calls
are therefore filtered to types `ODA` and `CA`.

**The sample.** A census would take weeks at 6 calls a minute. The frame
(`kap-anchor-sample-v2`) is 80 anchor indices spaced evenly across the range;
at each, the next 50 material-event disclosures; of those, the ones filed by
listed companies (member type `IGS`). About 42 per anchor, roughly 3,400
events on roughly 80 to 100 trading days. Anchors are processed in a strided
order, so a partial ingestion still spans the year. The frame was fixed by
constants before any price was read.

A first design (`v1`, unfiltered blocks) was abandoned after 9 blocks, before
any price was read, because it yielded about four usable events per block.

A request to MKK for production access has been drafted. Production would
give live data; the rate limit would stay.

## Prices

| | |
|---|---|
| Source | Yahoo Finance, `.IS` symbols, daily bars |
| Window | 2022-06-01 to 2024-03-01: 130 sessions before the first sampled event, 20 after the last |
| Fields | open, high, low, close (split-adjusted), adjusted close (also dividends), volume, dividends, splits |
| Not available | trading status, halts, intraday bars, order book |

Of the first 126 symbols requested, 97 were served and 29 were not. The 29
are mostly codes that are not equities (bond issuer codes, warrant series).
Delisted companies are among the unserved and cannot be told apart from them.

Defects found in the served bars and how each is handled are in
EVENT_STUDY_METHOD.md ("What is done to prices"). After those rules, 13 of
40,496 stock-sessions (0.03%) show a move the exchange's limit forbids.

## News

| | |
|---|---|
| Scored headlines | 7,158, all Turkish-language, March to October 2026 |
| With a session before 2026-08-10 | 3,727 (the rest are inside the sealed index window) |
| Of those, naming a listed issuer (confirmed link) | 84 |
| Of those, single-issuer and material | **7**, across 4 issuers |
| Needed for a stock-level news test | at least 200 |

The corpus was collected for an index study: its sources and relevance filter
favour macro and market news. It cannot support stock-level news tests.

## Not available at all

| Dataset | Why |
|---|---|
| Point-in-time market capitalisation | No verifiable source of historical shares outstanding or free float. Today's share count times an old price is look-ahead and is not used. |
| Complete company news | News is known through a sample of disclosures and a scraped set of headlines. "No news for this stock that day" cannot be established. |
| English-language news | No English source is collected. |
| Intraday prices | Daily bars only. |
| Sectors, index membership history, ticker-change history | Not collected. |

## What this means for each hypothesis

| | Runs on real data | Why not, if not |
|---|---|---|
| H1 | No | No market capitalisation; 7 news events |
| H2 | No | 7 news events; KAP disclosures have timestamps but no sentiment score |
| H3 | No | No per-stock news history |
| H4 | No | No per-stock news history |
| H5 | No | Limit-down closes are detectable from prices, but "no news" cannot be established |
| H6 | No | Streaks are detectable, but news intensity cannot be measured |
| H7 | **Yes**, when the sample is complete | |
| H8 | **Yes, if enough purchases are in the sample** | The small-versus-large comparison is blocked (no market capitalisation) |
| H9 | No | No English stream, no intraday prices |

Each "No" is reported by the code as `data_insufficient` with the reason
above, taken from `pipeline.availability`. None is reported as a weak result.

## What would unblock the rest

- **H1–H4:** stock-level news at volume. That means sources that cover
  companies, not the economy, and a price snapshot for the news period. From
  2026-08-10 onward such a study must also wait for the sealed index test.
- **H2 alone:** a sentiment score for KAP disclosures. The existing scorer was
  validated on news headlines, not on disclosure text, so that needs its own
  labelled validation first.
- **H1, H8's size split:** a licensed or verifiable history of shares
  outstanding.
- **H5, H6:** a census of disclosures (production access and several weeks of
  ingestion), so that the absence of news means something.
- **H9:** an English news archive with publication timestamps, and intraday
  prices.
