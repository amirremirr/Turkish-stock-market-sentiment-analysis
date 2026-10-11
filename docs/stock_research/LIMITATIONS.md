# Limitations

Ordered by how much each one limits what a result can mean.

## 1. Seven of the nine hypotheses cannot be tested on real data

H1–H6 and H9 report `data_insufficient`. The estimators exist and recover
planted effects in synthetic data, which shows the code works. It shows
nothing about the market. DATA_AVAILABILITY.md lists what each would need.

## 2. Survivorship

The issuer list is the provider's current roster, and prices come from a
provider that generally does not serve delisted tickers. A company that was
delisted, merged or renamed after 2023 is missing from both. Results are
conditional on surviving to the snapshot, and the size of the bias is not
known: unserved symbols are counted, but a delisted company cannot be told
apart from a code that was never an equity.

## 3. The KAP data is a sample of one year

About 3,400 disclosures at 80 anchors across 2023. Consequences:

- A year is one regime. 2023 in Turkey included an earthquake, a five-day
  market closure, an election and a sharp fall in the lira. Nothing here
  generalises beyond it without more years.
- A sample cannot establish the absence of news, which blocks H5 and H6.
- Each anchor covers about a day of filings, so events cluster on roughly 80
  to 100 dates. That is the effective sample size for anything market-wide.

## 4. Prices come from one free provider

The checks in EVENT_STUDY_METHOD.md found and handle four kinds of defect.
There may be others that leave no trace: a wrong price inside the limit, a
rights issue (bedelli) the provider did not adjust for, a split recorded on
the wrong day. The rights-issue category in H7 is the one most exposed, since
its own event is the thing most likely to be mis-adjusted.

There is no trading-status field. A halted stock looks like a stock with no
bar.

## 5. The price limit rule rests on one announcement

10% for all equity groups from 2020-03-13, "until a further announcement". No
later change was found, and the data agree on 99.97% of stock-sessions.
Instruments with their own margins are not distinguished.

## 6. Timing

- Session hours of 10:00 to 18:10 are assumed for 2023.
- Half days in 2023 are not recorded; afternoons before a closure are flagged
  ambiguous and excluded.
- A disclosure in the last minutes before the close is labelled
  during-session though almost nothing can be traded on it that day.
- With daily bars, a during-session disclosure has no clean first reaction.
  Those disclosures are outside H7's primary sample, which costs about half
  the events.

## 7. Categories are the filer's, and half are a catch-all

More than half of material-event disclosures use the general form. A short
phrase list on the summary line pulls buybacks, contracts, dividends and
capital increases out of it; everything else stays `general_other`. A missed
case dilutes its category. Financial reports are a different disclosure type
and are not in the sample at all.

## 8. Insider purchases

- "Insider" means anyone with a duty to notify trades in the issuer's shares:
  parents and large holders as well as directors.
- A purchase is taken to be on-exchange unless the text says otherwise. No
  field states the venue.
- The parser was written against five real forms and synthetic variants. A
  form whose table is empty (details in an attachment) is left unparsed.
- Transaction value is nominal bought times the midpoint of the disclosed
  price range, assuming a nominal value of 1 TL a share.

## 9. Benchmark

XU100 is a price index; stock returns include dividends. The market model's
intercept absorbs the average difference, but not on ex-dividend dates. There
are no size, sector or momentum factors: no point-in-time data exists here to
build them, so "abnormal" means relative to the index only.

## 10. Costs and fills are assumptions

See EXECUTION_ASSUMPTIONS.md. No order book, no measured fees, no order size.

## 11. Sample-size floors are not power

Meeting "30 events on 15 dates" means a number is worth printing. It does not
mean the test could detect an effect of realistic size. A `not_supported` or
`inconclusive` verdict from a small sample says little.

## 12. The audit was not independent of the author

AUDIT_REPORT.md was written by the author of the code. Its findings are
checkable because each has a test, but a second reviewer would look in places
the first did not think to.

## 13. Sentiment

Where sentiment is used (H1–H6), it comes from a model validated on
market-level headlines against a market-direction rubric. Its meaning for
company news has not been validated. KAP disclosures have no sentiment score.

## What a "supported" verdict would and would not mean

It would mean: in a 2023 sample of surviving issuers, with these prices and
these rules, the registered primary test passed after correction for nine
hypotheses, with an effect at least as large as the registered threshold.

It would not mean: the effect holds in other years, the disclosure caused the
return, or anyone could have earned it.
