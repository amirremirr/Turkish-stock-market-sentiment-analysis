# Results

*Status on 2026-10-11.*

## There are no real-data results yet

The registered run has not happened, and it cannot happen yet: the KAP sample
was 8 of 80 anchors complete when this was written, and the code refuses to
compute an outcome on a partial sample.

```
$ python -m stock_research.cli run
REFUSED  KAP sample frame is incomplete (8 of 80 anchors); no outcome is read on a partial sample
```

| | |
|---|---|
| Protocol | `92366aa0a6342e9c…`, registered 2026-10-11T00:15:04Z |
| Sample frame | `kap-anchor-sample-v2`, 80 anchors across 2023 |
| Ingested so far | 8 anchors, 382 disclosure details |
| Remaining | about 10 hours at 6 calls a minute |
| Real abnormal returns computed | **none** |

When ingestion finishes:

```bash
python -m stock_research.cli ingest-prices   # issuers that appeared since the last fetch
python -m stock_research.cli run             # runs once; writes the report
```

This file is then replaced by the outcome, whatever it is.

## What the sample looks like so far (counts only, no returns)

From `python -m stock_research.cli describe` at 8 of 80 anchors:

| | |
|---|---|
| Sampled disclosures | 351 on 13 distinct sessions |
| Issuer resolved to a price series | 280 |
| Published after the close / during the session / before the open / on a non-session day | 141 / 118 / 59 / 33 |
| Updates or corrections | 32 |
| Naming a company other than the filer | 21 |

| Category | Count |
|---|---|
| general_other | 103 |
| debt_issuance | 65 |
| buyback | 56 |
| administrative | 54 |
| other_form (unmapped form names) | 31 |
| new_contract | 10 |
| dividend | 10 |
| unusual_price_volume | 6 |
| rights_issue | 4 |
| insider_trade | 4 |
| asset_acquisition | 3 |
| bonus_issue | 3 |

Price data, 178 issuers: 74,413 stock-sessions checked against the 10% limit,
3,880 at the limit, 19 beyond it (0.03%), one calendar gap (2023-12-25).

### What these counts suggest, before any outcome

Scaled to 80 anchors, and with the reminder that a projection from 13
sessions is rough:

- **H7** should be runnable. Buybacks, debt issuance, administrative filings
  and the general category will pass the 30-events-on-15-dates rule easily;
  new contracts and dividends probably will; bonus issues, rights issues and
  asset transactions probably will not, and will be reported as too thin.
- **H8 will probably be `data_insufficient`.** Four share-transaction forms in
  351 disclosures projects to about 40 in the full sample. Of the five seen
  across all ingestion so far, two are purchases by a holder, one is a sale,
  one is a company's own buyback and one has an empty table. Around 15 to 20
  purchase days is below the 30 the rule needs.
- The buyback count is high because one anchor falls on 2023-02-16, the day
  after the market reopened from the earthquake closure, when the withholding
  tax on buybacks had just been suspended. Those events share a date and a
  cause; clustering by date is what stops them counting as 50 independent
  observations.

If H8 does come out insufficient, the honest summary of this study will be
that one hypothesis of nine could be tested.

## What has been shown: the estimators work on synthetic data

`python -m stock_research.cli demo` runs each hypothesis on synthetic data
containing the effect it looks for. All nine find it:

| | Verdict on synthetic data with a planted effect |
|---|---|
| H1–H4, H6–H9 | supported |
| H5 | execution_not_verifiable (its registered ceiling) |

**None of this is a finding.** It says the code detects an effect that was put
there. On synthetic data with no effect, the same code does not: across 100
independent null contexts each, the primary tests rejected at 3% (H1), 4%
(H2), 2% (H4), 6% (H5), 8% (H7) and 4% (H8), against a nominal 5%.

## What will be reported, and how

For each hypothesis: the verdict, the primary estimate with its 95% interval,
the raw and Holm-adjusted p-values, sample sizes and cluster counts, every
count along the admissibility chain, the sensitivity analyses, and the
limitations. A `data_insufficient` verdict is reported with its reason and is
not softened into a weak result.

Seven hypotheses will report `data_insufficient` regardless of how the run
goes, for the reasons in DATA_AVAILABILITY.md.
