# Statistical protocol

The rules below were fixed before any real outcome was read. They are
fingerprinted: `python -m stock_research.cli status` prints the protocol hash,
which covers the nine specifications, every rule version and constant, and a
SHA-256 of each source file that can change a result.

## Order of events

1. Sample frame fixed by constants (`kap-anchor-sample-v2`).
2. Estimators written and tested on synthetic data.
3. Independent audit; fixes (AUDIT_REPORT.md).
4. Protocol registered (`register`).
5. Ingestion completes.
6. One run on real data (`run`), stored under its manifest.

Steps 1 to 4 happened without a real abnormal return being computed. The code
enforces the order: step 6 refuses unless step 4 was done for the current
code and step 5 is finished. There is no interim look.

## Primary and exploratory

Each hypothesis has **one** primary test, named in its specification. Those
nine form the primary family. Everything else a hypothesis reports is
exploratory or a sensitivity analysis, is labelled so, and feeds no verdict.

## Multiple testing

- **Primary family: Holm.** The family size is nine, fixed. A hypothesis that
  cannot run contributes no p-value but still counts toward the nine. Thin
  data must not make the correction lighter.
- **Exploratory families: Benjamini-Hochberg**, within a hypothesis (H7's
  category-by-window grid).

## Verdicts

| Verdict | Condition |
|---|---|
| `data_insufficient` | A required dataset is unavailable or a sample-size rule fails. Nothing is estimated. |
| `supported` | Hypothesised sign, Holm-adjusted p below 0.05, and estimate at least the registered material effect. |
| `execution_not_verifiable` | As `supported`, for a claim that depends on a fill the data cannot confirm (H5). |
| `not_supported` | The 95% interval, read in the hypothesised direction, lies entirely below the material effect. |
| `inconclusive` | Everything else. |

Consequences worth stating plainly:

- An unadjusted p below 0.05 is not support.
- A significant effect smaller than the material effect is `not_supported`
  or `inconclusive`, never `supported`.
- A joint test (H7) has no interval, so it can be `supported` or
  `inconclusive`, never `not_supported`.
- A verdict from synthetic data, or from an incomplete frame, carries
  `claim_allowed = false` and is shown as such.

## Sample-size rules

| | Rule |
|---|---|
| Any group compared | at least 30 events on at least 15 dates |
| Any clustered test | at least 10 clusters in every clustering dimension, or no p-value |
| H2 | 100 events on 30 dates |
| H3 | 200 matched events, 30 test dates |
| H4 | 200 ticker-days on 30 dates |
| H6 | 300 at-risk sessions, 50 of each outcome, 20 test dates |
| H7 | 3 categories that each meet the group rule |
| H9 | 100 matched events on 30 dates |

These are floors below which a number is not worth reporting. They are not
power calculations, and meeting them does not mean a test can detect a
realistic effect.

## Dependence

Stock events are dependent across issuers on a day and within an issuer over
time. Primary tests with multi-day outcome windows cluster two ways, by day 0
and by issuer. EVENT_STUDY_METHOD.md gives the simulation that shows why.

## Predictive claims (H3, H6)

- Chronological split: the first 70% of dates train, the last 30% test. One
  split, fixed in advance; nothing is tuned on the test set.
- **Embargo:** training events whose outcome window had not closed before the
  first test date are dropped (5 sessions for H3, 1 for H6).
- Preprocessing (standardisation in H6) is fitted on the training set only.
- The comparison is a paired loss difference on the same test events, with
  clustered uncertainty.
- H6 also reports the Brier score, AUC and a calibration table.

## Effect sizes

Every primary block reports the estimate and its 95% interval, not only a
p-value. Each hypothesis registers a **material effect**: the size below
which the effect would not matter even if real.

| | Material effect |
|---|---|
| H1 | 0.5% of signed drift, small minus large caps |
| H2 | 5% of the gap continuing intraday |
| H3, H6 | any out-of-sample improvement |
| H4 | 0.2% of CAR per unit of log abnormal coverage |
| H5 | 0.5% of CAR(+1,+3) |
| H7 | 1% spread between category means |
| H8 | 1% of CAR(+1,+20) |
| H9 | English first in 55% of matched events |

## Diagnostics reported with every run

- **Missingness:** events lost at each admissibility step and by study status.
- **Survivorship:** symbols the provider could not serve.
- **Leakage:** covered by tests, listed in AUDIT_REPORT.md.
- **Overlap:** thinned and date-only variants beside the primary result.
- **Price data:** defects by type, calendar gaps, and the price-limit
  consistency check.

## What may change, and how

Any change to a rule changes the protocol hash. Before the first real run a
new hash is simply registered. After it, a new hash is a new study: its
results are stored beside the old ones, never over them, and a report must
say which protocol produced it.
