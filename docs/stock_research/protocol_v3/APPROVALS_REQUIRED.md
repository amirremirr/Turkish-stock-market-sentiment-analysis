# Approval checklist before registering v3

Nothing below has been done. v3 is not registered, the census has not been
started, and no real outcome has been computed. Each item is a decision; the
default shown is what the code on this branch does.

## Decisions on the design

| # | Decision | Default on this branch | Alternative | What it costs |
| --- | --- | --- | --- | --- |
| 1 | Replace the 80-anchor frame with a census of 2023 | Census (`kap-census-2023-v3`) | Keep 80 anchors: only buyback and perhaps one other class would be tested | About 37 hours of gateway calls after the anchor run ends |
| 2 | Primary test for H7 | Wild bootstrap of the two-way statistic, 9,999 draws, seed 20261011 | None of the alternatives held its level under issuer recurrence | — |
| 3 | The eight tested classes | buyback, dividend, bonus issue, rights issue, new contract, asset acquisition, asset sale, debt issuance | Add or remove a class now; not after outcomes | Each class is one more contrast |
| 4 | The reference group | Procedural filings only | Pool with unclassified filings (kept as a sensitivity): a larger, less clean baseline. This differs from the wording of the readiness decision, which suggested pooling | Reference of about 350 events in a census instead of about 2,700 |
| 5 | Leaving unclassified filings, insider notifications and mixed capital forms out of the test | Out, counted | Test them as classes, as v2 did | About four in ten filings are outside the test |
| 6 | Class minimums | 30 events / 20 dates / 20 issuers; reference 60 / 30 / 30 | Other values; the bootstrap held its level at 30 / 16 / 20 | Asset sale and possibly bonus issue may fall short |
| 7 | One event per issuer per six sessions | Thinned | Unthinned (a sensitivity) | A daily buyback filer contributes one event a week |
| 8 | Excluding filings dated to the 15 February 2023 reopening | Excluded | Include | 47 of 1,279 sampled disclosures so far |
| 9 | Verdict rule: a significant contrast of at least 1% in addition to the joint test | As stated | Joint test alone | — |

## Decisions on the data

| # | Decision | Default | Note |
| --- | --- | --- | --- |
| 10 | İş Yatırım as the second price source | Yes | A broker's public service, not the exchange. Its data are stored locally and not committed |
| 11 | Tolerance of 1 return point | 0.01 | 1.1% of returns differ by more; 1.8% by more than half a point |
| 12 | A return with no second-source counterpart is missing | Yes | Two stocks so far have no second-source series at all |
| 13 | Complete retrieval, with at most 0.1% accepted after three refusals on three days | Yes | v2 allowed 1% after one attempt |

## Steps that need a go-ahead, in order

| # | Step | Command | Effect |
| --- | --- | --- | --- |
| 14 | Let the anchor ingestion finish | none | It shares the rate limit with the census |
| 15 | Move the main checkout to this branch | `git worktree remove ../finance-v3`, `git switch research/stock-protocol-v3` | The scheduled task `stock-research-sample-ready` will then run v3 code |
| 16 | Create the v3 tables in the research database | `python -m stock_research.cli status` | Adds three tables and their triggers; rewrites nothing |
| 17 | Start the census | `python -m stock_research.data.kap run --max-hours 48` | About 12,000 calls |
| 18 | Fetch prices from both sources | `ingest-prices`, `ingest-prices-secondary` | Network only |
| 19 | Repeat calibration and power on the real census structure | `v3_validation.snapshot`, `calibrate` | Reads no return. If the level is not held, stop |
| 20 | Register v3 | `python -m stock_research.cli register` | Fixes the hash; push the commit that records it |
| 21 | Run | `python -m stock_research.cli run` | The one registered run. Your decision alone |

## What registration would fix

The protocol hash covers every file under `stock_research/` except the
fixtures, the report renderer, the CLI and the registry document; every
constant in CHANGES_FROM_V2.md; the exchange schedule; and digests of the
constants read from the repository-root `config`. Any change to a decision
above after registration is a v4.

## Known limits you would be accepting

* One year of data. A single small class differing by 3% is detected about
  one time in four; `inconclusive` will not rule out effects of 1 to 2%.
* Survivorship: issuers without a served price from both providers are
  absent, about a quarter of sampled issuer codes.
* The census structure is projected from 29 anchors; step 19 exists because
  of that.
* Where the two price sources disagree and no limit settles it, the return is
  dropped, not corrected.
