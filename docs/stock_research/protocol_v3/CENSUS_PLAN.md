# The 2023 KAP census: prepared, not started

Frame version `kap-census-2023-v3`. **No census request has been sent.** The
ingester below exists and is tested on a stand-in gateway; starting it is a
decision for approval (APPROVALS_REQUIRED.md, item 1), because it changes the
registered sampling frame and occupies the gateway's rate limit for about two
days.

## What the census is

Every material-event disclosure (`ODA` class, listing filter `ODA,CA`) filed
by a company that is listed with a stock code in the member roster, with a
disclosure number from 1,091,700 to 1,231,017: the range the development
gateway serves, which is calendar 2023. The range, the listing filter, the
class rule and the roster rule are the ones the anchor sample used
(`stock_research/config.py`, `stock_research/data/kap.py::is_target`). Only
the selection *within* the range changes: all of it, instead of 80 anchors of
50.

Why: the readiness audit found the 80-anchor sample has 80% power only for
spreads of about 3 to 5% with a valid test, and that the classes with
economic content stay below the 30-event minimum
(`docs/stock_research/readiness/POWER_ANALYSIS.md`, W-1 and W-2).

## How it is read

The listing endpoint returns up to 50 disclosures in ascending number from the
number asked for. The census asks for page 0 at the first number of the range,
then for the page that starts one past the last number it was given, until a
page comes back short or reaches the end of the range
(`data/kap.py::record_page`). Pages cannot be laid out in advance, so
`sr_kap_frame` grows by one row per page and `sr_meta` records the last page
once it is known. Each target's detail is then fetched, one call each.

A census has no processing order to choose. An interrupted run has covered
the start of the year and nothing after it; that is harmless because no
outcome may be read until the listing has reached the end and every target
has been asked for (`pipeline.gates`).

## What is reused, and how nothing is rewritten

Every disclosure the anchor sample holds lies in the census range and passed
the same listing filter, so it is a census member.

* `sr_raw_kap_listing` and `sr_raw_kap_detail` are append-only and keyed by
  disclosure number. A row that exists is left exactly as it is: same
  `frame_version` (`kap-anchor-sample-v2`), same block, same payload hash,
  same retrieval time. The census inserts with `INSERT OR IGNORE`.
* Membership of the census is recorded in a new append-only table,
  `sr_kap_frame_items (frame_version, disclosure_index, block_seq, is_target,
  recorded_at)`. A disclosure first listed by the anchor sample gets a row
  there when the census page that contains it is read, and its stored detail
  is not requested again.
* `sr_kap_frame` rows of the anchor frame stay, under their own frame version.
* `sr_ingest_log` gains `start` and `finish` rows for each census run with the
  ingester's hash; the earlier runs' rows stay.
* The manifest of a run reports how many targets were first listed by which
  frame (`provenance.targets_by_frame_that_first_listed_them`).

Test: `test_rows_of_an_earlier_frame_are_reused_and_left_as_they_were`.

One consequence to accept knowingly: about a quarter of the census's details
will have been retrieved in October 2026 by the anchor ingesters
(`62bb9ec0…` and `35b3dd6e…`) and the rest by the v3 ingester. The functions
that decide membership and extract text (`is_target`, `_listed_member_ids`,
`api_get`, `body_text`, `detail_row`) are byte-identical in all three; the
readiness audit checked the first two, and this branch did not touch them.

## Completeness

| | v2 | v3 |
| --- | --- | --- |
| Unavailable details tolerated | 1% of targets | none |
| When a refused detail counts as unavailable | after one attempt | after 3 refusals on 3 separate days |
| What happens then | the run proceeds | the run refuses; proceeding without the disclosure needs review and a new registration |
| Retry cadence | once per ingestion run | once per day, whatever the number of runs |
| Check of stored details | row count | every target detail must have a parseable timestamp and a subject (`kap.verify_details`) |

## Size and cost

From the anchor pages ingested so far: 0.108 targets per disclosure number,
so about 15,100 targets in the range, in about 350 listing pages. About 3,500
of those details will already be stored when the anchor run finishes.

| | Calls |
| --- | --- |
| Listing pages | about 350 |
| Details not yet stored | about 11,600 |
| Total | about 12,000, at 6 a minute: about 37 hours of awake time |

The anchor ingestion (PID 32720) was still running when this was written and
shares the same rate limit. The census should start after it finishes; the two
must not run together.

## How to start it, once approved

The branch was developed in a separate worktree (`../finance-v3`) so that the
running anchor ingestion and the research database were never touched. Git
will not check a branch out twice, so remove that worktree first.

```
git worktree remove ../finance-v3            # from the main checkout; add --force to drop its scratch outputs
git switch research/stock-protocol-v3        # in the checkout that holds stock_research.db and .env
python -m stock_research.cli status           # creates the new tables; reports the census at 0%
python -m stock_research.data.kap run --max-hours 48
python -m stock_research.cli ingest-prices
python -m stock_research.cli ingest-prices-secondary
python -m stock_research.cli gates            # every precondition, passed or not
```

`run` resumes where it stopped. If any refusal is logged, run it again on two
later days. Nothing above reads a return.

## What stays impossible

The gateway serves 2023 only. The census is one year; it is not a sample of
other years, and a result on it does not generalise to them.
