# KAP ingestion completeness

Audit output of 2026-10-11. Counts, identifiers, timestamps and digests read
from read-only copies of the research database (SQLite backup API). The
ingestion process was not stopped, restarted or reconfigured. No price was
joined to a disclosure for this document.

Reproduce: `python -m readiness_audit.make_structure` (takes the copy), then
`python -m readiness_audit.kap_checks`.

## State of the sample

Two copies were taken while ingestion ran: at 11:59 (the one every structural
number in this audit is based on) and at 12:33 local time.

| Item | 11:59 copy | 12:33 copy |
| --- | --- | --- |
| Anchors in the frame | 80 | 80 |
| Anchors attempted | 14 | 19 |
| Anchors complete (every target detail stored) | 14 | 19 |
| Anchors `incomplete_details` | 0 | 0 |
| Listing items in attempted anchors | 700 (50 each) | |
| Target disclosures | 611 | 826 |
| Details retrieved | 611 | 826 |
| Missing or unavailable disclosure IDs | none | none |
| Refused requests logged (`sr_kap_detail_failures`) | 0 | 0 |
| Retry count, any disclosure | 0 | 0 |
| Placeholder rows from the earlier ingester | 0 | 0 |
| Timestamps that do not parse | 0 | |
| Empty body text | 0 | |
| Missing subject | 0 | |
| Timestamps out of order against disclosure number | 0 | |

The process (PID 32720, started 11:39:44, 36-hour budget) was healthy at every
check: one anchor about every nine minutes, no refusal. **The frame is 24%
ingested. No real-data run is possible yet, and none was attempted.**

Targets per anchor: 37 to 49, median 44.5. A listing item is not a target if
its class is not a material-event class (47 of 700) or its filer is not in the
listed-company roster (42 of 700, from 18 filers).

## Source identity and digests

| Item | Value |
| --- | --- |
| Ingester running now | `data/kap.py`, SHA-256 `35b3dd6ea9ef852ace77bae9b0bcbe38166ea1757935c2e931107f5497215fe9`, commit `2f0054c`; logged at start (`sr_ingest_log`) and equal to the file on disk and to the v2 fingerprint entry |
| Ingester of the first 12 anchors (555 details) | SHA-256 `62bb9ec0fdd8…`, commit `ecdc532`; attested after the fact, since that version wrote no log |
| Do the two versions select and read the same thing? | Yes. `frame_blocks`, `is_target`, `_listed_member_ids`, `api_get`, `body_text` and `detail_row` are byte-identical in the two commits; the changes are in failure handling, completeness accounting and logging |
| KAP sample digest (11:59 copy, 646 details) | `d51d7b0bdfd3d75b2a4dc7d3445d306a713635fffb3770c9226edfeeaed45d5f` |
| Member roster digest (1,014 rows) | `a83a76612f043d6b9725ccfd56f2b209798fe732470d5eee193bb8493a463d75` |
| Price snapshot | `yahoo-2022-06-01-2024-03-01`; its digest changes as issuers are added (see K-2) |

The sample digest is of a partial sample and will change until ingestion ends.
What matters is that the run records it (`stock_research/pipeline.py:265-306`).

## Parser and mapping results (11:59 copy, 646 sampled disclosures)

| Item | Count |
| --- | --- |
| Classified by form name | 354 |
| General form, no phrase matched (`general_other`) | 203 |
| Form name not in the taxonomy (`other_form`) | 67 |
| Capital forms and share-transaction forms, by rule | 16 |
| General form, phrase matched | 6 |
| Insider notifications | 4: 3 parsed, 1 unparsed |
| Issuer resolved to a priced ticker | 516 |
| **Unresolved** | **130 (20.1%), from 58 filers** |

Unresolved share by category: debt issuance 65% (83 of 127), insider trade 25%
(1 of 4), asset acquisition 17%, other forms 15%, rights issue 14%,
administrative 10%, dividend 9%, general 8%, buyback 6%, and none of new
contract, asset sale, price query or bonus issue.

## Findings

### K-1. One fifth of sampled disclosures have no priced ticker, unevenly by category

* **Severity:** medium.
* **Evidence:** table above. Debt-issuance filings are mostly by issuers
  whose code is not a share the provider serves (bond-only issuers, banks'
  financing vehicles); they lose two thirds of their events. Other categories
  lose 6 to 17%.
* **Source:** `stock_research/events.py:193-217` (no line with 20 priced
  sessions before day 0 means no ticker), and the provider serving no data for
  90 of 344 requested symbols.
* **Impact.** Within a category, the events that remain are those of issuers
  with listed, traded, still-served shares. The provider does not serve
  delisted tickers and the roster is today's, so a company that left the
  exchange after 2023 is absent on both counts: survivorship in the direction
  of healthier issuers. The size of that group among the 58 unresolved filers
  was not established; most of the unresolved volume is debt issuance.
* **Recommended action.** Report unresolved counts by category with the
  result (the run already counts them). Before a run, classify the unresolved
  filers into "never had a listed share" and "share no longer served", from
  the roster and the second price source, so the survivorship part has a
  number.
* **Protocol amendment:** no for reporting; yes if a second price source is
  used to recover them.

### K-2. Nothing checks that prices were fetched for every sampled issuer

* **Severity:** medium (a run-order hazard that silently shrinks the sample).
* **Evidence.** Prices are fetched by a separate command for the codes among
  disclosures stored *at that moment* (`stock_research/pipeline.py:60-68`). At
  12:33 the production snapshot held 237 symbols while the sample named 404;
  168 had never been requested. An issuer without a price frame gets no
  ticker (`events.py:207-209`) and its events drop out as "unresolved". The
  run checks registration, ingestion and the missing-detail share
  (`pipeline.py:313-335`); it does not check price coverage.
* **How to verify:** on a copy, compare `events.candidate_symbols()` with
  `sr_price_availability`.
* **Impact.** Running straight after ingestion finishes would drop every
  issuer first seen in a later anchor: at the 11:59 copy, 75 issuers that the
  provider does serve.
* **Recommended action.** Run `ingest-prices` after ingestion completes and
  confirm zero never-requested symbols before any run. In a future protocol
  make the run refuse otherwise.
* **Protocol amendment:** yes for the gate; no for the manual step.

### K-3. A refused detail is asked for once per ingestion run

* **Severity:** low.
* **Evidence.** A refusal is logged and the disclosure is skipped for the
  rest of that run (`stock_research/data/kap.py:392-405`, with `visited` at
  `:359-366`); it is asked for again only when ingestion is run again, up to
  three attempts in total (`:336-343`). The run gate needs every anchor
  attempted and at most 1% unavailable; it does not need three attempts.
* **Impact.** After one long ingestion run, a disclosure refused once counts
  as unavailable although two more attempts were registered as allowed.
* **Recommended action.** If any refusal is logged, rerun `ingest-kap` twice
  on later days before the analysis.
* **Protocol amendment:** no.

### K-4. The roster is today's, not 2023's

* **Severity:** low to medium; part of K-1.
* **Evidence.** A disclosure is a target only if its filer is a listed
  company with a stock code in the member snapshot retrieved in October 2026
  (`kap.py:213-227`). 42 of 700 listing items in material-event classes are
  from 18 filers not in that roster. Some are filers that were never listed
  (funds, intermediaries); some may be companies listed in 2023 and not now.
* **Recommended action.** As in K-1.
* **Protocol amendment:** no.

## The 1% tolerance: five questions

**1. Can missing disclosures systematically bias H7?** Yes, in principle.
A refusal is a property of the disclosure (its size, attachments, a later
withdrawal), not a coin toss, so it can be related to category. And 1% is
loose against the small categories: 1% of about 3,500 targets is 35
disclosures, and the smallest category H7 will test has about 40 events.
If refusals concentrated in one small category the tolerance would allow
most of it to be missing.

**2. Are certain categories disproportionately unavailable?** It cannot be
seen from the refused items themselves: category comes from the detail, and a
refused disclosure has none. Its class, filer and listing title are known and
can be compared. So far the question is empty: 0 of 826 targets were refused.

**3. Is zero missingness achievable?** On the evidence so far, yes. With 0
refusals in 826 requests the 95% upper bound on the refusal rate is 0.36%,
which over about 3,500 targets is at most a dozen, each with two further
attempts available.

**4. Should the first confirmatory study require complete retrieval?** Yes.
The tolerance was a reasonable guard when refusals were feared; the data show
they are rare. A confirmatory test with categories of 40 events should not
have a gap it cannot describe. Recommended for a future protocol: zero
unavailable target details, or else every unavailable ID listed with three
attempts on three days and the sensitivity analysis below registered in
advance.

**5. If missing observations are accepted, what sensitivity analysis is
needed?** (a) Class, filer and title of each missing disclosure against the
retrieved ones. (b) The test repeated without any anchor that has a missing
detail. (c) A bound: assign the missing disclosures to the smallest tested
category with outcomes at the 5th and at the 95th percentile of that
category's CARs, and report whether the verdict survives both. (d) The count
of missing disclosures per anchor, since one bad hour at the gateway is one
date cluster.
