# Every rule changed from v2 to v3

v2 is `stock-research-protocol-v2`, hash `09e9382e…`, registered
2026-10-11T08:39:14Z, never run. v3 is a candidate and is **not registered**.
Finding ids refer to `docs/stock_research/readiness/`. Evidence for each
change is in VALIDATION.md.

Rules not listed here are unchanged: the event-study engine, the market model
and its windows, the benchmark, day-0 assignment on ordinary sessions, the
taxonomy that assigns a class to a filing, the insider parser, the Holm family
of nine, the specifications of H1–H6, H8 and H9, and the sealed-window guard.

## Inference (H7)

| # | v2 | v3 | Why | Where |
| --- | --- | --- | --- | --- |
| 1 | Joint p-value from F(q, G−1) on the two-way clustered Wald statistic | Wild cluster bootstrap of the same statistic: null imposed, Rademacher weights by day 0, 9,999 draws, seed 20261011; p = (1 + draws at least as extreme) / (1 + draws) | I-1: v2 rejected a true null 9.6% at 5% and 2.4% at the Holm level; v3 4.4–5.3% and 0.43–0.70% across eleven designs | `stats.wild_cluster_wald`, `h7.contrasts` |
| 2 | Events of one issuer with overlapping windows all kept; thinning a sensitivity | An issuer's eligible events thinned to one per six sessions (the earliest kept); the unthinned analysis is a sensitivity | Overlap is the dependence date-level procedures miss (22% rejection in the controlled design) | `h7.thin` |
| 3 | Sensitivities: all buckets, market-adjusted, thinned, day-0 clustering | All buckets; market-adjusted; unthinned; isolated events; pooled reference; date-statistic bootstrap; CR2/Satterthwaite; the v2 F test (marked as over-rejecting) | Report the alternatives that were evaluated | `h7.run` |

## What H7 tests

| # | v2 | v3 | Why | Where |
| --- | --- | --- | --- | --- |
| 4 | Any class with 30 events on 15 dates, against the largest class | Eight named classes (buyback, dividend, bonus issue, rights issue, new contract, asset acquisition, asset sale, debt issuance), each one contrast against a reference group | W-2: the classes that qualified were mostly residual | `h7.INTEREST` |
| 5 | No reference group | Reference group: procedural filings (`administrative`) | A contrast against filings with no transaction removes the level the benchmark leaves behind (M-2) | `h7.REFERENCE_CLASSES` |
| 6 | General-form filings with no matched phrase, unmapped forms, mixed capital forms and insider notifications were tested classes | Outside the test, counted, with the reason recorded; pooled into the reference in one sensitivity | Content unknown, or direction depends on the side (H8's subject) | `h7.NOT_TESTED` |
| 7 | Class minimum: 30 events on 15 dates | 30 events on 20 dates from 20 issuers; reference group 60 on 30 from 30 | I-2: a class could qualify on a handful of issuers | `config.H7_*`, `h7.sufficient` |
| 8 | At least three classes | At least one tested class and a sufficient reference group | The test is of contrasts, not of a spread | `h7.SPEC.sufficiency` |
| 9 | Size reported: highest class mean minus lowest | Each contrast with its standard error and bootstrap p-value (Holm across contrasts); the reported size is the largest contrast among those individually significant; a bias-corrected root-mean-square contrast beside it | W-3: the spread exceeded 1% in 85% of null samples | `h7.contrasts` |
| 10 | `supported`: Holm p < 0.05 and spread ≥ 1% | `supported`: Holm p < 0.05 and a significant contrast of at least 1% | False `supported` under the null 0.5% against 2.9% | `common.decide` (unchanged code; the estimate it reads changed) |
| 11 | Filings dated to the 15 February 2023 reopening included | Excluded (`closure_delayed`) | C-1: 17% of usable events in the first snapshot were the reopening week | `calendar._reopening`, `h7.prepare` |

## Sample

| # | v2 | v3 | Why | Where |
| --- | --- | --- | --- | --- |
| 12 | Frame `kap-anchor-sample-v2`: 80 anchors of 50 | Frame `kap-census-2023-v3`: every target disclosure in the same range, read page by page | W-1 | `config.KAP_FRAME_*`, `data/kap.py` |
| 13 | Membership by `frame_version` on the listing row | Membership in `sr_kap_frame_items`; rows of earlier frames reused and not rewritten | Preserve raw observations and provenance | `store`, `kap.record_page` |
| 14 | Up to 1% of details may be unavailable after one attempt | None. A refused detail is retried on 3 separate days; if any is still missing the run refuses and the case goes to review | Completeness questions 1 and 4 of the audit | `config.KAP_MAX_UNAVAILABLE_SHARE`, `KAP_MIN_ATTEMPT_DAYS`, `kap.progress` |
| 15 | A refused detail retried once per ingestion run | Once per day | K-3 | `kap._may_ask` |
| 15a | Fixed 11 s between calls; a rate-limit refusal waits 65 s and uses up one of five attempts | Spacing widens by 2 s after a refusal (ceiling 30 s) and narrows by 1 s after 30 accepted calls (floor 11 s); the gateway's `Retry-After` is honoured; a refusal does not use up an attempt | On 2026-10-11 the gateway began refusing one call in four, tripling the time per call. Pacing changes how fast the sample arrives, not what is in it | `kap.Pace`, `config.KAP_THROTTLE_*` |
| 16 | Stored details counted | Stored details verified: parseable timestamp, subject, payload hash | A detail with no timestamp has no day 0 | `kap.verify_details` |

## Prices

| # | v2 | v3 | Why | Where |
| --- | --- | --- | --- | --- |
| 17 | One provider | A second provider's closes and capital actions stored beside the first | A-1, A-2 | `data/secondary.py`, two new append-only tables |
| 18 | A return is kept if it passes the cleaning rules | Also only if the second provider's adjusted return is within 1 point; otherwise missing (`source_disagreement`), and missing too when there is nothing to compare (`no_second_source`) | A-1: 1,611 wrong returns, 26 caught by v2 | `prices.derive`, `config.RECONCILE_TOLERANCE` |
| 19 | Action dates exempt from the jump and limit tests | Adjusted return on an action date tested against the price limit (tolerance 1 point) | A-2 | `prices.ACTION_LIMIT_TOLERANCE` |
| 20 | No comparison of corporate-action records | Reported with every run | A-3, A-4 | `prices.action_consistency` |

## Calendar

| # | v2 | v3 | Why | Where |
| --- | --- | --- | --- | --- |
| 21 | 2023 half days inferred from the next weekday being closed | Recorded: 20 April and 27 June; holidays, the cancelled session and the closure recorded too | C-2, C-3: six full sessions wrongly treated as half days | `calendar.EXCHANGE_SCHEDULE` |
| 22 | Half-day cut at 12:30, flagged ambiguous | Close at 12:40, not ambiguous | C-4 | same |
| 23 | No check of sessions against a schedule | The data's 2023 sessions must equal the recorded schedule, apart from known benchmark gaps | Calendar integrity | `SessionCalendar.schedule_mismatches` |

## Gates and provenance

| # | v2 | v3 | Why | Where |
| --- | --- | --- | --- | --- |
| 24 | Checks in `run_real` only | Gates evaluated in one place and carried by the real-data context; every hypothesis refuses a real context whose gates have not passed | P-1 | `pipeline.gates`, `common.base_result` |
| 25 | No price-coverage check | Refuses unless both providers were asked for every sampled issuer code | K-2 | `pipeline.price_coverage` |
| 26 | Root-`config` constants outside the hash | Digested into the protocol document | P-3 | `registry.external_constants` |
| 27 | Manifest without library versions, second-source digests or gates | All three recorded; targets counted by the frame that first listed them | P-4, P-5 | `pipeline.environment`, `dataset_provenance` |
| 28 | Protocol document names its predecessor | Names the full lineage (v1, v2) | Continuity | `registry.LINEAGE` |

## Versions

`PROTOCOL_VERSION` v2 → v3; schema v1 → v2; calendar rule v1 → v2; price rules
v2 → v3; frame as above. Taxonomy, entity linker, insider parser, limit rule,
cost model and engine versions are unchanged.

## Not changed although the audit raised it

* The benchmark (M-1) and the price index against dividend-adjusted stocks
  (M-3): the contrast against a reference group addresses the level problem;
  a different benchmark was not asked for.
* The roster of listed companies is still today's (K-4).
* Prices for 2023 were in the database before any protocol (P-2). True of v3
  as well; no event outcome has been computed under any version.
