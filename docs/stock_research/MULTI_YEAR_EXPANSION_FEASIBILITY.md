# Extending the KAP study to 2024–2026: feasibility

Investigated 2026-10-11 on branch `research/multi-year-feasibility`. This is a
feasibility report. Nothing was ingested for any additional year, protocol v3
was not changed or registered, no hypothesis was run and no return was
computed. The 2023 census kept running throughout.

## Answer

| Year | Accessible today with what we have? | Why |
| --- | --- | --- |
| 2024 | **No** | The only gateway our credentials reach serves a fixed 2023 dataset |
| 2025 | **No** | same |
| January–September 2026 | **No** | same; and outcomes from 10 August 2026 are sealed by the index study |

The data exist and MKK distributes them. What is missing is production
access to MKK's KAP data dissemination service, which is obtained by request
and whose terms and price are not published. Nothing can be collected for
2024–2026 until MKK answers.

## 1. What was verified, and how

### The gateway we use is a sandbox with a frozen 2023 dataset

| Fact | Evidence |
| --- | --- |
| The portal describes itself as a test environment | MKK API Developer Portal home page: "build your application and test our APIs through the developer environment"; its FAQ: the portal "enables developers to discover and test APIs in a sandbox environment" (<https://apiportal.mkk.com.tr>, <https://apiportal.mkk.com.tr/faq>, read 2026-10-11) |
| The KAP product has one plan | Product page "KAP Data Dissemination Services": "Plan — Free: Throttling for 6 calls per min"; application approval automatic; specification served from `https://apigwdev.mkk.com.tr/api/vyk?openapi`; contact "Public Disclosure Services Directorate, kapdestek@mkk.com.tr" (<https://apiportal.mkk.com.tr/products/6765bc408e6d51715bccf5a5/detail>) |
| The dataset has not grown | `GET /lastDisclosureIndex` returned `1231017` today, the same value recorded on 2026-06-13 (`docs/kap_api_notes.md`). A live feed adds several hundred disclosures a day; this one added none in four months |
| That last disclosure is from 31 December 2023 | recorded on 2026-06-13 and consistent with every detail ingested since (publication dates 2023-01-02 to 2023-12-31) |

So the limit to 2023 is not a parameter we set and not a restriction on our
account: it is the content of the test environment. No request to it can
return a 2024 disclosure.

### No production endpoint is reachable

Requests **without credentials** to the obvious production host names
(`apigw.mkk.com.tr`, `apigwprod.mkk.com.tr`, `api.mkk.com.tr`,
`apigwtest.mkk.com.tr`) do not resolve in public DNS. The published
specification lists only the dev server. Our credentials were therefore not
tried anywhere else, and whether they would be accepted by a production
gateway is **unknown**. Production access is requested through
`kapdestek@mkk.com.tr`; a request has been drafted earlier in this project
and has not been sent.

### What the service would deliver if production mirrors the sandbox

From the published specification (`kap_openapi.json`) and from 2023:

| Need | Available | Note |
| --- | --- | --- |
| Full disclosure text | yes | `disclosureDetail`, HTML body, plus attachments |
| Company identifiers | yes | sender id, exchange codes, related companies; `/members`, `/memberSecurities` |
| Timestamp to the second | yes | in the detail only, not in the listing |
| Category | partly | disclosure type and class, and the form's subject line; the economic class is assigned by our taxonomy |
| Complete annual sets, not samples | yes in principle | `/disclosures` pages through every disclosure in ascending number; the 2023 census is doing exactly this. `/lastDisclosureIndex` gives the end |
| Withdrawn disclosures | yes | `/blockedDisclosures` |

Whether production has the same endpoints, the same history depth and the
same 6-calls-a-minute throttle is **not verified**. The portal FAQ also
describes quotas per day or month as a possible plan feature; none is stated
for the KAP product.

### Other routes

| Route | Finding | Usable? |
| --- | --- | --- |
| KAP public website | MKK describes KAP as "an electronic archive which allows easy and low-cost access to historical information", with a detailed search for wider periods (<https://kap.org.tr/en/about/general-information>). Individual disclosures of 2024–2026 are public pages | Not for collection. An automated request for its `robots.txt` was answered by an access-denied page that refers the caller to `kapdestek@mkk.com.tr`. The site defends itself against automated clients; collecting a census from it would be working around that, and no terms permitting it were found. Not pursued |
| KAP data dissemination to data vendors | MKK: "Instant sharing of data with data broadcasting organizations via KAP data distribution service" (<https://www.mkk.com.tr/en/corporate-governance-services/kap-public-disclosure-platform>) | This is the production service. Contract terms and fees are not published; the MKK fee schedules found cover KAP *membership* fees for filers, not data recipients |
| Licensed data vendors | They carry KAP disclosures in real time | Possibly, as a purchased archive. No vendor's archive depth, export terms or academic price was verified |
| Borsa İstanbul DataStore | Sells historical *market* data, with academic discounts | For prices, not for disclosures. No KAP archive product was found there |

### Prices for 2024–2026

Both providers serve the period. Checked today for three symbols: Yahoo
returns 249–251 sessions for 2024, 250–251 for 2025 and 186 for January to
September 2026; İş Yatırım returns 250, 251 and 186, and its XU100 series
covers the same range. The two-source reconciliation built for v3 would apply
unchanged. Coverage of the full issuer universe of those years is not
verified.

### A side effect of this investigation

Four requests were sent to the dev gateway while the census was running, of
which only the first returned usable output. They share the census's rate
limit, and the census logged one rate-limit refusal a few minutes later (a
65-second wait, after which its spacing widened from 11 to 13 seconds). No
detail was refused and nothing was lost, but the census was slowed by about
a minute by this work. No further gateway requests were made.

## 2. Feasibility by year

Counts for 2024–2026 are **estimates, not measurements**: no listing for
those years could be requested. They scale 2023, where the census range holds
139,317 disclosure numbers and about 15,100 target disclosures (0.108 per
number). Public disclosure numbers seen in search results are near 1,660,000,
which against 1,231,017 at the end of 2023 suggests about 150,000 to 160,000
numbers a year.

| | 2024 | 2025 | Jan–Sep 2026 |
| --- | --- | --- | --- |
| Disclosure numbers (est.) | ~155,000 | ~155,000 | ~115,000 |
| Target disclosures: material events of listed companies (est.) | ~17,000 | ~17,000 | ~12,500 |
| API calls: one per detail plus ~1 listing page per 43 targets | ~17,400 | ~17,400 | ~12,800 |
| Ingestion time at 6 calls a minute | ~49 h | ~49 h | ~36 h |
| Usable H7 events after v3 eligibility and thinning (est. from the 2023 projection of ~950–1,450) | ~1,100–1,600 | ~1,100–1,600 | ~800–1,200 |
| Price history, both providers | available | available | available |
| Outcome window sealed by the index study | no | no | **from 2026-08-10** |

**Issuer coverage.** For 2023, about a quarter of sampled issuer codes have no
served price (bond-only issuers, and tickers since delisted or renamed). The
roster and both price providers are current, so the share of delisted issuers
falls as the year gets closer to today: coverage should be better for 2025
and 2026 than for 2023, and the survivorship problem smaller. Not measured.

**Sealed window.** `SEALED_INDEX_BOUNDARY` forbids any stock-level outcome on
or after 2026-08-10 until the index study's single sealed result is recorded.
A CAR(0,+5) needs five sessions after day 0, and the registered windows reach
+20. Until the index result exists, 2026 can contribute events with day 0 up
to about mid-July only.

**Data-quality risks specific to the later years.**

* The exchange schedule (half days, holidays, any cancelled or suspended
  session) must be recorded for each year as it was for 2023. Market-wide
  circuit breakers were triggered in this period and need checking against
  the calendar and the price-limit rule.
* The price-limit rule in force (10% since 2020) must be confirmed for each
  year; the cleaning rules depend on it.
* The first provider's bad closing prices were concentrated on days of
  market-wide stress in 2022–2023. The same should be expected in 2024–2026;
  the second-source rule is what catches them.
* KAP form names change. The taxonomy maps form titles and phrases seen in
  2023; new or renamed forms would fall into the unclassified group until the
  taxonomy is extended, and extending it after seeing a year's data is a
  protocol change.
* Share-code changes and new listings: an issuer listed in 2025 has too
  little price history for a 130-session estimation window early on.
* Throttling is not stable: on 2026-10-11 the dev gateway refused about one
  call in four for an hour and a half. Times above assume it does not.

## 3. Research design

2023 is not an untouched sample. Protocol v3 was shaped on it: its class
taxonomy, its price rules, its calendar and its sample-size rules were all
written with sight of 2023's disclosures and prices, though not of any event
outcome. A result on 2023 alone is therefore a first result, not a confirmed
one.

| | A. 2023 only | B. 2023, then 2024–2025 as replication | C. One pooled 2023–2026 study |
| --- | --- | --- | --- |
| Statistical power | Low: one class at 3% detected about one time in four | Replication sample about twice 2023's; pooled 2024–2025 detects roughly 30% smaller effects than one year | Highest: about 3.5 years, effects roughly half the size detectable in one year |
| Independence from development | None | **Full for 2024–2025**: no disclosure, price rule or outcome of those years has been seen | Mixed: a pooled estimate includes the development year |
| Market relevance | 2023 only: an election year with an earthquake closure | Recent | Recent |
| Cost and time | Nothing further | Production access (unknown terms), ~100 h of ingestion, schedules and limits for two years | As B plus 2026, which the sealed window truncates |
| Data completeness | Census in progress | Unknown until access is granted | Unknown; 2026 partial |
| Risk of flexibility | Low | Low if the replication rules are frozen before any 2024 data arrive | Higher: one study, no second look |

**Recommendation: B, with one refinement.**

1. Finish and register the 2023 study as designed (v3), and report it as what
   it is: the development-period study.
2. **Before requesting any 2024–2026 data**, write a replication protocol
   that fixes everything now: the same test, classes, reference group,
   thinning, minimums, price rules and gates as v3; the years (2024 and 2025
   as the confirmatory replication, pooled, with year as a stratum); the
   treatment of 2026 (January to mid-July as a third stratum only if the
   sealed window allows, otherwise excluded); the rule for form names the
   taxonomy does not know (unclassified, no extension); and the reading of
   outcomes (a 2023 finding counts as confirmed only if the replication
   rejects in the same direction).
3. Do not pool 2023 into the confirmatory estimate. A pooled 2023–2026 figure
   can be reported as a descriptive summary, labelled as including the
   development year.

Why not C: its extra power comes partly from the one year that cannot count
as independent, and a single pooled test leaves nothing to check it against.
Why not A: the power analysis already says a null result on 2023 would rule
out very little, and a positive one would rest on a sample the rules were
tuned on.

The order matters more than the choice. The replication protocol is only
worth something if it is frozen while 2024–2026 are still unseen, which they
are today.

## 4. What would have to be approved to begin

Nothing below has been done.

1. **Send the access request to MKK** (`kapdestek@mkk.com.tr`) asking for
   production access to KAP Data Dissemination Services for academic
   research: the history depth available, the rate limit and any quota, the
   licence terms (storage, derived data, publication of results), and the
   fee. It must come from you, as the account holder.
2. **Accept MKK's terms and any fee**, once known. No cost can be stated
   today: none is published.
3. **Decide the design** (A, B or C) and, for B or C, approve the drafting of
   a replication or multi-year protocol *before* any later-year data are
   requested.
4. **Decide what happens to 2026** given the sealed index window.
5. Only then: record each year's exchange schedule and price-limit rule,
   extend the frame constants under a new frame version, and ingest. That is
   a new protocol version; it does not alter v3 or the 2023 census.

If MKK declines or prices access out of reach, the alternatives are a
licensed vendor's archive (terms unverified) or staying with 2023. Collecting
from the public website is not an option this project should take.

## Sources

* MKK API Developer Portal: <https://apiportal.mkk.com.tr>,
  <https://apiportal.mkk.com.tr/faq>,
  <https://apiportal.mkk.com.tr/products/6765bc408e6d51715bccf5a5/detail>
  (read 2026-10-11).
* KAP, general information: <https://kap.org.tr/en/about/general-information>.
* MKK, KAP service page:
  <https://www.mkk.com.tr/en/corporate-governance-services/kap-public-disclosure-platform>.
* MKK fee schedules (KAP membership fees for filers, not data access):
  <https://www.mkk.com.tr/sites/default/files/2025-02/MKK-Ucret-Tarifesi-03-02-2025.pdf>,
  <https://www.mkk.com.tr/sites/default/files/2025-10/2025_Yili_KAP_Hizmet_Bedeli_Tarifesi.pdf>.
* Borsa İstanbul historical data sales and academic pricing:
  <https://www.borsaistanbul.com/sayfa/2816>.
* This repository: `kap_openapi.json`, `docs/kap_api_notes.md`,
  `docs/stock_research/DATA_AVAILABILITY.md`,
  `docs/stock_research/protocol_v3/VALIDATION.md`.

## Not verified

The existence, endpoints, history depth, throttle and price of a production
KAP gateway; whether our credentials work on it; disclosure counts for
2024–2026; issuer price coverage for those years; any vendor's archive; the
KAP website's terms of use (not found).
