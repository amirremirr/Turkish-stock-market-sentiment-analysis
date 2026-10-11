# Trading calendar and disclosure timing

Audit output of 2026-10-11. No return was read for this document; it uses
session dates, publication timestamps and the timing code.

## What was checked

| Item | Result |
| --- | --- |
| Sessions of 2023 | Correct. The calendar is the set of dates the benchmark has a bar on. Its 2023 weekday closures are 9, 10, 13, 14 February, 21 April, 1 May, 19 May, 28–30 June and 30 August, which is the official list plus the earthquake closure. |
| Cancelled session | Correct. 8 February 2023 is removed from every series (`stock_research/data/prices.py:51-60`, `:309-314`). The second price source has no bar for that date either. |
| Missing benchmark session | 25 December 2023 was a real session: 252 of 252 stocks have a second-source bar for it and so does the second source's index series. Yahoo's index has none. The panel folds it into 26 December for stock and market alike (`prices.py:320-332`). |
| Half days | 20 April and 27 June 2023 (arife). Both are caught by the rule. Six full sessions are wrongly treated as half days (finding C-2). |
| Closing time | Continuous trading ended at 18:00 and the closing session ran to 18:10 on full days; on half days continuous trading ended at 12:30 and the closing session ran to 12:40. The code uses 18:10 and 12:30 (`stock_research/calendar.py:42-43`). |
| Time zone | Correct. KAP stamps are parsed as `Europe/Istanbul`; aware timestamps are converted before comparison (`calendar.py:65-71`, `:143-145`). |
| After-close filings | Assigned to the next session (`calendar.py:170`). Verified at every boundary by test. |

Sources: Borsa İstanbul, "2023 Yılı Resmi Tatil Günleri Nedeniyle Borsa Piyasa
ve Pazarlarında Uygulanacak İşlem, Tescil ve Takas Esasları", no.
E-18454353-100.04.02-13031, 7 December 2022
(<https://borsaistanbul.com/files/2023_tatil-duyurusu.pdf>); member notices
relaying its equity-market table for 20–21 April and 27–30 June 2023
(<https://www.denizyatirim.com/asset/AnnouncementsPdf/1681722855.pdf>,
<https://www.denizyatirim.com/asset/AnnouncementsPdf/1687247589.pdf>). The
official table itself (an attachment to the announcement) could not be
retrieved; the cover letter and the member notices could.

## Sessions from data against the official list

Weekdays with no session in the price range (2022-06-01 to 2024-02-29), from
the panel built on the snapshot copy:

2022-07-11, 2022-07-12, 2022-07-15, 2022-08-30, 2023-02-08 (cancelled),
2023-02-09, 2023-02-10, 2023-02-13, 2023-02-14, 2023-04-21, 2023-05-01,
2023-05-19, 2023-06-28, 2023-06-29, 2023-06-30, 2023-08-30, 2023-12-25 (data
gap, merged), 2024-01-01.

Each is an official holiday, the earthquake closure, or the documented data
gap. No session is missing without a reason and no holiday carries a bar.

## Where sampled disclosures fall (metadata only, snapshot of 646)

| Publication time | Count | Treatment |
| --- | --- | --- |
| During the session | 339 | day 0 same session; excluded from H7's primary sample |
| After the close | 194 | day 0 next session |
| Before the open | 78 | day 0 same session |
| Non-session day | 35 | day 0 next session |
| 18:00–18:10 (closing session) | 13 | counted as during the session, so excluded from the primary sample |
| 09:40–10:00 (opening session) | 14 | counted as before the open |
| Flagged timing-ambiguous | 0 | |
| Published 8–14 February 2023 | 33 | day 0 is 15 February (finding C-1) |

## Findings

### C-1. Filings made during the February 2023 closure are dated to the reopening day

* **Severity:** medium.
* **Evidence:** 33 of the 198 events usable for H7 so far (17%) have day 0 on
  2023-02-15, seven or eight calendar days after publication. They come from
  one anchor that fell on 7–8 February. Their CAR(0,+5) is the return of the
  week the market reopened after a five-session closure, for every stock at
  once.
* **Source:** `stock_research/calendar.py:156-157` (a non-session publication
  goes to the next session, however far away).
* **How to verify:** `outputs/stock_research/readiness/h7_structure.csv`,
  count rows by `day0`; join `kap_events.pkl` on `event_id` for the
  publication date.
* **Impact:** the rule is applied correctly, and all 33 events are one date
  cluster, so the inference does not count them as 33 independent outcomes.
  But the outcome is not a reaction to the filing in any usual sense, and one
  cluster with an extreme common return moves the mean of every small category
  that has members in it. In a full sample of about 1,000 events they are 3%.
* **Recommended action:** in a future protocol, exclude events whose day 0 is
  more than a stated number of calendar days after publication because of an
  exchange closure, or register the test without that date as a sensitivity.
* **Protocol amendment:** yes, if adopted (an eligibility rule).

### C-2. Six full sessions of 2023 are treated as half days

* **Severity:** low.
* **Evidence:** for years without a recorded half-day list the rule assumes an
  early close whenever the next weekday has no session
  (`calendar.py:132-137`). In 2023 that flags 7 February, 28 April, 18 May,
  29 August, 22 December and 29 December, all of which ran to 18:10, besides
  the two real half days.
* **Effect:** a filing between 12:30 and 18:10 on those dates is moved to the
  next session and flagged `timing_ambiguous`; H7 drops flagged rows
  (`stock_research/hypotheses/h7.py:79`). Day 0 is delayed, never advanced, so
  no event is given a session it could not have traded in. In the snapshot no
  disclosure is affected: 25 were published on two of those dates, all before
  12:30 or after 18:10.
* **How to verify:** `tests/stock_research/test_readiness.py::
  test_full_sessions_before_a_closure_are_wrongly_assumed_half_days`.
* **Recommended action:** record the 2023 half days (20 April, 27 June) and
  holidays explicitly in a future protocol instead of inferring them.
* **Protocol amendment:** yes, if adopted (it changes the calendar rule).

### C-3. The inferred half-day rule would miss a half day before a weekend holiday

* **Severity:** low; no effect on the 2023 sample.
* **Evidence:** 28 October 2022 was a half day (29 October fell on a
  Saturday). The next weekday was a session, so the rule does not flag it. In
  2023, 28 October was itself a Saturday, so no 2023 session is affected.
* **Recommended action:** same as C-2.

### C-4. The half-day cut is at 12:30; the closing session ran to 12:40

* **Severity:** low.
* **Effect:** a filing between 12:30 and 12:40 on a half day is treated as
  after the close and flagged. Conservative, and such rows are dropped.
* **How to verify:** `test_half_day_closing_auction_is_treated_as_after_the_close`.

### C-5. Filings during the closing session count as "during the session"

* **Severity:** informational.
* **Effect:** 13 sampled disclosures were published between 18:00 and 18:10,
  5 of them after 18:05. They are excluded from H7's primary sample along with
  every other during-session filing, which is the safe side. A filing at
  09:57 counts as before the open although the opening price was already set;
  for a close-to-close window that is right, because the return starts from
  the previous close.

### C-6. The calendar rule reads holiday lists from a file outside the fingerprint

See PROTOCOL_INTEGRITY.md, finding P-3 (`calendar.py:125`).

## Tests added

`tests/stock_research/test_readiness.py`, on a calendar built from the
official 2023 closure list: fourteen boundary cases (before the opening
session, 09:59:59, 10:00:00, mid-session, closing session, 18:10:00,
18:10:01, late evening, Friday evening, Saturday, an official holiday, Kurban
Bayramı, the cancelled session, the earthquake closure); a sweep over the
year asserting day 0 is never before publication; both half days; the
12:30–12:40 window; the four wrongly flagged full sessions; UTC and naive
timestamps.
