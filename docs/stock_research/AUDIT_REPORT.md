# Independent audit of the stock-level implementation

*Carried out 2026-10-11 on the H1–H9 implementation in `stock_research/`,
before the protocol was registered and before any real abnormal return had
been computed. Every confirmed finding was fixed and has a regression test.
Line numbers refer to the code as fixed.*

## Scope and method

The review read the implementation, not its documentation, and tried to break
it: by simulating data with no effect and measuring how often each test
claimed one, by feeding it real KAP forms and real provider bars, and by
checking each rule that is supposed to stop information from the future
reaching a test.

**It was not independent in one respect that should be stated.** The same
author wrote the implementation and the audit. What makes the findings
checkable is that each has a test that fails on the original code.

No real outcome was read during the audit. The provider's price bars were
examined for defects, and KAP form text was read to check parsing; neither
involves an abnormal return.

## Summary

| Severity | Count | Fixed |
|---|---|---|
| Critical | 1 | 1 |
| High | 8 | 8 |
| Medium | 7 | 6 (one is in code outside this module and is recorded, not changed) |
| Low | 6 | 2 fixed, 4 recorded as limitations |

The critical finding would have invalidated the two hypotheses that can run on
real data. It was found by simulation, not by inspection.

## Critical

### C1. Date-only clustering treats one issuer's overlapping windows as independent

- **Where:** primary tests of H1, H3, H4, H5, H6, H7, H8 (originally
  `mean_test(values, dates)` and `regress(..., ["day0"])`).
- **Mechanism:** a holder who buys on ten consecutive days produces ten
  events whose 20-session windows share most of their returns. Clustering by
  event date assumes events on different dates are independent. They are
  nearly the same number repeated.
- **Measured:** 300 simulated panels with no effect, 12 issuers each filing on
  10 nearby sessions, CAR(+1,+20). Date-only clustering rejected the true null
  **48%** of the time at the 5% level. Clustering by date and issuer rejected
  **6.7%**.
- **Test:** `test_audit.py::test_overlapping_windows_need_issuer_clustering`.
- **Fix:** primary tests with multi-day windows cluster two ways, by day 0 and
  by issuer (`hypotheses/common.py:245`, `TWO_WAY` at `:268`). Date-only
  results are kept as a labelled sensitivity. H8 also reports a version with
  one purchase per issuer per 21 sessions (`eventstudy.py:252`).

## High

### H1. Holder share trades were attributed to the filer, not the traded company

- **Where:** `events.py:247`, `insider.py:98`.
- **Mechanism:** on the "Pay Alım Satım Bildirimi" form the filer is often a
  parent or shareholder, and the company whose shares were traded is named in
  the "related companies" field. The event builder used the filer's ticker. A
  purchase of İş Finansal Kiralama shares by İş Bankası would have entered the
  study as an İş Bankası event.
- **Found by:** reading the first real form in the sample (disclosure 1114477).
- **Tests:** `test_components.py::test_holder_purchase_is_attributed_to_the_related_issuer`,
  `::test_event_builder_uses_related_issuer_for_holder_trades_and_prior_liquidity`.
- **Fix:** for share transactions the ticker comes from the related company
  when the filer is a different entity; if the form names more than one, no
  ticker is assigned.

### H2. Company buybacks were counted as insider purchases

- **Where:** `events.py` (`classify`), `insider.py:98` (`transaction_kind`).
- **Mechanism:** companies report trades in their own shares on the same form
  (disclosure 1114489, Net Holding). H8 excludes buybacks; the form name alone
  did not.
- **Test:** `test_components.py::test_an_issuers_own_buyback_on_the_share_form_is_not_an_insider_trade`.
- **Fix:** a filer trading its own shares is classified as a buyback.

### H3. H7 could be "supported" for mechanical reasons

- **Where:** `hypotheses/h7.py:28` (`PRICE_TRIGGERED`), `prepare`.
- **Mechanism, two parts.** (a) A company answering an exchange query about
  unusual price moves files *because* the price moved, so that category has an
  abnormal return by construction. (b) For a disclosure published during the
  session, the day-0 close-to-close return includes everything before
  publication. Either can make category means differ with no reaction to any
  disclosure.
- **Tests:** `test_audit.py::test_h7_ignores_price_triggered_disclosures`,
  `::test_h7_primary_uses_only_events_whose_day0_starts_before_publication`.
- **Fix:** the primary sample is limited to disclosures published before the
  open, after the close or on a non-session day, and excludes price-triggered
  categories. All timing buckets together are a sensitivity.

### H4. H3 and H6 trained on outcomes that were not yet known

- **Where:** `hypotheses/common.py:271` (`chronological_split`).
- **Mechanism:** the split was by event date. A training event five sessions
  before the first test date has an outcome window that closes *after* the
  test period begins, so a model "fitted at the test start" used returns from
  inside the test period.
- **Test:** `test_audit.py::test_training_events_with_unresolved_outcomes_are_embargoed`.
- **Fix:** training events whose outcome window had not closed before the
  first test date are dropped (5 sessions for H3, 1 for H6).

### H5. A cancelled session's leftover bar corrupted the reopening-day return

- **Where:** `data/prices.py:51` (`CANCELLED_SESSIONS`).
- **Mechanism:** Borsa Istanbul cancelled all trades of 2023-02-08 and
  reopened on 2023-02-15. The provider still serves a bar for 2023-02-08. The
  reopening return was measured from that bar: 65 of 96 stocks showed a move
  beyond the 10% limit on 2023-02-15. Measured from 2023-02-07, AKBNK's move
  is exactly +10.0%.
- **Found by:** `limits.empirical_check`, which compares the bars with the
  exchange's price-limit rule.
- **Test:** `test_core.py::test_a_cancelled_session_is_removed_so_the_next_return_starts_before_it`.
- **Fix:** annulled sessions are removed from every series, with the source
  recorded beside the date.

### H6. Carried-forward prices were read as zero returns

- **Where:** `data/prices.py:253`.
- **Mechanism:** on a day a stock did not trade the provider repeats the last
  price with zero volume. That is a fabricated return of zero. One ticker had
  441 such bars.
- **Test:** `test_core.py::test_a_zero_volume_bar_is_a_missing_session_not_a_zero_return`.
- **Fix:** a zero-volume stock bar is a missing session; the return after it
  is missing too.

### H7. A sample with no studiable event crashed instead of reporting why

- **Where:** `eventstudy.py` (`event_table`).
- **Mechanism:** when every event fell in the sealed window, no outcome column
  existed and H7 raised `KeyError`. A crash is not a verdict, and the next
  person fixes a crash by loosening something.
- **Test:** `test_hypotheses.py::test_sealed_window_blocks_limit_and_event_outcomes`.
- **Fix:** outcome columns always exist; the hypothesis returns
  `data_insufficient` with counts by status.

### H8. A blank table cell turned a sale into "a purchase and a sale"

- **Where:** `insider.py` (`transaction_rows`).
- **Mechanism:** in the text extracted from a share-transaction form an empty
  cell leaves no trace. A sale of 190,000 shares with the "bought" cell blank
  reads `190.000 190.000 22.966.000 22.776.000` (real form 1205583), which a
  positional parser takes as 190,000 bought and 190,000 sold. With the "sold"
  cell blank instead, a sale would be read as a purchase.
- **Tests:** `test_components.py::test_a_blank_table_cell_cannot_turn_a_sale_into_a_purchase`,
  `::test_a_row_that_does_not_reconcile_with_holdings_is_not_parsed`.
- **Fix:** the side is read from the change in holdings the form reports
  (start of day against end of day), and a row is accepted only if its amounts
  reconcile with that change within 1%. A form with any row that does not
  reconcile is left unparsed.

## Medium

### M7. "Nominal bedelli" was read as a rights issue

- **Where:** `insider.py` (`_EXCLUSION_STEMS`).
- **Mechanism:** forms are excluded when the text shows the shares came from a
  capital increase. The stem `bedelli` (rights issue) also matched "190.000 TL
  nominal bedelli", which means "with a nominal value of", so an ordinary
  on-exchange trade was dropped.
- **Test:** `test_components.py::test_a_blank_table_cell_cannot_turn_a_sale_into_a_purchase`.
- **Fix:** the stems name the capital increase (`bedelli sermaye`,
  `bedelsiz sermaye`, `rüçhan`), not the bare word.

### M1. H3 compared two models on different sentiment numbers

- **Where:** `hypotheses/h3.py` (`sentiment_baseline`, `run`).
- **Mechanism:** the raw model used the admissible event's sentiment; the
  surprise was computed from the mean of *all* mentions that day. A difference
  between the models could come from that mismatch.
- **Test:** `test_audit.py::test_surprise_is_the_events_own_sentiment_against_its_baseline`.
- **Fix:** the surprise is the event's own sentiment against the ticker's
  prior baseline.

### M2. The protocol hash covered constants but not code

- **Where:** `hypotheses/registry.py:48` (`code_fingerprint`), `Spec.parameters`.
- **Mechanism:** thresholds defined inside hypothesis modules (H2's limit-open
  cut-off, H5's negative-news threshold, H6's ridge) and any rule inside a
  function body could change without changing the hash.
- **Tests:** `test_audit.py::test_protocol_hash_covers_source_files_and_hypothesis_parameters`,
  `::test_code_fingerprint_ignores_line_endings_but_not_content`.
- **Fix:** every module constant is in its specification, and the hash
  includes a SHA-256 of each result-bearing source file, line endings
  normalised.

### M3. Strategies paid for one leg and assumed short sales

- **Where:** `costs.py:89`; execution blocks of H1, H2, H5, H8.
- **Mechanism:** index-hedged trades were charged one round trip, and H1 and
  H2 traded in both directions, which needs short sales in small caps.
- **Test:** `test_audit.py::test_hedged_strategies_pay_for_both_legs_and_do_not_assume_short_sales`.
- **Fix:** hedged trades pay two round trips; the executable versions are
  long only and say what they do not assume.

### M4. A session missing from the index series merged silently into the next

- **Where:** `data/prices.py` (`build_panel`, `calendar_gaps`).
- **Mechanism:** the index has no bar for 2023-12-25 although 96 stocks
  traded. The calendar comes from the index, so that day folded into
  2023-12-26, where 15 stocks then showed two-day moves read as one-day moves.
- **Test:** `test_core.py::test_a_session_the_benchmark_lacks_is_merged_and_flagged`.
- **Fix:** the gap is detected and reported; close-to-close returns on the
  merged session are two-day on both the stock and the index side, and
  same-day quantities are withheld there.

### M5. Partial provider bars implied moves the exchange forbids

- **Where:** `data/prices.py:283`.
- **Mechanism:** bars for 2022-12-16 have about a third of normal volume, and
  eight stocks then "moved" 11% to 18% on 2022-12-19.
- **Test:** `test_core.py::test_a_move_beyond_the_price_limit_is_a_defect_not_a_return`.
- **Fix:** under a verified limit rule a one-session move beyond the limit is
  set to missing and recorded. After all three price fixes, 13 of 40,496
  stock-sessions (0.03%) contradict the rule, down from 95.

### M6. Defects in existing code outside this module (recorded, not changed)

- `kap_ingest.py` takes tickers from `relatedStocks`, which is empty for
  ordinary company filings (the codes are in `senderExchCodes`), and stores
  the publication hour only, which cannot separate 18:05 from 18:55 around an
  18:10 close.
- `config.BIST_HOLIDAYS` covers 2025–2026 only, so `trading_calendar` treats
  every other weekday as a session.
- `events/entities.py` lists `ZIRAAT` as an issuer; Ziraat Bankası is not a
  listed company.

These feed the index pipeline, whose derivations are versioned and whose
study is sealed. `kap_ingest.py` is disabled in production. Changing them is a
decision for the index study, so they are left as they are and the stock
module does not use them.

## Low

| | Finding | Status |
|---|---|---|
| L1 | Events sharing a day 0 but published at different kinds of time kept the first one's timing label | **Fixed**: labelled `mixed`, outside clean-timing samples (`test_audit.py::test_events_sharing_a_day0_but_not_a_timing_are_marked_mixed`) |
| L2 | Details the gateway could not serve were not counted | **Fixed**: reported by `describe` |
| L3 | H7's joint test rejected 8 of 100 null simulations | Recorded. Within sampling error of 5% (95% interval 3.5% to 15%), possibly mildly liberal. The Holm factor of nine leaves a wide margin |
| L4 | Stock returns include dividends; the XU100 benchmark is a price index | Recorded in LIMITATIONS. About one basis point a day on average |
| L5 | A disclosure in the last minutes before 18:10 is labelled during-session though it can barely be traded that day | Recorded. Such events are outside H2 and H7's primary samples |
| L6 | The entity linker's accuracy is measured on examples written by the rule author | Recorded. It is a development set, not an accuracy estimate |

## Checked and found sound

Each of these was a place a leak could have been, and has a test:

- The expected-return model cannot see the event: an injected return on days
  −5..+20 leaves alpha and beta unchanged.
- Size terciles use capitalisation from strictly before day 0.
- The sentiment baseline (H3) and the coverage baseline (H4) are strictly
  lagged; a later event changes nothing earlier.
- An issuer's listed line is chosen from turnover before day 0.
- A window with a missing return has no CAR; nothing is filled.
- Holm uses the registered family of nine even when one test runs.
- An unadjusted p below 0.05 is not support; nor is a significant effect below
  the material threshold, or one of the wrong sign.
- No hypothesis returns anything but `data_insufficient` on a thin sample or
  with a required dataset missing.
- The sealed index window blocks stock-level outcomes, limit days included.
- A real-data run refuses without a registered protocol or on a partial frame.

## Tests

Executed on Windows 11, Python 3.10, 2026-10-11:

- `python -m pytest`: **740 passed**, 0 failed, 0 skipped. 640 are the
  repository's existing tests; 100 are new, in `tests/stock_research/`.
- `python -m scripts.freeze_result --verify-only`: frozen index artifact intact.
- Null rejection rates of the full tests over 100 independent synthetic
  contexts each, nominal 5%: H1 3%, H2 4%, H4 2%, H5 6%, H7 8%, H8 4%.

Nothing was blocked by the environment. Two things are not covered by
automated tests because they need the network, and were exercised by hand in
this session: `YahooProvider.fetch` and the KAP client (`data/kap.py::api_get`).

## What the executed tests support, and what they do not

**Supported by executed tests:**

- Each of the nine estimators recovers an effect planted in synthetic data
  and does not find one that was not planted.
- The listed leak guards hold.
- The verdict rule behaves as registered.
- The real-data path runs end to end on synthetic inputs shaped like real ones.

**Not verified by anything in this repository:**

- Any statement about Borsa Istanbul. No real abnormal return has been
  computed. The KAP sample was still being ingested when this was written.
- That the 10% price limit applied throughout the sample. The source is the
  exchange's March 2020 announcement; no later change was found, and the data
  agree with 10% on 99.97% of stock-sessions.
- That the provider's adjusted prices handle rights issues correctly.
- Session hours in 2023 (10:00 to 18:10 is assumed).
- The insider parser beyond the five real forms seen and synthetic variants.
- The entity linker on real headlines.
- The size of survivorship bias. Delisted companies are absent from both the
  roster and the price provider, and how many is not known.
- Anything about H9. There is no English news stream.
