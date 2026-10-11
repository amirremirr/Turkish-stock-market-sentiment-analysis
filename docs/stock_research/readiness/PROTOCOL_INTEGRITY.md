# Reproducibility and protocol integrity

Audit output of 2026-10-11. Nothing here ran a hypothesis on real data. The
registered code under `stock_research/` was not edited: the protocol hash of
the working tree is the registered v2 hash before and after this audit.

## Registrations

| Protocol | SHA-256 | Registered (UTC) | Code | Reproduced by this audit |
| --- | --- | --- | --- | --- |
| v1, interim | `b8a5c8fba4881ada78574a6346be1ab0bb0c14aae064f95ecddbcdd5e2187735` | 2026-10-11 00:09:09 | not a commit | stored document hashes to its key |
| v1 | `92366aa0a6342e9c78c02700a7253b45e5dc4ce0cac6476467057164cb66dfb8` | 2026-10-11 00:15:04 | `ecdc532` (`main`) | yes: `git archive ecdc532`, then `registry.protocol_hash()` |
| v2 | `09e9382e77ec44bc06aaaf5c4bde0d64e06a6584d2aeb4b65ed35da0b2e7d99a` | 2026-10-11 08:39:14 | `df6be00` | yes: same method, and from the working tree |

* Each stored protocol document hashes to the key it is stored under.
* v1's fingerprint covers 26 files and omits `data/kap.py`; v2's covers 29 and
  includes it. v2 names v1 as superseded, with its hash, and v1's row is
  untouched.
* `sr_manifests` and `sr_results` hold **zero rows** in both copies of the
  database taken today. No real-data result has been stored under any
  protocol.
* The frozen index artifact verifies:
  `python -m scripts.freeze_result --verify-only` reports
  `walk-forward-protocol-v1 intact=True verbatim=True file_matches=True`.
* The v2 hash is written in documents on a pushed branch
  (`docs/stock_research/STATISTICAL_PROTOCOL.md`, `RESULTS.md`,
  `AUDIT_REPORT.md` at `origin/audit/stock-research-hardening`), so the
  registration has a timestamp outside this machine.

## Append-only storage

Update and delete triggers exist on `sr_protocols`, `sr_manifests`,
`sr_results`, `sr_raw_kap_listing`, `sr_raw_kap_detail`, `sr_raw_price_bars`,
`sr_raw_members`, `sr_kap_detail_failures` and `sr_ingest_log` (18 triggers).
`sr_kap_frame`, `sr_price_availability` and `sr_meta` are working state and
are mutable by design.

## Determinism

`pipeline.synthetic_demo()` run twice in one process gives byte-identical
canonical JSON (test), and run in two processes with different
`PYTHONHASHSEED` gives the same SHA-256
(`6e0de1e0bf0b345665e3438fdb04a8a658b29606a8432523a5ce545297d72b2b`). The only
random element in the registered code is the cluster bootstrap, which takes a
fixed seed.

Environment of this audit: Python 3.10.11, numpy 1.26.4, pandas 2.3.3, scipy
1.15.3, statsmodels 0.14.6, beautifulsoup4 4.14.3, lxml 6.0.2, yfinance 1.4.0.

## Findings

### P-1. The run gate sits in one function; the hypothesis runners are open

* **Severity:** medium.
* **Evidence.** Registration, ingestion completeness and the missing-detail
  tolerance are checked in `pipeline.run_real`
  (`stock_research/pipeline.py:313-335`) and nowhere else.
  `pipeline.real_context` (`:164-186`) builds the real context for anyone;
  `registry.run_all` (`stock_research/hypotheses/registry.py:175-194`) and
  each `hN.run` take whatever context they are given. A call such as
  `h7.run(real_context()[0])` would compute real outcomes on a partial sample,
  with no registration check, and store nothing.
* **Tests.** The existing tests show the gate refusing an unregistered
  protocol, a partial frame and an over-tolerance sample
  (`tests/stock_research/test_pipeline.py:96-102`,
  `test_hardening.py:189-201`). None shows that the gate cannot be walked
  around, because it can.
* **Impact.** The "no interim look" property is a discipline, not a
  mechanism. This audit did not use that path, and the empty result tables are
  consistent with nobody having used it, but an empty table cannot prove it.
* **Recommended action.** In a future protocol, move the three checks into
  the construction of a real-origin context, so a context holding real
  outcomes cannot exist before they pass; keep the diagnostics that read no
  return on a separate, outcome-free object.
* **Protocol amendment:** yes (it changes fingerprinted code).

### P-2. Prices and disclosures were in the database before the protocol was registered

* **Severity:** low, disclosed for completeness.
* **Evidence.** First KAP detail retrieved 2026-10-10 23:01:26 UTC; first
  price bar 23:31:38; v1 registered 2026-10-11 00:09 and 00:15; v2 at 08:39.
  The price-cleaning rules (cancelled session, zero-volume bars, the calendar
  gap) were written after looking at those prices for defects.
* **Impact.** Cleaning rules were chosen with sight of the price data, though
  not of any event outcome: no event table with returns existed before
  registration, and none has been stored since. That statement rests on the
  development record, not on a cryptographic guarantee.
* **Recommended action:** state it in LIMITATIONS; a later protocol on a new
  sample year would not have the problem.
* **Protocol amendment:** no.

### P-3. Three fingerprinted modules read constants from a file the fingerprint does not cover

* **Severity:** low.
* **Evidence.** The fingerprint covers `stock_research/` only
  (`registry.py:81-89`). The repository-root `config.py` is imported at
  `stock_research/calendar.py:125` (`BIST_HOLIDAYS`, `BIST_HALF_DAYS`),
  `stock_research/hypotheses/h4.py:136` (`ECONOMIC_CALENDAR`) and
  `stock_research/data/kap.py:88` (`KAP_BASE_URL`).
* **Impact.** Adding a 2023 entry to the holiday list would change how every
  2023 filing is timed without changing the protocol hash; a different base
  URL is a different data source; H4's calendar of announcements is a
  result-bearing input. Today none of the three affects H7: the holiday lists
  cover 2025 and 2026 only.
* **Recommended action.** In a future protocol, hash those constants into the
  protocol document, as is already done for the frame constants.
* **Protocol amendment:** yes.

### P-4. The manifest does not record library versions

* **Severity:** low.
* **Evidence.** `registry.manifest` (`registry.py:197-220`) records the
  protocol hash, commit, dates, seeds and data digests, but not the versions
  of numpy, pandas, scipy, or of the HTML parser that turns a disclosure body
  into the text the taxonomy reads (`data/kap.py:142`).
* **Impact.** A later rerun on other versions could differ in the last
  digits, or in parsed text, with nothing to explain it.
* **Recommended action.** Record them (the list above) with the run; add them
  to the manifest in a future protocol.
* **Protocol amendment:** only for the manifest change.

### P-5. A price snapshot is a name, and its contents grow

* **Severity:** low; mitigated by the digest.
* **Evidence.** The snapshot id is the date range. Bars for new issuers are
  appended under the same id whenever `ingest-prices` runs
  (`stock_research/data/prices.py:145-181`). Two runs with the same snapshot
  id can therefore have read different data.
* **Impact.** Identity is carried by `price_snapshot_digest`
  (`pipeline.py:281-299`), which the manifest stores. That is adequate to
  detect a difference, not to restore the data: the database is one local
  file with licensed content and is not committed.
* **Recommended action.** Before the run: finish `ingest-prices`, copy the
  database, and record the copy's SHA-256 next to the result.
* **Protocol amendment:** no.

### P-6. Append-only triggers protect against accidents, not against the owner

* **Severity:** informational.
* **Evidence.** A trigger can be dropped by anyone who can write the file.
* **Impact.** The protection that binds is outside the database: the hash in
  a pushed commit, and results committed with their manifest.
* **Recommended action.** Push the commit that records a registration before
  the run, as was done for v2.

### P-7. A stale sentence in the inference module

* **Severity:** informational.
* **Evidence.** `stock_research/stats.py:9` says the primary test clusters by
  event date. The registered H7 test clusters two ways, by day 0 and by
  issuer (`stock_research/hypotheses/h7.py`, `TWO_WAY`). The code is right;
  the sentence predates the change.
* **Recommended action.** Correct with the next protocol version; editing it
  now would change the fingerprint.

## Is the provenance enough to reproduce a future analysis?

Enough to know whether two runs saw the same data and code: protocol hash,
commit, ingester hash per ingestion run, digests of the KAP sample, the price
bars and the roster, the list of unavailable disclosure IDs. Not enough for a
third party to rebuild the data: KAP details come from a rate-limited
development gateway and prices from a provider that revises history (see
PRICE_ADJUSTMENT_AUDIT.md). Reproduction depends on keeping the database
file.

## What this audit left untouched

No file under `stock_research/` was modified, no protocol was registered, the
production database was opened read-only for two backups and never written,
the ingestion process was not interrupted, and no branch was merged. The
audit's code is in `readiness_audit/`, outside the fingerprint (test:
`test_the_audit_code_is_outside_the_fingerprint`).
