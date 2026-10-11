# Pre-run readiness audit of the stock-level study

Date: 2026-10-11. Branch: `audit/stock-research-readiness`, from
`audit/stock-research-hardening` at `df6be00`. Subject: the registered
stock-level framework (protocol v2,
`09e9382e77ec44bc06aaaf5c4bde0d64e06a6584d2aeb4b65ed35da0b2e7d99a`), and H7
in particular, before its first run on real data.

**Decision: BLOCKED.** See [READINESS_DECISION.md](READINESS_DECISION.md).

**Nothing in this directory is a research result.** No abnormal return of a
sampled event was computed or read. Every probability reported comes from
synthetic returns; every count comes from identifiers, dates and categories.

## In one page

* The registered H7 test is implemented correctly and is miscalibrated on the
  sample it will get: with no effect present it rejects 10.5% of the time at
  5% and 3.1% at the level a single test must meet under Holm (0.56%). A wild
  cluster bootstrap by date holds the level.
* The sample will be about 1,000 usable events on 74 dates. With a valid test
  it has 80% power only for spreads of about 5% (evenly spread) or a little
  over 3% (one large category). A census of the year, about 42 more hours of
  ingestion, brings that to about 1.7%.
* The provider's closing prices are wrong for most stocks on 16 dates, one
  pair of them in the sample year, and four capital increases are adjusted
  with a wrong factor that the cleaning rules let through. Dividends, plain
  bonus issues, rights issues and the benchmark check out against a second
  source.
* The calendar, the timing rule, the market model, the protocol
  registrations, the append-only storage and the ingestion so far (826 of 826
  details, no refusal) held up.
* A new protocol version is needed before any run. It is proposed in the
  decision file and not implemented.

## Files

| File | Phase | Content |
| --- | --- | --- |
| [READINESS_DECISION.md](READINESS_DECISION.md) | 8 | the decision, its justification, all findings, proposed v3, next steps |
| [POWER_ANALYSIS.md](POWER_ANALYSIS.md) | 1 | projected sample, power tables, detectable effects, sample-size and dependence sensitivity |
| [POWER_RESULTS.csv](POWER_RESULTS.csv) | 1 | every simulated cell with its standard error |
| [PRICE_ADJUSTMENT_AUDIT.md](PRICE_ADJUSTMENT_AUDIT.md) | 2 | two-source comparison of prices and corporate actions |
| [MARKET_MODEL_AUDIT.md](MARKET_MODEL_AUDIT.md) | 3 | expected-return model and benchmark |
| [INFERENCE_CALIBRATION.md](INFERENCE_CALIBRATION.md) | 4 | reference implementations, null rejection rates, alternatives |
| [TRADING_CALENDAR_AUDIT.md](TRADING_CALENDAR_AUDIT.md) | 5 | 2023 calendar, closing times, timing boundaries |
| [KAP_COMPLETENESS.md](KAP_COMPLETENESS.md) | 6 | ingestion state, digests, mapping, the 1% tolerance |
| [PROTOCOL_INTEGRITY.md](PROTOCOL_INTEGRITY.md) | 7 | registrations, fingerprints, determinism, bypass review |
| `calibration_*.json`, `calibration_power_comparison*.csv`, `mean_test_calibration.json`, `power_designs.json` | 1, 4 | simulation outputs behind the tables |
| `figures/` | 1, 4 | power curves, nominal against Holm, power by sample size, null rejection rates |

Each finding carries a severity, its evidence, the source file and line, how
to verify it, its impact, a recommended action, and whether it needs a
protocol amendment. Finding ids: W (power), A (prices), M (market model), I
(inference), C (calendar), K (KAP), P (protocol).

## How the audit stayed read-only

* The registered package `stock_research/` was not edited. Its protocol hash
  was the v2 hash before and after
  (`test_the_code_still_is_the_registered_protocol`).
* Audit code is in `readiness_audit/`, outside the fingerprint.
* The research database was opened read-only, twice, to copy it with
  SQLite's backup API. Everything else ran on the copies in
  `outputs/stock_research/readiness/`, which is not committed.
* `python -m stock_research.cli run`, `pipeline.run_real`,
  `registry.run_all` on real data and the real-data hypothesis runners were
  not called. The audit used `h7.prepare` (eligibility, no returns) and, for
  whether a window's returns exist, `notna()` only.
* Real returns were read in three places, none of them an event window of the
  sample: residuals of 2022 for the noise calibration; single-session returns
  of every stock on every session for the two-source comparison; and price
  ratios on corporate-action dates.
* The ingestion process was not stopped, restarted or reconfigured. Two
  groups of the audit's own simulation processes were stopped and restarted.
* Nothing was merged and no protocol was registered.

One correction made during the audit: the first version of the sample
projection left out anchors that yield no usable event and overstated the
projected sample by about a sixth. It was found before any document was
final; every simulation was rerun, and a test now covers it
(`test_anchors_that_yield_nothing_stay_in_the_projection_pool`).

## Reproduce

```
python -m readiness_audit.make_structure           # copy the database, read structure, calibrate on 2022
python -m readiness_audit.kap_checks               # ingestion and mapping counts
python -m readiness_audit.price_checks             # corporate-action consistency
python -m readiness_audit.independent_prices fetch # second source (network)
python -m readiness_audit.independent_prices compare
python -m readiness_audit.market_model             # pre-sample model diagnostics
python -m readiness_audit.run_simulations          # all simulations, about an hour on 12 cores if run in parallel parts
python -m readiness_audit.figures
python -m readiness_audit.tables
pytest tests/stock_research/test_readiness.py
```

`make_structure` takes a new copy, so numbers that depend on ingestion
progress will move as anchors arrive. The documents here use the copy of
11:59 (14 anchors complete) for structure and that of 12:33 (19 anchors) for
ingestion status.

## Tests

`tests/stock_research/test_readiness.py`, 34 tests: the covariance against a
brute-force reference and against statsmodels; the audit's joint test against
`h7._joint`; the simulator's mean and variance; the planted spread; the
over-rejection in a small fixed design; the projection rules; fourteen timing
boundaries, both half days, the wrongly flagged full sessions, time zones;
the protocol hash; the fingerprint's scope; determinism of the synthetic
results.

Run on 2026-10-11: the whole repository suite, 793 passed (759 before this
audit, plus 34); `tests/stock_research`, 153 passed.
