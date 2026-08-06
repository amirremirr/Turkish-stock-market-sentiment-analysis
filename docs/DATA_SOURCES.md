# Data sources

What the project collects, from where, under what terms, and what is
deliberately absent.

## Turkish financial news (RSS)

Twelve public RSS feeds. Only headline text, URL, source and publication time are
stored; article bodies are not retrieved or redistributed.

| Source | Tier | Notes |
|---|---|---|
| `aa_ekonomi`, `aa_politika` | B | State news agency; also used as the "official" side of the official-vs-general-media gap |
| `bloomberght` | B | Financial wire |
| `dunya`, `haberturk_ekonomi`, `hurriyet_ekonomi`, `investing_tr_economy`, `ntv_ekonomi`, `sabah_ekonomi`, `sozcu_gundem`, `sozcu_ekonomi`, `cumhuriyet_ekonomi` | C | General and financial press |

Tiers reflect source type, not quality: A = structured primary sources,
B = wires and official statements, C = general press.

Every fetched item is preserved in `raw_headline_observations` before
canonicalization, so a headline carried by several feeds retains all of its
source-distinct observations. That is what makes coverage-breadth counting
honest.

## Market data

| Series | Provider | Coverage | Fields |
|---|---|---|---|
| BIST 100 (`XU100.IS`) | yfinance | daily | OHLCV + derived return + completeness status |
| USD/TRY | Alpha Vantage | daily | OHLC |
| EEM, `BZ=F`, `USDTRY=X` | yfinance | daily | close + return |

yfinance's `XU100.IS` daily series lags roughly a day, so the most recent session
is often unavailable at fetch time. This is a provider characteristic, not a
pipeline fault.

Every daily bar carries a completeness status — see
[OPERATIONS.md](OPERATIONS.md).

## Scoring

OpenAI `gpt-5-mini`, prompt version `p3`, stored as
`gpt-5-mini-2025-08-07/p3`. Requires `OPENAI_API_KEY`. Roughly half a cent per
daily run.

The `p_positive`/`p_neutral`/`p_negative` fields are **synthetic compatibility
values**, not calibrated probabilities, and are labelled
`synthetic_compatibility` in `score_components_kind`.

A local XLM-RoBERTa backend remains available as a free offline fallback.

## Exploratory external series

GDELT global media tone and Google Trends search interest, in
`external_series`. **Present only in a local research database — production has
never collected them**, because the fetchers are not part of the cloud run.

## Not available

| Item | Why it matters | Status |
|---|---|---|
| Intraday BIST prices | 62% of headlines publish during the session; without intraday data their execution timing cannot be tested | **Blocked.** Descriptive use only. |
| Consensus/expectations data | A macro release is only a surprise relative to an expectation | **Blocked.** Licensed. Surprise fields stay unavailable; expectation is never inferred from headline tone. |
| KAP disclosures (production) | Structured issuer events would populate `company_kap` properly | **Blocked.** The development gateway serves a 2023 sample dataset; production access is pending. `KAP_ENABLED=False`. |
| Bond / rate proxy, volatility index | Would widen the control menu | Not collected. |

Nothing above is approximated or fabricated to fill a gap.

## Licensing and redistribution

- RSS headlines: headline metadata only, no article bodies, no redistribution of
  full text.
- yfinance / Alpha Vantage: personal research use under provider terms. Derived
  daily values are stored; raw vendor feeds are not redistributed.
- No paid or licensed dataset is required to run the pipeline, and none is
  bundled.
- The offline demo uses only committed synthetic fixtures and needs no
  credential, network access or model download.

## Credentials

`OPENAI_API_KEY` (scoring), `ALPHA_VANTAGE_KEY` (FX, optional),
`MKK_API_KEY`/`MKK_API_SECRET` (KAP, disabled). Held in `.env` locally and as
GitHub repository secrets in CI. `.env` is gitignored; no credential is written
to the database, the dashboard, or any published artifact.

The after-close price workflow receives **no** credential, so it cannot invoke
the scorer even if a future edit tried to.
