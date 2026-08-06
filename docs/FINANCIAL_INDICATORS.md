# Financial news indicators

Descriptive measures of Turkish financial-news conditions. **None of these is a
trading signal.** They describe what the press published and how unusual it was
against its own history; no result here is a validated predictive relationship.

## Signal families

The detailed `category` field is a measurement input, assigned by a specific
scoring prompt. It is frozen. Families are *derived* from it plus transparent
headline rules and carry their own version (`signal_family_version`), so the
mapping can be revised without redefining the historical record.

| Family | Contents |
|---|---|
| `monetary_policy` | TCMB, policy rate, PPK, bonds, treasury |
| `inflation_macro` | CPI, growth, employment, trade, budget |
| `political_regulatory_risk` | Political events, SPK/market regulation |
| `fx_lira` | Currency, exchange rates, lira |
| `banking_financial_sector` | Sector-level banking: credit growth, deposit rates, BDDK, liquidity, capital, sector profitability |
| `company_kap` | A **named** listed issuer's own event: earnings, dividends, capital increases, acquisitions, KAP disclosures |
| `global_risk` | Fed/ECB, geopolitics, commodities, global markets |
| `market_recap` | Reports of a price move that already happened |
| `media_narrative` | Explicit commentary and press-about-press framing |
| `other` | Relevant but fitting no family |

### The banking boundary

**Entity specificity, not industry.** "Banking sector loan growth slows" is
`banking_financial_sector`; "Garanti BBVA announced results" is `company_kap`,
the same as any other issuer. Because the rule keys on whether a specific listed
entity is named, it applies uniformly across sectors instead of special-casing
banks.

A named bank *without* an issuer-level event is assigned to the sector and
**flagged ambiguous** rather than forced.

### Domestic-only aggregate

Stored under the family key `__domestic__`. It covers `monetary_policy`,
`inflation_macro`, `political_regulatory_risk`, `fx_lira`,
`banking_financial_sector` and `company_kap`.

`global_risk` is excluded because a Fed decision is real information about a
different economy, and folding it into a Turkish domestic tone series conflates
the two. `market_recap` is excluded because it reports moves that already
happened rather than describing conditions.

**The pre-existing overall `daily_signal_variants` table is unchanged.** The
domestic aggregate is an addition, not a replacement.

## Market recap

A recap carries no new information about the future: "BIST closed lower" tells
you what the market did, not what it learned. Mixing recaps into a directional
signal creates a reverse-causality trap — the tone follows the return by
construction, so any apparent predictive relationship is the return predicting
itself.

Recaps are **kept**, not deleted or hidden. They measure attention and are the
right sample for reverse-causality checks. The flag simply makes the distinction
available, and recaps are excluded by default only from directional research
outputs.

Detection requires **both** a market subject (index, venue, instrument, sector
shares) **and** a movement predicate in a reporting frame. An exemption pass then
removes headlines announcing genuinely new information — a new index, a listing,
an appointment, a regulatory decision, a company disclosure — so market
vocabulary alone cannot trigger it.

Fields: `is_market_recap`, `market_recap_version`, `market_recap_rule`,
`market_recap_evidence`, `market_recap_confidence`. Confidence reports which rule
fired (0.9 unambiguous frame, 0.7 subject-plus-movement), not a calibrated
probability.

## Daily family signals — `daily_family_signals`

Keyed by `signal_date + signal_family + experiment_id + family_version`.

Mean, relevance-weighted mean, median, min, max, standard deviation, headline
count, independent source count, positive/neutral/negative shares, average
relevance, market-recap count, unknown-timing count, excluded and unresolved
counts, ambiguous count, and a sample-sufficiency status.

Arithmetic is reused from `aggregation.signals.compute_signal_variants` rather
than reimplemented, so there is one definition of "the signal".

| Sufficiency | Meaning |
|---|---|
| `sufficient` | ≥3 headlines from ≥2 sources |
| `thin_sample` | fewer than 3 headlines |
| `single_source` | enough headlines, one outlet |
| `insufficient` | no usable observations |

**NULL is used wherever a value cannot be defensibly computed.** A single
observation has no dispersion; reporting 0.0 would claim consensus.

## Abnormal tone — `abnormal_tone_daily`

Scopes: `outlet`, `outlet_family`, `family`.

Absolute tone says little — some outlets are structurally more negative than
others, so −0.2 from a habitually gloomy paper is unremarkable while the same
value from an upbeat one is news. Normalizing against each key's own history is
what turns a level into a signal.

**Every value for date *t* uses observations strictly before *t*.** A
full-sample mean would leak the future into every historical value and make any
downstream evaluation meaningless. Below the minimum history the value is NULL;
a zero-variance prior yields NULL rather than an infinite z-score.

Default window 20 sessions, minimum history 5, both configurable and versioned.

This is **time-series** normalization — each key against its own past. It is not
cross-sectional; no key is ranked against other keys on the same date.

## News disagreement — `news_disagreement_daily`

Within-day dispersion, cross-outlet dispersion, max-minus-min outlet tone,
strongly positive/negative shares, entropy (base 3, so 1.0 is an even three-way
split), government/opposition camp gap, official-versus-general-media gap, and
independent source breadth.

**This measures disagreement among observed news sources. It is not market
uncertainty** and is never relabelled as such: outlets can disagree loudly about
a story markets ignore, and agree completely about one that moves prices.

Cross-outlet fields require at least 3 independently represented sources and are
NULL otherwise. The camp mapping is reused from the existing polarization work
so the daily indicator and the inferential analysis describe the same construct.

## Volume and attention — `news_volume_daily`

Per family plus an all-news series keyed `__all__`.

| Field | Counts |
|---|---|
| `headline_count` | rows |
| `observation_count` | distinct events |
| `source_breadth` | outlets that carried it |

Wires syndicate, so ten copies of one agency story is one event covered widely,
not ten independent signals. Only `source_breadth` is named as breadth, so the
numbers cannot be quietly swapped for each other.

Rolling prior mean, standard deviation, z-score, percentile, and changes over 1,
5 and 20 sessions — all from sessions strictly before the date described.

## Regime report

Deterministic JSON and CSV. Separates four things:

- **level** — where tone sits now
- **change** — movement over 5 and 20 sessions
- **abnormal** — position against its own prior history
- **attention** — coverage volume and breadth

Plus disagreement, sample sufficiency, timing quality, and market-recap share,
with the individual headlines carrying the most weight in each number.

**No causal explanation is generated.** Listing a headline as a driver describes
its weight in an average; it does not assert that it caused anything.

## Coverage report

Counts and shares by detailed category and family, the rule responsible for each
assignment, ambiguous cases with their reasons and examples, `other` share,
market-recap share, sources per family, family × timing bucket, family ×
experiment, and representative examples.

This report exists to find weaknesses in the rules. **Rules must not be tuned
against financial outcomes** — that would fit the taxonomy to the answer.

## Limitations

- Families are derived from a keyword taxonomy; ambiguous cases are reported,
  not resolved.
- Rules are Turkish-language string matching, not semantic understanding.
- Disagreement and volume describe the press, not the market.
- Prior windows are short at current corpus size, so many abnormal values are
  NULL for want of history.
- Nothing here has been evaluated out-of-sample, and no result is a validated
  predictive relationship.
