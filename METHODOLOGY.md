# BIST 100 Sentiment Pipeline — Methodology

This document describes every design decision in the pipeline so future
work can replicate, audit, or improve it.

---

## 1. Relevance Filter

Every scraped headline passes a **two-tier filter** before entering the DB.

### Tier 1 — Blocklist

If a blocklist term is present **and** no strong Turkey-market anchor is
present, the headline is dropped immediately.  The blocklist targets
noise that appears in financial feeds but has no BIST relevance:

| Term | Rationale |
|------|-----------|
| `bitcoin`, `ethereum`, `kripto`, `btc`, ` nft` | Crypto-only stories |
| `nasdaq`, `wall street`, `new york borsasi` | US-market stories |
| `nikkei`, `dax `, `ftse`, `hang seng` | Foreign index stories |
| `sterling`, `pound` | Non-TRY FX stories |
| `hindistan `, ` cin`, `japonya` | Country-specific EM news |

**Guard note:** `" cin"` (space prefix) avoids matching `"için"` (Turkish
"for/because"), which after ASCII folding ends in `...icin` and would
otherwise trigger a false positive.

### Strong overrides

If any of the following appear in the headline, the blocklist is bypassed
regardless — these are always BIST-relevant:

`turkiye`, `turk`, `bist`, `borsa istanbul`, `tcmb`, `turk lirasi`,
`tl kuru`, `imamoglu`, `kilicdaroglu`

### Tier 2 — Keyword gate

After passing (or bypassing) the blocklist, the headline must contain at
least one term from `RELEVANCE_KEYWORDS`.  This catches remaining
off-topic headlines that slipped through tier 1.

**Substring guard:** `" fon"` (space prefix) is used instead of bare
`"fon"` to avoid matching `"telefon"` (telephone).

---

## 2. Category Taxonomy

Headlines are classified by `classify_headline()` in `scraper.py` using
first-match priority.  Categories are ordered from most-specific to
least-specific:

| Priority | Category | Key signals |
|----------|----------|-------------|
| 1 | `bist_company` | bist, hisse, halka arz, spk, ozel durum, finansal sonuc |
| 2 | `rates_tcmb` | tcmb, faiz, hazine, tahvil, para politikasi |
| 3 | `political_risk` | Named leaders only (imamoglu, kilicdaroglu) + event words: gozalti, tutuklama, istifa, erken secim, siyasi kriz |
| 4 | `turkey_macro` | enflasyon, buyume, gsyh, ihracat, turizm |
| 5 | `crypto` | bitcoin, ethereum, kripto (before fx_lira to win on "bitcoin+dolar") |
| 6 | `global_risk` | fed, ecb, jeopolitik, kredi notu, moodys, fitch, s&p |
| 7 | `fx_lira` | dolar, euro, doviz, " kur" (space-guarded), lira |
| 8 | `banks` | banka, kredi, mevduat |
| 9 | `energy_commodities` | petrol, dogalgaz, altin, emtia, metals, grains |
| — | `other` | No rule matched |

**Design decision for `political_risk`:** Bare party acronyms (`chp`,
`akp`, `mhp`) are intentionally excluded from this category.  Routine
party news (regional meetings, press conferences) adds noise without
market signal.  Only named leaders and explicit event words are kept.
Party-only headlines that pass the relevance filter land in `other` or
`turkey_macro`.

**`" kur"` guard:** The FX category uses a space-prefixed `" kur"` to
avoid miscategorising headlines about `kurul` (board/council) as FX.

---

## 3. Sentiment Model & Thresholds

**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- XLM-RoBERTa fine-tuned on Twitter data in 8 languages including Turkish
- Outputs probabilities for three classes: negative, neutral, positive
- Continuous score = P(positive) − P(negative) ∈ [−1, +1]

**Threshold tuning** (validated on 198 human-labeled headlines, 2026-06-08):

| Score range | Label |
|-------------|-------|
| score > +0.05 | positive |
| score < −0.05 | negative |
| else | neutral |

- Pre-tuning accuracy (argmax): 69.2%
- Post-tuning accuracy: 76.8% (+7.6 pp)
- The wide neutral band corrects the model's tendency to over-call
  negative on routine financial language ("reserves fell", "forecast
  lowered").

---

## 4. Daily Aggregate: Confidence Weighting

The daily `avg_score` is a **confidence-weighted average**, not a simple
mean:

```
weight_i  = max(|score_i|, 0.10)        # high-conviction signals count more
avg_score = Σ(score_i × weight_i) / Σ(weight_i)
```

Rationale: a headline scored ±0.88 (clear positive/negative framing)
contains more information than a headline scored ±0.06 (near-neutral).
Simple averaging dilutes high-confidence signals with noise.

**Why the 0.10 floor** (added 2026-06-11): without it a perfectly neutral
headline has ~zero weight, so a day with ten neutral headlines and one
+0.6 headline would aggregate to ~+0.6 — the lone strong headline erases
the calm of the day. A neutral headline *is* information ("nothing
happened") and should pull the daily average toward 0. With the floor, a
neutral counts 1/10 of a max-conviction headline.

The same weighting (confidence floor × time-of-day) is used for the
per-category aggregates in `category_daily_sentiment`, so category-level
signals are directly comparable to the overall daily score.

**Validation note:** confidence weighting improves signal smoothness but
models can be confidently wrong.  Track accuracy on labeled holdout data
as the primary validation metric, not smoothness.

Expected side effect: `avg_score` and `bull_bear_ratio` will disagree on
some days (a single high-confidence negative can outweigh several weak
positives).  This is by design.  The evaluate.py L3 audit flags days
where the disagreement is extreme (below 55%), not the normal range.

---

## 5. Daily Aggregate: Time-of-Day Weighting

Publication hour (Istanbul local time, UTC+3) is stored in
`headlines.published_hour`.  The weight multiplier is combined with the
confidence weight:

```
combined_weight_i = |score_i| × time_weight(hour_i)
```

| Istanbul hour | Time window | Weight |
|---------------|-------------|--------|
| < 10:00 | Pre-market | 1.5× |
| 10:00 – 18:00 | Market hours | 1.0× |
| > 18:00 | Post-market | 0.8× |
| NULL (unknown) | Default | 1.0× |

Rationale: pre-market news sets investor mood before the open and has
a larger impact on the first session prices.  Post-market news is
discounted because it cannot affect the same day's close.

**Current status:** `published_hour` is populated for new headlines
scraped after 2026-06-08.  Older rows have NULL and use 1.0× (neutral).
Run `python evaluate.py --layer 1` to see current population rate.

---

## 6. Deduplication Strategy

Headlines are deduplicated by **normalised title prefix** (first 80
characters after ASCII folding), regardless of URL.

```python
dedup_key = _normalise(title)[:80]   # normalise = ASCII-fold + lowercase
```

This prevents the same news story published by multiple RSS sources
from inflating per-day headline counts.  URL-based dedup (via
`INSERT OR IGNORE` on the URL UNIQUE constraint) remains as a second
layer for exact-URL duplicates within the same source.

**False-merge risk:** two distinct headlines whose first 80 normalised
characters are identical.  The evaluate.py L1 title-dedup audit reports
any hash collisions seen in the DB.  Watch for:
- Daily market-open/close summaries (common prefix: "Borsa İstanbul güne
  ... ile başladı")
- Templated headlines from the same source on different days

---

## 7. Known Limitations

1. **Model language mismatch:** The XLM-RoBERTa model was trained on
   Twitter data.  Turkish financial text is more formal and uses
   domain-specific vocabulary.  Accuracy on in-domain text (76.8%) is
   an improvement over the pre-tuned baseline but well below a
   fine-tuned financial model.

2. **Negation handling:** The model does not reliably handle Turkish
   negation (e.g., "düşmedi" — did not fall, "artmadı" — did not rise).
   Such headlines may be scored in the wrong direction.

3. **No intraday signal:** All headlines are aggregated to a single
   daily score.  Intraday timing within the trading session is partially
   captured by time-of-day weights but not fully resolved.

4. **Source selection bias:** The current sources (Bloomberg HT, AA
   Ekonomi, Dünya, etc.) over-represent mainstream economic press.
   Opposition political events may be under-represented; added aa_politika
   and sozcu_gundem in June 2026 to partially address this.

5. **Data volume gate:** L5 signal statistics (Pearson r, hit rate) are
   only shown after 30 overlapping reliable trading days.  Until then,
   all signal claims are anecdotal.

6. **`published_hour` coverage:** Pre-June-2026 headlines have no
   publication hour; time-of-day weighting is fully active only for
   headlines scraped after 2026-06-08.

7. **Confidence weighting unvalidated on holdout:** The confidence and
   time-of-day weights improve theoretical signal quality but have not
   yet been validated against a holdout label set.  Target: 300–500
   labeled headlines split into tune/test sets.


---

## 12. Scorer switch: XLM-RoBERTa -> gpt-5-mini (2026-06-12)

The production sentiment scorer was switched from local XLM-RoBERTa to the
OpenAI API (gpt-5-mini) after a controlled benchmark on the 198 human-labeled
headlines:

| Scorer | Accuracy vs human labels | Methodology |
|---|---|---|
| XLM-R raw argmax | 69.2% | in-sample |
| XLM-R + tuned thresholds (old production) | 76.8% | thresholds tuned ON these labels |
| Gemini 2.5 Flash zero-shot | 77.3% | honest zero-shot |
| gpt-5-mini zero-shot | 82.3% | honest zero-shot |
| **gpt-5-mini + 30 few-shot examples** | **84.5%** | examples from a stratified split, evaluated on the held-out 168 |

Key facts:
- The 30 few-shot examples live in `fewshot_examples.json` and are the exact
  set the benchmark validated (stratified 10/label, random_state=42).
- The LLM returns label + strength; `sentiment_llm._to_tuple` derives a score
  in [-1, +1] that always agrees with the label under the +-0.05 thresholds,
  and pseudo-probabilities with p_pos - p_neg == score, so the existing
  confidence-weighted aggregation and `relabel` machinery work unchanged.
- All 755 historical headlines were re-scored with gpt-5-mini on 2026-06-12 so
  the entire time series uses ONE scorer (pre-switch DB backed up to
  `backups/pre_llm_rescore_2026-06-12.db`). Mixing scorers across history
  would corrupt the signal analysis.
- XLM-RoBERTa remains available as the offline fallback
  (`SENTIMENT_BACKEND = "xlmr"` in config.py).
- Cost: ~half a cent per daily run; the full-history re-score cost ~$0.07.
- Remaining errors are dominated by genuinely ambiguous headlines (label
  noise), not model failures — accuracy near 85% is likely close to the
  inter-annotator ceiling for this task.

---

## 13. Graded relevance instead of deletion (2026-06-12)

The LLM analysis grades every headline with a relevance score in [0, 1]
(stored in `headlines.relevance`) rather than making a binary keep/delete
decision:

| Grade | Meaning |
|---|---|
| 1.0 | Directly about Turkish markets/economy/policy |
| 0.7 | Global financial/geopolitical news with Turkish implications |
| 0.4 | Business news with weak/indirect connection |
| 0.1 | Barely related |
| 0.0 | Unrelated (celebrity, sports, prayer times, crime, lottery) |

Usage in aggregation: `weight = max(|score|, 0.10) x time_weight x relevance`.
Rows with relevance below `RELEVANCE_MIN_FOR_AGGREGATION` (0.25) are excluded
from daily aggregates entirely but NEVER deleted — the judgment stays in the
DB, auditable, and the threshold is tunable without data loss. Ungraded rows
(NULL) count fully.

Rationale: the LLM's sentiment was validated against human labels (84.5%);
its relevance judgment was not. An unvalidated judgment may downweight, but
must not destroy data. (An earlier same-day version deleted 60 "irrelevant"
rows; they were restored from backup and regraded under this scheme.)

The keyword relevance filter in scraper.py still runs at scrape time as a
cheap pre-filter; the LLM grade refines it at scoring time.

---

## 14. Label conventions v2 and prompt recalibration (2026-06-13)

The 300-headline label set collected under the written LABELING.md rubric
revealed a **convention shift**: the original 198 labels were ~26% neutral,
the new set is 59% neutral (the rubric''s "stuck >20s -> neutral" and
"foreign-economy data -> neutral" rules bite hard). The two sets disagree
about where neutral begins and are NOT merged.

**Decision: the 300-row set (labels_validated.csv conventions) is the
canonical ground truth.** The 198-row set is legacy reference only.

Production scorer recalibrated to the new convention ("prompt p3"):
- LABELING.md conventions written into the scoring prompt verbatim
  (neutral-default, Turkey-lens commodities, hike-expectations = negative,
  dollarization = negative, intra-party politics = neutral, etc.)
- 30 few-shot examples re-drawn from the canonical set (10/label, seed 42)
- benchmark_llm.py now imports the production prompt (single source of truth)

Held-out evaluation (270 rows not in the few-shot set):

| Scorer | Accuracy | Direction flips (pos<->neg) |
|---|---|---|
| Majority class ("always neutral") | 61.5% | - |
| Production p2 (pre-recalibration) | 68.5% | ~0 |
| **Production p3 (recalibrated)** | **83.3%** | **0** |

Relevance grading validated on the same set: 90.7% agreement with human y/n
at the 0.25 cutoff; only 1/300 human-relevant headlines excluded.

Full history re-scored under p3 (EXPERIMENT_ID v1-p3; pre-rescore backup
`backups/pre_p3_rescore_2026-06-13.db`). Residual errors are concentrated on
the neutral boundary (e.g. ministerial PR phrased as investment news) and
approach the practical inter-annotator ceiling for single-title judgment.
