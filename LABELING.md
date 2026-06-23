# Labeling Guide

You are the ground truth. The model was benchmarked against *your* judgment
(84.5%), and every future improvement is measured against these labels — so
**consistency matters more than any individual "correct" answer**. Pick a
convention, write it down, stick to it.

## Setup (5 minutes)

```powershell
# Export 300 fresh headlines, skipping the 198 you already labeled:
python main.py export-labels --n 300 --exclude "labels_to_validate - labels_to_validate.csv.csv"
```

Open `labels_to_validate.csv` in Excel / Google Sheets, then:

1. **Hide columns `model_label`, `model_score`, `model_relevance`** before you
   start. Seeing the model's answer first anchors you and silently inflates
   agreement — the single biggest labeling mistake. Unhide them only when done.
2. Freeze the header row, widen the `title` column.
3. You fill exactly two columns:
   - `human_label`: `positive` / `neutral` / `negative`
   - `human_relevant`: `y` / `n`

## The question you are answering

> "A Turkish equity investor skims this headline. Does it nudge their mood
> about the Turkish market up, down, or not at all?"

Judge **market-relevant sentiment, not emotional tone**, and judge **only the
title** — no googling the story, no opening the article. If the title alone
doesn't move you, it's `neutral`.

## Label definitions

| Label | Meaning | Examples from this corpus |
|---|---|---|
| `positive` | Good for Turkish economy/market mood | export records, rate-cut expectations, government support packages, trade deals, rating upgrades, strong earnings, oil/gas prices falling |
| `negative` | Bad for Turkish economy/market mood | inflation above expectations, lira weakness, company losing 93% of market cap, bankruptcies, political arrests, sanctions, energy prices rising |
| `neutral` | No directional read | data announced without surprise ("TCMB rezerv verileri açıklandı"), schedules, live price tickers ("Altın fiyatları canlı"), balanced/mixed reports |

## Conventions for the recurring hard cases

These came from analyzing where you and the model disagreed. Decide once,
apply always:

1. **Commodities through Turkey's lens.** Turkey imports nearly all its
   energy: oil/gas **down = positive**, up = negative. But **US-specific
   inventory/production stats = neutral** (e.g. "ABD'de petrol stoklarında
   sert düşüş" — a US statistic, not a price move). Gold price moves =
   neutral unless explicitly tied to TL or crisis flight.
2. **Foreign economies** (Eurozone confidence, German industry, UK PMI) =
   `neutral`, unless it's a clear global risk event (Fed surprise, crash,
   war escalation) that would hit all emerging markets including Turkey.
3. **Company-level news counts.** A bankruptcy, fine, or disclosure of
   problems = `negative` even for a small company; record results or major
   contracts = `positive`. (Index impact is handled by weighting later —
   your job is just direction.)
4. **Politics:** arrests/resignations/crises that markets would notice =
   `negative` (+ relevant). Routine party events, speeches, condolence
   messages = `neutral`, and usually `human_relevant = n`.
5. **Stuck for more than ~20 seconds?** It's `neutral`. Genuine ambiguity IS
   the neutral class. Optionally add a `notes` column and mark these — they
   make great few-shot examples later.

## Settled judgment calls (decided 2026-06-13, follow these)

Documented while labeling the 300-row canonical set — these are now binding:

| Case | Convention |
|---|---|
| Rising FX deposits / dollarization | `negative` (bad for TL sentiment) |
| Surprise ECB/Fed hike **announcement** | `negative` (global tightening event) |
| Hike **preview** ("bekleniyor") or currency-**reaction** story | `neutral` |
| TCMB/Fed hike *expectations* | `negative` (easing deferred — symmetric with "cut expectations = positive"). Note: a hike call can read credibility-positive in the Turkish context; we deliberately chose the symmetry reading. |
| Foreign bank/investor entering Turkey (e.g. Sber) | `positive` (investment interest beats sanction-exposure angle) |
| Intra-party turmoil (kurultay calls, internal resignations) | `neutral` / `n` — below the "markets would notice" bar |
| Arrests/probes of mayors, opposition leaders | `negative` / `y` |
| Hostile-tone but ambiguous market read | 20-second rule → `neutral` |

## `human_relevant` — y/n

> "Does this headline belong in a Turkish market-sentiment index at all?"

- `y`: economy, markets, companies, commodities, currencies, monetary policy,
  market-moving politics, global financial events — **when in doubt, `y`**
- `n`: sports, celebrities, lifestyle, ordinary crime, prayer times, lottery,
  holiday greetings, tourism listicles — even when they mention lira amounts
  ("3,5 milyon liralık altın bulunan çantayı otogarda unuttu" → `n`)

Note: `n` rows still get a sentiment label (just label the literal tone —
it won't be used for the market index, but it validates the model's grading).

## Process tips

- **Batches of ~50, then a break.** Label quality degrades sharply after
  30–40 minutes. 300 labels ≈ 2–3 hours total; spread over a week is fine.
- **Label in order** — no cherry-picking easy ones; the hard ones are the
  most valuable data.
- Use Excel autocomplete/dropdowns to avoid typos (`positive`, not `pos` or
  `Positive` — lowercase exactly).
- **Save as CSV (UTF-8)**, keep all column names unchanged.
- If you realize mid-way that you changed a convention, go back and fix the
  earlier rows — drift between row 1 and row 300 is worse than any single
  wrong label.

## Label-quality tools (methodological audit)

Beyond "does the model agree with me," three tools check whether the ground
truth *itself* is trustworthy — run them when you want to improve the labeling
systematically rather than just collect more of it.

**1. Adjudicate disagreements** — recover the true error rate. Some "model
errors" are the model being right and you having drifted; tagging them tells
you real accuracy and surfaces rubric gaps.
```powershell
python label_audit.py disagreements labels_validated_p3.csv   # -> disagreements_to_review.csv
# fill the 'verdict' column: model | human | ambiguous, then:
python label_audit.py disagreements disagreements_to_review.csv --summary
```

**2. Measure your own consistency** (the ceiling the scorer is graded against).
Re-label a blind subset after a ~2-week gap; if you only agree with your past
self 85% of the time, an 83% scorer is essentially maxed out.
```powershell
python label_audit.py consistency-export labels_validated.csv --n 50  # -> consistency_relabel.csv
# re-label it blind (don't peek at the original), then:
python label_audit.py consistency-check labels_validated.csv consistency_relabel.csv
```

**3. Label what's informative, not what's random** — active learning. Pulls the
headlines nearest a decision boundary instead of a stratified random sample.
```powershell
python main.py export-labels --n 100 --uncertain --exclude labels_validated.csv
```

## When you're done

```powershell
# Accuracy, confusion matrix, threshold sweep, holdout split:
python main.py validate-labels labels_to_validate.csv --save-report
```

Then tell Claude — next steps from there: merge with the original 198 (~500
total), measure relevance agreement against the LLM's 0.25 cutoff, refresh
the few-shot examples from the richer pool, and re-benchmark the scorer.
Later (migration Phase 4) a ~200-row subset gets a second pass with a
continuous direction (−1..+1) annotation — don't worry about that now.
