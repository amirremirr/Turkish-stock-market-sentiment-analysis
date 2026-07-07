# ROADMAP

Ordered by impact. Each item has a clear done-when so you know when to move on.

---

## Done (2026-06-12) — sentiment scorer switched to gpt-5-mini

| # | What | Why |
|---|------|-----|
| ✓ | **Production scorer is now gpt-5-mini via OpenAI API** (`sentiment_llm.py`, `SENTIMENT_BACKEND="llm"`) | Benchmarked on the 198 human labels: 84.5% held-out accuracy with 30 few-shot examples vs 76.8% for XLM-R tuned in-sample. Gemini free tier tested too (77.3%) but capped at 20 req/day. See METHODOLOGY §12. |
| ✓ | All 755 historical headlines re-scored with gpt-5-mini | One consistent scorer across history; pre-switch DB at `backups/pre_llm_rescore_2026-06-12.db` |
| ✓ | XLM-R kept as offline fallback backend | `SENTIMENT_BACKEND="xlmr"` — no API dependency |
| ✓ | `benchmark_llm.py` decision tool (OpenAI + Gemini providers) | Re-run any time labels grow |

**Also done same day:** LLM **category classification + graded relevance** (METHODOLOGY §13).
The daily scoring call now returns sentiment + category + relevance (0-1) together. Relevance
multiplies the aggregation weight; rows under 0.25 are excluded from aggregates but never
deleted (user decision: unvalidated judgments may downweight, not destroy). `other` category:
18% → ~11%. Still possible later: LLM-extracted ticker mentions (KAP/company-level analysis),
event detection, GPT-written daily mood summary on the dashboard.

## Done (2026-06-11) — critical-review fixes

| # | What | Why |
|---|------|-----|
| ✓ | **`next_return` misalignment fixed** | `shift(-1)` ran AFTER merge/filter in evaluate.py L5 and visualize.py — ~40% of "next-day" pairs actually spanned 2–15 days. Now computed on the consecutive price series before merging. All L5 stats before this fix were partly invalid. |
| ✓ | **Cross-run duplicate prevention + cleanup** | NULL-url headlines (ntv_ekonomi) re-inserted every run (`NULL != NULL` in SQLite). 54 redundant rows deleted; `insert_headlines` now dedups by (normalized title[:80], published date) against the DB. Note: every ntv story had been triplicated. |
| ✓ | **Reliable-day count corrected: 26 → 13** | Half the "reliable" (≥3 headline) days were only above the gate because of duplicate rows. Honest gate ETA: ~17 more trading days. |
| ✓ | **kap_bildirimler removed** | URL 404s — KAP's redesigned site serves no RSS. Feed contributed 0 headlines ever. KAP-style keywords stay in the relevance filter. |
| ✓ | **Confidence-weight floor (0.10)** | Score-0 headlines had zero weight, so one strong headline could define a 10-neutral day. `SENTIMENT_CONFIDENCE_FLOOR` in config.py; METHODOLOGY §4 updated. |
| ✓ | **Per-category weighting unified** (was M2) | `category_daily_sentiment` now uses the same confidence + time weighting as the daily score. |
| ✓ | **`relabel` command** (enables M3) | Recomputes labels from stored probabilities with current thresholds — no re-inference. Threshold changes no longer leave mixed-regime labels. |
| ✓ | **Recovery runs recorded** | Successful step-by-step recovery now writes a `status='recovered'` row to pipeline_runs instead of leaving the day marked crashed. |
| ✓ | **Granger double-shift fixed** | Test ran on `next_return` while statsmodels applies its own lags (testing a 2-day lead by accident). Now uses same-day return. |
| ✓ | **Loud failure on unknown model labels** | `_extract_score` raised-on-empty instead of silently scoring 0.0/neutral after a model swap. |
| ✓ | `ECONOMIC_CALENDAR` comment honesty | Config claimed chart overlay/event tables; nothing reads it yet. Comment now says so. |

## Done (2026-06-10)

| # | What | Why |
|---|------|-----|
| ✓ | `other` category 24% → 18.6% | Added keywords (THY, holding, iflas, küresel piyasalar, LNG, fındık…); added `RELEVANCE_HARD_BLOCKLIST` (piyango, baraj doluluk, dunya kupas, futbol transfer, konser) |
| ✓ | Crash recovery now includes scrape step | Previously recovery retried score/aggregate/prices on stale headlines if scrape was what crashed |
| ✓ | HF offline guard | `HF_HUB_OFFLINE=1` now only set if model cache exists; fresh install no longer fails silently |
| ✓ | BIST holiday skip | `run_scheduled.py` exits early on dates in `BIST_HOLIDAYS` (was running on closed-market days) |
| ✓ | `executemany` rowcount fix | SQLite returns `-1` for `cur.rowcount` after `executemany`; replaced with `total_changes()` delta — inserted count in logs is now accurate |
| ✓ | `statsmodels` in requirements.txt | Granger causality test in evaluate.py L5 was silently unavailable on fresh installs |

---

## Signal exploration findings (2026-07-07, `explore_signal.py`)

At ~30 overlap days (underpowered; min detectable |r| ≈ 0.5), a disciplined,
FDR-corrected sweep of targets and aggregations found:
- **No target shows a signal**; direction, volatility, and FX are all ~zero.
- **The FX "p=0.05" was a false positive** — killed by FDR correction (q=0.43).
- **Aggregation choice does not matter** — mean, confidence-weighted, intensity,
  shock-count, net-direction all ~zero vs next return. Rules out "bad averaging"
  as the reason for the null.
- **One lead to watch as data grows:** abnormal return vs EM (r≈+0.26, sensible
  sign — good news → Turkey outperforms EM). NOT significant (p=0.20), does not
  survive correction. Re-check at 60+ days. Do not report as a finding.

---

## Standing rule — aggregation weights are FROZEN (2026-06-12)

The confidence floor (0.10), time-of-day multipliers (1.5×/1.0×/0.8×), and
relevance cutoff (0.25) are unvalidated hyperparameters justified by narrative.
**Do not adjust them** until there are ≥45 reliable overlap days, then run a
proper ablation: chronological train/test split, each weight toggled on/off,
judged by held-out r and hit rate — not by whether the chart looks smoother.
Tuning them earlier is overfitting with extra steps.

---

## Short-term (next 2 weeks)

### S1 — Labels sprint *(manual, highest-impact)*

The 76.8% accuracy figure is in-sample on the same 198 labels used to tune the ±0.05 thresholds. Until a holdout run passes, this number is optimistic.

```bash
python main.py export-labels --n 300        # exports labels_to_validate.csv
# Open in Excel / Google Sheets
# Fill human_label column: positive / neutral / negative
python main.py validate-labels labels_to_validate.csv --save
```

**Done when:** ≥300 unique labels; `validate_labels.py` holdout report runs and shows:
- holdout test accuracy ≥ 70%
- overfitting gap (tune − test) ≤ 10%

Do not re-tune `SENTIMENT_POSITIVE_THRESHOLD` / `SENTIMENT_NEGATIVE_THRESHOLD` using these labels and then report the accuracy on the same set.

---

### S2 — Temporal alignment (`signal_date`)

Current issue: a headline published at 17:00 Istanbul is grouped into calendar day *t*, but most of day *t*'s return has already been realized. L5 "same-day" correlation may be sentiment reacting to price, not leading it.

**What to do:**

1. Add a `signal_date` field to the aggregation — map `(published_at, published_hour)` to the trading session the market hasn't priced yet:
   - `published_hour < 10` (Istanbul) → `signal_date = published_at` (pre-market, before open)
   - `10 ≤ published_hour ≤ 18` → `signal_date = published_at` (during session — arguable, flag separately)
   - `published_hour > 18` or `NULL` → `signal_date = next_trading_day(published_at)`
2. Store `signal_date` in `daily_sentiment` or compute it in L5.
3. Re-run `aggregate` and regenerate the chart.
4. In evaluate.py L5: report Pearson *r* for both calendar-date and signal-date alignment side by side so you can see whether the "lead" was an artifact.

**Done when:** L5 primary stat uses `signal_date → return(t+1)`, not raw `published_at`.

---

## Medium-term (next month)

### M1 — DB backup

Single SQLite file with no backup. One corrupted write loses everything.

Add to `run_scheduled.py` after the pipeline completes:

```python
import shutil
backup_dir = HERE / "backups"
backup_dir.mkdir(exist_ok=True)
# Keep last 7 daily backups
dst = backup_dir / f"finance_sentiment_{TODAY}.db"
shutil.copy2(HERE / "finance_sentiment.db", dst)
# Prune older than 7 days
for old in sorted(backup_dir.glob("finance_sentiment_*.db"))[:-7]:
    old.unlink()
```

**Done when:** `backups/` directory has rolling 7-day copies; one-command restore possible.

---

### ~~M2 — Per-category aggregation consistency~~ ✓ Done 2026-06-11

Per-category aggregates now use the same confidence + time weighting as the daily score.

---

### M3 — Threshold from holdout only

Currently `SENTIMENT_POSITIVE_THRESHOLD = 0.05` was tuned on the full 198-label set. After S1 is done:

1. Run `validate_labels.py` holdout (60/40 split on ≥300 labels).
2. The optimal threshold from the **tune set** (60%) is the only number you're allowed to set in `config.py`.
3. Report the **test set** (40%) accuracy as the honest number everywhere.
4. After changing thresholds, run `run.bat relabel` — it recomputes all stored labels from the saved probabilities (no re-inference needed).

**Done when:** Config comment says "tuned on tune set only; test-set accuracy = X%".

---

## Long-term (2–3 months, needs data)

### L1 — Walk-forward signal validation

Requires ~6 months of daily data (≥120 reliable overlap days). One-window Pearson *r* on 26–30 points is not statistically meaningful.

**What to build:** Rolling 20-day train / 5-day test window over the full history. Report mean *r* and hit rate across all windows.

**Done when:** Walk-forward mean hit rate > 52% with p < 0.05 over 6+ months, or the opposite (confirmed non-signal).

---

### L2 — Naive strategy with transaction costs

Before the current L5 naive strategy means anything, add a spread/slippage assumption (e.g. 0.1% per trade round-trip). Most apparent edge in daily-signal strategies disappears once you add realistic costs.

**Done when:** L5 shows strategy return net of 0.1% per trade.

---

### L3 — Category-level signal

Once data is sufficient, break L5 down by `category_daily_sentiment`. The hypothesis is that `rates_tcmb` and `fx_lira` sentiment leads BIST returns more strongly than `energy_commodities` or `turkey_macro`.

---

### L4 — Model upgrade path (needs ≥500 labels)

If holdout accuracy stays below 77% after threshold tuning, consider:
1. Benchmark 1–2 finance-specific or Turkish multilingual models from HuggingFace on the same label set.
2. Fine-tune the XLM-RoBERTa head on Turkish financial headlines (needs 500+ labels; even 300 may be enough for a calibration layer on top of `model_score`).

**Do not start this until S1 is done.**

---

## Maintenance (yearly)

- **BIST_HOLIDAYS** in `config.py`: update each January from [KAP market calendar](https://www.kap.org.tr).
- **ECONOMIC_CALENDAR** in `config.py`: update TCMB PPK and TÜİK TÜFE dates each January.
- **RSS feed health**: run `python discover_sources.py --url <feed>` on all feeds if headlines drop unexpectedly.
- **`RELEVANCE_HARD_BLOCKLIST`**: add terms when new social-content noise patterns appear from `sozcu_gundem` or `aa_politika`.

---

## Won't do (and why)

| Item | Reason |
|------|--------|
| Multiple-testing corrections (Bonferroni etc.) | Overkill at current sample size; address if/when publishing |
| SSL certificate allowlist hardening | Personal scraper on trusted home network; acceptable risk |
| Fine-tuning before 500+ labels | Not enough data to outperform zero-shot baseline reliably |
| Migrate off SQLite | No concurrent writers; SQLite is appropriate for this scale |
| Windows portability (ctypes sleep prevention) | Project is intentionally Windows-only (Task Scheduler) |
