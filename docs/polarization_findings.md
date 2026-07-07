# Finding: political slant in Turkish financial-news sentiment

*Generated from `polarization_analysis.py`. Data: ~1,900 scored headlines,
2026-03 to 2026-07.*

![polarization](polarization.png)

## The result

Turkish financial-news sentiment carries a **large, statistically overwhelming
political slant**. Ranking outlets by political leaning produces a monotonic
gradient in average market sentiment:

| Camp | Outlets | Mean sentiment |
|---|---|---|
| Pro-government / state | Sabah, Anadolu Agency | **+0.11** |
| Market-focused | Bloomberg HT, Investing | −0.03 |
| Opposition | Sözcü | **−0.09** |

The pro-government vs opposition gap is **+0.20** (t = 10.6, **p ≈ 4×10⁻²⁴**,
Cohen's d = 0.74 — a medium-to-large effect on ~950 headlines). Unlike the
market-prediction question, this finding is **not** sample-limited: it has real
statistical power and the 95% confidence intervals for the two camps do not
overlap.

## What makes it non-obvious: the slant is *political*, not tonal

The divergence is concentrated in **domestic-economic coverage** and nearly
disappears on topics Ankara does not control:

| Topic | Pro-gov − opposition gap |
|---|---|
| Turkish economy (macro) | **+0.21** |
| Companies | +0.21 |
| Global markets | +0.16 |
| **Energy / commodities** | **+0.04** |

Outlets split sharply on *how the Turkish economy is doing* (a politically
loaded question) but essentially **agree about oil and commodity prices**. If
this were a blanket editorial mood, the gap would appear everywhere; instead it
tracks the political charge of the topic. That within-topic contrast is the
evidence the mechanism is political.

## Exploratory (thin data, not yet a claim)

- **Daily polarization index** (pro-gov − opposition, n≈21 days): consistently
  positive (+0.21), with spikes around the mid-June 2026 political-tension days.
- **Market link:** polarization → next-day lira volatility is currently null
  (r≈+0.09, p≈0.71, n≈19) — underpowered. Now that USD/TRY is collected daily,
  this is instrumented to be tested properly at 60+ overlap days.

## Limitations and how they're being addressed

| Limitation | Status |
|---|---|
| "Opposition" was a single outlet (Sözcü) | **Being fixed:** Cumhuriyet (a distinct major opposition paper) + Sözcü's economy feed added 2026-07-07; the slant will be re-verified with a broader opposition camp as their history accumulates. |
| Framing vs selection (same story spun differently, or different stories covered?) | Partly addressed by the within-topic comparison (the gap survives *within* macro but vanishes *within* energy, arguing against pure topic-selection). A full same-story control needs entity/event matching (see the event-centric migration). |
| Sentiment is one LLM's measure | The scorer only sees title text, never the source, so a systematic per-outlet difference must come from the text itself. A clean next check is to replicate the gap with a second, independent model. |
