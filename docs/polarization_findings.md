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

## Deeper: who drives the slant (`polarization_dynamics.py`)

*Updated 2026-10-07: 5,014 scored headlines over 136 publication dates
(2026-03-12 .. 2026-10-06). Intervals resample whole publication dates and
resample the baseline with the camps, per
[POLARIZATION_METHODS.md](POLARIZATION_METHODS.md); the July draft used an iid
headline bootstrap with a fixed baseline, which overstated precision.*

Using the **market-focused press as a neutral baseline** (Bloomberg HT, Investing;
mean −0.06), the polarization is **asymmetric**, and not in the obvious direction:

| Camp | n | Mean | Deviation from market baseline (95% CI) |
|---|---|---|---|
| Pro-government | 2,120 | +0.09 | **+0.15** [+0.12, +0.17] |
| Opposition | 2,032 | −0.13 | −0.08 [−0.10, −0.05] |

The pro-government press sits about twice as far from the baseline as the
opposition press (asymmetry +0.068, date-cluster 95% CI [+0.019, +0.118];
excludes 0, and consistent with July's +0.077 on a quarter of the data). The
split is driven **more by pro-government optimism than by opposition
pessimism**. Descriptive and observational: it describes outlet tone, says
nothing about intent, and depends on the market press being a fair midpoint.

**Stress hypothesis: not supported.** Does the gap widen in weeks the lira
weakens? Over 18 weeks with enough coverage in both camps, Pearson r = +0.61
(p = 0.01) looks like a yes, but it rests on two adjacent weeks (24 Aug: smallest
gap, lira firmer; 31 Aug: large gap, largest depreciation). The rank correlation
is ρ = +0.08 (p = 0.76), and leave-one-week-out Pearson ranges from +0.32 to
+0.67. Read as no reliable relationship at this sample size. The public-anxiety
(Google Trends) leg could not be run: `external_series` holds no `gt_dolar` rows.

![dynamics](polarization_dynamics.png)

## Limitations and how they're being addressed

| Limitation | Status |
|---|---|
| "Opposition" was a single outlet (Sözcü) | **Being fixed:** Cumhuriyet (a distinct major opposition paper) + Sözcü's economy feed added 2026-07-07; the slant will be re-verified with a broader opposition camp as their history accumulates. |
| Sentiment is one LLM's measure | **Addressed.** Replicated with an independent model (`replicate_slant.py`): on the same 150+150 headlines, Gemini finds gap **+0.16** vs gpt-5-mini's **+0.20** (both *p* < 1e-7), and the two models agree headline-by-headline (*r* = 0.74). The slant is in the text, not one scorer's artifact. |
| Framing vs selection (same story spun differently, or different stories covered?) | **Examined — and it complicates the story.** A same-story matcher (`same_story_analysis.py`) found 33 cross-camp pairs about the same event; *within* those pairs the gap shrinks to **+0.08 and is not significant** (p=0.12). So a large share of the overall slant appears to be **selection** (which stories each camp covers) rather than **framing** (spinning the same story). Framing is present on genuine matches (e.g. a Bosphorus transit-fee rise: pro-gov +0.30 vs opposition −0.30) but the crude lexical matcher is noisy; cleanly decomposing selection vs framing needs entity/event linking (migration Phase 6). **The robust claim is the *existence and topic-structure* of the slant, not that it is primarily framing bias.** |
