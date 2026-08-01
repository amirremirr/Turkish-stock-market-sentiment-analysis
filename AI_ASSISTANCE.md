# AI Assistance Disclosure

**Published:** 2026-08-01

This project was developed with AI assistance. The disclosure separates generated implementation help from researcher-owned decisions, documents concrete rejected/corrected output, and states what remains insufficiently verified.

## 1. What AI tools assisted with

Interactive coding assistants helped:

- draft and revise Python, SQL, tests, command-line interfaces, and documentation;
- inspect code paths, propose failure cases, and review methodological wording;
- scaffold data-quality audits and exploratory analysis scripts;
- suggest refactors, migration steps, and regression cases;
- summarize intermediate findings and prepare charts or written explanations.

AI models also appear inside the research system, which is a separate role:

- OpenAI `gpt-5-mini`, prompt `p3`, is the active headline scorer;
- Gemini was used as a second-scorer robustness check for part of the media-tone analysis;
- XLM-RoBERTa is retained as an offline fallback and historical baseline.

Using an AI model as the measured scorer is not the same as using an AI coding assistant to build the repository. Both require independent checks.

## 2. Decisions owned by the researcher

The researcher retained responsibility for consequential choices, including:

- defining the Turkish-equity sentiment and relevance rubric;
- deciding that neutral is the label for routine or genuinely ambiguous headlines, but never a substitute for missing processing output;
- resolving conventions for commodities, Turkey's import exposure, rate expectations, dollarization, and political events;
- selecting the canonical 300-label convention and keeping the earlier 198-label set separate;
- requiring model provenance and a held-out comparison before changing the production scorer;
- rejecting destructive relevance handling and requiring raw observation audit, reversible exclusions, restoration history, and explicit purge confirmation;
- requiring market-session alignment, an unweighted primary baseline, frozen sensitivity specifications, and complete-price-series return construction;
- interpreting results as descriptive or exploratory and accepting a null predictive result as valid.

AI suggestions did not authorize data deletion, scorer changes, research claims, or deployment by themselves.

## 3. What was manually validated

Manual work documented in the repository includes:

- 300 canonical direction labels plus human relevance calls under the written rubric;
- a held-out prompt-`p3` comparison on the 270 labels not used as few-shot examples;
- relevance agreement at the 0.25 aggregation cutoff;
- review of model-versus-human disagreements;
- blind re-labeling support for intra-annotator consistency;
- spot checks of parsing, Turkish normalization, topic assignment, duplicate behavior, and generated output;
- review of return alignment and market-session assumptions;
- an independent-model check for the direction of the outlet tone difference.

"Manually validated" is limited to those checks. It does not mean every generated line, headline, provider response, statistical assumption, or migration path received an independent expert audit.

## 4. Rejected or corrected AI-assisted output

These examples are historical defects that were rejected or corrected; they are not descriptions of current behavior.

1. **Destructive relevance cleanup.** An early implementation deleted rows judged irrelevant. The researcher rejected the design and restored affected rows from backup. The current scraper records exclusion metadata, `clean` creates a reversible history row, `restore-exclusion` restores it, and the low-level purge API requires `confirm=True`.
2. **Missing output fabricated as neutral.** An early batch parser filled omitted items with neutral values. The current scorer returns only explicit valid IDs; the pipeline retries the missing subset, preserves NULLs, and records `failed` after the attempt cap. Explicit neutral remains valid.
3. **Return misalignment.** Next-session returns were once shifted after filtering, pairing some news dates with returns 2-15 sessions later. Current predictive consumers form targets on the complete ordered price series before joining session signals.
4. **Duplicate inflation.** NULL-URL items could be reinserted across runs. Source/title/date canonical deduplication fixed replay inflation; the later raw-observation layer also preserves genuine cross-source observations instead of treating them as one ingest record.
5. **Model-name/provenance mix-up.** A provenance version tag was sent as the API request model name. The run failed loudly; request identity and stored provenance were separated.
6. **Calendar-expiring tests.** Hard-coded dates aged out of the active window. Affected tests were changed to stable or relative fixtures.
7. **Overstated terminology.** "Model confidence," "raw probabilities," and "reliable after 30 days" were used for LLM-derived values and a display gate. Current documentation uses model-reported sentiment intensity, synthetic compatibility components, categorical agreement with the project's rubric, and exploratory reporting eligibility.
8. **Weighted specification treated as the signal.** The historical intensity/relevance/time mean was presented without a simple stored baseline. The current session table makes `simple_mean` primary and keeps three weighted variants as sensitivities with no preferred-variant selection on the evaluation sample.
9. **Session storage mistaken for end-to-end alignment.** Initial work created `signal_date` while charts and scripts still used calendar aggregates. Current market-linked consumers read `daily_signal_variants.simple_mean`, with regression tests for the table choice and price lead.

These corrections do not imply all other AI output is correct. They show why preservation, provenance, explicit state, tests, and skeptical review are necessary.

## 5. Current technical safeguards influenced by review

- Source-distinct fetched observations are written to `raw_headline_observations` before canonical deduplication.
- Filter decisions are versioned and reversible; unrelated exclusions are not silently restored.
- Scoring state distinguishes pending, successful, retryable, and exhausted observations.
- Missing/invalid IDs remain absent, and malformed envelopes raise.
- LLM and XLM-R component semantics are stored explicitly.
- Full runs store final and component-level outcomes plus structured warnings/errors.
- Publication timestamps are normalized, timing buckets are stored separately from session assignment, and assignment rules are versioned.
- The unweighted session mean is primary; weighting assumptions are named sensitivities.
- Market-return targets are built before sparse signal joins.

The exact regression evidence for each safeguard is mapped in [docs/TEST_RISK_MAP.md](docs/TEST_RISK_MAP.md), including where coverage is only partial.

## 6. Remaining limitations in technical ownership

- The repository is primarily maintained and interpreted by one researcher; there is no independent maintainer or formal code audit.
- The human-label rubric is primarily one annotator's convention and may not generalize.
- Historical items rejected or deleted before raw-observation auditing cannot be reconstructed by an additive migration.
- Live RSS/provider schema changes and migration rollback are not fully exercised by deterministic tests; partial/all configured source failure decisions now have focused fixtures.
- Mixed scorer versions are reported by the audit but are not automatically blocked from aggregation.
- Exchange holiday/half-day configuration requires continuing authoritative maintenance.
- LLM intensity and derived compatibility fields remain uncalibrated.
- Polarization inference now models selected outlet/date/topic dependence, but few clusters, camp definitions, scorer error, and absent verified shared-event identity remain substantive limitations.
- Statistical and financial conclusions have not been independently replicated end to end. The project has no validated alpha or trading strategy.
- Fresh scoring depends on hosted model/API availability. Stored model and prompt provenance helps audit existing rows but does not make the hosted service immutable.

## 7. Research artifacts and reruns

Additive schema initialization can create tables/columns and classify legacy processing metadata, but it does not silently re-score observations, rebuild derived signal tables, or regenerate findings and figures. Aggregates change only when an aggregate-calling command is deliberately run. Dated research artifacts retain their original sample scope unless explicitly regenerated.

When an AI-assisted correction changes a research result, the update should record:

- the defect and decision owner;
- the affected sample and specification;
- whether raw data, derived data, or presentation changed;
- the before/after interpretation;
- the tests or audit evidence added.

## 8. Review standard for future AI-assisted changes

Future changes should identify the decision owner, preserve inputs, add risk-focused tests, record model/prompt/rule provenance, and distinguish generated suggestions from verified results. A passing suite is evidence for its mapped contracts, not proof that the methodology or conclusion is correct.
