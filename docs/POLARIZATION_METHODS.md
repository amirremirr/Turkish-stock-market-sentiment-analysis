# Polarization inference methods

**Implemented:** 2026-08-01
**Claim scope:** observational and descriptive; no causal political-bias claim

The maintained analysis is `analysis.polarization.inference`. It treats story
selection and story framing as different estimands and exposes failed or
inadequate inference rather than falling back to an independent-headline test.

Run the read-only report with:

```bash
python -m analysis.polarization.inference --db finance_sentiment.db
```

Nothing is written by default. Add `--json-output PATH` only when deliberately
creating a new dated research artifact.

## Analysis rows

Polarization is outlet-level analysis. The loader therefore prefers
source-distinct `raw_headline_observations` joined to the eligible canonical
score. This preserves two outlets observing the same canonical URL as two
outlet observations. Canonical headlines without a linked raw observation are
included as a legacy fallback. Pending, failed, and actively excluded headlines
are not analyzed.

Outlet camps are researcher-specified groupings. Their definitions are emitted
in every report and should be varied in sensitivity work; they are not inferred
facts about an outlet.

## Aggregate tone difference

The report includes:

- headline counts, raw means, and sample standard deviations by camp and outlet;
- the raw pro-government-minus-opposition mean difference;
- pooled within-camp Cohen's *d*;
- a deterministic percentile interval from resampling whole publication dates;
- an OLS association, `sentiment ~ camp + C(category) + C(date)`;
- conventional and cluster-robust uncertainty diagnostics.

Cluster-robust standard errors are attempted separately by outlet and date.
Event clustering is attempted only when a non-missing explicit canonical/shared
event identifier contains at least two repeated cross-camp events. The report
states cluster counts, rank or residual-degree problems, numerical failures, and
a small-cluster warning below 30 clusters.

The date-cluster bootstrap preserves within-date dependence. It does not solve
dependence shared across dates, scorer measurement error, camp-definition
uncertainty, or selection into the collected source set.

## Selection and framing

The selection section compares topic shares and, when canonical events exist,
cross-camp versus camp-only event coverage. These are descriptive coverage
differences.

The framing section holds an explicit repeated canonical event fixed and reports
the distribution of within-event tone gaps. The repository's current event
bridge is one event row per headline; its IDs are deliberately never treated as
shared-event evidence.

When no defensible shared identifier exists, the report may show a deterministic
lexical/date fallback. It globally selects one-to-one pairs without reusing a
headline. These pairs are labeled unverified sensitivity candidates, not known
same events, and should be manually inspected before substantive interpretation.

## Interpretation limits

The analysis can support phrases such as “systematic tone difference,”
“outlet-associated framing difference,” or “descriptive polarization pattern.”
It cannot establish editorial intent, causal political bias, scorer truth, or a
market effect. Fixed effects and clustered standard errors address selected
forms of dependence; they do not repair omitted variables, measurement error,
thin camps, few clusters, or inadequate event identity.

The checked-in `docs/polarization_findings.md` and figures remain dated research
snapshots. Implementing this method did not silently regenerate or revise those
results.
