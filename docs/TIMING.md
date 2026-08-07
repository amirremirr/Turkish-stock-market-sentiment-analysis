# Timing: what every date field means

A return is only a valid target if someone could have earned it. That turns on
one question — *when could a position first be opened?* — and this document is
the single answer, replacing three call sites that used to imply three.

## The proven convention

**`signal_date` is the first trading session capable of reacting to the
publication.** It is not the session the news was published in.

This was proven, not assumed. `scripts/timing_audit.py` states two hypotheses
and lets production records refute one:

- **A** — `signal_date` is the publication / associated market session.
- **B** — `signal_date` is the first session capable of reacting.

`pre_open` and `during_session` cannot distinguish them: for those the
publication session *is* the reactable session. The verdict rests on the three
buckets where the two disagree.

| bucket | rows | matches A | A defined | matches B |
|---|---|---|---|---|
| `pre_open` | 813 | 813 | 813 | 813 |
| `during_session` | 2 222 | 2 222 | 2 222 | 2 222 |
| **`post_close`** | **172** | **0** | 172 | **172** |
| **`weekend_or_holiday`** | **251** | **0** | 0 | **251** |
| **`unknown`** | **435** | **0** | 435 | **435** |

Hypothesis A is refuted by 607 of 607 discriminating rows. Hypothesis B holds
for 3 893 of 3 893. Verdict: **`first_reactable_session`**.

The invariant that follows, exported by `research/timing.py` so nothing
re-derives it:

```
first_reactable_session == signal_date        for every bucket
```

## Field definitions

| Field | Meaning |
|---|---|
| `published_timestamp` | Publication moment, normalised to Europe/Istanbul. Naive values are read as Istanbul local. NULL when the source gave no time. |
| `timing_bucket` | Where the publication falls relative to the session: `pre_open`, `during_session`, `post_close`, `weekend_or_holiday`, `unknown`. |
| `signal_date` | The first reactable session. Kept for compatibility. |
| `first_reactable_session` | The same value under a name that states its meaning. |
| `first_reactable_at` | The opening bell of that session (10:00 Istanbul). For `during_session`, the publication moment itself — which is why it is blocked. |
| `information_cutoff` | The latest moment whose information a window's **entry** assumes. |
| `assumed_execution` | When the position is assumed to open. |
| `entry_date` / `exit_date` | The sessions supplying the entry and exit prices. |
| `event_information_cutoff` | For a group: the publication moment of its **last** member. |
| `event_timing_rule_version` | `event-timing-v1`. |
| `timing_conflict` | 1 when members react on different sessions. |
| `timing_conflict_reason` | `members_span_multiple_reactable_sessions`, `governing_member_timing_unknown`. |

## The defect this document exists because of

v1 read `signal_date` as the publication session and stepped forward to "the
next session". Since `signal_date` was *already* the reactable session, every
`post_close` and `weekend_or_holiday` window was built one session late.

Production row, headline 1647 — published 2026-06-08 21:00 Istanbul,
`signal_date` 2026-06-09:

| | v1 (wrong) | v2 (correct) |
|---|---|---|
| information cutoff | 2026-06-09 18:10 | 2026-06-08 18:10 |
| assumed execution | 2026-06-10 10:00 | 2026-06-09 10:00 |
| primary entry → exit | close 06-09 → open 06-10 | open 06-09 → close 06-09 |

This is **not a look-ahead leak** — a late window uses less information than it
could. It is worse in a subtler way: it measures the session *after* the one the
news could move, so a real relationship would surface as a null and be written
up as "no predictive content found". A leak makes you wrong loudly; this makes
you wrong quietly.

Every `post_close` and `weekend_or_holiday` window in the database was affected.
All were rebuilt; headlines, scores, labels, categories and experiment
identities were untouched.

## Windows (v2)

With `D` the first reactable session and `P` the trading session before it:

| Window | Entry → exit | Tradable | Why |
|---|---|---|---|
| `reactable_open_to_close` | open(D) → close(D) | **yes** | the news is public before the bell |
| `prior_close_to_reactable_open` | close(P) → open(D) | no | entry predates publication |
| `prior_close_to_reactable_close` | close(P) → close(D) | no | entry predates publication |

The gap windows are kept because that is where a pre-open story's reaction
actually lands — an open-to-close return can miss it entirely. They are labelled
for what they are: a measurement of reaction, not an achievable return.

`P` is the calendar-previous trading session, not "the last bar we happen to
have". Across a missing bar the latter would silently span several sessions and
report the total as one overnight gap.

Because both `pre_open` and `post_close` execute at open(D), **the primary
window has the same shape for both** — which is what makes them poolable into
one statistical unit.

## Event-level timing

A candidate event is not fully known until its **last** member is published.
So:

- `first_reactable_session` = the **latest** `signal_date` among members.
- Ties on that session break toward the most restrictive bucket.
- The bucket is then read from the **same member** that supplied the session.

`governing_headline_id` records which member that was. Taking the earliest
`signal_date` — the behaviour this replaces — claimed an event was actionable
before part of it existed. Pairing one member's bucket with another's session
was the same error wearing a different hat.

Groups whose members straddle sessions are flagged `timing_conflict`, excluded
from primary evaluation, and retained descriptively.

## What stays blocked

| Bucket | Reason |
|---|---|
| `during_session` | no intraday BIST prices, so no defensible entry price |
| `unknown` | publication time unknown, so no known reaction session |

Only `complete` and `corrected` price bars are visible to the window builder.
A provisional bar is an intraday snapshot; using one would reintroduce the fault
that corrupted 2026-07-31.

## Re-running the audit

```bash
python -m scripts.timing_audit --db finance_sentiment.db --per-bucket 25
```

Exit code 0 means the verdict matches the declared semantics *and* every sampled
row's generated window matches the independently derived expectation. The audit
derives expectations from the calendar rather than from the window builder, so
it is not comparing the code with itself.
