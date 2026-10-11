"""Pre-run readiness audit of the stock-level study.

Lives outside ``stock_research/`` on purpose: the protocol fingerprint covers
that package, and an audit must not change what it audits.

Nothing here computes a real abnormal return around a sampled event. Real data
is read from a snapshot copy of the research database, and only for three
things: the *structure* of the sample (who, when, which category), whether a
return *exists* for a session (never its value in an event window), and
price-level facts about corporate-action dates and the pre-sample period
(2022, before any sampled event).
"""
