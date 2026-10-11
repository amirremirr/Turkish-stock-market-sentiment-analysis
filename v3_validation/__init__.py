"""Validation of the candidate protocol v3, on synthetic returns only.

Like ``readiness_audit``, this package sits outside ``stock_research/`` and so
outside the protocol fingerprint. It reuses the readiness audit's simulator
(``readiness_audit.power.simulate_cars``) and sample projection, so the audit's
calibrated numbers are the benchmarks the candidate procedures are held to.

From real data it reads the same three things the audit read, and nothing
else: which issuer filed in which category on which session; whether the
returns an event would need exist; and residuals of 2022 for the noise
calibration. No event-window return of a sampled disclosure is computed.
"""
