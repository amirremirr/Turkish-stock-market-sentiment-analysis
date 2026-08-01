"""Pure aggregation helpers for auditable sentiment signals."""

from .signals import compute_signal_variants, legacy_time_weight

__all__ = ["compute_signal_variants", "legacy_time_weight"]
