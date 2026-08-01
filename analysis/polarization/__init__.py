"""Transparent, non-causal media-polarization inference helpers."""

from importlib import import_module

__all__ = [
    "DEFAULT_OPPOSITION_SOURCES",
    "DEFAULT_PRO_GOVERNMENT_SOURCES",
    "analyze_polarization",
    "date_cluster_bootstrap",
    "lexical_date_pairs",
    "load_headlines",
]


def __getattr__(name):
    """Keep public conveniences lazy so ``python -m ...inference`` is warning-free."""

    if name in __all__:
        return getattr(import_module(".inference", __name__), name)
    raise AttributeError(name)
