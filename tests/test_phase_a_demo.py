"""The Phase A offline demo must stay credential-free, fast and deterministic.

Its purpose is to let a reader see the descriptive layer without being granted
access to anything, so the absence of network, model and private-database access
is part of the contract and is asserted rather than assumed.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.demo_phase_a import run_demo


@pytest.fixture
def demo_artifacts(tmp_path, monkeypatch):
    def _no_network(*args, **kwargs):                 # pragma: no cover
        raise AssertionError("the offline demo must not open a socket")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    return run_demo(tmp_path / "demo")


def test_demo_produces_all_artifacts(demo_artifacts):
    assert set(demo_artifacts) == {"summary", "family_signals", "drivers"}
    for path in demo_artifacts.values():
        assert Path(path).stat().st_size > 0


def test_demo_is_deterministic(tmp_path):
    first = run_demo(tmp_path / "a")["summary"].read_text(encoding="utf-8")
    second = run_demo(tmp_path / "b")["summary"].read_text(encoding="utf-8")
    assert first == second


def test_demo_demonstrates_every_phase_a_capability(demo_artifacts):
    summary = json.loads(demo_artifacts["summary"].read_text(encoding="utf-8"))

    # signal families
    assert len(summary["families_present"]) >= 5
    assert "monetary_policy" in summary["families_present"]
    assert "banking_financial_sector" in summary["families_present"]

    # market-recap classification and its exclusion from the directional sample
    assert summary["market_recap"]["count"] > 0
    assert summary["market_recap"]["excluded_from_directional_sample"] is True
    assert summary["market_recap"]["directional_sample_size"] > 0

    # domestic-only aggregation
    assert summary["domestic_only_latest"] is not None
    assert summary["domestic_only_latest"]["headline_count"] > 0

    # abnormal tone, disagreement, volume, drivers
    assert summary["abnormal_tone_latest"]
    assert summary["disagreement_latest"]
    assert summary["volume_latest"]
    assert summary["drivers_latest"]


def test_demo_shows_null_where_history_or_sources_are_insufficient(demo_artifacts):
    """A reader must be able to see the NULL discipline, not just be told about it."""

    summary = json.loads(demo_artifacts["summary"].read_text(encoding="utf-8"))
    assert any(
        row["abnormal_tone"] is None for row in summary["abnormal_tone_latest"]
    ), "at least one family should lack the history for an abnormal reading"
    assert any(
        row["cross_outlet_std"] is None for row in summary["disagreement_latest"]
    ), "a single-outlet family must report NULL, not a fabricated zero"


def test_demo_shows_syndication_collapsing_to_one_event(demo_artifacts):
    """Four outlets carrying one decision is one event, four sources."""

    summary = json.loads(demo_artifacts["summary"].read_text(encoding="utf-8"))
    monetary = next(
        row for row in summary["volume_latest"]
        if row["signal_family"] == "monetary_policy"
    )
    assert monetary["headline_count"] > monetary["observation_count"]
    assert monetary["source_breadth"] == monetary["headline_count"]


def test_demo_reports_camp_disagreement_where_both_camps_are_present(demo_artifacts):
    summary = json.loads(demo_artifacts["summary"].read_text(encoding="utf-8"))
    monetary = next(
        row for row in summary["disagreement_latest"]
        if row["signal_family"] == "monetary_policy"
    )
    assert monetary["camp_gap"] is not None
    assert monetary["min_sources_met"] == 1


def test_demo_makes_no_predictive_claim(demo_artifacts):
    """The demo may deny a predictive claim; it must never make one."""

    summary = json.loads(demo_artifacts["summary"].read_text(encoding="utf-8"))
    notes = " ".join(summary["notes"]).lower()
    assert "descriptive only" in notes
    assert "nothing here is a validated predictive signal" in notes

    # Claim-shaped language anywhere outside the disclaimers would be a problem.
    # Disclaimers are stripped wherever they appear, including inside nested
    # event briefs, so denying a claim never trips the check.
    disclaimer_fields = {"notes", "disclaimer", "status_note", "note",
                         "dispersion_note"}

    def _strip(value):
        if isinstance(value, dict):
            return {
                key: _strip(item) for key, item in value.items()
                if key not in disclaimer_fields
            }
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    body = json.dumps(_strip(summary), default=str).lower()
    for forbidden in ("alpha", "buy signal", "sell signal", "profitable",
                      "outperform", "validated"):
        assert forbidden not in body, f"{forbidden!r} appears outside a disclaimer"


def test_demo_runs_quickly(tmp_path):
    import time

    started = time.perf_counter()
    run_demo(tmp_path / "timed")
    assert time.perf_counter() - started < 10.0
