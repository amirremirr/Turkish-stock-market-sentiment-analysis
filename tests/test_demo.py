"""End-to-end contract for the committed, fully offline demo."""

from __future__ import annotations

import json
import socket

import pandas as pd
import pytest

from scripts.demo import run_demo


def test_offline_demo_writes_audited_deterministic_artifacts(tmp_path, monkeypatch):
    def reject_network(*_args, **_kwargs):
        raise AssertionError("the offline demo attempted a network connection")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(socket, "create_connection", reject_network)

    artifacts = run_demo(tmp_path)

    assert set(artifacts) == {"results", "audit", "chart"}
    assert all(path.parent == tmp_path and path.is_file() for path in artifacts.values())
    assert artifacts["chart"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    results = pd.read_csv(artifacts["results"], dtype={"signal_date": str})
    assert list(results.columns[:5]) == [
        "signal_date",
        "simple_mean",
        "relevance_weighted",
        "intensity_relevance_weighted",
        "full_weighted",
    ]
    assert len(results) == 5

    # There is no news signal on June 12.  June 11 must nevertheless target
    # the immediately subsequent exchange session (June 12), not June 15.
    june_11 = results.loc[results["signal_date"] == "2026-06-11"].iloc[0]
    expected = 10300.0 / 10150.0 - 1.0
    sparse_join_error = 10250.0 / 10150.0 - 1.0
    assert june_11["next_session_return"] == pytest.approx(expected, abs=1e-10)
    assert june_11["next_session_return"] != pytest.approx(sparse_join_error)

    # The unknown-time neutral and the post-close positive both belong to the
    # June 11 session; the zero is an observed score and remains in the mean.
    assert june_11["headline_count"] == 2
    assert june_11["simple_mean"] == pytest.approx(0.125)

    audit = json.loads(artifacts["audit"].read_text(encoding="utf-8"))
    assert audit["all_checks_passed"] is True
    assert audit["demo_contract"]["offline"] is True
    assert audit["demo_contract"]["primary_signal"] == "simple_mean"
    assert audit["demo_contract"]["weighted_variants_role"] == "sensitivity_only"
    assert audit["record_classification"] == {
        "excluded": 1,
        "explicit_neutral": 2,
        "failed": 1,
        "missing": 1,
        "scored_non_neutral": 5,
    }
