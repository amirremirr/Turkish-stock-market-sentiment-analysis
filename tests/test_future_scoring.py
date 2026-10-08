"""The untouched-future runner and its scoring, on synthetic data only.

No test here reads the production database: the untouched sample's outcomes
must not be computed before it is eligible, and that includes by a test.
"""

import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

import database as db
from research.future_scoring import score
from research.future_validation import definition, definition_hash
from research.protocol import _spec, protocol_hash
from research.walkforward import FITTED, Fold
from scripts import run_future_validation as runner

FROZEN = Path(__file__).resolve().parent.parent / "docs" / "frozen"
NEWS_FEATURES = ("mean_tone", "breadth_weighted_tone", "tone_dispersion", "positive_share")


def _contract():
    return definition(protocol_hash=protocol_hash())


def _units(n, *, signal, seed=11):
    """n synthetic sessions from 2026-08-10; *signal* scales how much
    mean_tone drives the return."""

    rng = random.Random(seed)
    day, units, previous = date(2026, 8, 10), [], None
    while len(units) < n:
        if day.weekday() < 5:
            tone = rng.gauss(0, 1)
            ret = signal * tone + rng.gauss(0, 1)
            unit = {
                "first_reactable_session": day.isoformat(),
                "exit_date": day.isoformat(),
                "raw_return": ret,
                "prev_return": previous,
                "prev_return_sign": None if previous is None else (1.0 if previous > 0 else -1.0),
                "headline_count": rng.randint(5, 40),
                "net_tone_share": rng.gauss(0, 1),
                "eem_lag1": rng.gauss(0, 1), "brent_lag1": rng.gauss(0, 1),
                "usdtry_lag1": rng.gauss(0, 1),
                "mean_tone": tone, "breadth_weighted_tone": tone + rng.gauss(0, 0.1),
                "tone_dispersion": abs(rng.gauss(0, 1)), "positive_share": rng.random(),
                "abnormal_tone": rng.gauss(0, 1), "abnormal_tone_domestic": rng.gauss(0, 1),
                "mean_cross_source_dispersion": rng.gauss(0, 1),
                "source_breadth": rng.randint(1, 8), "event_count": rng.randint(1, 20),
                "max_novelty": rng.random(), "multi_source_events": rng.randint(0, 5),
                "dominant_family": "other", "dominant_timing_bucket": "pre_open",
                "regime": None,
            }
            units.append(unit)
            previous = ret
        day += timedelta(days=1)
    return units


# -- End to end on synthetic sessions ----------------------------------------

def test_a_real_signal_on_enough_sessions_is_a_success():
    result = runner.evaluate_untouched(_units(90, signal=3.0), _contract())
    assert result["folds_fittable"] >= 3
    assert result["verdict"] == "success"
    winners = {r["feature_set"] for r in result["comparisons"] if r["meets_success_criteria"]}
    # Only sets carrying the tone that drives the synthetic return can win.
    assert winners <= {"family_signals", "event_tone_novelty", "controls_plus_news"}


def test_pure_noise_is_not_a_success():
    result = runner.evaluate_untouched(_units(90, signal=0.0), _contract())
    assert result["verdict"] in {"failure", "inconclusive"}
    assert not any(r["meets_success_criteria"] for r in result["comparisons"])


def test_fewer_than_three_folds_is_inconclusive_even_with_a_signal():
    # 51-70 sessions build one or two primary folds.
    result = runner.evaluate_untouched(_units(65, signal=3.0), _contract())
    assert result["folds_fittable"] < 3
    assert result["verdict"] == "inconclusive"


def test_bonferroni_level_tracks_the_number_of_news_specifications():
    result = runner.evaluate_untouched(_units(90, signal=3.0), _contract())
    m = result["news_specifications"]
    assert m > 1
    assert result["interval_level"] == pytest.approx(1 - 0.05 / m)


# -- The majority-of-folds rule, on handcrafted predictions -------------------

def _spec_with(kind, name, predictions):
    return {"feature_set": name, "model": "ridge", "target": "raw_return",
            "kind": kind, "status": FITTED, "predictions": predictions}


def _crafted(perfect_folds, folds=4, per_fold=10):
    """A baseline that always predicts +0.1, and a news spec that is perfect in
    *perfect_folds* and identical to the baseline elsewhere."""

    rng = random.Random(3)
    built, base, news, day = [], [], [], 0
    for index in range(1, folds + 1):
        sessions = []
        for _ in range(per_fold):
            session = f"2026-09-{day + 1:02d}" if day < 30 else f"2026-10-{day - 29:02d}"
            day += 1
            sessions.append(session)
            actual = rng.choice([-1.0, 1.0]) * (0.5 + rng.random())
            row = {"fold": index, "first_reactable_session": session,
                   "exit_date": session, "actual": actual}
            base.append({**row, "predicted": 0.1})
            news.append({**row, "predicted": actual if index in perfect_folds else 0.1})
        built.append(Fold(index=index, train=("x",), test=tuple(sessions), embargoed=()))
    return built, [_spec_with("baseline", "none", base), _spec_with("news", "family_signals", news)]


def test_margins_in_only_half_the_folds_is_inconclusive():
    folds, specs = _crafted(perfect_folds={1, 2})
    result = score(specs, folds, _spec()["decision_thresholds"])
    row = result["comparisons"][0]
    assert row["clears_pooled_margins"] and row["folds_cleared"] == 2
    assert not row["majority_of_folds"]
    assert result["verdict"] == "inconclusive"


def test_margins_in_a_majority_of_folds_is_a_success():
    folds, specs = _crafted(perfect_folds={1, 2, 3})
    result = score(specs, folds, _spec()["decision_thresholds"])
    assert result["comparisons"][0]["majority_of_folds"]
    assert result["verdict"] == "success"


def test_no_fitted_baseline_is_a_failure():
    folds, specs = _crafted(perfect_folds={1, 2, 3})
    result = score(specs[1:], folds, _spec()["decision_thresholds"])
    assert result["verdict"] == "failure"


# -- Sealed inputs -------------------------------------------------------------

def test_todays_code_reproduces_the_sealed_protocol_and_definition():
    frozen = json.loads((FROZEN / "walk-forward-protocol-v1.json").read_text(encoding="utf-8"))
    sealed = json.loads((FROZEN / "untouched_future_v1.json").read_text(encoding="utf-8"))
    frozen_spec = frozen["study"]["protocol_specification"]
    assert protocol_hash(frozen_spec) == sealed["protocol_hash"]

    current = json.loads(json.dumps(_spec()))
    section, key = runner._PERMITTED_SPEC_DIFFERENCE
    assert current[section][key] == sealed["allowed_feature_versions"]["dataset_version"]
    current[section][key] = frozen_spec[section][key]
    assert protocol_hash(current) == sealed["protocol_hash"]

    rebuilt = definition(protocol_hash=sealed["protocol_hash"],
                         frozen_artifact_hash=sealed["frozen_retrospective_artifact_hash"])
    assert definition_hash(rebuilt) == sealed["definition_hash"]


def test_the_structurally_unfittable_first_fold_is_not_counted():
    result = runner.evaluate_untouched(_units(90, signal=3.0), _contract())
    assert result["folds_built"] == result["folds_fittable"] + 1


def test_not_ready_refuses_before_any_outcome_is_read(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "check_sealed_inputs", lambda path: {
        "stored": {"definition_hash": "h" * 64}, "definition": _contract()})
    monkeypatch.setattr("scripts.future_readiness.build_report", lambda path: {
        "eligible_to_run": False, "blocking_reasons": ["42 untouched sessions < 51 required"]})

    def must_not_run(*args, **kwargs):
        raise AssertionError("outcomes were read before readiness")

    monkeypatch.setattr(runner, "_untouched_units", must_not_run)
    monkeypatch.setattr(runner, "evaluate_untouched", must_not_run)
    with pytest.raises(runner.Refused, match="not ready"):
        runner.run(str(tmp_path / "gate.db"), write_file=False)


# -- One result, append-only ---------------------------------------------------

def _result(**overrides):
    base = {"definition_hash": "d" * 64, "version": "untouched_future_v1",
            "verdict": "failure", "verdict_reason": "r", "untouched_sessions": 80,
            "folds": 4, "news_specifications": 12, "result_hash": "r" * 64,
            "completed_at": "2026-12-07T00:00:00+00:00"}
    return {**base, **overrides}


def test_a_second_result_for_the_same_definition_is_refused(tmp_path):
    path = str(tmp_path / "f.db")
    db.init_db(db_path=path)
    db.record_future_validation_result(_result(), db_path=path)
    with pytest.raises(RuntimeError, match="runs once"):
        db.record_future_validation_result(_result(verdict="success"), db_path=path)

    connection = sqlite3.connect(path)
    try:
        for statement in ("UPDATE future_validation_results SET verdict='success'",
                          "DELETE FROM future_validation_results"):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement)
    finally:
        connection.close()
