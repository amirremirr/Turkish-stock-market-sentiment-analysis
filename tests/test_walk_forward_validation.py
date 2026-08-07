"""Walk-forward validation: leakage, fold boundaries, the unit, and the freeze.

The properties asserted here are the ones whose violation would not show up as
an error but as a *better number*. A leaked test fold, a full-sample
standardiser, or a duplicated outcome all make results improve, so nothing in
the ordinary run would complain. They have to be tested directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.controls import CONTROL_SETS, KIND_TRADABLE
from research.modelling_unit import (
    DEFAULT_EMBARGO_SESSIONS, attach_lagged_features, build_session_units,
    eligible_rows, unit_counts,
)
from research.protocol import (
    FEATURE_SETS, PROTOCOL_STATUS, canonical_json, is_tradable_specification,
    protocol_document, protocol_hash, select_geometry,
)
from research.return_windows import PRIMARY_WINDOW
from research.walkforward import (
    FITTED, INSUFFICIENT, Standardiser, build_folds, cluster_bootstrap,
    direction_metrics, evaluate_specification, fold_boundaries_are_safe,
    regression_metrics, sample_gate,
)


def _dataset(sessions=60, events_per_session=3):
    """A synthetic dataset shaped like production: many events, one outcome."""

    rows = []
    for index in range(sessions):
        session = f"2026-{3 + index // 28:02d}-{1 + index % 28:02d}"
        outcome = ((index * 37) % 21 - 10) / 10.0
        for event in range(events_per_session):
            rows.append({
                "group_key": f"g{index}-{event}",
                "window_name": PRIMARY_WINDOW,
                "first_reactable_session": session,
                "signal_date": session,
                "eligibility_status": "eligible",
                "is_tradable_window": 1,
                "timing_conflict": 0,
                "entry_date": session, "exit_date": session,
                "raw_return": outcome,
                "residual_none": outcome,
                "residual_em_lagged": None,
                "residual_em_oil_fx_lagged": None,
                "residual_em_contemporaneous": None,
                "headline_count": 2, "source_count": 2,
                "mean_sentiment": ((index + event) % 7 - 3) / 3.0,
                "cross_source_dispersion": 0.2,
                "novelty": 0.5,
                "signal_family": "monetary_policy",
                "timing_bucket": "pre_open",
            })
    return rows


# ---------------------------------------------------------------------------
# The statistical unit
# ---------------------------------------------------------------------------
class TestModellingUnit:
    def test_one_row_per_session_not_per_event(self):
        dataset = _dataset(sessions=10, events_per_session=4)
        units = build_session_units(dataset)
        assert len(units) == 10
        assert len({u["first_reactable_session"] for u in units}) == 10

    def test_duplicated_outcomes_are_reported_not_hidden(self):
        dataset = _dataset(sessions=10, events_per_session=4)
        counts = unit_counts(dataset, build_session_units(dataset))
        assert counts["event_rows"] == 40
        assert counts["distinct_sessions"] == 10
        assert counts["session_units"] == 10
        # 40 rows, at most 10 independent outcomes. The gap is the point.
        assert counts["distinct_outcomes"] <= 10
        assert counts["duplication_factor"] >= 4.0

    def test_conflicting_outcomes_on_one_session_are_an_error(self):
        dataset = _dataset(sessions=2, events_per_session=2)
        dataset[1]["raw_return"] = 99.0        # same session, different target
        with pytest.raises(ValueError, match="must share one outcome"):
            build_session_units(dataset)

    def test_blocked_and_untradable_rows_never_enter_the_sample(self):
        dataset = _dataset(sessions=5, events_per_session=2)
        dataset[0]["eligibility_status"] = "blocked"
        dataset[1]["is_tradable_window"] = 0
        dataset[2]["timing_conflict"] = 1
        selected = eligible_rows(dataset)
        keys = {row["group_key"] for row in selected}
        assert dataset[0]["group_key"] not in keys
        assert dataset[1]["group_key"] not in keys
        assert dataset[2]["group_key"] not in keys

    def test_lagged_features_never_read_the_current_session(self):
        units = attach_lagged_features(build_session_units(_dataset(sessions=6, events_per_session=1)))
        assert units[0]["prev_return"] is None, "the first session has no predecessor"
        for earlier, later in zip(units, units[1:]):
            assert later["prev_return"] == earlier["raw_return"]

    def test_missing_predecessor_is_null_not_zero(self):
        units = attach_lagged_features(build_session_units(_dataset(sessions=3, events_per_session=1)))
        assert units[0]["prev_return"] is None
        assert units[0]["prev_return_sign"] is None

    def test_only_lagged_factors_are_exposed(self):
        panel = {"2026-03-01": {"EEM": 1.5, "EEM_lag1": 0.5}}
        units = attach_lagged_features(
            build_session_units(_dataset(sessions=3, events_per_session=1)),
            factor_panel=panel,
        )
        first = units[0]
        assert first["eem_lag1"] == 0.5
        assert "EEM" not in first and "eem" not in first


# ---------------------------------------------------------------------------
# Fold boundaries
# ---------------------------------------------------------------------------
class TestFolds:
    def test_folds_are_chronological_and_disjoint(self):
        sessions = [f"2026-03-{d:02d}" for d in range(1, 29)]
        folds = build_folds(sessions, initial_train=10, test_size=5, step=5)
        assert folds
        assert fold_boundaries_are_safe(folds)["safe"] is True
        for fold in folds:
            assert max(fold.train) < min(fold.test)
            assert not set(fold.train) & set(fold.test)

    def test_embargo_removes_the_adjacent_session_from_training(self):
        sessions = [f"2026-03-{d:02d}" for d in range(1, 29)]
        folds = build_folds(
            sessions, initial_train=10, test_size=5, step=5,
            embargo=DEFAULT_EMBARGO_SESSIONS,
        )
        first = folds[0]
        assert len(first.embargoed) == DEFAULT_EMBARGO_SESSIONS
        assert first.embargoed[0] not in first.train
        assert first.embargoed[0] not in first.test
        assert len(first.train) == 10 - DEFAULT_EMBARGO_SESSIONS

    def test_training_always_expands_and_never_wraps(self):
        sessions = [f"2026-03-{d:02d}" for d in range(1, 29)]
        folds = build_folds(sessions, initial_train=10, test_size=5, step=5)
        for earlier, later in zip(folds, folds[1:]):
            assert len(later.train) > len(earlier.train)
            assert set(earlier.train) <= set(later.train)

    def test_no_fold_when_the_sample_is_too_short(self):
        assert build_folds(["2026-03-01"], initial_train=10, test_size=5, step=5) == []

    def test_unsafe_design_is_detected(self):
        from research.walkforward import Fold

        broken = [Fold(index=1, train=("2026-03-05",), test=("2026-03-01",),
                       embargoed=())]
        report = fold_boundaries_are_safe(broken)
        assert report["safe"] is False
        assert report["violations"]


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------
class TestLeakage:
    def test_standardiser_uses_only_the_training_fold(self):
        train = [[1.0], [2.0], [3.0]]
        scaler = Standardiser().fit(train)
        before = list(scaler.means), list(scaler.deviations)
        scaler.transform([[100.0]])          # a wild test value
        assert (list(scaler.means), list(scaler.deviations)) == before

    def test_transform_does_not_recentre_the_test_fold(self):
        scaler = Standardiser().fit([[0.0], [2.0]])
        transformed = scaler.transform([[10.0], [12.0]])
        # If the scaler had been refitted on the test fold these would be
        # symmetric about zero.
        assert all(value > 0 for row in transformed for value in row)

    def test_constant_column_does_not_explode(self):
        scaler = Standardiser().fit([[5.0], [5.0], [5.0]])
        assert scaler.transform([[5.0]]) == [[0.0]]

    def test_no_random_splitting_anywhere_in_the_module(self):
        source = (REPOSITORY_ROOT / "research" / "walkforward.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("train_test_split", "shuffle", "random.sample",
                          "rng.permutation"):
            assert forbidden not in source

    def test_contemporaneous_controls_are_never_tradable(self):
        for name, definition in CONTROL_SETS.items():
            if "contemporaneous" in name:
                assert definition["kind"] != KIND_TRADABLE
        assert is_tradable_specification("controls_plus_news", "raw_return") is True
        assert is_tradable_specification(
            "controls_plus_news", "residual_em_contemporaneous"
        ) is False

    def test_no_feature_set_contains_a_contemporaneous_input(self):
        for name, spec in FEATURE_SETS.items():
            for feature in spec["features"]:
                assert not feature.endswith("_contemporaneous")
                # A lagged factor is fine; an unlagged one is not.
                assert feature not in ("eem", "brent", "usdtry")


# ---------------------------------------------------------------------------
# The sample gate
# ---------------------------------------------------------------------------
class TestSampleGate:
    def test_gate_refuses_and_names_the_binding_requirement(self):
        rows = [
            {"first_reactable_session": f"2026-03-{d:02d}", "raw_return": 1.0,
             "x": 1.0}
            for d in range(1, 6)
        ]
        folds = build_folds(
            [r["first_reactable_session"] for r in rows],
            initial_train=40, test_size=10, step=10,
        )
        gate = sample_gate(rows, ["x"], "raw_return", minimum_sessions=40,
                           folds=folds, minimum_test_sessions=8)
        assert gate["status"] == INSUFFICIENT
        assert "5 usable rows" in gate["binding_requirement"]
        assert "40" in gate["binding_requirement"]

    def test_gate_counts_missing_values_per_column(self):
        rows = [
            {"first_reactable_session": "2026-03-01", "raw_return": 1.0, "x": None},
            {"first_reactable_session": "2026-03-02", "raw_return": None, "x": 1.0},
        ]
        gate = sample_gate(rows, ["x"], "raw_return", minimum_sessions=1,
                           folds=[], minimum_test_sessions=1)
        assert gate["missing_by_column"] == {"x": 1, "raw_return": 1}
        assert gate["rows_complete"] == 0

    def test_blocked_specification_is_not_fitted(self):
        dataset = _dataset(sessions=6, events_per_session=1)
        units = attach_lagged_features(build_session_units(dataset))
        folds = build_folds(
            [u["first_reactable_session"] for u in units],
            initial_train=40, test_size=10, step=10,
        )
        outcome = evaluate_specification(
            units, feature_set_name="ar1", model_name="ridge",
            target="raw_return", folds=folds, minimum_sessions=40,
            minimum_test_sessions=8,
        )
        assert outcome["status"] == INSUFFICIENT
        assert outcome["predictions"] == []
        assert outcome["gate"]["binding_requirement"]

    def test_a_sufficient_sample_actually_fits(self):
        dataset = _dataset(sessions=60, events_per_session=2)
        units = attach_lagged_features(build_session_units(dataset))
        folds = build_folds(
            [u["first_reactable_session"] for u in units],
            initial_train=25, test_size=6, step=6,
        )
        outcome = evaluate_specification(
            units, feature_set_name="ar1", model_name="ridge",
            target="raw_return", folds=folds, minimum_sessions=25,
            minimum_test_sessions=5,
        )
        assert outcome["status"] == FITTED
        assert outcome["predictions"]
        assert outcome["pooled"]["prediction_coverage"] is not None

        # Per fold, not pooled: an expanding window is *supposed* to train on
        # sessions an earlier fold tested, so pooling every fold's training set
        # would flag correct behaviour. What must never happen is a prediction
        # for a session that its own fold trained on.
        by_index = {fold.index: fold for fold in folds}
        for prediction in outcome["predictions"]:
            fold = by_index[prediction["fold"]]
            session = prediction["first_reactable_session"]
            assert session in fold.test
            assert session not in fold.train
            assert session not in fold.embargoed


# ---------------------------------------------------------------------------
# The freeze
# ---------------------------------------------------------------------------
class TestProtocolFreeze:
    def test_hash_is_stable_across_calls(self):
        assert protocol_hash() == protocol_hash()

    def test_hash_ignores_provenance_but_covers_the_specification(self):
        first = protocol_document(code_commit="aaa", database_snapshot="1")
        second = protocol_document(code_commit="bbb", database_snapshot="2")
        assert first["protocol_hash"] == second["protocol_hash"]

        mutated = json.loads(canonical_json(first["specification"]))
        mutated["decision_thresholds"]["alpha"] = 0.10
        assert protocol_hash(mutated) != first["protocol_hash"]

    def test_protocol_is_labelled_retrospective(self):
        document = protocol_document()
        assert document["specification"]["status"] == PROTOCOL_STATUS
        assert "not an untouched future test" in (
            document["specification"]["status_note"].lower()
        )

    def test_prohibitions_are_recorded_in_the_protocol(self):
        prohibited = protocol_document()["specification"]["prohibited"]
        assert "random train/test splitting" in prohibited
        assert "full-sample normalisation" in prohibited
        assert "feature selection using test results" in prohibited
        assert any("transaction-cost" in item for item in prohibited)

    def test_geometry_depends_only_on_session_count(self):
        assert select_geometry(80)["name"] == "primary"
        assert select_geometry(49)["name"] == "reduced"
        assert select_geometry(10)["name"] == "none"

    def test_every_geometry_can_fit_at_its_own_threshold(self):
        """A geometry that applies where it cannot fit is a broken geometry.

        The first production run selected `primary` at exactly 50 sessions and
        then blocked all 72 specifications, because 40 training + 1 embargo +
        10 test needs 51. Thresholds are now derived, and this pins it.
        """

        from research.protocol import _spec

        specification = _spec()
        for name, geometry in specification["folds"]["geometries"].items():
            threshold = geometry["applies_when_sessions_at_least"]
            sessions = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
                        for i in range(threshold)]
            folds = build_folds(
                sessions,
                initial_train=geometry["initial_train_sessions"],
                test_size=geometry["test_sessions"],
                step=geometry["step_sessions"],
                embargo=specification["folds"]["embargo_sessions"],
            )
            assert folds, f"{name} applies at {threshold} but produces no fold"
            first = folds[0]
            assert len(first.train) >= geometry["initial_train_sessions"] - (
                specification["folds"]["embargo_sessions"]
            )
            assert len(first.test) >= geometry["minimum_test_sessions_per_fold"]

    def test_geometry_thresholds_are_not_hand_written(self):
        from research.protocol import _spec

        specification = _spec()
        embargo = specification["folds"]["embargo_sessions"]
        for geometry in specification["folds"]["geometries"].values():
            assert geometry["applies_when_sessions_at_least"] == (
                geometry["initial_train_sessions"] + embargo
                + geometry["test_sessions"]
            )

    def test_reduced_geometry_cannot_declare_success(self):
        assert select_geometry(49)["can_declare_success"] is False
        assert select_geometry(80)["can_declare_success"] is True

    def test_every_feature_set_is_buildable_from_declared_columns(self):
        from research.modelling_unit import AGGREGATED_FEATURES

        derived = set(AGGREGATED_FEATURES) | {
            "prev_return", "prev_return_sign", "eem_lag1", "brent_lag1",
            "usdtry_lag1", "abnormal_tone", "abnormal_tone_domestic",
        }
        for name, spec in FEATURE_SETS.items():
            missing = set(spec["features"]) - derived
            assert not missing, f"{name} needs undeclared columns {missing}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_balanced_accuracy_penalises_always_up(self):
        actual = [1.0, 1.0, 1.0, -1.0]
        predicted = [1.0, 1.0, 1.0, 1.0]
        metrics = direction_metrics(actual, predicted)
        assert metrics["directional_accuracy"] == 0.75
        assert metrics["balanced_accuracy"] == 0.5

    def test_regression_metrics_report_reference_r2(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], reference=2.0)
        assert metrics["mae"] == 0.0
        assert metrics["r2_vs_reference"] == 1.0

    def test_cluster_bootstrap_resamples_clusters_not_rows(self):
        # Twenty rows but two clusters: the interval must reflect two draws.
        values = [1.0] * 10 + [0.0] * 10
        clusters = ["a"] * 10 + ["b"] * 10
        interval = cluster_bootstrap(values, clusters, resamples=200)
        assert interval["clusters"] == 2
        assert interval["lower"] == 0.0 and interval["upper"] == 1.0

    def test_bootstrap_is_deterministic(self):
        values = [0.1 * i for i in range(20)]
        clusters = [f"c{i % 5}" for i in range(20)]
        first = cluster_bootstrap(values, clusters, resamples=100)
        second = cluster_bootstrap(values, clusters, resamples=100)
        assert first == second

    def test_single_cluster_reports_no_interval(self):
        interval = cluster_bootstrap([1.0, 2.0], ["a", "a"], resamples=50)
        assert interval["lower"] is None and interval["upper"] is None
