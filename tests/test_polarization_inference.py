import builtins
import sqlite3

import pandas as pd
import pytest

from analysis.polarization import inference


PRO = ("pro_a", "pro_b")
OPP = ("opp_a", "opp_b")


def _analyze(rows, **kwargs):
    return inference.analyze_polarization(
        pd.DataFrame(rows),
        pro_government_sources=PRO,
        opposition_sources=OPP,
        bootstrap_repetitions=400,
        bootstrap_seed=17,
        **kwargs,
    )


def test_console_output_escapes_only_characters_unsupported_by_active_encoding():
    text = "kayıp ölçüm"

    assert inference._console_safe(text, "utf-8") == text
    cp1252_safe = inference._console_safe(text, "cp1252")
    assert cp1252_safe == "kay\\u0131p ölçüm"
    cp1252_safe.encode("cp1252", errors="strict")


def test_exact_descriptives_effect_size_and_cluster_bootstrap_are_deterministic():
    rows = [
        {"source": "pro_a", "date": "2026-01-01", "sentiment": 1.0, "category": "macro", "title": "a"},
        {"source": "opp_a", "date": "2026-01-01", "sentiment": 0.0, "category": "macro", "title": "b"},
        {"source": "pro_a", "date": "2026-01-02", "sentiment": 3.0, "category": "macro", "title": "c"},
        {"source": "opp_a", "date": "2026-01-02", "sentiment": 0.0, "category": "macro", "title": "d"},
    ]

    first = _analyze(rows)
    second = _analyze(rows)

    assert first["mean_difference"]["estimate"] == pytest.approx(2.0)
    assert first["mean_difference"]["standardized_effect_size"] == pytest.approx(2.0)
    bootstrap = first["date_cluster_bootstrap"]
    assert bootstrap == second["date_cluster_bootstrap"]
    assert bootstrap["status"] == "ok"
    assert bootstrap["cluster_count"] == 2
    assert bootstrap["repetitions_completed"] == 400
    assert bootstrap["lower"] == pytest.approx(1.0)
    assert bootstrap["upper"] == pytest.approx(3.0)


def test_canonical_events_hold_event_fixed_and_selection_remains_separate():
    rows = []
    for index, (pro_score, opposition_score) in enumerate(
        [(0.8, 0.2), (0.4, -0.2), (0.1, -0.1), (0.6, 0.0)], start=1
    ):
        event = f"event-{index}"
        day = f"2026-02-{index:02d}"
        rows.extend(
            [
                {
                    "source": "pro_a",
                    "date": day,
                    "sentiment": pro_score,
                    "category": "macro",
                    "title": f"pro story {index}",
                    "canonical_event_id": event,
                },
                {
                    "source": "opp_a",
                    "date": day,
                    "sentiment": opposition_score,
                    "category": "macro",
                    "title": f"opp story {index}",
                    "canonical_event_id": event,
                },
            ]
        )
    rows.extend(
        [
            {
                "source": "pro_b",
                "date": "2026-02-05",
                "sentiment": 0.9,
                "category": "macro",
                "title": "pro-only story",
                "canonical_event_id": "pro-only",
            },
            {
                "source": "opp_b",
                "date": "2026-02-06",
                "sentiment": -0.9,
                "category": "macro",
                "title": "opposition-only story",
                "canonical_event_id": "opp-only",
            },
        ]
    )

    report = _analyze(rows)

    assert report["matching_audit"]["method"] == "canonical_event_id"
    assert report["matching_audit"]["verified_shared_events"] == 4
    assert report["framing"]["unit"] == "explicit repeated canonical event"
    assert report["framing"]["event_or_pair_count"] == 4
    assert report["framing"]["mean_gap"] == pytest.approx(0.5)
    assert report["selection"]["verified_shared_event_count"] == 4
    assert report["selection"]["event_coverage"]["shared_event_count"] == 4
    assert report["selection"]["event_coverage"]["pro_government_only_event_count"] == 1
    assert report["selection"]["event_coverage"]["opposition_only_event_count"] == 1
    coverage = {row["camp"]: row for row in report["selection"]["story_coverage_by_camp"]}
    assert coverage["pro_government"]["matched_headline_count"] == 4
    assert coverage["pro_government"]["unmatched_headline_count"] == 1
    assert coverage["opposition"]["matched_headline_count"] == 4
    assert coverage["opposition"]["unmatched_headline_count"] == 1

    regression = report["regression"]
    assert regression["status"] == "ok"
    event_sensitivity = regression["cluster_robust_sensitivities"]["event"]
    assert event_sensitivity["cluster_count"] == 6
    assert event_sensitivity["status"] == "ok_with_few_clusters"


def test_bridge_ids_are_not_treated_as_shared_events_and_fallback_has_no_reuse():
    rows = [
        {
            "headline_id": 1,
            "source": "pro_a",
            "date": "2026-03-01",
            "sentiment": 0.8,
            "category": "rates",
            "title": "Merkez Bankasi politika karari piyasayi etkiledi",
            "bridge_event_id": 101,
        },
        {
            "headline_id": 2,
            "source": "pro_b",
            "date": "2026-03-01",
            "sentiment": 0.4,
            "category": "rates",
            "title": "Merkez Bankasi politika karari sonrasi degerlendirme",
            "bridge_event_id": 102,
        },
        {
            "headline_id": 3,
            "source": "opp_a",
            "date": "2026-03-01",
            "sentiment": -0.3,
            "category": "rates",
            "title": "Merkez Bankasi politika karari tartisiliyor",
            "bridge_event_id": 103,
        },
        {
            "headline_id": 4,
            "source": "pro_a",
            "date": "2026-03-02",
            "sentiment": 0.2,
            "category": "rates",
            "title": "Petrol uretim kesintisi fiyatlari yukseltti",
            "bridge_event_id": 104,
        },
        {
            "headline_id": 5,
            "source": "opp_b",
            "date": "2026-03-02",
            "sentiment": -0.2,
            "category": "rates",
            "title": "Petrol uretim kesintisi fiyatlari baskiladi",
            "bridge_event_id": 105,
        },
    ]

    report = _analyze(rows)

    audit = report["matching_audit"]
    assert audit["method"] == "lexical_date_fallback"
    assert audit["one_to_one_no_reuse"] is True
    assert audit["verified_shared_events"] == 0
    assert len(audit["details"]) == 2
    assert len({pair["pro_row_key"] for pair in audit["details"]}) == 2
    assert len({pair["opposition_row_key"] for pair in audit["details"]}) == 2
    assert report["framing"]["status"] == "sensitivity_only"
    assert report["regression"]["cluster_robust_sensitivities"]["event"]["status"] == "skipped"
    assert "1:1 headline-event bridge" in (
        report["regression"]["cluster_robust_sensitivities"]["event"]["diagnostic"]
    )


def test_cluster_and_rank_diagnostics_are_explicit():
    same_date_rows = [
        {"source": "pro_a", "date": "2026-04-01", "sentiment": score, "category": "macro", "title": str(i)}
        for i, score in enumerate((0.1, 0.2, 0.3))
    ] + [
        {"source": "opp_a", "date": "2026-04-01", "sentiment": score, "category": "macro", "title": str(i)}
        for i, score in enumerate((-0.1, -0.2, -0.3))
    ]
    report = _analyze(same_date_rows)
    date_sensitivity = report["regression"]["cluster_robust_sensitivities"]["date"]
    assert date_sensitivity["status"] == "skipped"
    assert date_sensitivity["cluster_count"] == 1
    assert "at least two clusters" in date_sensitivity["diagnostic"]
    assert report["date_cluster_bootstrap"]["status"] == "inadequate"

    saturated = _analyze(
        [
            {"source": "pro_a", "date": "2026-04-01", "sentiment": 0.2, "category": "a", "title": "a"},
            {"source": "opp_a", "date": "2026-04-01", "sentiment": -0.2, "category": "a", "title": "b"},
        ]
    )
    assert saturated["regression"]["status"] == "inadequate_residual_degrees_of_freedom"
    assert saturated["regression"]["residual_degrees_of_freedom"] == 0


def test_missing_statsmodels_is_reported_without_losing_descriptives(monkeypatch):
    real_import = builtins.__import__

    def reject_statsmodels(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("synthetic missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_statsmodels)
    report = _analyze(
        [
            {"source": "pro_a", "date": "2026-05-01", "sentiment": 0.5, "category": "a", "title": "a"},
            {"source": "opp_a", "date": "2026-05-01", "sentiment": -0.5, "category": "a", "title": "b"},
            {"source": "pro_a", "date": "2026-05-02", "sentiment": 0.4, "category": "a", "title": "c"},
            {"source": "opp_a", "date": "2026-05-02", "sentiment": -0.4, "category": "a", "title": "d"},
        ]
    )
    assert report["mean_difference"]["estimate"] == pytest.approx(0.9)
    assert report["regression"]["status"] == "unavailable"
    assert "statsmodels is not installed" in report["regression"]["diagnostics"][0]


def test_database_loader_exposes_bridge_id_but_not_canonical_event(tmp_path):
    path = tmp_path / "polarization.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE headlines (
            id INTEGER PRIMARY KEY, source TEXT, title TEXT, published_at TEXT,
            category TEXT, sentiment_score REAL, processing_status TEXT
        );
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY, headline_id INTEGER UNIQUE
        );
        CREATE TABLE headline_exclusions (
            headline_id INTEGER, restored_at TEXT
        );
        INSERT INTO headlines VALUES
            (1, 'pro_a', 'one', '2026-06-01', 'macro', 0.2, 'scored'),
            (2, 'opp_a', 'two', '2026-06-01', 'macro', -0.2, 'scored'),
            (3, 'opp_a', 'excluded', '2026-06-01', 'macro', -0.9, 'scored');
        INSERT INTO events VALUES (11, 1), (12, 2), (13, 3);
        INSERT INTO headline_exclusions VALUES (3, NULL);
        """
    )
    connection.commit()
    connection.close()

    loaded = inference.load_headlines(path)

    assert list(loaded["headline_id"]) == [1, 2]
    assert list(loaded["bridge_event_id"]) == [11, 12]
    assert "canonical_event_id" not in loaded
    report = _analyze(loaded)
    assert report["matching_audit"]["method"] == "lexical_date_fallback"


def test_database_loader_preserves_cross_source_shared_url_observations(tmp_path):
    path = tmp_path / "polarization-raw.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE headlines (
            id INTEGER PRIMARY KEY, source TEXT, title TEXT, published_at TEXT,
            category TEXT, sentiment_score REAL, processing_status TEXT
        );
        CREATE TABLE raw_headline_observations (
            observation_id INTEGER PRIMARY KEY, headline_id INTEGER,
            source TEXT, title TEXT, published_at TEXT, published_timestamp TEXT
        );
        CREATE TABLE headline_exclusions (headline_id INTEGER, restored_at TEXT);
        INSERT INTO headlines VALUES
            (1, 'pro_a', 'canonical shared story', '2026-06-01', 'macro', 0.4, 'scored'),
            (2, 'opp_b', 'legacy canonical story', '2026-06-02', 'macro', -0.2, 'scored');
        INSERT INTO raw_headline_observations VALUES
            (10, 1, 'pro_a', 'pro rendering', '2026-06-01', NULL),
            (11, 1, 'opp_a', 'opposition rendering', '2026-06-01', NULL);
        """
    )
    connection.commit()
    connection.close()

    loaded = inference.load_headlines(path)

    assert list(loaded["source"]) == ["pro_a", "opp_a", "opp_b"]
    assert list(loaded["headline_id"]) == [1, 1, 2]
    assert loaded.loc[loaded["headline_id"] == 1, "sentiment"].tolist() == [0.4, 0.4]
    report = _analyze(loaded)
    counts = {row["camp"]: row["count"] for row in report["raw_descriptives"]["by_camp"]}
    assert counts == {"opposition": 2, "pro_government": 1}
