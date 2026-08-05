"""The publish guard must refuse a stale database before it reaches the branch.

The failure this prevents is unrecoverable: the daily workflow force-pushes an
orphan commit, so overwriting a newer snapshot leaves no history to restore.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.guard_db_snapshot import (
    compare_snapshots,
    main,
    snapshot_freshness,
)


def _make_db(
    path: Path,
    *,
    headlines: int,
    scraped_at: str,
    published_at: str,
    price_date: str = "2026-07-31",
    run_started_at: str = "2026-07-31T09:00:00Z",
) -> Path:
    con = sqlite3.connect(str(path))
    try:
        con.executescript(
            """
            CREATE TABLE headlines (
                id INTEGER PRIMARY KEY, scraped_at TEXT, published_at TEXT
            );
            CREATE TABLE bist100_prices (date TEXT PRIMARY KEY);
            CREATE TABLE pipeline_runs (
                run_id INTEGER PRIMARY KEY, started_at TEXT
            );
            """
        )
        con.executemany(
            "INSERT INTO headlines VALUES (?,?,?)",
            [(index + 1, scraped_at, published_at) for index in range(headlines)],
        )
        con.execute("INSERT INTO bist100_prices VALUES (?)", (price_date,))
        con.execute("INSERT INTO pipeline_runs VALUES (1, ?)", (run_started_at,))
        con.commit()
    finally:
        con.close()
    return path


@pytest.fixture
def canonical(tmp_path) -> Path:
    return _make_db(
        tmp_path / "canonical.db",
        headlines=3465,
        scraped_at="2026-07-31T09:22:00Z",
        published_at="2026-07-31",
    )


def test_newer_candidate_is_allowed(tmp_path, canonical):
    candidate = _make_db(
        tmp_path / "newer.db",
        headlines=3600,
        scraped_at="2026-08-05T09:00:00Z",
        published_at="2026-08-05",
        price_date="2026-08-05",
        run_started_at="2026-08-05T09:00:00Z",
    )
    report = compare_snapshots(candidate, canonical)
    assert report["safe_to_publish"]
    assert not report["regressions"]


def test_identical_snapshot_is_allowed(tmp_path, canonical):
    same = _make_db(
        tmp_path / "same.db",
        headlines=3465,
        scraped_at="2026-07-31T09:22:00Z",
        published_at="2026-07-31",
    )
    assert compare_snapshots(same, canonical)["safe_to_publish"]


def test_stale_local_database_is_refused(tmp_path, canonical):
    """The exact Phase 0 hazard: the 2026-07-07 local copy over 2026-07-31."""

    stale = _make_db(
        tmp_path / "stale.db",
        headlines=1991,
        scraped_at="2026-07-07T09:00:00Z",
        published_at="2026-07-07",
        price_date="2026-07-07",
        run_started_at="2026-07-07T09:00:00Z",
    )
    report = compare_snapshots(stale, canonical)
    assert not report["safe_to_publish"]
    regressed = {item["marker"] for item in report["regressions"]}
    assert regressed == {
        "headline_count", "max_scraped_at", "max_published_at",
        "max_price_date", "max_run_started_at",
    }


def test_a_single_regressed_marker_is_enough_to_refuse(tmp_path, canonical):
    """Row count alone can look healthy while the corpus frontier went backwards."""

    partial = _make_db(
        tmp_path / "partial.db",
        headlines=4000,
        scraped_at="2026-08-05T09:00:00Z",
        published_at="2026-07-20",
        price_date="2026-08-05",
        run_started_at="2026-08-05T09:00:00Z",
    )
    report = compare_snapshots(partial, canonical)
    assert not report["safe_to_publish"]
    assert [item["marker"] for item in report["regressions"]] == ["max_published_at"]


def test_missing_reference_table_constrains_nothing(tmp_path, canonical):
    reference = tmp_path / "bare.db"
    con = sqlite3.connect(str(reference))
    con.executescript("CREATE TABLE headlines (id INTEGER PRIMARY KEY);")
    con.close()

    candidate = _make_db(
        tmp_path / "candidate.db",
        headlines=10,
        scraped_at="2026-07-01T00:00:00Z",
        published_at="2026-07-01",
    )
    report = compare_snapshots(candidate, reference)
    verdicts = {item["marker"]: item["verdict"] for item in report["comparisons"]}
    assert verdicts["max_price_date"] == "unconstrained"
    assert report["safe_to_publish"]


def test_freshness_markers_report_none_for_absent_tables(tmp_path):
    bare = tmp_path / "empty.db"
    sqlite3.connect(str(bare)).close()
    markers = snapshot_freshness(bare)
    assert set(markers) == {
        "headline_count", "max_scraped_at", "max_published_at",
        "max_price_date", "max_run_started_at",
    }
    assert all(value is None for value in markers.values())


def test_cli_exits_nonzero_on_regression(tmp_path, canonical, capsys):
    stale = _make_db(
        tmp_path / "stale.db",
        headlines=1991,
        scraped_at="2026-07-07T09:00:00Z",
        published_at="2026-07-07",
        price_date="2026-07-07",
        run_started_at="2026-07-07T09:00:00Z",
    )
    code = main([str(stale), "--reference", str(canonical)])
    assert code == 1
    assert "REFUSED" in capsys.readouterr().out


def test_cli_override_is_recorded_in_the_output(tmp_path, canonical, capsys):
    stale = _make_db(
        tmp_path / "stale.db",
        headlines=10,
        scraped_at="2026-01-01T00:00:00Z",
        published_at="2026-01-01",
        price_date="2026-01-01",
        run_started_at="2026-01-01T00:00:00Z",
    )
    code = main([
        str(stale), "--reference", str(canonical),
        "--allow-regression", "deliberate rebuild after corruption",
    ])
    assert code == 0
    assert "deliberate rebuild after corruption" in capsys.readouterr().out


def test_cli_reports_missing_candidate(tmp_path, canonical, capsys):
    code = main([str(tmp_path / "absent.db"), "--reference", str(canonical)])
    assert code == 2
