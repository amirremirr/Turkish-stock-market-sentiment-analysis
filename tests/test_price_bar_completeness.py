"""Daily-bar completeness contracts.

The production fault these prevent: the scheduled run fires at 06:30 UTC, hours
before the 18:10 Istanbul close, so a same-day price fetch returns an intraday
snapshot. Stored without qualification it becomes that session's "close", and
every return into or out of that session is wrong with nothing to reveal it.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import database as db
from price_bars import (
    REVIEW_BEFORE_SETTLEMENT,
    REVIEW_MISSING_VOLUME_FULL,
    REVIEW_NON_TRADING_DAY,
    REVIEW_UNVERIFIED,
    REVIEW_ZERO_VOLUME_FULL,
    REVIEW_ZERO_VOLUME_HALF,
    STATUS_COMPLETE,
    STATUS_CORRECTED,
    STATUS_PROVIDER_INVALID,
    STATUS_PROVISIONAL,
    classify_price_bar,
    may_replace,
    session_type,
    settlement_time,
)

# 2026-07-30 Thu full session, 2026-07-31 Fri full session,
# 2026-08-01/02 weekend, 2026-07-15 holiday, 2026-05-26 half day (13:00 close).
FULL_DAY = "2026-07-30"
HALF_DAY = "2026-05-26"
HOLIDAY = "2026-07-15"
SATURDAY = "2026-08-01"


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume", "daily_return"],
    )


@pytest.fixture
def price_db(tmp_path):
    path = str(tmp_path / "prices.db")
    db.init_db(path)
    return path


# -- Scenario: run during market hours ----------------------------------------

def test_run_during_market_hours_yields_a_provisional_bar():
    """The exact production fault: 06:30 UTC is 09:30 Istanbul, mid-session."""

    result = classify_price_bar(
        FULL_DAY, volume=0.0, observed_at=f"{FULL_DAY}T06:30:00Z"
    )
    assert result.status == STATUS_PROVISIONAL
    assert result.review_reason == REVIEW_BEFORE_SETTLEMENT
    assert not result.is_analysable


def test_a_bar_observed_at_the_closing_bell_is_still_provisional():
    """The provider needs a moment after the bell to publish the settled bar."""

    result = classify_price_bar(
        FULL_DAY, volume=5e9, observed_at=f"{FULL_DAY}T18:10:00+03:00"
    )
    assert result.status == STATUS_PROVISIONAL


# -- Scenario: run after market close -----------------------------------------

def test_run_after_close_plus_delay_yields_a_complete_bar():
    result = classify_price_bar(
        FULL_DAY, volume=7.9e9, observed_at=f"{FULL_DAY}T18:41:00+03:00"
    )
    assert result.status == STATUS_COMPLETE
    assert result.review_reason is None
    assert result.is_analysable


def test_next_morning_observation_is_complete():
    result = classify_price_bar(
        FULL_DAY, volume=7.9e9, observed_at="2026-07-31T06:30:00Z"
    )
    assert result.status == STATUS_COMPLETE


# -- Scenario: weekend and holiday --------------------------------------------

@pytest.mark.parametrize("day", [SATURDAY, HOLIDAY])
def test_bar_on_a_non_trading_day_is_provider_invalid(day):
    result = classify_price_bar(
        day, volume=1e9, observed_at=f"{day}T23:00:00+03:00"
    )
    assert result.status == STATUS_PROVIDER_INVALID
    assert result.review_reason == REVIEW_NON_TRADING_DAY
    assert result.session_type == "closed"
    assert result.settles_at is None
    assert not result.is_analysable


# -- Scenario: half-day close --------------------------------------------------

def test_half_day_settles_from_the_official_early_close():
    assert session_type(HALF_DAY) == "half"
    settles = settlement_time(HALF_DAY)
    assert settles.hour == 13 and settles.minute == 30

    # 14:00 is after a half-day close but well before a regular one.
    after_early_close = classify_price_bar(
        HALF_DAY, volume=3.4e9, observed_at=f"{HALF_DAY}T14:00:00+03:00"
    )
    assert after_early_close.status == STATUS_COMPLETE

    during = classify_price_bar(
        HALF_DAY, volume=1e9, observed_at=f"{HALF_DAY}T11:00:00+03:00"
    )
    assert during.status == STATUS_PROVISIONAL


def test_half_day_zero_volume_is_flagged_under_its_own_reason():
    """Half-days trade thinly; the flag stays distinguishable for triage."""

    result = classify_price_bar(
        HALF_DAY, volume=0.0, observed_at=f"{HALF_DAY}T14:00:00+03:00"
    )
    assert result.status == STATUS_COMPLETE
    assert result.review_reason == REVIEW_ZERO_VOLUME_HALF


# -- Scenario: zero-volume historical row flagged ------------------------------

def test_zero_volume_on_a_full_session_is_flagged_for_review():
    result = classify_price_bar(
        FULL_DAY, volume=0.0, observed_at="2026-08-05T12:00:00+03:00"
    )
    assert result.status == STATUS_COMPLETE, "a flag is not a rejection"
    assert result.review_reason == REVIEW_ZERO_VOLUME_FULL
    assert result.needs_review


def test_missing_volume_is_flagged():
    result = classify_price_bar(
        FULL_DAY, volume=None, observed_at="2026-08-05T12:00:00+03:00"
    )
    assert result.review_reason == REVIEW_MISSING_VOLUME_FULL


def test_unverifiable_observation_time_is_treated_as_unfinished():
    result = classify_price_bar(FULL_DAY, volume=5e9, observed_at=None)
    assert result.status == STATUS_PROVISIONAL
    assert result.review_reason == REVIEW_UNVERIFIED


# -- Scenario: provisional row later replaced by a complete row ----------------

def test_complete_bar_replaces_a_provisional_bar(price_db):
    db.upsert_prices(
        _frame([(FULL_DAY, 100.0, 101.0, 99.0, 99.5, 0.0, -0.5)]),
        db_path=price_db, observed_at=f"{FULL_DAY}T06:30:00Z",
    )
    stored = db.get_prices(db_path=price_db, complete_only=False)
    assert stored.loc[0, "bar_status"] == STATUS_PROVISIONAL
    assert db.get_prices(db_path=price_db).empty, "provisional is withheld"

    counts = db.upsert_prices(
        _frame([(FULL_DAY, 100.0, 103.0, 99.0, 102.5, 7.9e9, 2.5)]),
        db_path=price_db, observed_at="2026-07-31T06:30:00Z",
        mark_corrected=True,
    )
    assert counts["corrected"] == 1

    settled = db.get_prices(db_path=price_db)
    assert len(settled) == 1
    assert settled.loc[0, "close"] == 102.5
    assert settled.loc[0, "bar_status"] == STATUS_CORRECTED


def test_a_provisional_refetch_never_demotes_a_settled_bar(price_db):
    db.upsert_prices(
        _frame([(FULL_DAY, 100.0, 103.0, 99.0, 102.5, 7.9e9, 2.5)]),
        db_path=price_db, observed_at="2026-07-31T06:30:00Z",
    )
    counts = db.upsert_prices(
        _frame([(FULL_DAY, 100.0, 101.0, 99.0, 99.5, 0.0, -0.5)]),
        db_path=price_db, observed_at=f"{FULL_DAY}T06:30:00Z",
    )
    assert counts["skipped_would_demote"] == 1

    stored = db.get_prices(db_path=price_db, complete_only=False)
    assert stored.loc[0, "close"] == 102.5, "settled close must survive"
    assert stored.loc[0, "bar_status"] == STATUS_COMPLETE


@pytest.mark.parametrize(
    "existing, incoming, allowed",
    [
        (None, STATUS_PROVISIONAL, True),
        (STATUS_PROVISIONAL, STATUS_COMPLETE, True),
        (STATUS_PROVISIONAL, STATUS_CORRECTED, True),
        (STATUS_PROVIDER_INVALID, STATUS_COMPLETE, True),
        (STATUS_COMPLETE, STATUS_COMPLETE, True),
        (STATUS_COMPLETE, STATUS_PROVISIONAL, False),
        (STATUS_CORRECTED, STATUS_PROVISIONAL, False),
        (STATUS_COMPLETE, STATUS_PROVIDER_INVALID, False),
    ],
)
def test_completeness_only_moves_forward(existing, incoming, allowed):
    assert may_replace(existing, incoming) is allowed


# -- Backfill and analysis defaults --------------------------------------------

def test_backfill_resolves_history_from_recorded_run_times(price_db):
    """A run that started after a session settled has already refreshed it."""

    with db._conn(price_db) as con:
        con.executemany(
            "INSERT INTO bist100_prices (date, open, high, low, close, volume,"
            " daily_return) VALUES (?,?,?,?,?,?,?)",
            [
                ("2026-07-29", 1.0, 1.0, 1.0, 100.0, 6.9e9, 0.0),
                ("2026-07-30", 1.0, 1.0, 1.0, 101.0, 7.9e9, 1.0),
                ("2026-07-31", 1.0, 1.0, 1.0, 99.0, 0.0, -2.0),
            ],
        )
        con.execute(
            "INSERT INTO pipeline_runs (started_at, status) VALUES (?, 'ok')",
            ("2026-07-31T09:22:26Z",),
        )

    counts = db.backfill_price_bar_status(db_path=price_db)
    assert counts["classified"] == 3

    stored = db.get_prices(db_path=price_db, complete_only=False).set_index("date")
    # The run started 12:22 Istanbul on 07-31: after 07-30 settled, before 07-31.
    assert stored.loc["2026-07-29", "bar_status"] == STATUS_COMPLETE
    assert stored.loc["2026-07-30", "bar_status"] == STATUS_COMPLETE
    assert stored.loc["2026-07-31", "bar_status"] == STATUS_PROVISIONAL

    assert len(db.get_prices(db_path=price_db)) == 2


def test_backfill_without_run_history_withholds_rather_than_trusts(price_db):
    with db._conn(price_db) as con:
        con.execute(
            "INSERT INTO bist100_prices (date, close, volume) VALUES (?,?,?)",
            ("2026-07-30", 101.0, 7.9e9),
        )
    db.backfill_price_bar_status(db_path=price_db)
    stored = db.get_prices(db_path=price_db, complete_only=False)
    assert stored.loc[0, "bar_status"] == STATUS_PROVISIONAL
    assert db.get_prices(db_path=price_db).empty


def test_get_prices_withholds_unclassified_rows(price_db):
    with db._conn(price_db) as con:
        con.execute(
            "INSERT INTO bist100_prices (date, close) VALUES ('2026-07-30', 101.0)"
        )
    assert db.get_prices(db_path=price_db).empty
    assert len(db.get_prices(db_path=price_db, complete_only=False)) == 1


def test_review_listing_surfaces_flagged_and_withheld_bars(price_db):
    db.upsert_prices(
        _frame([
            (FULL_DAY, 1.0, 1.0, 1.0, 101.0, 0.0, 1.0),
            (SATURDAY, 1.0, 1.0, 1.0, 102.0, 1e9, 1.0),
        ]),
        db_path=price_db, observed_at="2026-08-05T12:00:00+03:00",
    )
    flagged = {row["date"]: row for row in db.list_price_bars_for_review(price_db)}
    assert flagged[FULL_DAY]["bar_review_reason"] == REVIEW_ZERO_VOLUME_FULL
    assert flagged[SATURDAY]["bar_status"] == STATUS_PROVIDER_INVALID


# -- Return recalculation after correction -------------------------------------

def test_returns_recompute_on_the_complete_series_after_a_correction(price_db):
    """A corrected close must flow into the neighbouring session's return."""

    db.upsert_prices(
        _frame([
            ("2026-07-29", 1.0, 1.0, 1.0, 100.0, 6.9e9, None),
            ("2026-07-30", 1.0, 1.0, 1.0, 200.0, 7.9e9, 100.0),
        ]),
        db_path=price_db, observed_at="2026-07-31T06:30:00Z",
    )
    before = db.get_prices(db_path=price_db)
    assert before["close"].tolist() == [100.0, 200.0]

    # 07-30's close was an intraday snapshot; the settled close was 110.
    db.upsert_prices(
        _frame([("2026-07-30", 1.0, 1.0, 1.0, 110.0, 7.9e9, 10.0)]),
        db_path=price_db, observed_at="2026-07-31T06:30:00Z",
        mark_corrected=True,
    )
    after = db.get_prices(db_path=price_db)
    assert after["close"].tolist() == [100.0, 110.0]

    recomputed = after["close"].pct_change().mul(100).tolist()
    assert recomputed[1] == pytest.approx(10.0)
    assert after.loc[1, "daily_return"] == pytest.approx(10.0)
