from datetime import date, datetime

import pytest

from trading_calendar import (
    ISTANBUL,
    assign_trading_session,
    is_trading_day,
    session_close,
    signal_date,
)


@pytest.mark.parametrize(
    ("timestamp", "bucket", "assigned_date"),
    [
        ("2026-06-10T09:59:59+03:00", "pre_open", "2026-06-10"),
        ("2026-06-10T10:00:00+03:00", "during_session", "2026-06-10"),
        ("2026-06-10T18:10:00+03:00", "during_session", "2026-06-10"),
        ("2026-06-10T18:10:01+03:00", "post_close", "2026-06-11"),
    ],
)
def test_regular_session_boundaries(timestamp, bucket, assigned_date):
    assignment = assign_trading_session(timestamp)

    assert assignment.timing_bucket == bucket
    assert assignment.signal_date == assigned_date


def test_aware_timestamp_is_normalized_before_date_and_bucket_assignment():
    # The source timestamp is still June 9 at UTC-08, but it is 10:30 on
    # June 10 in Istanbul and therefore belongs to the June 10 session.
    assignment = assign_trading_session("2026-06-09T23:30:00-08:00")

    assert assignment.published_at_istanbul == datetime(
        2026, 6, 10, 10, 30, tzinfo=ISTANBUL
    )
    assert assignment.timing_bucket == "during_session"
    assert assignment.signal_date == "2026-06-10"


def test_naive_timestamp_is_interpreted_as_istanbul_local_time():
    assignment = assign_trading_session("2026-06-10 10:30:00")

    assert assignment.published_at_istanbul == datetime(
        2026, 6, 10, 10, 30, tzinfo=ISTANBUL
    )
    assert assignment.signal_date == "2026-06-10"


def test_weekend_publication_maps_to_monday():
    assignment = assign_trading_session("2026-06-13T11:00:00+03:00")

    assert assignment.timing_bucket == "weekend_or_holiday"
    assert assignment.signal_date == "2026-06-15"


def test_full_day_holiday_is_skipped():
    assignment = assign_trading_session("2026-07-15T11:00:00+03:00")

    assert assignment.timing_bucket == "weekend_or_holiday"
    assert assignment.signal_date == "2026-07-16"


def test_ramazan_half_day_closes_at_1300_then_skips_consecutive_closures():
    at_close = assign_trading_session("2026-03-19T13:00:00+03:00")
    after_close = assign_trading_session("2026-03-19T13:00:01+03:00")

    assert session_close(date(2026, 3, 19)).isoformat() == "13:00:00"
    assert at_close.timing_bucket == "during_session"
    assert at_close.signal_date == "2026-03-19"
    assert after_close.timing_bucket == "post_close"
    assert after_close.signal_date == "2026-03-23"


def test_kurban_half_day_rolls_to_june_1_not_legacy_false_holiday():
    assignment = assign_trading_session("2026-05-26T13:00:01+03:00")

    assert assignment.timing_bucket == "post_close"
    assert assignment.signal_date == "2026-06-01"
    assert is_trading_day(date(2026, 6, 1))


def test_sunday_fixed_holiday_does_not_create_an_unpublished_monday_closure():
    assert is_trading_day(date(2026, 8, 31))


def test_republic_day_eve_half_day_rolls_past_october_29_closure():
    assignment = assign_trading_session("2026-10-28T13:00:01+03:00")

    assert assignment.timing_bucket == "post_close"
    assert assignment.signal_date == "2026-10-30"


@pytest.mark.parametrize(
    ("timestamp", "fallback_date", "assigned_date"),
    [
        (None, "2026-06-10", "2026-06-11"),
        ("2026-06-10", None, "2026-06-11"),
        ("not-a-timestamp", "2026-06-12", "2026-06-15"),
    ],
)
def test_unknown_time_conservatively_maps_to_next_session(
    timestamp, fallback_date, assigned_date
):
    assignment = assign_trading_session(timestamp, fallback_date)

    assert assignment.timing_bucket == "unknown"
    assert assignment.published_at_istanbul is None
    assert assignment.signal_date == assigned_date


def test_unknown_time_on_holiday_uses_non_trading_day_bucket():
    assignment = assign_trading_session(None, "2026-07-15")

    assert assignment.timing_bucket == "weekend_or_holiday"
    assert assignment.signal_date == "2026-07-16"


def test_missing_timestamp_and_date_is_rejected_deterministically():
    with pytest.raises(ValueError, match="publication date"):
        assign_trading_session(None)


def test_signal_date_compatibility_wrapper_uses_new_policy():
    assert signal_date("2026-06-10", 9) == "2026-06-10"
    assert signal_date("2026-06-10", 14) == "2026-06-10"
    assert signal_date("2026-06-10", 22) == "2026-06-11"
    assert signal_date("2026-06-10", None) == "2026-06-11"
