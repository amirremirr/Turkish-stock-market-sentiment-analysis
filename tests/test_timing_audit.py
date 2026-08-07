"""The timing convention, proven rather than assumed.

These tests exist because the convention was previously *implied* by three call
sites that disagreed, and the disagreement shipped: every post-close and weekend
return window was built one session late. The regression tests below pin the
corrected alignment to concrete dates, so a future refactor that reintroduces
the shift fails loudly instead of quietly producing a null.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.return_windows import (
    PRIMARY_WINDOW, REASON_INTRADAY_UNAVAILABLE, REASON_NO_PRIOR_SESSION_BAR,
    REASON_TIMING_CONFLICT, REASON_UNKNOWN_TIMING, WINDOW_PRIOR_CLOSE_TO_CLOSE,
    WINDOW_PRIOR_CLOSE_TO_OPEN, PriceSeries, build_return_windows,
    timing_eligibility,
)
from research.timing import (
    CONFLICT_GOVERNING_UNKNOWN, CONFLICT_MULTIPLE_SESSIONS,
    SIGNAL_DATE_SEMANTICS, derive_event_timing, expected_publication_session,
    expected_signal_date, first_reactable_at, previous_session,
)
from scripts.timing_audit import audit_rows, classify_semantics
from trading_calendar import assign_trading_session

# 2026-06-08 (Mon) .. 2026-06-12 (Fri) are consecutive trading sessions.
BARS = [
    {"date": "2026-06-05", "open": 100.0, "close": 101.0, "bar_status": "complete"},
    {"date": "2026-06-08", "open": 101.0, "close": 102.0, "bar_status": "complete"},
    {"date": "2026-06-09", "open": 102.0, "close": 103.0, "bar_status": "complete"},
    {"date": "2026-06-10", "open": 103.0, "close": 104.0, "bar_status": "complete"},
]


@pytest.fixture
def prices():
    return PriceSeries(BARS)


# ---------------------------------------------------------------------------
# What signal_date means
# ---------------------------------------------------------------------------
class TestSignalDateSemantics:
    """Hypothesis A (publication session) vs B (first reactable session)."""

    @pytest.mark.parametrize("timestamp,bucket,expected", [
        ("2026-06-09T08:00:00+03:00", "pre_open", "2026-06-09"),
        ("2026-06-09T12:00:00+03:00", "during_session", "2026-06-09"),
        # The discriminating cases: the assigned session is not the publication
        # session.
        ("2026-06-08T21:00:00+03:00", "post_close", "2026-06-09"),
        ("2026-06-06T11:00:00+03:00", "weekend_or_holiday", "2026-06-08"),
    ])
    def test_assignment_matches_first_reactable_session(
        self, timestamp, bucket, expected
    ):
        assignment = assign_trading_session(timestamp)
        assert assignment.timing_bucket == bucket
        assert assignment.signal_date == expected
        assert expected_signal_date(timestamp) == expected

    def test_hypothesis_a_is_refuted_by_post_close(self):
        """A post-close story is assigned to a session it was not published in."""

        timestamp = "2026-06-08T21:00:00+03:00"
        assert expected_publication_session(timestamp) == "2026-06-08"
        assert expected_signal_date(timestamp) == "2026-06-09"
        assert assign_trading_session(timestamp).signal_date != "2026-06-08"

    def test_unknown_time_waits_for_the_next_session(self):
        assignment = assign_trading_session(None, "2026-06-08")
        assert assignment.timing_bucket == "unknown"
        assert assignment.signal_date == "2026-06-09"

    def test_classifier_returns_hypothesis_b_on_production_shaped_rows(self):
        records = [
            {"id": 1, "timing_bucket": "pre_open", "signal_date": "2026-06-09",
             "published_at": "2026-06-09", "published_timestamp": "2026-06-09T08:00:00+03:00"},
            {"id": 2, "timing_bucket": "post_close", "signal_date": "2026-06-09",
             "published_at": "2026-06-08", "published_timestamp": "2026-06-08T21:00:00+03:00"},
            {"id": 3, "timing_bucket": "weekend_or_holiday", "signal_date": "2026-06-08",
             "published_at": "2026-06-06", "published_timestamp": "2026-06-06T11:00:00+03:00"},
            {"id": 4, "timing_bucket": "unknown", "signal_date": "2026-06-09",
             "published_at": "2026-06-08", "published_timestamp": None},
        ]
        verdict = classify_semantics(records)
        assert verdict["verdict"] == SIGNAL_DATE_SEMANTICS
        assert verdict["hypothesis_b_holds"] is True
        assert verdict["hypothesis_a_holds"] is False
        assert "post_close" in verdict["discriminating_buckets"]

    def test_classifier_would_report_hypothesis_a_if_rows_said_so(self):
        """The verdict is read off the data, not hard-coded to the answer."""

        records = [
            {"id": 1, "timing_bucket": "post_close", "signal_date": "2026-06-08",
             "published_at": "2026-06-08",
             "published_timestamp": "2026-06-08T21:00:00+03:00"},
        ]
        verdict = classify_semantics(records)
        assert verdict["hypothesis_b_holds"] is False
        assert verdict["verdict"] != SIGNAL_DATE_SEMANTICS
        assert verdict["agrees_with_declared"] is False


# ---------------------------------------------------------------------------
# The one-session shift
# ---------------------------------------------------------------------------
class TestReturnWindowAlignment:
    def test_pre_open_trades_the_session_it_was_published_before(self, prices):
        windows = {w.window_name: w for w in
                   build_return_windows("2026-06-09", "pre_open", prices)}
        primary = windows[PRIMARY_WINDOW]
        assert (primary.entry_date, primary.exit_date) == ("2026-06-09", "2026-06-09")
        assert primary.entry_price == 102.0 and primary.exit_price == 103.0
        assert primary.is_tradable is True

    def test_post_close_is_not_shifted_by_a_session(self, prices):
        """The v1 regression: entry must be the reactable open, not the next one.

        News published 2026-06-08 21:00 has signal_date 2026-06-09. v1 built
        close(2026-06-09) -> open(2026-06-10) and so measured the session after
        the one the news could move.
        """

        windows = {w.window_name: w for w in
                   build_return_windows("2026-06-09", "post_close", prices)}
        primary = windows[PRIMARY_WINDOW]
        assert primary.entry_date == "2026-06-09"
        assert primary.exit_date == "2026-06-09"
        assert primary.entry_date != "2026-06-10"
        assert primary.assumed_execution.startswith("2026-06-09T10:00")

        gap = windows[WINDOW_PRIOR_CLOSE_TO_OPEN]
        assert (gap.entry_date, gap.exit_date) == ("2026-06-08", "2026-06-09")
        assert gap.information_cutoff.startswith("2026-06-08T18:10")

    def test_post_close_and_pre_open_share_the_primary_window(self, prices):
        """Both execute at the same open on the same session, so both are poolable."""

        pre = build_return_windows("2026-06-09", "pre_open", prices)[0]
        post = next(
            w for w in build_return_windows("2026-06-09", "post_close", prices)
            if w.window_name == PRIMARY_WINDOW
        )
        assert pre.entry_date == post.entry_date
        assert pre.exit_date == post.exit_date
        assert pre.raw_return == post.raw_return

    def test_weekend_reacts_at_the_next_session_open(self, prices):
        windows = {w.window_name: w for w in
                   build_return_windows("2026-06-08", "weekend_or_holiday", prices)}
        primary = windows[PRIMARY_WINDOW]
        assert (primary.entry_date, primary.exit_date) == ("2026-06-08", "2026-06-08")
        gap = windows[WINDOW_PRIOR_CLOSE_TO_CLOSE]
        # 2026-06-05 is the Friday before; the weekend is spanned, not skipped.
        assert gap.entry_date == "2026-06-05"

    def test_gap_windows_are_never_tradable(self, prices):
        for bucket in ("pre_open", "post_close", "weekend_or_holiday"):
            windows = build_return_windows("2026-06-09", bucket, prices)
            for window in windows:
                if window.window_name == PRIMARY_WINDOW:
                    assert window.is_tradable is True
                else:
                    assert window.is_tradable is False
                    assert window.not_tradable_reason

    def test_missing_prior_bar_blocks_only_the_gap_windows(self):
        prices = PriceSeries([
            {"date": "2026-06-09", "open": 102.0, "close": 103.0,
             "bar_status": "complete"},
        ])
        windows = {w.window_name: w for w in
                   build_return_windows("2026-06-09", "post_close", prices)}
        assert windows[PRIMARY_WINDOW].is_available is True
        assert windows[WINDOW_PRIOR_CLOSE_TO_OPEN].unavailable_reason == (
            REASON_NO_PRIOR_SESSION_BAR
        )

    def test_provisional_bars_are_invisible(self):
        prices = PriceSeries([
            {"date": "2026-06-08", "open": 101.0, "close": 102.0,
             "bar_status": "complete"},
            {"date": "2026-06-09", "open": 102.0, "close": 103.0,
             "bar_status": "provisional"},
        ])
        windows = build_return_windows("2026-06-09", "pre_open", prices)
        assert all(not window.is_available for window in windows)

    def test_during_session_and_unknown_stay_blocked(self, prices):
        during = build_return_windows("2026-06-09", "during_session", prices)
        assert during[0].unavailable_reason == REASON_INTRADAY_UNAVAILABLE
        unknown = build_return_windows("2026-06-09", "unknown", prices)
        assert unknown[0].unavailable_reason == REASON_UNKNOWN_TIMING

    def test_previous_session_crosses_a_nine_day_holiday(self):
        # 2026-05-27..2026-05-30 are BIST holidays; 2026-05-26 is a half day.
        assert previous_session("2026-06-01") == "2026-05-26"


# ---------------------------------------------------------------------------
# Event-level timing
# ---------------------------------------------------------------------------
class TestEventTiming:
    def test_last_member_governs_not_the_earliest(self):
        """An event is not actionable before its last defining headline exists."""

        timing = derive_event_timing([
            {"headline_id": 1, "signal_date": "2026-06-09",
             "timing_bucket": "pre_open",
             "published_timestamp": "2026-06-09T08:00:00+03:00"},
            {"headline_id": 2, "signal_date": "2026-06-10",
             "timing_bucket": "post_close",
             "published_timestamp": "2026-06-09T21:00:00+03:00"},
        ])
        assert timing.first_reactable_session == "2026-06-10"
        assert timing.timing_bucket == "post_close"
        assert timing.governing_headline_id == 2

    def test_bucket_and_session_come_from_the_same_member(self):
        """The defect this rule replaces: bucket from one, session from another."""

        timing = derive_event_timing([
            {"headline_id": 1, "signal_date": "2026-06-09",
             "timing_bucket": "during_session"},
            {"headline_id": 2, "signal_date": "2026-06-11",
             "timing_bucket": "pre_open"},
        ])
        assert timing.first_reactable_session == "2026-06-11"
        # during_session is more restrictive but belongs to an earlier session,
        # so it must not be borrowed.
        assert timing.timing_bucket == "pre_open"
        assert timing.governing_headline_id == 2

    def test_most_restrictive_wins_within_one_session(self):
        timing = derive_event_timing([
            {"headline_id": 1, "signal_date": "2026-06-09",
             "timing_bucket": "pre_open"},
            {"headline_id": 2, "signal_date": "2026-06-09",
             "timing_bucket": "during_session"},
        ])
        assert timing.timing_bucket == "during_session"
        assert timing.governing_headline_id == 2
        assert timing.timing_conflict == 0

    def test_members_spanning_sessions_are_flagged(self):
        timing = derive_event_timing([
            {"headline_id": 1, "signal_date": "2026-06-09", "timing_bucket": "pre_open"},
            {"headline_id": 2, "signal_date": "2026-06-11", "timing_bucket": "pre_open"},
        ])
        assert timing.timing_conflict == 1
        assert CONFLICT_MULTIPLE_SESSIONS in timing.timing_conflict_reason
        assert timing.member_session_count == 2

    def test_governing_member_without_timing_is_flagged(self):
        timing = derive_event_timing([
            {"headline_id": 1, "signal_date": "2026-06-09", "timing_bucket": "pre_open"},
            {"headline_id": 2, "signal_date": "2026-06-09", "timing_bucket": "unknown"},
        ])
        assert timing.timing_conflict == 1
        assert CONFLICT_GOVERNING_UNKNOWN in timing.timing_conflict_reason

    def test_single_member_group_never_conflicts(self):
        timing = derive_event_timing([
            {"headline_id": 7, "signal_date": "2026-06-09",
             "timing_bucket": "post_close",
             "published_timestamp": "2026-06-08T20:00:00+03:00"},
        ])
        assert timing.timing_conflict == 0
        assert timing.event_information_cutoff == "2026-06-08T20:00:00+03:00"
        assert timing.first_reactable_at.startswith("2026-06-09T10:00")

    def test_cutoff_is_the_latest_publication_not_the_first(self):
        timing = derive_event_timing([
            {"headline_id": 1, "signal_date": "2026-06-09",
             "timing_bucket": "pre_open",
             "published_timestamp": "2026-06-09T07:00:00+03:00"},
            {"headline_id": 2, "signal_date": "2026-06-09",
             "timing_bucket": "pre_open",
             "published_timestamp": "2026-06-09T09:30:00+03:00"},
        ])
        assert timing.event_information_cutoff == "2026-06-09T09:30:00+03:00"

    def test_conflicted_events_are_blocked_from_primary_evaluation(self):
        assert timing_eligibility("pre_open", timing_conflict=True) == {
            "status": "blocked", "reason": REASON_TIMING_CONFLICT,
        }
        assert timing_eligibility("pre_open", timing_conflict=False)["status"] == (
            "eligible"
        )

    def test_first_reactable_at_is_the_opening_bell(self):
        assert first_reactable_at("2026-06-09", "post_close").startswith(
            "2026-06-09T10:00"
        )


# ---------------------------------------------------------------------------
# The audit table itself
# ---------------------------------------------------------------------------
class TestAuditTable:
    def test_audit_flags_a_shifted_window(self, prices, monkeypatch):
        """A deliberately shifted builder must produce a FAIL, not a PASS."""

        import scripts.timing_audit as audit

        def _shifted(session, bucket, series):
            from research.return_windows import build_return_windows as real
            if bucket in ("post_close", "weekend_or_holiday"):
                following = series.next_after(session)
                if following:
                    return real(str(following["date"]), bucket, series)
            return real(session, bucket, series)

        monkeypatch.setattr(audit, "build_return_windows", _shifted)
        rows = audit.audit_rows(
            [{"id": 1, "title": "t", "published_at": "2026-06-08",
              "published_hour": 21, "published_timestamp": None,
              "timing_bucket": "post_close", "signal_date": "2026-06-09"}],
            prices, per_bucket=5,
        )
        assert rows[0]["result"] == "FAIL"
        assert "expected" in rows[0]["reason"]

    def test_audit_passes_on_the_corrected_builder(self, prices):
        rows = audit_rows(
            [{"id": 1, "title": "t", "published_at": "2026-06-08",
              "published_hour": 21, "published_timestamp": None,
              "timing_bucket": "post_close", "signal_date": "2026-06-09"}],
            prices, per_bucket=5,
        )
        assert rows[0]["result"] == "PASS"
        assert rows[0]["generated_entry_date"] == "2026-06-09"
        assert rows[0]["previous_trading_session"] == "2026-06-08"

    def test_audit_runs_against_a_database(self, tmp_path):
        import database as db
        from scripts.timing_audit import run_audit

        path = str(tmp_path / "audit.db")
        db.init_db(db_path=path)
        with sqlite3.connect(path) as con:
            con.execute(
                """INSERT INTO headlines
                   (source, title, published_at, published_hour, timing_bucket,
                    signal_date, scraped_at)
                   VALUES ('s','t','2026-06-08',21,'post_close','2026-06-09','x')"""
            )
            con.execute(
                """INSERT INTO bist100_prices (date, open, close, bar_status)
                   VALUES ('2026-06-09',102.0,103.0,'complete')"""
            )
        result = run_audit(path, per_bucket=1)
        assert result["semantics"]["agrees_with_declared"] is True
        assert result["all_passed"] is True
