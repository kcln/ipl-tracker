"""Tests for the delay-tracking state helpers added to state.py.

The delay state lives per (day, match_id) so that the launchd job's repeated
ticks can detect transitions (none → active → none) and decide whether to
emit a status_update or status_resumed message.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src import state


@pytest.fixture
def now():
    # Fixed PT timestamp used by tests; aware so isoformat works as in prod
    return datetime(2026, 5, 13, 7, 0, 0, tzinfo=state.PT)


@pytest.fixture
def day_entry():
    s = {"days": {}}
    return state.day(s, "2026-05-13")


def test_get_delay_record_creates_default(day_entry):
    rec = state.get_delay_record(day_entry, "1001")
    assert rec == {
        "state": "none",
        "entered_at": None,
        "entered_phase": None,
        "last_overs_value": None,
        "last_overs_progress_at": None,
        "last_note_hash": None,
        "fired_count": 0,
    }


def test_set_delay_active_records_entered_phase(day_entry, now):
    rec = state.set_delay_active(
        day_entry, "1001", now=now, note_hash="abc", phase="pre_toss",
    )
    assert rec["entered_phase"] == "pre_toss"


def test_set_delay_active_keeps_entered_phase_on_re_fire(day_entry, now):
    state.set_delay_active(
        day_entry, "1001", now=now, note_hash="abc", phase="pre_toss",
    )
    rec = state.set_delay_active(
        day_entry, "1001", now=now, note_hash="def", phase="in_play",
    )
    # Same delay event — phase must not flip because new info arrived
    assert rec["entered_phase"] == "pre_toss"


def test_get_delay_record_returns_same_dict_on_repeat(day_entry):
    rec1 = state.get_delay_record(day_entry, "1001")
    rec1["state"] = "active"
    rec2 = state.get_delay_record(day_entry, "1001")
    assert rec2["state"] == "active"


def test_get_delay_record_isolates_per_match(day_entry):
    state.get_delay_record(day_entry, "1001")["state"] = "active"
    other = state.get_delay_record(day_entry, "1002")
    assert other["state"] == "none"


def test_set_delay_active_marks_transition(day_entry, now):
    rec = state.set_delay_active(day_entry, "1001", now=now, note_hash="abc")
    assert rec["state"] == "active"
    assert rec["entered_at"] == now.isoformat()
    assert rec["last_note_hash"] == "abc"
    assert rec["fired_count"] == 1


def test_set_delay_active_re_fires_on_note_change(day_entry, now):
    state.set_delay_active(day_entry, "1001", now=now, note_hash="abc")
    later = now + timedelta(minutes=20)
    rec = state.set_delay_active(day_entry, "1001", now=later, note_hash="def")
    assert rec["fired_count"] == 2
    assert rec["last_note_hash"] == "def"
    # entered_at is NOT updated on re-fire — same delay event, new info only
    assert rec["entered_at"] == now.isoformat()


def test_set_delay_active_does_not_re_fire_on_same_note(day_entry, now):
    state.set_delay_active(day_entry, "1001", now=now, note_hash="abc")
    later = now + timedelta(minutes=20)
    rec = state.set_delay_active(day_entry, "1001", now=later, note_hash="abc")
    assert rec["fired_count"] == 1  # no re-fire


def test_set_delay_cleared_resets_state(day_entry, now):
    state.set_delay_active(day_entry, "1001", now=now, note_hash="abc")
    later = now + timedelta(minutes=40)
    rec = state.set_delay_cleared(day_entry, "1001", now=later)
    assert rec["state"] == "none"
    # entered_at retained so resumption message can compute duration
    assert rec["entered_at"] == now.isoformat()


def test_set_delay_cleared_is_idempotent_when_already_clear(day_entry, now):
    rec = state.set_delay_cleared(day_entry, "1001", now=now)
    assert rec["state"] == "none"
    assert rec["fired_count"] == 0


def test_record_overs_progress_initial(day_entry, now):
    progressed = state.record_overs_progress(day_entry, "1001", overs=0.0, now=now)
    assert progressed is True  # first observation counts as progress
    rec = state.get_delay_record(day_entry, "1001")
    assert rec["last_overs_value"] == 0.0
    assert rec["last_overs_progress_at"] == now.isoformat()


def test_record_overs_progress_advance(day_entry, now):
    state.record_overs_progress(day_entry, "1001", overs=2.0, now=now)
    later = now + timedelta(minutes=5)
    progressed = state.record_overs_progress(day_entry, "1001", overs=2.3, now=later)
    assert progressed is True
    rec = state.get_delay_record(day_entry, "1001")
    assert rec["last_overs_value"] == 2.3
    assert rec["last_overs_progress_at"] == later.isoformat()


def test_record_overs_progress_no_change(day_entry, now):
    state.record_overs_progress(day_entry, "1001", overs=3.0, now=now)
    later = now + timedelta(minutes=12)
    progressed = state.record_overs_progress(day_entry, "1001", overs=3.0, now=later)
    assert progressed is False
    rec = state.get_delay_record(day_entry, "1001")
    # last_overs_progress_at not updated when no progress
    assert rec["last_overs_progress_at"] == now.isoformat()


def test_seconds_since_overs_progress(day_entry, now):
    state.record_overs_progress(day_entry, "1001", overs=5.0, now=now)
    later = now + timedelta(minutes=12)
    secs = state.seconds_since_overs_progress(day_entry, "1001", now=later)
    assert secs == 12 * 60


def test_seconds_since_overs_progress_returns_none_when_unobserved(day_entry, now):
    secs = state.seconds_since_overs_progress(day_entry, "1001", now=now)
    assert secs is None
