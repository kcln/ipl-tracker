"""Integration tests for tracker._maybe_generate_delay_phases.

Covers B1.1 (pre-toss overdue), B1.2 (post-toss no overs), B2 (overs frozen),
dedup (transition + material-change), soft cap, auto-clear, and resumption.

These tests don't hit any network — they pass synthetic match dicts directly
to the function and inspect the resulting day_entry messages.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src import state, tracker


def _match(*, toss_winner=None, overs1=None, overs2=None, status="scheduled",
           note=None, scheduled_ist="19:30", date_ist="2026-05-13", match_id="1001"):
    inn1 = {"runs": 50, "wkts": 1, "overs": overs1, "raw": ""} if overs1 is not None else None
    inn2 = {"runs": 50, "wkts": 1, "overs": overs2, "raw": ""} if overs2 is not None else None
    return {
        "id": match_id,
        "teams": ["RCB", "KKR"],
        "date_ist": date_ist,
        "scheduled_ist": scheduled_ist,
        "status": status,
        "toss_winner": toss_winner,
        "toss_decision": "field" if toss_winner else None,
        "inn1": inn1,
        "inn2": inn2,
        "note": note,
    }


@pytest.fixture
def day_entry():
    s = {"days": {}}
    return state.day(s, "2026-05-13")


@pytest.fixture(autouse=True)
def _silence_archive():
    """Don't touch docs/index.html during tests."""
    with patch("src.tracker.html_archive.upsert_message"):
        yield


def _msg_types(day_entry):
    return [m["type"] for m in day_entry.get("messages", [])]


# Scheduled start is 19:30 IST = 07:00 PT (May 13)
SCHED_PT = datetime(2026, 5, 13, 7, 0, 0, tzinfo=state.PT)


# ---------- B1.1: pre-toss overdue ----------

def test_b1_pre_toss_no_fire_under_threshold(day_entry):
    match = _match()  # scheduled, no toss
    now = SCHED_PT + timedelta(minutes=5)
    n = tracker._maybe_generate_delay_phases(day_entry, "2026-05-13", [match], now)
    assert n == 0
    assert "status_update_1_1" not in _msg_types(day_entry)


def test_b1_pre_toss_fires_at_threshold(day_entry):
    match = _match(note="Wet outfield, inspection delayed")
    now = SCHED_PT + timedelta(minutes=12)
    n = tracker._maybe_generate_delay_phases(day_entry, "2026-05-13", [match], now)
    assert n == 1
    msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert len(msgs) == 1
    assert "Wet outfield" in msgs[0]["body"]


def test_b1_pre_toss_no_refire_on_same_note(day_entry):
    match = _match(note="Wet outfield, inspection delayed")
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13", [match], SCHED_PT + timedelta(minutes=12),
    )
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13", [match], SCHED_PT + timedelta(minutes=25),
    )
    status_msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert len(status_msgs) == 1  # no re-fire


def test_b1_pre_toss_refires_on_changed_note(day_entry):
    m1 = _match(note="Wet outfield")
    m2 = _match(note="Inspection at 8:15 PM IST")
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13", [m1], SCHED_PT + timedelta(minutes=12),
    )
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13", [m2], SCHED_PT + timedelta(minutes=25),
    )
    status_msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert len(status_msgs) == 2
    assert "Inspection at 8:15" in status_msgs[1]["body"]


def test_b1_pre_toss_emits_resumed_on_toss(day_entry):
    # Delay opens
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13", [_match(note="Wet outfield")],
        SCHED_PT + timedelta(minutes=12),
    )
    # Toss happens 50 minutes after scheduled start
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13", [_match(toss_winner="RCB", note=None)],
        SCHED_PT + timedelta(minutes=50),
    )
    resumed = [m for m in day_entry["messages"] if m["type"].startswith("status_resumed_")]
    assert len(resumed) == 1
    # 50 - 12 = 38 minutes
    assert "38" in resumed[0]["body"]
    # Resumption should use the phase the delay STARTED in (pre_toss),
    # not the phase the match is in now (post_toss_pre_play). Pre-toss copy
    # is "Toss is underway".
    assert "toss" in resumed[0]["body"].lower()
    assert "first ball bowled" not in resumed[0]["body"].lower()


# ---------- B1.2: post-toss, no overs ----------

def test_b1_post_toss_no_fire_under_30min(day_entry):
    # Simulate the toss message having been emitted at SCHED_PT
    toss_msg = state.add_or_update_message(day_entry, "toss_1", "toss body")
    toss_msg["generated_at"] = SCHED_PT.isoformat()
    match = _match(toss_winner="RCB", status="live")
    now = SCHED_PT + timedelta(minutes=20)
    n = tracker._maybe_generate_delay_phases(day_entry, "2026-05-13", [match], now)
    assert n == 0


def test_b1_post_toss_fires_at_30min(day_entry):
    toss_msg = state.add_or_update_message(day_entry, "toss_1", "toss body")
    toss_msg["generated_at"] = SCHED_PT.isoformat()
    match = _match(toss_winner="RCB", status="live", note="Inspection at 9:00 PM IST")
    now = SCHED_PT + timedelta(minutes=35)
    n = tracker._maybe_generate_delay_phases(day_entry, "2026-05-13", [match], now)
    assert n == 1
    msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert "Inspection at 9:00" in msgs[0]["body"]


def test_b1_post_toss_clears_when_overs_appear(day_entry):
    toss_msg = state.add_or_update_message(day_entry, "toss_1", "toss body")
    toss_msg["generated_at"] = SCHED_PT.isoformat()
    # Open the delay
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", note="Inspection")],
        SCHED_PT + timedelta(minutes=35),
    )
    # Play starts (overs > 0)
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=0.4)],
        SCHED_PT + timedelta(minutes=45),
    )
    resumed = [m for m in day_entry["messages"] if m["type"].startswith("status_resumed_")]
    assert len(resumed) == 1


# ---------- B2: overs frozen ----------

def test_b2_no_fire_when_overs_advance(day_entry):
    # Tick 1: 3.0 overs
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=3.0)],
        SCHED_PT + timedelta(minutes=20),
    )
    # Tick 2: 5.2 overs (advance)
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=5.2)],
        SCHED_PT + timedelta(minutes=30),
    )
    status_msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert len(status_msgs) == 0


def test_b2_fires_when_overs_frozen_for_10min(day_entry):
    # Tick 1: observe 7.2 overs
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=7.2)],
        SCHED_PT + timedelta(minutes=30),
    )
    # Tick 2: 12 min later, still 7.2
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=7.2, note="Rain stoppage")],
        SCHED_PT + timedelta(minutes=42),
    )
    status_msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert len(status_msgs) == 1
    assert "Rain stoppage" in status_msgs[0]["body"]


def test_b2_suppressed_during_innings_break(day_entry):
    # Tick 1: innings 1 in progress, overs at 19.5
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=19.5)],
        SCHED_PT + timedelta(minutes=90),
    )
    # Tick 2: innings 1 ends (overs1=20.0), chase not yet started
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=20.0)],
        SCHED_PT + timedelta(minutes=92),
    )
    # Tick 3: 15 min into the innings break — should NOT fire (natural break)
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=20.0, note="Innings break")],
        SCHED_PT + timedelta(minutes=107),
    )
    status_msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert len(status_msgs) == 0


def test_b2_fires_after_innings_break_grace_window(day_entry):
    # Tick 1: innings 1 ends at 20.0
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=20.0)],
        SCHED_PT + timedelta(minutes=92),
    )
    # Tick 2: 35 min later — chase still hasn't started → genuine delay
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=20.0, note="Long rain delay")],
        SCHED_PT + timedelta(minutes=127),
    )
    status_msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert len(status_msgs) == 1
    assert "Long rain delay" in status_msgs[0]["body"]


def test_b2_fires_normally_after_chase_starts(day_entry):
    # Innings 1 ends, then chase begins and freezes mid-over
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=20.0)],
        SCHED_PT + timedelta(minutes=92),
    )
    # Chase begins
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=20.0, overs2=2.3)],
        SCHED_PT + timedelta(minutes=110),
    )
    # 12 min freeze during the chase → normal B2 alert
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=20.0, overs2=2.3, note="Rain stoppage")],
        SCHED_PT + timedelta(minutes=122),
    )
    status_msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert len(status_msgs) == 1
    assert "Rain stoppage" in status_msgs[0]["body"]


def test_b2_clears_when_overs_resume(day_entry):
    # Establish frozen baseline
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=7.2)],
        SCHED_PT + timedelta(minutes=30),
    )
    # Delay opens
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=7.2, note="Rain stoppage")],
        SCHED_PT + timedelta(minutes=42),
    )
    # Play resumes — overs advance
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB", status="live", overs1=7.3)],
        SCHED_PT + timedelta(minutes=80),
    )
    resumed = [m for m in day_entry["messages"] if m["type"].startswith("status_resumed_")]
    assert len(resumed) == 1
    # 80 - 42 = 38 min
    assert "38" in resumed[0]["body"]


# ---------- Archive URL is appended to in-match messages ----------

def test_status_update_body_includes_archive_url(day_entry):
    match = _match(note="Wet outfield, inspection delayed")
    now = SCHED_PT + timedelta(minutes=12)
    tracker._maybe_generate_delay_phases(day_entry, "2026-05-13", [match], now)
    body = next(
        m["body"] for m in day_entry["messages"] if m["type"].startswith("status_update_")
    )
    assert f"Archive: {tracker.ARCHIVE_URL}" in body


def test_status_resumed_body_includes_archive_url(day_entry):
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(note="Wet outfield, inspection delayed")],
        SCHED_PT + timedelta(minutes=12),
    )
    # Toss happens — delay clears
    tracker._maybe_generate_delay_phases(
        day_entry, "2026-05-13",
        [_match(toss_winner="RCB")],
        SCHED_PT + timedelta(minutes=20),
    )
    body = next(
        m["body"] for m in day_entry["messages"] if m["type"].startswith("status_resumed_")
    )
    assert f"Archive: {tracker.ARCHIVE_URL}" in body


# ---------- Soft cap ----------

def test_soft_cap_prevents_runaway_refires(day_entry):
    # Fire 7 distinct notes — only 5 should land
    notes = [f"Update {i}" for i in range(7)]
    for i, n in enumerate(notes):
        tracker._maybe_generate_delay_phases(
            day_entry, "2026-05-13", [_match(note=n)],
            SCHED_PT + timedelta(minutes=12 + i * 5),
        )
    status_msgs = [m for m in day_entry["messages"] if m["type"].startswith("status_update_")]
    assert len(status_msgs) <= 5
