"""Tests for status_update_message and status_resumed_message — the two new
message-builder entries that the delay-detection gates use.
"""
from __future__ import annotations

import pytest

from src import message_builder


@pytest.fixture
def match():
    return {
        "id": "1001",
        "teams": ["RCB", "KKR"],
        "date_ist": "2026-05-13",
        "scheduled_ist": "19:30",
        "status": "scheduled",
        "toss_winner": None,
        "inn1": None,
        "inn2": None,
    }


# ---------- status_update_message ----------

def test_status_update_pre_toss_with_note(match):
    body = message_builder.status_update_message(
        match, phase="pre_toss", note="Wet outfield, inspection at 8:15 PM IST",
    )
    assert "RCB vs KKR" in body
    assert "Wet outfield, inspection at 8:15 PM IST" in body
    # Should not invent a delay reason when upstream provided one
    assert body.count("Match delayed") <= 1


def test_status_update_pre_toss_no_note(match):
    body = message_builder.status_update_message(match, phase="pre_toss", note=None)
    assert "RCB vs KKR" in body
    # Generic fallback copy when upstream is silent
    assert "delay" in body.lower()
    assert "toss" in body.lower()


def test_status_update_post_toss_pre_play(match):
    match["toss_winner"] = "RCB"
    match["toss_decision"] = "field"
    body = message_builder.status_update_message(
        match, phase="post_toss_pre_play",
        note="Inspection scheduled for 9:00 PM IST",
    )
    assert "RCB vs KKR" in body
    assert "Inspection scheduled for 9:00 PM IST" in body


def test_status_update_in_play(match):
    match["status"] = "live"
    match["inn1"] = {"runs": 67, "wkts": 2, "overs": 7.2, "raw": "67/2 (7.2 Ov)"}
    body = message_builder.status_update_message(
        match, phase="in_play", note="Rain stoppage at 7.2 overs",
    )
    assert "RCB vs KKR" in body
    assert "Rain stoppage at 7.2 overs" in body


# ---------- status_resumed_message ----------

def test_status_resumed_pre_toss(match):
    body = message_builder.status_resumed_message(
        match, phase="pre_toss", delay_minutes=47,
    )
    assert "RCB vs KKR" in body
    assert "47" in body
    assert "toss" in body.lower()


def test_status_resumed_post_toss_pre_play(match):
    match["toss_winner"] = "RCB"
    match["toss_decision"] = "field"
    body = message_builder.status_resumed_message(
        match, phase="post_toss_pre_play", delay_minutes=35,
    )
    assert "RCB vs KKR" in body
    assert "35" in body
    assert any(word in body.lower() for word in ("play", "start", "first ball"))


def test_status_resumed_in_play(match):
    match["status"] = "live"
    body = message_builder.status_resumed_message(
        match, phase="in_play", delay_minutes=22,
    )
    assert "RCB vs KKR" in body
    assert "22" in body
    assert any(word in body.lower() for word in ("resumed", "resume"))


def test_status_resumed_rounds_minutes_to_int(match):
    # Caller passes float — body should still show clean integer
    body = message_builder.status_resumed_message(
        match, phase="in_play", delay_minutes=22.7,
    )
    assert "22" in body or "23" in body
    assert "22.7" not in body
