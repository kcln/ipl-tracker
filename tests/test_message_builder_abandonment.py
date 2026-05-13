"""Tests for post_match_message handling abandonment / no-result outcomes."""
from __future__ import annotations

import pytest

from src import message_builder


@pytest.fixture
def standings():
    return [
        {"team": "RCB", "played": 12, "won": 9, "lost": 3, "points": 18, "nrr": 0.81},
        {"team": "KKR", "played": 12, "won": 5, "lost": 7, "points": 10, "nrr": -0.20},
        {"team": "GT",  "played": 12, "won": 8, "lost": 4, "points": 16, "nrr": 0.60},
        {"team": "SRH", "played": 12, "won": 7, "lost": 5, "points": 14, "nrr": 0.30},
        {"team": "PBKS","played": 12, "won": 6, "lost": 6, "points": 12, "nrr": 0.10},
    ]


def test_post_match_abandoned_with_note(standings):
    match = {
        "id": "1001",
        "teams": ["RCB", "KKR"],
        "status": "complete",
        "result_type": "abandoned",
        "note": "Match abandoned due to wet outfield",
        "predicted_winner": "RCB",
        "inn1": None,
        "inn2": None,
    }
    body = message_builder.post_match_message(match, standings, [], [], {})
    # Reads as abandonment, not a fake win
    assert "abandoned" in body.lower() or "no result" in body.lower()
    assert "Match abandoned due to wet outfield" in body
    # Doesn't claim a winner
    assert "beat" not in body
    # Doesn't claim prediction correctness — we can't grade an abandoned match
    assert "correct" not in body.lower()


def test_post_match_no_result_without_note(standings):
    match = {
        "id": "1001",
        "teams": ["RCB", "KKR"],
        "status": "complete",
        "result_type": "no_result",
        "note": None,
        "predicted_winner": "RCB",
    }
    body = message_builder.post_match_message(match, standings, [], [], {})
    assert "no result" in body.lower() or "abandoned" in body.lower()
    assert "beat" not in body


def test_post_match_normal_win_unchanged(standings):
    # Regression: normal completion still produces the win body
    match = {
        "id": "1001",
        "teams": ["RCB", "KKR"],
        "status": "complete",
        "result_type": "win",
        "winner": "RCB",
        "actual_winner": "RCB",
        "result": "RCB won by 4 wickets",
        "predicted_winner": "RCB",
    }
    body = message_builder.post_match_message(match, standings, [], [], {})
    assert "RCB beat KKR" in body
    assert "abandoned" not in body.lower()
