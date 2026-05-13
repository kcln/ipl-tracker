"""Tests that _match_from_iplt20 (single-match fetcher used by
fetch_current_match) propagates the note and result_type fields. Without
this, the tracker merge loop in tracker.py:626-633 won't see fresh
delay-status changes between full-fixtures refreshes.
"""
from __future__ import annotations

from unittest.mock import patch

from src import data_fetcher


def test_match_from_iplt20_carries_note_and_result_type():
    fake_payload = {
        "Matchsummary": [
            {
                "MatchID": "1001",
                "FirstBattingTeamCode": "RCB",
                "SecondBattingTeamCode": "KKR",
                "MatchDate": "2026-05-13",
                "MatchTime": "19:30",
                "MatchStatus": "Live",
                "MatchProgress": "Rain stoppage at 7.2 overs",
            }
        ]
    }
    with patch.object(data_fetcher, "_iplt20_fetch", return_value=fake_payload):
        out = data_fetcher._match_from_iplt20("1001")
    assert out is not None
    assert out["note"] == "Rain stoppage at 7.2 overs"
    assert out["status"] == "live"
    assert out["result_type"] is None


def test_match_from_iplt20_carries_abandoned_result_type():
    fake_payload = {
        "Matchsummary": [
            {
                "MatchID": "1001",
                "FirstBattingTeamCode": "RCB",
                "SecondBattingTeamCode": "KKR",
                "MatchDate": "2026-05-13",
                "MatchTime": "19:30",
                "MatchStatus": "Abandoned",
                "MatchProgress": "Match abandoned due to wet outfield",
            }
        ]
    }
    with patch.object(data_fetcher, "_iplt20_fetch", return_value=fake_payload):
        out = data_fetcher._match_from_iplt20("1001")
    assert out["status"] == "complete"
    assert out["result_type"] == "abandoned"
    assert out["note"] == "Match abandoned due to wet outfield"
