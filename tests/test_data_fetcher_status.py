"""Tests for status-map extensions and note/result_type capture in the
iplt20 / Cricinfo / Cricbuzz parsers.

Goals:
- "abandoned" / "no result" / "washed out" raw statuses map to status=complete
- result_type field carries the semantic outcome: "win" | "no_result" | "abandoned" | None
- note field carries upstream free-text describing the current state, available
  regardless of whether status is scheduled / live / complete
"""
from __future__ import annotations

from src import data_fetcher


# ---------- iplt20 ----------

def _iplt20_base() -> dict:
    """Minimal valid iplt20 match payload."""
    return {
        "MatchID": "1001",
        "FirstBattingTeamCode": "RCB",
        "SecondBattingTeamCode": "KKR",
        "FirstBattingTeamName": "Royal Challengers Bengaluru",
        "SecondBattingTeamName": "Kolkata Knight Riders",
        "MatchDate": "2026-05-13",
        "MatchTime": "19:30",
        "MatchStatus": "Pre",
    }


def test_iplt20_status_abandoned_maps_to_complete_with_result_type():
    m = _iplt20_base() | {"MatchStatus": "Abandoned"}
    out = data_fetcher._parse_iplt20_match(m)
    assert out is not None
    assert out["status"] == "complete"
    assert out["result_type"] == "abandoned"


def test_iplt20_status_no_result_maps_to_complete():
    m = _iplt20_base() | {"MatchStatus": "No Result"}
    out = data_fetcher._parse_iplt20_match(m)
    assert out["status"] == "complete"
    assert out["result_type"] == "no_result"


def test_iplt20_status_washed_out_maps_to_complete():
    m = _iplt20_base() | {"MatchStatus": "Washed Out"}
    out = data_fetcher._parse_iplt20_match(m)
    assert out["status"] == "complete"
    assert out["result_type"] == "no_result"


def test_iplt20_status_post_with_winner_has_result_type_win():
    m = _iplt20_base() | {
        "MatchStatus": "Post",
        "WinningTeamID": "1",
        "FirstBattingTeamID": "1",
        "Commentss": "RCB won by 4 wickets",
    }
    out = data_fetcher._parse_iplt20_match(m)
    assert out["status"] == "complete"
    assert out["result_type"] == "win"
    assert out["winner"] == "RCB"


def test_iplt20_note_captures_match_progress_when_present():
    m = _iplt20_base() | {"MatchProgress": "Wet outfield, inspection at 8:15 PM IST"}
    out = data_fetcher._parse_iplt20_match(m)
    assert out["note"] == "Wet outfield, inspection at 8:15 PM IST"


def test_iplt20_note_is_none_when_no_status_text():
    out = data_fetcher._parse_iplt20_match(_iplt20_base())
    assert out["note"] is None


def test_iplt20_result_type_none_for_scheduled():
    out = data_fetcher._parse_iplt20_match(_iplt20_base())
    assert out["result_type"] is None
