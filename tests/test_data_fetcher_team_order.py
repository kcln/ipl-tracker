"""Tests that match.teams reflects the scheduled (home-first) order, not the
batting order. Without this, the toss message reorders teams post-toss
(today: morning reads "RCB vs KKR" but toss reads "KKR vs RCB" because
the iplt20 feed flips FirstBattingTeamCode after the toss).
"""
from __future__ import annotations

from src import data_fetcher


def _base():
    return {
        "MatchID": "1001",
        "FirstBattingTeamCode": "KKR",
        "SecondBattingTeamCode": "RCB",
        "FirstBattingTeamID": "1",
        "SecondBattingTeamID": "2",
        "MatchDate": "2026-05-13",
        "MatchTime": "19:30",
        "MatchStatus": "Live",
    }


def test_teams_uses_home_first_when_home_known():
    raw = _base() | {
        "HomeTeamID": "2",            # RCB is home
        "HomeTeamName": "Royal Challengers Bengaluru",
    }
    out = data_fetcher._parse_iplt20_match(raw)
    assert out["teams"] == ["RCB", "KKR"]
    # batting order still available separately
    assert out["first_batting"] == "KKR"
    assert out["second_batting"] == "RCB"


def test_teams_falls_back_to_feed_order_when_home_unknown():
    raw = _base()  # no HomeTeamID
    out = data_fetcher._parse_iplt20_match(raw)
    # Falls back to the feed's team order — t1 then t2
    assert out["teams"] == ["KKR", "RCB"]


def test_teams_does_not_flip_after_toss():
    """Same match, before toss (HomeTeamID known, batting order is the
    scheduled default) and after toss (batting order flipped). teams should
    not change."""
    pre_toss = _base() | {
        "MatchStatus": "Pre",
        "HomeTeamID": "2",
        "FirstBattingTeamCode": "RCB",
        "SecondBattingTeamCode": "KKR",
        "FirstBattingTeamID": "2",
        "SecondBattingTeamID": "1",
    }
    post_toss = _base() | {
        "MatchStatus": "Live",
        "HomeTeamID": "2",
        "FirstBattingTeamCode": "KKR",  # toss flipped this
        "SecondBattingTeamCode": "RCB",
        "FirstBattingTeamID": "1",
        "SecondBattingTeamID": "2",
    }
    pre = data_fetcher._parse_iplt20_match(pre_toss)
    post = data_fetcher._parse_iplt20_match(post_toss)
    assert pre["teams"] == post["teams"] == ["RCB", "KKR"]
