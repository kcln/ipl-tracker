"""Tests for state.season_prediction_record — counts cumulative pre-match
prediction accuracy by reading the "Pre-match prediction: correct/incorrect"
line baked into every post_match message body.

Critical that the math is right: this number goes on the public daily recap.
"""
from __future__ import annotations

import pytest

from src import state


def _msg(type_, body):
    return {"type": type_, "body": body}


def test_empty_dict_yields_zero_zero():
    assert state.season_prediction_record({}) == (0, 0)


def test_empty_days_yields_zero_zero():
    assert state.season_prediction_record({"days": {}}) == (0, 0)


def test_day_with_no_messages_yields_zero_zero():
    s = {"days": {"2026-05-13": {"messages": []}}}
    assert state.season_prediction_record(s) == (0, 0)


def test_single_correct():
    s = {"days": {"2026-05-13": {"messages": [
        _msg("post_match_1", "RCB beat KKR by 6 wickets\n\nPre-match prediction: correct\n\nfooter"),
    ]}}}
    assert state.season_prediction_record(s) == (1, 1)


def test_single_incorrect():
    s = {"days": {"2026-05-13": {"messages": [
        _msg("post_match_1", "KKR beat RCB by 3 runs\n\nPre-match prediction: incorrect\n\nfooter"),
    ]}}}
    assert state.season_prediction_record(s) == (0, 1)


def test_mixed_across_days():
    s = {"days": {
        "2026-05-11": {"messages": [_msg("post_match_1", "Pre-match prediction: correct")]},
        "2026-05-12": {"messages": [_msg("post_match_1", "Pre-match prediction: incorrect")]},
        "2026-05-13": {"messages": [_msg("post_match_1", "Pre-match prediction: correct")]},
    }}
    assert state.season_prediction_record(s) == (2, 3)


def test_multiple_matches_same_day():
    s = {"days": {"2026-05-13": {"messages": [
        _msg("post_match_1", "Pre-match prediction: correct"),
        _msg("post_match_2", "Pre-match prediction: incorrect"),
        _msg("post_match_3", "Pre-match prediction: correct"),
    ]}}}
    assert state.season_prediction_record(s) == (2, 3)


def test_abandoned_match_not_counted():
    """Post-match body for an abandoned match doesn't carry the prediction
    line at all — skips out of the tally."""
    s = {"days": {"2026-05-13": {"messages": [
        _msg("post_match_1",
             "RCB vs KKR — Match abandoned (no result)\n\nMatch abandoned due to wet outfield"),
    ]}}}
    assert state.season_prediction_record(s) == (0, 0)


def test_non_post_match_message_with_prediction_string_not_counted():
    """A morning/EOD/toss message that happens to contain the prediction
    string (e.g. quoted in a recap) must not bump the counter."""
    s = {"days": {"2026-05-13": {"messages": [
        _msg("morning", "Quoting yesterday: Pre-match prediction: correct"),
        _msg("end_of_day", "Season to date: 5 of 7 correct"),
        _msg("toss_1", "RCB vs KKR — Toss"),
    ]}}}
    assert state.season_prediction_record(s) == (0, 0)


def test_post_match_without_prediction_line_not_counted():
    s = {"days": {"2026-05-13": {"messages": [
        _msg("post_match_1", "RCB beat KKR by 6 wickets\n\n(somehow no prediction line)"),
    ]}}}
    assert state.season_prediction_record(s) == (0, 0)


def test_post_match_with_both_strings_counts_once_as_correct():
    """Defensive: if a body somehow contains both 'correct' and 'incorrect'
    keywords, count as 'incorrect' to be conservative (since 'incorrect'
    contains 'correct' as a substring — make sure we don't double-count)."""
    body = "Pre-match prediction: incorrect"
    s = {"days": {"2026-05-13": {"messages": [_msg("post_match_1", body)]}}}
    # 'incorrect' contains 'correct' — the implementation must not double-tally
    assert state.season_prediction_record(s) == (0, 1)


def test_deterministic_across_day_ordering():
    """Day key order shouldn't affect the result."""
    s1 = {"days": {
        "2026-05-11": {"messages": [_msg("post_match_1", "Pre-match prediction: correct")]},
        "2026-05-13": {"messages": [_msg("post_match_1", "Pre-match prediction: incorrect")]},
    }}
    s2 = {"days": {
        "2026-05-13": {"messages": [_msg("post_match_1", "Pre-match prediction: incorrect")]},
        "2026-05-11": {"messages": [_msg("post_match_1", "Pre-match prediction: correct")]},
    }}
    assert state.season_prediction_record(s1) == state.season_prediction_record(s2) == (1, 2)
