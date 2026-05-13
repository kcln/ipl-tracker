"""Tests for rewriting the EOD predictions line in a historical body, used
by the one-shot heal script. Validates the math-sensitive bits in isolation
before applying anything to state.json.
"""
from __future__ import annotations

from src import message_builder


def test_rewrite_replaces_today_line_with_season_line():
    body = (
        "IPL 2026 - Wednesday, May 13 - Day recap\n"
        "\n"
        "RCB beat KKR by 6 wickets\n"
        "\n"
        "Predictions today: 1 of 1 correct\n"
        "\n"
        "Updated top 4: RCB, GT, SRH, PBKS\n"
        "Archive: https://example.test/"
    )
    out = message_builder.rewrite_eod_predictions_line(body, correct=2, total=3)
    assert "Predictions today" not in out
    assert "Season to date: 2 of 3 correct (67%)" in out
    # Everything else is preserved
    assert "RCB beat KKR by 6 wickets" in out
    assert "Updated top 4: RCB, GT, SRH, PBKS" in out


def test_rewrite_handles_zero_total():
    body = "Predictions today: 0 of 0 correct"
    out = message_builder.rewrite_eod_predictions_line(body, correct=0, total=0)
    assert out == "Season to date: 0 of 0 correct"
    assert "%" not in out


def test_rewrite_handles_perfect_record():
    body = "...\nPredictions today: 1 of 1 correct\n..."
    out = message_builder.rewrite_eod_predictions_line(body, correct=5, total=5)
    assert "Season to date: 5 of 5 correct (100%)" in out


def test_rewrite_noop_when_no_predictions_line():
    body = "IPL 2026 - Day recap\n\nNo prediction reporting here\n"
    out = message_builder.rewrite_eod_predictions_line(body, correct=2, total=3)
    assert out == body


def test_rewrite_is_idempotent_on_already_rewritten_body():
    """If body already says 'Season to date', leave it alone — don't add a
    second line."""
    body = "Season to date: 2 of 3 correct (67%)"
    out = message_builder.rewrite_eod_predictions_line(body, correct=2, total=3)
    assert out == body
