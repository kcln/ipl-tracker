"""Tests for rewriting the team-vs-team header on the first line of a message
body. Used by the one-shot script that heals past messages baked before
the data_fetcher home-first fix.
"""
from __future__ import annotations

from src import message_builder


def test_rewrite_swaps_when_order_is_wrong():
    body = "KKR vs RCB — Toss\n\nToss: RCB won and chose to field."
    out = message_builder.rewrite_team_header(body, home_team="RCB")
    assert out.startswith("RCB vs KKR — Toss")
    # rest of the body is untouched
    assert "Toss: RCB won and chose to field." in out


def test_rewrite_noop_when_already_home_first():
    body = "RCB vs KKR — Toss\n\nbody"
    out = message_builder.rewrite_team_header(body, home_team="RCB")
    assert out == body


def test_rewrite_noop_when_home_team_not_in_header():
    body = "GT vs SRH — Innings break\n\nbody"
    out = message_builder.rewrite_team_header(body, home_team="MI")
    assert out == body


def test_rewrite_handles_no_team_header():
    body = "IPL 2026 - Wednesday, May 13\n\nbody"
    out = message_builder.rewrite_team_header(body, home_team="RCB")
    assert out == body


def test_rewrite_handles_trailing_dash():
    """Some headers have ' — Powerplay 1' suffix; preserve that exactly."""
    body = "KKR vs RCB — Powerplay 1\n\nKKR first innings: 56/2"
    out = message_builder.rewrite_team_header(body, home_team="RCB")
    assert out.splitlines()[0] == "RCB vs KKR — Powerplay 1"
    # The "KKR first innings" line stays — that's the batting team, semantic
    assert "KKR first innings: 56/2" in out


def test_rewrite_only_touches_first_line():
    """A team-vs-team mention later in the body shouldn't get swapped."""
    body = (
        "KKR vs RCB — Powerplay 1\n"
        "\n"
        "Comparison: KKR vs RCB last 5: 3-2"
    )
    out = message_builder.rewrite_team_header(body, home_team="RCB")
    lines = out.splitlines()
    assert lines[0] == "RCB vs KKR — Powerplay 1"
    assert lines[2] == "Comparison: KKR vs RCB last 5: 3-2"  # unchanged
