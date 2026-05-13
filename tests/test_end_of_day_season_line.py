"""Tests that end_of_day_message renders a 'Season to date' line in place
of the old 'Predictions today' line, with correct percentage formatting.
"""
from __future__ import annotations

import pytest

from src import message_builder


@pytest.fixture
def standings():
    return [
        {"team": "RCB", "played": 13, "won": 10, "lost": 3, "points": 20, "nrr": 0.8},
        {"team": "KKR", "played": 13, "won": 5,  "lost": 8, "points": 10, "nrr": -0.2},
        {"team": "GT",  "played": 13, "won": 9,  "lost": 4, "points": 18, "nrr": 0.6},
        {"team": "SRH", "played": 13, "won": 7,  "lost": 6, "points": 14, "nrr": 0.3},
        {"team": "PBKS","played": 13, "won": 6,  "lost": 7, "points": 12, "nrr": 0.1},
    ]


def test_season_line_with_percentage(standings):
    body = message_builder.end_of_day_message(
        "2026-05-13", [], standings, [], [], {}, "https://example.test/",
        season_correct=13, season_total=17,
    )
    assert "Season to date prediction: 13 of 17 correct (76%)" in body
    assert "Predictions today" not in body


def test_season_line_no_percentage_when_zero_total(standings):
    body = message_builder.end_of_day_message(
        "2026-05-13", [], standings, [], [], {}, "url",
        season_correct=0, season_total=0,
    )
    assert "Season to date prediction: 0 of 0 correct" in body
    # No "(X%)" segment when there's nothing to divide by
    assert "%" not in body.split("Season to date prediction:")[1].split("\n")[0]



def test_season_line_zero_correct_with_some_total(standings):
    body = message_builder.end_of_day_message(
        "2026-05-13", [], standings, [], [], {}, "url",
        season_correct=0, season_total=3,
    )
    assert "Season to date prediction: 0 of 3 correct (0%)" in body


def test_season_line_perfect_record(standings):
    body = message_builder.end_of_day_message(
        "2026-05-13", [], standings, [], [], {}, "url",
        season_correct=5, season_total=5,
    )
    assert "Season to date prediction: 5 of 5 correct (100%)" in body


def test_season_line_rounds_percentage_to_integer(standings):
    """13/17 = 76.47% → '76%' (rounded). 2/3 = 66.67% → '67%'."""
    body = message_builder.end_of_day_message(
        "2026-05-13", [], standings, [], [], {}, "url",
        season_correct=2, season_total=3,
    )
    assert "Season to date prediction: 2 of 3 correct (67%)" in body


def test_existing_recap_components_preserved(standings):
    """Day-recap line, Updated top 4, archive URL — all should remain."""
    body = message_builder.end_of_day_message(
        "2026-05-13", [], standings, [], [], {}, "https://example.test/",
        season_correct=1, season_total=1,
    )
    assert "Day recap" in body
    assert "Updated top 4" in body
    assert "https://example.test/" in body
