"""Tests for the standalone top-4 simulator."""
from __future__ import annotations

from ml.src.topfour_sim import simulate_top4


def _flat_predict(t1, t2):
    return 0.5


def test_unambiguous_separation_gives_deterministic_top4():
    """When point gaps are huge and outcomes are 50/50, top-4 should be the
    pre-simulation top 4 with probability ~1.0."""
    standings = [
        {"team": "A", "points": 20, "nrr": 1.0, "played": 10},
        {"team": "B", "points": 18, "nrr": 0.5, "played": 10},
        {"team": "C", "points": 16, "nrr": 0.4, "played": 10},
        {"team": "D", "points": 14, "nrr": 0.3, "played": 10},
        {"team": "E", "points":  4, "nrr": -1.0, "played": 10},
    ]
    fixtures = [{"team1": "A", "team2": "E"}]  # only 1 game left
    out = simulate_top4(standings, fixtures, _flat_predict, n_sims=200, seed=1)
    for t in ("A", "B", "C", "D"):
        assert out[t]["top4_prob"] == 1.0, f"{t} should always be top 4"
    assert out["E"]["top4_prob"] == 0.0


def test_unknown_teams_in_fixtures_skipped():
    standings = [{"team": "A", "points": 0, "nrr": 0.0, "played": 0}]
    fixtures = [{"team1": "A", "team2": "ZZZ"}]
    out = simulate_top4(standings, fixtures, _flat_predict, n_sims=50, seed=1)
    # No crash; A still in top 4 vacuously
    assert "A" in out
    assert out["A"]["top4_prob"] == 1.0


def test_meta_block_present():
    standings = [{"team": "A", "points": 0, "nrr": 0.0, "played": 0}]
    out = simulate_top4(standings, [], _flat_predict, n_sims=10, seed=42)
    assert "_meta" in out
    assert out["_meta"]["n_sims"] == 10
    assert out["_meta"]["seed"] == 42


def test_close_race_produces_nondeterministic_top4():
    standings = [
        {"team": "A", "points": 10, "nrr": 0.0, "played": 10},
        {"team": "B", "points": 10, "nrr": 0.0, "played": 10},
        {"team": "C", "points": 10, "nrr": 0.0, "played": 10},
        {"team": "D", "points": 10, "nrr": 0.0, "played": 10},
        {"team": "E", "points": 10, "nrr": 0.0, "played": 10},
    ]
    fixtures = [
        {"team1": "A", "team2": "B"}, {"team1": "C", "team2": "D"},
        {"team1": "A", "team2": "E"}, {"team1": "B", "team2": "C"},
    ]
    out = simulate_top4(standings, fixtures, _flat_predict, n_sims=500, seed=7)
    # In a 5-way tie with random outcomes, top-4 probabilities should be strictly
    # between 0 and 1 for at least one team.
    nondet = [t for t in ("A","B","C","D","E") if 0.0 < out[t]["top4_prob"] < 1.0]
    assert len(nondet) >= 1, "no team had a non-degenerate top-4 probability"
