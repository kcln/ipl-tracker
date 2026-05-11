"""Match winner & final top-4 predictor.

Weights:
  form (last 5)  40%
  NRR            30%
  squad form     30%
"""
from __future__ import annotations

import copy
from typing import Iterable


W_FORM = 0.40
W_NRR = 0.30
W_SQUAD = 0.30


def _form_score(team: str, recent_matches: Iterable[dict]) -> float:
    """Wins in last 5 matches for `team`, normalized to 0..1."""
    last_five = [m for m in recent_matches if team in m.get("teams", []) and m.get("status") == "complete"][-5:]
    if not last_five:
        return 0.5  # neutral prior
    wins = sum(1 for m in last_five if m.get("winner") == team)
    return wins / len(last_five)


def _nrr_score(team: str, standings: list[dict]) -> float:
    """Map NRR roughly into 0..1. NRR typically lives in [-2, +2]."""
    for row in standings:
        if row["team"] == team:
            nrr = max(-2.0, min(2.0, float(row.get("nrr", 0.0))))
            return (nrr + 2.0) / 4.0
    return 0.5


def _squad_score(team: str, squads: dict, all_teams: list[str]) -> float:
    """Sum top-3 batter runs + top-3 bowler wickets, normalized across league."""
    def raw(t: str) -> float:
        sq = squads.get(t, {})
        runs = sum(p.get("runs", 0) for p in sq.get("batters", [])[:3])
        wkts = sum(p.get("wickets", 0) for p in sq.get("bowlers", [])[:3])
        # rough scaling: 1 wicket ≈ 20 runs of impact
        return runs + wkts * 20

    own = raw(team)
    league_max = max((raw(t) for t in all_teams), default=0)
    if league_max <= 0:
        return 0.5
    return own / league_max


def predict_winner(
    team_a: str,
    team_b: str,
    standings: list[dict],
    recent_matches: list[dict],
    squads: dict,
    all_teams: list[str],
) -> tuple[str, str]:
    """Return (predicted_winner, one-sentence reason)."""
    a_score = (
        W_FORM * _form_score(team_a, recent_matches)
        + W_NRR * _nrr_score(team_a, standings)
        + W_SQUAD * _squad_score(team_a, squads, all_teams)
    )
    b_score = (
        W_FORM * _form_score(team_b, recent_matches)
        + W_NRR * _nrr_score(team_b, standings)
        + W_SQUAD * _squad_score(team_b, squads, all_teams)
    )

    winner = team_a if a_score >= b_score else team_b
    loser = team_b if winner == team_a else team_a

    # Build human reason: cite the strongest factor
    w_form = _form_score(winner, recent_matches)
    l_form = _form_score(loser, recent_matches)
    w_nrr = _nrr_score(winner, standings)
    l_nrr = _nrr_score(loser, standings)
    w_sq = _squad_score(winner, squads, all_teams)
    l_sq = _squad_score(loser, squads, all_teams)

    diffs = {
        "form": (w_form - l_form, f"{winner} have stronger recent form"),
        "nrr": (w_nrr - l_nrr, f"{winner} have a healthier net run rate"),
        "squad": (w_sq - l_sq, f"{winner}'s top batters and bowlers have produced more"),
    }
    factor, (_, reason) = max(diffs.items(), key=lambda kv: kv[1][0])
    return winner, reason


def predict_final_top4(
    standings: list[dict],
    remaining_fixtures: list[dict],
    recent_matches: list[dict],
    squads: dict,
) -> list[str]:
    """Forward-simulate remaining matches; return top 4 teams in final order."""
    # Mutable working table
    sim = {row["team"]: dict(row) for row in standings}
    all_teams = list(sim.keys())

    for fx in remaining_fixtures:
        if fx.get("status") == "complete":
            continue
        teams = fx.get("teams", [])
        if len(teams) != 2 or teams[0] not in sim or teams[1] not in sim:
            continue
        winner, _ = predict_winner(teams[0], teams[1], standings, recent_matches, squads, all_teams)
        loser = teams[1] if winner == teams[0] else teams[0]
        sim[winner]["played"] += 1
        sim[winner]["won"] += 1
        sim[winner]["points"] += 2
        sim[loser]["played"] += 1
        sim[loser]["lost"] += 1
        # NRR drift is too noisy to model without scores; leave unchanged

    ranked = sorted(
        sim.values(),
        key=lambda r: (r["points"], r.get("nrr", 0.0)),
        reverse=True,
    )
    return [r["team"] for r in ranked[:4]]


def current_top4(standings: list[dict]) -> list[str]:
    ranked = sorted(
        standings,
        key=lambda r: (r["points"], r.get("nrr", 0.0)),
        reverse=True,
    )
    return [r["team"] for r in ranked[:4]]
