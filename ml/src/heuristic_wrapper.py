"""Wrap the existing heuristic predictor (root src/predictor.py) for backtesting.

Reconstructs the heuristic's expected inputs (standings, recent_matches, squads)
from the backtest state. Read-only access to root src/predictor.py; the file is
imported via sys.path injection, never modified.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
# Add repo root so we can import the existing src.predictor
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import lazily to avoid affecting test collection if src has heavy imports
def _import_predictor():
    from src import predictor as P  # noqa
    return P


def _build_standings(season: int, state: dict) -> list[dict]:
    """Build iplt20-style standings from completed matches this season."""
    completed = [c for c in state["completed"] if c["season"] == season and not c.get("no_result")]
    table: dict[str, dict] = {}
    for m in completed:
        for t in (m["team1"], m["team2"]):
            row = table.setdefault(t, {
                "team": t, "played": 0, "won": 0, "lost": 0, "tied": 0,
                "points": 0, "nrr": 0.0, "for_runs": 0, "order": 0, "performance": "",
                "is_qualified": False,
            })
        winner = m.get("winner")
        if not winner:
            continue
        loser = m["team2"] if m["team1"] == winner else m["team1"]
        table[winner]["played"] += 1
        table[winner]["won"] += 1
        table[winner]["points"] += 2
        table[winner]["performance"] += "W"
        table[loser]["played"] += 1
        table[loser]["lost"] += 1
        table[loser]["performance"] += "L"

    rows = list(table.values())
    rows.sort(key=lambda r: (r["points"], r["won"]), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["order"] = i
    return rows


def _build_recent_matches(state: dict, limit: int = 50) -> list[dict]:
    """Recent completed matches in heuristic's expected shape."""
    out = []
    for m in state["completed"][-limit:]:
        if m.get("no_result"):
            continue
        if not m.get("winner"):
            continue
        out.append({
            "teams": [m["team1"], m["team2"]],
            "winner": m["winner"],
            "status": "complete",
            "venue_id": m.get("venue", ""),
            "venue_name": m.get("venue", ""),
            "second_batting": None,  # cricsheet matches.parquet doesn't expose this
        })
    return out


def _build_squads(state: dict, season: int) -> dict:
    """Build a placeholder squads dict. Real per-player stats live in players.parquet
    but the heuristic uses per-team aggregated batter/bowler ranks. We approximate
    via team season runs/wickets if available; else empty squads (heuristic falls back)."""
    return {}


def heuristic_predict(match_row: dict, state: dict) -> tuple[str, float]:
    """Adapter: call src.predictor.predict_winner using state-derived inputs.

    Returns (predicted_team, prob_team1_wins) — the heuristic produces a winner
    and a reason; we don't have a calibrated probability, so we use 0.55/0.45.
    """
    P = _import_predictor()

    season = int(match_row["season"])
    standings = _build_standings(season, state)
    recent_matches = _build_recent_matches(state)
    completed = recent_matches
    squads = _build_squads(state, season)
    all_teams = [r["team"] for r in standings] or [match_row["team1"], match_row["team2"]]

    # Build a `match` dict in the shape the heuristic expects
    match = {
        "venue_id": match_row.get("venue", ""),
        "venue_name": match_row.get("venue", ""),
        "home_team": None,  # not available in cricsheet
        "second_batting": None,
        "toss_winner": match_row.get("toss_winner"),
        "toss_decision": match_row.get("toss_decision"),
    }

    winner, _reason = P.predict_winner(
        match_row["team1"], match_row["team2"],
        standings, recent_matches, squads, all_teams,
        match=match, completed_matches=completed,
    )

    # We don't get a probability back. Mirror the heuristic's logistic constant
    # by recomputing the team-score gap to derive a soft probability.
    a_score = P._team_score(match_row["team1"], match_row["team2"], standings, recent_matches, squads, all_teams, match, completed)
    b_score = P._team_score(match_row["team2"], match_row["team1"], standings, recent_matches, squads, all_teams, match, completed)
    p_team1 = P._logistic(a_score - b_score)
    return winner, float(p_team1)
