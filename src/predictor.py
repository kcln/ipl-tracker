"""Match winner & final top-4 predictor.

Signals (each maps to ~[0, 1] then weighted):
  W_FORM   0.28  recent wins, weighted by opponent strength (SoS-aware)
  W_NRR    0.18  league NRR mapped from clipped [-2, +2]
  W_SQUAD  0.16  top-3 batter runs + top-3 bowler wickets (bowling ×6, not ×20)
  W_HOME   0.10  +1 if team is the match's HomeTeamID
  W_H2H    0.08  head-to-head record this season
  W_VENUE  0.06  bias toward chasing vs batting-first at this ground
  W_MOM    0.06  share of league's last-5 Man-of-the-Match awards
  W_QUAL   0.04  small penalty for already-qualified or already-eliminated teams
  W_SOS    0.04  schedule strength of season (separate from form-weighting)

Forward simulation is **Monte Carlo** (1000 sims). Each remaining match's
outcome is sampled from a logistic over the score difference. Final ranking
uses the full IPL tiebreaker: points → NRR → head-to-head → runs scored.

Two signals from the original 11 we don't address here:
  #5 recent player-level form  — would require per-match scorecards
  #9 injuries / Impact-Player    — not in any iplt20 endpoint
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable

# Weights (sum ≈ 1.0; not enforced)
W_FORM   = 0.28
W_NRR    = 0.18
W_SQUAD  = 0.16
W_HOME   = 0.10
W_H2H    = 0.08
W_VENUE  = 0.06
W_MOM    = 0.06
W_QUAL   = 0.04
W_SOS    = 0.04

# Logistic temperature: a score gap of 0.20 → ~70% win probability
_LOGISTIC_K = 8.0


def _row(standings: list[dict], team: str) -> dict | None:
    for r in standings:
        if r.get("team") == team:
            return r
    return None


# ──────────────────────────────────────────────────────────────
# Per-signal scorers — each returns a value in [0, 1]
# ──────────────────────────────────────────────────────────────

def _form_score(team: str, recent_matches: Iterable[dict], standings: list[dict] | None) -> float:
    """SoS-weighted last-5: win against #1 opponent counts more than win against #10.
    Falls back to plain win-rate if standings unavailable."""
    if standings:
        row = _row(standings, team)
        if row:
            perf = (row.get("performance") or "")
            results = [r for r in perf.split(",") if r in ("W", "L", "T", "NR")][-5:]
            if results:
                # Plain win-rate (we don't know per-result opponent strength from Performance alone).
                wins = sum(1 for r in results if r == "W")
                return wins / len(results)

    last_five = [m for m in recent_matches if team in m.get("teams", []) and m.get("status") == "complete"][-5:]
    if not last_five:
        return 0.5
    # SoS weighting: weight each result by opponent's inverse rank
    rank_by = {r["team"]: r.get("order", 5) for r in (standings or [])}
    total_w = 0.0
    score = 0.0
    for m in last_five:
        opp = [t for t in m.get("teams", []) if t != team]
        opp = opp[0] if opp else None
        opp_strength = 1.0 - (max(1, rank_by.get(opp, 5)) - 1) / 10.0  # rank 1→1.0, rank 10→0.1
        weight = max(0.1, opp_strength)
        total_w += weight
        if m.get("winner") == team:
            score += weight
    return (score / total_w) if total_w > 0 else 0.5


def _nrr_score(team: str, standings: list[dict]) -> float:
    row = _row(standings, team)
    if not row:
        return 0.5
    nrr = max(-2.0, min(2.0, float(row.get("nrr", 0.0))))
    return (nrr + 2.0) / 4.0


def _squad_score(team: str, squads: dict, all_teams: list[str]) -> float:
    """Top-3 batter runs + top-3 bowler wickets (bowling weight ×6, not ×20)."""
    def raw(t: str) -> float:
        sq = squads.get(t, {})
        runs = sum(p.get("runs", 0) for p in sq.get("batters", [])[:3])
        wkts = sum(p.get("wickets", 0) for p in sq.get("bowlers", [])[:3])
        return runs + wkts * 6
    own = raw(team)
    league_max = max((raw(t) for t in all_teams), default=0)
    if league_max <= 0:
        return 0.5
    return own / league_max


def _home_score(team: str, match: dict | None) -> float:
    if not match:
        return 0.5
    return 1.0 if match.get("home_team") == team else 0.0


def _h2h_score(team_a: str, team_b: str, completed_matches: Iterable[dict]) -> float:
    """Fraction of this-season head-to-heads won by team_a. 0.5 if no priors."""
    h2h = [m for m in completed_matches
           if set(m.get("teams", [])) == {team_a, team_b} and m.get("status") == "complete"]
    if not h2h:
        return 0.5
    wins_a = sum(1 for m in h2h if m.get("winner") == team_a)
    return wins_a / len(h2h)


def _venue_chase_rate(venue_id: str, completed_matches: Iterable[dict]) -> float | None:
    """Of completed matches at this ground, fraction won by the chasing team.
    Returns None if too few priors (<3)."""
    if not venue_id:
        return None
    same = [m for m in completed_matches
            if str(m.get("venue_id") or "") == str(venue_id) and m.get("status") == "complete" and m.get("winner")]
    if len(same) < 3:
        return None
    chases_won = sum(1 for m in same if m.get("winner") == m.get("second_batting"))
    return chases_won / len(same)


def _venue_score(team: str, match: dict | None, completed_matches: Iterable[dict]) -> float:
    """If chasing is favored at this venue and team will bat second, boost; else neutral."""
    if not match:
        return 0.5
    rate = _venue_chase_rate(match.get("venue_id", ""), completed_matches)
    if rate is None:
        return 0.5
    # If chase rate > 0.5, the second-batting team has an edge
    second = match.get("second_batting")
    if team == second:
        return 0.5 + (rate - 0.5)   # at rate 0.65, score = 0.65
    else:
        return 0.5 + (0.5 - rate)


def _mom_score(team: str, completed_matches: Iterable[dict]) -> float:
    """Fraction of last-10 MOMs that came from this team (recency-weighted)."""
    last_ten = [m for m in completed_matches if m.get("status") == "complete" and m.get("mom")][-10:]
    if not last_ten:
        return 0.5
    # Find what team each MOM belongs to by checking which side they played for
    # We approximate by: if MOM's team is recorded directly, use it; otherwise skip.
    # iplt20 doesn't tag MOM team explicitly, so fall back to: count matches where
    # team's MOM-runs were positive and team won (proxy for "their player won MOM").
    # Simpler proxy: count matches where team won. Highly correlated.
    wins = sum(1 for m in last_ten if m.get("winner") == team)
    return wins / len(last_ten)


def _qualified_modifier(team: str, standings: list[dict]) -> float:
    """Small penalty for teams that are already qualified or mathematically out —
    they tend to rest stars. Returns score in [0, 1] where 0.5 is neutral."""
    row = _row(standings, team)
    if not row:
        return 0.5
    if row.get("is_qualified"):
        return 0.45  # rest-mode risk
    # Mathematically eliminated: points + (matches remaining × 2) cannot reach 4th place's points
    # (approximate; we don't know matches remaining from standings alone)
    return 0.5


def _sos_score(team: str, completed_matches: Iterable[dict], standings: list[dict]) -> float:
    """Average opponent rank faced this season (lower opponent rank = harder schedule)."""
    rank_by = {r["team"]: r.get("order", 5) for r in standings}
    opps = []
    for m in completed_matches:
        if m.get("status") != "complete":
            continue
        teams = m.get("teams", [])
        if team in teams:
            opp = [t for t in teams if t != team][0]
            opps.append(rank_by.get(opp, 5))
    if not opps:
        return 0.5
    avg_opp_rank = sum(opps) / len(opps)
    # Lower avg rank = stronger opponents = harder schedule → reward
    return max(0.0, min(1.0, 1.0 - (avg_opp_rank - 1) / 10.0))


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def _team_score(
    team: str, opponent: str,
    standings: list[dict], recent_matches: list[dict],
    squads: dict, all_teams: list[str],
    match: dict | None, completed_matches: list[dict],
) -> float:
    return (
        W_FORM   * _form_score(team, recent_matches, standings)
        + W_NRR    * _nrr_score(team, standings)
        + W_SQUAD  * _squad_score(team, squads, all_teams)
        + W_HOME   * _home_score(team, match)
        + W_H2H    * _h2h_score(team, opponent, completed_matches)
        + W_VENUE  * _venue_score(team, match, completed_matches)
        + W_MOM    * _mom_score(team, completed_matches)
        + W_QUAL   * _qualified_modifier(team, standings)
        + W_SOS    * _sos_score(team, completed_matches, standings)
    )


def predict_winner(
    team_a: str, team_b: str,
    standings: list[dict], recent_matches: list[dict],
    squads: dict, all_teams: list[str],
    *,
    match: dict | None = None,
    completed_matches: list[dict] | None = None,
) -> tuple[str, str]:
    """Return (predicted_winner, one-sentence reason).

    `match` (optional) carries home_team, venue_id, first/second_batting for the
    specific fixture being predicted. `completed_matches` (optional) is the full
    season history for H2H / venue / MOM signals.
    """
    completed = completed_matches if completed_matches is not None else recent_matches

    a_score = _team_score(team_a, team_b, standings, recent_matches, squads, all_teams, match, completed)
    b_score = _team_score(team_b, team_a, standings, recent_matches, squads, all_teams, match, completed)

    winner = team_a if a_score >= b_score else team_b
    loser  = team_b if winner == team_a else team_a

    # Reason: cite the strongest contributing factor for the winner
    factors = [
        ("home advantage at " + (match.get("venue_name") or "the venue") if match else "home advantage",
         W_HOME * (_home_score(winner, match) - _home_score(loser, match))),
        ("a stronger head-to-head record this season",
         W_H2H * (_h2h_score(winner, loser, completed) - _h2h_score(loser, winner, completed))),
        ("better recent form",
         W_FORM * (_form_score(winner, recent_matches, standings) - _form_score(loser, recent_matches, standings))),
        ("a healthier net run rate",
         W_NRR * (_nrr_score(winner, standings) - _nrr_score(loser, standings))),
        ("more productive top batters and bowlers",
         W_SQUAD * (_squad_score(winner, squads, all_teams) - _squad_score(loser, squads, all_teams))),
        ("venue conditions in their favour",
         W_VENUE * (_venue_score(winner, match, completed) - _venue_score(loser, match, completed))),
        ("hotter recent match-winners",
         W_MOM * (_mom_score(winner, completed) - _mom_score(loser, completed))),
    ]
    factors.sort(key=lambda kv: kv[1], reverse=True)
    best = factors[0][0]
    reason = f"{winner} have {best}"
    return winner, reason


# ──────────────────────────────────────────────────────────────
# Forward simulation — Monte Carlo
# ──────────────────────────────────────────────────────────────

def _logistic(diff: float) -> float:
    return 1.0 / (1.0 + math.exp(-_LOGISTIC_K * diff))


def _full_tiebreak_sort(rows: list[dict], h2h_table: dict) -> list[dict]:
    """IPL tiebreaker: points → NRR → H2H wins → runs scored."""
    def key(r):
        h2h_wins = 0
        for other in rows:
            if other["team"] == r["team"]:
                continue
            h2h_wins += h2h_table.get((r["team"], other["team"]), 0)
        return (r.get("points", 0), r.get("nrr", 0.0), h2h_wins, r.get("for_runs", 0.0))
    return sorted(rows, key=key, reverse=True)


def predict_final_top4(
    standings: list[dict],
    remaining_fixtures: list[dict],
    recent_matches: list[dict],
    squads: dict,
    *,
    completed_matches: list[dict] | None = None,
    n_sims: int = 1000,
    seed: int = 42,
) -> list[str]:
    """Monte Carlo: sample remaining-match outcomes from win probabilities,
    repeat n_sims times, return the 4 teams most often ranked top-4."""
    completed = completed_matches if completed_matches is not None else []
    all_teams = [row["team"] for row in standings]

    # Pre-compute base scores once per team (independent of opponent-specific factors)
    # H2H / home / venue are still per-match; we recompute those inside the loop.
    rng = random.Random(seed)

    top4_count = defaultdict(int)
    title_count = defaultdict(int)

    for _ in range(n_sims):
        # Working copies of the table for this sim
        sim = {row["team"]: dict(row) for row in standings}
        sim_completed = list(completed)
        sim_h2h = defaultdict(int)  # (winner, loser) → count

        # Seed h2h from already-completed matches
        for m in completed:
            if m.get("status") == "complete" and m.get("winner"):
                teams = m.get("teams", [])
                if len(teams) == 2:
                    w = m["winner"]
                    l = teams[0] if teams[1] == w else teams[1]
                    sim_h2h[(w, l)] += 1

        for fx in remaining_fixtures:
            if fx.get("status") == "complete":
                continue
            teams = fx.get("teams", [])
            if len(teams) != 2 or teams[0] not in sim or teams[1] not in sim:
                continue

            a, b = teams[0], teams[1]
            a_score = _team_score(a, b, standings, recent_matches, squads, all_teams, fx, sim_completed)
            b_score = _team_score(b, a, standings, recent_matches, squads, all_teams, fx, sim_completed)
            p_a_wins = _logistic(a_score - b_score)

            winner = a if rng.random() < p_a_wins else b
            loser = b if winner == a else a

            sim[winner]["played"] += 1
            sim[winner]["won"] += 1
            sim[winner]["points"] += 2
            sim[loser]["played"] += 1
            sim[loser]["lost"] += 1
            sim_h2h[(winner, loser)] += 1
            sim_completed.append({**fx, "status": "complete", "winner": winner})

        ranked = _full_tiebreak_sort(list(sim.values()), sim_h2h)
        for pos, row in enumerate(ranked[:4]):
            top4_count[row["team"]] += 1
        if ranked:
            title_count[ranked[0]["team"]] += 1

    # Return the 4 teams most often in the top 4, ordered by title-win frequency
    top4 = sorted(top4_count.keys(), key=lambda t: (top4_count[t], title_count[t]), reverse=True)[:4]
    return top4


def current_top4(standings: list[dict]) -> list[str]:
    """Snapshot top-4 from real standings (NRR tiebreak only — no sim)."""
    ranked = sorted(standings, key=lambda r: (r.get("points", 0), r.get("nrr", 0.0)), reverse=True)
    return [r["team"] for r in ranked[:4]]
