"""Point-in-time feature engineering for match-winner prediction.

Each feature is a function (match_row, state) -> value where state contains
only data from matches with date strictly before match_row["date"].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _completed_for_team(state: dict, team: str, season: int | None = None):
    out = []
    for c in state["completed"]:
        if c.get("no_result"):
            continue
        if season is not None and c["season"] != season:
            continue
        if c["team1"] == team or c["team2"] == team:
            out.append(c)
    return out


def feat_form_last5_wins(match_row: dict, state: dict) -> float:
    """Wins in last 5 completed matches for team1 minus team2."""
    t1, t2 = match_row["team1"], match_row["team2"]
    def wr(team):
        recent = _completed_for_team(state, team)[-5:]
        if not recent:
            return 0.5
        wins = sum(1 for c in recent if c.get("winner") == team)
        return wins / len(recent)
    return wr(t1) - wr(t2)


def feat_season_nrr_diff(match_row: dict, state: dict) -> float:
    """NRR proxy: avg margin-runs differential per match this season."""
    season = int(match_row["season"])
    t1, t2 = match_row["team1"], match_row["team2"]
    def nrr_proxy(team):
        recent = _completed_for_team(state, team, season=season)
        if not recent:
            return 0.0
        wins = sum(1 for c in recent if c.get("winner") == team)
        losses = sum(1 for c in recent if c.get("winner") and c.get("winner") != team)
        n = len(recent)
        return (wins - losses) / max(1, n)
    return nrr_proxy(t1) - nrr_proxy(t2)


def feat_squad_strength_proxy(match_row: dict, state: dict) -> float:
    """Career win rate for team1 minus team2 (rough squad-strength proxy)."""
    t1, t2 = match_row["team1"], match_row["team2"]
    def career_wr(team):
        recent = _completed_for_team(state, team)
        if not recent:
            return 0.5
        wins = sum(1 for c in recent if c.get("winner") == team)
        return wins / len(recent)
    return career_wr(t1) - career_wr(t2)


def feat_home_advantage(match_row: dict, state: dict) -> float:
    """Cricsheet doesn't tag home_team. Use venue-team affinity instead.
    Returns: team1's historical win rate at this venue minus team2's."""
    venue = match_row.get("venue", "")
    if not venue:
        return 0.0
    t1, t2 = match_row["team1"], match_row["team2"]
    vh = state["venue_history"].get(venue, [])
    def venue_wr(team):
        team_games = [m for m in vh if m["team1"] == team or m["team2"] == team]
        if not team_games:
            return 0.5
        wins = sum(1 for m in team_games if m.get("winner") == team)
        return wins / len(team_games)
    return venue_wr(t1) - venue_wr(t2)


def feat_h2h_season(match_row: dict, state: dict) -> float:
    """team1's win share in h2h this season minus 0.5."""
    season = int(match_row["season"])
    t1, t2 = match_row["team1"], match_row["team2"]
    a, b = sorted([t1, t2])
    ent = state["h2h"].get((a, b, season))
    if not ent:
        return 0.0
    total = ent[a] + ent[b]
    if total == 0:
        return 0.0
    return ent[t1] / total - 0.5


def feat_venue_chase_rate(match_row: dict, state: dict) -> float:
    """Placeholder feature — always returns 0.0.

    The intent was historical chase-rate at this venue (chase rate − 0.5), but
    cricsheet's matches.parquet only exposes the WINNER per match, not the team
    that batted second. The richer phase_features module reads `win_by_wickets`
    from matches.parquet directly to recover chase information; this feature is
    kept here as a no-op so v1's saved logistic artifact stays loadable
    (it expects a 9-feature input vector that includes `venue_chase_rate`).
    """
    return 0.0


def feat_pom_recency(match_row: dict, state: dict) -> float:
    """Share of last 10 PoM awards that went to a player on each team.
    We don't have squads in state, so proxy = recent win rate (correlated)."""
    t1, t2 = match_row["team1"], match_row["team2"]
    last10 = state["completed"][-10:]
    if not last10:
        return 0.0
    def share(team):
        return sum(1 for m in last10 if m.get("winner") == team) / len(last10)
    return share(t1) - share(t2)


def feat_qualified_flag(match_row: dict, state: dict) -> float:
    """Late-season indicator: matches played in season for each team, late season
    can affect predictions due to resting stars. Returns team1.matches - team2.matches this season."""
    season = int(match_row["season"])
    t1, t2 = match_row["team1"], match_row["team2"]
    def count(team):
        return len(_completed_for_team(state, team, season=season))
    return count(t1) - count(t2)


def feat_strength_of_schedule(match_row: dict, state: dict) -> float:
    """Average opponent's career win rate faced so far this season, team1 vs team2."""
    season = int(match_row["season"])
    t1, t2 = match_row["team1"], match_row["team2"]
    def team_career_wr(team):
        c = _completed_for_team(state, team)
        if not c:
            return 0.5
        wins = sum(1 for m in c if m.get("winner") == team)
        return wins / len(c)
    def sos(team):
        sched = _completed_for_team(state, team, season=season)
        if not sched:
            return 0.0
        opps = [(m["team2"] if m["team1"] == team else m["team1"]) for m in sched]
        return float(np.mean([team_career_wr(o) for o in opps])) - 0.5
    return sos(t1) - sos(t2)


FEATURE_FUNCS = {
    "form_last5_diff": feat_form_last5_wins,
    "season_nrr_diff": feat_season_nrr_diff,
    "squad_strength_diff": feat_squad_strength_proxy,
    "home_advantage_diff": feat_home_advantage,
    "h2h_season_diff": feat_h2h_season,
    "venue_chase_rate": feat_venue_chase_rate,
    "pom_recency_diff": feat_pom_recency,
    "qualified_flag_diff": feat_qualified_flag,
    "strength_of_schedule_diff": feat_strength_of_schedule,
}


def build_features(match_row: dict, state: dict) -> dict:
    return {name: float(fn(match_row, state)) for name, fn in FEATURE_FUNCS.items()}


def build_features_dataset(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Walk matches chronologically and produce a feature row per match using
    point-in-time state. Same logic as backtest harness, but returns features."""
    df = matches_df.copy()
    df = df.dropna(subset=["date", "team1", "team2", "season"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    state = {
        "completed": [],
        "h2h": {},
        "team_form": {},
        "team_played": {},
        "team_won": {},
        "venue_history": {},
    }

    rows = []
    for _, row in df.iterrows():
        if row.get("no_result", False):
            state["completed"].append({
                "match_id": row["match_id"], "date": row["date"], "season": int(row["season"]),
                "team1": row["team1"], "team2": row["team2"], "winner": None, "no_result": True,
            })
            continue
        if pd.isna(row.get("winner")):
            continue
        feats = build_features(row.to_dict(), state)
        feats.update({
            "match_id": row["match_id"],
            "date": row["date"],
            "season": int(row["season"]),
            "team1": row["team1"],
            "team2": row["team2"],
            "winner": row["winner"],
            "winner_is_team1": int(row["winner"] == row["team1"]),
        })
        rows.append(feats)
        state["completed"].append({
            "match_id": row["match_id"], "date": row["date"], "season": int(row["season"]),
            "team1": row["team1"], "team2": row["team2"], "winner": row["winner"],
            "venue": row.get("venue", ""), "no_result": False,
        })
        a, b = sorted([row["team1"], row["team2"]])
        key = (a, b, int(row["season"]))
        ent = state["h2h"].setdefault(key, {a: 0, b: 0})
        if row["winner"] in ent:
            ent[row["winner"]] += 1
        v = row.get("venue", "")
        state["venue_history"].setdefault(v, []).append({
            "winner": row["winner"], "team1": row["team1"], "team2": row["team2"], "date": row["date"],
        })

    out = pd.DataFrame(rows)
    # Split column for downstream training
    out["split"] = "train"
    out.loc[out["season"] == 2024, "split"] = "val"
    out.loc[out["season"] == 2025, "split"] = "test"
    return out
