"""A1 — phase_player_stats.parquet

Per-match per-team aggregate of:
  - top-5 batters' strike rate by phase (pp / middle / death)
  - top-4 bowlers' economy by phase

PIT rules:
  - "Top 5 batters" / "top 4 bowlers" are identified from the *previous* match
    this team played (lineup proxy — actual XI for current match is unknown
    pre-match).
  - Each chosen player's phase SR / econ is their *career-cumulative* stat
    using only balls from matches strictly before the current match.
"""
from __future__ import annotations

import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd

from ml.src.v6._io import (
    HIST,
    V6_DIR,
    atomic_write_parquet,
    load_balls,
    load_matches,
    phase_of_over,
)

OUT = V6_DIR / "phase_player_stats.parquet"


def _annotate_balls(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Attach (date, season, phase) to balls, sort chronologically."""
    md = matches[["match_id", "date", "season"]].copy()
    b = balls.merge(md, on="match_id", how="inner")
    b["phase"] = b["over"].map(phase_of_over)
    # legal_ball flag — extras coded as wides/no-balls aren't legal deliveries
    # for SR/econ. cricsheet 'extras' is total extras count; we have no
    # 'extras_type'. Conservative: treat every recorded ball as legal — this
    # matches existing v5 player_features.py treatment which uses the same
    # balls.parquet without finer extras info.
    b = b.sort_values(["date", "match_id", "innings", "over", "ball"]).reset_index(drop=True)
    return b


def _build_team_lineup_map(balls: pd.DataFrame, matches: pd.DataFrame):
    """For each (match_id, team), return ordered list of batters (by balls faced
    desc) and bowlers (by balls bowled desc) for use as previous-match lineup proxy.
    """
    md = matches[["match_id", "team1", "team2"]]
    b = balls.merge(md, on="match_id", how="inner")
    # batters and their team = batting_team
    bat = (
        b.groupby(["match_id", "batting_team", "batter"]).size()
        .reset_index(name="balls_faced")
    )
    # bowlers: bowling team = the team that isn't batting
    b2 = b.copy()
    b2["bowling_team"] = np.where(b2["batting_team"] == b2["team1"], b2["team2"], b2["team1"])
    bowl = (
        b2.groupby(["match_id", "bowling_team", "bowler"]).size()
        .reset_index(name="balls_bowled")
    )
    return bat, bowl


def build() -> pd.DataFrame:
    matches = load_matches()
    balls = load_balls()
    balls = balls[balls["match_id"].isin(matches["match_id"])].copy()
    balls = _annotate_balls(balls, matches)

    # Build per-(match, team) lineup tables
    bat_per_match, bowl_per_match = _build_team_lineup_map(balls, matches)

    # ------ Cumulative PIT per-player phase stats ------
    # Iterate matches in chronological order. For each match M:
    #   1) emit the current cumulative stats per player.
    #   2) THEN update cumulative stats with M's balls.
    # We'll store, after processing matches 0..k-1, dicts that map player -> phase totals.

    # We need, per match, a snapshot of per-player stats BEFORE that match.
    # To avoid storing 1146 dict copies, we compute lineup-aggregates on the fly:
    # for each match, look up "team's previous match" -> its top players ->
    # current dict values -> aggregate -> append row.

    bat_runs = defaultdict(lambda: defaultdict(int))   # bat_runs[player][phase] = runs
    bat_balls = defaultdict(lambda: defaultdict(int))  # bat_balls[player][phase] = balls
    bowl_runs = defaultdict(lambda: defaultdict(int))  # runs conceded
    bowl_balls = defaultdict(lambda: defaultdict(int))

    # Index lineup tables by match for O(1) lookups
    bat_idx = {
        (mid, team): sub.sort_values("balls_faced", ascending=False)["batter"].tolist()
        for (mid, team), sub in bat_per_match.groupby(["match_id", "batting_team"])
    }
    bowl_idx = {
        (mid, team): sub.sort_values("balls_bowled", ascending=False)["bowler"].tolist()
        for (mid, team), sub in bowl_per_match.groupby(["match_id", "bowling_team"])
    }

    # Track each team's most recent match id (chronologically)
    last_match_for_team: dict[str, str] = {}

    rows = []
    matches_sorted = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    # Pre-group balls by match for fast incremental updates
    balls_by_match = {mid: sub for mid, sub in balls.groupby("match_id", sort=False)}

    def agg_sr(players, phase):
        if not players:
            return np.nan
        runs = sum(bat_runs[p][phase] for p in players)
        bf = sum(bat_balls[p][phase] for p in players)
        if bf == 0:
            return np.nan
        return 100.0 * runs / bf

    def agg_econ(players, phase):
        if not players:
            return np.nan
        runs = sum(bowl_runs[p][phase] for p in players)
        bb = sum(bowl_balls[p][phase] for p in players)
        if bb == 0:
            return np.nan
        return 6.0 * runs / bb  # runs per over

    for _, m in matches_sorted.iterrows():
        mid = m["match_id"]
        t1, t2 = m["team1"], m["team2"]

        feats = {"match_id": mid}
        for label, team in (("team1", t1), ("team2", t2)):
            prev_mid = last_match_for_team.get(team)
            top_bats = bat_idx.get((prev_mid, team), [])[:5] if prev_mid else []
            top_bowls = bowl_idx.get((prev_mid, team), [])[:4] if prev_mid else []
            for phase in ("pp", "middle", "death"):
                feats[f"{label}_top5_{phase}_sr_mean"] = agg_sr(top_bats, phase)
                feats[f"{label}_top4_{phase}_econ_mean"] = agg_econ(top_bowls, phase)
        rows.append(feats)

        # Update cumulative stats with this match's balls (so future matches see them)
        sub = balls_by_match.get(mid)
        if sub is not None and len(sub):
            # Bat updates
            for phase, group in sub.groupby("phase"):
                # runs off bat per batter
                runs_g = group.groupby("batter")["runs_off_bat"].sum()
                balls_g = group.groupby("batter").size()
                for p, r in runs_g.items():
                    bat_runs[p][phase] += int(r)
                for p, c in balls_g.items():
                    bat_balls[p][phase] += int(c)
                # bowler: runs conceded = total_runs per ball (includes extras),
                # balls bowled = count
                bowl_runs_g = group.groupby("bowler")["total_runs"].sum()
                bowl_balls_g = group.groupby("bowler").size()
                for p, r in bowl_runs_g.items():
                    bowl_runs[p][phase] += int(r)
                for p, c in bowl_balls_g.items():
                    bowl_balls[p][phase] += int(c)

        last_match_for_team[t1] = mid
        last_match_for_team[t2] = mid

    df = pd.DataFrame(rows)
    return df


def main():
    df = build()
    atomic_write_parquet(df, OUT)
    print(f"[A1] phase_player_stats: rows={len(df)} cols={len(df.columns)} -> {OUT}")


if __name__ == "__main__":
    main()
