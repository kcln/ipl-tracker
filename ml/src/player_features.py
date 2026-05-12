"""Phase 4: player-level point-in-time aggregates per match.

For each match and team, we compute:
  - avg career strike rate of the team's most recent 11 distinct batters
  - avg career economy of the team's most recent 6 distinct bowlers
  - top recent-30-day form of the team's batters / bowlers

All stats are PIT — only data from balls BEFORE the match's date is used.
"""
from __future__ import annotations

import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIST = ROOT / "ml" / "data" / "historical"


def _atomic_write(df, path):
    import pyarrow as pa, pyarrow.parquet as pq, os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp)
    os.replace(tmp, path)


def _pit_balls(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Annotate every ball with PIT cumulative stats for batter and bowler."""
    b = balls.merge(matches[["match_id", "date"]], on="match_id", how="left")
    b = b.sort_values(["date", "match_id", "innings", "over", "ball"]).reset_index(drop=True)

    # Batting PIT cumulatives — shift by one so "so_far" excludes the current ball.
    b["bat_runs_so_far"] = b.groupby("batter")["runs_off_bat"].cumsum() - b["runs_off_bat"]
    b["bat_balls_so_far"] = b.groupby("batter").cumcount()
    b["bat_sr_so_far"] = b["bat_runs_so_far"] / b["bat_balls_so_far"].clip(lower=1) * 100.0

    # Bowling PIT cumulatives
    wkt_mask = b["wicket_kind"].notna() & ~b["wicket_kind"].isin(
        ["run out", "retired hurt", "retired out", "obstructing the field"]
    )
    b["_wkt"] = wkt_mask.astype(int)
    b["bowl_runs_so_far"] = b.groupby("bowler")["total_runs"].cumsum() - b["total_runs"]
    b["bowl_balls_so_far"] = b.groupby("bowler").cumcount()
    b["bowl_wkts_so_far"] = b.groupby("bowler")["_wkt"].cumsum() - b["_wkt"]
    b["bowl_econ_so_far"] = b["bowl_runs_so_far"] / b["bowl_balls_so_far"].clip(lower=1) * 6.0
    return b


def _team_recent_lineup(matches_seq: pd.DataFrame, balls_indexed: dict, team: str, before_match_id: str, n_matches: int = 3) -> tuple[list[str], list[str]]:
    """Find the (batters, bowlers) for a team across its previous n_matches before before_match_id.
    matches_seq must be sorted ascending by date.
    """
    team_matches = matches_seq[(matches_seq["team1"] == team) | (matches_seq["team2"] == team)]
    idx = team_matches.index[team_matches["match_id"] == before_match_id]
    if len(idx) == 0:
        return [], []
    pos = idx[0]
    prev = team_matches.loc[:pos - 1].tail(n_matches) if pos > 0 else team_matches.iloc[0:0]

    batters = []
    bowlers = []
    seen_b, seen_w = set(), set()
    for _, m in prev.iterrows():
        mid = m["match_id"]
        bdf = balls_indexed.get(mid)
        if bdf is None:
            continue
        for who in bdf["batter"].dropna().unique():
            if who not in seen_b:
                seen_b.add(who)
                batters.append(who)
        for who in bdf["bowler"].dropna().unique():
            if who not in seen_w:
                seen_w.add(who)
                bowlers.append(who)
    return batters, bowlers


def build_player_features(matches: pd.DataFrame, balls_pit: pd.DataFrame) -> pd.DataFrame:
    """Per-match team-level player aggregates."""
    matches = matches.dropna(subset=["winner"]).sort_values(["date", "match_id"]).reset_index(drop=True)

    # Build a player -> (date, sr_so_far) timeline using last-balls-faced entry per date
    # We want for each player, latest career stats strictly before a given date.
    # Approach: for each player, take the LAST row per (player, date) — that gives end-of-day stats.
    # Then for any query date D, we want the entry with the latest date < D.

    # Build sorted lookup per player for batting
    bat = balls_pit[balls_pit["batter"].notna()][["batter", "date", "bat_runs_so_far", "bat_balls_so_far", "bat_sr_so_far"]].copy()
    bat = bat.groupby(["batter", "date"]).tail(1).reset_index(drop=True)
    bat = bat.sort_values(["batter", "date"]).reset_index(drop=True)
    bat_by_player = {p: g.reset_index(drop=True) for p, g in bat.groupby("batter")}

    bowl = balls_pit[balls_pit["bowler"].notna()][["bowler", "date", "bowl_runs_so_far", "bowl_balls_so_far", "bowl_wkts_so_far", "bowl_econ_so_far"]].copy()
    bowl = bowl.groupby(["bowler", "date"]).tail(1).reset_index(drop=True)
    bowl = bowl.sort_values(["bowler", "date"]).reset_index(drop=True)
    bowl_by_player = {p: g.reset_index(drop=True) for p, g in bowl.groupby("bowler")}

    # Index balls by match for lineup lookup
    balls_by_match = {mid: g for mid, g in balls_pit.groupby("match_id")}

    def stats_before(player_dict, player, date):
        g = player_dict.get(player)
        if g is None or len(g) == 0:
            return None
        # binary search by date
        idx = g["date"].searchsorted(date, side="left")
        if idx == 0:
            return None
        return g.iloc[idx - 1]

    rows = []
    matches_seq = matches.copy().reset_index(drop=True)
    for _, m in matches_seq.iterrows():
        mid, date = m["match_id"], m["date"]
        out = {"match_id": mid, "date": date, "season": m["season"]}
        for team_key, team in (("team1", m["team1"]), ("team2", m["team2"])):
            batters, bowlers = _team_recent_lineup(matches_seq, balls_by_match, team, mid, n_matches=3)

            bat_srs, bat_balls = [], []
            for p in batters:
                s = stats_before(bat_by_player, p, date)
                if s is not None and s["bat_balls_so_far"] >= 30:  # min sample to be meaningful
                    bat_srs.append(float(s["bat_sr_so_far"]))
                    bat_balls.append(int(s["bat_balls_so_far"]))
            bowl_econs, bowl_wkts = [], []
            for p in bowlers:
                s = stats_before(bowl_by_player, p, date)
                if s is not None and s["bowl_balls_so_far"] >= 60:
                    bowl_econs.append(float(s["bowl_econ_so_far"]))
                    bowl_wkts.append(int(s["bowl_wkts_so_far"]))

            # top-N aggregates
            if bat_srs:
                top_bat = sorted(bat_srs, reverse=True)[:5]
                out[f"{team_key}_top5_batter_sr_mean"] = float(np.mean(top_bat))
                out[f"{team_key}_top5_batter_sr_max"] = float(np.max(top_bat))
                out[f"{team_key}_n_batters_qual"] = int(len(bat_srs))
            else:
                out[f"{team_key}_top5_batter_sr_mean"] = float("nan")
                out[f"{team_key}_top5_batter_sr_max"] = float("nan")
                out[f"{team_key}_n_batters_qual"] = 0

            if bowl_econs:
                top_bowl = sorted(bowl_econs)[:4]  # lower is better
                out[f"{team_key}_top4_bowler_econ_mean"] = float(np.mean(top_bowl))
                out[f"{team_key}_top4_bowler_econ_min"] = float(np.min(top_bowl))
                out[f"{team_key}_n_bowlers_qual"] = int(len(bowl_econs))
            else:
                out[f"{team_key}_top4_bowler_econ_mean"] = float("nan")
                out[f"{team_key}_top4_bowler_econ_min"] = float("nan")
                out[f"{team_key}_n_bowlers_qual"] = 0

        rows.append(out)

    return pd.DataFrame(rows)


def main():
    matches = pd.read_parquet(HIST / "matches.parquet")
    balls = pd.read_parquet(HIST / "balls.parquet")
    matches = matches[matches["season"].between(2008, 2025)].copy()
    matches["date"] = pd.to_datetime(matches["date"])
    balls = balls[balls["match_id"].isin(matches["match_id"])].copy()

    print("Building PIT player stats...", flush=True)
    balls_pit = _pit_balls(balls, matches)
    print(f"  annotated {len(balls_pit)} balls", flush=True)

    print("Building per-match team aggregates...", flush=True)
    pf = build_player_features(matches, balls_pit)
    pf["date"] = pd.to_datetime(pf["date"])
    _atomic_write(pf, HIST / "player_features.parquet")
    print(f"wrote player_features.parquet ({len(pf)})", flush=True)


if __name__ == "__main__":
    main()
