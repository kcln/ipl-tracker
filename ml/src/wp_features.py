"""Per-delivery features for the live win probability model.

For each delivery in balls.parquet, produce a feature row containing the
state-as-of-that-delivery (NOT the outcome of that delivery, which would leak).

Target: did the batting team win the match?
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIST = ROOT / "ml" / "data" / "historical"

T20_TOTAL_BALLS = 120
T20_WICKETS = 10


def _team_career_winrate_through(matches: pd.DataFrame) -> dict:
    """Map of team -> (wins_so_far, played_so_far) cumulative, ordered by date.

    Returns a dict keyed by match_id: at the time this match starts,
    {team: (wins_so_far, played_so_far)} for both teams in the match.
    """
    matches = matches.dropna(subset=["winner"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    by_match = {}
    counts: dict[str, list[int]] = {}  # team -> [wins, played]
    for _, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        # snapshot BEFORE this match
        by_match[row["match_id"]] = {
            t1: tuple(counts.get(t1, [0, 0])),
            t2: tuple(counts.get(t2, [0, 0])),
        }
        # then update
        winner = row["winner"]
        for t in (t1, t2):
            counts.setdefault(t, [0, 0])
        counts[winner][0] += 1
        counts[t1][1] += 1
        counts[t2][1] += 1
    return by_match


def build_wp_features(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Build per-delivery feature rows for the win-probability model."""
    # team strength snapshot per match (career win-rate going in)
    print("computing pre-match team strengths...", flush=True)
    strength = _team_career_winrate_through(matches)

    # match-level info we need lookup-able
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    m_index = matches.set_index("match_id")

    # Determine the batting team for each match's 1st innings and 2nd innings,
    # and compute 1st-innings final total so we have a chase target.
    print("computing per-innings totals...", flush=True)
    per_innings_runs = (
        balls.groupby(["match_id", "innings"])["total_runs"].sum().unstack(fill_value=0)
    )

    print("scanning deliveries...", flush=True)
    out_rows = []
    # group balls by match for sequential walk
    for match_id, mballs in balls.groupby("match_id", sort=False):
        if match_id not in m_index.index:
            continue
        m = m_index.loc[match_id]
        if pd.isna(m["winner"]) or m.get("no_result", False):
            continue
        winner = m["winner"]

        # 1st innings final score
        first_innings_total = int(per_innings_runs.loc[match_id, 1]) if 1 in per_innings_runs.columns else None
        if first_innings_total is None:
            continue

        # iterate deliveries in order
        mballs = mballs.sort_values(["innings", "over", "ball"]).reset_index(drop=True)
        # Compute per-innings cumulative score, wickets, balls bowled
        cum_runs = 0
        wkts = 0
        balls_bowled = 0
        last_innings = None
        recent_runs = []  # rolling window of last 30 balls

        for _, b in mballs.iterrows():
            innings = int(b["innings"])
            if innings not in (1, 2):
                continue  # ignore super overs for WP
            if innings != last_innings:
                cum_runs = 0
                wkts = 0
                balls_bowled = 0
                recent_runs = []
                last_innings = innings

            # State BEFORE this delivery is what the model sees
            balls_remaining = T20_TOTAL_BALLS - balls_bowled
            wickets_remaining = T20_WICKETS - wkts
            current_score = cum_runs
            crr = (cum_runs / balls_bowled * 6.0) if balls_bowled > 0 else 0.0
            target = first_innings_total + 1 if innings == 2 else None
            if innings == 2:
                needed = max(0, target - cum_runs)
                rrr = (needed / balls_remaining * 6.0) if balls_remaining > 0 else np.nan
            else:
                rrr = np.nan
            last_30 = sum(recent_runs[-30:]) / max(1, min(30, len(recent_runs))) * 6.0 if recent_runs else 0.0
            batting_team = b["batting_team"]
            bowling_team = (m["team2"] if batting_team == m["team1"] else m["team1"])
            bs = strength.get(match_id, {}).get(batting_team, (0, 0))
            bb = strength.get(match_id, {}).get(bowling_team, (0, 0))
            batting_team_strength = (bs[0] / bs[1]) if bs[1] > 0 else 0.5
            bowling_team_strength = (bb[0] / bb[1]) if bb[1] > 0 else 0.5

            out_rows.append({
                "match_id": match_id,
                "season": int(m["season"]) if pd.notna(m["season"]) else None,
                "innings": innings,
                "ball_no_in_innings": balls_bowled,
                "balls_remaining": balls_remaining,
                "wickets_remaining": wickets_remaining,
                "current_score": current_score,
                "current_run_rate": crr,
                "required_run_rate": rrr,
                "target": target,
                "last_30_balls_rr": last_30,
                "batting_team_strength": batting_team_strength,
                "bowling_team_strength": bowling_team_strength,
                "is_chase": 1 if innings == 2 else 0,
                "venue": m.get("venue", ""),
                "batting_team_won": 1 if batting_team == winner else 0,
            })

            # Update cum AFTER recording the feature row
            cum_runs += int(b["total_runs"])
            balls_bowled += 1
            if b.get("wicket_kind") and b["wicket_kind"] not in {"retired hurt", "retired out"}:
                wkts += 1
            recent_runs.append(int(b["total_runs"]))

    out = pd.DataFrame(out_rows)
    print(f"built {len(out)} delivery rows", flush=True)
    return out


def main():
    matches = pd.read_parquet(HIST / "matches.parquet")
    balls = pd.read_parquet(HIST / "balls.parquet")
    matches = matches[matches["season"].between(2008, 2025)].copy()
    balls = balls[balls["match_id"].isin(matches["match_id"])]

    wp = build_wp_features(balls, matches)
    # split column
    wp["split"] = "train"
    wp.loc[wp["season"] == 2024, "split"] = "val"
    wp.loc[wp["season"] == 2025, "split"] = "test"

    import pyarrow as pa, pyarrow.parquet as pq, os
    out = HIST / "wp_features.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(wp, preserve_index=False), tmp)
    os.replace(tmp, out)
    print(f"wrote {out} (rows={len(wp)})")


if __name__ == "__main__":
    main()
