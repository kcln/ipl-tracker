"""Phase 3: build 4 per-phase feature parquets (pre-match, post-toss, post-PP1,
innings-break) using point-in-time state plus phase-specific signals."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIST = ROOT / "ml" / "data" / "historical"


def _atomic_write(df: pd.DataFrame, path: pathlib.Path):
    import pyarrow as pa, pyarrow.parquet as pq, os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp)
    os.replace(tmp, path)


def _build_pre_match_clean(matches: pd.DataFrame) -> pd.DataFrame:
    """Cleaner re-implementation of pre-match PIT features."""
    df = matches.sort_values(["date", "match_id"]).reset_index(drop=True).copy()
    career_w, career_p = {}, {}
    season_w, season_p = {}, {}
    last5 = {}
    venue_team_w, venue_team_p = {}, {}
    venue_chase_w, venue_p = {}, {}

    def wr(w, p, k, default=0.5):
        if p.get(k, 0) == 0:
            return default
        return w.get(k, 0) / p[k]

    rows = []
    for _, m in df.iterrows():
        t1, t2 = m["team1"], m["team2"]
        season = int(m["season"]) if pd.notna(m["season"]) else -1
        venue = m.get("venue", "")
        rows.append({
            "match_id": m["match_id"],
            "season": season,
            "date": m["date"],
            "venue": venue,
            "team1": t1,
            "team2": t2,
            "team1_career_wr": wr(career_w, career_p, t1),
            "team2_career_wr": wr(career_w, career_p, t2),
            "team1_career_played": career_p.get(t1, 0),
            "team2_career_played": career_p.get(t2, 0),
            "team1_season_wr": wr(season_w, season_p, (season, t1)),
            "team2_season_wr": wr(season_w, season_p, (season, t2)),
            "team1_season_played": season_p.get((season, t1), 0),
            "team2_season_played": season_p.get((season, t2), 0),
            "team1_form5": float(np.mean(last5.get(t1, [0.5])[-5:])) if last5.get(t1) else 0.5,
            "team2_form5": float(np.mean(last5.get(t2, [0.5])[-5:])) if last5.get(t2) else 0.5,
            "team1_venue_wr": wr(venue_team_w, venue_team_p, (venue, t1)),
            "team2_venue_wr": wr(venue_team_w, venue_team_p, (venue, t2)),
            "venue_chase_rate": (venue_chase_w.get(venue, 0) / venue_p[venue]) if venue_p.get(venue, 0) > 0 else 0.5,
            "venue_played": venue_p.get(venue, 0),
            "toss_winner": m.get("toss_winner"),
            "toss_decision": m.get("toss_decision"),
            "winner": m.get("winner"),
            "win_by_runs": m.get("win_by_runs"),
            "win_by_wickets": m.get("win_by_wickets"),
            "no_result": bool(m.get("no_result", False)),
        })
        if pd.notna(m["winner"]) and not m.get("no_result", False):
            w = m["winner"]
            for t in (t1, t2):
                career_p[t] = career_p.get(t, 0) + 1
                season_p[(season, t)] = season_p.get((season, t), 0) + 1
                venue_team_p[(venue, t)] = venue_team_p.get((venue, t), 0) + 1
                last5.setdefault(t, []).append(1 if t == w else 0)
            career_w[w] = career_w.get(w, 0) + 1
            season_w[(season, w)] = season_w.get((season, w), 0) + 1
            venue_team_w[(venue, w)] = venue_team_w.get((venue, w), 0) + 1
            venue_p[venue] = venue_p.get(venue, 0) + 1
            if pd.notna(m.get("win_by_wickets")):
                venue_chase_w[venue] = venue_chase_w.get(venue, 0) + 1
    return pd.DataFrame(rows)


def _add_post_toss(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["toss_decision_field"] = (df["toss_decision"] == "field").astype(int)
    return df


def _add_powerplay_signals(df: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """For each match, compute innings-1 PP1 runs/wickets (overs 0..5).
    These are observable after PP1 of innings 1 only; for post-PP1 model,
    a match enters the dataset only if innings 1 PP1 has completed."""
    pp1 = balls[balls["innings"] == 1].copy()
    pp1 = pp1[pp1["over"] <= 5]
    by_m = pp1.groupby("match_id").agg(
        pp1_runs=("total_runs", "sum"),
        pp1_wickets=("wicket_kind", lambda s: int((s.notna() & ~s.isin(["retired hurt", "retired out"])).sum())),
    ).reset_index()
    df = df.merge(by_m, on="match_id", how="left")
    df["pp1_runs"] = df["pp1_runs"].fillna(0)
    df["pp1_wickets"] = df["pp1_wickets"].fillna(0)
    return df


def _add_innings_break(df: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """At innings break: first innings total + first innings wickets."""
    inn1 = balls[balls["innings"] == 1].copy()
    by_m = inn1.groupby("match_id").agg(
        first_innings_total=("total_runs", "sum"),
        first_innings_balls=("ball", "count"),
        first_innings_wickets=("wicket_kind", lambda s: int((s.notna() & ~s.isin(["retired hurt", "retired out"])).sum())),
    ).reset_index()
    df = df.merge(by_m, on="match_id", how="left")
    # Required run rate at innings break for the chaser
    df["target"] = df["first_innings_total"] + 1
    df["innings_break_required_rr"] = df["target"] / 20.0
    return df


def main():
    matches = pd.read_parquet(HIST / "matches.parquet")
    balls = pd.read_parquet(HIST / "balls.parquet")

    # filter 2008-2025
    matches = matches[matches["season"].between(2008, 2025)].copy()
    balls = balls[balls["match_id"].isin(matches["match_id"])]

    pre = _build_pre_match_clean(matches)
    # drop no-result and null-winner rows for training
    pre_train = pre[(~pre["no_result"]) & pre["winner"].notna()].copy()
    pre_train["winner_is_team1"] = (pre_train["winner"] == pre_train["team1"]).astype(int)
    pre_train["split"] = "train"
    pre_train.loc[pre_train["season"] == 2024, "split"] = "val"
    pre_train.loc[pre_train["season"] == 2025, "split"] = "test"

    _atomic_write(pre_train, HIST / "features_pre_match.parquet")
    print(f"wrote features_pre_match.parquet ({len(pre_train)})", flush=True)

    post_toss = _add_post_toss(pre_train)
    _atomic_write(post_toss, HIST / "features_post_toss.parquet")
    print(f"wrote features_post_toss.parquet ({len(post_toss)})", flush=True)

    post_pp1 = _add_powerplay_signals(post_toss, balls)
    _atomic_write(post_pp1, HIST / "features_post_pp1.parquet")
    print(f"wrote features_post_pp1.parquet ({len(post_pp1)})", flush=True)

    inn_break = _add_innings_break(post_pp1, balls)
    _atomic_write(inn_break, HIST / "features_innings_break.parquet")
    print(f"wrote features_innings_break.parquet ({len(inn_break)})", flush=True)


if __name__ == "__main__":
    main()
