"""Cricsheet ingestion (Phase 0).

Downloads ipl_json.zip, extracts, and builds matches / balls / players parquets
with team-rename mapping.

Run:
    python -m ml.src.ingest          # uses cached zip if present
    python -m ml.src.ingest --force  # re-downloads zip (used by the daily launchd refresh)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"

ROOT = pathlib.Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ml"
RAW_DIR = ML_ROOT / "data" / "historical" / "raw_json"
HIST_DIR = ML_ROOT / "data" / "historical"

TEAM_RENAME = {
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


def _rename_team(name: str | None) -> str | None:
    if name is None:
        return None
    return TEAM_RENAME.get(name, name)


def _atomic_write_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def download_zip(force: bool = False) -> pathlib.Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ipl_json.zip"
    if zip_path.exists() and not force:
        return zip_path
    print(f"Downloading {CRICSHEET_URL} ...", flush=True)
    with requests.get(CRICSHEET_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = zip_path.with_suffix(".zip.tmp")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        os.replace(tmp, zip_path)
    print(f"Saved {zip_path}", flush=True)
    return zip_path


def extract_zip(zip_path: pathlib.Path) -> list[pathlib.Path]:
    out_dir = RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        for n in names:
            target = out_dir / pathlib.Path(n).name
            if target.exists():
                continue
            with z.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())
    json_files = sorted(out_dir.glob("*.json"))
    print(f"{len(json_files)} match JSONs available", flush=True)
    return json_files


def _parse_match(path: pathlib.Path) -> tuple[dict, list[dict]] | None:
    """Return (match_row, ball_rows). Returns None if match is not parseable."""
    with open(path) as f:
        data = json.load(f)

    info = data.get("info", {})
    event = info.get("event", {})
    season_raw = info.get("season")
    # cricsheet seasons can be like "2007/08" — normalise to 2008.
    if isinstance(season_raw, str) and "/" in season_raw:
        season = int("20" + season_raw.split("/")[1])
    else:
        try:
            season = int(season_raw)
        except (TypeError, ValueError):
            season = None
    dates = info.get("dates") or []
    date = dates[0] if dates else None
    teams = [_rename_team(t) for t in info.get("teams", [])]
    if len(teams) != 2:
        return None
    team1, team2 = teams[0], teams[1]

    venue = info.get("venue") or ""
    toss = info.get("toss", {}) or {}
    toss_winner = _rename_team(toss.get("winner"))
    toss_decision = toss.get("decision")

    outcome = info.get("outcome", {}) or {}
    winner = _rename_team(outcome.get("winner"))
    by = outcome.get("by", {}) or {}
    win_by_runs = by.get("runs")
    win_by_wickets = by.get("wickets")
    no_result = outcome.get("result") in ("no result", "tie")

    poms = info.get("player_of_match") or []
    pom = poms[0] if poms else None

    match_id = path.stem  # cricsheet filename is e.g. "1473461"

    match_row = {
        "match_id": match_id,
        "season": season,
        "date": date,
        "venue": venue,
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "winner": winner,
        "win_by_runs": win_by_runs,
        "win_by_wickets": win_by_wickets,
        "player_of_match": pom,
        "no_result": bool(no_result),
        "match_number": event.get("match_number"),
    }

    ball_rows: list[dict] = []
    innings_list = data.get("innings") or []
    for innings_num, inn in enumerate(innings_list, start=1):
        batting_team = _rename_team(inn.get("team"))
        for over in inn.get("overs", []) or []:
            over_num = over.get("over")
            for ball_idx, delivery in enumerate(over.get("deliveries", []) or [], start=1):
                runs_obj = delivery.get("runs", {}) or {}
                runs_off_bat = runs_obj.get("batter") or 0
                extras = runs_obj.get("extras") or 0
                wickets = delivery.get("wickets") or []
                wicket_kind = None
                player_out = None
                fielders: list[str] = []
                if wickets:
                    w0 = wickets[0]
                    wicket_kind = w0.get("kind")
                    player_out = w0.get("player_out")
                    fielders = [(f.get("name") or "") for f in (w0.get("fielders") or [])]
                ball_rows.append({
                    "match_id": match_id,
                    "innings": innings_num,
                    "batting_team": batting_team,
                    "over": over_num,
                    "ball": ball_idx,
                    "batter": delivery.get("batter"),
                    "bowler": delivery.get("bowler"),
                    "non_striker": delivery.get("non_striker"),
                    "runs_off_bat": runs_off_bat,
                    "extras": extras,
                    "total_runs": runs_off_bat + extras,
                    "wicket_kind": wicket_kind,
                    "player_out": player_out,
                    "fielders": fielders,
                })

    return match_row, ball_rows


def build_parquets(json_files: list[pathlib.Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_rows: list[dict] = []
    ball_rows: list[dict] = []
    t0 = time.time()
    for i, path in enumerate(json_files):
        parsed = _parse_match(path)
        if parsed is None:
            continue
        m, balls = parsed
        match_rows.append(m)
        ball_rows.extend(balls)
        if (i + 1) % 200 == 0:
            print(f"  parsed {i + 1}/{len(json_files)} in {time.time() - t0:.1f}s", flush=True)

    matches_df = pd.DataFrame(match_rows)
    balls_df = pd.DataFrame(ball_rows)

    matches_df["date"] = pd.to_datetime(matches_df["date"], errors="coerce")
    matches_df = matches_df.sort_values("date").reset_index(drop=True)

    _atomic_write_parquet(matches_df, HIST_DIR / "matches.parquet")
    _atomic_write_parquet(balls_df, HIST_DIR / "balls.parquet")
    print(f"Wrote matches.parquet ({len(matches_df)}) and balls.parquet ({len(balls_df)})", flush=True)
    return matches_df, balls_df


def build_players_rollup(balls_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Career rollups: runs, balls faced, strike rate, wickets taken, economy."""
    legal_runs = balls_df.copy()
    bat = legal_runs.groupby("batter", dropna=True).agg(
        career_runs=("runs_off_bat", "sum"),
        career_balls_faced=("ball", "count"),
    ).reset_index().rename(columns={"batter": "player"})
    bat["career_strike_rate"] = (bat["career_runs"] / bat["career_balls_faced"].clip(lower=1)) * 100.0

    wickets_mask = balls_df["wicket_kind"].notna() & ~balls_df["wicket_kind"].isin(["run out", "retired hurt", "retired out", "obstructing the field"])
    bowl_wkts = balls_df.loc[wickets_mask].groupby("bowler", dropna=True).size().rename("career_wickets").reset_index()
    bowl_runs = balls_df.groupby("bowler", dropna=True)["total_runs"].sum().rename("career_runs_conceded").reset_index()
    bowl_balls = balls_df.groupby("bowler", dropna=True).size().rename("career_balls_bowled").reset_index()
    bowl = bowl_runs.merge(bowl_balls, on="bowler", how="outer").merge(bowl_wkts, on="bowler", how="outer").rename(columns={"bowler": "player"})
    bowl["career_wickets"] = bowl["career_wickets"].fillna(0).astype(int)
    bowl["career_economy"] = (bowl["career_runs_conceded"] / (bowl["career_balls_bowled"].clip(lower=1) / 6.0))

    players = bat.merge(bowl, on="player", how="outer")
    for col in ["career_runs", "career_balls_faced", "career_runs_conceded", "career_balls_bowled", "career_wickets"]:
        if col in players.columns:
            players[col] = players[col].fillna(0)
    players = players.dropna(subset=["player"])

    _atomic_write_parquet(players, HIST_DIR / "players.parquet")
    print(f"Wrote players.parquet ({len(players)})", flush=True)
    return players


def main() -> None:
    ap = argparse.ArgumentParser(description="Cricsheet IPL ingestion")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download ipl_json.zip even if the cached copy exists (used by daily refresh).",
    )
    args = ap.parse_args()
    zip_path = download_zip(force=args.force)
    json_files = extract_zip(zip_path)
    matches_df, balls_df = build_parquets(json_files)
    build_players_rollup(balls_df, matches_df)


if __name__ == "__main__":
    main()
