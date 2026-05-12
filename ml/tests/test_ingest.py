"""Smoke tests for ingest output. Run only after `python -m ml.src.ingest`."""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIST = ROOT / "ml" / "data" / "historical"


def _skip_if_missing(p):
    if not p.exists():
        pytest.skip(f"{p} not built yet; run `python -m ml.src.ingest` first")


def test_matches_count_reasonable():
    p = HIST / "matches.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    assert len(df) >= 1000, f"only {len(df)} matches"


def test_distinct_teams_bounded():
    p = HIST / "matches.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    teams = pd.unique(df[["team1", "team2"]].values.ravel("K"))
    assert len(teams) <= 20, f"too many distinct teams: {sorted(teams)}"


def test_null_winners_only_no_result():
    p = HIST / "matches.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    null_winner_mask = df["winner"].isna()
    if null_winner_mask.any():
        assert df.loc[null_winner_mask, "no_result"].all(), \
            "null winners exist on matches that are not flagged no_result"


def test_team_rename_applied():
    p = HIST / "matches.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    teams = set(pd.unique(df[["team1", "team2"]].values.ravel("K")))
    assert "Kings XI Punjab" not in teams, "Kings XI Punjab should be renamed to Punjab Kings"
    assert "Delhi Daredevils" not in teams, "Delhi Daredevils should be renamed to Delhi Capitals"


def test_balls_parquet_rows():
    p = HIST / "balls.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    assert len(df) >= 200_000, f"only {len(df)} balls"
