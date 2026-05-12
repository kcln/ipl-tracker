"""Smoke tests for ml/src/player_features.py."""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIST = ROOT / "ml" / "data" / "historical"


def _skip_if_missing(p):
    if not p.exists():
        pytest.skip(f"{p} not built yet")


def test_player_features_built_for_each_match():
    p = HIST / "player_features.parquet"
    matches_p = HIST / "matches.parquet"
    _skip_if_missing(p); _skip_if_missing(matches_p)
    pf = pd.read_parquet(p)
    matches = pd.read_parquet(matches_p)
    matches = matches[matches["season"].between(2008, 2025) & matches["winner"].notna()]
    assert len(pf) == len(matches), f"player_features has {len(pf)} rows, expected {len(matches)}"


def test_top5_batter_sr_in_plausible_range():
    p = HIST / "player_features.parquet"
    _skip_if_missing(p)
    pf = pd.read_parquet(p)
    for col in ("team1_top5_batter_sr_mean", "team2_top5_batter_sr_mean"):
        v = pf[col].dropna()
        assert v.min() >= 50.0, f"{col} too low: {v.min()}"
        assert v.max() <= 250.0, f"{col} too high: {v.max()}"


def test_top4_bowler_econ_in_plausible_range():
    p = HIST / "player_features.parquet"
    _skip_if_missing(p)
    pf = pd.read_parquet(p)
    for col in ("team1_top4_bowler_econ_mean", "team2_top4_bowler_econ_mean"):
        v = pf[col].dropna()
        assert v.min() >= 3.0, f"{col} too low: {v.min()}"
        assert v.max() <= 12.0, f"{col} too high: {v.max()}"


def test_team_recent_lineup_uses_positional_indexing():
    """Smoke test that _team_recent_lineup handles a non-contiguous index."""
    from ml.src.player_features import _team_recent_lineup

    m = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "match_id": ["m1", "m2", "m3"],
        "team1": ["A", "A", "B"],
        "team2": ["B", "C", "A"],
    })
    balls = {
        "m1": pd.DataFrame({"batter": ["p1", "p2"], "bowler": ["p3", "p4"]}),
        "m2": pd.DataFrame({"batter": ["p5"], "bowler": ["p6"]}),
    }
    batters, bowlers = _team_recent_lineup(m, balls, "A", "m3", n_matches=3)
    # team A's prior matches are m1 (idx 0) and m2 (idx 1) — should pick up players from both
    assert set(batters) == {"p1", "p2", "p5"}
    assert set(bowlers) == {"p3", "p4", "p6"}
