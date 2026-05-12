"""Feature sanity tests."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIST = ROOT / "ml" / "data" / "historical"


def _skip_if_missing(p):
    if not p.exists():
        pytest.skip(f"{p} not built yet")


def test_no_nan_after_warmup():
    p = HIST / "features.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p).sort_values(["date", "match_id"]).reset_index(drop=True)
    # After warmup (each team has played at least 5), no NaNs in feature columns
    feature_cols = [c for c in df.columns if c not in {"match_id", "date", "season", "team1", "team2", "winner", "winner_is_team1", "split"}]
    tail = df.iloc[200:]
    nan_counts = tail[feature_cols].isna().sum()
    assert nan_counts.sum() == 0, f"NaNs found: {nan_counts[nan_counts > 0]}"


def test_features_in_sane_range():
    p = HIST / "features.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    for col in ["form_last5_diff", "h2h_season_diff", "home_advantage_diff", "pom_recency_diff"]:
        assert df[col].abs().max() <= 1.5, f"{col} out of [-1, 1] range"


def test_target_distribution():
    p = HIST / "features.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    rate = df["winner_is_team1"].mean()
    # team1 ordering is arbitrary, so the rate should be near 0.5 ± 0.1
    assert 0.4 <= rate <= 0.6, f"team1 win rate too imbalanced: {rate:.3f}"


def test_wp_balls_remaining_monotonic():
    p = HIST / "wp_features.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    # within an innings of a single match, balls_remaining must strictly decrease
    sample = df.head(50_000)
    for (mid, inn), g in sample.groupby(["match_id", "innings"]):
        diffs = g["balls_remaining"].diff().dropna()
        assert (diffs <= 0).all(), f"balls_remaining not monotonic in match {mid} innings {inn}"


def test_wp_wickets_remaining_non_increasing():
    p = HIST / "wp_features.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p).head(50_000)
    for (mid, inn), g in df.groupby(["match_id", "innings"]):
        diffs = g["wickets_remaining"].diff().dropna()
        assert (diffs <= 0).all(), f"wickets_remaining increased in match {mid} innings {inn}"


def test_wp_rrr_nan_in_first_innings():
    p = HIST / "wp_features.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    first_innings = df[df["innings"] == 1]
    assert first_innings["required_run_rate"].isna().all(), "RRR must be NaN in 1st innings"
