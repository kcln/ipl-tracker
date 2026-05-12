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
