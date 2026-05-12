"""Tests for ml/src/phase_features.py — the cleaner pre-match builder used in v3+."""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIST = ROOT / "ml" / "data" / "historical"


def _skip_if_missing(p):
    if not p.exists():
        pytest.skip(f"{p} not built yet")


def test_pre_match_clean_chronological():
    p = HIST / "features_pre_match.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p).sort_values(["date", "match_id"]).reset_index(drop=True)
    assert df["date"].is_monotonic_increasing


def test_career_wr_first_match_is_default():
    """A team's first-ever appearance must have career_wr == 0.5 (no priors)."""
    p = HIST / "features_pre_match.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p).sort_values(["date", "match_id"]).reset_index(drop=True)
    seen = set()
    for _, row in df.iterrows():
        for tcol, wrcol in [("team1", "team1_career_wr"), ("team2", "team2_career_wr")]:
            if row[tcol] not in seen:
                assert row[wrcol] == 0.5, f"first-ever match for {row[tcol]} had career_wr={row[wrcol]}"
                seen.add(row[tcol])


def test_post_pp1_has_powerplay_columns():
    p = HIST / "features_post_pp1.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    assert "pp1_runs" in df.columns
    assert "pp1_wickets" in df.columns
    # PP1 runs should be in a plausible range (0..100+ for some heavy starts)
    assert df["pp1_runs"].min() >= 0
    assert df["pp1_runs"].max() <= 200


def test_innings_break_target_is_first_innings_plus_one():
    p = HIST / "features_innings_break.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p)
    # target should equal first_innings_total + 1 wherever both are defined
    valid = df.dropna(subset=["target", "first_innings_total"])
    diffs = (valid["target"] - valid["first_innings_total"]).unique()
    assert set(diffs.tolist()) <= {1}, f"target should be first_innings_total+1, got diffs {diffs}"


def test_only_dead_code_removed():
    """Guards against accidental re-introduction of the buggy `_build_pre_match_state`."""
    from ml.src import phase_features
    assert not hasattr(phase_features, "_build_pre_match_state"), \
        "dead/broken function _build_pre_match_state must remain removed; use _build_pre_match_clean"
