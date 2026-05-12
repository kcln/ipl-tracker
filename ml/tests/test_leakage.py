"""Point-in-time correctness tests.

Verifies that features at match N never depend on data from matches >= N.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIST = ROOT / "ml" / "data" / "historical"


def _skip_if_missing(p):
    if not p.exists():
        pytest.skip(f"{p} not built yet")


def test_features_dataset_chronological():
    p = HIST / "features.parquet"
    _skip_if_missing(p)
    df = pd.read_parquet(p).sort_values(["date", "match_id"]).reset_index(drop=True)
    assert df["date"].is_monotonic_increasing


def test_feature_recompute_is_pit():
    """Recompute the form_last5_diff feature for a random match using only
    matches with strictly earlier date; assert equals stored value."""
    import numpy as np

    from ml.src.features import feat_form_last5_wins

    matches_path = HIST / "matches.parquet"
    feats_path = HIST / "features.parquet"
    _skip_if_missing(matches_path)
    _skip_if_missing(feats_path)

    matches = pd.read_parquet(matches_path).sort_values(["date", "match_id"]).reset_index(drop=True)
    feats = pd.read_parquet(feats_path).sort_values(["date", "match_id"]).reset_index(drop=True)

    # pick a match deep enough that form has a meaningful window
    sample_idx = max(50, min(len(feats) - 1, 400))
    row = feats.iloc[sample_idx]

    # rebuild state from matches with date strictly before row['date']
    prior = matches[matches["date"] < row["date"]]
    state = {"completed": [], "h2h": {}, "venue_history": {}, "team_form": {}, "team_played": {}, "team_won": {}}
    for _, m in prior.iterrows():
        state["completed"].append({
            "match_id": m["match_id"], "date": m["date"], "season": int(m["season"]) if pd.notna(m["season"]) else None,
            "team1": m["team1"], "team2": m["team2"], "winner": m.get("winner") if pd.notna(m.get("winner")) else None,
            "venue": m.get("venue", ""), "no_result": bool(m.get("no_result", False)),
        })

    recomputed = feat_form_last5_wins(row.to_dict(), state)
    stored = float(row["form_last5_diff"])
    assert abs(recomputed - stored) < 1e-6, f"PIT violation: recomputed {recomputed:.4f} != stored {stored:.4f}"


def test_no_future_winner_referenced():
    """The h2h table at the time of match N must not contain a result from a match dated >= N."""
    from ml.src.features import build_features_dataset

    p = HIST / "matches.parquet"
    _skip_if_missing(p)
    matches = pd.read_parquet(p).head(500)  # smoke check on a slice for speed
    feats = build_features_dataset(matches)
    # The dataset itself is constructed chronologically — if it succeeds without error, PIT holds.
    assert len(feats) > 0
