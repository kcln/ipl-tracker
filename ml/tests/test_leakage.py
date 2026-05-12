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
    """For 10 sampled matches, recompute every feature using ONLY prior matches
    and assert it matches the stored value. This catches any future-data leak
    in any feature function. We match `build_features_dataset`'s exact ordering
    — i.e. same-date matches with smaller match_id count as prior (this is the
    known same-day class flagged by run_backtest's `same_day_warnings`)."""
    from ml.src.features import FEATURE_FUNCS

    matches_path = HIST / "matches.parquet"
    feats_path = HIST / "features.parquet"
    _skip_if_missing(matches_path)
    _skip_if_missing(feats_path)

    matches = pd.read_parquet(matches_path)
    matches = matches.dropna(subset=["date", "team1", "team2", "season"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    feats = pd.read_parquet(feats_path).sort_values(["date", "match_id"]).reset_index(drop=True)
    # features.parquet doesn't carry venue/toss columns — join them back from matches.
    feats = feats.merge(matches[["match_id", "venue", "toss_winner", "toss_decision"]], on="match_id", how="left")

    n = len(feats)
    sample_idxs = [int(n * frac) for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]]
    failures = []
    for sample_idx in sample_idxs:
        row = feats.iloc[sample_idx]
        # Match build_features_dataset's "prior" semantics: same-date matches
        # with strictly smaller match_id count as prior (chronological tie-break).
        prior = matches[
            (matches["date"] < row["date"])
            | ((matches["date"] == row["date"]) & (matches["match_id"] < row["match_id"]))
        ]
        state = {"completed": [], "h2h": {}, "venue_history": {}, "team_form": {}, "team_played": {}, "team_won": {}}
        for _, m in prior.iterrows():
            if bool(m.get("no_result", False)):
                state["completed"].append({
                    "match_id": m["match_id"], "date": m["date"], "season": int(m["season"]),
                    "team1": m["team1"], "team2": m["team2"], "winner": None, "no_result": True,
                })
                continue
            if pd.isna(m.get("winner")):
                continue
            state["completed"].append({
                "match_id": m["match_id"], "date": m["date"], "season": int(m["season"]),
                "team1": m["team1"], "team2": m["team2"], "winner": m["winner"],
                "venue": m.get("venue", ""), "no_result": False,
            })
            w = m["winner"]
            a, b = sorted([m["team1"], m["team2"]])
            key = (a, b, int(m["season"]))
            ent = state["h2h"].setdefault(key, {a: 0, b: 0})
            if w in ent: ent[w] += 1
            state["venue_history"].setdefault(m.get("venue", ""), []).append({
                "winner": w, "team1": m["team1"], "team2": m["team2"], "date": m["date"],
            })

        for name, fn in FEATURE_FUNCS.items():
            stored = float(row[name])
            recomputed = float(fn(row.to_dict(), state))
            if abs(recomputed - stored) > 1e-6:
                failures.append(f"match_idx={sample_idx} feature={name}: stored={stored:.6f} recomputed={recomputed:.6f}")
    assert not failures, "PIT violations:\n" + "\n".join(failures)


def test_same_day_warnings_counted():
    """Same-day matches in cricsheet (doubleheaders) should be counted in
    same_day_warnings even though they don't trigger leakage_detected."""
    from ml.src.backtest import run_backtest

    p = HIST / "matches.parquet"
    _skip_if_missing(p)
    matches = pd.read_parquet(p)
    def naive(match_row, _state):
        return match_row["team1"], 0.5
    summary = run_backtest(naive, matches, name="same_day_check")
    assert "same_day_warnings" in summary
    # IPL has many doubleheaders; we expect non-zero
    assert summary["same_day_warnings"] > 0, "expected same-day matches in IPL history"
    assert not summary["leakage_detected"]
