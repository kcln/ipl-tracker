"""Parallel ML predict surface for the ml/ namespace.

Routes to the appropriate calibrated model based on the match stage:
  pre_match -> v3_phase_pre_match
  post_toss -> v3_phase_post_toss
  post_pp1  -> v3_phase_post_pp1
  innings_break -> v3_phase_innings_break

This does NOT replace the existing src/predictor.py used by the live tracker.
"""
from __future__ import annotations

import pathlib
import pickle
from functools import lru_cache
from typing import Any

import numpy as np

from ml.src.features import FEATURE_FUNCS, build_features

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "ml" / "data" / "models"


@lru_cache(maxsize=8)
def _load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _model_for_stage(stage: str, version: int = 3) -> dict:
    """Returns {'calibrator', 'feature_names', 'phase'}"""
    name = f"v{version}_phase_{stage}.pkl"
    return _load(str(MODELS_DIR / name))


def predict_for_stage(stage: str, features: dict, version: int = 3) -> tuple[float, list[str]]:
    """Returns (p_team1_wins, feature_order_used)."""
    art = _model_for_stage(stage, version)
    cols = art["feature_names"]
    X = np.array([[features.get(c, 0.0) for c in cols]])
    p = float(art["calibrator"].predict_proba(X)[0, 1])
    return p, cols


def predict_match(match_state: dict, version: int = 3) -> dict:
    """High-level predict. match_state shape:
        {
          "stage": "pre_match" | "post_toss" | "post_pp1" | "innings_break",
          "features": { name: value, ... }  # phase-appropriate features
        }
    Returns {predicted_winner, p_team1, p_team2, stage}
    """
    stage = match_state["stage"]
    p_team1, _ = predict_for_stage(stage, match_state["features"], version=version)
    return {
        "stage": stage,
        "p_team1": p_team1,
        "p_team2": 1.0 - p_team1,
        "predicted_winner": "team1" if p_team1 >= 0.5 else "team2",
    }
