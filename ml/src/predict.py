"""Parallel ML predict surface for the ml/ namespace.

Routes to the appropriate model based on the match stage:
  pre_match     -> v5 stacked ensemble (v1 + v3_pre_match + v4 -> logistic meta)
  post_toss     -> v3_phase_post_toss
  post_pp1      -> v3_phase_post_pp1
  innings_break -> v3_phase_innings_break
  live          -> v2_wp_lightgbm (per-delivery)

This does NOT replace the existing src/predictor.py used by the live tracker.
"""
from __future__ import annotations

import pathlib
import pickle
from functools import lru_cache

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "ml" / "data" / "models"


@lru_cache(maxsize=16)
def _load(path: str):
    # Pickle is safe here: the only files we ever load are our own trained model
    # artifacts produced under ml/data/models/ on this machine. We never load a
    # pickle from user input or the network. See ml/CLAUDE.md "atomic file writes".
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec B301


def predict_pre_match(
    *,
    p_v1: float, p_v3: float, p_v4: float,
) -> float:
    """v5 ensemble pre-match predict. Caller provides base-model probabilities;
    returns the stacked logistic meta-learner's calibrated p_team1."""
    art = _load(str(MODELS_DIR / "v5_ensemble.pkl"))
    meta = art["meta"]
    X = np.array([[p_v1, p_v3, p_v4]])
    return float(meta.predict_proba(X)[0, 1])


def predict_phase(stage: str, features: dict, version: int = 3) -> float:
    """Stage-specific predict (post_toss / post_pp1 / innings_break)."""
    if stage == "pre_match":
        raise ValueError("use predict_pre_match() for pre-match ensemble")
    name = f"v{version}_phase_{stage}.pkl"
    art = _load(str(MODELS_DIR / name))
    cols = art["feature_names"]
    X = np.array([[features.get(c, 0.0) for c in cols]])
    return float(art["calibrator"].predict_proba(X)[0, 1])


def predict_live(features: dict) -> float:
    """Per-delivery WP. features must include the keys from v2_wp_lightgbm.json."""
    art = _load(str(MODELS_DIR / "v2_wp_lightgbm.pkl"))
    cal = art["calibrator"]
    cols = art["feature_names"]
    X = np.array([[features.get(c, 0.0) for c in cols]])
    return float(cal.predict_proba(X)[0, 1])
