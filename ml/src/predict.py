"""Parallel ML predict surface. Does NOT replace the existing src/predictor.py.

This is the ml/ namespace's own predict function for downstream cutover. The
existing tracker continues to call its own predictor unmodified.
"""
from __future__ import annotations

import pathlib
import pickle
from typing import Any

import numpy as np
import pandas as pd

from ml.src.features import FEATURE_FUNCS, build_features

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "ml" / "data" / "models"


def load_model(version: int, name: str):
    path = MODELS_DIR / f"v{version}_{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_match(match_row: dict, state: dict, *, model=None, version: int = 1, name: str = "logistic") -> tuple[str, float]:
    """Returns (predicted_team, prob_team1_wins)."""
    if model is None:
        model = load_model(version, name)
    feats = build_features(match_row, state)
    X = np.array([[feats[k] for k in FEATURE_FUNCS]])
    p_team1 = float(model.predict_proba(X)[0, 1])
    winner = match_row["team1"] if p_team1 >= 0.5 else match_row["team2"]
    return winner, p_team1
