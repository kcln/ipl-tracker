"""Parallel ML predict surface for the ml/ namespace.

Routes to the best-known model per stage:
  pre_match     -> v9 ensemble (retrained on 2008-2024 base, meta on 2025) — 64.2% on 2026 held-out
                   v9_player alone hits 66.0% but the ensemble is more stable
  post_toss     -> v3_phase_post_toss
  post_pp1      -> v3_phase_post_pp1
  innings_break -> v6_phase_innings_break (72.9% on 2025 test — the headline mid-match model)
  live          -> v2_wp_lightgbm (71.6% per-delivery on 2025)

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
    # Trust boundary: only loads our own training artifacts from ml/data/models.
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec B301


def predict_pre_match(
    *,
    p_v9_pre_match: float,
    p_v9_player: float,
    p_v3_pre_match: float,
) -> float:
    """v9 ensemble pre-match predict.

    Caller provides the three base-model probabilities; returns the stacked
    logistic meta-learner's calibrated P(team1 wins). The meta was trained on
    2025 outcomes against base models trained on 2008-2024. On the held-out
    2026 window (n=53), v9 ensemble scored 64.2% accuracy.

    For a one-shot prediction, prefer `predict_pre_match_full(match_state)` —
    it builds the three base predictions for you.
    """
    art = _load(str(MODELS_DIR / "v9_ensemble.pkl"))
    meta = art["meta"]
    X = np.array([[p_v9_pre_match, p_v9_player, p_v3_pre_match]])
    return float(meta.predict_proba(X)[0, 1])


def predict_pre_match_full(features_v3: dict, features_player: dict) -> dict:
    """End-to-end pre-match predict from raw features.

    features_v3: dict matching v9_pre_match's feature_names (14 base PIT features)
    features_player: dict matching v9_player's player feature columns

    Returns {"p_team1": float, "p_v9_pre_match", "p_v9_player", "p_v3_pre_match"}.
    """
    v9pm = _load(str(MODELS_DIR / "v9_pre_match.pkl"))
    v9pl = _load(str(MODELS_DIR / "v9_player.pkl"))
    v3 = _load(str(MODELS_DIR / "v3_phase_pre_match.pkl"))

    X_v3 = np.array([[features_v3[c] for c in v9pm["feature_names"]]])
    p_v9pm = float(v9pm["calibrator"].predict_proba(X_v3)[0, 1])
    p_v3pm = float(v3["calibrator"].predict_proba(X_v3)[0, 1])

    full_feats = {**features_v3, **features_player}
    X_full = np.array([[full_feats.get(c, np.nan) for c in v9pl["feature_names"]]])
    p_v9pl = float(v9pl["calibrator"].predict_proba(X_full)[0, 1])

    p_team1 = predict_pre_match(p_v9_pre_match=p_v9pm, p_v9_player=p_v9pl, p_v3_pre_match=p_v3pm)
    return {"p_team1": p_team1, "p_v9_pre_match": p_v9pm, "p_v9_player": p_v9pl, "p_v3_pre_match": p_v3pm}


def predict_phase(stage: str, features: dict) -> float:
    """Stage-specific predict for post_toss / post_pp1 / innings_break.

    Routes to:
      post_toss     -> v3_phase_post_toss
      post_pp1      -> v3_phase_post_pp1
      innings_break -> v6_phase_innings_break (the strongest mid-match model — 72.9% on 2025)
    """
    if stage == "pre_match":
        raise ValueError("use predict_pre_match() / predict_pre_match_full() for pre-match")
    if stage == "innings_break":
        art = _load(str(MODELS_DIR / "v6_phase_innings_break.pkl"))
    elif stage in ("post_toss", "post_pp1"):
        art = _load(str(MODELS_DIR / f"v3_phase_{stage}.pkl"))
    else:
        raise ValueError(f"unknown stage {stage}")
    cols = art["feature_names"]
    X = np.array([[features.get(c, np.nan) for c in cols]])
    return float(art["calibrator"].predict_proba(X)[0, 1])


def predict_live(features: dict) -> float:
    """Per-delivery WP. features must include the keys from v2_wp_lightgbm.json.

    Used for live in-match win-probability charts. 71.6% per-delivery accuracy
    on 2025 test set. The artifact bundles the calibrator + the venue_categories
    mapping needed to encode the venue feature.
    """
    art = _load(str(MODELS_DIR / "v2_wp_lightgbm.pkl"))
    cal = art["calibrator"]
    cols = art["feature_names"]
    # If features dict has 'venue' as a string, map it to venue_code
    if "venue" in features and "venue_code" not in features:
        try:
            features = {**features, "venue_code": art["venue_categories"].index(features["venue"])}
        except (ValueError, KeyError):
            features = {**features, "venue_code": -1}  # unknown venue
    X = np.array([[features.get(c, np.nan) for c in cols]])
    return float(cal.predict_proba(X)[0, 1])
