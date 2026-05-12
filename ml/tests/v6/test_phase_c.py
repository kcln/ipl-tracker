"""Phase C smoke tests: load each v6 base model + v7 ensemble, confirm
.predict_proba works on sample rows, and check the kill-criterion gate
for v6_phase_pre_match (val acc >= heuristic baseline 54.3%).
"""
from __future__ import annotations

import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODELS = ROOT / "ml" / "data" / "models"
HIST = ROOT / "ml" / "data" / "historical"
V6 = HIST / "v6"

BASE_MODELS = [
    "v6_phase_pre_match",
    "v6_phase_post_toss",
    "v6_phase_post_pp1",
    "v6_phase_innings_break",
    "v6_player",
]


def _load_pkl(name: str):
    with open(MODELS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)  # nosec B301 — local training artifact


def _load_meta(name: str) -> dict:
    with open(MODELS / f"{name}.json") as f:
        return json.load(f)


def _build_joined_df() -> pd.DataFrame:
    base = pd.read_parquet(HIST / "features_innings_break.parquet")
    for fn in ["phase_player_stats", "venue_patterns", "recency_form",
               "h2h_career", "toss_conditioned", "weather"]:
        f = pd.read_parquet(V6 / f"{fn}.parquet")
        base = base.merge(f, on="match_id", how="left")
    pf = pd.read_parquet(HIST / "player_features.parquet")
    keep = [c for c in pf.columns if c == "match_id" or c.startswith("team")]
    base = base.merge(pf[keep], on="match_id", how="left")
    return base


@pytest.fixture(scope="module")
def joined_df() -> pd.DataFrame:
    return _build_joined_df()


@pytest.mark.parametrize("name", BASE_MODELS)
def test_base_model_loads_and_predicts(name, joined_df):
    art = _load_pkl(name)
    assert "calibrator" in art
    assert "feature_names" in art
    feats = art["feature_names"]
    # take any 3 rows -> predict_proba should produce shape (3, 2) with rows summing to 1
    sample = joined_df[feats].iloc[:3]
    p = art["calibrator"].predict_proba(sample.to_numpy())
    assert p.shape == (3, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_v7_ensemble_loads_and_predicts():
    art = _load_pkl("v7_ensemble")
    assert "meta" in art
    assert "input_columns" in art
    n_inputs = len(art["input_columns"])
    fake = np.array([[0.5] * n_inputs, [0.7] * n_inputs])
    p = art["meta"].predict_proba(fake)
    assert p.shape == (2, 2)


def test_pre_match_val_beats_heuristic_baseline():
    """v6_phase_pre_match must at minimum beat the 54.3% heuristic floor on val."""
    meta = _load_meta("v6_phase_pre_match")
    assert meta["validation_accuracy"] >= 0.543, (
        f"v6_phase_pre_match val acc {meta['validation_accuracy']:.3f} below heuristic 0.543"
    )


def test_v7_ensemble_metadata_present():
    meta = _load_meta("v7_ensemble")
    for k in ("base_models", "test_accuracy", "test_brier", "train_accuracy"):
        assert k in meta
    assert len(meta["base_models"]) >= 1
