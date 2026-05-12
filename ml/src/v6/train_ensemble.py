"""v7 ensemble: LightGBM meta-learner stacking v6_phase_pre_match,
v6_player, and v3_phase_pre_match (stable reference baseline).

Per v6 plan, v7 replaces v5's logistic meta with LightGBM. Base-model
predictions are computed on the 2024 val rows for meta training and
2025 test rows for evaluation -- the same split as v5.

If a base model hit its kill criterion (val acc < heuristic + 2%), it is
excluded from the v7 stack and logged.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import pickle
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = pathlib.Path(__file__).resolve().parents[3]
ML = ROOT / "ml"
HIST = ML / "data" / "historical"
V6 = HIST / "v6"
MODELS = ML / "data" / "models"


def _load_pkl(name: str):
    with open(MODELS / name, "rb") as f:
        return pickle.load(f)  # nosec B301 — own training artifacts


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _atomic_pickle(obj, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _atomic_json(obj, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _build_full_dataframe() -> pd.DataFrame:
    """Identical join logic as v6.train_phase._build_dataframe for pre_match + player.
    We need both v6_phase_pre_match features AND v6_player features (including v4 player aggregates).
    """
    base = pd.read_parquet(HIST / "features_pre_match.parquet")
    a1 = pd.read_parquet(V6 / "phase_player_stats.parquet")
    a2 = pd.read_parquet(V6 / "venue_patterns.parquet")
    a3 = pd.read_parquet(V6 / "recency_form.parquet")
    a4 = pd.read_parquet(V6 / "h2h_career.parquet")
    a5 = pd.read_parquet(V6 / "toss_conditioned.parquet")
    b1 = pd.read_parquet(V6 / "weather.parquet")
    pf = pd.read_parquet(HIST / "player_features.parquet")

    V4_PLAYER = [
        "team1_top5_batter_sr_mean", "team1_top5_batter_sr_max", "team1_n_batters_qual",
        "team1_top4_bowler_econ_mean", "team1_top4_bowler_econ_min", "team1_n_bowlers_qual",
        "team2_top5_batter_sr_mean", "team2_top5_batter_sr_max", "team2_n_batters_qual",
        "team2_top4_bowler_econ_mean", "team2_top4_bowler_econ_min", "team2_n_bowlers_qual",
    ]
    df = base.merge(a1, on="match_id", how="left") \
             .merge(a2, on="match_id", how="left") \
             .merge(a3, on="match_id", how="left") \
             .merge(a4, on="match_id", how="left") \
             .merge(a5, on="match_id", how="left") \
             .merge(b1, on="match_id", how="left") \
             .merge(pf[["match_id"] + V4_PLAYER], on="match_id", how="left")
    return df


def main():
    df = _build_full_dataframe()

    # Load base artifacts. Skip any that hit kill criterion.
    candidates = [
        ("v6_phase_pre_match", "v6_phase_pre_match.pkl"),
        ("v6_player",          "v6_player.pkl"),
        ("v3_phase_pre_match", "v3_phase_pre_match.pkl"),  # stable reference
    ]

    base_preds = {}
    used = []
    skipped = []
    for name, fn in candidates:
        art = _load_pkl(fn)
        if art.get("hit_kill_criterion", False):
            print(f"skipping {name} (kill criterion)")
            skipped.append(name)
            continue
        feats = art["feature_names"]
        # v3 model uses base 14 features that exist directly on the dataframe.
        # v6 models use enriched features. df has all columns merged.
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise RuntimeError(f"{name}: missing features in joined df: {missing}")
        p = art["calibrator"].predict_proba(df[feats].to_numpy())[:, 1]
        base_preds[f"p_{name}"] = p
        used.append(name)

    pred_cols = list(base_preds.keys())
    print(f"using base models: {used}")
    if skipped:
        print(f"skipped (kill criterion): {skipped}")

    preds_df = pd.DataFrame({"match_id": df["match_id"].values, **base_preds})
    joined = df[["match_id", "split", "date", "winner_is_team1"]].merge(preds_df, on="match_id", how="inner")

    meta_train = joined[joined["split"] == "val"].copy()    # 2024
    meta_test = joined[joined["split"] == "test"].copy()    # 2025

    X_train = meta_train[pred_cols].to_numpy()
    y_train = meta_train["winner_is_team1"].astype(int).to_numpy()
    X_test = meta_test[pred_cols].to_numpy()
    y_test = meta_test["winner_is_team1"].astype(int).to_numpy()

    print(f"meta train (2024) n={len(meta_train)}  test (2025) n={len(meta_test)}  inputs={pred_cols}")

    # LightGBM meta-learner. Small data -> shallow + low LR.
    meta = lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        num_leaves=7, min_child_samples=5, random_state=42,
        n_jobs=2, verbose=-1,
    )
    meta.fit(X_train, y_train)

    p_train = meta.predict_proba(X_train)[:, 1]
    train_acc = float(accuracy_score(y_train, (p_train >= 0.5).astype(int)))
    train_brier = float(brier_score_loss(y_train, p_train))

    p_test = meta.predict_proba(X_test)[:, 1]
    test_acc = float(accuracy_score(y_test, (p_test >= 0.5).astype(int)))
    test_brier = float(brier_score_loss(y_test, p_test))
    test_ll = float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6)))

    importance = sorted(
        zip(pred_cols, meta.booster_.feature_importance(importance_type="gain").tolist()),
        key=lambda x: -x[1],
    )

    print(f"meta train (2024): acc={train_acc:.3f} brier={train_brier:.3f}")
    print(f"meta test  (2025): acc={test_acc:.3f} brier={test_brier:.3f} log_loss={test_ll:.3f}")
    print(f"importance: {importance}")

    model_path = MODELS / "v7_ensemble.pkl"
    meta_path = MODELS / "v7_ensemble.json"
    if model_path.exists():
        raise RuntimeError(f"refusing to overwrite {model_path}")

    _atomic_pickle(
        {"meta": meta, "base_models": used, "input_columns": pred_cols},
        model_path,
    )
    art = {
        "version": 7,
        "name": "ensemble_stacked_lightgbm",
        "model_type": "stacked_lightgbm_meta",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": _git_sha(),
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "base_models": used,
        "skipped_base_models": skipped,
        "input_columns": pred_cols,
        "feature_importance_gain_ranked": importance,
        "hyperparameters": {
            "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
            "num_leaves": 7, "min_child_samples": 5, "random_state": 42,
        },
        "train_season": 2024, "n_train": int(len(X_train)),
        "test_season": 2025,  "n_test": int(len(X_test)),
        "train_accuracy": train_acc, "train_brier": train_brier,
        "test_accuracy": test_acc, "test_brier": test_brier, "test_log_loss": test_ll,
    }
    _atomic_json(art, meta_path)
    print(f"saved {model_path}")
    return art


if __name__ == "__main__":
    main()
