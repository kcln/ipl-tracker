"""Phase 4: v4 LightGBM with player-level features joined to pre-match."""
from __future__ import annotations

import argparse, datetime as dt, json, os, pathlib, pickle, subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

ROOT = pathlib.Path(__file__).resolve().parents[2]
ML = ROOT / "ml"; HIST = ML / "data" / "historical"; MODELS = ML / "data" / "models"

BASE_FEATURES = [
    "team1_career_wr", "team2_career_wr",
    "team1_career_played", "team2_career_played",
    "team1_season_wr", "team2_season_wr",
    "team1_season_played", "team2_season_played",
    "team1_form5", "team2_form5",
    "team1_venue_wr", "team2_venue_wr",
    "venue_chase_rate", "venue_played",
]
PLAYER_FEATURES = [
    "team1_top5_batter_sr_mean", "team1_top5_batter_sr_max", "team1_n_batters_qual",
    "team1_top4_bowler_econ_mean", "team1_top4_bowler_econ_min", "team1_n_bowlers_qual",
    "team2_top5_batter_sr_mean", "team2_top5_batter_sr_max", "team2_n_batters_qual",
    "team2_top4_bowler_econ_mean", "team2_top4_bowler_econ_min", "team2_n_bowlers_qual",
]
ALL_FEATURES = BASE_FEATURES + PLAYER_FEATURES


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _ap(o, p):
    p.parent.mkdir(parents=True, exist_ok=True); t=p.with_suffix(p.suffix+".tmp")
    with open(t,"wb") as f: pickle.dump(o,f)
    os.replace(t,p)


def _aj(o, p):
    p.parent.mkdir(parents=True, exist_ok=True); t=p.with_suffix(p.suffix+".tmp")
    with open(t,"w") as f: json.dump(o,f,indent=2,default=str)
    os.replace(t,p)


def train(version: int = 4):
    pm = pd.read_parquet(HIST / "features_pre_match.parquet")
    pf = pd.read_parquet(HIST / "player_features.parquet")
    df = pm.merge(pf[["match_id"] + PLAYER_FEATURES], on="match_id", how="left")
    print(f"merged: {len(df)} rows; nan rate per player col:")
    print(df[PLAYER_FEATURES].isna().mean())

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train, y_train = train_df[ALL_FEATURES], train_df["winner_is_team1"]
    X_val, y_val = val_df[ALL_FEATURES], val_df["winner_is_team1"]
    X_test, y_test = test_df[ALL_FEATURES], test_df["winner_is_team1"]

    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)} feats={len(ALL_FEATURES)}", flush=True)

    best_n, best_score = None, -np.inf
    for n_est in [100, 200, 400, 600]:
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for ti, vi in tscv.split(X_train):
            m = lgb.LGBMClassifier(n_estimators=n_est, max_depth=5, learning_rate=0.05,
                                   num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
            m.fit(X_train.iloc[ti], y_train.iloc[ti])
            p = m.predict_proba(X_train.iloc[vi])[:, 1]
            scores.append(-log_loss(y_train.iloc[vi], np.clip(p, 1e-6, 1 - 1e-6)))
        mean = float(np.mean(scores))
        if mean > best_score:
            best_score, best_n = mean, n_est
    print(f"best n_estimators={best_n} cv_score={best_score:.4f}", flush=True)

    base = lgb.LGBMClassifier(n_estimators=best_n, max_depth=5, learning_rate=0.05,
                              num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    base.fit(X_train, y_train)
    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)

    p_val = cal.predict_proba(X_val)[:, 1]
    val_acc = float(accuracy_score(y_val, (p_val >= 0.5).astype(int)))
    val_brier = float(brier_score_loss(y_val, p_val))
    val_ll = float(log_loss(y_val, np.clip(p_val, 1e-6, 1 - 1e-6)))
    p_test = cal.predict_proba(X_test)[:, 1]
    test_acc = float(accuracy_score(y_test, (p_test >= 0.5).astype(int)))
    test_brier = float(brier_score_loss(y_test, p_test))
    test_ll = float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6)))

    importance = sorted(zip(ALL_FEATURES, base.booster_.feature_importance(importance_type="gain").tolist()),
                        key=lambda x: -x[1])
    print("Top 10 features by gain:")
    for n, g in importance[:10]:
        print(f"  {n}: {g}")

    model_path = MODELS / f"v{version}_player_gbm.pkl"
    meta_path = MODELS / f"v{version}_player_gbm.json"
    if model_path.exists():
        raise RuntimeError(f"refusing to overwrite {model_path}")
    _ap({"calibrator": cal, "feature_names": ALL_FEATURES, "phase": "pre_match"}, model_path)

    meta = {
        "version": version, "name": "player_gbm",
        "model_type": "lightgbm_isotonic_prefit",
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(), "sklearn_version": sklearn.__version__, "lightgbm_version": lgb.__version__,
        "best_n_estimators": best_n, "cv_mean_neg_log_loss": best_score,
        "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "validation_accuracy": val_acc, "validation_brier": val_brier, "validation_log_loss": val_ll,
        "test_accuracy": test_acc, "test_brier": test_brier, "test_log_loss": test_ll,
        "feature_names": ALL_FEATURES, "feature_importance_gain_ranked": importance,
        "hyperparameters": {"n_estimators": best_n, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 31, "random_state": 42},
        "base_feature_count": len(BASE_FEATURES), "player_feature_count": len(PLAYER_FEATURES),
    }
    _aj(meta, meta_path)

    print(f"Saved {model_path}", flush=True)
    print(f"  val 2024: acc={val_acc:.3f} brier={val_brier:.3f}", flush=True)
    print(f"  test 2025: acc={test_acc:.3f} brier={test_brier:.3f}", flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--version", type=int, default=4); a=ap.parse_args()
    train(a.version)
