"""Phase 3: per-phase LightGBM training.

Trains 4 separate calibrated LightGBM models for each prediction stage:
  pre_match, post_toss, post_pp1, innings_break
"""
from __future__ import annotations

import argparse
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

ROOT = pathlib.Path(__file__).resolve().parents[2]
ML = ROOT / "ml"
HIST = ML / "data" / "historical"
MODELS = ML / "data" / "models"


FEATURES_BY_PHASE = {
    "pre_match": [
        "team1_career_wr", "team2_career_wr",
        "team1_career_played", "team2_career_played",
        "team1_season_wr", "team2_season_wr",
        "team1_season_played", "team2_season_played",
        "team1_form5", "team2_form5",
        "team1_venue_wr", "team2_venue_wr",
        "venue_chase_rate", "venue_played",
    ],
    "post_toss": [
        "team1_career_wr", "team2_career_wr",
        "team1_season_wr", "team2_season_wr",
        "team1_form5", "team2_form5",
        "team1_venue_wr", "team2_venue_wr",
        "venue_chase_rate",
        "toss_winner_is_team1", "toss_decision_field",
    ],
    "post_pp1": [
        "team1_career_wr", "team2_career_wr",
        "team1_season_wr", "team2_season_wr",
        "team1_form5", "team2_form5",
        "team1_venue_wr", "team2_venue_wr",
        "venue_chase_rate",
        "toss_winner_is_team1", "toss_decision_field",
        "pp1_runs", "pp1_wickets",
    ],
    "innings_break": [
        "team1_career_wr", "team2_career_wr",
        "team1_season_wr", "team2_season_wr",
        "team1_form5", "team2_form5",
        "team1_venue_wr", "team2_venue_wr",
        "venue_chase_rate",
        "toss_winner_is_team1", "toss_decision_field",
        "pp1_runs", "pp1_wickets",
        "first_innings_total", "first_innings_wickets",
        "innings_break_required_rr",
    ],
}

PARQUET_BY_PHASE = {
    "pre_match": "features_pre_match.parquet",
    "post_toss": "features_post_toss.parquet",
    "post_pp1": "features_post_pp1.parquet",
    "innings_break": "features_innings_break.parquet",
}


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _atomic_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _atomic_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def train_phase(phase: str, version: int = 3):
    parquet = HIST / PARQUET_BY_PHASE[phase]
    df = pd.read_parquet(parquet)
    feats = FEATURES_BY_PHASE[phase]

    train = df[df["split"] == "train"].sort_values(["date", "match_id"]).reset_index(drop=True)
    val = df[df["split"] == "val"].sort_values(["date", "match_id"]).reset_index(drop=True)
    test = df[df["split"] == "test"].sort_values(["date", "match_id"]).reset_index(drop=True)

    X_train, y_train = train[feats], train["winner_is_team1"]
    X_val, y_val = val[feats], val["winner_is_team1"]
    X_test, y_test = test[feats], test["winner_is_team1"]

    print(f"[{phase}] train={len(train)} val={len(val)} test={len(test)} features={len(feats)}", flush=True)

    # TimeSeriesSplit CV on n_estimators
    best_n, best_score = None, -np.inf
    for n_est in [100, 200, 400]:
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

    print(f"  best n_estimators={best_n} cv_neg_log_loss={best_score:.4f}", flush=True)

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

    importance = sorted(zip(feats, base.booster_.feature_importance(importance_type="gain").tolist()),
                        key=lambda x: -x[1])

    model_path = MODELS / f"v{version}_phase_{phase}.pkl"
    meta_path = MODELS / f"v{version}_phase_{phase}.json"
    if model_path.exists():
        raise RuntimeError(f"refusing to overwrite {model_path}")

    _atomic_pickle({"calibrator": cal, "feature_names": feats, "phase": phase}, model_path)
    meta = {
        "version": version, "name": f"phase_{phase}", "phase": phase,
        "model_type": "lightgbm_isotonic_prefit",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": _git_sha(), "sklearn_version": sklearn.__version__, "lightgbm_version": lgb.__version__,
        "best_n_estimators": best_n, "cv_mean_neg_log_loss": best_score,
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "validation_accuracy": val_acc, "validation_brier": val_brier, "validation_log_loss": val_ll,
        "test_accuracy": test_acc, "test_brier": test_brier, "test_log_loss": test_ll,
        "feature_names": feats, "feature_importance_gain_ranked": importance,
        "hyperparameters": {"n_estimators": best_n, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 31, "random_state": 42},
    }
    _atomic_json(meta, meta_path)

    print(f"  saved {model_path}")
    print(f"  val 2024: acc={val_acc:.3f} brier={val_brier:.3f}")
    print(f"  test 2025: acc={test_acc:.3f} brier={test_brier:.3f}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(FEATURES_BY_PHASE.keys()) + ["all"], default="all")
    ap.add_argument("--version", type=int, default=3)
    args = ap.parse_args()
    phases = list(FEATURES_BY_PHASE.keys()) if args.phase == "all" else [args.phase]
    results = []
    for p in phases:
        print(f"\n=== {p} ===")
        results.append(train_phase(p, args.version))
    print("\nSummary:")
    for r in results:
        print(f"  {r['phase']:14s} val={r['validation_accuracy']:.3f} test={r['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
