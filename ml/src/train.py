"""Training entry point. Phase 1: v1 logistic regression with isotonic calibration.

Run:
    python -m ml.src.train --version 1 --model logistic
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

ROOT = pathlib.Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ml"
HIST_DIR = ML_ROOT / "data" / "historical"
MODELS_DIR = ML_ROOT / "data" / "models"
PROGRESS_PATH = ML_ROOT / "PROGRESS.md"


FEATURE_COLS = [
    "form_last5_diff",
    "season_nrr_diff",
    "squad_strength_diff",
    "home_advantage_diff",
    "h2h_season_diff",
    "venue_chase_rate",
    "pom_recency_diff",
    "qualified_flag_diff",
    "strength_of_schedule_diff",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _atomic_pickle(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _atomic_json(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def train_logistic(version: int) -> dict:
    feats_path = HIST_DIR / "features.parquet"
    if not feats_path.exists():
        raise RuntimeError(f"missing {feats_path}; run ingest + features first")
    df = pd.read_parquet(feats_path)

    train_df = df[df["split"] == "train"].sort_values(["date", "match_id"]).reset_index(drop=True)
    val_df = df[df["split"] == "val"].sort_values(["date", "match_id"]).reset_index(drop=True)
    test_df = df[df["split"] == "test"].sort_values(["date", "match_id"]).reset_index(drop=True)

    print(f"train n={len(train_df)} val n={len(val_df)} test n={len(test_df)}", flush=True)

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["winner_is_team1"].to_numpy()
    X_val = val_df[FEATURE_COLS].to_numpy()
    y_val = val_df["winner_is_team1"].to_numpy()
    X_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df["winner_is_team1"].to_numpy()

    # CV over C
    best_C, best_score = None, -np.inf
    for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for ti, vi in tscv.split(X_train):
            m = LogisticRegression(C=C, max_iter=1000, random_state=42)
            m.fit(X_train[ti], y_train[ti])
            p = m.predict_proba(X_train[vi])[:, 1]
            # use negative log loss (higher is better)
            scores.append(-log_loss(y_train[vi], np.clip(p, 1e-6, 1 - 1e-6)))
        mean = float(np.mean(scores))
        if mean > best_score:
            best_score, best_C = mean, C

    print(f"best C={best_C} cv_neg_log_loss={best_score:.4f}", flush=True)

    # Fit final + calibrate
    base = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    base.fit(X_train, y_train)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
    cal.fit(X_train, y_train)

    # Validation (2024)
    p_val = cal.predict_proba(X_val)[:, 1]
    val_pred = (p_val >= 0.5).astype(int)
    val_acc = float(accuracy_score(y_val, val_pred))
    val_brier = float(brier_score_loss(y_val, p_val))
    val_ll = float(log_loss(y_val, np.clip(p_val, 1e-6, 1 - 1e-6)))

    # Test (2025) — touched ONCE
    p_test = cal.predict_proba(X_test)[:, 1]
    test_pred = (p_test >= 0.5).astype(int)
    test_acc = float(accuracy_score(y_test, test_pred))
    test_brier = float(brier_score_loss(y_test, p_test))
    test_ll = float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6)))

    coefs = base.coef_[0].tolist()
    importance = sorted(zip(FEATURE_COLS, [abs(c) for c in coefs]), key=lambda x: -x[1])

    model_path = MODELS_DIR / f"v{version}_logistic.pkl"
    meta_path = MODELS_DIR / f"v{version}_logistic.json"
    if model_path.exists():
        raise RuntimeError(f"refusing to overwrite {model_path}; bump version")
    _atomic_pickle(cal, model_path)

    meta = {
        "version": version,
        "name": "logistic",
        "model_type": "logistic_regression_isotonic",
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "sklearn_version": sklearn.__version__,
        "training_seasons": sorted({int(s) for s in train_df["season"].unique()}),
        "validation_season": 2024,
        "test_season": 2025,
        "best_C": best_C,
        "cv_mean_neg_log_loss": best_score,
        "validation_accuracy": val_acc,
        "validation_brier": val_brier,
        "validation_log_loss": val_ll,
        "test_accuracy": test_acc,
        "test_brier": test_brier,
        "test_log_loss": test_ll,
        "feature_names": FEATURE_COLS,
        "coefficients": dict(zip(FEATURE_COLS, coefs)),
        "intercept": float(base.intercept_[0]),
        "feature_importance_ranked": importance,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    _atomic_json(meta, meta_path)

    print(f"Saved {model_path}", flush=True)
    print(f"  val 2024: acc={val_acc:.3f} brier={val_brier:.3f} log_loss={val_ll:.3f}", flush=True)
    print(f"  test 2025: acc={test_acc:.3f} brier={test_brier:.3f} log_loss={test_ll:.3f}", flush=True)

    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--model", choices=["logistic"], default="logistic")
    args = ap.parse_args()
    if args.model == "logistic":
        train_logistic(args.version)


if __name__ == "__main__":
    main()
