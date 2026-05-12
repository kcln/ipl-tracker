"""Phase 5: v5 logistic meta-learner stacking v1 + v3_pre_match + v4.

Meta-learner training set: 2024 base-model predictions (the val set from earlier phases).
Test set: 2025 base-model predictions — touched once for v5.
"""
from __future__ import annotations

import datetime as dt, json, os, pathlib, pickle, subprocess

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = pathlib.Path(__file__).resolve().parents[2]
ML = ROOT / "ml"; HIST = ML / "data" / "historical"; MODELS = ML / "data" / "models"

V1_FEATURES = [
    "form_last5_diff", "season_nrr_diff", "squad_strength_diff",
    "home_advantage_diff", "h2h_season_diff", "venue_chase_rate",
    "pom_recency_diff", "qualified_flag_diff", "strength_of_schedule_diff",
]


def _load_pkl(name):
    with open(MODELS / name, "rb") as f:
        return pickle.load(f)


def _git_sha():
    try: return subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    except: return "unknown"


def _ap(o,p):
    p.parent.mkdir(parents=True, exist_ok=True); t=p.with_suffix(p.suffix+".tmp")
    with open(t,"wb") as f: pickle.dump(o,f)
    os.replace(t,p)


def _aj(o,p):
    p.parent.mkdir(parents=True, exist_ok=True); t=p.with_suffix(p.suffix+".tmp")
    with open(t,"w") as f: json.dump(o,f,indent=2,default=str)
    os.replace(t,p)


def main():
    # Load base models
    v1 = _load_pkl("v1_logistic.pkl")            # sklearn pipeline (CalibratedClassifierCV)
    v3 = _load_pkl("v3_phase_pre_match.pkl")    # {"calibrator", "feature_names"}
    v4 = _load_pkl("v4_player_gbm.pkl")         # {"calibrator", "feature_names"}

    # Load features and generate predictions in isolation, then merge by match_id
    f9 = pd.read_parquet(HIST / "features.parquet")
    fpm = pd.read_parquet(HIST / "features_pre_match.parquet")
    pf = pd.read_parquet(HIST / "player_features.parquet")
    fpm_pf = fpm.merge(pf, on="match_id", how="left")

    # v1 predictions on its own feature set
    p_v1 = v1.predict_proba(f9[V1_FEATURES].to_numpy())[:, 1]
    v1_preds = pd.DataFrame({"match_id": f9["match_id"].values, "p_v1": p_v1})

    # v3 predictions
    p_v3 = v3["calibrator"].predict_proba(fpm_pf[v3["feature_names"]].to_numpy())[:, 1]
    v3_preds = pd.DataFrame({"match_id": fpm_pf["match_id"].values, "p_v3": p_v3})

    # v4 predictions
    p_v4 = v4["calibrator"].predict_proba(fpm_pf[v4["feature_names"]].to_numpy())[:, 1]
    v4_preds = pd.DataFrame({"match_id": fpm_pf["match_id"].values, "p_v4": p_v4})

    base = fpm[["match_id", "split", "date", "winner_is_team1"]].merge(v1_preds, on="match_id", how="inner")
    base = base.merge(v3_preds, on="match_id", how="inner")
    base = base.merge(v4_preds, on="match_id", how="inner")

    meta_train = base[base["split"] == "val"]   # 2024
    meta_test = base[base["split"] == "test"]   # 2025
    print(f"meta train (2024) n={len(meta_train)} test (2025) n={len(meta_test)}")

    X_train = meta_train[["p_v1", "p_v3", "p_v4"]].to_numpy()
    y_train = meta_train["winner_is_team1"].astype(int).to_numpy()
    X_test = meta_test[["p_v1", "p_v3", "p_v4"]].to_numpy()
    y_test = meta_test["winner_is_team1"].astype(int).to_numpy()

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(X_train, y_train)

    p_test = meta.predict_proba(X_test)[:, 1]
    test_acc = float(accuracy_score(y_test, (p_test >= 0.5).astype(int)))
    test_brier = float(brier_score_loss(y_test, p_test))
    test_ll = float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6)))

    # Also report meta-learner's training accuracy (on 2024)
    p_train = meta.predict_proba(X_train)[:, 1]
    train_acc = float(accuracy_score(y_train, (p_train >= 0.5).astype(int)))
    train_brier = float(brier_score_loss(y_train, p_train))

    print("meta coefficients:", dict(zip(["p_v1","p_v3","p_v4"], meta.coef_[0].tolist())))
    print("meta intercept:", float(meta.intercept_[0]))
    print(f"meta train (2024): acc={train_acc:.3f} brier={train_brier:.3f}")
    print(f"meta test  (2025): acc={test_acc:.3f} brier={test_brier:.3f} log_loss={test_ll:.3f}")

    model_path = MODELS / "v5_ensemble.pkl"
    meta_path  = MODELS / "v5_ensemble.json"
    if model_path.exists():
        raise RuntimeError(f"refusing to overwrite {model_path}")
    _ap({"meta": meta, "base_models": ["v1_logistic", "v3_phase_pre_match", "v4_player_gbm"]}, model_path)

    art = {
        "version": 5, "name": "ensemble_stacked_logistic",
        "model_type": "stacked_logistic_meta",
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(), "sklearn_version": sklearn.__version__,
        "base_models": ["v1_logistic", "v3_phase_pre_match", "v4_player_gbm"],
        "meta_coefficients": {k: float(c) for k, c in zip(["p_v1","p_v3","p_v4"], meta.coef_[0].tolist())},
        "meta_intercept": float(meta.intercept_[0]),
        "train_season": 2024, "n_train": int(len(X_train)),
        "test_season": 2025,  "n_test": int(len(X_test)),
        "train_accuracy": train_acc, "train_brier": train_brier,
        "test_accuracy": test_acc, "test_brier": test_brier, "test_log_loss": test_ll,
    }
    _aj(art, meta_path)
    print(f"saved {model_path}")
    return art


if __name__ == "__main__":
    main()
