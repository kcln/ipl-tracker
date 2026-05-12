"""v6 Phase C: train 5 LightGBM base models on enriched v6 features.

Builds on the v3 per-phase trainer pattern. Joins all v6 phase A parquets
(A1 phase player stats, A2 venue patterns, A3 recency form, A4 H2H,
A5 toss-conditioned) plus B1 weather to the existing v3-era phase feature
parquets, then trains/calibrates per phase.

5 models, all saved under ml/data/models/:
  v6_phase_pre_match.pkl       -> v3 base + A1-A5 + weather
  v6_phase_post_toss.pkl       -> same + toss winner/decision
  v6_phase_post_pp1.pkl        -> same + pp1 runs/wickets
  v6_phase_innings_break.pkl   -> same + first innings total/wkts + RRR
  v6_player.pkl                -> v3 pre-match base + A1 phase player stats
                                  + v4 player_features aggregate

Rules carry forward from v1-v5:
- LightGBM with isotonic prefit calibration
- TimeSeriesSplit CV on n_estimators
- Calibrator fit on 2024 val ONLY
- 2025 test touched ONCE per model -> logged in ml/PROGRESS.md
- Atomic writes (.tmp + os.replace)
- Kill criteria: val acc < heuristic_val + 2% (i.e. < 58.3%) -> save anyway
  but flag for exclusion from v7 ensemble.
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

ROOT = pathlib.Path(__file__).resolve().parents[3]
ML = ROOT / "ml"
HIST = ML / "data" / "historical"
V6 = HIST / "v6"
MODELS = ML / "data" / "models"

# v3 base feature lists (kept identical for direct comparability)
V3_BASE_PRE_MATCH = [
    "team1_career_wr", "team2_career_wr",
    "team1_career_played", "team2_career_played",
    "team1_season_wr", "team2_season_wr",
    "team1_season_played", "team2_season_played",
    "team1_form5", "team2_form5",
    "team1_venue_wr", "team2_venue_wr",
    "venue_chase_rate", "venue_played",
]

# A1 phase-specific player aggregates
A1_FEATS = [
    "team1_top5_pp_sr_mean", "team1_top4_pp_econ_mean",
    "team1_top5_middle_sr_mean", "team1_top4_middle_econ_mean",
    "team1_top5_death_sr_mean", "team1_top4_death_econ_mean",
    "team2_top5_pp_sr_mean", "team2_top4_pp_econ_mean",
    "team2_top5_middle_sr_mean", "team2_top4_middle_econ_mean",
    "team2_top5_death_sr_mean", "team2_top4_death_econ_mean",
]

# A2 venue patterns
A2_FEATS = [
    "venue_avg_1st_innings",
    "venue_wickets_per_pp", "venue_wickets_per_middle", "venue_wickets_per_death",
    "venue_pace_econ", "venue_spin_econ", "venue_match_count",
]

# A3 recency form
A3_FEATS = [
    "team1_decay_form", "team2_decay_form", "team1_decay_form_diff",
]

# A4 H2H
A4_FEATS = ["h2h_career_team1_winrate", "h2h_last5_team1_winrate"]

# A5 toss-conditioned
A5_FEATS = [
    "venue_chase_rate_when_fielded",
    "venue_chase_rate_when_batted",
    "venue_total_matches",
]

# B1 weather
B1_FEATS = ["temp_c", "humidity_pct", "dewpoint_c", "wind_kmh", "precipitation_mm"]

# v4-era player aggregates (kept for v6_player only)
V4_PLAYER_FEATS = [
    "team1_top5_batter_sr_mean", "team1_top5_batter_sr_max", "team1_n_batters_qual",
    "team1_top4_bowler_econ_mean", "team1_top4_bowler_econ_min", "team1_n_bowlers_qual",
    "team2_top5_batter_sr_mean", "team2_top5_batter_sr_max", "team2_n_batters_qual",
    "team2_top4_bowler_econ_mean", "team2_top4_bowler_econ_min", "team2_n_bowlers_qual",
]

V6_ENRICHED = A1_FEATS + A2_FEATS + A3_FEATS + A4_FEATS + A5_FEATS + B1_FEATS

FEATURE_SETS = {
    "phase_pre_match": V3_BASE_PRE_MATCH + V6_ENRICHED,
    "phase_post_toss": V3_BASE_PRE_MATCH + V6_ENRICHED + [
        "toss_winner_is_team1", "toss_decision_field",
    ],
    "phase_post_pp1": V3_BASE_PRE_MATCH + V6_ENRICHED + [
        "toss_winner_is_team1", "toss_decision_field",
        "pp1_runs", "pp1_wickets",
    ],
    "phase_innings_break": V3_BASE_PRE_MATCH + V6_ENRICHED + [
        "toss_winner_is_team1", "toss_decision_field",
        "pp1_runs", "pp1_wickets",
        "first_innings_total", "first_innings_wickets",
        "innings_break_required_rr",
    ],
    "player": V3_BASE_PRE_MATCH + A1_FEATS + V4_PLAYER_FEATS,
}

BASE_PARQUET_BY_PHASE = {
    "phase_pre_match": "features_pre_match.parquet",
    "phase_post_toss": "features_post_toss.parquet",
    "phase_post_pp1": "features_post_pp1.parquet",
    "phase_innings_break": "features_innings_break.parquet",
    "player": "features_pre_match.parquet",
}

HEURISTIC_VAL_ACC = 0.563  # documented baseline from PROGRESS.md
KILL_THRESHOLD = HEURISTIC_VAL_ACC + 0.02  # 58.3%


# ----- utilities -----

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


def _build_dataframe(phase: str) -> pd.DataFrame:
    """Join v3 base features + v6 phase A parquets + B1 weather + (for player) v4 player_features."""
    base = pd.read_parquet(HIST / BASE_PARQUET_BY_PHASE[phase])

    # Load all v6 parquets
    a1 = pd.read_parquet(V6 / "phase_player_stats.parquet")
    a2 = pd.read_parquet(V6 / "venue_patterns.parquet")
    a3 = pd.read_parquet(V6 / "recency_form.parquet")
    a4 = pd.read_parquet(V6 / "h2h_career.parquet")
    a5 = pd.read_parquet(V6 / "toss_conditioned.parquet")
    b1 = pd.read_parquet(V6 / "weather.parquet")

    df = base.merge(a1, on="match_id", how="left") \
             .merge(a2, on="match_id", how="left") \
             .merge(a3, on="match_id", how="left") \
             .merge(a4, on="match_id", how="left") \
             .merge(a5, on="match_id", how="left") \
             .merge(b1, on="match_id", how="left")

    if phase == "player":
        pf = pd.read_parquet(HIST / "player_features.parquet")
        # avoid duplicate columns (player_features carries match_id/date/season only as keys)
        keep = ["match_id"] + V4_PLAYER_FEATS
        df = df.merge(pf[keep], on="match_id", how="left")

    return df


def train_one(phase: str, version: int = 6, fast: bool = False):
    df = _build_dataframe(phase)
    feats = FEATURE_SETS[phase]

    # column-by-column NaN check (>30% -> drop with warning)
    drop_cols = []
    for c in feats:
        if c not in df.columns:
            raise RuntimeError(f"missing feature column for {phase}: {c}")
        rate = df[c].isna().mean()
        if rate > 0.30:
            print(f"  [{phase}] dropping {c} (NaN rate {rate:.2%})")
            drop_cols.append(c)
    feats = [c for c in feats if c not in drop_cols]

    train = df[df["split"] == "train"].sort_values(["date", "match_id"]).reset_index(drop=True)
    val = df[df["split"] == "val"].sort_values(["date", "match_id"]).reset_index(drop=True)
    test = df[df["split"] == "test"].sort_values(["date", "match_id"]).reset_index(drop=True)

    X_train, y_train = train[feats], train["winner_is_team1"]
    X_val, y_val = val[feats], val["winner_is_team1"]
    X_test, y_test = test[feats], test["winner_is_team1"]

    print(f"[{phase}] train={len(train)} val={len(val)} test={len(test)} features={len(feats)}", flush=True)

    # Hyperparam search on n_estimators via TimeSeriesSplit
    n_grid = [200] if fast else [100, 200, 400, 600]
    best_n, best_score = None, -np.inf
    for n_est in n_grid:
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for ti, vi in tscv.split(X_train):
            m = lgb.LGBMClassifier(
                n_estimators=n_est, max_depth=5, learning_rate=0.05,
                num_leaves=31, random_state=42, n_jobs=2, verbose=-1,
            )
            m.fit(X_train.iloc[ti], y_train.iloc[ti])
            p = m.predict_proba(X_train.iloc[vi])[:, 1]
            scores.append(-log_loss(y_train.iloc[vi], np.clip(p, 1e-6, 1 - 1e-6)))
        mean = float(np.mean(scores))
        if mean > best_score:
            best_score, best_n = mean, n_est

    print(f"  best n_estimators={best_n} cv_neg_log_loss={best_score:.4f}", flush=True)

    base = lgb.LGBMClassifier(
        n_estimators=best_n, max_depth=5, learning_rate=0.05,
        num_leaves=31, random_state=42, n_jobs=2, verbose=-1,
    )
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

    importance = sorted(
        zip(feats, base.booster_.feature_importance(importance_type="gain").tolist()),
        key=lambda x: -x[1],
    )

    hit_kill = val_acc < KILL_THRESHOLD

    model_path = MODELS / f"v{version}_{phase}.pkl"
    meta_path = MODELS / f"v{version}_{phase}.json"
    if model_path.exists():
        raise RuntimeError(f"refusing to overwrite {model_path}")

    _atomic_pickle(
        {
            "calibrator": cal,
            "feature_names": feats,
            "phase": phase,
            "hit_kill_criterion": hit_kill,
        },
        model_path,
    )
    meta = {
        "version": version,
        "name": phase,
        "phase": phase,
        "model_type": "lightgbm_isotonic_prefit",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": _git_sha(),
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "best_n_estimators": best_n,
        "cv_mean_neg_log_loss": best_score,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "validation_accuracy": val_acc,
        "validation_brier": val_brier,
        "validation_log_loss": val_ll,
        "test_accuracy": test_acc,
        "test_brier": test_brier,
        "test_log_loss": test_ll,
        "feature_names": feats,
        "dropped_high_nan_features": drop_cols,
        "feature_importance_gain_ranked": importance,
        "hyperparameters": {
            "n_estimators": best_n, "max_depth": 5, "learning_rate": 0.05,
            "num_leaves": 31, "random_state": 42,
        },
        "kill_threshold": KILL_THRESHOLD,
        "hit_kill_criterion": hit_kill,
    }
    _atomic_json(meta, meta_path)
    print(f"  saved {model_path}")
    print(f"  val 2024: acc={val_acc:.3f} brier={val_brier:.3f}")
    print(f"  test 2025: acc={test_acc:.3f} brier={test_brier:.3f}")
    if hit_kill:
        print(f"  WARNING: val acc {val_acc:.3f} < kill threshold {KILL_THRESHOLD:.3f}")
    return meta


PHASES = ["phase_pre_match", "phase_post_toss", "phase_post_pp1", "phase_innings_break", "player"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES + ["all"], default="all")
    ap.add_argument("--version", type=int, default=6)
    ap.add_argument("--fast", action="store_true", help="Skip hyperparam grid; use n_estimators=200")
    args = ap.parse_args()
    phases = PHASES if args.phase == "all" else [args.phase]
    results = []
    for p in phases:
        print(f"\n=== {p} ===")
        results.append(train_one(p, args.version, fast=args.fast))
    print("\nSummary:")
    for r in results:
        flag = " KILL" if r["hit_kill_criterion"] else ""
        print(f"  {r['phase']:22s} val={r['validation_accuracy']:.3f} test={r['test_accuracy']:.3f} brier={r['test_brier']:.3f}{flag}")


if __name__ == "__main__":
    main()
