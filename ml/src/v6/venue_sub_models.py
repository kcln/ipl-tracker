"""v6 Phase D3 — per-venue sub-model experiment.

For each of the top-3 most-played venues, train a venue-specific LightGBM
using only matches at that venue (still point-in-time within the train
split), then compare its 2025 test-set accuracy against the overall
v3_phase_pre_match model evaluated on the same venue-subset.

Phase A's richer features aren't all wired up yet — this experiment uses the
v3_phase_pre_match feature set as a baseline. That's the same comparison
universe as the overall model, so the lift number is apples-to-apples.

Decision rule (V6_PLAN.md): commit the sub-model artifact only if it shows
> 2 percentage points of lift over the overall model on that venue.

Output:
  ml/docs/v6_venue_experiment.md   markdown summary (always)
  ml/data/models/v6_venue_<slug>.pkl   only if lift > 2pp

CLI:
    ml/.venv/bin/python -m ml.src.v6.venue_sub_models
    ml/.venv/bin/python -m ml.src.v6.venue_sub_models --top-n 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import pickle
import re
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss

ROOT = pathlib.Path(__file__).resolve().parents[3]
HIST = ROOT / "ml" / "data" / "historical"
MODELS = ROOT / "ml" / "data" / "models"
DOCS = ROOT / "ml" / "docs"

# Use the v3_phase_pre_match feature set so the comparison vs the overall
# v3 model is on the same feature space.
V3_FEATURES = [
    "team1_career_wr", "team2_career_wr",
    "team1_career_played", "team2_career_played",
    "team1_season_wr", "team2_season_wr",
    "team1_season_played", "team2_season_played",
    "team1_form5", "team2_form5",
    "team1_venue_wr", "team2_venue_wr",
    "venue_chase_rate", "venue_played",
]

LIFT_THRESHOLD_PP = 2.0  # commit sub-model only if lift > +2pp on venue test


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _atomic_write(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".pkl":
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
    elif path.suffix == ".json":
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    else:
        if isinstance(obj, str):
            tmp.write_text(obj)
        else:
            raise ValueError(f"unsupported suffix {path.suffix}")
    os.replace(tmp, path)


def _slug(venue: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", venue).strip("_").lower()
    return s[:60] or "venue"


def _load_overall_v3():
    with open(MODELS / "v3_phase_pre_match.pkl", "rb") as f:
        return pickle.load(f)  # nosec B301


def _train_venue_lgbm(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Train a calibrated LightGBM on a single venue's matches.
    Uses prefit isotonic calibration on the venue's val slice (if non-empty),
    otherwise falls back to cv=3 isotonic on train."""
    X_train = train_df[V3_FEATURES].astype(float).fillna(0.0).to_numpy()
    y_train = train_df["winner_is_team1"].astype(int).to_numpy()
    base = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                              num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    base.fit(X_train, y_train)
    if len(val_df) >= 5:
        X_val = val_df[V3_FEATURES].astype(float).fillna(0.0).to_numpy()
        y_val = val_df["winner_is_team1"].astype(int).to_numpy()
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
    else:
        # Very small venue val — fall back to CV-isotonic on train. Use
        # min(3, smallest_class) folds so CalibratedClassifierCV doesn't
        # blow up on tiny imbalanced venue slices.
        min_class = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
        cv_folds = max(2, min(3, min_class))
        if min_class < 2:
            # Degenerate venue split — return uncalibrated base wrapped to
            # match the calibrator interface.
            cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
            # "prefit" with no validation data: use train as a stand-in (this
            # is mostly noise but keeps the experiment runnable).
            cal.fit(X_train, y_train)
        else:
            cal = CalibratedClassifierCV(base, method="isotonic", cv=cv_folds)
            cal.fit(X_train, y_train)
    return cal


def _score(cal, df: pd.DataFrame) -> tuple[float, float]:
    if len(df) == 0:
        return float("nan"), float("nan")
    X = df[V3_FEATURES].astype(float).fillna(0.0).to_numpy()
    y = df["winner_is_team1"].astype(int).to_numpy()
    p = cal.predict_proba(X)[:, 1]
    return float(accuracy_score(y, (p >= 0.5).astype(int))), float(brier_score_loss(y, p))


def run(top_n: int = 3) -> dict:
    df = pd.read_parquet(HIST / "features_pre_match.parquet")

    # Pick top-N venues that have BOTH meaningful train and test history.
    # IPL venue names mutate over time (e.g. "Wankhede Stadium" vs
    # "Wankhede Stadium, Mumbai"), so naive `value_counts()` picks legacy
    # spellings with zero 2025 test rows. We pick by 2025-test-row count
    # and require at least some training history. This is what V6_PLAN.md
    # actually wants — comparable test slices.
    venue_test = df[df["split"] == "test"]["venue"].value_counts()
    venue_train = df[df["split"] == "train"]["venue"].value_counts()
    candidates = venue_test[(venue_test >= 3)]
    candidates = candidates[candidates.index.map(
        lambda v: venue_train.get(v, 0) >= 5
    )]
    top_venues = candidates.head(top_n).index.tolist()

    overall = _load_overall_v3()
    overall_cal = overall["calibrator"]

    rows = []
    saved_models = []
    for venue in top_venues:
        venue_df = df[df["venue"] == venue].copy()
        train_df = venue_df[venue_df["split"] == "train"]
        val_df = venue_df[venue_df["split"] == "val"]
        test_df = venue_df[venue_df["split"] == "test"]

        n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)

        # Overall v3 on this venue's test slice
        ov_test_acc, ov_test_brier = _score(overall_cal, test_df)

        # Skip training if there's nothing to test on, or if train is unusably tiny
        if n_test < 3 or n_train < 5:
            rows.append({
                "venue": venue, "n_train": n_train, "n_val": n_val, "n_test": n_test,
                "overall_test_acc": ov_test_acc,
                "venue_test_acc": float("nan"),
                "lift_pp": float("nan"),
                "committed": False,
                "note": "insufficient train/test rows",
            })
            continue

        sub = _train_venue_lgbm(train_df, val_df)
        v_test_acc, v_test_brier = _score(sub, test_df)

        if np.isnan(ov_test_acc):
            lift = float("nan")
        else:
            lift = (v_test_acc - ov_test_acc) * 100.0

        committed = bool(np.isfinite(lift) and lift > LIFT_THRESHOLD_PP)
        if committed:
            slug = _slug(venue)
            model_path = MODELS / f"v6_venue_{slug}.pkl"
            meta_path = MODELS / f"v6_venue_{slug}.json"
            if not model_path.exists():
                _atomic_write({
                    "calibrator": sub,
                    "feature_names": V3_FEATURES,
                    "venue": venue,
                    "decision_rule": f"lift > {LIFT_THRESHOLD_PP}pp",
                }, model_path)
                _atomic_write({
                    "version": 6,
                    "name": f"venue_{slug}",
                    "venue": venue,
                    "model_type": "lightgbm_isotonic_venue_sub_model",
                    "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "git_sha": _git_sha(),
                    "sklearn_version": sklearn.__version__,
                    "lightgbm_version": lgb.__version__,
                    "n_train": n_train, "n_val": n_val, "n_test": n_test,
                    "overall_test_accuracy_on_venue": ov_test_acc,
                    "venue_test_accuracy": v_test_acc,
                    "lift_pp": lift,
                    "feature_names": V3_FEATURES,
                }, meta_path)
                saved_models.append(str(model_path.relative_to(ROOT)))

        rows.append({
            "venue": venue, "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "overall_test_acc": ov_test_acc,
            "venue_test_acc": v_test_acc,
            "overall_test_brier": ov_test_brier,
            "venue_test_brier": v_test_brier,
            "lift_pp": lift,
            "committed": committed,
            "note": "",
        })

    md = _format_markdown(rows, top_venues, saved_models)
    DOCS.mkdir(parents=True, exist_ok=True)
    _atomic_write(md, DOCS / "v6_venue_experiment.md")
    print(md)
    return {"rows": rows, "saved_models": saved_models}


def _format_markdown(rows: list[dict], top_venues: list[str],
                     saved_models: list[str]) -> str:
    lines = []
    lines.append("# Venue sub-model experiment (v6 D3)")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now(dt.UTC).isoformat()}")
    lines.append("")
    lines.append("Compares a per-venue LightGBM (trained only on matches at that")
    lines.append("venue) against the overall v3_phase_pre_match model on the same")
    lines.append("venue's 2025 test slice. Decision rule per V6_PLAN.md: commit")
    lines.append(f"the sub-model only if lift > {LIFT_THRESHOLD_PP}pp.")
    lines.append("")
    lines.append("## Top venues evaluated")
    lines.append("")
    for v in top_venues:
        lines.append(f"- {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| venue | n_train | n_val | n_test | overall_acc | venue_acc | lift (pp) | committed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for r in rows:
        def fmt(x, p="{:.3f}"):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "—"
            return p.format(x)
        committed = "yes" if r["committed"] else "no"
        lines.append(
            f"| {r['venue']} | {r['n_train']} | {r['n_val']} | {r['n_test']} | "
            f"{fmt(r['overall_test_acc'])} | {fmt(r['venue_test_acc'])} | "
            f"{fmt(r['lift_pp'], '{:+.1f}')} | {committed} |"
        )
        if r.get("note"):
            lines.append(f"|  | | | | | | | _{r['note']}_ |")
    lines.append("")
    if saved_models:
        lines.append("## Committed artifacts")
        lines.append("")
        for path in saved_models:
            lines.append(f"- `{path}`")
    else:
        lines.append("## Committed artifacts")
        lines.append("")
        lines.append("_No sub-models hit the +2pp threshold; nothing committed._")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Per-venue sub-model experiment (D3)")
    ap.add_argument("--top-n", type=int, default=3,
                    help="How many of the most-played venues to evaluate (default 3)")
    args = ap.parse_args()
    run(top_n=args.top_n)


if __name__ == "__main__":
    main()
