"""v6 Phase D2 — drift monitor.

Loads the v5 stacked-ensemble (or v7, once it exists) and scores every 2026
match that already has a recorded winner. Reports rolling accuracy by month
plus a cumulative season-to-date number, and compares to v5's training/val
baseline of 54.9% (from v5_ensemble.json's `train_accuracy`, which is the v5
meta-learner's fit on 2024).

Exit codes:
  0 — cumulative 2026 accuracy is within 5 percentage points of the baseline
  1 — cumulative 2026 accuracy has dropped > 5 percentage points

CLI:
    ml/.venv/bin/python -m ml.src.v6.drift_monitor
    ml/.venv/bin/python -m ml.src.v6.drift_monitor --ensemble v5_ensemble --baseline 0.549
    ml/.venv/bin/python -m ml.src.v6.drift_monitor --json   # machine-readable

Output is launchd-loggable: a structured-but-grep-friendly stdout block,
prefixed `[drift_monitor]` so it stands out in the log file.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
HIST = ROOT / "ml" / "data" / "historical"
MODELS = ROOT / "ml" / "data" / "models"

# Models we need to score the v5 ensemble against an arbitrary slice of matches.
# v5_ensemble's meta-learner takes [p_v1, p_v3, p_v4] -> p_team1.
_BASE_MODEL_FILES = {
    "v1": MODELS / "v1_logistic.pkl",
    "v3": MODELS / "v3_phase_pre_match.pkl",
    "v4": MODELS / "v4_player_gbm.pkl",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_pickle(path: str):
    # Trusted artifacts only — produced by our training scripts under ml/data/models/.
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec B301


def _v1_predict_proba(df_v1_features: pd.DataFrame) -> np.ndarray:
    art = _load_pickle(str(_BASE_MODEL_FILES["v1"]))
    # v1 is a saved CalibratedClassifierCV directly.
    return art.predict_proba(df_v1_features.to_numpy())[:, 1]


def _v3_predict_proba(df_v3_features: pd.DataFrame) -> np.ndarray:
    art = _load_pickle(str(_BASE_MODEL_FILES["v3"]))
    cols = art["feature_names"]
    return art["calibrator"].predict_proba(df_v3_features[cols].to_numpy())[:, 1]


def _v4_predict_proba(df_v4_features: pd.DataFrame) -> np.ndarray:
    art = _load_pickle(str(_BASE_MODEL_FILES["v4"]))
    cols = art["feature_names"]
    X = df_v4_features[cols].astype(float).fillna(0.0).to_numpy()
    return art["calibrator"].predict_proba(X)[:, 1]


def _ensemble_predict(p_v1: np.ndarray, p_v3: np.ndarray, p_v4: np.ndarray,
                       ensemble_name: str) -> np.ndarray:
    art = _load_pickle(str(MODELS / f"{ensemble_name}.pkl"))
    X = np.column_stack([p_v1, p_v3, p_v4])
    return art["meta"].predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Feature rebuild — point-in-time, walks ALL matches (2008..2026) once.
# ---------------------------------------------------------------------------

def _build_pre_match_pit(matches: pd.DataFrame) -> pd.DataFrame:
    """Same PIT logic as phase_features._build_pre_match_clean, but accepts the
    full matches frame (no season filter) so 2026 rows get features computed
    using only data from earlier matches."""
    df = matches.sort_values(["date", "match_id"]).reset_index(drop=True).copy()
    career_w, career_p = {}, {}
    season_w, season_p = {}, {}
    last5: dict = {}
    venue_team_w, venue_team_p = {}, {}
    venue_chase_w, venue_p = {}, {}

    def wr(w, p, k, default=0.5):
        if p.get(k, 0) == 0:
            return default
        return w.get(k, 0) / p[k]

    rows = []
    for _, m in df.iterrows():
        t1, t2 = m["team1"], m["team2"]
        season = int(m["season"]) if pd.notna(m["season"]) else -1
        venue = m.get("venue", "") or ""
        rows.append({
            "match_id": m["match_id"],
            "season": season,
            "date": m["date"],
            "venue": venue,
            "team1": t1,
            "team2": t2,
            "team1_career_wr": wr(career_w, career_p, t1),
            "team2_career_wr": wr(career_w, career_p, t2),
            "team1_career_played": career_p.get(t1, 0),
            "team2_career_played": career_p.get(t2, 0),
            "team1_season_wr": wr(season_w, season_p, (season, t1)),
            "team2_season_wr": wr(season_w, season_p, (season, t2)),
            "team1_season_played": season_p.get((season, t1), 0),
            "team2_season_played": season_p.get((season, t2), 0),
            "team1_form5": float(np.mean(last5.get(t1, [0.5])[-5:])) if last5.get(t1) else 0.5,
            "team2_form5": float(np.mean(last5.get(t2, [0.5])[-5:])) if last5.get(t2) else 0.5,
            "team1_venue_wr": wr(venue_team_w, venue_team_p, (venue, t1)),
            "team2_venue_wr": wr(venue_team_w, venue_team_p, (venue, t2)),
            "venue_chase_rate": (venue_chase_w.get(venue, 0) / venue_p[venue]) if venue_p.get(venue, 0) > 0 else 0.5,
            "venue_played": venue_p.get(venue, 0),
            "winner": m.get("winner"),
            "no_result": bool(m.get("no_result", False)),
        })
        if pd.notna(m["winner"]) and not m.get("no_result", False):
            w = m["winner"]
            for t in (t1, t2):
                career_p[t] = career_p.get(t, 0) + 1
                season_p[(season, t)] = season_p.get((season, t), 0) + 1
                venue_team_p[(venue, t)] = venue_team_p.get((venue, t), 0) + 1
                last5.setdefault(t, []).append(1 if t == w else 0)
            career_w[w] = career_w.get(w, 0) + 1
            season_w[(season, w)] = season_w.get((season, w), 0) + 1
            venue_team_w[(venue, w)] = venue_team_w.get((venue, w), 0) + 1
            venue_p[venue] = venue_p.get(venue, 0) + 1
            if pd.notna(m.get("win_by_wickets")):
                venue_chase_w[venue] = venue_chase_w.get(venue, 0) + 1
    return pd.DataFrame(rows)


def _build_v1_features(pre: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """v1 uses a different feature set (diff-style) — reuse features.py."""
    from ml.src.features import build_features_dataset
    full = build_features_dataset(matches)
    # build_features_dataset assigns split based on season; we don't care about split,
    # we just want the v1 9-feature columns indexed by match_id.
    return full


def _build_player_features_pit(matches: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """Reuse player_features._pit_balls + build_player_features on the FULL frame
    so 2026 rows get player aggregates."""
    from ml.src.player_features import _pit_balls, build_player_features
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    b = balls[balls["match_id"].isin(m["match_id"])].copy()
    balls_pit = _pit_balls(b, m)
    pf = build_player_features(m, balls_pit)
    pf["date"] = pd.to_datetime(pf["date"])
    return pf


# ---------------------------------------------------------------------------
# Main scoring & reporting
# ---------------------------------------------------------------------------

def score_season(season: int, ensemble: str) -> pd.DataFrame:
    """Return a DataFrame of (match_id, date, predicted_team1_p, predicted_winner,
    actual_winner, correct) for every completed match in `season`."""
    matches = pd.read_parquet(HIST / "matches.parquet")
    balls = pd.read_parquet(HIST / "balls.parquet")
    matches = matches.dropna(subset=["team1", "team2", "season"]).copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    # PIT pre-match features (full history, then filter to season)
    pre = _build_pre_match_pit(matches)
    pre["date"] = pd.to_datetime(pre["date"])

    # v1 features (the legacy 9 diff features)
    v1_full = _build_v1_features(pre, matches)
    v1_full["date"] = pd.to_datetime(v1_full["date"])

    # Player aggregates (PIT)
    pf = _build_player_features_pit(matches, balls)

    # Target slice: this season, completed (winner known, not no_result)
    season_matches = matches[
        (matches["season"] == season)
        & matches["winner"].notna()
        & (~matches["no_result"].fillna(False))
    ].copy()

    if season_matches.empty:
        return pd.DataFrame(columns=["match_id", "date", "p_team1", "predicted_winner",
                                     "actual_winner", "correct", "month"])

    # Assemble feature frames keyed on match_id.
    pre_s = pre[pre["match_id"].isin(season_matches["match_id"])].copy()
    pf_s = pf[pf["match_id"].isin(season_matches["match_id"])].copy()

    v3_features = ["team1_career_wr", "team2_career_wr",
                   "team1_career_played", "team2_career_played",
                   "team1_season_wr", "team2_season_wr",
                   "team1_season_played", "team2_season_played",
                   "team1_form5", "team2_form5",
                   "team1_venue_wr", "team2_venue_wr",
                   "venue_chase_rate", "venue_played"]

    df = pre_s.merge(pf_s.drop(columns=["date", "season"], errors="ignore"),
                     on="match_id", how="left")

    # Align v1 feature frame to the same match_ids.
    v1_features = ["form_last5_diff", "season_nrr_diff", "squad_strength_diff",
                   "home_advantage_diff", "h2h_season_diff", "venue_chase_rate",
                   "pom_recency_diff", "qualified_flag_diff",
                   "strength_of_schedule_diff"]
    v1_s = v1_full[v1_full["match_id"].isin(season_matches["match_id"])][
        ["match_id"] + v1_features].copy()
    df = df.merge(v1_s, on="match_id", how="left", suffixes=("", "_v1"))

    # Handle the duplicate `venue_chase_rate` column from the v1 join (suffixed).
    # v1 uses its own venue_chase_rate (which is the no-op constant 0.0) — keep the
    # v3 column for v3/v4 scoring and use the suffixed one for v1.
    if "venue_chase_rate_v1" in df.columns:
        v1_cols = [c if c != "venue_chase_rate" else "venue_chase_rate_v1"
                   for c in v1_features]
    else:
        v1_cols = v1_features

    # Drop any rows that ended up without v1 features (shouldn't happen given the
    # full-history build, but guard anyway).
    df = df.dropna(subset=v1_cols).reset_index(drop=True)

    # Score each base model.
    p_v1 = _v1_predict_proba(df[v1_cols].astype(float).fillna(0.0))
    p_v3 = _v3_predict_proba(df[v3_features].astype(float).fillna(0.0))
    p_v4 = _v4_predict_proba(df)
    p_meta = _ensemble_predict(p_v1, p_v3, p_v4, ensemble)

    df["p_team1"] = p_meta
    df["predicted_winner"] = np.where(df["p_team1"] >= 0.5, df["team1"], df["team2"])
    df["actual_winner"] = df["match_id"].map(
        season_matches.set_index("match_id")["winner"].to_dict()
    )
    df["correct"] = (df["predicted_winner"] == df["actual_winner"]).astype(int)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    return df[["match_id", "date", "month", "p_team1",
               "predicted_winner", "actual_winner", "correct"]]


def build_report(scored: pd.DataFrame, baseline: float) -> dict:
    """Aggregate to monthly + cumulative; compute drift vs baseline."""
    if scored.empty:
        return {
            "n_matches": 0,
            "cumulative_accuracy": None,
            "baseline_accuracy": baseline,
            "drift_pp": None,
            "monthly": [],
            "verdict": "no completed matches yet",
        }

    monthly = (scored.groupby("month")
                     .agg(n=("correct", "size"), acc=("correct", "mean"))
                     .reset_index()
                     .sort_values("month"))

    cum_acc = float(scored["correct"].mean())
    drift_pp = (cum_acc - baseline) * 100.0  # negative => model is worse than baseline

    if drift_pp >= -5.0:
        verdict = "within tolerance (drift <= 5pp)"
    else:
        verdict = "ALERT: accuracy dropped > 5pp vs baseline"

    return {
        "n_matches": int(len(scored)),
        "cumulative_accuracy": cum_acc,
        "baseline_accuracy": baseline,
        "drift_pp": drift_pp,
        "monthly": [
            {"month": r["month"], "n": int(r["n"]), "accuracy": float(r["acc"])}
            for _, r in monthly.iterrows()
        ],
        "verdict": verdict,
    }


def format_report(report: dict, season: int, ensemble: str) -> str:
    lines = []
    lines.append(f"[drift_monitor] ensemble={ensemble} season={season} "
                 f"n_matches={report['n_matches']}")
    if report["n_matches"] == 0:
        lines.append(f"[drift_monitor] {report['verdict']}")
        return "\n".join(lines)
    base = report["baseline_accuracy"]
    cum = report["cumulative_accuracy"]
    drift = report["drift_pp"]
    lines.append(f"[drift_monitor] baseline_acc={base:.3f} "
                 f"cumulative_acc={cum:.3f} drift={drift:+.1f}pp")
    lines.append("[drift_monitor] monthly:")
    for m in report["monthly"]:
        lines.append(f"[drift_monitor]   {m['month']}: n={m['n']:>2d} acc={m['accuracy']:.3f}")
    lines.append(f"[drift_monitor] verdict: {report['verdict']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="v6 drift monitor — check ensemble accuracy on current season")
    ap.add_argument("--season", type=int, default=2026,
                    help="Season to monitor (default: 2026)")
    ap.add_argument("--ensemble", default="v5_ensemble",
                    help="Ensemble pkl name without extension (default: v5_ensemble; "
                         "switch to v7_ensemble once it exists)")
    ap.add_argument("--baseline", type=float, default=0.549,
                    help="Baseline accuracy to compare against. Default 0.549 = "
                         "v5_ensemble train_accuracy from v5_ensemble.json")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of human-readable text")
    args = ap.parse_args()

    scored = score_season(args.season, args.ensemble)
    report = build_report(scored, args.baseline)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_report(report, args.season, args.ensemble))

    # Exit code: 0 if within tolerance OR no completed matches; 1 if drifted.
    if report["n_matches"] == 0:
        return 0
    if report["drift_pp"] is None or report["drift_pp"] >= -5.0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
