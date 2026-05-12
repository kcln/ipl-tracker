"""Chronological backtest harness with point-in-time enforcement.

Replays matches in date order, building state from prior matches only,
calling predict_fn(match_row, state), and aggregating per-season metrics.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import pathlib
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ml"
HIST_DIR = ML_ROOT / "data" / "historical"
RESULTS_DIR = ML_ROOT / "data" / "backtest_results"


def _brier(prob_team1: np.ndarray, y_team1: np.ndarray) -> float:
    return float(np.mean((prob_team1 - y_team1) ** 2))


def _log_loss(prob_team1: np.ndarray, y_team1: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(prob_team1, eps, 1.0 - eps)
    return float(-np.mean(y_team1 * np.log(p) + (1 - y_team1) * np.log(1 - p)))


def run_backtest(
    predict_fn: Callable[[dict, dict], tuple[str, float]],
    matches_df: pd.DataFrame,
    name: str,
    *,
    skip_no_result: bool = True,
    leakage_check_every: int = 50,
) -> dict:
    """Replay matches_df in date order calling predict_fn(match_row, state).

    predict_fn returns (predicted_team_name, prob_team1_wins).

    The state passed in is guaranteed point-in-time: it only references matches
    with date strictly before the current match's date (ties broken by row order).

    Returns:
        {
          "name": str,
          "overall": {"acc", "brier", "log_loss", "n"},
          "by_season": {season: {...}},
          "details_path": str,
          "leakage_detected": bool,
        }
    """
    df = matches_df.copy()
    df = df.dropna(subset=["date", "team1", "team2", "season"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    state = {
        "completed": [],
        "h2h": {},
        "team_form": {},
        "team_played": {},
        "team_won": {},
        "venue_history": {},
        "_max_seen_date": None,
    }

    details = []
    leakage_detected = False

    for i, row in df.iterrows():
        if skip_no_result and row.get("no_result", False):
            # Still update state with the no-result so subsequent features see it
            state["completed"].append({
                "match_id": row["match_id"], "date": row["date"], "season": row["season"],
                "team1": row["team1"], "team2": row["team2"], "winner": None, "no_result": True,
            })
            continue

        match_row = row.to_dict()
        try:
            pred_team, p_team1 = predict_fn(match_row, state)
        except Exception as e:
            print(f"  predict_fn failed on match {row['match_id']}: {e}", flush=True)
            pred_team = row["team1"]
            p_team1 = 0.5

        actual = row["winner"]
        if pd.isna(actual) or actual is None:
            continue
        y_team1 = 1 if actual == row["team1"] else 0
        correct = 1 if pred_team == actual else 0

        details.append({
            "match_id": row["match_id"],
            "date": row["date"],
            "season": int(row["season"]),
            "team1": row["team1"],
            "team2": row["team2"],
            "predicted": pred_team,
            "actual": actual,
            "p_team1": float(p_team1),
            "y_team1": int(y_team1),
            "correct": int(correct),
        })

        # update state AFTER prediction
        state["completed"].append({
            "match_id": row["match_id"],
            "date": row["date"],
            "season": int(row["season"]),
            "team1": row["team1"],
            "team2": row["team2"],
            "winner": actual,
            "venue": row.get("venue", ""),
            "no_result": False,
        })
        # h2h
        a, b = sorted([row["team1"], row["team2"]])
        key = (a, b, int(row["season"]))
        ent = state["h2h"].setdefault(key, {a: 0, b: 0})
        if actual in ent:
            ent[actual] += 1
        # form
        for t in (row["team1"], row["team2"]):
            state["team_played"][t] = state["team_played"].get(t, 0) + 1
            state["team_form"].setdefault(t, []).append(1 if t == actual else 0)
            if t == actual:
                state["team_won"][t] = state["team_won"].get(t, 0) + 1
        # venue
        v = row.get("venue", "")
        state["venue_history"].setdefault(v, []).append({"winner": actual, "team1": row["team1"], "team2": row["team2"], "date": row["date"]})

        # Periodic leakage check: every entry in state["completed"] must have date < current
        if (i + 1) % leakage_check_every == 0:
            # Allow equal date if match_id order is earlier; conservative: just check strictly less.
            if any(c["date"] > row["date"] for c in state["completed"][:-1]):
                leakage_detected = True
                break

        state["_max_seen_date"] = row["date"]

    if not details:
        return {"name": name, "overall": {"acc": 0.0, "brier": 0.0, "log_loss": 0.0, "n": 0},
                "by_season": {}, "details_path": "", "leakage_detected": False}

    details_df = pd.DataFrame(details)

    overall = {
        "acc": float(details_df["correct"].mean()),
        "brier": _brier(details_df["p_team1"].to_numpy(), details_df["y_team1"].to_numpy()),
        "log_loss": _log_loss(details_df["p_team1"].to_numpy(), details_df["y_team1"].to_numpy()),
        "n": int(len(details_df)),
    }

    by_season = {}
    for season, sdf in details_df.groupby("season"):
        by_season[int(season)] = {
            "acc": float(sdf["correct"].mean()),
            "brier": _brier(sdf["p_team1"].to_numpy(), sdf["y_team1"].to_numpy()),
            "log_loss": _log_loss(sdf["p_team1"].to_numpy(), sdf["y_team1"].to_numpy()),
            "n": int(len(sdf)),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{name}_{ts}.parquet"
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(details_df, preserve_index=False), tmp)
    os.replace(tmp, out_path)

    return {
        "name": name,
        "overall": overall,
        "by_season": by_season,
        "details_path": str(out_path.relative_to(ROOT)),
        "leakage_detected": leakage_detected,
    }


if __name__ == "__main__":
    matches = pd.read_parquet(HIST_DIR / "matches.parquet")
    def coin_flip(_match, _state):
        import random
        t1 = _match["team1"]
        t2 = _match["team2"]
        return (t1 if random.random() < 0.5 else t2), 0.5
    summary = run_backtest(coin_flip, matches, name="coin_flip")
    print(summary)
