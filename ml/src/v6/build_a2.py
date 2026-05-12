"""A2 — venue_patterns.parquet

Per-match snapshot of the venue's prior statistics:
  - avg 1st-innings total at this venue (rolling window of last 20 matches there)
  - wicket-fall rate by phase (avg wickets per phase per match at the venue)
  - pace vs spin economy split — we don't have bowler-style data in
    balls.parquet, so use a proxy: bowler career-economy quartile at point in
    time. Bowlers in the top half of overall career economy at the moment of
    the match are tagged 'expensive' (rough pace/aggressive proxy) and bottom
    half 'economical' (rough spin/containment proxy). Document the limitation.

Columns (per spec):
  match_id, venue_avg_1st_innings, venue_wickets_per_pp,
  venue_wickets_per_middle, venue_wickets_per_death,
  venue_pace_econ, venue_spin_econ, venue_match_count
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from ml.src.v6._io import (
    V6_DIR,
    atomic_write_parquet,
    load_balls,
    load_matches,
    phase_of_over,
)

OUT = V6_DIR / "venue_patterns.parquet"
WINDOW = 20


def build() -> pd.DataFrame:
    matches = load_matches()
    balls = load_balls()
    balls = balls[balls["match_id"].isin(matches["match_id"])].copy()
    balls["phase"] = balls["over"].map(phase_of_over)

    # Pre-aggregate per-match summaries we'll need
    first_innings_total = (
        balls[balls["innings"] == 1]
        .groupby("match_id")["total_runs"].sum()
        .to_dict()
    )

    # Wickets per phase per match (counting only deliveries where player_out is set)
    wkts = balls[balls["player_out"].notna()].copy()
    wkts_per_match_phase = (
        wkts.groupby(["match_id", "phase"]).size().unstack(fill_value=0)
    )
    for col in ("pp", "middle", "death"):
        if col not in wkts_per_match_phase.columns:
            wkts_per_match_phase[col] = 0

    # For pace/spin proxy: compute per-bowler running career economy snapshot
    # before each match. To approximate "pace vs spin" we'll classify each
    # bowler at the moment of the match by whether their career economy is
    # >= the global median bowler economy seen so far. We then compute the
    # venue's prior-match average economy from "high-econ" vs "low-econ"
    # bowlers separately.
    bowler_runs = defaultdict(int)
    bowler_balls = defaultdict(int)

    matches_sorted = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    balls_by_match = {mid: sub for mid, sub in balls.groupby("match_id", sort=False)}

    # Per-venue rolling state
    venue_first_innings_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
    venue_wkts_pp_sum: dict[str, int] = defaultdict(int)
    venue_wkts_mid_sum: dict[str, int] = defaultdict(int)
    venue_wkts_death_sum: dict[str, int] = defaultdict(int)
    venue_match_count: dict[str, int] = defaultdict(int)
    venue_pace_runs: dict[str, int] = defaultdict(int)
    venue_pace_balls: dict[str, int] = defaultdict(int)
    venue_spin_runs: dict[str, int] = defaultdict(int)
    venue_spin_balls: dict[str, int] = defaultdict(int)

    rows = []
    for _, m in matches_sorted.iterrows():
        mid = m["match_id"]
        venue = m.get("venue", "")
        count = venue_match_count[venue]
        avg_1st = (
            float(np.mean(venue_first_innings_history[venue]))
            if venue_first_innings_history[venue]
            else np.nan
        )
        rows.append({
            "match_id": mid,
            "venue_avg_1st_innings": avg_1st,
            "venue_wickets_per_pp": (venue_wkts_pp_sum[venue] / count) if count else np.nan,
            "venue_wickets_per_middle": (venue_wkts_mid_sum[venue] / count) if count else np.nan,
            "venue_wickets_per_death": (venue_wkts_death_sum[venue] / count) if count else np.nan,
            "venue_pace_econ": (
                6.0 * venue_pace_runs[venue] / venue_pace_balls[venue]
                if venue_pace_balls[venue] else np.nan
            ),
            "venue_spin_econ": (
                6.0 * venue_spin_runs[venue] / venue_spin_balls[venue]
                if venue_spin_balls[venue] else np.nan
            ),
            "venue_match_count": count,
        })

        # Update with this match's data
        sub = balls_by_match.get(mid)
        if sub is None or not len(sub):
            continue
        # First innings total
        fi = first_innings_total.get(mid)
        if fi is not None:
            venue_first_innings_history[venue].append(fi)
        # Wickets per phase
        if mid in wkts_per_match_phase.index:
            row = wkts_per_match_phase.loc[mid]
            venue_wkts_pp_sum[venue] += int(row.get("pp", 0))
            venue_wkts_mid_sum[venue] += int(row.get("middle", 0))
            venue_wkts_death_sum[venue] += int(row.get("death", 0))
        venue_match_count[venue] += 1

        # Pace vs spin proxy: at the moment of this match, classify each bowler
        # by whether their *current* career economy is above or below the
        # global-median career econ across all bowlers with >= 30 balls bowled.
        # For each ball in this match, attribute to pace or spin bucket.
        eligible = [
            (b, 6.0 * bowler_runs[b] / bowler_balls[b])
            for b in bowler_balls
            if bowler_balls[b] >= 30
        ]
        if eligible:
            median_econ = float(np.median([e for _, e in eligible]))
        else:
            median_econ = np.nan

        def bowler_class(b):
            if bowler_balls[b] < 30 or np.isnan(median_econ):
                return None
            cur = 6.0 * bowler_runs[b] / bowler_balls[b]
            return "pace" if cur >= median_econ else "spin"

        # Vectorize per-bowler aggregation
        per_bowler = sub.groupby("bowler").agg(runs=("total_runs", "sum"), balls=("ball", "size"))
        for bowler, r in per_bowler.iterrows():
            cls = bowler_class(bowler)
            if cls == "pace":
                venue_pace_runs[venue] += int(r["runs"])
                venue_pace_balls[venue] += int(r["balls"])
            elif cls == "spin":
                venue_spin_runs[venue] += int(r["runs"])
                venue_spin_balls[venue] += int(r["balls"])
            # else: skip (not yet enough data to classify)

        # Update bowler career totals
        for bowler, r in per_bowler.iterrows():
            bowler_runs[bowler] += int(r["runs"])
            bowler_balls[bowler] += int(r["balls"])

    df = pd.DataFrame(rows)
    return df


def main():
    df = build()
    atomic_write_parquet(df, OUT)
    print(f"[A2] venue_patterns: rows={len(df)} cols={len(df.columns)} -> {OUT}")


if __name__ == "__main__":
    main()
