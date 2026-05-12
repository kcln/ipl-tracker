"""Standalone top-4 Monte Carlo simulator for a season.

Given:
  - the current standings (team -> points, nrr, ...)
  - the list of remaining fixtures
  - a predict_match_fn(team1, team2) -> p_team1_wins

Run N bootstrap simulations: sample each remaining fixture outcome from its
predicted probability, recompute final standings, record top-4 finishers.
Return each team's probability of finishing top-4 with a 95% bootstrap CI.

This is parallel infrastructure for future cutover; the existing tracker's
top-4 logic in src/predictor.py is untouched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import defaultdict
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]


def simulate_top4(
    standings: list[dict],
    remaining_fixtures: list[dict],
    predict_fn: Callable[[str, str], float],
    *,
    n_sims: int = 5000,
    seed: int = 42,
) -> dict:
    """standings rows: {team, points, nrr, played}
       remaining_fixtures rows: {team1, team2}

    Returns:
        {
          team: {
            "top4_prob": float,
            "ci95_lo": float, "ci95_hi": float,
            "title_prob": float,
          },
          ...,
          "_meta": {"n_sims": ..., "seed": ...}
        }
    """
    rng = random.Random(seed)
    top4_counts = defaultdict(int)
    title_counts = defaultdict(int)
    per_sim_top4 = []  # list of frozenset(top4 teams) per sim for CI

    teams = [r["team"] for r in standings]

    for _ in range(n_sims):
        sim = {r["team"]: dict(r) for r in standings}
        for fx in remaining_fixtures:
            t1, t2 = fx["team1"], fx["team2"]
            if t1 not in sim or t2 not in sim:
                continue
            p1 = predict_fn(t1, t2)
            winner = t1 if rng.random() < p1 else t2
            loser = t2 if winner == t1 else t1
            sim[winner]["points"] += 2
            sim[winner]["played"] += 1
            sim[loser]["played"] += 1
        ranked = sorted(sim.values(), key=lambda r: (r["points"], r.get("nrr", 0.0)), reverse=True)
        top4 = [r["team"] for r in ranked[:4]]
        per_sim_top4.append(frozenset(top4))
        for t in top4:
            top4_counts[t] += 1
        title_counts[ranked[0]["team"]] += 1

    # Per-team CI via bootstrap subsampling
    out = {}
    n = float(n_sims)
    boot_n = 500
    for t in teams:
        appearances = [(1 if t in s else 0) for s in per_sim_top4]
        # subsample with replacement to get a 95% CI on top-4 prob
        boot_means = []
        for _ in range(boot_n):
            sample = [appearances[rng.randrange(n_sims)] for _ in range(n_sims)]
            boot_means.append(sum(sample) / n_sims)
        boot_means.sort()
        out[t] = {
            "top4_prob": top4_counts[t] / n,
            "title_prob": title_counts[t] / n,
            "ci95_lo": boot_means[int(0.025 * boot_n)],
            "ci95_hi": boot_means[int(0.975 * boot_n)],
        }
    out["_meta"] = {"n_sims": n_sims, "seed": seed}
    return out


def _demo_predict(t1: str, t2: str) -> float:
    """Demo predict: just 0.5. Useful for sanity-checking the simulator."""
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", help="path to standings.json")
    ap.add_argument("--fixtures",  help="path to fixtures.json")
    ap.add_argument("--n-sims",    type=int, default=5000)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    if not args.standings or not args.fixtures:
        # Run a smoke demo
        standings = [
            {"team": "MI",  "points": 14, "nrr":  0.4, "played": 12},
            {"team": "CSK", "points": 14, "nrr":  0.3, "played": 12},
            {"team": "RCB", "points": 12, "nrr":  0.1, "played": 12},
            {"team": "DC",  "points": 12, "nrr":  0.0, "played": 12},
            {"team": "GT",  "points": 10, "nrr": -0.1, "played": 12},
            {"team": "RR",  "points":  8, "nrr": -0.2, "played": 12},
        ]
        fixtures = [
            {"team1": "MI", "team2": "RR"},
            {"team1": "CSK", "team2": "GT"},
            {"team1": "RCB", "team2": "DC"},
        ]
        out = simulate_top4(standings, fixtures, _demo_predict, n_sims=args.n_sims, seed=args.seed)
        print(json.dumps(out, indent=2))
        return

    with open(args.standings) as f:
        standings = json.load(f)
    with open(args.fixtures) as f:
        fixtures = json.load(f)
    out = simulate_top4(standings, fixtures, _demo_predict, n_sims=args.n_sims, seed=args.seed)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
