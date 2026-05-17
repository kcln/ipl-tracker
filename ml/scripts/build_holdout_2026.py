"""Build 2026-only holdout feature parquets for auto-promotion.

The 2025 IPL is a LOCKED test set per ml/CLAUDE.md, so we cannot use it to
drive any automated decision. The 2026 in-season matches, however, are
genuinely out-of-sample (no v3/v6/v10 model has ever seen them). Each
nightly_retrain run rebuilds these from the freshly-ingested cricsheet
parquet, then evaluates candidate vs live on the resulting holdout.

Output (under ml/data/historical/):
    holdout_2026_pre_match.parquet
    holdout_2026_post_toss.parquet
    holdout_2026_post_pp1.parquet
    holdout_2026_innings_break.parquet

Each row mirrors the training parquet schema for that phase, including
`winner_is_team1` so accuracy + Brier can be computed directly.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

# Reuse the existing point-in-time feature builders — same logic as the
# training parquets, just filtered to season 2026 at the end.
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.src.phase_features import (  # noqa: E402
    _atomic_write,
    _build_pre_match_clean,
    _add_post_toss,
    _add_powerplay_signals,
    _add_innings_break,
)

HIST = ROOT / "ml" / "data" / "historical"


def build() -> dict[str, int]:
    matches = pd.read_parquet(HIST / "matches.parquet")
    balls = pd.read_parquet(HIST / "balls.parquet")

    # Include 2008-2026 so state accumulates correctly through 2025 before
    # 2026 features are computed. The existing builder is point-in-time
    # correct — for each row, state contains only data from prior matches.
    matches = matches[matches["season"].between(2008, 2026)].copy()
    balls = balls[balls["match_id"].isin(matches["match_id"])]

    pre = _build_pre_match_clean(matches)
    # Keep only matches with a definitive winner
    pre = pre[(~pre["no_result"]) & pre["winner"].notna()].copy()
    pre["winner_is_team1"] = (pre["winner"] == pre["team1"]).astype(int)

    # Filter to 2026 only AFTER state has accumulated
    pre_2026 = pre[pre["season"] == 2026].copy()

    counts: dict[str, int] = {}
    _atomic_write(pre_2026, HIST / "holdout_2026_pre_match.parquet")
    counts["pre_match"] = len(pre_2026)

    post_toss = _add_post_toss(pre_2026)
    _atomic_write(post_toss, HIST / "holdout_2026_post_toss.parquet")
    counts["post_toss"] = len(post_toss)

    post_pp1 = _add_powerplay_signals(post_toss, balls)
    # Drop rows where powerplay data couldn't be derived from balls.parquet
    post_pp1_complete = post_pp1.dropna(subset=["pp1_runs", "pp1_wickets"], how="all")
    _atomic_write(post_pp1_complete, HIST / "holdout_2026_post_pp1.parquet")
    counts["post_pp1"] = len(post_pp1_complete)

    inn_break = _add_innings_break(post_pp1, balls)
    inn_break_complete = inn_break.dropna(subset=["first_innings_total"]) \
        if "first_innings_total" in inn_break.columns else inn_break
    _atomic_write(inn_break_complete, HIST / "holdout_2026_innings_break.parquet")
    counts["innings_break"] = len(inn_break_complete)

    return counts


def main():
    counts = build()
    for phase, n in counts.items():
        print(f"holdout_2026_{phase}.parquet: {n} rows")


if __name__ == "__main__":
    main()
