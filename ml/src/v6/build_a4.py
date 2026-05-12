"""A4 — h2h_career.parquet

Career H2H + last-5-meetings H2H between the two teams.

Columns:
  match_id, h2h_career_team1_winrate, h2h_last5_team1_winrate
NaN where no prior meetings exist between the two teams.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from ml.src.v6._io import V6_DIR, atomic_write_parquet, load_matches

OUT = V6_DIR / "h2h_career.parquet"


def _pair_key(a: str, b: str) -> tuple:
    return tuple(sorted([a, b]))


def build() -> pd.DataFrame:
    matches = load_matches()

    pair_wins: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pair_played: dict[tuple, int] = defaultdict(int)
    # last5 stores tuple (winner, ) for the most recent 5 meetings
    pair_last5: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=5))

    rows = []
    for _, m in matches.iterrows():
        t1, t2 = m["team1"], m["team2"]
        key = _pair_key(t1, t2)
        played = pair_played[key]
        if played == 0:
            career_t1 = np.nan
        else:
            career_t1 = pair_wins[key].get(t1, 0) / played
        last5 = list(pair_last5[key])
        if not last5:
            last5_t1 = np.nan
        else:
            last5_t1 = sum(1 for w in last5 if w == t1) / len(last5)

        rows.append({
            "match_id": m["match_id"],
            "h2h_career_team1_winrate": career_t1,
            "h2h_last5_team1_winrate": last5_t1,
        })

        # Update with this match's result
        if m.get("no_result", False) or pd.isna(m.get("winner")):
            continue
        w = m["winner"]
        pair_played[key] += 1
        pair_wins[key][w] += 1
        pair_last5[key].append(w)

    return pd.DataFrame(rows)


def main():
    df = build()
    atomic_write_parquet(df, OUT)
    print(f"[A4] h2h_career: rows={len(df)} cols={len(df.columns)} -> {OUT}")


if __name__ == "__main__":
    main()
