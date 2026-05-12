"""A5 — toss_conditioned.parquet

Venue chase rate conditioned on toss decision (bat vs field).

For each match:
  venue_chase_rate_when_fielded = of prior matches at this venue where the
    toss winner elected to FIELD, what fraction were won by chase (win_by_wickets
    not null).
  venue_chase_rate_when_batted = same but toss winner elected to BAT.
  venue_total_matches = total prior matches at this venue.

NaN for any (venue, decision) bucket with zero prior matches.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ml.src.v6._io import V6_DIR, atomic_write_parquet, load_matches

OUT = V6_DIR / "toss_conditioned.parquet"


def build() -> pd.DataFrame:
    matches = load_matches()

    field_chase_wins: dict[str, int] = defaultdict(int)
    field_total: dict[str, int] = defaultdict(int)
    bat_chase_wins: dict[str, int] = defaultdict(int)
    bat_total: dict[str, int] = defaultdict(int)
    venue_total: dict[str, int] = defaultdict(int)

    rows = []
    for _, m in matches.iterrows():
        venue = m.get("venue", "")
        field_n = field_total[venue]
        bat_n = bat_total[venue]
        rows.append({
            "match_id": m["match_id"],
            "venue_chase_rate_when_fielded": (
                field_chase_wins[venue] / field_n if field_n else np.nan
            ),
            "venue_chase_rate_when_batted": (
                bat_chase_wins[venue] / bat_n if bat_n else np.nan
            ),
            "venue_total_matches": venue_total[venue],
        })

        # Update with this match's outcome
        if m.get("no_result", False):
            continue
        decision = m.get("toss_decision")
        was_chase = pd.notna(m.get("win_by_wickets"))
        venue_total[venue] += 1
        if decision == "field":
            field_total[venue] += 1
            if was_chase:
                field_chase_wins[venue] += 1
        elif decision == "bat":
            bat_total[venue] += 1
            if was_chase:
                bat_chase_wins[venue] += 1

    return pd.DataFrame(rows)


def main():
    df = build()
    atomic_write_parquet(df, OUT)
    print(f"[A5] toss_conditioned: rows={len(df)} cols={len(df.columns)} -> {OUT}")


if __name__ == "__main__":
    main()
