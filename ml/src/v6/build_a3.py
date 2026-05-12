"""A3 — recency_form.parquet

Exponential-decay form over last 10 matches per team (alpha=0.7).
Weight of most recent match = 1.0, second most recent = 0.7, third = 0.49, ...

`team_decay_form` is the weighted mean of (1 if won, 0 if lost, 0.5 if no_result)
over up to the last 10 matches before the current match, weights = alpha**k
where k=0 is most recent.

Columns: match_id, team1_decay_form, team2_decay_form, team1_decay_form_diff
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from ml.src.v6._io import V6_DIR, atomic_write_parquet, load_matches

OUT = V6_DIR / "recency_form.parquet"
ALPHA = 0.7
WINDOW = 10


def _decay(history) -> float:
    """history is iterable of {0, 0.5, 1.0}, oldest first."""
    if not history:
        return np.nan
    seq = list(history)
    # We want most-recent weight = 1.0, so reverse so seq[0] is most recent
    seq = list(reversed(seq))[:WINDOW]
    weights = np.array([ALPHA**k for k in range(len(seq))])
    return float(np.dot(weights, seq) / weights.sum())


def build() -> pd.DataFrame:
    matches = load_matches()
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW * 2))

    rows = []
    for _, m in matches.iterrows():
        t1, t2 = m["team1"], m["team2"]
        f1 = _decay(history[t1])
        f2 = _decay(history[t2])
        rows.append({
            "match_id": m["match_id"],
            "team1_decay_form": f1,
            "team2_decay_form": f2,
            "team1_decay_form_diff": (f1 - f2) if not (np.isnan(f1) or np.isnan(f2)) else np.nan,
        })

        # Update history (skip no-result)
        if m.get("no_result", False) or pd.isna(m.get("winner")):
            # treat as 0.5 contribution to both
            history[t1].append(0.5)
            history[t2].append(0.5)
            continue
        w = m["winner"]
        for t in (t1, t2):
            history[t].append(1.0 if t == w else 0.0)

    return pd.DataFrame(rows)


def main():
    df = build()
    atomic_write_parquet(df, OUT)
    print(f"[A3] recency_form: rows={len(df)} cols={len(df.columns)} -> {OUT}")


if __name__ == "__main__":
    main()
