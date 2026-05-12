"""v6 Phase B2 — actual playing XI per match from balls.parquet.

For each match, the playing XI of a team is the set of players who appeared
in the ball-by-ball record as batter, bowler, or fielder for that team.

- A `batter` is on the batting_team's XI.
- A `bowler` is on the OTHER team's XI (bowling team).
- A `fielder` is on the bowling team's XI (catches/run-outs by the fielding
  side while the other side bats).

We then map batting_team / bowling_team to team1 / team2 using matches.parquet
and emit `team1_xi`, `team2_xi` as lists of player names (typically up to 11).

CLI: `python -m ml.src.v6.build_b2` writes
`ml/data/historical/v6/playing_xi.parquet`.
"""
from __future__ import annotations

import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.src.v6._io import V6_DIR, load_balls, load_matches

OUTPUT = V6_DIR / "playing_xi.parquet"


def _normalise_fielders(cell) -> list[str]:
    """balls.parquet stores fielders as a list-like; coerce to a clean list[str]."""
    if cell is None:
        return []
    # Numpy arrays + lists both iterable; strings would be a single name we
    # don't expect but handle defensively.
    if isinstance(cell, str):
        return [cell] if cell else []
    try:
        return [str(x) for x in cell if x is not None and str(x) != ""]
    except TypeError:
        return []


def build_xi() -> pd.DataFrame:
    matches = load_matches()  # 2008-2025 filter applied
    balls = load_balls()

    # match_id types must align (both object/string per inspection)
    keep_ids = set(matches["match_id"].astype(str))
    balls = balls[balls["match_id"].astype(str).isin(keep_ids)].copy()

    # team1/team2 lookup
    team_lookup = matches.set_index("match_id")[["team1", "team2"]].to_dict("index")

    # Per-(match, team) sets of players: batters belong to batting_team;
    # bowlers and fielders belong to the OTHER team in that match.
    # We'll iterate batters and bowlers via groupby for speed, then handle
    # fielders row-by-row (it's a list column).
    rows: dict[tuple[str, str], set[str]] = {}

    # Batters
    bat = balls.dropna(subset=["batter"])[["match_id", "batting_team", "batter"]]
    for (mid, team), grp in bat.groupby(["match_id", "batting_team"], sort=False):
        rows.setdefault((str(mid), team), set()).update(grp["batter"].astype(str).unique())

    # For non-strikers (also batting-team players who took the crease)
    ns = balls.dropna(subset=["non_striker"])[["match_id", "batting_team", "non_striker"]]
    for (mid, team), grp in ns.groupby(["match_id", "batting_team"], sort=False):
        rows.setdefault((str(mid), team), set()).update(grp["non_striker"].astype(str).unique())

    # Bowlers: bowler's team is the OTHER team in the match
    bowl = balls.dropna(subset=["bowler"])[["match_id", "batting_team", "bowler"]]
    for (mid, bat_team), grp in bowl.groupby(["match_id", "batting_team"], sort=False):
        teams = team_lookup.get(str(mid))
        if teams is None:
            continue
        bowling_team = teams["team2"] if bat_team == teams["team1"] else teams["team1"]
        rows.setdefault((str(mid), bowling_team), set()).update(grp["bowler"].astype(str).unique())

    # Fielders: same as bowlers — belong to the bowling team.
    # Iterate rows where fielders is non-empty.
    f_balls = balls[["match_id", "batting_team", "fielders"]]
    for mid, bat_team, fielders in f_balls.itertuples(index=False, name=None):
        names = _normalise_fielders(fielders)
        if not names:
            continue
        teams = team_lookup.get(str(mid))
        if teams is None:
            continue
        bowling_team = teams["team2"] if bat_team == teams["team1"] else teams["team1"]
        rows.setdefault((str(mid), bowling_team), set()).update(names)

    # Assemble output: one row per match, with team1_xi / team2_xi as lists
    out = []
    for mid, info in team_lookup.items():
        t1 = info["team1"]
        t2 = info["team2"]
        xi1 = sorted(rows.get((str(mid), t1), set()))
        xi2 = sorted(rows.get((str(mid), t2), set()))
        out.append({
            "match_id": str(mid),
            "team1_xi": xi1,
            "team2_xi": xi2,
        })

    df = pd.DataFrame(out)
    return df


def write_atomic_xi(df: pd.DataFrame) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")

    schema = pa.schema([
        ("match_id", pa.string()),
        ("team1_xi", pa.list_(pa.string())),
        ("team2_xi", pa.list_(pa.string())),
    ])
    table = pa.table({
        "match_id": pa.array(df["match_id"].astype(str).tolist(), type=pa.string()),
        "team1_xi": pa.array(df["team1_xi"].tolist(), type=pa.list_(pa.string())),
        "team2_xi": pa.array(df["team2_xi"].tolist(), type=pa.list_(pa.string())),
    }, schema=schema)
    pq.write_table(table, tmp)
    import os
    os.replace(tmp, OUTPUT)


def main() -> int:
    df = build_xi()
    write_atomic_xi(df)
    sizes_t1 = df["team1_xi"].str.len()
    sizes_t2 = df["team2_xi"].str.len()
    total = sizes_t1 + sizes_t2
    print(f"wrote {OUTPUT}  ({len(df)} matches)")
    print(f"  team1 XI size: mean={sizes_t1.mean():.2f}, median={sizes_t1.median():.0f}, "
          f"min={sizes_t1.min()}, max={sizes_t1.max()}")
    print(f"  team2 XI size: mean={sizes_t2.mean():.2f}, median={sizes_t2.median():.0f}, "
          f"min={sizes_t2.min()}, max={sizes_t2.max()}")
    print(f"  total players per match: mean={total.mean():.2f} (expect ~22)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
