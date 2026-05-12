"""Shared I/O helpers for v6 Phase A builders."""
from __future__ import annotations

import os
import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[3]
HIST = ROOT / "ml" / "data" / "historical"
V6_DIR = HIST / "v6"


def atomic_write_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write df to path atomically via .tmp + os.replace, using pyarrow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp)
    os.replace(tmp, path)


def load_matches() -> pd.DataFrame:
    m = pd.read_parquet(HIST / "matches.parquet")
    # Filter 2008-2025; exclude 2026 (ongoing) — same rule as v1-v5
    m = m[(m["season"] >= 2008) & (m["season"] <= 2025)].copy()
    m = m.sort_values(["date", "match_id"]).reset_index(drop=True)
    return m


def load_balls() -> pd.DataFrame:
    return pd.read_parquet(HIST / "balls.parquet")


def phase_of_over(over: int) -> str:
    """IPL phase: PP = overs 0-5, middle = 6-14, death = 15-19."""
    if over <= 5:
        return "pp"
    if over <= 14:
        return "middle"
    return "death"
