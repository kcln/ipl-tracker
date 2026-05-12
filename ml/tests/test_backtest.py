"""Backtest harness tests."""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIST = ROOT / "ml" / "data" / "historical"


def _skip_if_missing(p):
    if not p.exists():
        pytest.skip(f"{p} not built yet")


def test_coin_flip_baseline():
    p = HIST / "matches.parquet"
    _skip_if_missing(p)
    matches = pd.read_parquet(p)
    from ml.src.backtest import run_backtest
    import random

    rng = random.Random(42)
    def cf(match_row, _state):
        t1, t2 = match_row["team1"], match_row["team2"]
        return (t1 if rng.random() < 0.5 else t2), 0.5

    summary = run_backtest(cf, matches, name="coin_flip_test")
    assert summary["overall"]["n"] > 1000
    # coin flip ≈ 50% ± 5%
    assert 0.42 <= summary["overall"]["acc"] <= 0.58
    assert not summary["leakage_detected"]
