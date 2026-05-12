"""Phase A PIT smoke tests.

For each of the 5 parquets, pick a sample match somewhere mid-history and
recompute one feature from the raw inputs using only data strictly prior to
that match, then assert the stored value matches.

Smoke level: one feature per parquet, with a tolerance.
"""
from __future__ import annotations

import pathlib
from collections import deque

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
HIST = ROOT / "ml" / "data" / "historical"
V6 = HIST / "v6"


@pytest.fixture(scope="module")
def matches():
    m = pd.read_parquet(HIST / "matches.parquet")
    m = m[(m["season"] >= 2008) & (m["season"] <= 2025)].copy()
    return m.sort_values(["date", "match_id"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def balls():
    return pd.read_parquet(HIST / "balls.parquet")


def _phase_of(over: int) -> str:
    if over <= 5:
        return "pp"
    if over <= 14:
        return "middle"
    return "death"


# -------------- A3 recency_form --------------

def test_a3_pit_decay_form(matches):
    df = pd.read_parquet(V6 / "recency_form.parquet")
    assert len(df) == len(matches)

    # Pick a match well into the dataset to ensure both teams have history
    sample_idx = 500
    m = matches.iloc[sample_idx]
    mid = m["match_id"]
    t1 = m["team1"]

    # Reconstruct team1's last 10 outcomes from prior matches
    prior = matches.iloc[:sample_idx]
    history = []
    for _, pm in prior.iterrows():
        if pm["team1"] == t1 or pm["team2"] == t1:
            if pm.get("no_result", False) or pd.isna(pm.get("winner")):
                history.append(0.5)
            else:
                history.append(1.0 if pm["winner"] == t1 else 0.0)
    # Most recent first
    seq = list(reversed(history))[:10]
    weights = np.array([0.7**k for k in range(len(seq))])
    expected = float(np.dot(weights, seq) / weights.sum()) if seq else np.nan

    stored = df[df["match_id"] == mid].iloc[0]["team1_decay_form"]
    assert abs(stored - expected) < 1e-9, f"expected {expected}, got {stored}"


# -------------- A4 h2h_career --------------

def test_a4_pit_career_h2h(matches):
    df = pd.read_parquet(V6 / "h2h_career.parquet")
    assert len(df) == len(matches)

    # Find a match where the team pair has met before
    for idx in range(200, len(matches)):
        m = matches.iloc[idx]
        t1, t2 = m["team1"], m["team2"]
        prior = matches.iloc[:idx]
        meetings = prior[
            ((prior["team1"] == t1) & (prior["team2"] == t2))
            | ((prior["team1"] == t2) & (prior["team2"] == t1))
        ]
        meetings_decided = meetings[~meetings["no_result"].fillna(False) & meetings["winner"].notna()]
        if len(meetings_decided) >= 5:
            mid = m["match_id"]
            t1_wins = (meetings_decided["winner"] == t1).sum()
            expected = t1_wins / len(meetings_decided)
            stored = df[df["match_id"] == mid].iloc[0]["h2h_career_team1_winrate"]
            assert abs(stored - expected) < 1e-9
            return
    pytest.fail("Could not find a sample match with prior H2H history")


# -------------- A5 toss_conditioned --------------

def test_a5_pit_chase_rate(matches):
    df = pd.read_parquet(V6 / "toss_conditioned.parquet")
    assert len(df) == len(matches)

    # Find a match where the venue has prior "field" decisions
    for idx in range(100, len(matches)):
        m = matches.iloc[idx]
        venue = m["venue"]
        prior = matches.iloc[:idx]
        prior_v = prior[prior["venue"] == venue]
        prior_v = prior_v[~prior_v["no_result"].fillna(False)]
        prior_field = prior_v[prior_v["toss_decision"] == "field"]
        if len(prior_field) >= 5:
            chases = prior_field["win_by_wickets"].notna().sum()
            expected = chases / len(prior_field)
            stored = df[df["match_id"] == m["match_id"]].iloc[0]["venue_chase_rate_when_fielded"]
            assert abs(stored - expected) < 1e-9
            return
    pytest.fail("No suitable venue with prior 'field' decisions found")


# -------------- A2 venue_patterns --------------

def test_a2_pit_avg_1st_innings(matches, balls):
    df = pd.read_parquet(V6 / "venue_patterns.parquet")
    assert len(df) == len(matches)

    # First innings totals per match
    fi_totals = balls[balls["innings"] == 1].groupby("match_id")["total_runs"].sum()

    # Pick a match well into the data
    sample_idx = 600
    m = matches.iloc[sample_idx]
    mid = m["match_id"]
    venue = m["venue"]
    prior = matches.iloc[:sample_idx]
    prior_v = prior[prior["venue"] == venue]
    # Last 20 prior matches at this venue (window=20)
    prior_v_last = prior_v.tail(20)
    totals = [fi_totals.get(pmid) for pmid in prior_v_last["match_id"] if pmid in fi_totals.index]
    if not totals:
        pytest.skip("No prior matches at venue for this sample")
    expected = float(np.mean(totals))
    stored = df[df["match_id"] == mid].iloc[0]["venue_avg_1st_innings"]
    assert abs(stored - expected) < 1e-6


# -------------- A1 phase_player_stats --------------

def test_a1_pit_top5_pp_sr(matches, balls):
    df = pd.read_parquet(V6 / "phase_player_stats.parquet")
    assert len(df) == len(matches)

    # Find a match well into the data
    sample_idx = 700
    m = matches.iloc[sample_idx]
    mid = m["match_id"]
    t1 = m["team1"]

    # Identify team1's "previous match" chronologically
    prior = matches.iloc[:sample_idx]
    t1_prior = prior[(prior["team1"] == t1) | (prior["team2"] == t1)]
    if len(t1_prior) == 0:
        pytest.skip("team1 has no prior match")
    prev_match = t1_prior.iloc[-1]
    prev_mid = prev_match["match_id"]

    # Top 5 batters from team1 in prev match by balls faced
    prev_balls = balls[(balls["match_id"] == prev_mid) & (balls["batting_team"] == t1)]
    if len(prev_balls) == 0:
        pytest.skip("team1 didn't bat in prev match (?)")
    top5 = (
        prev_balls.groupby("batter").size().sort_values(ascending=False).head(5).index.tolist()
    )

    # PIT cumulative pp SR for those players using only balls from matches
    # strictly prior to the current match.
    prior_mids = set(prior["match_id"].tolist())
    pp_balls = balls[balls["match_id"].isin(prior_mids) & balls["batter"].isin(top5)]
    pp_balls = pp_balls[pp_balls["over"] <= 5]
    if pp_balls.empty:
        pytest.skip("No pp balls in prior data for top5")
    runs = pp_balls["runs_off_bat"].sum()
    bf = len(pp_balls)
    expected = 100.0 * runs / bf
    stored = df[df["match_id"] == mid].iloc[0]["team1_top5_pp_sr_mean"]
    assert stored is not None and not np.isnan(stored)
    assert abs(stored - expected) < 1e-6, f"expected {expected}, got {stored}"


# -------------- Common: rows match & no future leakage smoke --------------

def test_all_parquets_have_match_id_alignment(matches):
    expected_mids = set(matches["match_id"].tolist())
    for name in ("phase_player_stats", "venue_patterns", "recency_form", "h2h_career", "toss_conditioned"):
        df = pd.read_parquet(V6 / f"{name}.parquet")
        got = set(df["match_id"].tolist())
        assert got == expected_mids, f"{name}: match_id set mismatch"
