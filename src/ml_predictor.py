"""ML predictor bridge for the live tracker.

Wraps the v9 ensemble (and v6_phase_innings_break, v3 stage models, v2 WP)
from ml/data/models/ behind a tracker-friendly API.

CRITICAL: every public function in this module must catch all exceptions and
return None on failure. The existing heuristic flow in src/predictor.py must
remain bulletproof if any ML dependency, file, or model is missing or stale.

The integration is *parallel-display only*: callers should run the heuristic
flow as before and optionally surface the ML prediction alongside. Nothing in
the tracker depends on the ML path for correctness.
"""
from __future__ import annotations

import pathlib
import sys
from functools import lru_cache
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _import_ml():
    """Lazy-import the ml/ namespace. Returns a tuple of callables or None."""
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from ml.src.predict import predict_pre_match_full
        from ml.src.phase_features import _build_pre_match_clean
        from ml.src.player_features import _pit_balls, build_player_features
        return predict_pre_match_full, _build_pre_match_clean, _pit_balls, build_player_features
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_matches_balls():
    """Load cricsheet matches + balls parquets once per process."""
    try:
        import pandas as pd
        m = pd.read_parquet(ROOT / "ml" / "data" / "historical" / "matches.parquet")
        m["date"] = pd.to_datetime(m["date"])
        balls = pd.read_parquet(ROOT / "ml" / "data" / "historical" / "balls.parquet")
        return m, balls
    except Exception:
        return None, None


def predict_pre_match(
    team1: str,
    team2: str,
    venue: str,
    season: int,
    *,
    toss_winner: str | None = None,
    toss_decision: str | None = None,
    match_date: Any = None,
) -> dict | None:
    """Predict pre-match using the v9 ensemble.

    Returns a dict on success:
        {
            "predicted_winner": str,    # team1 or team2
            "p_team1": float,            # P(team1 wins) ∈ [0, 1]
            "confidence": float,         # max(p_team1, 1-p_team1)
            "source": "v9",
        }

    Returns None on ANY failure. Callers MUST handle None as "ML unavailable".

    `match_date` is optional; if omitted, today is used. This affects which
    historical matches the feature builders treat as priors — same-day matches
    with smaller match_ids are included as priors.
    """
    try:
        import pandas as pd
    except Exception:
        return None

    deps = _import_ml()
    if deps is None:
        return None
    predict_full, build_pm, pit_balls, build_pf = deps

    m, balls = _load_matches_balls()
    if m is None or balls is None:
        return None

    try:
        date = pd.to_datetime(match_date) if match_date is not None else pd.Timestamp.now()
        synth_id = f"PREDICT_{team1[:3]}_{team2[:3]}_{date.strftime('%Y%m%d%H%M%S')}"
        synth = {
            "match_id": synth_id, "date": date, "season": int(season),
            "venue": venue, "team1": team1, "team2": team2,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "winner": None, "win_by_runs": None, "win_by_wickets": None,
            "player_of_match": None, "no_result": False, "match_number": None,
        }
        m_ext = pd.concat([m, pd.DataFrame([synth])], ignore_index=True)
        m_ext["date"] = pd.to_datetime(m_ext["date"])
        m_ext = m_ext.sort_values(["date", "match_id"]).reset_index(drop=True)

        pm = build_pm(m_ext)
        # Sentinel-fill any null-winner rows so build_player_features doesn't drop them.
        m_sent = m_ext.copy()
        m_sent.loc[m_sent["winner"].isna(), "winner"] = "SENTINEL"
        balls_pit = pit_balls(balls, m_ext)
        pf = build_pf(m_sent, balls_pit)

        v3_row = pm[pm["match_id"] == synth_id]
        pf_row = pf[pf["match_id"] == synth_id]
        if len(v3_row) == 0 or len(pf_row) == 0:
            return None

        v3_feats = v3_row.iloc[0].to_dict()
        player_feats = pf_row.iloc[0].to_dict()
        out = predict_full(v3_feats, player_feats)
        p_team1 = float(out["p_team1"])
        winner = team1 if p_team1 >= 0.5 else team2
        return {
            "predicted_winner": winner,
            "p_team1": p_team1,
            "confidence": max(p_team1, 1.0 - p_team1),
            "source": "v9",
        }
    except Exception:
        return None


def _build_pit_pre_match_row(
    team1: str, team2: str, venue: str, season: int,
    toss_winner: str | None = None, toss_decision: str | None = None,
    match_date: Any = None,
):
    """Internal: synthesize the fixture, build PIT v3 features for it.

    Returns (v3_feats_dict, player_feats_dict, synth_id) or None on failure.
    """
    import pandas as pd
    deps = _import_ml()
    if deps is None:
        return None
    _, build_pm, pit_balls, build_pf = deps
    m, balls = _load_matches_balls()
    if m is None or balls is None:
        return None
    try:
        date = pd.to_datetime(match_date) if match_date is not None else pd.Timestamp.now()
        synth_id = f"PREDICT_{team1[:3]}_{team2[:3]}_{date.strftime('%Y%m%d%H%M%S')}"
        synth = {
            "match_id": synth_id, "date": date, "season": int(season),
            "venue": venue, "team1": team1, "team2": team2,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "winner": None, "win_by_runs": None, "win_by_wickets": None,
            "player_of_match": None, "no_result": False, "match_number": None,
        }
        m_ext = pd.concat([m, pd.DataFrame([synth])], ignore_index=True)
        m_ext["date"] = pd.to_datetime(m_ext["date"])
        m_ext = m_ext.sort_values(["date", "match_id"]).reset_index(drop=True)
        pm = build_pm(m_ext)
        m_sent = m_ext.copy()
        m_sent.loc[m_sent["winner"].isna(), "winner"] = "SENTINEL"
        balls_pit = pit_balls(balls, m_ext)
        pf = build_pf(m_sent, balls_pit)
        v3_row = pm[pm["match_id"] == synth_id]
        pf_row = pf[pf["match_id"] == synth_id]
        if len(v3_row) == 0 or len(pf_row) == 0:
            return None
        return v3_row.iloc[0].to_dict(), pf_row.iloc[0].to_dict(), synth_id
    except Exception:
        return None


def _predict_with_phase_model(stage: str, features: dict, team1: str, team2: str) -> dict | None:
    """Internal: route to ml.src.predict.predict_phase and shape the result."""
    try:
        from ml.src.predict import predict_phase
        p_team1 = float(predict_phase(stage, features))
        winner = team1 if p_team1 >= 0.5 else team2
        return {
            "predicted_winner": winner,
            "p_team1": p_team1,
            "confidence": max(p_team1, 1.0 - p_team1),
            "source": f"v6_{stage}" if stage == "innings_break" else f"v3_{stage}",
        }
    except Exception:
        return None


def predict_post_toss(
    team1: str, team2: str, venue: str, season: int,
    toss_winner: str, toss_decision: str,
    match_date: Any = None,
) -> dict | None:
    """Post-toss ML prediction via v3_phase_post_toss. Returns None on any failure."""
    built = _build_pit_pre_match_row(team1, team2, venue, season, toss_winner, toss_decision, match_date)
    if built is None:
        return None
    v3_feats, _, _ = built
    feats = dict(v3_feats)
    feats["toss_winner_is_team1"] = int(toss_winner == team1)
    feats["toss_decision_field"] = int(toss_decision == "field")
    return _predict_with_phase_model("post_toss", feats, team1, team2)


def predict_post_pp1(
    team1: str, team2: str, venue: str, season: int,
    toss_winner: str, toss_decision: str,
    pp1_runs: int, pp1_wickets: int,
    match_date: Any = None,
) -> dict | None:
    """Post-powerplay-1 ML prediction via v3_phase_post_pp1. Returns None on failure."""
    built = _build_pit_pre_match_row(team1, team2, venue, season, toss_winner, toss_decision, match_date)
    if built is None:
        return None
    v3_feats, _, _ = built
    feats = dict(v3_feats)
    feats["toss_winner_is_team1"] = int(toss_winner == team1)
    feats["toss_decision_field"] = int(toss_decision == "field")
    feats["pp1_runs"] = int(pp1_runs)
    feats["pp1_wickets"] = int(pp1_wickets)
    return _predict_with_phase_model("post_pp1", feats, team1, team2)


def predict_innings_break(
    team1: str, team2: str, venue: str, season: int,
    toss_winner: str, toss_decision: str,
    pp1_runs: int, pp1_wickets: int,
    first_innings_total: int, first_innings_wickets: int,
    match_date: Any = None,
) -> dict | None:
    """Innings-break ML prediction via v6_phase_innings_break.

    This is the strongest mid-match model: 72.9% on the 2025 test set.
    Returns None on any failure.
    """
    built = _build_pit_pre_match_row(team1, team2, venue, season, toss_winner, toss_decision, match_date)
    if built is None:
        return None
    v3_feats, _, _ = built
    feats = dict(v3_feats)
    feats["toss_winner_is_team1"] = int(toss_winner == team1)
    feats["toss_decision_field"] = int(toss_decision == "field")
    feats["pp1_runs"] = int(pp1_runs)
    feats["pp1_wickets"] = int(pp1_wickets)
    feats["first_innings_total"] = int(first_innings_total)
    feats["first_innings_wickets"] = int(first_innings_wickets)
    feats["target"] = first_innings_total + 1
    feats["innings_break_required_rr"] = (first_innings_total + 1) / 20.0
    return _predict_with_phase_model("innings_break", feats, team1, team2)


def format_ml_line(ml_result: dict | None) -> str | None:
    """Format the parallel-display line for a message. Returns None if ML unavailable.

    Output example: 'ML model: CSK wins (62%)'
    """
    if not ml_result:
        return None
    try:
        return f"ML model: {ml_result['predicted_winner']} wins ({ml_result['confidence']:.0%})"
    except Exception:
        return None
