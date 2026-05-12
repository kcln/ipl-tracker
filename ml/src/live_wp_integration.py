"""Live WP integration helper for src/tracker.py.

This module provides a single function the live tracker can call each cycle
during an in-match window. It does NOT modify src/tracker.py — the user wires
this in manually. The function is designed to be a thin adapter:

  from ml.src.live_wp_integration import wp_for_current_delivery
  wp = wp_for_current_delivery(match_state)  # returns float in [0, 1]

Where `match_state` is the live in-match state dict that the existing tracker
already builds (innings, balls bowled, current score, wickets, target, etc).

The integration point in src/tracker.py would be inside the post-PP1 / chase
generation flow, alongside the existing heuristic call.

Example integration (add to src/tracker.py, NOT modified by this module):

    try:
        from ml.src.live_wp_integration import wp_for_current_delivery
        ml_wp = wp_for_current_delivery(match_state)
        print(f"  ML WP (v2): {ml_wp:.1%} for {match_state['batting_team']}")
    except Exception as e:
        # Fall back silently — the existing heuristic path is unaffected
        print(f"  ML WP skipped: {e}")

This pattern lets the user observe the ML prediction alongside the heuristic
for one match window. If it's reliable, they can promote it to a published
message field. If not, the heuristic flow stays untouched.
"""
from __future__ import annotations

from typing import Any


def wp_for_current_delivery(match_state: dict[str, Any]) -> float:
    """Return P(batting_team wins) for the current delivery.

    Required keys in match_state:
      innings: int (1 or 2)
      ball_no_in_innings: int (0-indexed within innings)
      balls_remaining: int (out of 120 per innings)
      wickets_remaining: int (out of 10)
      current_score: int
      current_run_rate: float
      last_30_balls_rr: float (use 0 if innings just started)
      batting_team_strength: float (use 0.5 if unknown)
      bowling_team_strength: float (use 0.5 if unknown)
      is_chase: int (0 for innings 1, 1 for innings 2)
      venue: str  (matched against the v2 artifact's venue_categories)

    Optional keys (innings 2 only):
      target: int (first innings total + 1)
      required_run_rate: float

    Returns probability in [0, 1] of the batting team eventually winning.
    Conservative defaults are applied for missing keys; missing keys do not
    raise.
    """
    from ml.src.predict import predict_live

    required = ["innings", "ball_no_in_innings", "balls_remaining", "wickets_remaining",
                "current_score", "current_run_rate", "venue"]
    for k in required:
        if k not in match_state:
            raise ValueError(f"missing required key '{k}' in match_state")

    feats = {
        "innings": match_state["innings"],
        "ball_no_in_innings": match_state["ball_no_in_innings"],
        "balls_remaining": match_state["balls_remaining"],
        "wickets_remaining": match_state["wickets_remaining"],
        "current_score": match_state["current_score"],
        "current_run_rate": match_state["current_run_rate"],
        "required_run_rate": match_state.get("required_run_rate", float("nan")),
        "target": match_state.get("target", float("nan")),
        "last_30_balls_rr": match_state.get("last_30_balls_rr", 0.0),
        "batting_team_strength": match_state.get("batting_team_strength", 0.5),
        "bowling_team_strength": match_state.get("bowling_team_strength", 0.5),
        "is_chase": match_state.get("is_chase", 1 if match_state["innings"] == 2 else 0),
        "venue": match_state["venue"],
    }
    return predict_live(feats)


def wp_curve_for_innings(deliveries: list[dict[str, Any]], smooth_window: int = 6) -> list[float]:
    """Build a smoothed WP curve over a full innings.

    deliveries: list of per-delivery state dicts (same shape as match_state above)
    smooth_window: rolling average window in balls (default 6 = one over)

    Returns: list of smoothed P(batting_team wins) per delivery.
    """
    raw = [wp_for_current_delivery(d) for d in deliveries]
    smoothed = []
    for i, _ in enumerate(raw):
        lo = max(0, i - smooth_window + 1)
        smoothed.append(sum(raw[lo:i + 1]) / (i - lo + 1))
    return smoothed
