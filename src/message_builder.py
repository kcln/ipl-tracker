"""Compose morning / post-match / end-of-day message bodies."""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from . import predictor
from .state import IST

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
PT = ZoneInfo("America/Los_Angeles")


def _fmt_times(date_ist: str, hhmm_ist: str) -> str:
    """'15:30' on '2026-05-12' (IST) → '3:30 PM IST / 5:00 AM ET / 4:00 AM CT / 2:00 AM PT'."""
    try:
        dt_ist = datetime.strptime(f"{date_ist} {hhmm_ist}", "%Y-%m-%d %H:%M").replace(tzinfo=IST)
    except ValueError:
        return f"{hhmm_ist} IST"

    def stamp(dt: datetime) -> str:
        # %-I is POSIX, works on macOS
        s = dt.strftime("%-I:%M %p")
        return s

    return (
        f"{stamp(dt_ist)} IST / "
        f"{stamp(dt_ist.astimezone(ET))} ET / "
        f"{stamp(dt_ist.astimezone(CT))} CT / "
        f"{stamp(dt_ist.astimezone(PT))} PT"
    )


def _format_date_long(date_iso: str) -> str:
    dt = datetime.strptime(date_iso, "%Y-%m-%d")
    return dt.strftime("%A, %B %-d")


def _top4_line(label: str, teams: list[str]) -> str:
    return f"{label}: " + (", ".join(teams) if teams else "(unavailable)")


def morning_message(
    date_iso: str,
    todays_matches: list[dict],
    standings: list[dict],
    remaining_fixtures: list[dict],
    recent_matches: list[dict],
    squads: dict,
) -> str:
    lines: list[str] = []
    lines.append(f"IPL 2026 - {_format_date_long(date_iso)}")
    lines.append("")
    lines.append("Today's matches:")
    lines.append("")

    all_teams = [r["team"] for r in standings] or list({t for m in todays_matches for t in m["teams"]})

    for i, m in enumerate(todays_matches, 1):
        t1, t2 = m["teams"]
        winner, reason = predictor.predict_winner(
            t1, t2, standings, recent_matches, squads, all_teams,
            match=m, completed_matches=recent_matches,
        )
        lines.append(f"Match {i}: {t1} vs {t2}")
        lines.append(_fmt_times(m["date_ist"], m["scheduled_ist"]))
        lines.append(f"Prediction: {winner} wins")
        lines.append(f"Reason: {reason}")
        lines.append("")

    lines.append(_top4_line("Current top 4", predictor.current_top4(standings)))
    lines.append(_top4_line(
        "Predicted final top 4",
        predictor.predict_final_top4(
            standings, remaining_fixtures, recent_matches, squads,
            completed_matches=recent_matches,
        ),
    ))
    return "\n".join(lines).rstrip()


def post_match_message(
    match: dict,
    standings: list[dict],
    remaining_fixtures: list[dict],
    recent_matches: list[dict],
    squads: dict,
) -> str:
    """`match` must have winner, result (margin text), and predicted_winner fields."""
    winner = match.get("actual_winner") or match.get("winner")
    teams = match.get("teams", [])
    loser = teams[1] if teams and winner == teams[0] else (teams[0] if teams else "")
    margin = match.get("result") or "the deciding result"
    predicted = match.get("predicted_winner")
    correct = "correct" if predicted == winner else "incorrect"

    lines = [
        f"{winner} beat {loser} by {_strip_winner_prefix(margin, winner, loser)}",
        "",
        f"Pre-match prediction: {correct}",
        "",
        _top4_line("Updated top 4", predictor.current_top4(standings)),
        _top4_line(
            "Predicted final top 4",
            predictor.predict_final_top4(
                standings, remaining_fixtures, recent_matches, squads,
                completed_matches=recent_matches,
            ),
        ),
    ]
    return "\n".join(lines)


def _strip_winner_prefix(margin: str, winner: str, loser: str) -> str:
    """Source feeds vary: ESPN's statusText reads 'CSK won by 12 runs';
    iplt20 Commentss reads 'Delhi Capitals Won by 3  Wickets '. We render
    the winner ourselves, so strip any leading 'X won by' phrasing and
    normalize whitespace."""
    if not margin:
        return ""
    m = re.sub(r"\s+", " ", margin).strip()
    lower = m.lower()
    for prefix in (
        f"{winner.lower()} won by ",
        f"{winner.lower()} beat {loser.lower()} by ",
        f"{winner.lower()} won the match by ",
    ):
        if lower.startswith(prefix):
            return m[len(prefix):]
    # Pull anything after "by "
    if " by " in lower:
        return m.split(" by ", 1)[1]
    return m


def end_of_day_message(
    date_iso: str,
    days_matches: list[dict],
    standings: list[dict],
    remaining_fixtures: list[dict],
    recent_matches: list[dict],
    squads: dict,
    archive_url: str,
) -> str:
    lines: list[str] = [f"IPL 2026 - {_format_date_long(date_iso)} - Day recap", ""]
    correct = 0
    total = 0
    for m in days_matches:
        if m.get("status") != "complete":
            continue
        winner = m.get("actual_winner") or m.get("winner")
        teams = m.get("teams", [])
        if not winner or len(teams) != 2:
            continue
        loser = teams[1] if winner == teams[0] else teams[0]
        margin = _strip_winner_prefix(m.get("result") or "", winner, loser)
        lines.append(f"{winner} beat {loser} by {margin}".rstrip())
        total += 1
        if m.get("predicted_winner") == winner:
            correct += 1

    lines.append("")
    lines.append(f"Predictions today: {correct} of {total} correct")
    lines.append("")
    lines.append(_top4_line("Updated top 4", predictor.current_top4(standings)))
    lines.append(_top4_line(
        "Predicted final top 4",
        predictor.predict_final_top4(
            standings, remaining_fixtures, recent_matches, squads,
            completed_matches=recent_matches,
        ),
    ))
    lines.append("")
    lines.append(f"Archive: {archive_url}")
    return "\n".join(lines)


def season_recap_message(standings: list[dict], archive_url: str) -> str:
    top4 = predictor.current_top4(standings)
    lines = [
        "IPL 2026 - Season complete",
        "",
        _top4_line("Final top 4", top4),
        "",
        f"Archive: {archive_url}",
        "",
        "Tracker will now disable itself.",
    ]
    return "\n".join(lines)
