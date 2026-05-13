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
        # Parallel ML prediction (silent fallback if unavailable)
        try:
            from . import ml_predictor as _ml
            _ml_pred = _ml.predict_pre_match(
                t1, t2,
                venue=m.get("venue_name") or m.get("venue", ""),
                season=2026,
                match_date=m.get("date_ist"),
            )
            _ml_line = _ml.format_ml_line(_ml_pred)
            if _ml_line:
                lines.append(_ml_line)
        except Exception:
            pass
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
    # Abandoned / no-result branch: skip the win body and the prediction-grade
    # line. An abandoned game has no real winner, so grading the prediction is
    # meaningless (and would dilute the daily accuracy count).
    if match.get("result_type") in ("no_result", "abandoned"):
        teams = match.get("teams", [])
        header = f"{teams[0]} vs {teams[1]}" if len(teams) == 2 else "Match"
        note = match.get("note") or "Match abandoned — no result"
        lines = [
            f"{header} — Match abandoned (no result)",
            "",
            note,
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


def _pct(p: float) -> str:
    return f"{int(round(p * 100))}%"


# ---------------------------------------------------------------------------
# Delay messages: status_update fires when a B1/B2 gate detects an overdue or
# frozen milestone; status_resumed fires when the delay clears.
# Generic, phase-aware copy so the body reads naturally without upstream text.
# ---------------------------------------------------------------------------

_PHASE_DELAY_FALLBACK = {
    "pre_toss":          "Match delayed — toss not yet held",
    "post_toss_pre_play": "Play delayed — first ball not yet bowled",
    "in_play":           "Play stopped — overs counter has not advanced",
}

_PHASE_RESUMED_COPY = {
    "pre_toss":          "Toss is underway",
    "post_toss_pre_play": "Play has started — first ball bowled",
    "in_play":           "Play has resumed",
}


_TEAM_HEADER_RE = re.compile(r"^([A-Z]{2,5}) vs ([A-Z]{2,5})(.*)$")


def rewrite_team_header(body: str, *, home_team: str) -> str:
    """If the first line of `body` is a 'TeamA vs TeamB[ — suffix]' header and
    home_team is one of the two teams but is in the wrong position, swap
    them. Otherwise return body unchanged. Only the first line is touched.
    """
    if not body:
        return body
    lines = body.splitlines(keepends=False)
    first = lines[0]
    m = _TEAM_HEADER_RE.match(first)
    if not m:
        return body
    a, b, suffix = m.group(1), m.group(2), m.group(3)
    if home_team not in (a, b):
        return body
    if a == home_team:
        return body  # already correct
    # Swap so home is first
    new_first = f"{b} vs {a}{suffix}"
    lines[0] = new_first
    # Preserve trailing newline if original had one
    out = "\n".join(lines)
    if body.endswith("\n"):
        out += "\n"
    return out


def status_update_message(match: dict, *, phase: str, note: str | None) -> str:
    t1, t2 = match["teams"]
    lines = [f"{t1} vs {t2} — Status update", ""]
    if note:
        lines.append(note)
    else:
        lines.append(_PHASE_DELAY_FALLBACK.get(phase, "Match delayed"))
    return "\n".join(lines)


def status_resumed_message(match: dict, *, phase: str, delay_minutes: float) -> str:
    t1, t2 = match["teams"]
    mins = int(round(delay_minutes))
    headline = _PHASE_RESUMED_COPY.get(phase, "Play resumed")
    return (
        f"{t1} vs {t2} — {headline}\n"
        f"\n"
        f"Resumed after a {mins}-minute delay."
    )


def toss_message(
    match: dict,
    standings: list[dict],
    recent_matches: list[dict],
    squads: dict,
    completed_matches: list[dict],
) -> str:
    """Fires once the toss is known. Adjusted prediction with toss bias."""
    t1, t2 = match["teams"]
    all_teams = [r["team"] for r in standings] or list(set(match["teams"]))
    winner, prob, reason = predictor.predict_after_toss(
        t1, t2, standings, recent_matches, squads, all_teams,
        match=match, completed_matches=completed_matches,
    )
    toss_winner = match.get("toss_winner") or "?"
    toss_decision = match.get("toss_decision") or "?"
    lines = [
        f"{t1} vs {t2} — Toss",
        "",
        f"Toss: {toss_winner} won and chose to {toss_decision}.",
        "",
        f"Updated prediction: {winner} wins ({_pct(prob)})",
        f"Reason: {reason}",
    ]
    # Parallel ML prediction (silent fallback if unavailable)
    try:
        from . import ml_predictor as _ml
        _ml_pred = _ml.predict_post_toss(
            t1, t2,
            venue=match.get("venue_name") or match.get("venue", ""),
            season=2026,
            toss_winner=match.get("toss_winner") or "",
            toss_decision=match.get("toss_decision") or "",
            match_date=match.get("date_ist"),
        )
        _ml_line = _ml.format_ml_line(_ml_pred)
        if _ml_line:
            lines.append(_ml_line)
    except Exception:
        pass
    return "\n".join(lines)


def powerplay_1_message(
    match: dict,
    standings: list[dict],
    completed_matches: list[dict],
) -> str:
    """Fires after innings-1 powerplay (over 6.0)."""
    inn1 = match.get("inn1") or {}
    runs = inn1.get("runs", 0)
    wkts = inn1.get("wkts", 0)
    overs = inn1.get("overs", 6.0)
    batting = match.get("first_batting") or match["teams"][0]
    bowling = match.get("second_batting") or match["teams"][1]

    # Backfill case: tracker first saw the match already complete, so the
    # only inn1 data we have is the FINAL innings score, not the PP snapshot.
    if overs > 10:
        lines = [
            f"{batting} vs {bowling} — Powerplay 1",
            "",
            f"Live powerplay snapshot not captured.",
            f"{batting} first innings: {runs}/{wkts} in {overs} overs.",
        ]
        return "\n".join(lines)

    winner, prob, reason = predictor.predict_after_powerplay(
        batting, bowling, runs, wkts, innings_num=1, target=None,
        standings=standings, completed_matches=completed_matches,
    )
    lines = [
        f"{batting} vs {bowling} — Powerplay 1",
        "",
        f"{batting}: {runs}/{wkts} after {overs} overs.",
        "",
        f"Updated prediction: {winner} wins ({_pct(prob)})",
        f"Reason: {reason}",
    ]
    # Parallel ML prediction
    try:
        from . import ml_predictor as _ml
        _ml_pred = _ml.predict_post_pp1(
            match["teams"][0], match["teams"][1],
            venue=match.get("venue_name") or match.get("venue", ""),
            season=2026,
            toss_winner=match.get("toss_winner") or "",
            toss_decision=match.get("toss_decision") or "",
            pp1_runs=runs, pp1_wickets=wkts,
            match_date=match.get("date_ist"),
        )
        _ml_line = _ml.format_ml_line(_ml_pred)
        if _ml_line:
            lines.append(_ml_line)
    except Exception:
        pass
    return "\n".join(lines)


def innings_break_message(
    match: dict,
    standings: list[dict],
    completed_matches: list[dict],
) -> str:
    """Fires when innings 1 ends — target set, chase prediction."""
    inn1 = match.get("inn1") or {}
    runs = inn1.get("runs", 0)
    wkts = inn1.get("wkts", 0)
    overs = inn1.get("overs", 20.0)
    batting = match.get("first_batting") or match["teams"][0]
    chasing = match.get("second_batting") or match["teams"][1]
    target = match.get("revised_target") or (runs + 1)
    winner, prob, reason = predictor.predict_chase(
        chasing, batting, target, runs, wkts,
        completed_matches=completed_matches,
        venue_id=match.get("venue_id"),
    )
    lines = [
        f"{batting} vs {chasing} — Innings break",
        "",
        f"{batting} finished {runs}/{wkts} in {overs} overs.",
        f"{chasing} need {target} to win.",
        "",
        f"Updated prediction: {winner} wins ({_pct(prob)})",
        f"Reason: {reason}",
    ]
    # Parallel ML prediction — innings_break is the strongest mid-match model (v6, 72.9%)
    try:
        from . import ml_predictor as _ml
        # Compute PP1 stats from match state for the feature vector
        pp1 = match.get("inn1_pp") or {}
        pp1_runs = pp1.get("runs", 0)
        pp1_wkts = pp1.get("wkts", 0)
        _ml_pred = _ml.predict_innings_break(
            match["teams"][0], match["teams"][1],
            venue=match.get("venue_name") or match.get("venue", ""),
            season=2026,
            toss_winner=match.get("toss_winner") or "",
            toss_decision=match.get("toss_decision") or "",
            pp1_runs=pp1_runs, pp1_wickets=pp1_wkts,
            first_innings_total=runs, first_innings_wickets=wkts,
            match_date=match.get("date_ist"),
        )
        _ml_line = _ml.format_ml_line(_ml_pred)
        if _ml_line:
            lines.append(_ml_line)
    except Exception:
        pass
    return "\n".join(lines)


def powerplay_2_message(
    match: dict,
    standings: list[dict],
    completed_matches: list[dict],
) -> str:
    """Fires after innings-2 powerplay (over 6.0 of the chase)."""
    inn1 = match.get("inn1") or {}
    inn2 = match.get("inn2") or {}
    chasing = match.get("second_batting") or match["teams"][1]
    defending = match.get("first_batting") or match["teams"][0]
    target = match.get("revised_target") or (inn1.get("runs", 0) + 1)
    runs = inn2.get("runs", 0)
    wkts = inn2.get("wkts", 0)
    overs = inn2.get("overs", 6.0)

    if overs > 10:
        lines = [
            f"{defending} vs {chasing} — Powerplay 2",
            "",
            f"Live chase powerplay snapshot not captured.",
            f"{chasing} chase: {runs}/{wkts} in {overs} overs (target {target}).",
        ]
        return "\n".join(lines)

    winner, prob, reason = predictor.predict_after_powerplay(
        chasing, defending, runs, wkts, innings_num=2, target=target,
        standings=standings, completed_matches=completed_matches,
    )
    lines = [
        f"{defending} vs {chasing} — Powerplay 2",
        "",
        f"{chasing}: {runs}/{wkts} after {overs} overs, chasing {target}.",
        "",
        f"Updated prediction: {winner} wins ({_pct(prob)})",
        f"Reason: {reason}",
    ]
    # Parallel ML prediction — use post_pp1 model in chase mode (same shape)
    try:
        from . import ml_predictor as _ml
        _ml_pred = _ml.predict_post_pp1(
            match["teams"][0], match["teams"][1],
            venue=match.get("venue_name") or match.get("venue", ""),
            season=2026,
            toss_winner=match.get("toss_winner") or "",
            toss_decision=match.get("toss_decision") or "",
            pp1_runs=runs, pp1_wickets=wkts,
            match_date=match.get("date_ist"),
        )
        _ml_line = _ml.format_ml_line(_ml_pred)
        if _ml_line:
            lines.append(_ml_line)
    except Exception:
        pass
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
