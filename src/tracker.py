"""IPL 2026 daily tracker — main entry point.

Invoked every 15 minutes by launchd. On each run:

  1. Fetch fixtures + standings (cached).
  2. Determine which message types should exist for today.
  3. Generate any missing ones; log to docs/index.html; record in state.json.
  4. Send only the newest undelivered message via iMessage; skip older ones.
  5. Commit & push state.json + docs/ to GitHub.

Exit codes:
  0  success (or nothing to do)
  1  fatal failure
  2  partial success (HTML written but iMessage failed, etc.)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, date
from pathlib import Path

# Allow running as `python3 src/tracker.py` (script) or `python3 -m src.tracker` (module)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import data_fetcher, html_archive, imessage_sender, message_builder, state  # noqa: E402
    from src.state import IST, PT, now_pt, today_pt_iso  # noqa: E402
else:
    from . import data_fetcher, html_archive, imessage_sender, message_builder, state
    from .state import IST, PT, now_pt, today_pt_iso

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_URL = "https://kcln.github.io/ipl-tracker/"
SEASON_END = date(2026, 5, 31)


def _log(msg: str, level: str = "info") -> None:
    color = {"info": "\033[36m", "warn": "\033[33m", "err": "\033[31m", "ok": "\033[32m"}.get(level, "")
    reset = "\033[0m"
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"{color}[{ts}] [{level}]{reset} {msg}", file=sys.stderr)


# ─────────────────────────────────────────
# Helpers: filter today's matches, ordering, etc.
# ─────────────────────────────────────────

def _matches_for_date_pt(all_matches: list[dict], date_iso_pt: str) -> list[dict]:
    """Return matches whose IST start time falls on `date_iso_pt` after
    converting IST → PT."""
    out = []
    for m in all_matches:
        date_ist = m.get("date_ist")
        time_ist = m.get("scheduled_ist") or "00:00"
        if not date_ist:
            continue
        try:
            dt_ist = datetime.strptime(f"{date_ist} {time_ist}", "%Y-%m-%d %H:%M").replace(tzinfo=IST)
        except ValueError:
            continue
        if dt_ist.astimezone(PT).strftime("%Y-%m-%d") == date_iso_pt:
            out.append(m)
    out.sort(key=lambda m: (m.get("date_ist", ""), m.get("scheduled_ist", "")))
    return out


def _completed_match_lookup(all_matches: list[dict]) -> list[dict]:
    """Flatten completed matches with a `winner` field for predictor input."""
    out = []
    for m in all_matches:
        if m.get("status") != "complete":
            continue
        winner = m.get("winner") or m.get("actual_winner")
        if not winner:
            continue
        out.append({"teams": m.get("teams", []), "winner": winner, "status": "complete"})
    return out


# ─────────────────────────────────────────
# Per-message generation
# ─────────────────────────────────────────

def _maybe_generate_morning(
    day_entry: dict, date_iso: str,
    todays: list[dict], standings: list[dict],
    remaining: list[dict], recent: list[dict], squads: dict,
) -> bool:
    if state.find_message(day_entry, "morning"):
        return False
    if not todays:
        return False
    # Per spec: morning fires any time after 00:00 PT — i.e. always present once we know matches
    body = message_builder.morning_message(
        date_iso, todays, standings, remaining, recent, squads,
    )
    msg = state.add_or_update_message(day_entry, "morning", body)
    html_archive.upsert_message(date_iso, "morning", msg["generated_at"], body)
    _log(f"generated morning brief ({len(todays)} matches)", "ok")
    return True


def _maybe_generate_post_match(
    day_entry: dict, date_iso: str,
    todays: list[dict], standings: list[dict],
    remaining: list[dict], recent: list[dict], squads: dict,
) -> int:
    generated = 0
    # Sort matches by scheduled time so "Match 1/2" is stable
    for idx, match in enumerate(todays, 1):
        if match.get("status") != "complete":
            continue
        msg_type = f"post_match_{idx}"
        if state.find_message(day_entry, msg_type):
            continue
        # Find recorded prediction for this match (from morning brief generation we re-derive)
        all_teams = [r["team"] for r in standings] or list({t for m in todays for t in m["teams"]})
        from . import predictor as _p  # local import to avoid cycles in standalone script mode
        predicted, _ = _p.predict_winner(
            match["teams"][0], match["teams"][1],
            standings, recent, squads, all_teams,
        )
        match_for_msg = dict(match)
        match_for_msg["predicted_winner"] = predicted
        body = message_builder.post_match_message(
            match_for_msg, standings, remaining, recent, squads,
        )
        # Also record the predicted/actual winner on the match entry in state
        _record_match_outcome(day_entry, match, predicted)
        msg = state.add_or_update_message(day_entry, msg_type, body)
        html_archive.upsert_message(date_iso, msg_type, msg["generated_at"], body)
        generated += 1
        _log(f"generated {msg_type}: {match['teams']} → {match.get('winner')}", "ok")
    return generated


def _record_match_outcome(day_entry: dict, match: dict, predicted_winner: str) -> None:
    """Store {id, teams, predicted_winner, actual_winner, status, result} on day_entry.matches."""
    entries = day_entry.setdefault("matches", [])
    mid = match.get("id")
    actual = match.get("winner") or match.get("actual_winner")
    for e in entries:
        if e.get("id") == mid:
            e["status"] = match.get("status")
            e["result"] = match.get("result")
            e["predicted_winner"] = predicted_winner
            e["actual_winner"] = actual
            return
    entries.append({
        "id": mid,
        "teams": match.get("teams"),
        "scheduled_ist": match.get("scheduled_ist"),
        "status": match.get("status"),
        "result": match.get("result"),
        "predicted_winner": predicted_winner,
        "actual_winner": actual,
    })


def _maybe_generate_end_of_day(
    day_entry: dict, date_iso: str,
    todays: list[dict], standings: list[dict],
    remaining: list[dict], recent: list[dict], squads: dict,
) -> bool:
    if state.find_message(day_entry, "end_of_day"):
        return False
    if not todays:
        return False
    if not all(m.get("status") == "complete" for m in todays):
        return False
    # Use the matches stored on day_entry (carry predicted_winner)
    enriched = day_entry.get("matches") or todays
    body = message_builder.end_of_day_message(
        date_iso, enriched, standings, remaining, recent, squads, ARCHIVE_URL,
    )
    msg = state.add_or_update_message(day_entry, "end_of_day", body)
    html_archive.upsert_message(date_iso, "end_of_day", msg["generated_at"], body)
    _log("generated end-of-day recap", "ok")
    return True


# ─────────────────────────────────────────
# Git push
# ─────────────────────────────────────────

def _git(*args: str) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            ["git", *args], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except (subprocess.SubprocessError, OSError) as e:
        return 1, "", str(e)


def _git_push_if_changes() -> bool:
    rc, status_out, _ = _git("status", "--porcelain")
    if rc != 0:
        _log("git not a repo or git missing; skipping push", "warn")
        return False
    if not status_out.strip():
        _log("no git changes to push")
        return True
    _git("add", "state.json", "docs/")
    rc, _, err = _git("commit", "-m", f"tracker update {now_pt().isoformat(timespec='minutes')}")
    if rc != 0:
        _log(f"git commit failed: {err}", "warn")
        return False
    rc, _, err = _git("push")
    if rc != 0:
        _log(f"git push failed (will retry next run): {err}", "warn")
        return False
    _log("pushed to remote", "ok")
    return True


# ─────────────────────────────────────────
# Season end handling
# ─────────────────────────────────────────

def _disable_launchd() -> None:
    plist = Path.home() / "Library/LaunchAgents/com.kcln.ipltracker.plist"
    if not plist.exists():
        return
    subprocess.run(
        ["launchctl", "unload", str(plist)],
        capture_output=True, timeout=10,
    )
    _log("launchd job unloaded — tracker disabled for the offseason", "ok")


def _handle_season_end(state_obj: dict) -> bool:
    if now_pt().date() <= SEASON_END:
        return False
    if state_obj.get("season_ended"):
        return False
    standings = data_fetcher.fetch_standings()
    body = message_builder.season_recap_message(standings, ARCHIVE_URL)
    today = today_pt_iso()
    day_entry = state.day(state_obj, today)
    msg = state.add_or_update_message(day_entry, "season_recap", body)
    html_archive.upsert_message(today, "season_recap", msg["generated_at"], body)
    if imessage_sender.send(body):
        state.mark_delivered(msg)
    state_obj["season_ended"] = True
    state.save(state_obj)
    _disable_launchd()
    return True


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main() -> int:
    _log("tracker run starting")
    partial = False

    state_obj = state.load()

    if _handle_season_end(state_obj):
        return 0

    try:
        all_matches = data_fetcher.fetch_fixtures()
        standings = data_fetcher.fetch_standings()
        squads = data_fetcher.fetch_squads()
    except Exception as e:  # data layer should never raise, but be safe
        _log(f"data fetch failed: {e}", "err")
        state.save(state_obj)
        return 1

    if not all_matches:
        _log("no fixtures available — likely ESPN/Cricbuzz blocked or season not yet started", "warn")
        state.save(state_obj)
        return 0  # graceful: nothing to do

    today = today_pt_iso()
    todays = _matches_for_date_pt(all_matches, today)
    if not todays:
        _log(f"no matches scheduled for {today} (PT); nothing to do")
        state.save(state_obj)
        return 0

    # Refresh live match status (TTL 60s) for any not-yet-complete match
    for m in todays:
        if m.get("status") != "complete":
            latest = data_fetcher.fetch_current_match(m["id"])
            if latest:
                m["status"] = latest.get("status", m.get("status"))
                m["result"] = latest.get("result") or m.get("result")
                if latest.get("winner"):
                    m["winner"] = latest["winner"]

    day_entry = state.day(state_obj, today)
    recent = _completed_match_lookup(all_matches)
    remaining = [m for m in all_matches if m.get("status") != "complete"]

    # Generate any messages that should exist
    _maybe_generate_morning(day_entry, today, todays, standings, remaining, recent, squads)
    _maybe_generate_post_match(day_entry, today, todays, standings, remaining, recent, squads)
    _maybe_generate_end_of_day(day_entry, today, todays, standings, remaining, recent, squads)

    # Send only the newest undelivered message
    newest = state.latest_undelivered_message(day_entry)
    if newest is not None:
        skipped = state.mark_older_as_skipped(day_entry, newest, "newer message available")
        if skipped:
            _log(f"marked {skipped} older message(s) as skipped")
        ok = imessage_sender.send(newest["body"])
        if ok:
            state.mark_delivered(newest)
        else:
            _log("iMessage send failed", "warn")
            partial = True

    state.save(state_obj)

    if not _git_push_if_changes():
        partial = True

    _log("tracker run complete", "ok")
    return 2 if partial else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:  # last-resort safety net
        _log(f"unhandled exception: {e}", "err")
        sys.exit(1)
