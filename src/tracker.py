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
import re
import subprocess
import sys
import hashlib
from datetime import datetime, date, timedelta
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
    """Flatten completed matches preserving fields the predictor uses
    for head-to-head, venue, MOM, and home-advantage signals."""
    out = []
    for m in all_matches:
        if m.get("status") != "complete":
            continue
        winner = m.get("winner") or m.get("actual_winner")
        if not winner:
            continue
        out.append({
            "teams": m.get("teams", []),
            "winner": winner,
            "status": "complete",
            "venue_id": m.get("venue_id", ""),
            "venue_name": m.get("venue_name", ""),
            "home_team": m.get("home_team"),
            "first_batting": m.get("first_batting"),
            "second_batting": m.get("second_batting"),
            "mom": m.get("mom"),
            "toss_winner": m.get("toss_winner"),
        })
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


# Thresholds for the B1 / B2 delay-detection gates. Kept module-level so
# tests can override if desired without touching internals.
_PRE_TOSS_DELAY_MINUTES = 10
_POST_TOSS_DELAY_MINUTES = 30
_IN_PLAY_FREEZE_MINUTES = 10
_DELAY_SOFT_CAP = 5  # max status_update messages per match per day


def _note_hash(note: str | None) -> str | None:
    if not note:
        return None
    return hashlib.sha1(note.strip().lower().encode("utf-8")).hexdigest()[:12]


def _scheduled_dt_pt(match: dict) -> datetime | None:
    try:
        dt_ist = datetime.strptime(
            f"{match['date_ist']} {match['scheduled_ist']}", "%Y-%m-%d %H:%M",
        ).replace(tzinfo=IST)
        return dt_ist.astimezone(PT)
    except (KeyError, ValueError, TypeError):
        return None


def _delay_gate_status(
    match: dict, idx: int, day_entry: dict, now_pt: datetime,
) -> tuple[str, bool]:
    """Returns (phase, gate_open) for the delay-detection gates.

    phase ∈ {"pre_toss", "post_toss_pre_play", "in_play"}.
    """
    toss_winner = match.get("toss_winner")
    inn1 = match.get("inn1") or {}
    inn2 = match.get("inn2") or {}
    overs1 = inn1.get("overs") or 0
    overs2 = inn2.get("overs") or 0

    # B1.1 — pre-toss overdue
    if not toss_winner:
        sched = _scheduled_dt_pt(match)
        if sched is None:
            return ("pre_toss", False)
        return ("pre_toss", now_pt > sched + timedelta(minutes=_PRE_TOSS_DELAY_MINUTES))

    # B1.2 — post-toss but first ball not yet bowled
    if overs1 == 0 and overs2 == 0:
        toss_msg = state.find_message(day_entry, f"toss_{idx}")
        if not toss_msg:
            # We see toss_winner in the feed but the tracker hasn't generated
            # its toss message yet (same tick). Defer — next tick handles it.
            return ("post_toss_pre_play", False)
        try:
            toss_ts = datetime.fromisoformat(toss_msg["generated_at"])
        except (KeyError, ValueError):
            return ("post_toss_pre_play", False)
        return (
            "post_toss_pre_play",
            now_pt > toss_ts + timedelta(minutes=_POST_TOSS_DELAY_MINUTES),
        )

    # B2 — in play, check overs-frozen against state-tracked last-progress timestamp
    secs = state.seconds_since_overs_progress(day_entry, match["id"], now=now_pt)
    if secs is None:
        return ("in_play", False)
    return ("in_play", secs >= _IN_PLAY_FREEZE_MINUTES * 60)


def _maybe_generate_delay_phases(
    day_entry: dict, date_iso: str, todays: list[dict], now_pt: datetime,
) -> int:
    """Emit status_update / status_resumed messages based on B1 + B2 gates.

    Runs before _maybe_generate_in_match_phases so a transition like
    pre_toss → toss-fires-this-tick produces both a status_resumed (here) and
    the toss milestone message (there). The iMessage sender will deliver the
    newer of the two and mark the other skipped, which is the right behavior:
    the milestone carries more info than the bare resumption ping.
    """
    generated = 0
    for idx, match in enumerate(todays, 1):
        match_id = match["id"]
        status = match.get("status", "scheduled")

        # Track overs progress regardless of gate state — gives us the
        # last_overs_progress_at timestamp B2 needs.
        inn1 = match.get("inn1") or {}
        inn2 = match.get("inn2") or {}
        if status in ("live", "complete"):
            cur_overs = (inn2.get("overs") if (inn2 and inn2.get("overs"))
                         else (inn1.get("overs") if inn1 else None))
            if cur_overs is not None:
                state.record_overs_progress(day_entry, match_id, overs=cur_overs, now=now_pt)

        if status == "complete":
            # Don't open new delays on a completed match. If we had an active
            # delay, treat completion (or transition to complete) as a clear.
            rec = state.get_delay_record(day_entry, match_id)
            if rec["state"] == "active":
                resumed_phase = rec.get("entered_phase") or "in_play"
                _emit_status_resumed(
                    day_entry, date_iso, idx, match, rec, now_pt, resumed_phase,
                )
                state.set_delay_cleared(day_entry, match_id, now=now_pt)
                generated += 1
            continue

        phase, gate_open = _delay_gate_status(match, idx, day_entry, now_pt)
        rec = state.get_delay_record(day_entry, match_id)
        prev_state = rec["state"]
        prev_hash = rec["last_note_hash"]
        note = (match.get("note") or "").strip() or None
        nh = _note_hash(note)

        if gate_open:
            state.set_delay_active(day_entry, match_id, now=now_pt, note_hash=nh, phase=phase)
            fresh = state.get_delay_record(day_entry, match_id)
            should_emit = (prev_state != "active") or (prev_hash != nh)
            if fresh["fired_count"] > _DELAY_SOFT_CAP:
                should_emit = False
            if should_emit:
                key = f"status_update_{idx}_{fresh['fired_count']}"
                body = message_builder.status_update_message(match, phase=phase, note=note)
                msg = state.add_or_update_message(day_entry, key, body)
                html_archive.upsert_message(date_iso, key, msg["generated_at"], body)
                generated += 1
                _log(f"generated {key}: {match['teams']} ({phase})", "ok")
        elif prev_state == "active":
            # Use the phase the delay STARTED in, not the current phase — a
            # pre-toss delay clearing should read "Toss is underway", not
            # "Play has started", even though the match has now advanced.
            resumed_phase = rec.get("entered_phase") or phase
            _emit_status_resumed(day_entry, date_iso, idx, match, rec, now_pt, resumed_phase)
            state.set_delay_cleared(day_entry, match_id, now=now_pt)
            generated += 1

    return generated


def _emit_status_resumed(
    day_entry: dict, date_iso: str, idx: int, match: dict, rec: dict,
    now_pt: datetime, phase: str,
) -> None:
    entered_at = rec.get("entered_at")
    delay_minutes = 0.0
    if entered_at:
        try:
            t0 = datetime.fromisoformat(entered_at)
            delay_minutes = (now_pt - t0).total_seconds() / 60.0
        except ValueError:
            pass
    key = f"status_resumed_{idx}_{rec.get('fired_count', 0)}"
    body = message_builder.status_resumed_message(match, phase=phase, delay_minutes=delay_minutes)
    msg = state.add_or_update_message(day_entry, key, body)
    html_archive.upsert_message(date_iso, key, msg["generated_at"], body)
    _log(f"generated {key}: {match['teams']} ({phase}, {int(round(delay_minutes))}min)", "ok")


def _maybe_generate_in_match_phases(
    day_entry: dict, date_iso: str,
    todays: list[dict], standings: list[dict],
    recent: list[dict], squads: dict,
) -> int:
    """For each live match, emit toss / powerplay_1 / innings_break / powerplay_2
    messages when the milestone is newly crossed (idempotent — once per match)."""
    generated = 0
    for idx, match in enumerate(todays, 1):
        st = match.get("status")
        if st not in ("live", "complete"):
            continue  # not started yet

        # Toss — fires as soon as TossTeam appears
        if match.get("toss_winner"):
            key = f"toss_{idx}"
            if not state.find_message(day_entry, key):
                body = message_builder.toss_message(match, standings, recent, squads, recent)
                msg = state.add_or_update_message(day_entry, key, body)
                html_archive.upsert_message(date_iso, key, msg["generated_at"], body)
                generated += 1
                _log(f"generated {key}: {match['teams']}", "ok")

        inn1 = match.get("inn1") or {}
        inn2 = match.get("inn2") or {}

        # Powerplay 1 — innings 1 has crossed 6 overs (or first innings complete)
        if inn1.get("overs", 0) >= 6.0:
            key = f"powerplay_1_{idx}"
            if not state.find_message(day_entry, key):
                body = message_builder.powerplay_1_message(match, standings, recent)
                msg = state.add_or_update_message(day_entry, key, body)
                html_archive.upsert_message(date_iso, key, msg["generated_at"], body)
                generated += 1
                _log(f"generated {key}: {match['teams']}", "ok")

        # Innings break — first innings reached 20 overs OR current_innings flipped to 2
        innings_done = (
            inn1.get("overs", 0) >= 19.99
            or match.get("current_innings") == 2
            or (inn2 and inn2.get("runs") is not None)
        )
        if innings_done:
            key = f"innings_break_{idx}"
            if not state.find_message(day_entry, key):
                body = message_builder.innings_break_message(match, standings, recent)
                msg = state.add_or_update_message(day_entry, key, body)
                html_archive.upsert_message(date_iso, key, msg["generated_at"], body)
                generated += 1
                _log(f"generated {key}: {match['teams']}", "ok")

        # Powerplay 2 — chase has crossed 6 overs
        if inn2.get("overs", 0) >= 6.0:
            key = f"powerplay_2_{idx}"
            if not state.find_message(day_entry, key):
                body = message_builder.powerplay_2_message(match, standings, recent)
                msg = state.add_or_update_message(day_entry, key, body)
                html_archive.upsert_message(date_iso, key, msg["generated_at"], body)
                generated += 1
                _log(f"generated {key}: {match['teams']}", "ok")

    return generated


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
        # Use already-imported predictor (worked around in module-vs-script init at top)
        if __package__ in (None, ""):
            from src import predictor as _p
        else:
            from . import predictor as _p
        predicted, _ = _p.predict_winner(
            match["teams"][0], match["teams"][1],
            standings, recent, squads, all_teams,
            match=match, completed_matches=recent,
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
# Signup → recipients.txt sync
# ─────────────────────────────────────────

SIGNUP_SHEET_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbx8wwSgBEPz-SMMTtsi2sEt9xzAUWgDBAtN7Wdg94wJb8VLT-Q5dctZDO0rl_1s4yV6/exec"
)


def _sync_recipients_from_sheet() -> list[str]:
    """Pull approved signup phone numbers from the Apps Script Sheet and merge
    them into recipients.txt. Best-effort — silent on network/auth failures
    so a brief network blip never blocks message generation.

    Returns the list of phones newly added in this run (used for catch-up).

    Requires SHEET_SYNC_TOKEN env var (matches a constant in the Apps Script).
    """
    import os as _os
    import requests as _requests

    token = _os.environ.get("SHEET_SYNC_TOKEN", "").strip()
    if not token:
        return []  # sync disabled — recipients.txt manually managed

    try:
        r = _requests.get(
            SIGNUP_SHEET_URL,
            params={"action": "recipients", "token": token},
            timeout=10,
            allow_redirects=True,
        )
        if r.status_code != 200:
            _log(f"recipient sync HTTP {r.status_code}", "warn")
            return []
        data = r.json()
    except (_requests.RequestException, ValueError) as e:
        _log(f"recipient sync failed: {e}", "warn")
        return []

    if not data.get("ok"):
        _log(f"recipient sync rejected: {data.get('error')}", "warn")
        return []

    raw_phones = data.get("recipients") or []

    # Google Sheets strips the leading + from E.164 numbers because they look
    # numeric. Restore it for any all-digit value that's 10–15 chars.
    def _normalize(p):
        p = str(p or "").strip()
        if not p:
            return ""
        if p.startswith("+"):
            return p
        if p.isdigit() and 10 <= len(p) <= 15:
            return "+" + p
        return p

    sheet_phones = [_normalize(p) for p in raw_phones]
    sheet_phones = [p for p in sheet_phones if p]
    if not sheet_phones:
        return []

    path = REPO_ROOT / "recipients.txt"
    existing: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                existing.add(line)

    # Skip phones that previously opted out — they must START to rejoin
    opted_out = _read_phone_list(REPO_ROOT / "optout.txt")
    added = [p for p in sheet_phones if p and p not in existing and p not in opted_out]
    if not added:
        return []

    # Append (preserves ordering and any header comments already in the file)
    with path.open("a", encoding="utf-8") as f:
        if existing or path.stat().st_size == 0:
            pass  # plain append
        f.write("\n# auto-synced from signup form\n")
        for p in added:
            f.write(p + "\n")
    _log(f"synced {len(added)} new recipient(s) from signup sheet", "ok")
    return added


# ─────────────────────────────────────────
# Catch-up — send today's earlier messages to a newly-signed-up phone
# ─────────────────────────────────────────

def _catch_up_new_recipients(new_phones: list[str], day_entry: dict) -> int:
    """For each brand-new recipient, send today's already-generated messages
    in chronological order so they don't miss what fired earlier in the day.

    Skips messages that haven't been delivered to anyone yet (those will go
    out via the normal broadcast). Skips the catch-up if there's nothing to
    catch up on. Returns the number of catch-up sends attempted.
    """
    if not new_phones:
        return 0

    earlier = [
        m for m in day_entry.get("messages", [])
        if m.get("delivered") and m.get("body")
    ]
    earlier.sort(key=lambda m: m.get("generated_at", ""))
    if not earlier:
        _log(f"catch-up: {len(new_phones)} new recipient(s) but no prior messages today")
        return 0

    total_sends = 0
    for phone in new_phones:
        _log(f"catch-up: sending {len(earlier)} message(s) to {phone}", "ok")
        for msg in earlier:
            ok = imessage_sender.send_to(msg["body"], phone)
            total_sends += 1
            if not ok:
                _log(f"  catch-up send to {phone} failed at {msg['type']}", "warn")
                break  # likely a delivery issue; stop spamming this phone
            # Small pause between sends so iMessage doesn't bundle/throttle
            import time as _time
            _time.sleep(0.5)
    return total_sends


# ─────────────────────────────────────────
# STOP / START opt-out handling
# ─────────────────────────────────────────

def _read_phone_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def _write_phone_list(path: Path, phones: list[str], header_comment: str | None = None) -> None:
    """Atomically rewrite a phone list file."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        if header_comment:
            f.write(header_comment.rstrip() + "\n\n")
        for p in phones:
            f.write(p + "\n")
    tmp.replace(path)


def _process_optout_commands() -> None:
    """Read STOP / START commands from chat.db and apply them."""
    try:
        if __package__ in (None, ""):
            from src import messages_reader  # type: ignore
        else:
            from . import messages_reader
    except Exception as e:
        _log(f"messages_reader import failed: {e}", "warn")
        return

    recipients_path = REPO_ROOT / "recipients.txt"
    optout_path = REPO_ROOT / "optout.txt"

    actives = _read_phone_list(recipients_path)
    opted = _read_phone_list(optout_path)
    eligible = actives | opted
    if not eligible:
        return

    commands = messages_reader.poll(eligible)
    if not commands:
        return

    # De-duplicate by phone — last command wins
    final: dict[str, str] = {}
    for c in commands:
        final[c["phone"]] = c["command"]

    changes_made = False
    for phone, command in final.items():
        if command == "stop":
            if phone in actives:
                actives.discard(phone)
                opted.add(phone)
                changes_made = True
                _log(f"STOP from {phone} — removed from recipients", "ok")
                _send_confirmation(phone, "stop")
            elif phone in opted:
                _log(f"STOP from {phone} — already opted out, ignoring")
        elif command == "start":
            if phone in opted:
                opted.discard(phone)
                actives.add(phone)
                changes_made = True
                _log(f"START from {phone} — restored to recipients", "ok")
                _send_confirmation(phone, "start")
            elif phone in actives:
                _log(f"START from {phone} — already active, ignoring")

    if changes_made:
        _write_phone_list(
            recipients_path,
            sorted(actives),
            header_comment=(
                "# iMessage recipients for IPL tracker. One per line.\n"
                "# Phones: +14155551234 (E.164, with country code).\n"
                "# Lines starting with # are comments. Edit anytime — picked up next run."
            ),
        )
        _write_phone_list(
            optout_path,
            sorted(opted),
            header_comment="# Opted-out recipients. STOP via iMessage adds them here; START removes.",
        )


def _send_confirmation(phone: str, kind: str) -> None:
    """Send a short confirmation iMessage to the recipient after STOP/START."""
    if kind == "stop":
        body = "You're off the IPL tracker list. Reply START anytime to rejoin."
    elif kind == "start":
        body = "Welcome back to the IPL tracker. Match-day texts will resume on the next match."
    else:
        return
    ok = imessage_sender.send_to(body, phone)
    if not ok:
        _log(f"confirmation send to {phone} failed", "warn")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main() -> int:
    _log("tracker run starting")
    partial = False

    state_obj = state.load()

    # Process inbound STOP / START commands first so opt-outs land before
    # any new sync re-adds them.
    _process_optout_commands()

    # Pull any new signups before generating messages so newcomers get
    # whichever message is next due to fire. Hold onto the list so we can
    # catch them up on earlier messages from today after the generators run.
    new_signups = _sync_recipients_from_sheet()

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

    # If today has any non-complete match, force-refresh the fixtures cache
    # so we get the latest toss + innings state from iplt20. The default
    # 24h cache hides mid-match updates (toss_winner, FirstBattingSummary,
    # etc.) that we need for phase-message generation.
    if any(m.get("status") != "complete" for m in todays):
        try:
            all_matches = data_fetcher.fetch_fixtures(force=True)
            todays = _matches_for_date_pt(all_matches, today)
            _log("forced fixtures refresh — live match(es) present today", "ok")
        except Exception as e:
            _log(f"fixtures force-refresh failed: {e}", "warn")

    # Refresh live match status (TTL 60s) for any not-yet-complete match
    for m in todays:
        if m.get("status") != "complete":
            latest = data_fetcher.fetch_current_match(m["id"])
            if latest:
                m["status"] = latest.get("status", m.get("status"))
                m["result"] = latest.get("result") or m.get("result")
                if latest.get("winner"):
                    m["winner"] = latest["winner"]
                # Carry through delay-detection fields so 60s-refreshed status
                # text reaches _maybe_generate_delay_phases.
                if latest.get("note") is not None:
                    m["note"] = latest["note"]
                if latest.get("result_type") is not None:
                    m["result_type"] = latest["result_type"]

    day_entry = state.day(state_obj, today)
    recent = _completed_match_lookup(all_matches)
    remaining = [m for m in all_matches if m.get("status") != "complete"]

    # Generate any messages that should exist
    _maybe_generate_morning(day_entry, today, todays, standings, remaining, recent, squads)
    # Delay-phase gates run BEFORE in_match_phases so a resumption + milestone
    # firing on the same tick produce both messages; the iMessage sender will
    # auto-skip the resumption in favor of the more-informative milestone.
    _maybe_generate_delay_phases(day_entry, today, todays, now_pt())
    _maybe_generate_in_match_phases(day_entry, today, todays, standings, recent, squads)
    _maybe_generate_post_match(day_entry, today, todays, standings, remaining, recent, squads)
    _maybe_generate_end_of_day(day_entry, today, todays, standings, remaining, recent, squads)

    # Catch up brand-new signups on today's earlier messages BEFORE the broadcast,
    # so they receive messages in chronological order:
    #   1. catch-up (earlier delivered messages, chronological)
    #   2. broadcast (the latest undelivered message, sent to everyone)
    _catch_up_new_recipients(new_signups, day_entry)

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

    # Fill the hero cards (Most recent / Leader) from live data
    _update_hero(all_matches, standings)

    if not _git_push_if_changes():
        partial = True

    _log("tracker run complete", "ok")
    return 2 if partial else 0


def _update_hero(all_matches: list[dict], standings: list[dict]) -> None:
    """Replace __HERO_*__ placeholders in docs/index.html with current data."""
    index = REPO_ROOT / "docs" / "index.html"
    if not index.exists():
        return
    try:
        html = index.read_text(encoding="utf-8")
    except OSError:
        return

    # Most recent completed match
    completed = [m for m in all_matches if m.get("status") == "complete" and m.get("winner")]
    completed.sort(key=lambda m: (m.get("date_ist", ""), m.get("scheduled_ist", "")), reverse=True)
    most_recent = completed[0] if completed else None

    if most_recent:
        teams = most_recent.get("teams", ["", ""])
        winner = most_recent.get("winner")
        loser = teams[1] if winner == teams[0] else teams[0]
        team_line = f'{teams[0]} <span class="vs">vs</span> {teams[1]}'
        venue = most_recent.get("venue_name", "")
        date = most_recent.get("date_ist", "")
        try:
            from datetime import datetime as _dt
            date_disp = _dt.strptime(date, "%Y-%m-%d").strftime("%b %-d")
        except (ValueError, TypeError):
            date_disp = date
        meta_line = f"{date_disp}" + (f" · {venue}" if venue else "")
        # Extract just "X by Y" from the result text
        result_txt = most_recent.get("result") or ""
        margin_match = re.search(r" by (.+?)$", result_txt, re.IGNORECASE)
        margin = margin_match.group(1).strip() if margin_match else result_txt
        win_line = f"{winner} won by {margin}" if margin else f"{winner} won"
    else:
        team_line = "Season opens soon"
        meta_line = "IPL 2026"
        win_line = "First match coming"

    # Leader (top of standings)
    if standings:
        leader = standings[0]
        leader_team = leader.get("team", "—")
        wins = leader.get("won", 0)
        played = leader.get("played", 0)
        pts = leader.get("points", 0)
        nrr = leader.get("nrr", 0.0)
        leader_desc = f"{pts} pts · NRR {nrr:+.3f} · {wins} of {played} won"
    else:
        leader_team = "—"
        leader_desc = "Standings pending"

    # Use BeautifulSoup to update by element ID so this works whether the
    # HTML still has raw __HERO_X__ placeholders (first run) or already-
    # substituted values from a previous run (every subsequent run).
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    def _set(el_id: str, html_content: str):
        el = soup.find(id=el_id)
        if el is None:
            return
        el.clear()
        # team_line contains <span class="vs">vs</span> markup; parse as fragment
        if "<" in html_content:
            for child in BeautifulSoup(html_content, "html.parser").contents:
                el.append(child)
        else:
            el.string = html_content

    _set("hero-match",       team_line)
    _set("hero-meta",        meta_line)
    _set("hero-win",         win_line)
    _set("hero-leader",      leader_team)
    _set("hero-leader-desc", leader_desc)

    count_el = soup.find(id="match-count")
    if count_el is not None:
        count_el.clear()
        count_el.string = f"Day {len(completed)} of 74"

    new_html = str(soup)
    if new_html != html:
        index.write_text(new_html, encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:  # last-resort safety net
        _log(f"unhandled exception: {e}", "err")
        sys.exit(1)
