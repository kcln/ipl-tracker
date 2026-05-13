"""State persistence for the IPL tracker.

Atomic JSON read/write keyed by date (YYYY-MM-DD in America/Los_Angeles).
Idempotency-safe: callers add messages/matches by stable id.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
IST = ZoneInfo("Asia/Kolkata")

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "state.json"
# last_run lives in a separate gitignored file so heartbeat updates from
# no-op tracker runs don't dirty state.json (and therefore don't produce
# empty commits / spurious Pages rebuilds).
LAST_RUN_PATH = REPO_ROOT / "data" / "last_run.txt"


def now_pt() -> datetime:
    return datetime.now(PT)


def today_pt_iso() -> str:
    return now_pt().strftime("%Y-%m-%d")


def load() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"days": {}}
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Strip any legacy last_run field on read so it never re-enters state.json
        data.pop("last_run", None)
        return data
    except (json.JSONDecodeError, OSError):
        # Corrupted state — start fresh rather than crashing the launchd job.
        return {"days": {}}


def save(state: dict[str, Any]) -> None:
    """Atomic write of state.json (without heartbeat timestamp) + the
    last_run heartbeat to a separate gitignored file. Splitting these keeps
    no-op tracker runs from producing committable diffs."""
    # Defensive: strip any last_run that snuck into the dict
    state.pop("last_run", None)

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(STATE_PATH.parent), prefix=".state-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, STATE_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Write heartbeat to gitignored file (atomic, no failure-back-into-state)
    try:
        LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_hb = LAST_RUN_PATH.with_suffix(".tmp")
        tmp_hb.write_text(now_pt().isoformat() + "\n", encoding="utf-8")
        tmp_hb.replace(LAST_RUN_PATH)
    except OSError:
        pass  # heartbeat is informational, don't fail the run


def day(state: dict, date_iso: str) -> dict:
    days = state.setdefault("days", {})
    return days.setdefault(date_iso, {"matches": [], "messages": []})


def find_message(day_entry: dict, msg_type: str) -> dict | None:
    for m in day_entry.get("messages", []):
        if m.get("type") == msg_type:
            return m
    return None


def add_or_update_message(day_entry: dict, msg_type: str, body: str) -> dict:
    existing = find_message(day_entry, msg_type)
    if existing is not None:
        existing["body"] = body
        return existing
    new_msg = {
        "type": msg_type,
        "body": body,
        "generated_at": now_pt().isoformat(),
        "delivered": False,
        "delivered_at": None,
        "delivery_skipped": False,
        "skipped_reason": None,
    }
    day_entry.setdefault("messages", []).append(new_msg)
    return new_msg


def latest_undelivered_message(day_entry: dict) -> dict | None:
    candidates = [
        m for m in day_entry.get("messages", [])
        if not m.get("delivered") and not m.get("delivery_skipped")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.get("generated_at") or "")


def mark_older_as_skipped(day_entry: dict, keep_msg: dict, reason: str) -> int:
    skipped = 0
    keep_ts = keep_msg.get("generated_at") or ""
    for m in day_entry.get("messages", []):
        if m is keep_msg:
            continue
        if m.get("delivered") or m.get("delivery_skipped"):
            continue
        if (m.get("generated_at") or "") < keep_ts:
            m["delivery_skipped"] = True
            m["skipped_reason"] = reason
            skipped += 1
    return skipped


def mark_delivered(msg: dict) -> None:
    msg["delivered"] = True
    msg["delivered_at"] = now_pt().isoformat()


# Substrings used by message_builder.post_match_message — keep in sync if
# wording ever changes. The trailing keyword makes the match unambiguous
# against the substring "correct" appearing inside "incorrect".
_PREDICTION_CORRECT_MARKER = "Pre-match prediction: correct"
_PREDICTION_INCORRECT_MARKER = "Pre-match prediction: incorrect"


def season_prediction_record(state_obj: dict) -> tuple[int, int]:
    """Walk every post_match message body across all days and return
    (correct, total). Only counts messages whose `type` starts with
    "post_match" (so a morning/recap body that happens to quote the
    marker string doesn't bump the tally). Abandoned matches don't
    contain the marker — naturally skipped.
    """
    correct = 0
    total = 0
    days = state_obj.get("days") or {}
    for day_entry in days.values():
        for msg in day_entry.get("messages", []) or []:
            if not msg.get("type", "").startswith("post_match"):
                continue
            body = msg.get("body") or ""
            # Order matters: check the longer ("incorrect") phrase first so
            # the substring "correct" inside "incorrect" doesn't false-fire.
            if _PREDICTION_INCORRECT_MARKER in body:
                total += 1
            elif _PREDICTION_CORRECT_MARKER in body:
                correct += 1
                total += 1
    return correct, total


# ---------------------------------------------------------------------------
# Delay tracking: per (day, match_id) state used by the B1 / B2 detection
# gates in tracker._maybe_generate_delay_phases.
#
# `state` transitions: "none" → "active" → "none". A status_update message
# fires on none → active (and on active → active when the upstream note text
# materially changes). A status_resumed message fires on active → none.
# ---------------------------------------------------------------------------

def _delay_bucket(day_entry: dict) -> dict:
    return day_entry.setdefault("delay_state", {})


def get_delay_record(day_entry: dict, match_id: str) -> dict:
    bucket = _delay_bucket(day_entry)
    rec = bucket.setdefault(str(match_id), {
        "state": "none",
        "entered_at": None,
        "entered_phase": None,
        "last_overs_value": None,
        "last_overs_progress_at": None,
        "last_note_hash": None,
        "fired_count": 0,
    })
    # Backfill for records persisted before entered_phase existed
    rec.setdefault("entered_phase", None)
    return rec


def set_delay_active(
    day_entry: dict, match_id: str, *, now: datetime,
    note_hash: str | None, phase: str | None = None,
) -> dict:
    rec = get_delay_record(day_entry, match_id)
    if rec["state"] != "active":
        rec["state"] = "active"
        rec["entered_at"] = now.isoformat()
        rec["entered_phase"] = phase
        rec["last_note_hash"] = note_hash
        rec["fired_count"] += 1
    elif note_hash != rec["last_note_hash"]:
        # Same delay event, new upstream info — re-fire but keep entered_at/phase
        rec["last_note_hash"] = note_hash
        rec["fired_count"] += 1
    return rec


def set_delay_cleared(day_entry: dict, match_id: str, *, now: datetime) -> dict:
    rec = get_delay_record(day_entry, match_id)
    rec["state"] = "none"
    return rec


def record_overs_progress(day_entry: dict, match_id: str, *, overs: float, now: datetime) -> bool:
    rec = get_delay_record(day_entry, match_id)
    prev = rec["last_overs_value"]
    if prev is None or overs != prev:
        rec["last_overs_value"] = overs
        rec["last_overs_progress_at"] = now.isoformat()
        return True
    return False


def seconds_since_overs_progress(day_entry: dict, match_id: str, *, now: datetime) -> float | None:
    rec = get_delay_record(day_entry, match_id)
    ts = rec["last_overs_progress_at"]
    if not ts:
        return None
    prev = datetime.fromisoformat(ts)
    return (now - prev).total_seconds()
