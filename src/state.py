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


def now_pt() -> datetime:
    return datetime.now(PT)


def today_pt_iso() -> str:
    return now_pt().strftime("%Y-%m-%d")


def load() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"last_run": None, "days": {}}
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupted state — start fresh rather than crashing the launchd job.
        return {"last_run": None, "days": {}}


def save(state: dict[str, Any]) -> None:
    """Atomic write: tmp file in same dir, then rename."""
    state["last_run"] = now_pt().isoformat()
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
