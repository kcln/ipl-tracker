"""Read inbound STOP / START commands from macOS Messages (chat.db).

Used by the tracker each run to auto-process unsubscribe and resubscribe
requests texted in by recipients. The Messages database is opened
read-only; we never write to it.

Requires Full Disk Access for the Python interpreter (granted via
System Settings → Privacy & Security → Full Disk Access).

Cursor (last processed message rowid) lives at data/messages_cursor.json
so commands are processed exactly once across tracker runs.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CURSOR_FILE = REPO_ROOT / "data" / "messages_cursor.json"
CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"

# Whole-message keywords. Match is case-insensitive, on the trimmed text.
_STOP_WORDS  = {"stop", "unsubscribe", "opt out", "optout", "opt-out", "end", "quit", "cancel"}
_START_WORDS = {"start", "subscribe", "opt in", "optin", "opt-in", "resume", "yes"}


def _log(msg: str, level: str = "info") -> None:
    colour = {"info": "\033[36m", "warn": "\033[33m", "err": "\033[31m", "ok": "\033[32m"}.get(level, "")
    reset = "\033[0m"
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"{colour}[{ts}] [msgreader] [{level}]{reset} {msg}", file=sys.stderr)


def _load_cursor() -> int:
    if not CURSOR_FILE.exists():
        return 0
    try:
        with CURSOR_FILE.open("r", encoding="utf-8") as f:
            return int(json.load(f).get("last_rowid", 0))
    except (OSError, ValueError, TypeError):
        return 0


def _save_cursor(rowid: int) -> None:
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CURSOR_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"last_rowid": int(rowid)}, f)
    tmp.replace(CURSOR_FILE)


def _classify(text: str) -> str | None:
    """Return 'stop' / 'start' / None for the trimmed lowercased text."""
    t = (text or "").strip().lower()
    if t in _STOP_WORDS:
        return "stop"
    if t in _START_WORDS:
        return "start"
    return None


def poll(eligible_phones: set[str]) -> list[dict]:
    """Return list of {phone, command, rowid} for new STOP/START commands.

    Only honors commands where the sender is in `eligible_phones` (either
    currently subscribed or currently opted-out — random texts are ignored).
    Cursor is advanced past the highest scanned rowid even if no commands
    matched, so the next call only sees newer messages.
    """
    if not CHAT_DB.exists():
        _log(f"chat.db not found at {CHAT_DB}", "warn")
        return []

    last_rowid = _load_cursor()
    commands: list[dict] = []
    max_seen = last_rowid

    try:
        conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.rowid, h.id AS sender, m.text
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.rowid
            WHERE m.is_from_me = 0
              AND m.rowid > ?
              AND m.text IS NOT NULL
              AND length(m.text) BETWEEN 2 AND 60
            ORDER BY m.rowid ASC
            """,
            (last_rowid,),
        )
        rows = cur.fetchall()
        conn.close()
    except sqlite3.OperationalError as e:
        _log(f"chat.db read failed: {e} — grant Full Disk Access to {sys.executable}", "warn")
        return []
    except sqlite3.DatabaseError as e:
        _log(f"chat.db error: {e}", "warn")
        return []

    if not rows:
        return []

    for rowid, sender, text in rows:
        max_seen = max(max_seen, int(rowid))
        if not sender or sender not in eligible_phones:
            continue
        cmd = _classify(text)
        if not cmd:
            continue
        commands.append({"phone": sender, "command": cmd, "rowid": int(rowid)})

    # Advance cursor past everything we scanned, even if no commands matched
    if max_seen > last_rowid:
        _save_cursor(max_seen)

    if commands:
        _log(f"found {len(commands)} STOP/START command(s)", "ok")
    return commands


if __name__ == "__main__":
    # Manual probe: read from current recipients + optout files
    actives = set()
    rp = REPO_ROOT / "recipients.txt"
    if rp.exists():
        for line in rp.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                actives.add(line)
    op = REPO_ROOT / "optout.txt"
    if op.exists():
        for line in op.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                actives.add(line)
    print(f"eligible phones: {sorted(actives)}")
    cmds = poll(actives)
    print(f"commands: {cmds}")
