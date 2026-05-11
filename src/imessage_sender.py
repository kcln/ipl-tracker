"""Send iMessages via Messages.app on macOS using AppleScript.

Reads target handle from $IMESSAGE_RECIPIENT (phone number in E.164,
e.g. +14155551234, or an email tied to iMessage).
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime


def _log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"\033[2m[{ts}] [imessage]\033[0m {msg}", file=sys.stderr)


def _escape_for_applescript(s: str) -> str:
    # Escape backslashes and quotes, preserve newlines as literal \n in script
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _ensure_messages_running() -> None:
    # `tell application "Messages" to activate` will launch it if not running.
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "Messages" to activate'],
            check=False, capture_output=True, timeout=8,
        )
    except (subprocess.SubprocessError, OSError) as e:
        _log(f"could not activate Messages.app: {e}")


def send(body: str) -> bool:
    """Returns True on success, False on failure. Never raises."""
    recipient = os.environ.get("IMESSAGE_RECIPIENT", "").strip()
    if not recipient:
        _log("IMESSAGE_RECIPIENT env var not set; skipping send")
        return False
    if not body or not body.strip():
        _log("empty message body; skipping send")
        return False

    _ensure_messages_running()

    safe_body = _escape_for_applescript(body)
    safe_recipient = recipient.replace('"', '\\"')

    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{safe_recipient}" of targetService
        send "{safe_body}" to targetBuddy
    end tell
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=20,
        )
    except subprocess.TimeoutExpired:
        _log("osascript timed out")
        return False
    except OSError as e:
        _log(f"osascript not available: {e}")
        return False

    if result.returncode != 0:
        _log(f"osascript failed: rc={result.returncode} stderr={result.stderr.strip()}")
        return False
    _log(f"sent {len(body)} chars to {recipient}")
    return True


if __name__ == "__main__":
    ok = send("ipl-tracker self-test " + datetime.now().isoformat(timespec="seconds"))
    sys.exit(0 if ok else 1)
