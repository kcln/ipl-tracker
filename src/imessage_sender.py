"""Send iMessages via Messages.app on macOS using AppleScript.

Recipient sources, in priority order:
  1. `recipients.txt` at the repo root — one handle per line; `#` starts a
     comment. Edit it anytime; the next run picks it up. No reload needed.
  2. `$IMESSAGE_RECIPIENT` env var — single handle or comma-separated list.

A handle is either an E.164 phone (`+14155551234`) or an Apple-ID email
tied to iMessage. Each recipient gets the message as an individual iMessage
(not a group chat).
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPIENTS_FILE = REPO_ROOT / "recipients.txt"


def _log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"\033[2m[{ts}] [imessage]\033[0m {msg}", file=sys.stderr)


def _escape_for_applescript(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _ensure_messages_running() -> None:
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "Messages" to activate'],
            check=False, capture_output=True, timeout=8,
        )
    except (subprocess.SubprocessError, OSError) as e:
        _log(f"could not activate Messages.app: {e}")


def _parse_recipients() -> list[str]:
    """File wins; env var is the fallback. De-duplicates while preserving order."""
    recipients: list[str] = []
    seen: set[str] = set()

    if RECIPIENTS_FILE.exists():
        try:
            for raw_line in RECIPIENTS_FILE.read_text(encoding="utf-8").splitlines():
                line = raw_line.split("#", 1)[0].strip()
                if line and line not in seen:
                    seen.add(line)
                    recipients.append(line)
        except OSError as e:
            _log(f"could not read {RECIPIENTS_FILE.name}: {e}")

    if not recipients:
        env_val = os.environ.get("IMESSAGE_RECIPIENT", "")
        for r in env_val.split(","):
            r = r.strip()
            if r and r not in seen:
                seen.add(r)
                recipients.append(r)

    return recipients


def _send_one(body: str, recipient: str) -> bool:
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
        _log(f"osascript timed out for {recipient}")
        return False
    except OSError as e:
        _log(f"osascript not available: {e}")
        return False
    if result.returncode != 0:
        _log(f"send to {recipient} failed: rc={result.returncode} {result.stderr.strip()}")
        return False
    _log(f"sent {len(body)} chars to {recipient}")
    return True


def send_to(body: str, phone: str) -> bool:
    """Send to a single specific phone, bypassing recipients.txt entirely.
    Used for STOP/START confirmation messages."""
    if not phone or not body or not body.strip():
        return False
    _ensure_messages_running()
    return _send_one(body, phone)


def send(body: str) -> bool:
    """Send `body` to every recipient in IMESSAGE_RECIPIENT.

    Returns True if **at least one** recipient received it. Best-effort —
    a single bad number won't block delivery to the others.
    """
    recipients = _parse_recipients()
    if not recipients:
        _log("IMESSAGE_RECIPIENT env var not set; skipping send")
        return False
    if not body or not body.strip():
        _log("empty message body; skipping send")
        return False

    _ensure_messages_running()
    ok_count = sum(1 for r in recipients if _send_one(body, r))
    if ok_count == 0:
        _log(f"all {len(recipients)} sends failed")
        return False
    if ok_count < len(recipients):
        _log(f"partial: {ok_count}/{len(recipients)} sends succeeded")
    return True


if __name__ == "__main__":
    ok = send("ipl-tracker self-test " + datetime.now().isoformat(timespec="seconds"))
    sys.exit(0 if ok else 1)
