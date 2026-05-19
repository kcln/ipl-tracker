"""Send messages via Telegram Bot API. Parallel sender to imessage_sender.

Setup:
  1. Create bot with @BotFather → store token in `.env` as TELEGRAM_BOT_TOKEN
  2. /start the bot from each recipient's Telegram account
  3. Run `python -m src.telegram_sender --discover` to print discovered chat_ids
  4. Add them to `telegram_chat_ids.txt` (one per line, # for comments)

Both files (`.env` and `telegram_chat_ids.txt`) are gitignored.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAT_IDS_FILE = REPO_ROOT / "telegram_chat_ids.txt"
ENV_FILE = REPO_ROOT / ".env"


def _log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"\033[2m[{ts}] [telegram]\033[0m {msg}", file=sys.stderr)


def _load_env() -> None:
    """Minimal .env loader — no python-dotenv dependency."""
    if not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _get_token() -> str | None:
    _load_env()
    return os.environ.get("TELEGRAM_BOT_TOKEN")


def _api(method: str, **params) -> dict:
    token = _get_token()
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not set"}
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/{method}",
            data=params,
            timeout=15,
        )
        return r.json()
    except requests.RequestException as e:
        return {"ok": False, "error": str(e)}


def _parse_chat_ids() -> List[str]:
    if not CHAT_IDS_FILE.exists():
        return []
    out: List[str] = []
    seen: set[str] = set()
    for raw in CHAT_IDS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _send_one(body: str, chat_id: str) -> bool:
    result = _api("sendMessage", chat_id=chat_id, text=body)
    if result.get("ok"):
        _log(f"sent {len(body)} chars to {chat_id}")
        return True
    err = result.get("description") or result.get("error") or "unknown"
    _log(f"send to {chat_id} failed: {err}")
    return False


def send(body: str) -> bool:
    """Send to every chat_id in telegram_chat_ids.txt.
    Returns True if at least one succeeded — same semantics as imessage_sender.send.
    """
    chat_ids = _parse_chat_ids()
    if not chat_ids:
        _log("no chat_ids configured; skipping")
        return False
    if not body or not body.strip():
        _log("empty body; skipping")
        return False

    ok = sum(1 for c in chat_ids if _send_one(body, c))
    if ok == 0:
        _log(f"all {len(chat_ids)} sends failed")
        return False
    if ok < len(chat_ids):
        _log(f"partial: {ok}/{len(chat_ids)} succeeded")
    return True


def send_to(body: str, chat_id: str) -> bool:
    if not body or not body.strip() or not chat_id:
        return False
    return _send_one(body, chat_id)


def _fetch_chats() -> dict[int, dict]:
    """Return {chat_id: {first_name, last_name, username}} for every private
    chat visible in the bot's 24h update window. Empty dict on API failure."""
    result = _api("getUpdates", limit=100)
    if not result.get("ok"):
        _log(f"getUpdates failed: {result.get('error') or result.get('description')}")
        return {}
    chats: dict[int, dict] = {}
    for u in result.get("result", []):
        msg = u.get("message") or u.get("edited_message") or {}
        c = msg.get("chat", {})
        if c.get("id") and c.get("type") == "private":
            chats[c["id"]] = {
                "chat_id": c["id"],
                "first_name": c.get("first_name") or "",
                "last_name": c.get("last_name") or "",
                "username": c.get("username") or "",
            }
    return chats


def _format_name(info: dict) -> str:
    name = (info.get("first_name") or "").strip()
    last = (info.get("last_name") or "").strip()
    if last:
        name = f"{name} {last}".strip()
    return name or "(no name)"


def discover() -> List[dict]:
    """Print chat_ids visible via getUpdates so the user can populate
    telegram_chat_ids.txt."""
    chats = _fetch_chats()
    out = list(chats.values())
    if not out:
        print("No updates. /start the bot in Telegram first.")
    else:
        print(f"Found {len(out)} chat(s):")
        for c in out:
            print(f"  {c}")
    return out


def discover_and_add() -> List[dict]:
    """Auto-add any new /start-ers to telegram_chat_ids.txt and DM the admin.

    Diffs `getUpdates` against the existing file; appends any private chats
    not yet listed and pings the admin (first existing chat_id, or
    `TELEGRAM_ADMIN_CHAT_ID` if set) with a one-line signup notice. Returns
    the list of newly-added entries. Safe to call repeatedly — idempotent on
    chat_ids already in the file."""
    chats = _fetch_chats()
    if not chats:
        return []

    existing = set(_parse_chat_ids())
    new = [info for cid, info in chats.items() if str(cid) not in existing]
    if not new:
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    # Ensure file exists with a header if this is the first signup ever
    if not CHAT_IDS_FILE.exists():
        CHAT_IDS_FILE.write_text(
            "# Telegram chat IDs for ipl-tracker. One per line.\n"
            "# Auto-populated by discover_and_add() from web signups (/start @Kipl26bot).\n",
            encoding="utf-8",
        )
    with CHAT_IDS_FILE.open("a", encoding="utf-8") as f:
        for info in new:
            name = _format_name(info)
            handle = f"@{info['username']}" if info["username"] else "no @handle"
            f.write(f"{info['chat_id']}  # {name} ({handle}) — joined {today}\n")
    _log(f"auto-added {len(new)} new chat_id(s) to {CHAT_IDS_FILE.name}")

    # Pick admin: env override, else the first pre-existing chat_id in the file
    admin_id = os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "").strip()
    if not admin_id:
        pre_existing = [c for c in _parse_chat_ids() if c not in {str(n["chat_id"]) for n in new}]
        admin_id = pre_existing[0] if pre_existing else ""
    if admin_id:
        for info in new:
            name = _format_name(info)
            handle = f"@{info['username']}" if info["username"] else "no @handle"
            body = (
                f"🆕 New IPL tracker signup\n"
                f"Name: {name}\n"
                f"Handle: {handle}\n"
                f"chat_id: {info['chat_id']}\n"
                f"Auto-added to recipients."
            )
            _send_one(body, admin_id)
    else:
        _log("no admin chat_id available; new signups added but no DM sent")

    return new


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--discover":
        discover()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--discover-and-add":
        added = discover_and_add()
        print(f"Added {len(added)} new chat(s).")
        sys.exit(0)
    body = " ".join(sys.argv[1:]) or f"ipl-tracker telegram self-test {datetime.now().isoformat(timespec='seconds')}"
    ok = send(body)
    sys.exit(0 if ok else 1)
