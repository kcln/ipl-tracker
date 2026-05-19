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
OPTOUT_FILE = REPO_ROOT / "telegram_optout.txt"
ENV_FILE = REPO_ROOT / ".env"

WELCOME_MESSAGE = (
    "🏏 You're on the IPL 2026 tracker, {first_name}.\n"
    "\n"
    "Match-day messages will start landing here:\n"
    "• Prediction before the match\n"
    "• Powerplay + innings updates\n"
    "• Final result and a short recap at night\n"
    "\n"
    "About 7 messages per match day, only on match days. No spam between.\n"
    "\n"
    "Send /stop anytime to leave. Questions or feedback: @kcla21.\n"
    "\n"
    "— KC · kcln.github.io/ipl-tracker"
)

WELCOME_BACK_MESSAGE = (
    "🏏 You're back on the IPL 2026 tracker, {first_name}. "
    "Next match-day update will land here. Send /stop anytime to leave."
)

OPTOUT_CONFIRM_MESSAGE = (
    "You're off the IPL 2026 tracker, {first_name}. No more messages. "
    "Send /start anytime to rejoin."
)


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


def _parse_id_file(path: Path) -> List[str]:
    """Parse a chat-id-per-line file, ignoring comments and blanks."""
    if not path.exists():
        return []
    out: List[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _parse_chat_ids() -> List[str]:
    return _parse_id_file(CHAT_IDS_FILE)


def _parse_optout_ids() -> List[str]:
    return _parse_id_file(OPTOUT_FILE)


def _remove_id_from_file(path: Path, chat_id: str) -> bool:
    """Remove every line whose id field equals chat_id. Preserves headers
    and comment-only lines. Returns True if any line was removed."""
    if not path.exists():
        return False
    kept: List[str] = []
    removed = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line == chat_id:
            removed = True
            continue
        kept.append(raw)
    if removed:
        text = "\n".join(kept)
        if text and not text.endswith("\n"):
            text += "\n"
        path.write_text(text, encoding="utf-8")
    return removed


def _append_chat_id(path: Path, chat_id: str, info: dict, action: str) -> None:
    """Append `<chat_id>  # <name> (<handle>) — <action> <date>` to path."""
    today = datetime.now().strftime("%Y-%m-%d")
    if not path.exists():
        header = (
            "# Telegram chat IDs for ipl-tracker. One per line.\n"
            "# Auto-populated by discover_and_add() from /start signups.\n"
            if path == CHAT_IDS_FILE
            else
            "# Chat IDs that opted out via /stop. discover_and_add skips these.\n"
            "# /start re-subscribes (the entry is removed from this file).\n"
        )
        path.write_text(header, encoding="utf-8")
    name = _format_name(info)
    handle = f"@{info['username']}" if info.get("username") else "no @handle"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{chat_id}  # {name} ({handle}) — {action} {today}\n")


def _send_one(body: str, chat_id: str) -> bool:
    result = _api("sendMessage", chat_id=chat_id, text=body)
    if result.get("ok"):
        _log(f"sent {len(body)} chars to {chat_id}")
        return True
    err = result.get("description") or result.get("error") or "unknown"
    _log(f"send to {chat_id} failed: {err}")
    return False


def send(body: str) -> bool:
    """Send to every chat_id in telegram_chat_ids.txt that has NOT opted out.
    Returns True if at least one succeeded — same semantics as imessage_sender.send.
    """
    optout = set(_parse_optout_ids())
    chat_ids = [c for c in _parse_chat_ids() if c not in optout]
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


def _format_name(info: dict) -> str:
    name = (info.get("first_name") or "").strip()
    last = (info.get("last_name") or "").strip()
    if last:
        name = f"{name} {last}".strip()
    return name or "(no name)"


def _fetch_chats() -> dict[int, dict]:
    """Return {chat_id: info} for every private chat in the bot's 24h update
    window. `info` carries chat metadata plus `text` = the user's most-recent
    message (for /stop, /start command routing). Empty dict on API failure."""
    result = _api("getUpdates", limit=100)
    if not result.get("ok"):
        _log(f"getUpdates failed: {result.get('error') or result.get('description')}")
        return {}
    chats: dict[int, dict] = {}
    # Sort by update_id so later messages overwrite earlier ones per chat
    for u in sorted(result.get("result", []), key=lambda x: x.get("update_id", 0)):
        msg = u.get("message") or u.get("edited_message") or {}
        c = msg.get("chat", {})
        if c.get("id") and c.get("type") == "private":
            chats[c["id"]] = {
                "chat_id": c["id"],
                "first_name": c.get("first_name") or "",
                "last_name": c.get("last_name") or "",
                "username": c.get("username") or "",
                "text": (msg.get("text") or "").strip(),
            }
    return chats


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


def _resolve_admin_id(exclude_ids: set[str]) -> str:
    """First existing chat_id (excluding the just-added set), or the env override."""
    admin_id = os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "").strip()
    if admin_id:
        return admin_id
    for cid in _parse_chat_ids():
        if cid not in exclude_ids:
            return cid
    return ""


def discover_and_add() -> List[dict]:
    """Process all pending Telegram updates: handle /stop opt-outs, /start
    rejoins, and brand-new signups. Returns the list of first-time signups
    that were just added (used for admin DM notices).

    Per-chat routing, based on the user's most-recent message text:
      • starts with '/stop'  → remove from chat_ids, add to optout, send confirm
      • starts with '/start' → if in optout, rejoin (welcome-back); if absent
        from both lists, first-time signup (welcome + admin DM)
      • any other text       → first-time signup if absent from both lists,
        otherwise no-op (already known and not asking to leave)

    Opted-out chat_ids are NEVER silently re-added; they must send /start.
    """
    chats = _fetch_chats()
    if not chats:
        return []

    active = set(_parse_chat_ids())
    optout = set(_parse_optout_ids())
    new_signups: List[dict] = []

    for cid_int, info in chats.items():
        cid = str(cid_int)
        text = info.get("text", "").lower()
        first = (info.get("first_name") or "").strip() or "there"

        if text.startswith("/stop"):
            # Opt out — remove from active list, append to optout, confirm
            if cid in active:
                _remove_id_from_file(CHAT_IDS_FILE, cid)
                active.discard(cid)
            if cid not in optout:
                _append_chat_id(OPTOUT_FILE, cid, info, "opted out")
                optout.add(cid)
                _send_one(OPTOUT_CONFIRM_MESSAGE.format(first_name=first), cid)
                _log(f"opted out {cid} ({_format_name(info)})")
            continue

        is_start_cmd = text.startswith("/start")
        if cid in optout:
            # Previously opted out — only /start can bring them back
            if is_start_cmd:
                _remove_id_from_file(OPTOUT_FILE, cid)
                optout.discard(cid)
                _append_chat_id(CHAT_IDS_FILE, cid, info, "rejoined")
                active.add(cid)
                _send_one(WELCOME_BACK_MESSAGE.format(first_name=first), cid)
                _log(f"rejoined {cid} ({_format_name(info)})")
            continue

        if cid not in active:
            # First-time signup
            _append_chat_id(CHAT_IDS_FILE, cid, info, "joined")
            active.add(cid)
            _send_one(WELCOME_MESSAGE.format(first_name=first), cid)
            new_signups.append(info)
            _log(f"new signup {cid} ({_format_name(info)})")

    # DM the admin about brand-new signups (mirror the prior behavior)
    if new_signups:
        new_ids = {str(n["chat_id"]) for n in new_signups}
        admin_id = _resolve_admin_id(new_ids)
        if admin_id:
            for info in new_signups:
                name = _format_name(info)
                handle = f"@{info['username']}" if info.get("username") else "no @handle"
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

    return new_signups


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
