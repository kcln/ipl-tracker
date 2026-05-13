"""One-shot: re-write the team-vs-team header on past message bodies so the
home team appears first. Updates both state.json (source of truth) and
docs/index.html (rendered) by calling html_archive.upsert_message with the
corrected body. Safe to re-run — idempotent.

Run: ./venv/bin/python3 scripts/heal_past_team_headers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import html_archive, message_builder, state  # noqa: E402


def _home_team_lookup() -> dict[str, str | None]:
    """Build {date_iso -> home_team} from the fixtures cache."""
    cache = REPO / "data" / "_cache" / "fixtures.json"
    if not cache.exists():
        return {}
    raw = json.loads(cache.read_text())
    out: dict[str, str | None] = {}
    for m in raw.get("matches", []):
        d = m.get("date_ist")
        if not d:
            continue
        # Multi-match days: only first wins, fine for our purposes (header
        # rewrite is no-op when teams don't intersect with home)
        out.setdefault(d, m.get("home_team"))
    return out


def main() -> int:
    homes = _home_team_lookup()
    if not homes:
        print("no fixtures cache; nothing to do", file=sys.stderr)
        return 1

    state_obj = state.load()
    days = state_obj.get("days", {})
    rewrites = 0

    for date_iso, day_entry in days.items():
        home = homes.get(date_iso)
        if not home:
            continue
        for msg in day_entry.get("messages", []):
            body = msg.get("body") or ""
            new_body = message_builder.rewrite_team_header(body, home_team=home)
            if new_body != body:
                msg["body"] = new_body
                # Refresh the HTML article so the page picks up the swap
                html_archive.upsert_message(
                    date_iso, msg["type"], msg["generated_at"], new_body,
                )
                rewrites += 1
                print(f"  rewrote {date_iso} / {msg['type']}: "
                      f"{body.splitlines()[0][:40]} -> {new_body.splitlines()[0][:40]}")

    if rewrites == 0:
        print("no rewrites needed")
    else:
        state.save(state_obj)
        print(f"rewrote {rewrites} message(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
