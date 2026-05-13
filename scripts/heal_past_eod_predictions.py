"""One-shot: rewrite the legacy 'Predictions today: X of Y correct' line in
past end_of_day messages to 'Season to date: X of Y correct (Z%)' using a
running cumulative tally through each day. Updates both state.json and
docs/index.html. Safe to re-run — idempotent (bodies already in the new
format are left alone by rewrite_eod_predictions_line).

Run: ./venv/bin/python3 scripts/heal_past_eod_predictions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import html_archive, message_builder, state  # noqa: E402


_INCORRECT = "Pre-match prediction: incorrect"
_CORRECT = "Pre-match prediction: correct"


def _count_day(day_entry: dict) -> tuple[int, int]:
    c = t = 0
    for msg in day_entry.get("messages", []) or []:
        if not msg.get("type", "").startswith("post_match"):
            continue
        body = msg.get("body") or ""
        if _INCORRECT in body:
            t += 1
        elif _CORRECT in body:
            c += 1
            t += 1
    return c, t


def main() -> int:
    state_obj = state.load()
    days = state_obj.get("days") or {}
    if not days:
        print("no days in state.json")
        return 0

    rewrites = 0
    cum_c = cum_t = 0
    for date_iso in sorted(days.keys()):
        day = days[date_iso]
        day_c, day_t = _count_day(day)
        cum_c += day_c
        cum_t += day_t

        eod = state.find_message(day, "end_of_day")
        if not eod:
            continue
        old_body = eod.get("body") or ""
        new_body = message_builder.rewrite_eod_predictions_line(
            old_body, correct=cum_c, total=cum_t,
        )
        if new_body != old_body:
            eod["body"] = new_body
            html_archive.upsert_message(
                date_iso, "end_of_day", eod["generated_at"], new_body,
            )
            rewrites += 1
            old_first = next((l for l in old_body.splitlines() if "Predictions today" in l), "")
            new_first = next((l for l in new_body.splitlines() if "Season to date" in l), "")
            print(f"  {date_iso}: {old_first.strip()}  →  {new_first.strip()}")

    if rewrites == 0:
        print("no rewrites needed")
    else:
        state.save(state_obj)
        print(f"rewrote {rewrites} end_of_day body(ies); cumulative {cum_c}/{cum_t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
