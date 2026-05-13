"""End-to-end simulation of today's RCB/KKR wet-field scenario.

Drives _maybe_generate_delay_phases across a 12-tick (3-hour) window with
the live match data evolving in the way iplt20 would expose it. Verifies the
full sequence of messages the tracker should emit.

Run: ./venv/bin/python3 scripts/smoke_test_wet_field.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import state, tracker  # noqa: E402


SCHED_PT = datetime(2026, 5, 13, 7, 0, 0, tzinfo=state.PT)  # 19:30 IST


def _match(*, toss_winner=None, overs1=None, status="scheduled", note=None):
    inn1 = {"runs": 0, "wkts": 0, "overs": overs1, "raw": ""} if overs1 is not None else None
    return {
        "id": "1001",
        "teams": ["RCB", "KKR"],
        "date_ist": "2026-05-13",
        "scheduled_ist": "19:30",
        "status": status,
        "toss_winner": toss_winner,
        "toss_decision": "field" if toss_winner else None,
        "inn1": inn1,
        "inn2": None,
        "note": note,
    }


def main() -> int:
    state_obj = {"days": {}}
    day_entry = state.day(state_obj, "2026-05-13")

    # Simulate the ticks. (minute_offset_from_scheduled_start, match_snapshot)
    timeline = [
        # 19:30 IST, on-time tick — no toss yet, no upstream note yet
        (0,  _match()),
        # 19:45 IST — 15 min late, still no toss. iplt20 surfaces "Wet outfield".
        (15, _match(note="Wet outfield, inspection at 8:15 PM IST")),
        # 20:00 IST — still delayed, no new info
        (30, _match(note="Wet outfield, inspection at 8:15 PM IST")),
        # 20:15 IST — new inspection text
        (45, _match(note="Inspection: pitch ready, toss at 8:30 PM IST")),
        # 20:32 IST — toss happens, but no overs yet
        (62, _match(toss_winner="RCB", status="live", note=None)),
        # 21:00 IST — 28 min after toss, still no overs (under 30-min gate)
        (90, _match(toss_winner="RCB", status="live", note=None)),
        # 21:15 IST — 43 min after toss, still no overs. Gate opens.
        (105, _match(toss_winner="RCB", status="live",
                     note="Outfield damp; toss complete, play delayed")),
        # 21:35 IST — first ball finally bowled
        (125, _match(toss_winner="RCB", status="live", overs1=0.1)),
        # 22:00 IST — play in progress, 4 overs in (advancing)
        (150, _match(toss_winner="RCB", status="live", overs1=4.0)),
        # 22:15 IST — frozen at 4.0 (rain stoppage in real time but no upstream note yet)
        (165, _match(toss_winner="RCB", status="live", overs1=4.0)),
        # 22:30 IST — still frozen, upstream now reports rain
        (180, _match(toss_winner="RCB", status="live", overs1=4.0,
                     note="Rain stoppage at 4.0 overs")),
        # 22:55 IST — play resumes (overs advance)
        (205, _match(toss_winner="RCB", status="live", overs1=4.3)),
    ]

    with patch.object(tracker.html_archive, "upsert_message"):
        for offset, m in timeline:
            now_pt = SCHED_PT + timedelta(minutes=offset)
            # Simulate the toss message having been generated when toss_winner first appears.
            # In real run, _maybe_generate_in_match_phases handles this; we shortcut here.
            if m.get("toss_winner") and not state.find_message(day_entry, "toss_1"):
                tmsg = state.add_or_update_message(day_entry, "toss_1", "toss body")
                tmsg["generated_at"] = now_pt.isoformat()
            tracker._maybe_generate_delay_phases(day_entry, "2026-05-13", [m], now_pt)

    # Report
    print("\n=== Generated messages (in order) ===")
    for msg in day_entry.get("messages", []):
        ts = msg.get("generated_at", "")
        t = msg.get("type", "")
        body_first_line = (msg.get("body") or "").splitlines()[0] if msg.get("body") else ""
        print(f"  [{ts[11:19]}] {t:32s}  {body_first_line}")

    print("\n=== Bodies ===")
    for msg in day_entry.get("messages", []):
        if msg["type"].startswith(("status_update_", "status_resumed_")):
            print(f"\n----- {msg['type']} @ {msg['generated_at']}")
            print(msg["body"])

    # Verify: should have at least 1 status_update + 1 status_resumed during pre-toss,
    # 1 status_update + 1 status_resumed during post-toss-pre-play,
    # 1 status_update + 1 status_resumed during in-play freeze.
    types = [m["type"] for m in day_entry["messages"]]
    update_count = sum(1 for t in types if t.startswith("status_update_"))
    resumed_count = sum(1 for t in types if t.startswith("status_resumed_"))

    print(f"\n=== Summary ===")
    print(f"status_update messages:  {update_count}")
    print(f"status_resumed messages: {resumed_count}")

    expected_updates_min = 3   # pre-toss (at least 1), post-toss (1), in-play (1)
    expected_resumed_min = 3   # one per phase

    if update_count >= expected_updates_min and resumed_count >= expected_resumed_min:
        print(f"\nPASS — at least {expected_updates_min} updates and {expected_resumed_min} resumptions emitted")
        return 0
    print(f"\nFAIL — expected ≥{expected_updates_min} updates and ≥{expected_resumed_min} resumptions")
    return 1


if __name__ == "__main__":
    sys.exit(main())
