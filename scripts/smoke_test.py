"""Offline smoke test: stub out data_fetcher with synthetic IPL data and run
the orchestrator end-to-end. Verifies:

  * morning + post_match + end_of_day messages are generated
  * docs/index.html contains today's section + articles
  * state.json carries delivered/skipped flags correctly

Run:   ./venv/bin/python3 scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Force a known date so we don't depend on "today" having matches
from src import state, data_fetcher, tracker, imessage_sender  # noqa: E402
from datetime import datetime  # noqa: E402

TODAY = state.today_pt_iso()

# Today's PT date in IST terms: an IPL "afternoon" match at 15:30 IST falls on
# the same PT calendar date as ~02:00 PT same morning, so build a fixture set
# whose IST date matches PT today.
date_ist = datetime.strptime(TODAY, "%Y-%m-%d").strftime("%Y-%m-%d")

FIXTURES = [
    {"id": "1001", "teams": ["CSK", "MI"], "date_ist": date_ist,
     "scheduled_ist": "15:30", "status": "complete", "result": "CSK won by 18 runs",
     "winner": "CSK"},
    {"id": "1002", "teams": ["RCB", "KKR"], "date_ist": date_ist,
     "scheduled_ist": "19:30", "status": "complete", "result": "RCB won by 4 wickets",
     "winner": "RCB"},
    # An already-played earlier match in the season, for "form" input
    {"id": "0999", "teams": ["CSK", "RR"], "date_ist": "2026-05-10",
     "scheduled_ist": "19:30", "status": "complete", "result": "CSK won by 22 runs",
     "winner": "CSK"},
    # Remaining matches for forward-sim
    {"id": "1003", "teams": ["DC", "PBKS"], "date_ist": "2026-05-12",
     "scheduled_ist": "19:30", "status": "scheduled", "result": None},
    {"id": "1004", "teams": ["SRH", "GT"], "date_ist": "2026-05-14",
     "scheduled_ist": "19:30", "status": "scheduled", "result": None},
]

STANDINGS = [
    {"team": "RCB", "played": 12, "won": 9, "lost": 3, "points": 18, "nrr": 0.81},
    {"team": "SRH", "played": 12, "won": 8, "lost": 4, "points": 16, "nrr": 0.65},
    {"team": "GT",  "played": 12, "won": 7, "lost": 5, "points": 14, "nrr": 0.41},
    {"team": "PBKS","played": 12, "won": 7, "lost": 5, "points": 14, "nrr": 0.10},
    {"team": "CSK", "played": 12, "won": 6, "lost": 6, "points": 12, "nrr": -0.05},
    {"team": "MI",  "played": 12, "won": 6, "lost": 6, "points": 12, "nrr": -0.10},
    {"team": "KKR", "played": 12, "won": 5, "lost": 7, "points": 10, "nrr": -0.25},
    {"team": "DC",  "played": 12, "won": 5, "lost": 7, "points": 10, "nrr": -0.31},
    {"team": "LSG", "played": 12, "won": 4, "lost": 8, "points":  8, "nrr": -0.55},
    {"team": "RR",  "played": 12, "won": 3, "lost": 9, "points":  6, "nrr": -0.78},
]

SQUADS = {
    "CSK": {"batters": [{"name": "R Gaikwad", "runs": 540}, {"name": "S Dube", "runs": 410}, {"name": "MS Dhoni", "runs": 220}],
            "bowlers": [{"name": "M Pathirana", "wickets": 18}, {"name": "R Jadeja", "wickets": 14}, {"name": "M Ali", "wickets": 9}]},
    "MI":  {"batters": [{"name": "R Sharma", "runs": 480}, {"name": "S Yadav", "runs": 460}, {"name": "T Mills", "runs": 90}],
            "bowlers": [{"name": "J Bumrah", "wickets": 20}, {"name": "G Coetzee", "wickets": 13}, {"name": "H Pandya", "wickets": 8}]},
    "RCB": {"batters": [{"name": "V Kohli", "runs": 620}, {"name": "F du Plessis", "runs": 380}, {"name": "G Maxwell", "runs": 290}],
            "bowlers": [{"name": "M Siraj", "wickets": 17}, {"name": "Y Chahal", "wickets": 15}, {"name": "R Topley", "wickets": 8}]},
    "KKR": {"batters": [{"name": "P Salt", "runs": 420}, {"name": "S Iyer", "runs": 380}, {"name": "A Russell", "runs": 310}],
            "bowlers": [{"name": "S Narine", "wickets": 16}, {"name": "V Chakravarthy", "wickets": 15}, {"name": "M Starc", "wickets": 12}]},
    "DC":  {"batters": [], "bowlers": []},
    "PBKS":{"batters": [], "bowlers": []},
    "SRH": {"batters": [], "bowlers": []},
    "GT":  {"batters": [], "bowlers": []},
    "LSG": {"batters": [], "bowlers": []},
    "RR":  {"batters": [], "bowlers": []},
}


# Monkey-patch the data fetchers
data_fetcher.fetch_fixtures = lambda force=False: FIXTURES
data_fetcher.fetch_standings = lambda force=False: STANDINGS
data_fetcher.fetch_squads = lambda force=False: SQUADS
data_fetcher.fetch_current_match = lambda mid: None

# Also patch in the tracker module's reference (it imports the symbols at top level)
tracker.data_fetcher.fetch_fixtures = data_fetcher.fetch_fixtures
tracker.data_fetcher.fetch_standings = data_fetcher.fetch_standings
tracker.data_fetcher.fetch_squads = data_fetcher.fetch_squads
tracker.data_fetcher.fetch_current_match = data_fetcher.fetch_current_match

# Disable real iMessage send during smoke
sent_messages: list[str] = []
def fake_send(body: str) -> bool:
    sent_messages.append(body)
    return True
imessage_sender.send = fake_send
tracker.imessage_sender.send = fake_send

# Disable git push during smoke (tracker.py uses subprocess; we no-op)
tracker._git_push_if_changes = lambda: True

# Reset state so the run is clean
STATE_PATH = REPO / "state.json"
INDEX_PATH = REPO / "docs" / "index.html"
STATE_PATH.write_text('{"last_run": null, "days": {}}\n')
if INDEX_PATH.exists():
    INDEX_PATH.unlink()

rc = tracker.main()
print(f"\n--- tracker.main() rc = {rc} ---\n")

# Verify state
st = json.loads(STATE_PATH.read_text())
assert TODAY in st["days"], f"today missing from state: {st}"
day = st["days"][TODAY]
msgs = {m["type"]: m for m in day["messages"]}
assert "morning" in msgs, f"no morning brief: {list(msgs)}"
assert any(t.startswith("post_match_") for t in msgs), f"no post_match: {list(msgs)}"
assert "end_of_day" in msgs, f"no end_of_day: {list(msgs)}"

# Newest message should be end_of_day, delivered=True, others skipped
eod = msgs["end_of_day"]
assert eod["delivered"], "end_of_day not delivered"
morning = msgs["morning"]
assert morning["delivery_skipped"], "older message should be skipped"

# Verify HTML
html = INDEX_PATH.read_text()
assert "IPL 2026" in html
assert f'data-day="{TODAY}"' in html, "today's <details> missing"
assert f'msg-{TODAY}-morning' in html, "morning article missing"
assert f'msg-{TODAY}-end_of_day' in html, "end_of_day article missing"

# Verify iMessage send was attempted (one — the newest)
assert len(sent_messages) == 1, f"expected exactly 1 send, got {len(sent_messages)}"
assert "Day recap" in sent_messages[0] or "Updated top 4" in sent_messages[0]

print("PASS — all assertions held")
print(f"  • {len(msgs)} messages in state.json")
print(f"  • {len(sent_messages)} iMessage send (the newest)")
print(f"  • HTML size: {INDEX_PATH.stat().st_size} bytes")
print()
print("--- newest message (what would ship to iPhone) ---")
print(sent_messages[0])
