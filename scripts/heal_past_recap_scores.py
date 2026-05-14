"""One-shot: backfill final innings scores onto every completed match in
state.json (from the fixtures feed, which carries inn1/inn2 for past matches)
and re-render all `post_match_*` and `end_of_day` messages so each carries
the new `TEAM R/W (O)  ·  TEAM R/W (O)` score line. Idempotent.

Run: ./venv/bin/python3 scripts/heal_past_recap_scores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import data_fetcher, html_archive, message_builder, predictor, state, tracker  # noqa: E402


def main() -> int:
    fixtures = data_fetcher.fetch_fixtures(force=True)
    by_id = {str(f.get("id")): f for f in fixtures}

    state_obj = state.load()
    days = state_obj.get("days", {})
    if not days:
        print("no days in state.json; nothing to heal")
        return 0

    standings = data_fetcher.fetch_standings(force=False) or []
    squads = data_fetcher.fetch_squads(force=False) or {}
    recent = tracker._completed_match_lookup(fixtures)
    remaining = [m for m in fixtures if m.get("status") != "complete"]

    backfilled = 0
    rewrites = 0

    for date_iso in sorted(days.keys()):
        day_entry = days[date_iso]
        # Pass 1: backfill innings data onto day_entry["matches"]
        for m in day_entry.get("matches", []):
            mid = str(m.get("id") or "")
            src = by_id.get(mid)
            if not src or src.get("status") != "complete":
                continue
            changed = False
            for key in ("inn1", "inn2", "first_batting", "second_batting"):
                if not m.get(key) and src.get(key):
                    m[key] = src[key]
                    changed = True
            if changed:
                backfilled += 1
                print(f"  backfilled {date_iso} / {m.get('teams')} (id={mid})")

        # Pass 2: rewrite post_match_* and end_of_day bodies
        completed_matches = day_entry.get("matches") or []
        for msg in day_entry.get("messages", []):
            t = msg.get("type", "")
            if not (t.startswith("post_match_") or t == "end_of_day"):
                continue
            new_body = _rebuild_body(
                msg, day_entry, date_iso, completed_matches,
                state_obj, standings, remaining, recent, squads,
            )
            if new_body and new_body != msg.get("body"):
                msg["body"] = new_body
                html_archive.upsert_message(date_iso, t, msg["generated_at"], new_body)
                rewrites += 1
                print(f"  rewrote {date_iso} / {t}")

    state.save(state_obj)
    print(f"\nDone. backfilled={backfilled}, rewrites={rewrites}")
    return 0


def _rebuild_body(
    msg: dict, day_entry: dict, date_iso: str,
    completed_matches: list[dict],
    state_obj: dict, standings: list[dict],
    remaining: list[dict], recent: list[dict], squads: dict,
) -> str | None:
    t = msg["type"]
    if t.startswith("post_match_"):
        # post_match_<idx> — find the corresponding match by index
        try:
            idx = int(t.split("_")[-1]) - 1
        except ValueError:
            return None
        if idx < 0 or idx >= len(completed_matches):
            return None
        match = dict(completed_matches[idx])
        # Backwards-compat: tracker.message_builder.post_match_message uses
        # 'winner' and 'actual_winner'. day_entry stores 'actual_winner'.
        if not match.get("winner"):
            match["winner"] = match.get("actual_winner")
        # Re-grade prediction using stored predicted_winner
        return message_builder.post_match_message(
            match, standings, remaining, recent, squads,
        )
    if t == "end_of_day":
        season_correct, season_total = state.season_prediction_record(state_obj)
        # Promote winner field for the message builder
        enriched = []
        for m in completed_matches:
            mm = dict(m)
            if not mm.get("winner"):
                mm["winner"] = mm.get("actual_winner")
            enriched.append(mm)
        return message_builder.end_of_day_message(
            date_iso, enriched, standings, remaining, recent, squads,
            tracker.ARCHIVE_URL,
            season_correct=season_correct, season_total=season_total,
        )
    return None


if __name__ == "__main__":
    sys.exit(main())
