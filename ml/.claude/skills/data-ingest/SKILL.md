---
name: data-ingest
description: Download cricsheet IPL JSON, normalise into parquets with team-rename mapping. Use once per refresh of historical data.
---

# data-ingest

Ingest cricsheet IPL JSON dump into normalised parquets.

## Steps
1. `requests.get("https://cricsheet.org/downloads/ipl_json.zip")` → `ml/data/historical/raw_json/ipl_json.zip`.
2. Extract per-match JSONs into `ml/data/historical/raw_json/`.
3. For each match JSON, produce:
   - `matches` row: match_id, season, date, venue, team1, team2, toss_winner, toss_decision, winner, win_by_runs, win_by_wickets, player_of_match.
   - `balls` rows: match_id, innings, over, ball, batter, bowler, runs_off_bat, extras, wicket_kind, player_out, fielders (list).
4. Apply team-rename mapping:
   - "Kings XI Punjab" → "Punjab Kings"
   - "Delhi Daredevils" → "Delhi Capitals"
   - "Royal Challengers Bangalore" → "Royal Challengers Bengaluru" (cricsheet uses Bangalore historically)
   - Defunct teams (Pune Warriors, Kochi Tuskers, Rising Pune Supergiant, Gujarat Lions, Deccan Chargers) stay as-is.
5. Build `players` rollups parquet from balls.parquet.
6. Write parquets atomically (tmp → replace) using pyarrow.

## Tests
- Total match count > 1100.
- Distinct teams in matches ≤ 16 (incl. defunct).
- Null winners only for matches where outcome.result == "no result" or "tie".
