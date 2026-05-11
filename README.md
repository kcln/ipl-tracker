# IPL 2026 daily tracker

A macOS-native cron job that watches the 2026 Indian Premier League, predicts each day's matches, and pings me on iMessage. Archives every brief and result at <https://kcln.github.io/ipl-tracker/>.

## What it does

Runs every 15 minutes via launchd. On each run:

1. Fetches today's fixtures, standings, and squad stats — tries four tiers in order:
   - **iplt20.com official feed** (primary; S3-backed, public, no key) — this is the source iplt20.com itself uses, hosted on `ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com`. Full 74-match schedule, points table with last-5 form, top run scorers, most wickets.
   - **ESPN Cricinfo JSON API** (fallback, no key needed — often 403-blocked by Akamai)
   - **CricAPI** (set `CRICAPI_KEY` env var; free tier ≈100 calls/day from <https://cricapi.com/>)
   - **Cricbuzz HTML scrape** (floor; ~4 matches visible, no standings)
2. Generates whichever of these are due but missing for today (PT):
   - **morning brief** — once after 00:00 PT, lists today's matches with predictions
   - **post-match** — once each match completes
   - **end-of-day recap** — once the last match finishes
3. Logs every message to `docs/index.html` (published via GitHub Pages).
4. Sends only the newest undelivered message via iMessage; older ones are marked skipped.
5. Commits and pushes the diff to `kcln/ipl-tracker`.

After **May 31, 2026** (final day), it posts a season recap and unloads its own launchd job.

## Setup

```bash
git clone https://github.com/kcln/ipl-tracker.git
cd ipl-tracker
./install.sh
```

The installer will:

- Create `venv/` and install dependencies.
- Prompt you for `IMESSAGE_RECIPIENT` (your phone in `+14155551234` form, or your Apple ID email) and append it to `~/.zshrc`.
- Render `launchd/com.kcln.ipltracker.plist` with absolute paths and load it into `~/Library/LaunchAgents/`.

Then manually:

1. **Grant Automation permission.** System Settings → Privacy & Security → Automation. Enable `osascript → Messages` (and `Terminal → Messages` if you'll run manually). Without this, iMessage sends silently fail.
2. **Create the GitHub repo and push:**
   ```bash
   gh repo create kcln/ipl-tracker --public --source=. --remote=origin --push
   ```
3. **Enable GitHub Pages.** Repo settings → Pages → source = `main` branch, folder = `/docs`. The included workflow then redeploys on every push to `docs/`.

## Operations

```bash
# Tail the logs
tail -f ~/Library/Logs/ipl-tracker.log

# Trigger a run manually
./venv/bin/python3 src/tracker.py

# Stop the scheduled job
launchctl unload ~/Library/LaunchAgents/com.kcln.ipltracker.plist

# Restart it
launchctl load ~/Library/LaunchAgents/com.kcln.ipltracker.plist

# Force-refresh a particular cache (delete then run)
rm data/_cache/fixtures.json
./venv/bin/python3 src/tracker.py
```

## File layout

```
src/tracker.py           # orchestrator (called every 15 min)
src/data_fetcher.py      # ESPN primary, Cricbuzz fallback, file cache
src/predictor.py         # form + NRR + squad-form weighted prediction
src/message_builder.py   # morning / post-match / end-of-day text
src/imessage_sender.py   # osascript wrapper for Messages.app
src/html_archive.py      # idempotent BS4-based docs/index.html updates
src/state.py             # atomic JSON state.json read/write

data/teams.json          # team metadata + aliases (committed)
data/fixtures.json       # last successful fixtures pull (committed)
data/squads.json         # last successful squad-stats pull (committed)
data/_cache/             # short-TTL HTTP caches (gitignored)

state.json               # source of truth for what's been generated/sent
docs/index.html          # archive page served by GitHub Pages
docs/style.css           # ported from kcl-brand
launchd/com.kcln.ipltracker.plist
install.sh
.github/workflows/deploy.yml
```

## Prediction logic

For any matchup A vs B we compute a score for each team:

```
score = 0.40 · form(last 5)
      + 0.30 · NRR (clipped to [-2, +2] then mapped to [0,1])
      + 0.30 · squad_form (top-3 batters' runs + top-3 bowlers' wickets · 20,
                           normalized to league max)
```

Higher score wins. For the final top-4 prediction we forward-simulate every remaining match the same way and rank by (points, NRR).

This is deliberately simple — no ML, no historical training data, no external feature pipeline. The model is in one file (`src/predictor.py`) and any reader can sanity-check or tune the weights inline.

## Troubleshooting

**ESPN returns 403.** Expected. The iplt20.com official feed is the primary source and serves IPL data from a public S3 bucket — no key, no auth, no rate limit. ESPN/CricAPI/Cricbuzz remain configured as fallbacks. If you want them active:

- **CricAPI:** sign up at <https://cricapi.com/>, copy your API key, then re-run the installer with `CRICAPI_KEY="your-key" bash install.sh`. The plist will be re-rendered with the key.
- **All sources failing** is logged and exits cleanly — no broken messages are sent.

**iMessage send fails silently.** Most often missing Automation permission. After granting it, run `./venv/bin/python3 src/imessage_sender.py` — it sends a single self-test line. You can also confirm `IMESSAGE_RECIPIENT` is set: `launchctl getenv IMESSAGE_RECIPIENT` (set via the plist) or `echo $IMESSAGE_RECIPIENT` in a new shell.

**git push fails.** Make sure `gh auth status` is clean and that `kcln/ipl-tracker` has a remote configured. The tracker logs the failure and retries on its next run; nothing is lost.

**No matches today.** Expected — the tracker exits with 0 and does nothing. The launchd job will check again 15 minutes later.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success, or nothing to do |
| 1 | Fatal failure |
| 2 | Partial success (e.g. HTML written but iMessage or git push failed) |
