# v6 build plan

Branch: `ml-engine-v6` · Started: 2026-05-12 · Goal: lift pre-match 2025 test accuracy from v5's 47% toward the academic 67–72% band, while keeping the v2 live WP model untouched.

## Rules carried forward from v1-v5

1. **Immutable versions.** v6_* models never overwrite v5. v7 ensemble never overwrites v5.
2. **Test set integrity.** 2025 test touched once per model version. Log every touch in `PROGRESS.md`.
3. **Point-in-time correctness.** Every feature at match N uses only data from matches before N.
4. **Calibration mandatory.** Every classifier wrapped in `CalibratedClassifierCV(method='isotonic')`. Report Brier alongside accuracy.
5. **Isolation.** All work under `ml/`. Existing tracker `src/` is read-only.
6. **Atomic writes.** `.tmp` → `os.replace`.

## Phase A — derived features from cricsheet (no scrapers)

Five parallel workstreams writing to `ml/data/historical/v6/`.

| ID | Output | Source | Notes |
|----|--------|--------|-------|
| A1 | `phase_player_stats.parquet` | balls.parquet | Per-player PP/middle/death over SR + bowler economy by phase. PIT cumulative. |
| A2 | `venue_patterns.parquet` | matches.parquet + balls.parquet | Per-venue avg 1st innings total, wicket-fall rate per phase, pace-vs-spin economy split |
| A3 | `recency_form.parquet` | matches.parquet | Exponential-decay form over last 10 matches per team (α=0.7), beats flat last-5 |
| A4 | `h2h_career.parquet` | matches.parquet | Career H2H + last-5-meetings outcomes per team pair |
| A5 | `toss_conditioned.parquet` | matches.parquet + balls.parquet | Venue chase rate conditioned on toss-winner decision (field vs bat) |

Each must include `match_id` as the join key and respect PIT (no data from match M or later).

## Phase B — external data

| ID | Output | Source | Notes |
|----|--------|--------|-------|
| B1 | `weather.parquet` | open-meteo.com archive API (free, no key) | Backfill weather for ~1225 historical matches by venue lat/lon + date. Capture temp, humidity, dew point at match start time. |
| B2 | `playing_xi.parquet` | balls.parquet (historical) | Extract actual XI per match (anyone who batted or bowled). At inference time, the XI is unknown until iplt20 announcement — for training, use actual. |
| B3 | `venue_geocodes.json` | static seed file | venue name → (lat, lon, city, IST timezone). One-time geocode from a fixed seed of ~50 IPL venues. |

## Phase C — train base models (5 parallel, then ensemble)

After Phase A completes:

| ID | Model | Features | Notes |
|----|-------|----------|-------|
| C.1.1 | `v6_phase_pre_match.pkl` | base + A1-A5 + B1 + B2 | LightGBM + isotonic calibration. TimeSeriesSplit CV. |
| C.1.2 | `v6_phase_post_toss.pkl` | C.1.1 features + toss winner + decision | |
| C.1.3 | `v6_phase_post_pp1.pkl` | C.1.2 features + PP1 runs + wickets | |
| C.1.4 | `v6_phase_innings_break.pkl` | C.1.3 features + first innings total + RRR | |
| C.1.5 | `v6_player.pkl` | C.1.1 features + richer player aggregates (phase-specific via A1) | |

Then C.2: train `v7_ensemble.pkl` as a LightGBM meta-learner stacking C.1.1 + C.1.5 predictions (plus optionally v3_pre_match as a stable reference). LightGBM meta is the v7 architecture upgrade per the roadmap.

## Phase D — operational

| ID | Output | Notes |
|----|--------|-------|
| D1 | `launchd/com.kcln.ipl-cricsheet-refresh.plist` + install script | Daily cricsheet refresh at 23:30 IST. Won't touch `src/tracker.py`. |
| D2 | `ml/src/v6/drift_monitor.py` | Weekly accuracy check on the rolling 2026 season; alert via stdout if it drops 5%+ from val baseline. |
| D3 | `ml/src/v6/venue_sub_models.py` | Per-venue sub-model experiment (high-scoring vs low-scoring venues). If gain > 2%, commit. |
| D4 | `ml/src/v6/retrain.py` | One-shot retraining script that produces v6_*+1, v7+1 from the current parquet snapshot. |

## Output paths

```
ml/data/historical/v6/         # new feature parquets (A1-A5, B1-B2)
ml/data/historical/v6/raw/     # raw weather API responses cached (gitignored)
ml/data/models/v6_*.pkl        # base models + metadata json sidecars
ml/data/models/v7_ensemble.pkl # meta-learner
launchd/com.kcln.ipl-cricsheet-refresh.plist
ml/src/v6/                     # all new module code
ml/tests/v6/                   # all new tests
ml/docs/v6_model.html          # new model card (won't overwrite v5's)
```

## Verification

Before declaring v6 done:

- All 33 existing `ml/tests/` tests still pass
- New v6 tests pass: feature PIT, weather schema, lineup extraction, training reproducibility
- `git diff main..ml-engine-v6 --name-only | grep -v '^ml/' | grep -v '^launchd/' | grep -v '^\.gitignore$'` returns empty (isolation)
- Per-model test-set integrity log entries added to PROGRESS.md
- v7 ensemble test accuracy reported with calibrated probabilities

## Token budget reminder

Multi-agent dispatch costs approximately 15x baseline (BrowseComp finding via context-engineering:multi-agent-patterns). Set explicit budgets per workstream. Use filesystem coordination over message-passing per `multi-agent-patterns` skill.
