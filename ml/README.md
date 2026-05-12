# ml/ — IPL match prediction engine

**Status:** parallel infrastructure on the `ml-engine` branch. The live tracker
(root `src/`, `scripts/`, `docs/index.html`, `launchd/`) is unchanged.

## What this is

Five model versions trained from Cricsheet IPL data (2008–2025), kept under
`ml/data/models/` with metadata sidecars and immutable version numbers.

| Version | Model | Test 2025 acc | Brier | Notes |
|---|---|---|---|---|
| v1 | logistic + isotonic | 41.4% | 0.261 | killed; weak feature set |
| v2 | LightGBM live WP per-delivery | 71.6% | 0.190 | **in target band** |
| v3_phase_pre_match | LightGBM + isotonic | 47.1% | 0.266 | beats v1 by +11% on val |
| v3_phase_post_toss | LightGBM + isotonic | 41.4% | 0.269 | toss alone adds no signal |
| v3_phase_post_pp1 | LightGBM + isotonic | 67.1% | 0.208 | **in target band** |
| v3_phase_innings_break | LightGBM + isotonic | 64.3% | 0.205 | strongest pre-2nd-innings signal |
| v4 | player-features LightGBM | 40.0% | 0.293 | overfits small train; kept for ensemble |
| v5 | stacked logistic over v1+v3+v4 | 47.1% | 0.255 | best Brier on test |

Baselines on 2025: coin flip 50%, always-team2 52.9%, heuristic (cricsheet
inputs) 54.3%, academic baseline 67–72%.

## Layout

```
ml/
├── CLAUDE.md              isolation + correctness rules
├── PROGRESS.md            phase log, version log, test-set integrity log
├── requirements.txt       pinned
├── README.md              this file
├── .venv/                 isolated env (gitignored)
├── .claude/               skills + agents for this project only
├── data/
│   ├── historical/        cricsheet parquets (gitignored except .gitkeep)
│   ├── models/            committed model artifacts (.pkl + .json)
│   └── backtest_results/  per-run details (gitignored)
├── src/
│   ├── ingest.py           cricsheet download + parse
│   ├── features.py         9 pre-match PIT features (used by v1)
│   ├── phase_features.py   14+ pre-match PIT features (used by v3, v4, v5)
│   ├── player_features.py  team-level player career aggregates
│   ├── wp_features.py      per-delivery features
│   ├── backtest.py         chronological replay harness
│   ├── train.py            v1 logistic
│   ├── train_wp.py         v2 LightGBM live WP
│   ├── train_phase.py      v3 stage models
│   ├── train_player.py     v4 player-features
│   ├── train_ensemble.py   v5 stacked logistic meta
│   ├── topfour_sim.py      standalone 5000-sim top-4 simulator
│   ├── predict.py          ml-namespace predict surface (routes by stage)
│   ├── heuristic_wrapper.py read-only adapter for root src/predictor.py
│   └── calibrate.py        reliability diagram
├── tests/                  18 tests (ingest, features, leakage, backtest)
└── docs/
    ├── model.html          model card (open in a browser)
    ├── model_history.md    one row per version
    ├── calibration_v1.png  v1 reliability diagram
    ├── calibration_v2.png  v2 reliability diagram
    └── wp_demo_data.json   5 dramatic IPL 2024/25 matches for the WP chart
```

## How to test locally

```bash
# from repo root
ml/.venv/bin/python -m pytest ml/ -v
```

If `ml/data/historical/*.parquet` aren't built yet, the data-dependent tests
will skip. Build them:

```bash
ml/.venv/bin/python -m ml.src.ingest
ml/.venv/bin/python -m ml.src.wp_features
ml/.venv/bin/python -m ml.src.phase_features
ml/.venv/bin/python -m ml.src.player_features
```

## How to inspect a model

Open `ml/docs/model.html` in a browser. All metrics, reliability diagrams, and
the WP demo chart for 5 dramatic matches are inline.

Each model artifact has a metadata JSON sidecar with full provenance:

```bash
cat ml/data/models/v2_wp_lightgbm.json | jq
```

## Cutover plan (manual, by KCL)

Nothing in this repo is wired into the live tracker yet. When you want to
integrate, here's a safe path. None of these steps should be taken without
human review.

1. **Smoke-test predictions side-by-side.** Wrap `ml.src.predict.predict_pre_match`
   in a thin adapter under `src/` and run it in parallel with the existing
   heuristic on the next 10 fixtures. Log both predictions; do not change
   downstream behaviour. Decide if the ML prediction is worth promoting.

2. **Promote the WP model first.** v2 is the most validated (71.6% on 2025
   per-delivery). The live tracker can call `predict_live()` in addition to,
   not in place of, the heuristic's `predict_after_powerplay()` / `predict_chase()`.
   Display both for a week.

3. **Top-4 simulator parallel run.** Wire `ml.src.topfour_sim.simulate_top4`
   into a separate output cell — keep `src/predictor.predict_final_top4`
   unchanged. Compare top-4 distributions weekly.

4. **Pre-match ensemble cutover.** Only if v5 (or a future v6) clearly beats
   the heuristic on a fresh held-out season. 2025 results suggest the pre-match
   ensemble is competitive but not clearly better.

5. **Do NOT modify `launchd/`, `requirements.txt`, or `recipients.sh`** as part
   of cutover. Add a new `requirements-ml.txt` to root if you need to install
   ML deps into the live `venv/`; pin them identically to `ml/requirements.txt`.

## Verifying nothing in the existing tracker was modified

```bash
# from repo root, with main fetched
git fetch origin main
git diff origin/main..ml-engine --name-only | grep -v '^ml/' | grep -v '^\.gitignore$'
# Expected output: nothing.
```

If anything beyond `ml/` and `.gitignore` shows up, that file was changed by
mistake — revert it.

## What was learned

- Cricsheet has 1,224 IPL JSONs across 2008–2026 (~290k deliveries).
- The 9-feature heuristic-style signal set is not informative enough on
  cricsheet-only inputs (no squad ranks, no home_team) — v1 fell apart.
- The richer 14-feature pre-match set beats v1 by +11.2% on val but loses on
  the small (n=70) 2025 test slice.
- Per-delivery WP (v2) is the strongest model overall and the safest to
  deploy first.
- Player-level features dominate gain rankings but overfit with only 1,005
  training matches — they add value as part of an ensemble, not standalone.
- 2025 is a noisy test slice; 2024 val numbers track the academic baselines
  more closely.

See `PROGRESS.md` for full phase logs and kill-criteria entries.
