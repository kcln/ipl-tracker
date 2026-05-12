# ml/ — ML engine for IPL 2026 tracker

Isolated ML experiment. Lives entirely under `ml/`. The existing heuristic tracker (root `src/`, `scripts/`, `docs/index.html`, `launchd/`, root `requirements.txt`) is never modified by code in this directory.

## Hard rules

1. **2025 IPL season is locked test set.** Touch it once per model version. Never iterate on 2025 results. If you read 2025 predictions back into feature design, document the leakage and treat the run as invalid.
2. **Point-in-time correctness.** Every feature at match N may only reference data from matches 0..N-1. No future data. `ml/tests/test_leakage.py` enforces this and must pass before any model is saved.
3. **Versioned models, never overwritten.** `v1` stays `v1` forever. New work creates `v2`, `v3`. Each model artifact is `ml/data/models/v{N}_<name>.pkl` plus a sibling `.json` with full metadata (git sha, sklearn version, training seasons, metrics, feature names).
4. **Calibration mandatory.** Every classifier is wrapped in `CalibratedClassifierCV(method='isotonic')` and reports Brier score alongside accuracy.
5. **Pinned deps.** `ml/requirements.txt` only — fully version-pinned. The repo-root `requirements.txt` is not touched.
6. **Never modify files outside `ml/`.** Read-only access to root `src/predictor.py` is allowed for understanding the heuristic; wrappers live in `ml/src/heuristic_wrapper.py`.

## Conventions

- All times are ISO 8601 UTC.
- File writes are atomic: write to `path.tmp`, then `os.replace(path.tmp, path)`.
- Parquet files use pyarrow.
- All scripts can be run as modules: `python -m ml.src.train ...`.
- Use the `ml-train`, `backtest`, `data-ingest`, `model-card` skills for repeatable workflows.
- Use the `backtest-runner`, `feature-engineer`, `data-scientist` subagents for verbose work.

## Layout

```
ml/
├── CLAUDE.md              this file
├── PROGRESS.md            phase checklist + baselines + version log
├── requirements.txt       pinned
├── README.md              cutover instructions (Phase 5)
├── .venv/                 isolated env (gitignored)
├── .claude/
│   ├── skills/
│   └── agents/
├── data/
│   ├── historical/        cricsheet parquets (gitignored except .gitkeep)
│   ├── models/            committed artifacts
│   └── backtest_results/  per-run details (gitignored)
├── src/
│   ├── ingest.py
│   ├── features.py
│   ├── backtest.py
│   ├── train.py
│   ├── calibrate.py
│   ├── predict.py
│   └── heuristic_wrapper.py
├── tests/
│   ├── test_leakage.py
│   ├── test_features.py
│   ├── test_backtest.py
│   └── test_ingest.py
└── docs/
    ├── model.html
    └── model_history.md
```

## Kill criteria

- 60% accuracy = floor. Below this, document and continue.
- 67–72% = target band.
- 75%+ = excellent.
- 85%+ = audit immediately, likely leakage.

If a phase fails its kill criterion, record in `PROGRESS.md` and move on. Never deploy a failing model.
