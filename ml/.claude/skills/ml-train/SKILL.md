---
name: ml-train
description: Train and calibrate a versioned classifier with TimeSeriesSplit CV, isotonic calibration, and full metadata artifacts. Use when training any new model version under ml/.
---

# ml-train

Train a versioned, calibrated classifier and persist artifacts.

## Inputs
- features parquet (must have `split` column with values `train|val|test`)
- target column name (default `winner_is_team1`)
- model type (`logistic`, `lightgbm`)
- hyperparameter grid

## Steps
1. Load features parquet. Split into train (≤2023), val (2024), test (2025).
2. For each hyperparameter setting, run TimeSeriesSplit(n_splits=5) CV on train+val ordered chronologically. Pick the best by mean validation log loss.
3. Refit best model on train. Wrap in `CalibratedClassifierCV(method='isotonic', cv=5)` using train only.
4. Score on val (2024) — record acc, brier, log_loss.
5. Score on test (2025) ONCE. Append to `PROGRESS.md` test-set integrity log.
6. Save model to `ml/data/models/v{N}_{name}.pkl` (atomic via `.tmp` then `os.replace`).
7. Save metadata sidecar `v{N}_{name}.json`:
   - version, model_type, created_at (UTC), git_sha, sklearn_version, lightgbm_version (if used)
   - training_seasons, validation_season, test_season
   - validation_accuracy, validation_brier, test_accuracy, test_brier, log_loss
   - feature_names, feature_importance_ranked
   - hyperparameters
8. Regenerate `ml/docs/calibration_v{N}.png` reliability diagram.

## Rules
- Never overwrite an existing `v{N}` file. If `v1` exists, the next is `v2`.
- 2025 is touched exactly once per version. Subsequent re-touches must be rejected.
- Kill if validation accuracy < heuristic baseline + 2%. Document and skip integration.
