---
name: backtest
description: Chronological replay backtest of a predict function across seasons with point-in-time enforcement. Use when evaluating a model on historical matches.
---

# backtest

Chronological replay of a `predict_fn(match_row, state)` across the historical matches parquet, with leakage tests.

## Function signature
```
run_backtest(predict_fn, matches_df, name) -> dict
```
returns:
```
{
  "name": str,
  "overall": {"acc": float, "brier": float, "log_loss": float, "n": int},
  "by_season": {"2008": {...}, ..., "2025": {...}},
  "details_path": "ml/data/backtest_results/<name>_<ts>.parquet"
}
```

## Steps
1. Sort matches_df ascending by date.
2. Initialise `state = {"completed": [], "standings_by_season": {}, ...}` — empty.
3. For each match row:
   a. Build feature vector from state-as-of-now via `features.build_features(match_id, state)`.
   b. `prob, pred = predict_fn(features, match_row)`.
   c. Compare to ground truth; record `(match_id, date, season, predicted, actual, prob, correct)`.
   d. Update state with this match's outcome.
4. Run leakage checks every K matches: assert no feature references any match_id > current.
5. Compute per-season accuracy, brier, log loss. Halt with explicit error on detected leakage.
6. Save details parquet to `ml/data/backtest_results/<name>_<UTC-ISO>.parquet`.
7. Return summary only (not the details).

## Subagent
Use the `backtest-runner` subagent when results would be verbose; it returns ONLY the summary dict.
