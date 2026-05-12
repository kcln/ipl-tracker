# ML Engine Progress

Branch: `ml-engine` · Started: 2026-05-12

## Phase checklist

- [x] Phase -1: Infrastructure setup
- [x] Phase 0: Data foundation + backtest harness + heuristic baseline
- [x] Phase 1: v1 logistic regression with calibration (KILLED — see below)
- [ ] Phase 2: Live win-probability model
- [ ] Phase 3: Per-phase GBM models
- [ ] Phase 4: Player-level features
- [ ] Phase 5: Ensemble + documentation

## Dataset

- Cricsheet IPL JSON (downloaded 2026-05-12)
- 1,224 match JSONs covering 2008–2026 seasons
- After filter to 2008–2025 with non-null winner: 1,146 feature rows
- Splits: train ≤2023 (1,005), val 2024 (71), test 2025 (70)
- 2026 (current season, 55 matches) excluded from features.

## Baselines

| Model | 2024 val acc | 2025 test acc | Brier | Log loss | Notes |
|---|---|---|---|---|---|
| Heuristic (wrapper) | 56.3% | 54.3% | 0.288 | 0.801 | overall on 1146 matches: 52.3% |

### Kill criteria for Phase 0
2025 test accuracy is 54.3% — below the 60% floor. Per plan rules ("If <60%, document concerns, continue anyway"), proceeding to Phase 1.

Concerns to keep in mind:
- The heuristic wrapper has degraded inputs vs the live tracker (no squad ranks, no home_team, no second_batting), so this is a lower bound on heuristic capability with cricsheet-only inputs. ML models also use cricsheet-only inputs, so the comparison is apples-to-apples.
- Year-to-year variance is large (41% in 2023, 66% in 2014). The 2025 sample is small (n=70).

## Model version log

| Version | Name | Date | Train seasons | Test season | Acc | Brier | Notes |
|---|---|---|---|---|---|---|---|
| v1 | logistic + isotonic | 2026-05-12 | 2008–2023 | 2025 | 41.4% | 0.261 | 9 features, C=0.1. KILLED. |

## Test-set integrity log

Each row records the single time 2025 was touched for a given model version.

| Version | Touched at (UTC) | Acc | Brier |
|---|---|---|---|
| v1_logistic | 2026-05-12 | 41.4% | 0.261 |

## Kill-criteria log

Phases that hit kill criteria and what was done.

| Phase | Issue | Action |
|---|---|---|
| 0 | 2025 heuristic accuracy 54.3% < 60% floor | Documented, continued. |
| 1 | v1 val accuracy 46.5% < heuristic + 2% (58.3%) | Documented, did not integrate, continued. |

### Phase 1 post-mortem
- v1 logistic test accuracy 41.4% — worse than always-team2 baseline (52.9%).
- Predicted probability std on test = 0.035; features are nearly uninformative on cricsheet-only inputs.
- Likely causes:
  1. No feature scaling — `qualified_flag_diff` is integer counts, others are in [-1, 1]; the optimiser hit overflow warnings on early CV folds.
  2. `venue_chase_rate` returns 0 always (placeholder; cricsheet matches.parquet has `win_by_wickets` but my state only stored winners).
  3. The single most predictive heuristic signals (squad ranks, home_team) are unavailable from cricsheet alone.
- For Phase 3 (per-phase GBMs) I'll add `StandardScaler` and richer features that read `win_by_wickets` from matches.parquet directly.
