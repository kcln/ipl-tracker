# ML Engine Progress

Branch: `ml-engine` · Started: 2026-05-12

## Phase checklist

- [x] Phase -1: Infrastructure setup
- [x] Phase 0: Data foundation + backtest harness + heuristic baseline
- [ ] Phase 1: v1 logistic regression with calibration
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

## Test-set integrity log

Each row records the single time 2025 was touched for a given model version.

| Version | Touched at (UTC) | Acc | Brier |
|---|---|---|---|

## Kill-criteria log

Phases that hit kill criteria and what was done.

| Phase | Issue | Action |
|---|---|---|
