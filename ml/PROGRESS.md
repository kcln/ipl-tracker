# ML Engine Progress

Branch: `ml-engine` · Started: 2026-05-12

## Phase checklist

- [x] Phase -1: Infrastructure setup
- [x] Phase 0: Data foundation + backtest harness + heuristic baseline
- [x] Phase 1: v1 logistic regression with calibration (KILLED — see below)
- [x] Phase 2: Live win-probability model (71.6% test, in target band)
- [x] Phase 3: Per-phase GBM models
- [x] Phase 4: Player-level features (added signal but did not improve val accuracy)
- [x] Phase 5: Ensemble + documentation

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
| v2 | wp_lightgbm + isotonic | 2026-05-12 | 2008–2023 | 2025 | 71.6% | 0.190 | per-delivery WP, 12 numeric + venue, n_train=239,693 |
| v3 | phase_pre_match (LightGBM+iso) | 2026-05-12 | 2008–2023 | 2025 | 47.1% | 0.266 | 14 PIT features; val 57.7% (beats v1 by +11.2%, beats heuristic val by +1.4%, just under 3% kill threshold) |
| v3 | phase_post_toss (LightGBM+iso) | 2026-05-12 | 2008–2023 | 2025 | 41.4% | 0.269 | 11 features incl. toss; toss adds no signal here |
| v3 | phase_post_pp1 (LightGBM+iso) | 2026-05-12 | 2008–2023 | 2025 | 67.1% | 0.208 | 13 features incl. PP1 runs/wkts; **in target band** |
| v3 | phase_innings_break (LightGBM+iso) | 2026-05-12 | 2008–2023 | 2025 | 64.3% | 0.205 | 16 features incl. first innings total + RRR |
| v4 | player_gbm (LightGBM+iso) | 2026-05-12 | 2008–2023 | 2025 | 40.0% | 0.293 | 14 base + 12 player aggregates. Player features dominate gain rankings but val acc dropped 2.8% vs v3_pre_match — overfits on n_train=1005. |
| v5 | ensemble_stacked_logistic | 2026-05-12 | meta on 2024 | 2025 | 47.1% | 0.255 | meta over (v1, v3_pre_match, v4); meta weights v1≈0, v3=0.54, v4=0.69; **best test Brier** but ties v3 on accuracy |
| v6 | phase_pre_match (LightGBM+iso) | 2026-05-11 | 2008–2023 | 2025 | 42.9% | 0.288 | 46 feats (v3 base + A1-A5 + B1 weather). val 56.3% = heuristic baseline, **KILL** (< 58.3%). Top gain: team1_top4_pp_econ, wind_kmh, venue_wickets_per_death, venue_spin_econ. New features rank highly but don't generalise to 2025. |
| v6 | phase_post_toss (LightGBM+iso) | 2026-05-11 | 2008–2023 | 2025 | 44.3% | 0.280 | 48 feats. val 57.7%, **KILL** (< 58.3%). |
| v6 | phase_post_pp1 (LightGBM+iso) | 2026-05-11 | 2008–2023 | 2025 | 57.1% | 0.324 | 50 feats. val 59.2%, passed kill gate. Test acc drops vs v3 (67.1%) — added features hurt 2025 generalisation. |
| v6 | phase_innings_break (LightGBM+iso) | 2026-05-11 | 2008–2023 | 2025 | 72.9% | 0.195 | 53 feats. val 73.2%, test 72.9% — **new high**, beats v3 innings_break (64.3%) by +8.6%; in target band. |
| v6 | player (LightGBM+iso) | 2026-05-11 | 2008–2023 | 2025 | 47.1% | 0.276 | 38 feats (v3 base + A1 + v4 player aggregates). val 56.3%, **KILL**. Player features still high gain, low generalisation. |
| v7 | ensemble_stacked_lightgbm | 2026-05-11 | meta on 2024 | 2025 | 47.1% | 0.265 | LightGBM meta over surviving v6 base + v3_pre_match. v6_pre_match and v6_player skipped (kill). Only v3_pre_match passed -> single-input meta -> ties v5 on accuracy, slightly worse Brier (0.265 vs 0.255). |

## Test-set integrity log

Each row records the single time 2025 was touched for a given model version.

| Version | Touched at (UTC) | Acc | Brier |
|---|---|---|---|
| v1_logistic | 2026-05-12 | 41.4% | 0.261 |
| v2_wp_lightgbm | 2026-05-12 | 71.6% | 0.190 |
| v3_phase_pre_match | 2026-05-12 | 47.1% | 0.266 |
| v3_phase_post_toss | 2026-05-12 | 41.4% | 0.269 |
| v3_phase_post_pp1 | 2026-05-12 | 67.1% | 0.208 |
| v3_phase_innings_break | 2026-05-12 | 64.3% | 0.205 |
| v4_player_gbm | 2026-05-12 | 40.0% | 0.293 |
| v5_ensemble | 2026-05-12 | 47.1% | 0.255 |
| v6_phase_pre_match | 2026-05-11 | 42.9% | 0.288 |
| v6_phase_post_toss | 2026-05-11 | 44.3% | 0.280 |
| v6_phase_post_pp1 | 2026-05-11 | 57.1% | 0.324 |
| v6_phase_innings_break | 2026-05-11 | 72.9% | 0.195 |
| v6_player | 2026-05-11 | 47.1% | 0.276 |
| v7_ensemble | 2026-05-11 | 47.1% | 0.265 |

## Kill-criteria log

Phases that hit kill criteria and what was done.

| Phase | Issue | Action |
|---|---|---|
| 0 | 2025 heuristic accuracy 54.3% < 60% floor | Documented, continued. |
| 1 | v1 val accuracy 46.5% < heuristic + 2% (58.3%) | Documented, did not integrate, continued. |
| 4 | v4 val 54.9% < v3_pre_match 57.7%; test 40.0% | Documented; player features add high-gain but high-variance signal. Likely useful in ensemble. |
| C (v6_phase_pre_match) | val 56.3% < heuristic + 2% (58.3%) | Saved artifact (immutable), excluded from v7 stack. New v6 features (A1-A5, B1 weather) rank top by gain but do not generalise — overfit on 1005 train rows. |
| C (v6_phase_post_toss) | val 57.7% < 58.3% | Saved, excluded from v7. |
| C (v6_player) | val 56.3% < 58.3% | Saved, excluded from v7. Same overfit pattern as v4. |
| C (v7_ensemble) | Only v3_pre_match survived kill gate -> degenerate single-input meta | v7 test 47.1% ties v5 baseline (no improvement). v6 enrichment did not lift pre-match. v6 innings_break is the real win (72.9% test, +8.6% over v3). |

### Phase 1 post-mortem
- v1 logistic test accuracy 41.4% — worse than always-team2 baseline (52.9%).
- Predicted probability std on test = 0.035; features are nearly uninformative on cricsheet-only inputs.
- Likely causes:
  1. No feature scaling — `qualified_flag_diff` is integer counts, others are in [-1, 1]; the optimiser hit overflow warnings on early CV folds.
  2. `venue_chase_rate` returns 0 always (placeholder; cricsheet matches.parquet has `win_by_wickets` but my state only stored winners).
  3. The single most predictive heuristic signals (squad ranks, home_team) are unavailable from cricsheet alone.
- For Phase 3 (per-phase GBMs) I'll add `StandardScaler` and richer features that read `win_by_wickets` from matches.parquet directly.
