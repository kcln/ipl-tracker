# Model history

Append one row per model version, newest at bottom.

| Version | Name | Date (UTC) | Train seasons | Val acc | Test acc | Brier | Notes |
|---|---|---|---|---|---|---|---|
| v1 | logistic + isotonic | 2026-05-12 | 2008–2023 | 46.5% | 41.4% | 0.261 | KILLED — features uninformative; see PROGRESS.md post-mortem. |
| v2 | wp_lightgbm + isotonic (prefit) | 2026-05-12 | 2008–2023 | 74.1% | 71.6% | 0.190 | Per-delivery win probability; 12 numeric + venue features; n_train=239,693 deliveries. |
| v3 | phase_pre_match (LightGBM+iso) | 2026-05-12 | 2008–2023 | 57.7% | 47.1% | 0.266 | 14 PIT features. Val beats v1 by 11.2% and heuristic by 1.4%; test below heuristic — 2025 is noisy (n=70). |
| v3 | phase_post_toss (LightGBM+iso) | 2026-05-12 | 2008–2023 | 56.3% | 41.4% | 0.269 | Adding toss decision didn't help; documented. |
| v3 | phase_post_pp1 (LightGBM+iso) | 2026-05-12 | 2008–2023 | 64.8% | 67.1% | 0.208 | Powerplay-1 info lifts to target band. |
| v3 | phase_innings_break (LightGBM+iso) | 2026-05-12 | 2008–2023 | 71.8% | 64.3% | 0.205 | First-innings total + RRR is the strongest in-match pre-2nd-innings signal. |
