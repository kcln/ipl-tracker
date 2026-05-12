# Model history

Append one row per model version, newest at bottom.

| Version | Name | Date (UTC) | Train seasons | Val acc | Test acc | Brier | Notes |
|---|---|---|---|---|---|---|---|
| v1 | logistic + isotonic | 2026-05-12 | 2008–2023 | 46.5% | 41.4% | 0.261 | KILLED — features uninformative; see PROGRESS.md post-mortem. |
| v2 | wp_lightgbm + isotonic (prefit) | 2026-05-12 | 2008–2023 | 74.1% | 71.6% | 0.190 | Per-delivery win probability; 12 numeric + venue features; n_train=239,693 deliveries. |
