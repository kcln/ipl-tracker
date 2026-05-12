---
name: model-card
description: Publish the model card HTML at ml/docs/model.html with calibration diagram, feature importance, version history. Use after every successful model train.
---

# model-card

Render `ml/docs/model.html` summarising the current model lineup.

## Sections
1. Headline accuracy on 2025 (latest model only).
2. Calibration reliability diagram (PNG inline).
3. Feature importance (top 20).
4. Per-phase accuracy table (pre-match, post-toss, post-PP1, innings-break).
5. Baseline comparison: coin flip (50%), toss-wins-match (~53%), heuristic, academic best (~70%).
6. Known limitations.
7. Version history table from `model_history.md`.
8. Last-updated UTC timestamp.

## Style
- White background `#ffffff`, dark text `#0f0f0f`.
- System UI font stack; no Google Fonts.
- No animations, no JS except Chart.js (CDN) for WP demo chart.
- Single column, max-width 720px.

## Output
- `ml/docs/model.html` (atomic write).
- Append a row to `ml/docs/model_history.md` describing this version.
