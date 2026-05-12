# Venue sub-model experiment (v6 D3)

Generated: 2026-05-12T22:32:21.502930+00:00

Compares a per-venue LightGBM (trained only on matches at that
venue) against the overall v3_phase_pre_match model on the same
venue's 2025 test slice. Decision rule per V6_PLAN.md: commit
the sub-model only if lift > 2.0pp.

## Top venues evaluated

- Narendra Modi Stadium, Ahmedabad
- Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow
- Wankhede Stadium, Mumbai

## Results

| venue | n_train | n_val | n_test | overall_acc | venue_acc | lift (pp) | committed |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Narendra Modi Stadium, Ahmedabad | 16 | 8 | 9 | 0.889 | 0.222 | -66.7 | no |
| Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow | 6 | 7 | 8 | 0.125 | 0.750 | +62.5 | yes |
| Wankhede Stadium, Mumbai | 38 | 7 | 7 | 0.429 | 0.429 | +0.0 | no |

## Committed artifacts

_No sub-models hit the +2pp threshold; nothing committed._
