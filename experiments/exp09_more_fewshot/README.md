# exp09 — more few_shot shot-count views (does the lever keep scaling?)

> Pre-registration. Numbers in `experiments/NOTES.md` after the runs.

## Goal
exp08 (0.7988, 5-member with fs{2,4,8}) showed more orthogonal few_shot views
help. Does adding fs{1,3,6} push further toward 0.8088, or saturate?

## Cells
- e09a/b/c: few_shot 1/3/6-shot @ eff192.
- e09d/e/f: ensembles up to 8 members (2 zero_shot + 6 few_shot views) and subsets.

## Metric
Kaggle test. Result: saturated — see NOTES.
