# exp07 — rebuild the best ensemble from eff192-trained members

> Pre-registration. Numbers in `experiments/NOTES.md` after the runs.

## Goal
exp06 found eff batch 192 is the best single-adapter recipe (zs256 0.7766→0.7844),
but the exp04 best ensemble (0.7888) was built from under-trained (eff768) members.
Retrain the ensemble members at eff192 and re-ensemble — does upgrading every
member's recipe (especially the orthogonal few_shot member) clear 0.7888?

## Cells
- e07a: zero_shot r192 @ eff192 (retrain). e07b: few_shot r256 @ eff192 (retrain).
  (eff192 zs256 already exists = exp06 e06c.)
- e07c: all-eff192 trio {zs256, zs192, fs256} (the bet).
- e07d: eff192 zs-pair {zs256, zs192} (no few_shot — isolates few_shot's value).
- e07e: eff192 trio + old eff768 van256 (extra member).

## Metric
Kaggle test (proxy misranks). Members scored by the cross-strategy LL scorer.
