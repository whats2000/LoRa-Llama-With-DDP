# exp08 — few_shot shot-count diversity (more orthogonal members)

> Pre-registration. Numbers in `experiments/NOTES.md` after the runs.

## Goal
exp07 (0.7922) confirmed orthogonal members help. few_shot is the orthogonal
axis; do *different shot counts* (2/4/8) give orthogonal-enough views to stack?

## Cells
- e08a: few_shot 8-shot @ eff192 (maxlen 768). e08b: few_shot 2-shot @ eff192.
  (4-shot = exp07 e07b.)
- e08c/d/e: LL ensembles mixing zero_shot + few_shot {2,4,8}-shot members.
  Scorer extended: each few_shot member scored with its OWN trained shot count
  (per-member `shots` field) — a 4-shot prompt for an 8-shot-trained adapter is a
  train/inference mismatch.

## Metric
Kaggle test. Diversity precheck: fs2/fs4/fs8 mutually agree only 0.84–0.87
(as orthogonal as zero_shot-vs-few_shot) — unlike DoRA (same-strategy, correlated).
