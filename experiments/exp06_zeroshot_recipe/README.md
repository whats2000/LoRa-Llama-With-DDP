# exp06 — zero_shot training recipe (prompt/loss/hyperparameters)

> Pre-registration. Numbers in `experiments/NOTES.md` after the runs.

## Goal
Per user steer (CoT/few_shot don't help at 1B; focus on *training* the zero_shot
model). Improve the zero_shot r=256 recipe: learning rate, effective batch /
optimizer-step count, and the loss function. Control = exp02 e02c (lr 5e-5, eff
batch 768) → Kaggle 0.7766.

## Cells
- e06a/b: learning rate 1e-4 / 2e-4 (config).
- e06c/e/f: effective-batch / #steps — 192 / 96 / 384 (config). 768 → too few
  steps (~110); smaller batch = more optimizer steps. Hypothesis: undertraining.
- e06d: **restricted 4-way option loss** — cross-entropy over {A,B,C,D} at the
  answer position (cell-local `train.py`), aligning training with the LL
  inference objective. LL-scored (e06g) since it doesn't supervise EOS/generation.
- e06g/h: inference — LL-score e06d; ensemble eff192 + van192 + fs256.

## Metric
Kaggle test (proxy misranks). LR/batch cells use the root generation submission;
the loss cell and ensembles use the LL scorer.
