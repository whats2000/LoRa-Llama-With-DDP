# exp01 — LoRA rank (adapter capacity) sweep

> Pre-registration. Hypotheses and design live here; **observed numbers go in
> `experiments/NOTES.md` only after the cells finish** (lab-journal discipline).

## Goal

Find the LoRA rank that maximises the validation-accuracy ceiling for the
zero-shot PathoQA fine-tune. The baseline runs overfit hard, so the question
is whether **reducing adapter capacity** raises the ceiling that overfitting
currently caps.

## Motivation (from the baseline)

Baseline `training_history.json` for both zero_shot and few_shot shows a clean
overfitting signature:

- `train_loss` collapses to ~0.017–0.034 by epoch 10 (near-perfect memorisation).
- `val_loss` bottoms at **epoch 2–3 (~0.38)** then climbs to ~0.70.
- `val_accuracy` peaks at **epoch 5–6 (~0.744 zero / ~0.733 few)** then declines.

`save_best=true` already harvests the peak epoch, so the *ceiling* (~0.744),
not late-epoch decay, is the limiter. The baseline adapter is large:
`r=64, alpha=128` on all 7 linear modules of a 1B model, trained on ~8.2k
examples in ~110 optimiser steps. Excess capacity is the leading suspect for
the memorisation.

## Hypothesis

Lower rank → less capacity to memorise → better generalisation → higher peak
`val_accuracy`. I expect the peak to land at **r=8–32** rather than r=64, with
r=128 (e01e) overfitting at least as badly as the r=64 anchor. If accuracy is
flat across ranks, capacity is *not* the binding constraint and the next
experiment should target the LR schedule or training target instead.

## Design — one knob: `lora.r` (with `alpha = 2r` held proportional)

| Cell | r | alpha | Role |
|------|----|------|------|
| e01a | 8  | 16  | low capacity |
| e01b | 16 | 32  | |
| e01c | 32 | 64  | |
| e01d | 64 | 128 | **baseline anchor / control** |
| e01e | 128| 256 | high capacity |

`alpha = 2r` keeps the effective LoRA scaling (`alpha/r = 2`) constant so the
sweep isolates *capacity*, not *update magnitude*.

### Fixed across all cells

- Strategy: **zero_shot** (strongest *and* cheapest baseline; no rationale gen).
- Optimiser: lr 5e-5, weight_decay 0.01, warmup_ratio 0.05, 10 epochs, `save_best`.
- LoRA: dropout 0.05, bias none, all 7 linear target modules.
- Effective batch **768** = 8 GPU × batch 48 × grad_accum 2 (matches baseline).
- Data split: val_ratio 0.1, seed 42 (identical val set across cells).

### Hardware / launch

Single node, 8× H200, partition `8gpus`, account `gov108018`. Each cell has its
own `run.sbatch`; submit from the repo root so `SLURM_SUBMIT_DIR` is the root
(the root pipeline — `main.py`, `src/`, `configs/base.yaml` — is used
unmodified; only the cell `config.yaml` differs, so no root `.py` is copied
into the cells).

## Metrics

- **Proxy (steering):** best `val_accuracy` over epochs, read from each cell's
  `saved_models/training_history.json`.
- **Real (reported by user):** private-test score from the benchmark submission
  CSV at `eNNx/outputs/zero_shot_submission.csv`. Validation accuracy here is
  only an estimate; the private test is authoritative.

## Files

- `e01{a..e}/config.yaml` — per-cell overrides (the varied knob).
- `e01{a..e}/run.sbatch` — per-cell launcher.
- `e01{a..e}/saved_models/` — adapter + `training_history.json` (gitignored).
- `e01{a..e}/outputs/` — benchmark submission CSV + details JSONL (gitignored).
