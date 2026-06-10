# exp05 — DoRA (weight-decomposed LoRA) as a LoRA-method lever

> Pre-registration. Observed numbers in `experiments/NOTES.md` after the runs.

## Goal

Stay on the LoRA *method* (base model fixed at Llama-3.2-1B). Test whether
**DoRA** (weight-decomposed LoRA, `use_dora=True`) — the current SOTA LoRA
variant — (a) beats vanilla LoRA at matched rank on the Kaggle test, and (b)
adds a *method-diverse* member that lifts the ensemble past 0.7888.

## Hypothesis

DoRA decomposes each weight update into magnitude + direction and often beats
vanilla LoRA ~1pt at equal rank. Expected: DoRA ≥ vanilla solo at r∈{128,192,256},
and DoRA adapters diverse enough from vanilla to improve the ensemble.

## Design

- Phase 1 (e05a/b/c): train DoRA at r∈{256,192,128}, zero_shot, α=2r, effective
  batch 768. Config-driven `use_dora` via cell-local `model.py`/`main.py`
  (copies of root, add `use_dora`/`use_rslora`/`init_lora_weights`). NOTE: DoRA
  is memory-heavy — needed per-GPU batch 8 × accum 12 (still 768 effective) to
  avoid OOM at r=256 on the H200.
- Phase 2 (e05d/e/f): fold DoRA adapters into the exp04 best ensemble
  {van256, van192, fs256} via the cross-strategy LL scorer (DoRA loads normally).

## Metric

Kaggle test (val proxy misranks — established exp01–04). DoRA adapters are
zero_shot and LL-scorable; they load via `PeftModel.from_pretrained` like vanilla
LoRA (no PiSSA-style save conversion needed).
