# exp02 — rank push (continue the monotonic-rank result past r=128)

> Pre-registration. Observed numbers go in `experiments/NOTES.md` only after the
> cells finish AND their Kaggle test scores are collected.

## Goal

exp01 found that on the **Kaggle test** (not the val proxy, which misranks),
zero_shot accuracy rose **monotonically with LoRA rank**: r=8→128 gave
0.7100→0.7666, with no sign of saturation at r=128. This experiment asks: **does
the gain continue past r=128, and where does it saturate (or destabilise)?**

## Why scaling (γ) is held constant, and why rsLoRA is NOT used here

Literature pass (real papers, see NOTES / chat): Kalajdzievski's **rsLoRA**
(arXiv:2312.03732) shows vanilla LoRA's scaling γ=α/r *shrinks* as rank grows
(fixed α), collapsing gradients at high rank — which is why most ablations see
rank "saturate" early (e.g. [doi:10.1111/exsy.70208](https://doi.org/10.1111/exsy.70208)
found r∈{8..64} flat within 1–2%, chose r=8). **We already sidestep this** by
setting **α = 2r**, so γ = α/r = 2 is constant across ranks — i.e. our setup is
*already* rank-stable, the property rsLoRA adds. Enabling `use_rslora` on top of
α=2r would make γ=α/√r blow up to 22–45 (a scaling-magnitude change, not a clean
rsLoRA test), so it is deliberately **not** used here. A proper γ / fixed-α
rsLoRA comparison (with sane γ) is deferred to exp03.

This experiment therefore varies **one knob — `lora.r`** — with α=2r holding
γ=2 fixed, isolating capacity.

## Hypothesis

If exp01's monotonic trend reflects genuine capacity headroom (not noise), test
score should keep rising from r=128 toward r=512, plateauing somewhere. If it
plateaus immediately at r=128 or degrades, the exp01 trend was near its ceiling
and the next lever is not rank (→ strategy / inference-time methods). Test
variance (~0.033, exp01) means only a clear multi-point trend is decisive.

## Design — one knob: `lora.r` (α=2r, γ=2 constant, vanilla LoRA)

| Cell | r | α | role |
|------|----|----|------|
| e02a | 128 | 256  | anchor — reproduces exp01 e01e (test 0.7666) |
| e02b | 192 | 384  | |
| e02c | 256 | 512  | |
| e02d | 384 | 768  | |
| e02e | 512 | 1024 | top of the push |

### Fixed across all cells
zero_shot; lr 5e-5; 10 epochs; dropout 0.05; all 7 linear target modules;
effective batch 768 (8 GPU × 48 × accum 2); val_ratio 0.1 / seed 42. Pure config
over the unmodified root pipeline (`main.py` + `src/`), one `config.yaml` +
`run.sbatch` per cell. Single node, 8× H200, partition `8gpus`, account
`gov108018`.

## Metric

**Kaggle hw-1-question-answering test score is the decision metric** (val proxy
misranks capacity — exp01). Each cell's benchmark CSV at
`e02{a..e}/outputs/zero_shot_submission.csv` is submitted; val proxy from
`saved_models/training_history.json` is recorded but not trusted for ranking.

## Files
- `e02{a..e}/config.yaml`, `e02{a..e}/run.sbatch`
- `e02{a..e}/saved_models/training_history.json` (proxy, gitignored)
- `e02{a..e}/outputs/zero_shot_submission.csv` (gitignored)
