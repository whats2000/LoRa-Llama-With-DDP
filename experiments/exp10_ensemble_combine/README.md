# exp10 — ensemble combination method (free micro-optimization)

> Pre-registration. Numbers in `experiments/NOTES.md` after the runs.

## Goal
Best = 5-member {zs256,zs192,fs2,fs4,fs8} uniform-arithmetic LL ensemble (0.7988).
Can a better *combination* (geometric mean / weighting) squeeze more, no training?

## Cells (same 5 members, vary combine)
- e10a geometric uniform; e10b arith zs×1.5; e10c geom zs×1.5; e10d arith solo-weighted.
Scorer extended with `ensemble.combine` (arithmetic|geometric) and per-member `weight`.

## Metric
Kaggle test. Result: none beat uniform arithmetic — see NOTES.
