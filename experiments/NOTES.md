# Lab journal — LoRA-Llama PathoQA

Running record of completed experiments. **Append an entry only after a cell
has run and its metrics are committed; every number must trace to a real run
artifact.** Validation accuracy is an internal proxy — the private-test
submission score (reported by the user) is authoritative.

## Pipeline evolution (headline val_accuracy proxy)

| Ref | Strategy | Knob | Best val_acc (proxy) | Source artifact |
|-----|----------|------|----------------------|-----------------|
| baseline (root) | zero_shot | r=64/α=128, lr5e-5, 10ep | 0.7441 (epoch 6) | `saved_models/checkpoint/zero_shot/training_history.json` |
| baseline (root) | few_shot  | r=64/α=128, lr5e-5, 10ep | 0.7333 (epoch 6) | `saved_models/checkpoint/few_shot/training_history.json` |
| baseline (root) | cot       | — | not run (`SKIP_COT=1`) | — |
| **exp01** | zero_shot | LoRA rank sweep r∈{8..128} | **test monotonic in rank: r=128 best 0.7666**, r=64 0.7611, r=32 0.7444, r=16 0.7277, r=8 0.7100 (proxy inverted-U *misranks* — see entry) | `experiments/exp01_lora_rank/e01{a..e}/`; Kaggle hw-1-question-answering 2026-06-09 |

Submission-set recompute (separate inference run, 900 examples,
`outputs/validation/*.jsonl`): zero_shot 0.7367 (663/900), few_shot 0.7322
(659/900). These differ slightly from the in-training proxy above because they
are a separate generation pass over a 900-example submission split rather than
the per-epoch validation split.

---

## exp01 — LoRA rank (adapter capacity) sweep

**Goal:** Find the LoRA rank that maximises the zero_shot val-accuracy ceiling,
testing whether reducing adapter capacity raises the ceiling that overfitting
caps in the baseline.

**Hypothesis (pre-registered in `exp01_lora_rank/README.md`):** lower rank →
less memorisation → higher peak val_acc; peak expected at r=8–32, r=128 to
overfit at least as badly as the r=64 anchor. **This hypothesis was wrong in
its monotonic form** — see findings.

**Method:** five cells, one knob (`lora.r`, with `alpha=2r` held proportional).
Fixed: zero_shot, lr 5e-5, 10 epochs, dropout 0.05, effective batch 768
(1 node × 8 H200 × batch 48 × grad_accum 2), val_ratio 0.1 / seed 42. Each cell
trains the root pipeline unmodified via its own `config.yaml` + `run.sbatch`
(jobs 88529–88533, partition `8gpus`).

**Results.** Best `val_accuracy` (proxy, from each cell's
`training_history.json`) vs **Kaggle test score** (all five cells submitted to
hw-1-question-answering, 2026-06-09; public == private):

| Cell | r | α | best val_acc *(proxy)* | peak epoch | **Kaggle test** | proxy curve shape |
|------|----|----|-------------|-----------|--------------|-------------|
| e01a | 8  | 16  | 0.6970 | 9  | 0.7100 | underfit — train_loss stalls ~0.30, never memorises |
| e01b | 16 | 32  | 0.7266 | 10 | 0.7277 | still rising at ep10, no overfit yet |
| e01c | 32 | 64  | 0.7457 | 10 | 0.7444 | monotonic rise, still climbing at ep10 |
| e01d | 64 | 128 | 0.7422 | 5  | 0.7611 | peaks ep5, then "overfits" down on val |
| **e01e** | **128**| **256** | 0.7370 | 8  | **0.7666** | proxy "worst overfit" (train_loss→0.00, val_loss→0.76) — **yet best on test** |

Reference: root zero_shot baseline proxy 0.7441 (peak ep6); prior baseline
Kaggle submissions (r=64) ranged 0.7366–0.7700, best 0.7700. e01d (r=64) scored
0.7611 here — inside that historical band.

> **⚠️ The proxy and the test rank capacity in OPPOSITE directions.** On the
> validation proxy the order is r=32 > r=64 > r=128 (inverted-U). On the **real
> test the order is monotonic in capacity: r=128 > r=64 > r=32 > r=16 > r=8**
> (0.7666 → 0.7100). The proxy's "overfitting" signal (high val_loss at r=128)
> did **not** predict test degradation — it predicted the *opposite*. The
> proxy-based findings below are kept for the record but are **overridden by the
> test results** (see "Real-test findings").

**Key findings (PROXY-ONLY — overridden by the real-test findings below):**
1. **Capacity↔accuracy is an inverted-U, not monotonic.** r=8 underfits
   (0.697); accuracy peaks at **r=32 (0.7457)**; r=64/128 overfit and land
   lower. The smallest adapter is *not* best — my "lower is better" hypothesis
   was wrong.
2. **The two best cells (r=16, r=32) had val_acc still monotonically rising at
   epoch 10 — they never peaked.** They are *epoch-limited*, not
   capacity-limited at the top. r=32's best == its final == ep10 (0.7457).
   This is the highest-value lead: lower-capacity adapters overfit later, so
   the 10-epoch budget is cutting them off early.
3. **Accuracy/loss decouple at r=32:** val_loss rises from ep3 (0.37→0.54)
   while val_acc keeps climbing — the model gets more confident-wrong on a few
   while getting more right overall. `save_best` keys on accuracy, so this is
   fine for checkpoint selection, but means `val_loss` is a poor early-stop
   signal here.
4. Net proxy gain over baseline is small (0.7457 vs 0.7441, +0.0016) and may be
   within noise; the actionable result is the *dynamic* (lower r still
   improving at ep10), not the headline delta.

**Failed variants (root cause):**
- **e01a (r=8): underfit.** train_loss never drops below ~0.30 and val_acc
  plateaus ~0.69 — 8-dim adapters across 7 modules lack the capacity to fit the
  task. Root cause: too few trainable parameters, not overfitting.
- **e01e (r=128): worst overfit.** train_loss→0.00 by ep7, val_loss blows up to
  0.76, val_acc peaks early (ep3) and stays flat ~0.733. Root cause: excess
  capacity memorises the ~8.1k train set in ~110 steps.
- **e01d (r=64, anchor): overfits after ep5.** Same failure mode as baseline —
  confirms the baseline's capacity is past the sweet spot.

**Real-test findings (Kaggle, all five cells submitted 2026-06-09 — these
OVERRIDE the proxy findings above):**
1. **On the real test, capacity helps monotonically.** Test score rises
   cleanly with rank: r=8 0.7100 → r=16 0.7277 → r=32 0.7444 → r=64 0.7611 →
   r=128 0.7666. Five points, strictly increasing. The low-rank cells (r=8/16)
   are clearly worse (well outside the historical ~0.033 noise band); the high
   end (r=64/128) is the good regime.
2. **The validation proxy actively misranks capacity at the top.** Proxy order
   was r=32 > r=64 > r=128; test order is the reverse. The val "overfitting"
   signal (r=128: train_loss→0.00, val_loss→0.76) was the cell that scored
   **best** on test. Conclusion: the per-epoch val split (900 ex, seed 42) is a
   *misleading* model-selection signal for this task — apparent val overfitting
   does not transfer to test degradation. **Do not trust proxy capacity
   rankings; the proxy↔test correlation is negative at the high-capacity end.**
3. **e01e (r=128, 0.7666) essentially ties the historical best (0.7700)** —
   within the run-to-run band — and e01d (r=64, 0.7611) reproduces the baseline
   regime. Neither *cleanly* beat 0.7700, but both confirm r≥64 is where the
   good scores live.
4. **Test variance is still large within a rank** (~0.033 across prior r=64
   runs), so the r=64 vs r=128 gap (0.7611 vs 0.7666) is not individually
   decisive — but the *cross-rank monotonic trend* clears the noise band.

**Conclusion / shipped?** **Nothing shipped yet.** The actionable, evidence-based
takeaways: (a) **more LoRA capacity is good, not bad** — the baseline r=64 is
fine and r=128 is at least as good; do not reduce rank. (b) **The validation
proxy is unreliable for ranking and must not be used to pick capacity** — future
selection should lean on the Kaggle test directly (100/day budget) or a better
proxy. (c) Capacity alone plateaus around the historical best (~0.76–0.77); to
clear 0.77 needs a *different lever* than rank — candidates: higher rank still
(r≥256), strategy (few_shot/CoT, never tested on Kaggle), inference-time methods
(self-consistency / option-likelihood scoring), or ensembling. Next: a
literature pass on LoRA for small-model MCQA to pick the highest-evidence lever
before the next sweep.

**Files:**
- Configs/runners: `exp01_lora_rank/e01{a..e}/config.yaml`, `…/run.sbatch`
- Curves/metrics: `exp01_lora_rank/e01{a..e}/saved_models/training_history.json`
- Benchmark submissions: `exp01_lora_rank/e01{a..e}/outputs/zero_shot_submission.csv`
- Design/hypothesis: `exp01_lora_rank/README.md`
