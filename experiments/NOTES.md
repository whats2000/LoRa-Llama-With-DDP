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
| **exp02** | zero_shot | rank push r∈{128..512} @ γ=2 | **test peaks at r=256: 0.7766 (NEW BEST)**; r=192 0.7700, r=128 0.7666, r=384 0.7577, r=512 0.7533 (inverted-U on test; r=128 reproduces exp01 exactly) | `experiments/exp02_rank_scaling/e02{a..e}/`; Kaggle 2026-06-09 |
| **exp03** | zero_shot | option-LL ensemble of top adapters | **top-2 {r256,r192} test 0.7811 (NEW BEST)**; top-3 0.7777, top-4 0.7744 (single r256 LL≈gen). Ensembling beats best single by +0.0045 | `experiments/exp03_ll_ensemble/e03{a..d}/`; Kaggle 2026-06-10 |
| **exp04** | zero_shot + few_shot | cross-strategy LL ensemble | **{zs256,zs192,fs256} test 0.7888 (NEW BEST)**; pair {zs256,fs256} 0.7855; fs solo 0.7544; CoT dropped (proxy 0.59). Weak-but-diverse few_shot member lifts ensemble +0.0077 | `experiments/exp04_strategy_diverse/e04{a..e}/`; Kaggle 2026-06-10 |
| **exp05** | zero_shot | DoRA (weight-decomposed LoRA) | **DoRA r256 solo 0.7811 beats vanilla 0.7766**; but r192/r128 worse, and DoRA *hurts* the ensemble (4-mem 0.7855, 5-mem 0.7811 < best 0.7888). Best stays exp04 0.7888 | `experiments/exp05_dora_variant/e05{a..f}/`; Kaggle 2026-06-10 |
| **exp06** | zero_shot | training recipe (LR / steps / loss) | **eff-batch 192 (more steps) single 0.7844 — best single adapter** (+0.0078 vs control 0.7766); lr↑ and eff96 overfit; restricted-loss failed (val 0.7189); eff192 in ensemble 0.7866 < best 0.7888 | `experiments/exp06_zeroshot_recipe/e06{a..h}/`; Kaggle 2026-06-10 |
| **exp07** | zs+fewshot @ eff192 | rebuild ensemble from eff192 members | **eff192 trio {zs256,zs192,fs256} test 0.7922 (NEW BEST)**; few_shot solo jumped 0.7544→0.7733 with eff192; val misranked (zs-pair best on val 0.7767 but test 0.7822 < trio) | `experiments/exp07_eff192_ensemble/e07{a..e}/`; Kaggle 2026-06-10 |
| **exp08** | zs + multi-shot fewshot | few_shot shot-count diversity | **5-member {zs256,zs192,fs2,fs4,fs8} test 0.7988 (NEW BEST)**; 4-mem 0.7955; fs shot-counts mutually orthogonal (agree 0.84–0.87, unlike DoRA). More orthogonal views = better | `experiments/exp08_fewshot_diversity/e08{a..e}/`; Kaggle 2026-06-10 |
| **exp09** | zs + 6 fewshot views | scale up shot-count diversity | **negative: more views saturate/regress** — 8-member 0.7911, 6fs-only 0.7855, all < exp08 5-member 0.7988. Sweet spot ~5 members {zs256,zs192,fs2,fs4,fs8}. Best stays 0.7988 | `experiments/exp09_more_fewshot/e09{a..f}/`; Kaggle 2026-06-10 |
| **exp10** | combination method | weighted / geometric ensemble | **negative: none beat uniform arithmetic 0.7988** — geom uniform 0.7922, zs-weighted 0.7955–0.7966, solo-weighted ties 0.7988. Uniform avg is optimal | `experiments/exp10_ensemble_combine/e10{a..d}/`; Kaggle 2026-06-10 |

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

---

## exp02 — rank push past r=128 (find the capacity ceiling)

**Goal:** exp01 showed test accuracy rising monotonically with rank up to the
edge of its sweep (r=128, 0.7666). Does the gain continue past 128, and where
does it peak/saturate?

**Hypothesis:** if exp01's trend reflected real capacity headroom, test score
keeps rising toward r=512 before plateauing. (Pre-registered in
`exp02_rank_scaling/README.md`.)

**Method:** one knob, `lora.r ∈ {128,192,256,384,512}`, with α=2r holding the
effective scaling γ=α/r=2 constant (vanilla LoRA — *not* rsLoRA; with α=2r the
scaling is already rank-stable, so `use_rslora` would only inflate γ to 22–45, a
confound). zero_shot, lr 5e-5, 10 epochs, effective batch 768, val_ratio 0.1 /
seed 42. Pure config over the unmodified root pipeline. Jobs 88574–88578,
partition `8gpus`. Decision metric = Kaggle test (proxy misranks, per exp01).

**Results** (proxy = best val_acc from `training_history.json`; **test** =
Kaggle hw-1-question-answering, all 5 submitted 2026-06-09, public == private):

| Cell | r | α | proxy (best val) | proxy peak ep | **Kaggle test** |
|------|----|----|------|----|------|
| e02a | 128 | 256  | 0.7431 | 4 | 0.7666 *(= exp01 e01e exactly)* |
| e02b | 192 | 384  | 0.7405 | 5 | 0.7700 |
| **e02c** | **256** | **512** | 0.7465 | 6 | **0.7766 ← NEW BEST** |
| e02d | 384 | 768  | 0.7474 | 7 | 0.7577 |
| e02e | 512 | 1024 | 0.7405 | 4 | 0.7533 |

**Key findings:**
1. **Test accuracy is an inverted-U in rank, peaking at r=256 (0.7766).** The
   left arm (r=8→256) is the monotonic rise exp01 saw; the right arm
   (r=384→512) *declines* (0.7577, 0.7533). exp01's "monotonic" reading was an
   artifact of stopping the sweep at r=128 — the true optimum is r=256.
2. **r=256 (0.7766) beats the prior best (historical 0.7700) and exp01's best
   (r=128, 0.7666) — first genuine improvement over baseline.**
3. **Reproducibility confirmed:** e02a (r=128) scored **0.7666, identical to
   exp01 e01e (r=128)** — same config, same harness, same score. So the score
   differences across cells are real signal, and the ~0.033 spread in the
   *historical* baseline submissions was config drift, not seed noise. This
   raises confidence the r=256 peak is real, not a lucky draw.
4. **Proxy still useless for ranking:** proxy ranked e02d (r=384) highest
   (0.7474), but test ranked it 4th (0.7577); the test-best r=256 had a middling
   proxy (0.7465). Confirms exp01 — do not select capacity on val.

**Failed variants (root cause):**
- **e02d (r=384, 0.7577) and e02e (r=512, 0.7533): over-capacity.** Past r=256,
  extra adapter capacity hurts test accuracy — the model has enough degrees of
  freedom to fit train-set idiosyncrasies that don't transfer. Note proxy peak
  epoch crept later then collapsed (e02e peaked ep4), consistent with
  faster/harder overfitting at very high rank.

**Conclusion / shipped?** **r=256 (α=512) is the new best zero_shot config at
test 0.7766**, beating baseline 0.7700. Not yet merged into the root pipeline
(still zero_shot-only; defer shipping until a strategy/ensemble decision). The
rank lever is now *mapped*: optimum r=256, declining beyond. Further rank
tuning has low headroom — next levers should be orthogonal to capacity.
Candidates for exp03: (1) ensemble the top adapters (e02c r=256 + e02b r=192 +
e02a r=128) via option-likelihood averaging — free, no training; (2)
inference-time option-likelihood scoring vs free-generation; (3) untested
strategies (CoT rationale distillation is already coded, few_shot); (4) a
fine rank refine around 256 (224/256/288) — lowest expected payoff.

**Files:**
- Configs/runners: `exp02_rank_scaling/e02{a..e}/config.yaml`, `…/run.sbatch`
- Curves/metrics: `exp02_rank_scaling/e02{a..e}/saved_models/training_history.json`
- Benchmark submissions: `exp02_rank_scaling/e02{a..e}/outputs/zero_shot_submission.csv`
- Design/hypothesis: `exp02_rank_scaling/README.md`

---

## exp03 — option-likelihood ensemble of the top rank adapters

**Goal:** rank is mapped (exp02: best single r=256, 0.7766). Pursue a lever
orthogonal to capacity: ensemble the already-trained top adapters. No training.

**Hypothesis:** the top adapters disagree on ~15.6% of benchmark questions
(pairwise agreement 0.86–0.93), so averaging their per-option probabilities
should correct complementary errors and beat the best single (0.7766).

**Method:** pure inference (no training). For each adapter, score the 4 answer
letters by the first-token log-prob after the zero_shot prompt, softmax over the
4 options → per-option probabilities; **average those probabilities across
adapters**, argmax. Implemented in cell-local `main.py` (copy of root main.py,
repurposed for inference; reuses `src.data` helpers unmodified). Single-GPU jobs
on `dev`. Verified offline first: single-adapter r=256 LL-acc on val = 0.7467,
matching its generation proxy 0.7465 → LL scoring is lossless. Adapters reused:
r=256 (exp02 e02c), r=192 (e02b), r=128 (e02a), r=64 (exp01 e01d).

**Results** (val LL-acc = offline, gold val split; **test** = Kaggle, submitted
2026-06-10, public == private):

| Cell | ensemble | val LL-acc | **Kaggle test** |
|------|----------|-----------|------|
| e03a | single r=256 | 0.7467 | (not submitted; LL≈gen → ≈ e02c 0.7766) |
| **e03b** | **top-2 {r256,r192}** | 0.7500 | **0.7811 ← NEW BEST** |
| e03c | top-3 {r256,r192,r128} | 0.7533 | 0.7777 |
| e03d | top-4 {+r64} | 0.7489 | 0.7744 |

**Key findings:**
1. **The top-2 ensemble {r256,r192} is the new best at test 0.7811** — beats the
   best single (r256, 0.7766) by +0.0045 and the original baseline (0.7700) by
   +0.0111. Ensembling is a real, training-free gain.
2. **Fewer strong models > more models.** Test order: top-2 (0.7811) > top-3
   (0.7777) > top-4 (0.7744). Adding the weaker r=64 adapter monotonically
   dilutes the ensemble. Use only the strongest 2.
3. **LL scoring is lossless vs generation** for a single model (val 0.7467 ≈
   gen proxy 0.7465) — so the entire exp03 gain is from ensembling, not from
   switching the decode method.
4. **The val proxy misranked ensemble breadth too:** val peaked at top-3
   (0.7533), test peaked at top-2 (0.7811). Val correctly called the *direction*
   ("ensembling helps", "drop r64") but not the exact optimum — consistent with
   exp01/exp02: trust val for direction, the Kaggle test for the final pick.

**Failed variants (root cause):**
- **e03d (top-4, 0.7744): over-broad ensemble.** Including r=64 (single test
  0.7611, the weakest member) drags the averaged probabilities toward its
  errors. Ensemble members should be strong and comparable; a member ~0.015
  below the others is net-negative.

**Conclusion / shipped?** **New best pipeline = top-2 option-LL ensemble of the
r=256 + r=192 zero_shot adapters, test 0.7811.** Progress arc: baseline 0.7700 →
exp02 r=256 single 0.7766 → exp03 top-2 ensemble 0.7811. Still not merged into
the root pipeline (root has no LL-ensemble inference path). Next-lever
candidates: (1) weighted ensemble (weight by single-model strength) or add a
r=320/r=224 sibling to the top-2 for more *diverse-but-strong* members; (2)
untested strategies (CoT/few_shot) as additional diverse ensemble members;
(3) self-consistency on CoT. Rank/capacity itself is exhausted.

**Files:**
- Inference entry + configs/runners: `exp03_ll_ensemble/e03{a..d}/main.py`,
  `…/config.yaml`, `…/run.sbatch`
- Submissions + per-cell summaries: `exp03_ll_ensemble/e03{a..d}/outputs/ll_submission.csv`,
  `…/ll_submission_summary.json`

---

## exp04 — cross-strategy ensemble (diverse prompt strategies as members)

**Goal:** exp03's ensemble gain came from diversity among *same-strategy*
(zero_shot) adapters. Test whether a *different prompt strategy* member —
few_shot and/or CoT at the peak rank r=256 — adds more diversity and lifts the
ensemble past 0.7811.

**Hypothesis:** different strategies make *different* errors, so a strategy-diverse
member should help the ensemble even if it is individually weaker (contrast
exp03, where a weaker *same-strategy* member, r=64, hurt).

**Method:** Phase 1 — train few_shot @ r=256 (e04a) and CoT @ r=256 (e04b) via
the root pipeline (config-only, effective batch 768). Phase 2 — extend the exp03
LL scorer so each ensemble *member* is scored with its own prompt strategy
(few_shot uses its seed-42 4-example prompt; left-truncation preserves the
"Answer:" position); average per-option probs across members. Cells e04c
{zs256,zs192,fs256}, e04d {zs256,fs256}, e04e {zs256,zs192} (control). Adapters
reused: zs256=exp02 e02c, zs192=exp02 e02b. Verified offline on val, then Kaggle.

**Results** (val LL-acc offline; **test** = Kaggle 2026-06-10, public==private):

| Cell | members | val LL-acc | **Kaggle test** |
|------|---------|-----------|------|
| e04a | few_shot solo r256 | 0.7378 (proxy) | 0.7544 |
| e04b | CoT solo r256 | 0.5885 (proxy) | not submitted (dropped) |
| e04e | {zs256, zs192} *(control)* | 0.7500 | 0.7811 *(= exp03 top-2 exactly)* |
| e04d | {zs256, fs256} | 0.7556 | 0.7855 |
| **e04c** | {zs256, zs192, fs256} | 0.7600 | **0.7888 ← NEW BEST** |

**Key findings:**
1. **Cross-strategy ensemble is the new best: {zs256,zs192,fs256} = 0.7888**,
   beating exp03's same-strategy top-2 (0.7811) by +0.0077 and the original
   baseline (0.7700) by +0.0188.
2. **A weak-but-diverse member helps; a weak-but-similar member hurts.** few_shot
   solo (0.7544) is *weaker* than zs256 (0.7766), yet adding it lifts the
   ensemble (+0.0077 in the trio, +0.0089 in the pair vs their zs-only
   counterparts). Contrast exp03 where the weak *same-strategy* r=64 (0.7611)
   *dragged the ensemble down*. The discriminator is error **diversity**
   (different strategy → different mistakes), not member accuracy alone.
3. **The control reproduced exp03's 0.7811 exactly**, on both val (0.7500) and
   test (0.7811) — the extended cross-strategy scorer is correct.
4. **Val ranking held on test this time** (trio > pair > control both ways),
   unlike exp03 where val misranked ensemble breadth. For cross-strategy
   composition the val signal was reliable; still, the absolute val→test gap
   (~+0.03) persists, so val stays a directional guide only.

**Failed variants (root cause):**
- **e04b (CoT @ r=256): proxy 0.5885, dropped before submission.** Two causes:
  (a) a 1B model produces weak chains-of-thought for pathology MCQA; (b) the
  rationales are long and name "Option A/B/C/D" throughout the reasoning, often
  without reaching a clean "Answer: X" inside max_new_tokens — so the
  reverse-token-scan extraction grabs an option letter from the *reasoning*, not
  the final answer. CoT is not first-token LL-scorable and its generation
  extraction is unreliable; salvaging it (forced "Answer:" parsing or
  answer-position scoring) is deferred. Too weak to help the ensemble (cf.
  finding 2 — diversity helps only if the member isn't *this* far below).

**Conclusion / shipped?** **New best = cross-strategy LL ensemble
{zs256, zs192, fs256}, test 0.7888.** Progress arc: baseline 0.7700 → exp02
r256 0.7766 → exp03 zs top-2 0.7811 → exp04 cross-strategy trio 0.7888
(+0.0188 over baseline). Still not merged into root (no LL-ensemble inference
path there). Next levers (diminishing returns): (1) a *properly decoded* CoT
member (fix answer extraction) to add a third strategy; (2) more few_shot
variants (different example sets/seeds) as cheap diverse members; (3) weighted
averaging by member strength; (4) self-consistency on a fixed CoT.

**Files:**
- Training (phase 1): `exp04_strategy_diverse/e04{a,b}/config.yaml`, `…/run.sbatch`
- Ensemble scorer (phase 2): `exp04_strategy_diverse/e04{c,d,e}/main.py`,
  `…/config.yaml`, `…/run.sbatch`
- Submissions + summaries: `exp04_strategy_diverse/e04*/outputs/`

---

## exp05 — DoRA (weight-decomposed LoRA) as a LoRA-method lever

**Goal:** stay on the LoRA *method* (model fixed at 1B). Does DoRA beat vanilla
LoRA at matched rank, and does it add a method-diverse ensemble member past 0.7888?

**Hypothesis:** DoRA ≥ vanilla solo at equal rank, and DoRA adapters diversify
the ensemble. (Pre-registered in `exp05_dora_variant/README.md`.)

**Method:** Phase 1 — DoRA (`use_dora=True`) at r∈{256,192,128}, zero_shot,
α=2r, effective batch 768. DoRA is memory-heavy → OOM at batch 48 on the 140 GB
H200; fixed with batch 8 × accum 12 (768 effective unchanged) + expandable
segments. Config-driven via cell-local `model.py`/`main.py`. Phase 2 — fold DoRA
adapters into the exp04 best ensemble via the cross-strategy LL scorer (DoRA
loads normally onto the base).

**Results** (proxy = best val_acc; **test** = Kaggle 2026-06-10):

| Cell | adapter / ensemble | proxy | **Kaggle test** | vs vanilla |
|------|--------------------|-------|------|-----|
| e05a | DoRA r256 solo | 0.7438 | **0.7811** | vanilla 0.7766 → **DoRA +0.0045** |
| e05b | DoRA r192 solo | 0.7427 | 0.7566 | vanilla 0.7700 → DoRA −0.0134 |
| e05c | DoRA r128 solo | 0.7438 | 0.7577 | vanilla 0.7666 → DoRA −0.0089 |
| e05d | {van256,van192,fs256,**dora256**} | val 0.7600 | 0.7855 | best 0.7888 → **−0.0033** |
| e05e | {…,**dora256,dora192**} | val 0.7656 | 0.7811 | best 0.7888 → **−0.0077** |
| e05f | {van256,dora256,fs256} | val 0.7567 | not submitted (weakest on val) | — |

**Key findings:**
1. **DoRA r=256 solo (0.7811) beats vanilla r=256 (0.7766)** by +0.0045 — a real
   LoRA-method win at the peak rank, and it single-handedly matches the exp03
   2-model vanilla ensemble. But DoRA is **inconsistent**: at r=192/128 it is
   *worse* than vanilla (−0.013/−0.009). DoRA helps only at the high rank here.
2. **DoRA HURT the ensemble** (4-member 0.7855, 5-member 0.7811, both < best
   0.7888) — even though DoRA r256 is individually strong (0.7811) and disagrees
   with vanilla on 11.3% of questions. **This is the central lesson refined:**
   exp04's few_shot helped the ensemble despite being *weaker* because it is a
   different *strategy* (orthogonal errors); DoRA *hurt* despite being *stronger*
   because it is the same strategy (zero_shot) — its errors are **correlated**
   with vanilla zero_shot. **Ensemble gains require orthogonal (cross-strategy)
   diversity, not a different LoRA parameterisation of the same view.**
3. **Val misranked again** — it ranked the 5-member ensemble best (0.7656); test
   ranked it *worst* of the three DoRA ensembles, and all three below the
   3-member best. Consistent with every prior experiment: trust val for
   direction only.

**Failed variants (root cause):**
- **e05b/e05c (DoRA r192/r128 solo): worse than vanilla.** DoRA's magnitude
  decomposition seems to help only near the capacity sweet spot (r256); at lower
  rank it underperforms plain LoRA. Root cause unclear — possibly DoRA's extra
  magnitude params need more capacity/epochs to pay off.
- **e05d/e05e (DoRA in ensemble): regressed vs best.** Root cause: same-strategy
  correlation (finding 2). Adding a member whose errors correlate with existing
  members averages in shared mistakes without orthogonal correction; the weaker
  dora192 (0.7566) in e05e dragged it down further (cf. exp03 r=64).

**Conclusion / shipped?** **Nothing shipped; best stays exp04 {van256,van192,fs256}
= 0.7888.** DoRA is not a net win here: a single-rank solo gain (r256) that does
not transfer to the ensemble, plus inconsistency across ranks. The actionable
takeaway for closing the gap to 0.8088 is finding 2: **add orthogonal views
(different strategies/prompts), not more same-strategy LoRA variants.** Candidate
exp06 levers: multiple few_shot variants (different example sets), expert-persona
/ paraphrased zero_shot prompts, or a working different decision method — each a
genuinely different view to diversify the ensemble.

**Files:**
- DoRA training: `exp05_dora_variant/e05{a,b,c}/{config.yaml,run.sbatch,model.py,main.py}`
- DoRA ensembles: `exp05_dora_variant/e05{d,e,f}/{config.yaml,run.sbatch,main.py}`
- Submissions/summaries: `exp05_dora_variant/e05*/outputs/`

---

## exp06 — zero_shot training recipe (LR / steps / loss)

**Goal:** improve how the zero_shot r=256 adapter is *trained* (user steer: stop
chasing CoT/few_shot at 1B). Control = exp02 e02c (lr 5e-5, eff batch 768) → 0.7766.

**Method:** one knob per cell, all zero_shot r=256. LR {1e-4,2e-4}; effective
batch {384,192,96} (= more optimizer steps at fixed 10 epochs); restricted 4-way
option loss (cell-local `train.py`). LR/batch cells = root pipeline (config); loss
cell scored by the LL scorer.

**Results** (Kaggle test; proxy in parens):

| Cell | recipe | **test** | proxy |
|------|--------|------|-------|
| (e02c) | eff768, lr5e-5 *(control)* | 0.7766 | 0.7465 |
| e06a | lr 1e-4 | 0.7522 | 0.7439 |
| e06b | lr 2e-4 | 0.7666 | 0.7309 |
| e06f | eff384 (2× steps) | 0.7666 | 0.7517 |
| **e06c** | **eff192 (4× steps)** | **0.7844** | 0.7542 |
| e06e | eff96 (8× steps) | 0.7488 | 0.7646 |
| e06d | restricted 4-way loss | — (LL val 0.7189) | gen 0.5972 |
| e06h | ensemble {eff192, van192, fs256} | 0.7866 | val 0.7656 |

**Key findings:**
1. **eff batch 192 is the best single-adapter recipe (0.7844)** — +0.0078 over the
   eff768 control, and our best single adapter (> vanilla 0.7766, DoRA 0.7811).
   The eff768 default under-trained (only ~110 optimizer steps); 4× more steps
   helped.
2. **But the steps curve is non-monotonic / has a sweet spot at eff192.** eff96
   (8× steps) *overfit* — best proxy (0.7646) yet worst test (0.7488). eff384
   (0.7666) even dipped below the control, so the curve is also noisy (~±0.01).
   The proxy badly misranked: it rose monotonically with steps while test peaked
   at eff192 then crashed. Same proxy-misranks lesson as every prior experiment.
3. **Learning rate 5e-5 was already near-optimal** — 1e-4 (0.7522) and 2e-4
   (0.7666) both underperformed the control.
4. **eff192 did NOT improve the ensemble** (e06h 0.7866 < best 0.7888): swapping
   the stronger zero_shot member for van256 *hurt*, consistent with exp05 — a
   stronger but same-strategy member doesn't help (correlated errors).

**Failed variants (root cause):**
- **e06d (restricted 4-way option loss): LL val 0.7189, well below control 0.7467.**
  Root cause: restricting the loss to the 4 option logits discards the full-vocab
  next-token signal, which evidently regularises/teaches useful structure. The
  generative full-vocab CE is the better objective even though we score by option
  LL at inference.
- **e06e (eff96, 8× steps): overfit, 0.7488.** Too many optimizer steps on ~8.1k
  examples memorise the train set; test generalisation collapses.

**Conclusion / shipped?** **Best single-adapter recipe = eff batch 192 (0.7844)**
— a real training-recipe win, and the clearest "how to train zero_shot" result.
But it did not lift the **ensemble** (best stays exp04 {van256,van192,fs256} =
0.7888). Plateau at 0.7888; single-best 0.7844. Open next step (uncertain):
retrain *all* ensemble members (zs192, few_shot) with the eff192 recipe and
re-ensemble — whether stronger-but-correlated members net help is unknown.

**Files:**
- LR/steps (config): `exp06_zeroshot_recipe/e06{a,b,c,e,f}/{config.yaml,run.sbatch}`
- Restricted loss (code): `exp06_zeroshot_recipe/e06d/{train.py,main.py,config.yaml,run.sbatch}`
- Inference: `exp06_zeroshot_recipe/e06{g,h}/{main.py,config.yaml,run.sbatch}`

---

## exp07 — rebuild the best ensemble from eff192-trained members

**Goal:** the exp04 best ensemble (0.7888) used under-trained eff768 members.
exp06 showed eff192 is the better recipe. Retrain the members at eff192 and
re-ensemble — does it clear 0.7888?

**Method:** retrain zs192 (e07a) and few_shot r256 (e07b) at eff batch 192
(batch 24 × accum 1 × 8 GPU); eff192 zs256 reuses exp06 e06c. Ensemble via the
cross-strategy LL scorer.

**Results** (Kaggle test):

| Cell | adapter / ensemble | proxy/val | **test** |
|------|--------------------|-----------|------|
| e07a | zs192 @ eff192 solo | proxy 0.7604 (was 0.7405) | 0.7722 (was 0.7700) |
| e07b | few_shot @ eff192 solo | proxy 0.7524 (was 0.7378) | **0.7733 (was 0.7544, +0.019)** |
| e07d | eff192 zs-pair {zs256,zs192} | val 0.7767 | 0.7822 |
| **e07c** | eff192 trio {zs256,zs192,fs256} | val 0.7689 | **0.7922 ← NEW BEST** |
| e07e | eff192 trio + old van256 | val 0.7700 | 0.7922 |

**Key findings:**
1. **All-eff192 trio = 0.7922, new best** (+0.0034 over exp04's 0.7888). Upgrading
   the members' training recipe lifted the ensemble.
2. **The eff192 recipe helped the orthogonal few_shot member most** (+0.019 solo:
   0.7544→0.7733). few_shot's prompt is longer / harder, so it benefited more from
   the extra optimizer steps — and being the ensemble's only orthogonal member,
   its lift drove the ensemble gain. (zs192 only +0.0022 solo.)
3. **Val misranked yet again:** it ranked the few_shot-free zs-pair best (0.7767),
   but on test the trio WITH few_shot won (0.7922 > pair 0.7822). The cross-strategy
   member helps test even when it lowers val — the exp04 lesson, reconfirmed.
4. Adding the old eff768 van256 (e07e) didn't change the score (0.7922) — a 4th
   correlated zero_shot member is inert.

**Conclusion / shipped?** **New best pipeline = eff192 trio {zs256, zs192, fs256}
LL-ensemble = 0.7922.** Arc: baseline 0.7700 → exp04 0.7888 → exp07 0.7922. The
two reusable wins compounded: eff192 training recipe (exp06) + cross-strategy
ensemble (exp04). Gap to leaderboard 0.8088 now +0.0166. Next candidates: more
orthogonal members at eff192 (few_shot with different example sets / shot counts —
each a different in-context view), which is the only axis that has ever helped.

**Files:**
- Retrain: `exp07_eff192_ensemble/e07{a,b}/{config.yaml,run.sbatch}`
- Ensembles: `exp07_eff192_ensemble/e07{c,d,e}/{main.py,config.yaml,run.sbatch}`
- Submissions: `exp07_eff192_ensemble/e07*/outputs/`

---

## exp08 — few_shot shot-count diversity

**Goal:** orthogonal members help (exp04/07). Are different few_shot *shot counts*
(2/4/8) orthogonal enough to stack as extra ensemble members?

**Method:** train few_shot 8-shot (e08a, maxlen 768) and 2-shot (e08b) at eff192;
4-shot = exp07 e07b. Ensemble via the cross-strategy LL scorer, **extended so each
few_shot member is scored with its own trained shot count** (per-member `shots`;
a mismatch would invalidate fs8/fs2).

**Diversity precheck (benchmark preds):** fs2/fs4/fs8 mutually agree 0.843–0.866,
and ~0.84 vs zero_shot — i.e. as orthogonal to each other as few_shot is to
zero_shot. (Contrast DoRA in exp05: same-strategy, correlated, hurt the ensemble.)

**Results** (Kaggle test):

| Cell | ensemble | val | **test** |
|------|----------|-----|------|
| e08e | {zs256, fs2, fs4, fs8} | 0.7611 | 0.7900 |
| e08d | {zs256, zs192, fs4, fs8} | 0.7667 | 0.7955 |
| **e08c** | {zs256, zs192, fs2, fs4, fs8} | 0.7711 | **0.7988 ← NEW BEST** |

**Key findings:**
1. **5-member multi-shot ensemble = 0.7988, new best** (+0.0066 over exp07 0.7922).
   Stacking few_shot views at different shot counts works.
2. **Shot-count variants are genuinely orthogonal** (0.84–0.87 mutual agreement) —
   the discriminator vs DoRA: DoRA was a different *parameterisation* of the same
   zero_shot decision (correlated); different shot counts are different *decision
   contexts* (orthogonal). Confirms the exp05 lesson from the other side.
3. **More orthogonal views = monotonically better** (5-mem 0.7988 > 4-mem 0.7955 >
   {zs256,fs×3} 0.7900). Each diverse member adds signal.
4. Val ranked these correctly for once (c>d>e both ways) — but absolute val→test
   gap (~+0.03) persists; still trust test.

**Conclusion / shipped?** **New best = 5-member multi-shot LL ensemble = 0.7988.**
Arc: 0.7700 → 0.7888 → 0.7922 → 0.7988. Gap to 0.8088 now +0.0100. The lever is
clearly "more orthogonal few_shot views" — next: additional shot counts (1/3/6/16)
and/or different example sets, all at eff192, then re-ensemble.

**Files:** `exp08_fewshot_diversity/e08{a,b}/` (train), `e08{c,d,e}/` (ensembles).

---

## exp09 — scaling up few_shot shot-count views (negative)

**Goal:** exp08's 5-member multi-shot ensemble hit 0.7988. Does adding more
few_shot views (1/3/6-shot) keep helping toward 0.8088?

**Method:** train few_shot 1/3/6-shot @ eff192 (e09a/b/c); ensemble up to
8 members (2 zero_shot + 6 few_shot views {1,2,3,4,6,8}) and subsets.

**Results** (Kaggle test; best prior = exp08 5-member 0.7988):

| Cell | ensemble | val | test |
|------|----------|-----|------|
| e09e | 6 few_shot only {1,2,3,4,6,8} | 0.7667 | 0.7855 |
| e09f | {zs256, zs192, fs1, fs3, fs4, fs6} | 0.7756 | 0.7888 |
| e09d | 8-member {zs256, zs192, fs1,2,3,4,6,8} | 0.7733 | 0.7911 |
| (e08c) | **{zs256, zs192, fs2, fs4, fs8}** *(prior best)* | 0.7711 | **0.7988** |

**Key findings:**
1. **The orthogonal-views lever saturated and reversed.** All exp09 ensembles
   (0.7855–0.7911) are *below* exp08's 5-member 0.7988. Adding fs1/fs3/fs6 diluted
   rather than helped.
2. **There is an ensemble-composition sweet spot (~5 members, {zs2,zs+fs2,4,8}).**
   Beyond it: (a) the extra few_shot views are weaker (fs1 proxy 0.7635, fs6 0.7466)
   and/or more mutually correlated than the well-spaced {2,4,8}; (b) a 6-few_shot
   ensemble shifts the average too far toward few_shot (individually weaker than
   zero_shot). Same "weak/correlated member dilutes" mechanism as exp03 r64 / exp05
   DoRA, now from the member-count direction.
3. Shot-count spacing matters: geometric {2,4,8} beat {1,3,4,6} (e09f 0.7888).

**Conclusion / shipped?** **No improvement; best stays exp08 5-member = 0.7988.**
The ensemble-composition lever is exhausted — diverse-member stacking peaks at
~5 well-chosen members. Arc holds: 0.7700 → 0.7888 → 0.7922 → 0.7988. Remaining
gap to 0.8088 (+0.0100) likely needs a different lever (weighted combination, or
something orthogonal we haven't found), not more members.

**Files:** `exp09_more_fewshot/e09{a,b,c}/` (train), `e09{d,e,f}/` (ensembles).

---

## exp10 — ensemble combination method (negative)

**Goal:** squeeze the fixed best 5-member set {zs256,zs192,fs2,fs4,fs8} (uniform
arithmetic = 0.7988) via a better combination — geometric mean and/or weighting.

**Method:** extend the LL scorer with `combine` (arithmetic|geometric) and
per-member `weight`. Same 5 members, no retraining.

**Results** (Kaggle test; control uniform-arithmetic = 0.7988):

| Cell | combine | test |
|------|---------|------|
| e10a | geometric, uniform | 0.7922 |
| e10c | geometric, zs×1.5 | 0.7955 |
| e10b | arithmetic, zs×1.5 | 0.7966 |
| e10d | arithmetic, solo-strength weighted | 0.7988 (tie) |
| (e08c) | **arithmetic, uniform** | **0.7988** |

**Key findings:**
1. **Uniform arithmetic averaging is optimal.** No variant beat it; geometric mean
   *hurt* (−0.0066), and up-weighting the stronger zero_shot members *hurt*
   (−0.002 to −0.003).
2. **Why weighting zero_shot up hurts:** it suppresses the few_shot members, which
   are the *orthogonal* contributors driving the ensemble gain. The ensemble wants
   the diverse-but-weaker members at full weight — strength-weighting is
   counterproductive here (the exp04/05 "diversity > strength" lesson, again).
3. Solo-strength weights (1.3/1.2/1/1/1, near-uniform) tied at 0.7988 — barely
   shifted predictions.

**Conclusion / shipped?** **No improvement; best holds at 0.7988** (5-member
uniform-arithmetic LL ensemble). The combination lever is exhausted. After
sweeping capacity, recipe, LoRA-method, ensemble composition, and combination,
the approach plateaus at **0.7988** (+0.0288 over baseline 0.7700). The remaining
+0.0100 to the leaderboard top (0.8088) appears to need a qualitatively different
technique not reachable by the LoRA/ensemble levers explored here.

**Files:** `exp10_ensemble_combine/e10{a..d}/`.
