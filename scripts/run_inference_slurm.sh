#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm batch script: quick validation-set inference (first N examples)
#
# Runs infer_validation.py for all three fine-tuned LoRA variants
# (zero_shot, few_shot, cot) on a single node / single GPU.  The output is
# one JSONL file per variant in WORKDIR/validation_outputs/.
#
# Usage:
#   sbatch scripts/run_inference_slurm.sh
#
# Optional environment overrides (via --export or before sbatch):
#   N_EXAMPLES          Number of validation examples per variant (default: 30)
#   CHECKPOINT_ROOT     Root dir of adapter checkpoints (default: saved_models/checkpoint)
#   OUTPUT_DIR          Where to write JSONL files (default: validation_outputs)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=val-infer
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --account=GOV108018

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/work/whats2000/generative-artificial-intelligence-2026/HW1-Question-Answering"
cd "$WORKDIR"

source .venv/bin/activate

mkdir -p logs

# ── Redirect TMPDIR away from /tmp ────────────────────────────────────────────
export TMPDIR="$WORKDIR/.tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# ── Parameters (can be overridden via environment) ────────────────────────────
N_EXAMPLES="${N_EXAMPLES:-30}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-saved_models/checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-validation_outputs}"

echo "══════════════════════════════════════════════════════════════"
echo "  Validation inference"
echo "  Node      : $(hostname)"
echo "  GPUs      : $SLURM_GPUS_ON_NODE"
echo "  Examples  : $N_EXAMPLES per variant"
echo "  Adapters  : $CHECKPOINT_ROOT"
echo "  Output    : $OUTPUT_DIR"
echo "══════════════════════════════════════════════════════════════"

python src/infer_validation.py \
    --base            configs/base.yaml \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --dataset         dataset/dataset.csv \
    --output_dir      "$OUTPUT_DIR" \
    --n               "$N_EXAMPLES" \
    --variants        zero_shot few_shot cot

echo ""
echo "Done.  JSONL files written to $OUTPUT_DIR/"
