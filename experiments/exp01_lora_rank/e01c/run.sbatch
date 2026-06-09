#!/bin/bash
#SBATCH --job-name=e01c_r32
#SBATCH --partition=8gpus
#SBATCH --account=gov108018
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=experiments/exp01_lora_rank/e01c/logs/slurm_%j.out
#SBATCH --error=experiments/exp01_lora_rank/e01c/logs/slurm_%j.err

# exp01 / e01c — LoRA rank sweep: r=32, alpha=64, zero_shot.
# Single-node 8×H200. Effective batch 768 (48 × grad_accum 2 × 8 GPU).
set -euo pipefail

# Repo root: SLURM records the submit dir; fall back to script location (../../..).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    WORKDIR="${SLURM_SUBMIT_DIR}"
else
    WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$WORKDIR"

source .venv/bin/activate

# ── Redirect all caches to /work (home quota is limited) ──────────────────────
export HF_HOME=/work/whats2000/hf_cache
export TORCH_HOME=/work/whats2000/torch_hub
# meta-llama/Llama-3.2-1B-Instruct is gated; pass the existing HF token through.
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"

CELL="experiments/exp01_lora_rank/e01c"
mkdir -p "$CELL/logs" "$CELL/outputs"

# ── TMPDIR off /tmp to avoid node swap pressure ───────────────────────────────
export TMPDIR="$WORKDIR/.tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR" || true' EXIT

echo "Cell e01c  r=32 alpha=64  host=$(hostname)  GPUs=$CUDA_VISIBLE_DEVICES"

accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    main.py \
        --base configs/base.yaml \
        --config "$CELL/config.yaml"

echo "Done: e01c (r=32)"
