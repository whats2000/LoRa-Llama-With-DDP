#!/bin/bash
#SBATCH --job-name=LoRa-Llama-3.2
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
WORKDIR="/work/whats2000/generative-artificial-intelligence-2026/HW1-Question-Answering/"
cd "$WORKDIR"

source .venv/bin/activate

mkdir -p logs

# ── Redirect TMPDIR away from /tmp to avoid swap pressure ────────────────────
export TMPDIR="$WORKDIR/.tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# ── Clean /tmp on this node (user-owned files only) ───────────────────────────
echo "Cleaning /tmp on $(hostname)…"
find /tmp -maxdepth 1 -user "$USER" -mindepth 1 -exec rm -rf {} + 2>/dev/null || true

# ── Multi-node accelerate settings ───────────────────────────────────────────
NUM_GPUS_PER_NODE=8
NUM_PROCESSES=$(( SLURM_NNODES * NUM_GPUS_PER_NODE ))
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "Nodes: $SLURM_NNODES  |  Master: $MASTER_ADDR  |  Total GPUs: $NUM_PROCESSES"

# Helper: run one experiment across all allocated nodes
run_experiment() {
    local config_file="$1"
    local extra_args="${2:-}"
    local exp_name
    exp_name=$(basename "$config_file" .yaml)

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Starting experiment: ${exp_name}  (${NUM_PROCESSES} GPUs across ${SLURM_NNODES} nodes)"
    echo "════════════════════════════════════════════════════════════"

    # Flush dirty pages and purge /tmp between runs to keep node RAM clean
    sync
    srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
        bash -c 'find /tmp -maxdepth 1 -user "$USER" -mindepth 1 -exec rm -rf {} + 2>/dev/null || true' || true

    srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
        bash -c "
            cd '$WORKDIR'
            source .venv/bin/activate
            export TMPDIR='$TMPDIR'
            accelerate launch \\
                --multi_gpu \\
                --num_machines=$SLURM_NNODES \\
                --num_processes=$NUM_PROCESSES \\
                --machine_rank=\$SLURM_NODEID \\
                --main_process_ip=$MASTER_ADDR \\
                --main_process_port=$MASTER_PORT \\
                --mixed_precision=bf16 \\
                --dynamo_backend=no \\
                main.py \\
                    --base configs/base.yaml \\
                    --config '$config_file' \\
                    $extra_args
        "

    echo "  Done: ${exp_name}"
}

# ── Skip flags — set to 1 to skip that experiment ────────────────────────────
SKIP_ZERO_SHOT=0
SKIP_FEW_SHOT=0
SKIP_COT=0

# ── Predict-only mode — set to 1 to skip training and run inference only ──────
# Loads the saved adapter from paths.saved_models defined in each config.
PREDICT_ONLY="${PREDICT_ONLY:-0}"
_PRED_FLAG=""
[[ "$PREDICT_ONLY" -eq 1 ]] && _PRED_FLAG="--predict-only" && echo "predict-only mode enabled"

# ── Run all experiments sequentially ─────────────────────────────────────────
[[ "$SKIP_ZERO_SHOT" -eq 1 ]] && echo "Skipping zero_shot (SKIP_ZERO_SHOT=1)" || run_experiment configs/zero_shot.yaml "$_PRED_FLAG"
[[ "$SKIP_FEW_SHOT"  -eq 1 ]] && echo "Skipping few_shot  (SKIP_FEW_SHOT=1)"  || run_experiment configs/few_shot.yaml  "$_PRED_FLAG"
[[ "$SKIP_COT"       -eq 1 ]] && echo "Skipping cot       (SKIP_COT=1)"       || run_experiment configs/cot.yaml       "$_PRED_FLAG"

echo ""
echo "All experiments completed."
