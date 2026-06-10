#!/bin/bash
#SBATCH --job-name=e06c_zs
#SBATCH --partition=8gpus
#SBATCH --account=gov108018
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=experiments/exp06_zeroshot_recipe/e06c/logs/slurm_%j.out
#SBATCH --error=experiments/exp06_zeroshot_recipe/e06c/logs/slurm_%j.err
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then WORKDIR="${SLURM_SUBMIT_DIR}"; else
    WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
cd "$WORKDIR"; source .venv/bin/activate
export HF_HOME=/work/whats2000/hf_cache TORCH_HOME=/work/whats2000/torch_hub
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
CELL="experiments/exp06_zeroshot_recipe/e06c"; mkdir -p "$CELL/logs" "$CELL/outputs"
export TMPDIR="$WORKDIR/.tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR" || true' EXIT
echo "Cell e06c  lr=5e-5 eff192 (4x steps)  host=$(hostname)"
accelerate launch --multi_gpu --num_machines=1 --num_processes=8 \
    --mixed_precision=bf16 --dynamo_backend=no \
    main.py --base configs/base.yaml --config "$CELL/config.yaml"
echo "Done: e06c"
