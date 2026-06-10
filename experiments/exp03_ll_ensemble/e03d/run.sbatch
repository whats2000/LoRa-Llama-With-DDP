#!/bin/bash
#SBATCH --job-name=e03d_ll
#SBATCH --partition=dev
#SBATCH --account=gov108018
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/exp03_ll_ensemble/e03d/logs/slurm_%j.out
#SBATCH --error=experiments/exp03_ll_ensemble/e03d/logs/slurm_%j.err
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then WORKDIR="${SLURM_SUBMIT_DIR}"; else
    WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
cd "$WORKDIR"
source .venv/bin/activate
export HF_HOME=/work/whats2000/hf_cache
export TORCH_HOME=/work/whats2000/torch_hub
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
CELL="experiments/exp03_ll_ensemble/e03d"
mkdir -p "$CELL/logs" "$CELL/outputs"
echo "Cell e03d  LL ensemble (top-4 {r256,r192,r128,r64})  host=$(hostname)  GPU=$CUDA_VISIBLE_DEVICES"
python "$CELL/main.py" --base configs/base.yaml --config "$CELL/config.yaml"
echo "Done: e03d"
