#!/bin/bash
#SBATCH --job-name=e09f_inf
#SBATCH --partition=dev
#SBATCH --account=gov108018
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=experiments/exp09_more_fewshot/e09f/logs/slurm_%j.out
#SBATCH --error=experiments/exp09_more_fewshot/e09f/logs/slurm_%j.err
set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"; cd "$WORKDIR"; source .venv/bin/activate
export HF_HOME=/work/whats2000/hf_cache TORCH_HOME=/work/whats2000/torch_hub
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
CELL="experiments/exp09_more_fewshot/e09f"; mkdir -p "$CELL/logs" "$CELL/outputs"
python "$CELL/main.py" --base configs/base.yaml --config "$CELL/config.yaml"
