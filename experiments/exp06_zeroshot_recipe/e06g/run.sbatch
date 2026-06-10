#!/bin/bash
#SBATCH --job-name=e06g_inf
#SBATCH --partition=dev
#SBATCH --account=gov108018
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=experiments/exp06_zeroshot_recipe/e06g/logs/slurm_%j.out
#SBATCH --error=experiments/exp06_zeroshot_recipe/e06g/logs/slurm_%j.err
set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"; cd "$WORKDIR"; source .venv/bin/activate
export HF_HOME=/work/whats2000/hf_cache TORCH_HOME=/work/whats2000/torch_hub
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
CELL="experiments/exp06_zeroshot_recipe/e06g"; mkdir -p "$CELL/logs" "$CELL/outputs"
echo "Cell e06g  LL-score e06d restricted-loss"
python "$CELL/main.py" --base configs/base.yaml --config "$CELL/config.yaml"
