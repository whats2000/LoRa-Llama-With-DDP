#!/bin/bash
#SBATCH --job-name=e05f_ens
#SBATCH --partition=dev
#SBATCH --account=gov108018
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=experiments/exp05_dora_variant/e05f/logs/slurm_%j.out
#SBATCH --error=experiments/exp05_dora_variant/e05f/logs/slurm_%j.err
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then WORKDIR="${SLURM_SUBMIT_DIR}"; else
    WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
cd "$WORKDIR"; source .venv/bin/activate
export HF_HOME=/work/whats2000/hf_cache TORCH_HOME=/work/whats2000/torch_hub
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
CELL="experiments/exp05_dora_variant/e05f"; mkdir -p "$CELL/logs" "$CELL/outputs"
echo "Cell e05f  van256 + dora256 + fs256  host=$(hostname)"
python "$CELL/main.py" --base configs/base.yaml --config "$CELL/config.yaml"
echo "Done: e05f"
