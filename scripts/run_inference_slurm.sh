#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Slurm batch script: vLLM-based parallel validation-set inference
#
# Launches vLLM server(s) with LoRA adapters on each node, then runs
# infer_validation.py clients in parallel.  Each client processes a data
# shard, and results are merged after all ranks complete.
#
# Default (2 nodes × 8 GPUs = 16 GPUs):
#   sbatch scripts/run_inference_slurm.sh
#
# Optional environment overrides (via --export or before sbatch):
#   N_EXAMPLES          Number of validation examples per variant (default: 0 = all)
#   CHECKPOINT_ROOT     Root dir of adapter checkpoints (default: saved_models/checkpoint)
#   OUTPUT_DIR          Where to write JSONL files (default: validation_outputs)
#   TP_SIZE             Tensor-parallel GPUs per vLLM instance (default: 1)
#   MAX_MODEL_LEN       vLLM max context length (default: 2048)
#   GPU_MEM_UTIL        GPU memory utilization ratio (default: 0.90)
#   MAX_WORKERS         Concurrent API requests per client (default: 16)
#   MODEL_ID            Base model ID (default: read from configs/base.yaml)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=vllm-infer
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
if [ -z "${WORKDIR:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        WORKDIR="${SLURM_SUBMIT_DIR}"
    elif [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
        WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        WORKDIR="$(dirname "${WORKDIR}")"
    else
        echo "ERROR: Cannot auto-detect WORKDIR. Please set it manually." >&2
        exit 1
    fi
fi
cd "$WORKDIR"

source .venv/bin/activate

mkdir -p logs

# ── Redirect TMPDIR away from /tmp ────────────────────────────────────────────
export TMPDIR="$WORKDIR/.tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$TMPDIR"' EXIT

# ── Parameters (can be overridden via environment) ────────────────────────────
N_EXAMPLES="${N_EXAMPLES:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-saved_models/checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/validation}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_WORKERS="${MAX_WORKERS:-16}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VARIANTS="${VARIANTS:-zero_shot few_shot}" # We skip "cot" for faster inference, but can add it back if needed

# Read MODEL_ID from config if not set
if [ -z "${MODEL_ID:-}" ]; then
    MODEL_ID=$(python -c "
import yaml
with open('configs/base.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg['model']['model_id'])
")
fi

# Avoid port collisions across concurrent SLURM jobs
BASE_PORT=$((8000 + (${SLURM_JOB_ID:-$$} % 1000) * 10))

# Export variables for node workers
export WORKDIR N_EXAMPLES CHECKPOINT_ROOT OUTPUT_DIR TP_SIZE MAX_MODEL_LEN
export GPU_MEM_UTIL MAX_WORKERS VLLM_STARTUP_TIMEOUT MODEL_ID BASE_PORT VARIANTS

# ── Helper: wait for vLLM health check ────────────────────────────────────────
wait_for_vllm() {
    local url="$1"
    local timeout="$2"
    local deadline=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "  vLLM ready at $url"
            return 0
        fi
        sleep 5
    done

    echo "  ERROR: vLLM failed to start at $url within ${timeout}s" >&2
    return 1
}
export -f wait_for_vllm

# ══════════════════════════════════════════════════════════════════════════════
# NODE WORKER MODE — executed by srun on each node
# ══════════════════════════════════════════════════════════════════════════════
node_worker() {
    local TIMESTAMP="$1"

    cd "$WORKDIR"
    source .venv/bin/activate

    # How many GPUs does this node have?
    GPUS_ON_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
    INSTANCES_PER_NODE=$((GPUS_ON_NODE / TP_SIZE))
    if [ "$INSTANCES_PER_NODE" -lt 1 ]; then
        INSTANCES_PER_NODE=1
    fi

    # Global rank offset: node 0 gets ranks 0..N-1, node 1 gets N..2N-1, etc.
    NODE_ID="${SLURM_NODEID:-0}"
    GLOBAL_RANK_OFFSET=$((NODE_ID * INSTANCES_PER_NODE))
    WORLD_SIZE=$((${SLURM_NNODES:-1} * INSTANCES_PER_NODE))

    echo "══════════════════════════════════════════════════════════════"
    echo "  Node worker on $(hostname)"
    echo "  GPUs: $GPUS_ON_NODE  TP: $TP_SIZE  Instances: $INSTANCES_PER_NODE"
    echo "  Global rank offset: $GLOBAL_RANK_OFFSET  World size: $WORLD_SIZE"
    echo "══════════════════════════════════════════════════════════════"

    # Clean up stale temp files from prior runs
    rm -rf /tmp/vllm_cache_* /tmp/inductor_* /tmp/triton_* \
           /tmp/tiktoken_rs_* /tmp/vllm_*.log 2>/dev/null || true
    rm -f /dev/shm/nccl-* /dev/shm/torch_* 2>/dev/null || true

    # Build --lora-modules argument for all available adapters
    LORA_MODULES_ARG=""
    for variant in $VARIANTS; do
        adapter_path="${WORKDIR}/${CHECKPOINT_ROOT}/${variant}"
        if [ -d "$adapter_path" ]; then
            if [ -n "$LORA_MODULES_ARG" ]; then
                LORA_MODULES_ARG="${LORA_MODULES_ARG} "
            fi
            LORA_MODULES_ARG="${LORA_MODULES_ARG}${variant}=${adapter_path}"
        fi
    done

    if [ -z "$LORA_MODULES_ARG" ]; then
        echo "  ERROR: No LoRA adapters found under ${CHECKPOINT_ROOT}/" >&2
        return 1
    fi

    # ── Start vLLM instances with staggered launch ────────────────────────────
    VLLM_PIDS=()
    for i in $(seq 0 $((INSTANCES_PER_NODE - 1))); do
        GPU_START=$((i * TP_SIZE))
        GPU_END=$((GPU_START + TP_SIZE - 1))
        CUDA_DEVS=$(seq -s, "$GPU_START" "$GPU_END")
        PORT=$((BASE_PORT + i))
        GLOBAL_RANK=$((GLOBAL_RANK_OFFSET + i))

        # Isolate temp directories per instance
        export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor_${GLOBAL_RANK}"
        export TRITON_CACHE_DIR="${TMPDIR}/triton_${GLOBAL_RANK}"
        export VLLM_CACHE_ROOT="${TMPDIR}/vllm_cache_${GLOBAL_RANK}"
        mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

        # Stagger startup to avoid NCCL port conflicts
        if [ "$i" -gt 0 ]; then
            sleep $((i * 10))
        fi

        echo "  Starting vLLM instance $i (GPU $CUDA_DEVS, port $PORT, rank $GLOBAL_RANK)..."

        CUDA_VISIBLE_DEVICES=$CUDA_DEVS \
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_ID" \
            --tensor-parallel-size "$TP_SIZE" \
            --port "$PORT" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --enable-lora \
            --lora-modules $LORA_MODULES_ARG \
            --max-lora-rank 64 \
            > "${TMPDIR}/vllm_rank${GLOBAL_RANK}.log" 2>&1 &

        VLLM_PIDS+=($!)
    done

    # ── Health check all instances ────────────────────────────────────────────
    for i in $(seq 0 $((INSTANCES_PER_NODE - 1))); do
        PORT=$((BASE_PORT + i))
        if ! wait_for_vllm "http://localhost:${PORT}/v1/models" "$VLLM_STARTUP_TIMEOUT"; then
            echo "  FATAL: vLLM instance $i failed to start. Logs:" >&2
            GLOBAL_RANK=$((GLOBAL_RANK_OFFSET + i))
            cat "${TMPDIR}/vllm_rank${GLOBAL_RANK}.log" >&2
            # Kill all vLLM processes before exiting
            for pid in "${VLLM_PIDS[@]}"; do
                kill "$pid" 2>/dev/null || true
            done
            return 1
        fi
    done

    echo "  All $INSTANCES_PER_NODE vLLM instances ready."

    # ── Launch inference clients (one per instance, in parallel) ──────────────
    CLIENT_PIDS=()
    for i in $(seq 0 $((INSTANCES_PER_NODE - 1))); do
        PORT=$((BASE_PORT + i))
        GLOBAL_RANK=$((GLOBAL_RANK_OFFSET + i))

        python src/infer_validation.py \
            --base configs/base.yaml \
            --checkpoint_root "$CHECKPOINT_ROOT" \
            --dataset dataset/dataset.csv \
            --output_dir "$OUTPUT_DIR" \
            --n "$N_EXAMPLES" \
            --variants $VARIANTS \
            --base_url "http://localhost:${PORT}/v1" \
            --rank "$GLOBAL_RANK" \
            --world_size "$WORLD_SIZE" \
            --timestamp "$TIMESTAMP" \
            --max_workers "$MAX_WORKERS" &

        CLIENT_PIDS+=($!)
    done

    # Wait for all clients to finish
    local exit_code=0
    for pid in "${CLIENT_PIDS[@]}"; do
        if ! wait "$pid"; then
            exit_code=1
        fi
    done

    # Cleanup: kill vLLM servers
    echo "  Shutting down vLLM servers..."
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true

    return $exit_code
}
export -f node_worker

# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR MODE — main entry point
# ══════════════════════════════════════════════════════════════════════════════

# Generate unified timestamp for all ranks
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export TIMESTAMP

NNODES="${SLURM_NNODES:-1}"

echo "══════════════════════════════════════════════════════════════"
echo "  vLLM parallel validation inference"
echo "  Nodes     : $NNODES"
echo "  TP size   : $TP_SIZE"
echo "  Model     : $MODEL_ID"
echo "  Examples  : ${N_EXAMPLES:-0} per variant (0 = all)"
echo "  Adapters  : $CHECKPOINT_ROOT"
echo "  Output    : $OUTPUT_DIR"
echo "  Timestamp : $TIMESTAMP"
echo "  Base port : $BASE_PORT"
echo "══════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"

if [ "$NNODES" -gt 1 ]; then
    # Multi-node: use srun to launch node_worker on each node
    srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
        bash -c 'node_worker "$TIMESTAMP"'
else
    # Single-node: run directly (no srun needed)
    node_worker "$TIMESTAMP"
fi

echo ""
echo "All nodes complete. Merging results..."

# Determine how many total instances ran
GPUS_ON_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
TOTAL_INSTANCES=$((NNODES * (GPUS_ON_NODE / TP_SIZE)))
if [ "$TOTAL_INSTANCES" -lt 1 ]; then
    TOTAL_INSTANCES=1
fi

# Only merge if there were multiple ranks
if [ "$TOTAL_INSTANCES" -gt 1 ]; then
    python src/infer_validation.py --merge \
        --output_dir "$OUTPUT_DIR" \
        --timestamp "$TIMESTAMP" \
        --variants $VARIANTS \
        --cleanup
fi

echo ""
echo "Done. JSONL files written to $OUTPUT_DIR/"
