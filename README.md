# HW1: Question Answering

Fine-tuning `meta-llama/Llama-3.2-1B-Instruct` with LoRA on the **PathoQA** dataset for medical multiple-choice question answering.

---

## Setup

### Option A — Standard `venv` (always available)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Option B — `uv` (faster, recommended)

```bash
# 1. Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the virtual environment and install all dependencies
uv sync

# 3. Activate (optional — uv run will use it automatically)
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

> **After adding or changing dependencies** re-export the pinned file:
> ```bash
> uv sync && uv export --no-hashes -o requirements.txt
> ```

---

## Configuration

All hyperparameters, prompt templates, and file paths live in **`configs/`** — no hard-coded strings in the source code.

Each experiment config **overrides only what differs** from `configs/base.yaml`; the two are deep-merged at runtime.

| File | Experiment |
|------|------------|
| `configs/base.yaml` | Shared settings (model, LoRA, training, prompt text) |
| `configs/zero_shot.yaml` | Zero-shot prompting |
| `configs/few_shot.yaml` | Few-shot prompting (4 examples) |
| `configs/cot.yaml` | Chain-of-Thought prompting |

Key sections in `base.yaml`:

| Section | Purpose |
|---------|----------|
| `paths` | Dataset, checkpoint, and output CSV paths |
| `data` | Train/val split ratio and random seed |
| `model` | HuggingFace model ID and quantization toggle |
| `lora` | LoRA rank, alpha, target modules, dropout |
| `training` | Epochs, batch size, learning rate, gradient accumulation |
| `prompting` | Strategy and all prompt text templates |

---

## Running

```bash
# Run a single experiment (zero-shot, few-shot, or CoT)
python main.py --config configs/zero_shot.yaml
python main.py --config configs/few_shot.yaml
python main.py --config configs/cot.yaml

# With uv (no manual activation needed)
uv run main.py --config configs/zero_shot.yaml
uv run main.py --config configs/few_shot.yaml
uv run main.py --config configs/cot.yaml
```

> [!TIP]
> Effective batch size with 2 nodes × 8 GPUs (16 total): **768** for all strategies.
> zero_shot/few_shot: 48 per-GPU (VRAM bound at 512 tokens) × grad_accum 1 × 16 = 768. CoT: 24 per-GPU × grad_accum 2 × 16 = 768 (smaller per-GPU batch to fit 1024-token context).
> Edit `configs/base.yaml` to adjust `training.batch_size` and `training.grad_accumulation_steps` for your hardware.

---

## Reproducing with Slurm

A ready-made Slurm job script is provided at `scripts/run_experience_slurm.sh`. 
It runs all three experiments (zero-shot → few-shot → CoT) sequentially on one node with 8 GPUs using `accelerate launch`.

```bash
# The experiment is run at nano4.nchc.org.tw, but the script can be adapted for any Slurm cluster with similar resource availability.
sbatch --account=<USER_OR_PROJECT_ID> scripts/run_experience_slurm.sh
```

Replace `<USER_OR_PROJECT_ID>` with your cluster account name (e.g. your user ID or compute-project ID).

The script defaults to:
- Partition: `dev`
- 8 GPUs per node (`--gres=gpu:8`)
- 64 CPUs, all available memory
- Wall-time: 2 hours
- Logs written to `logs/slurm_<JOBID>.out` / `.err`

Adjust `#SBATCH` directives inside the script if your cluster uses different partition names or resource limits.

---

## Project Structure

```
HW1_{student_id}/
├── dataset/
│   ├── dataset.csv          # 9 000 labeled PathoQA examples
│   └── benchmark.csv        # 900 unlabeled Kaggle test questions
├── saved_models/
│   └── checkpoint/
│       ├── cot/             # Best CoT checkpoint (epoch with highest val accuracy)
│       ├── few_shot/        # Best few-shot checkpoint
│       └── zero_shot/       # Best zero-shot checkpoint
├── main.py
├── README.md
├── requirements.txt
│
│   (additional project files — not required in zip but present in repo)
├── configs/
│   ├── base.yaml
│   ├── zero_shot.yaml
│   ├── few_shot.yaml
│   └── cot.yaml
├── pyproject.toml
├── uv.lock
└── src/
    ├── data.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```

---

## Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Quantization | None (bfloat16) |
| LoRA rank (`r`) | 64 |
| LoRA alpha | 128 |
| Target modules | all linear layers |
| Epochs | 10 |
| Batch size (`training.batch_size`) | 48 per GPU; **reduce to 4 on a single consumer GPU** (Total effective batch size: 768) |
| Gradient accumulation (`training.grad_accumulation_steps`) | 2 **increase to 96 if batch size is reduced to 4** (Remaining effective batch size: 768) |
| Learning rate | 1e-4 |
| Prompting strategy | CoT |
