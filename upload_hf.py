"""upload_hf.py — publish all trained LoRA/DoRA adapters to the HF Hub.

Public model repo so the course teacher can download/reproduce. Uploads every
experiments/**/saved_models/ dir (the 28 ablation adapters). Resumable.
"""
import os
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

TOKEN = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
REPO = "whats2000/lora-llama-pathoqa-checkpoints"

README = r"""---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B-Instruct
library_name: peft
tags: [lora, dora, medical-qa, multiple-choice, pathology, ablation-study]
---

# LoRA-Llama PathoQA — all trained adapters (ablation study)

LoRA / DoRA adapters for **`meta-llama/Llama-3.2-1B-Instruct`** fine-tuned on
**PathoQA** (4-option medical pathology MCQA). This repo holds **every adapter**
from a 10-experiment ablation study (base model fixed at 1B; the study is about
*method*). Full lab journal and code: see the course submission package
(`report.ipynb` + `experiments/NOTES.md`).

Metric = Kaggle `hw-1-question-answering` test accuracy (public == private).

## Best pipeline — 0.7988 (5-member option-likelihood ensemble)

Uniform-average the per-option probabilities of these 5 adapters (all rank 256,
effective-batch 192):

| role | path |
|---|---|
| zero-shot r256 | `experiments/exp06_zeroshot_recipe/e06c/saved_models` |
| zero-shot r192 | `experiments/exp07_eff192_ensemble/e07a/saved_models` |
| few-shot 2-shot | `experiments/exp08_fewshot_diversity/e08b/saved_models` |
| few-shot 4-shot | `experiments/exp07_eff192_ensemble/e07b/saved_models` |
| few-shot 8-shot | `experiments/exp08_fewshot_diversity/e08a/saved_models` |

Progression: baseline 0.7700 → rank=256 0.7766 → +ensemble 0.7811 → +few-shot
0.7888 → @eff192 0.7922 → +multi-shot **0.7988**.

## Load an adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

base_id = "meta-llama/Llama-3.2-1B-Instruct"
local = snapshot_download("whats2000/lora-llama-pathoqa-checkpoints",
                          allow_patterns="experiments/exp06_zeroshot_recipe/e06c/saved_models/*")
adapter = f"{local}/experiments/exp06_zeroshot_recipe/e06c/saved_models"

tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, adapter).eval()
```

## All adapters (organised by experiment)

- `exp01_lora_rank/e01a..e` — LoRA rank sweep r∈{8,16,32,64,128} (zero-shot)
- `exp02_rank_scaling/e02a..e` — rank push r∈{128,192,256,384,512}
- `exp04_strategy_diverse/e04a` — few-shot @ eff768
- `exp05_dora_variant/e05a..c` — DoRA r∈{128,192,256}
- `exp06_zeroshot_recipe/e06a..f` — LR / effective-batch / loss recipe
- `exp07_eff192_ensemble/e07a,b` — zero/few-shot retrained @ eff-batch 192
- `exp08_fewshot_diversity/e08a,b` — few-shot 8-shot / 2-shot @ eff192
- `exp09_more_fewshot/e09a..c` — few-shot 1/3/6-shot @ eff192

Each dir is a standard PEFT adapter (`adapter_config.json` +
`adapter_model.safetensors`) plus its `training_history.json` and loss/accuracy
curves.
"""

api = HfApi(token=TOKEN)
print("Creating repo (public)…")
create_repo(REPO, repo_type="model", private=False, exist_ok=True, token=TOKEN)
print("Uploading README…")
upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md",
            repo_id=REPO, repo_type="model", token=TOKEN,
            commit_message="Add model card")
print("Uploading all adapters (experiments/**/saved_models/**)… this is ~17GB.")
upload_folder(repo_id=REPO, repo_type="model", folder_path="experiments",
              allow_patterns=["**/saved_models/**"], token=TOKEN,
              commit_message="Upload all 28 LoRA/DoRA adapters (PathoQA ablation)")
print("UPLOAD COMPLETE:", f"https://huggingface.co/{REPO}")
