"""infer_validation.py — quick qualitative evaluation on the first N validation examples.

For each of the three fine-tuned variants (zero_shot, few_shot, cot) this script:
  1. Loads the saved LoRA adapter from ``saved_models/checkpoint/<variant>/``.
  2. Runs greedy generation on the first ``--n`` validation rows.
  3. Writes one JSONL file per variant to ``<output_dir>/<variant>_validation.jsonl``.

Each JSONL line contains:
  {
    "question_id": <int>,
    "question":    <str>,
    "options":     {"A": ..., "B": ..., "C": ..., "D": ...},
    "gold_label":  "A" | "B" | "C" | "D",
    "gold_index":  0-3,
    "prompt":      <str>,          # full prompt fed to the model
    "raw_output":  <str>,          # verbatim model generation
    "pred_label":  "A"|"B"|"C"|"D"|null,
    "pred_index":  0-3 | null,
    "correct":     true | false
  }

Usage (single GPU):
    python scripts/infer_validation.py \\
        --base configs/base.yaml \\
        --checkpoint_root saved_models/checkpoint \\
        --output_dir validation_outputs \\
        --n 10

The script is intentionally single-process / single-GPU so it is cheap to run
as a quick sanity-check before a full evaluation job.
"""

import argparse
import copy
import json
import os
import sys
from typing import Any

import pandas as pd
import torch
import yaml
from peft import PeftModel
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Path helpers ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
# Allow ``from src.xxx import ...`` when run directly.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import OPTION_LABELS, extract_answer_from_token_ids, format_prompt, get_option_token_ids  # noqa: E402




# ── Config helpers ─────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _build_config(base_path: str, strategy: str) -> dict[str, Any]:
    """Load base config and merge the experiment-specific override for *strategy*."""
    base = _load_yaml(base_path)
    override_path = os.path.join(os.path.dirname(base_path), f"{strategy}.yaml")
    if os.path.exists(override_path):
        base = _deep_merge(base, _load_yaml(override_path))
    return base


# ── Few-shot example builder (mirrors main.py) ────────────────────────────────

def _build_few_shot_examples(train_df: pd.DataFrame, n: int) -> list[pd.Series]:
    import random
    random.seed(42)
    indices = random.sample(range(len(train_df)), min(n, len(train_df)))
    return [train_df.iloc[i] for i in indices]


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model(model_id: str, adapter_path: str, use_4bit: bool = False):
    """Load the base causal-LM and attach a saved LoRA adapter (inference only)."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        if use_4bit
        else None
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if use_4bit else None,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
    )

    model: PeftModel = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    if not use_4bit:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    return model, tokenizer


# ── Per-variant inference ─────────────────────────────────────────────────────

def _run_variant(
    strategy: str,
    base_config_path: str,
    checkpoint_root: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n: int,
    output_dir: str,
    max_new_tokens_override: int | None,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Variant: {strategy}  (first {n} validation examples)")
    print(f"{'=' * 60}")

    cfg = _build_config(base_config_path, strategy)
    model_cfg = cfg["model"]
    prompt_cfg = cfg["prompting"]
    train_cfg = cfg["training"]

    adapter_path = os.path.join(checkpoint_root, strategy)
    if not os.path.isdir(adapter_path):
        print(f"  [WARN] Adapter not found at {adapter_path!r} — skipping variant.")
        return

    model_id: str = str(model_cfg["model_id"])
    use_4bit: bool = bool(model_cfg.get("use_4bit", False))
    max_new_tokens: int = (
        max_new_tokens_override
        if max_new_tokens_override is not None
        else int(train_cfg.get("max_new_tokens", 256))
    )

    print(f"  Loading adapter from {adapter_path!r} …")
    model, tokenizer = _load_model(model_id, adapter_path, use_4bit=use_4bit)
    device = next(model.parameters()).device
    print(f"  Model on device: {device}")

    # Few-shot examples (only needed for few_shot strategy)
    few_shot_examples: list[pd.Series] | None = None
    if strategy == "few_shot":
        n_shots: int = int(prompt_cfg.get("num_few_shot_examples", 4))
        few_shot_examples = _build_few_shot_examples(train_df, n_shots)

    subset = val_df.head(n).reset_index(drop=True)

    # Option token ID variants — identical to _evaluate in train.py.
    option_ids_per_label: list[list[int]] = get_option_token_ids(tokenizer)
    pad_id: int = tokenizer.pad_token_id or tokenizer.eos_token_id

    results: list[dict] = []
    tokenizer.padding_side = "left"

    with torch.no_grad():
        for _, row in subset.iterrows():
            prompt: str = format_prompt(row, prompt_cfg, examples=few_shot_examples)

            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=int(train_cfg.get("max_length", 512)),
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            prompt_len: int = input_ids.shape[1]

            # Greedy generation — identical to _evaluate in train.py.
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            new_token_ids: list[int] = gen_ids[0, prompt_len:].tolist()
            raw_output: str = tokenizer.decode(new_token_ids, skip_special_tokens=True)

            # Reverse scan new tokens; fall back to full sequence if not found.
            pred_index: int | None = extract_answer_from_token_ids(
                new_token_ids, option_ids_per_label
            )
            if pred_index is None:
                pred_index = extract_answer_from_token_ids(
                    gen_ids[0].tolist(), option_ids_per_label
                )
            if pred_index is None:
                pred_index = 0  # last-resort fallback
            gold_index: int = int(row["ans"])

            results.append({
                "question_id": int(row["question_id"]),
                "question": str(row["question"]),
                "options": {
                    "A": str(row["opa"]),
                    "B": str(row["opb"]),
                    "C": str(row["opc"]),
                    "D": str(row["opd"]),
                },
                "gold_label": OPTION_LABELS[gold_index],
                "gold_index": gold_index,
                "prompt": prompt,
                "raw_output": raw_output,
                "pred_label": OPTION_LABELS[pred_index],
                "pred_index": pred_index,
                "correct": pred_index == gold_index,
            })

            status = "✓" if results[-1]["correct"] else "✗"
            label_str = results[-1]["pred_label"]
            print(f"  [{status}] Q{results[-1]['question_id']:>4}  pred={label_str}  gold={results[-1]['gold_label']}")

    # ── Write JSONL ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{strategy}_validation.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in results:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    correct = sum(r["correct"] for r in results)
    print(f"\n  Accuracy on first {n}: {correct}/{n} = {correct / n:.1%}")
    print(f"  Output written → {out_path}")

    # Free GPU memory before loading the next variant
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick validation-set inference for zero_shot / few_shot / cot adapters."
    )
    parser.add_argument(
        "--base",
        default="configs/base.yaml",
        help="Path to base YAML config (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="saved_models/checkpoint",
        help="Directory containing zero_shot/, few_shot/, cot/ adapter subdirs "
             "(default: saved_models/checkpoint)",
    )
    parser.add_argument(
        "--dataset",
        default="dataset/dataset.csv",
        help="Path to labelled dataset CSV (default: dataset/dataset.csv)",
    )
    parser.add_argument(
        "--output_dir",
        default="validation_outputs",
        help="Directory to write JSONL files into (default: validation_outputs)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of validation examples to run (default: 10)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for generation (uses config value if omitted)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["zero_shot", "few_shot", "cot"],
        choices=["zero_shot", "few_shot", "cot"],
        help="Which variants to run (default: all three)",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root so the script can be invoked from any cwd.
    os.chdir(REPO_ROOT)

    # ── Load and split dataset ────────────────────────────────────────────────
    base_cfg = _load_yaml(args.base)
    data_cfg = base_cfg["data"]

    df = pd.read_csv(args.dataset)
    train_df, val_df = train_test_split(
        df,
        test_size=float(data_cfg["val_ratio"]),
        random_state=int(data_cfg["seed"]),
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Dataset split → train={len(train_df)}  val={len(val_df)}")
    print(f"Running first {args.n} validation examples for variants: {args.variants}")

    for variant in args.variants:
        _run_variant(
            strategy=variant,
            base_config_path=args.base,
            checkpoint_root=args.checkpoint_root,
            train_df=train_df,
            val_df=val_df,
            n=args.n,
            output_dir=args.output_dir,
            max_new_tokens_override=args.max_new_tokens,
        )

    print("\nAll variants done.")


if __name__ == "__main__":
    main()
