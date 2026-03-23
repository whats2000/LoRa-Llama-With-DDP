"""infer_validation.py — vLLM-based parallel validation-set inference.

For each fine-tuned variant (zero_shot, few_shot, cot) this script:
  1. Connects to a running vLLM server that serves the LoRA adapters.
  2. Shards the first ``--n`` validation rows by ``--rank`` / ``--world_size``.
  3. Sends concurrent requests via ThreadPoolExecutor.
  4. Writes one JSONL file per variant (with rank suffix when world_size > 1).

A separate ``--merge`` mode collects per-rank shards into a single JSONL per
variant and re-computes accuracy.

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

Usage (single vLLM server already running):
    python src/infer_validation.py \\
        --base configs/base.yaml \\
        --checkpoint_root saved_models/checkpoint \\
        --output_dir validation_outputs \\
        --n 30 \\
        --base_url http://localhost:8000/v1

Merge mode (after all ranks finish):
    python src/infer_validation.py --merge \\
        --output_dir validation_outputs \\
        --timestamp 20260323_120000 \\
        --variants zero_shot few_shot cot
"""

import argparse
import copy
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
import yaml
from openai import OpenAI
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Path helpers ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
# Allow ``from src.xxx import ...`` when run directly.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import OPTION_LABELS, extract_answer_from_text, format_prompt  # noqa: E402


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


# ── vLLM API call helper ─────────────────────────────────────────────────────

def _call_vllm(
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_tokens: int,
) -> str:
    """Send a single completion request to the vLLM server and return the text."""
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].text


# ── Per-variant inference ─────────────────────────────────────────────────────

def _run_variant(
    strategy: str,
    client: OpenAI,
    base_config_path: str,
    checkpoint_root: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n: int,
    output_dir: str,
    max_new_tokens_override: int | None,
    rank: int,
    world_size: int,
    timestamp: str | None,
    max_workers: int,
) -> None:
    cfg = _build_config(base_config_path, strategy)
    prompt_cfg = cfg["prompting"]
    train_cfg = cfg["training"]

    adapter_path = os.path.join(checkpoint_root, strategy)
    if not os.path.isdir(adapter_path):
        print(f"  [WARN] Adapter not found at {adapter_path!r} — skipping variant.")
        return

    max_new_tokens: int = (
        max_new_tokens_override
        if max_new_tokens_override is not None
        else int(train_cfg.get("max_new_tokens", 256))
    )

    # Few-shot examples (only needed for few_shot strategy)
    few_shot_examples: list[pd.Series] | None = None
    if strategy == "few_shot":
        n_shots: int = int(prompt_cfg.get("num_few_shot_examples", 4))
        few_shot_examples = _build_few_shot_examples(train_df, n_shots)

    # Take first N (0 means all), then shard by rank
    subset = val_df if n <= 0 else val_df.head(n)
    subset = subset.reset_index(drop=True)
    if world_size > 1:
        chunk_size = (len(subset) + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, len(subset))
        subset = subset.iloc[start_idx:end_idx].reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"  Variant: {strategy}  (rank {rank}/{world_size}, {len(subset)} examples)")
    print(f"{'=' * 60}")

    if len(subset) == 0:
        print("  No examples for this rank — skipping.")
        return

    # The vLLM model name is the adapter name registered at server startup
    model_name = strategy

    # Build all prompts first
    prompts: list[str] = []
    for _, row in subset.iterrows():
        prompts.append(format_prompt(row, prompt_cfg, examples=few_shot_examples))

    # Concurrent API calls
    results: list[dict | None] = [None] * len(subset)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict = {}
        for i, (_, row) in enumerate(subset.iterrows()):
            future = executor.submit(
                _call_vllm, client, model_name, prompts[i], max_new_tokens,
            )
            futures[future] = i

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {strategy} rank={rank}",
        ):
            i = futures[future]
            row = subset.iloc[i]
            raw_output: str = future.result()

            pred_index: int | None = extract_answer_from_text(raw_output)
            if pred_index is None:
                pred_index = 0  # last-resort fallback
            gold_index: int = int(row["ans"])

            results[i] = {
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
                "prompt": prompts[i],
                "raw_output": raw_output,
                "pred_label": OPTION_LABELS[pred_index],
                "pred_index": pred_index,
                "correct": pred_index == gold_index,
            }

            status = "✓" if results[i]["correct"] else "✗"  # type: ignore[index]
            label_str = results[i]["pred_label"]  # type: ignore[index]
            gold_str = results[i]["gold_label"]  # type: ignore[index]
            qid = results[i]["question_id"]  # type: ignore[index]
            print(f"  [{status}] Q{qid:>4}  pred={label_str}  gold={gold_str}")

    # ── Write JSONL ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    if world_size > 1 and timestamp:
        out_path = os.path.join(
            output_dir, f"{strategy}_validation_{timestamp}_rank{rank}.jsonl"
        )
    else:
        out_path = os.path.join(output_dir, f"{strategy}_validation.jsonl")

    with open(out_path, "w", encoding="utf-8") as fh:
        for record in results:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    correct = sum(1 for r in results if r and r["correct"])
    total = len(results)
    print(f"\n  Accuracy: {correct}/{total} = {correct / total:.1%}")
    print(f"  Output written → {out_path}")


# ── Merge mode ────────────────────────────────────────────────────────────────

def _merge_results(
    output_dir: str,
    timestamp: str,
    variants: list[str],
    cleanup: bool = False,
) -> None:
    """Merge per-rank shard files into a single JSONL per variant."""
    print(f"\n{'=' * 60}")
    print(f"  Merging results  (timestamp={timestamp})")
    print(f"{'=' * 60}")

    for variant in variants:
        pattern = os.path.join(
            output_dir, f"{variant}_validation_{timestamp}_rank*.jsonl"
        )
        shard_files = sorted(glob.glob(pattern))
        if not shard_files:
            print(f"  [{variant}] No shard files found for pattern: {pattern}")
            continue

        all_records: list[dict] = []
        for fpath in shard_files:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))

        # Deduplicate by question_id and sort
        seen: set[int] = set()
        unique: list[dict] = []
        all_records.sort(key=lambda r: r["question_id"])
        for r in all_records:
            qid = r["question_id"]
            if qid not in seen:
                seen.add(qid)
                unique.append(r)

        merged_path = os.path.join(output_dir, f"{variant}_validation.jsonl")
        with open(merged_path, "w", encoding="utf-8") as fh:
            for record in unique:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        correct = sum(1 for r in unique if r.get("correct"))
        total = len(unique)
        acc = correct / total if total else 0
        print(
            f"  [{variant}] Merged {len(shard_files)} shards → "
            f"{total} examples, accuracy {correct}/{total} = {acc:.1%}"
        )
        print(f"  Output → {merged_path}")

        if cleanup:
            for fpath in shard_files:
                os.remove(fpath)
            print(f"  Cleaned up {len(shard_files)} shard files.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM-based parallel validation inference for LoRA adapters.",
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
        default=0,
        help="Number of validation examples to run (0 = all, default: 0)",
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
    # ── vLLM / parallel arguments ────────────────────────────────────────────
    parser.add_argument(
        "--base_url",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Global rank of this worker (default: 0)",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Unified timestamp for associating shard files from the same run",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="ThreadPoolExecutor concurrency for API calls (default: 16)",
    )
    # ── Merge mode ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge per-rank shard files instead of running inference",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete shard files after merging",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root so the script can be invoked from any cwd.
    os.chdir(REPO_ROOT)

    # ── Merge mode ───────────────────────────────────────────────────────────
    if args.merge:
        if not args.timestamp:
            parser.error("--merge requires --timestamp")
        _merge_results(
            output_dir=args.output_dir,
            timestamp=args.timestamp,
            variants=args.variants,
            cleanup=args.cleanup,
        )
        return

    # ── Inference mode ───────────────────────────────────────────────────────
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
    n_desc = "all" if args.n <= 0 else str(args.n)
    print(f"Running {n_desc} validation examples for variants: {args.variants}")
    print(f"Rank {args.rank} / world_size {args.world_size}")
    print(f"vLLM endpoint: {args.base_url}")

    client = OpenAI(base_url=args.base_url, api_key="unused")

    for variant in args.variants:
        _run_variant(
            strategy=variant,
            client=client,
            base_config_path=args.base,
            checkpoint_root=args.checkpoint_root,
            train_df=train_df,
            val_df=val_df,
            n=args.n,
            output_dir=args.output_dir,
            max_new_tokens_override=args.max_new_tokens,
            rank=args.rank,
            world_size=args.world_size,
            timestamp=args.timestamp,
            max_workers=args.max_workers,
        )

    print("\nAll variants done.")


if __name__ == "__main__":
    main()
