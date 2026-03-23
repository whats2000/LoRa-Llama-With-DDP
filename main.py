"""main.py — entry point; all behavior is driven by a config in configs/."""

import argparse
import copy
import hashlib
import json
import os

from accelerate import Accelerator
import torch
import pandas as pd
import yaml

from src.data import OPTION_LABELS, QADataset, generate_cot_rationales, load_benchmark, load_datasets
from src.evaluate import plot_training_history, predict
from src.model import load_model_and_tokenizer
from src.train import train


def load_config(path: str) -> dict[str, object]:
    """Load and return a YAML configuration file.

    Args:
        path: File-system path to the YAML config.

    Returns:
        Parsed config as a nested dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(
    base: dict[str, object],
    override: dict[str, object],
) -> dict[str, object]:
    """Recursively merge *override* into a deep copy of *base*.

    Nested dicts are merged; all other values are replaced by *override*.

    Args:
        base: The base configuration dict.
        override: Values to overlay on top of *base*.

    Returns:
        A new merged dict; *base* is not mutated.
    """
    result: dict[str, object] = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)  # type: ignore[arg-type]
        else:
            result[key] = val
    return result


def build_few_shot_examples(
    train_df,  # pd.DataFrame — avoid circular import at module level
    prompt_cfg: dict[str, object],
    n: int,
) -> list:
    """
    Sample n rows from train_df to use as few-shot examples.

    Args:
        train_df: Training DataFrame to sample from.
        prompt_cfg: The ``prompting`` sub-dict from config.
        n: Number of examples to sample.

    Returns:
        A list of :class:`pd.Series` rows.
    """
    import random  # noqa: PLC0415

    seed: int = 42
    random.seed(seed)
    indices = random.sample(range(len(train_df)), min(n, len(train_df)))
    return [train_df.iloc[i] for i in indices]


def main() -> None:
    """Parse arguments, load config, run training and inference."""
    parser = argparse.ArgumentParser(description="HW1 – PathoQA Question Answering")
    parser.add_argument(
        "--base",
        default="configs/base.yaml",
        help="Base YAML config with shared settings (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML config that overrides base (e.g. configs/cot.yaml)",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and load the saved adapter from paths.saved_models for inference only.",
    )
    args = parser.parse_args()

    cfg: dict[str, object] = load_config(args.base)
    if args.config:
        cfg = deep_merge(cfg, load_config(args.config))
    paths_cfg = dict(cfg["paths"])  # type: ignore[arg-type]
    data_cfg = dict(cfg["data"])  # type: ignore[arg-type]
    model_cfg = dict(cfg["model"])  # type: ignore[arg-type]
    lora_cfg = dict(cfg["lora"])  # type: ignore[arg-type]
    train_cfg = dict(cfg["training"])  # type: ignore[arg-type]
    prompt_cfg = dict(cfg["prompting"])  # type: ignore[arg-type]
    inference_cfg = dict(cfg.get("inference", {}))  # type: ignore[arg-type]

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading datasets…")
    train_df, val_df = load_datasets(data_cfg, str(paths_cfg["dataset"]))
    benchmark_df = load_benchmark(str(paths_cfg["benchmark"]))
    print(f"  train={len(train_df)}  val={len(val_df)}  benchmark={len(benchmark_df)}")

    few_shot_examples: list | None = None
    if str(prompt_cfg["strategy"]) == "few_shot":
        n_shots: int = int(prompt_cfg.get("num_few_shot_examples", 4))  # type: ignore[arg-type]
        few_shot_examples = build_few_shot_examples(train_df, prompt_cfg, n_shots)

    max_length: int = int(train_cfg["max_length"])  # type: ignore[arg-type]

    # ── Accelerator (single instance for the whole job) ───────────────────────
    grad_accumulation: int = int(train_cfg["grad_accumulation_steps"])  # type: ignore[arg-type]
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accumulation,
        mixed_precision="bf16",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model and tokenizer…")
    if args.predict_only:
        # Load base model + saved LoRA adapter directly; skip all training.
        from peft import PeftModel as _PeftModel
        from transformers import AutoTokenizer as _AutoTokenizer, AutoModelForCausalLM as _AutoModelForCausalLM
        adapter_path = str(paths_cfg["saved_models"])
        print(f"  predict-only mode: loading adapter from {adapter_path!r}")
        tokenizer = _AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = _AutoModelForCausalLM.from_pretrained(
            str(model_cfg["model_id"]),
            torch_dtype=torch.bfloat16,
        )
        trained_model = _PeftModel.from_pretrained(base, adapter_path)
        trained_model.eval()
        trained_model = trained_model.to(accelerator.device)
    else:
        model, tokenizer = load_model_and_tokenizer(model_cfg, lora_cfg)

    # ── Datasets ──────────────────────────────────────────────────────────────
    rationales_train: list[str] | None = None
    if not args.predict_only and str(prompt_cfg["strategy"]) == "cot":
        cot_cfg = dict(prompt_cfg.get("cot", {}))  # type: ignore[arg-type]
        if bool(cot_cfg.get("use_rationale_distillation", False)):
            rationale_max_new_tokens: int = int(cot_cfg.get("rationale_max_new_tokens", 200))  # type: ignore[arg-type]

            # Build a stable cache key from the inputs that determine rationale content.
            _cache_key = hashlib.md5(
                json.dumps({
                    "model_id": model_cfg.get("model_id"),
                    "seed": data_cfg.get("seed"),
                    "val_ratio": data_cfg.get("val_ratio"),
                    "rationale_prompt": str(cot_cfg.get("rationale_prompt", "")),
                    "max_new_tokens": rationale_max_new_tokens,
                    "max_length": max_length,
                    "repetition_penalty": 1.3,
                    "no_repeat_ngram_size": 8,
                }, sort_keys=True).encode()
            ).hexdigest()[:12]
            _cache_dir = str(paths_cfg["saved_models"])
            _cache_file = os.path.join(_cache_dir, f"rationale_cache_{_cache_key}.json")

            if os.path.exists(_cache_file):
                if accelerator.is_main_process:
                    print(f"Loading cached CoT rationales from {_cache_file}…")
                with open(_cache_file) as _f:
                    _cached = json.load(_f)
                rationales_train = _cached["train"]
            else:
                print("Generating CoT rationales for training set…")
                rationales_train = generate_cot_rationales(
                    model, tokenizer, train_df, prompt_cfg,
                    max_length=max_length,
                    max_new_tokens=rationale_max_new_tokens,
                    batch_size=int(inference_cfg.get("batch_size", 16)),
                    accelerator=accelerator,
                )
                if accelerator.is_main_process:
                    os.makedirs(_cache_dir, exist_ok=True)
                    with open(_cache_file, "w") as _f:
                        json.dump({"train": rationales_train}, _f)
                    print(f"CoT rationales cached → {_cache_file}")

    train_dataset = QADataset(train_df, tokenizer, prompt_cfg, max_length, few_shot_examples, rationales_train)
    val_dataset = QADataset(val_df, tokenizer, prompt_cfg, max_length, few_shot_examples, None)

    # ── Training ──────────────────────────────────────────────────────────────
    if args.predict_only:
        print("Skipping training (--predict-only)…")
    else:
        print("Starting training…")
        history, trained_model = train(
            model, tokenizer, train_dataset, val_dataset, train_cfg, paths_cfg, accelerator
        )

        # ── Plots (main process only — all ranks would corrupt the file) ──────────
        if accelerator.is_main_process:
            plot_training_history(history, str(paths_cfg["saved_models"]))

        # ── Sync all ranks before switching from training to inference ────────────
        # Wait for every rank to finish training + plotting, then free the DDP
        # wrapper references held by the accelerator so NCCL does not try to
        # execute leftover collective ops when inference starts.
        accelerator.wait_for_everyone()
        accelerator.free_memory()
        torch.cuda.empty_cache()
        # Move the unwrapped model to this rank's device explicitly.
        trained_model = trained_model.to(accelerator.device)

    # ── Benchmark inference ───────────────────────────────────────────────────
    max_length: int = int(train_cfg.get("max_length", 512))  # type: ignore[arg-type]
    max_new_tokens: int = int(train_cfg.get("max_new_tokens", 256))  # type: ignore[arg-type]
    inference_batch_size: int = int(inference_cfg.get("batch_size", 16))  # type: ignore[arg-type]

    print("Running inference on benchmark set…")
    benchmark_details = predict(
        trained_model,
        tokenizer,
        benchmark_df,
        prompt_cfg,
        max_length,
        max_new_tokens,
        inference_batch_size,
        accelerator,
        return_details=True,
        few_shot_examples=few_shot_examples,
    )
    if benchmark_details:
        benchmark_predictions: list[int] = [d["pred"] for d in benchmark_details]
        output_path: str = str(paths_cfg["output_csv"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(
            {"question_id": benchmark_df["question_id"].to_numpy(), "pred": benchmark_predictions}
        ).to_csv(output_path, index=False)
        print(f"Benchmark predictions saved → {output_path}")

        # ── Save raw model outputs as JSONL for qualitative analysis ──────────
        jsonl_path: str = output_path.replace(".csv", "_details.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as _jf:
            for row, detail in zip(benchmark_df.itertuples(index=False), benchmark_details):
                _jf.write(
                    json.dumps(
                        {
                            "question_id": row.question_id,
                            "question": row.question,
                            "options": {
                                "A": row.opa,
                                "B": row.opb,
                                "C": row.opc,
                                "D": row.opd,
                            },
                            "pred_label": OPTION_LABELS[detail["pred"]],
                            "pred_index": detail["pred"],
                            "raw_output": detail["raw_output"],
                            "prompt": detail["prompt"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Benchmark raw outputs saved → {jsonl_path}")


if __name__ == "__main__":
    main()
