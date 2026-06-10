"""main.py — exp03 cell (copy of root main.py, repurposed for inference).

Option-likelihood scoring + multi-adapter ensemble. NO training: loads one or
more already-trained LoRA adapters, scores the 4 answer letters (A/B/C/D) by the
first-token log-probability after the prompt, renormalises over the 4 options,
and averages those probabilities across adapters. Reports val accuracy (gold is
available) as an offline correctness check, and writes the benchmark submission.

Reuses root helpers from src.data unmodified (format_prompt, OPTION_LABELS,
load_datasets, load_benchmark); the new LL-scoring/ensemble logic lives here.
"""

import argparse
import copy
import json
import os
import sys

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# exp03 cell isolation: make repo root importable for src.* (no cell shadows here).
_CELL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CELL_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from src.data import OPTION_LABELS, format_prompt, load_benchmark, load_datasets


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_cfg(base_path: str, override_path: str | None) -> dict:
    cfg = yaml.safe_load(open(base_path))
    if override_path:
        cfg = _deep_merge(cfg, yaml.safe_load(open(override_path)))
    return cfg


@torch.no_grad()
def score_adapter(
    base_id: str,
    adapter_path: str,
    tokenizer,
    prompts: list[str],
    option_first_ids: list[int],
    device: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Return an (N, 4) tensor of per-option probabilities for one adapter.

    For each prompt, take the model logits at the final position, log-softmax
    over the vocabulary, read off the 4 option-letter first-token log-probs, and
    softmax over just those 4 so the row sums to 1 (a proper categorical over
    A/B/C/D). This is standard first-token MCQA scoring.
    """
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()

    tokenizer.padding_side = "left"
    opt_ids = torch.tensor(option_first_ids, device=device)
    n = len(prompts)
    probs = torch.zeros(n, 4)
    for s in range(0, n, batch_size):
        chunk = prompts[s : s + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        logits = model(**enc).logits[:, -1, :].float()  # (b, vocab)
        lp = torch.log_softmax(logits, dim=-1)
        opt_lp = lp[:, opt_ids]                          # (b, 4)
        probs[s : s + len(chunk)] = torch.softmax(opt_lp, dim=-1).cpu()

    del model, base
    torch.cuda.empty_cache()
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description="exp03 — option-LL ensemble inference")
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = _load_cfg(args.base, args.config)
    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    prompt_cfg = cfg["prompting"]
    ens_cfg = cfg["ensemble"]

    adapters: list[str] = list(ens_cfg["adapters"])
    base_id: str = str(model_cfg["model_id"])
    max_length: int = int(train_cfg.get("max_length", 512))
    batch_size: int = int(cfg.get("inference", {}).get("batch_size", 32))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # First-token ids for the space-prefixed answer letters (the form the model
    # emits after "Answer:" — matches the training target " {label}").
    option_first_ids = [tokenizer.encode(f" {L}", add_special_tokens=False)[0] for L in OPTION_LABELS]
    print(f"adapters={adapters}")
    print(f"option_first_ids={option_first_ids} -> "
          f"{[tokenizer.decode([i]) for i in option_first_ids]!r}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df = load_datasets(data_cfg, str(paths_cfg["dataset"]))
    benchmark_df = load_benchmark(str(paths_cfg["benchmark"]))
    val_prompts = [format_prompt(r, prompt_cfg) for _, r in val_df.iterrows()]
    bench_prompts = [format_prompt(r, prompt_cfg) for _, r in benchmark_df.iterrows()]

    # ── Score every adapter on val + benchmark; average probabilities ─────────
    val_acc_each: dict[str, float] = {}
    val_probs_sum = torch.zeros(len(val_df), 4)
    bench_probs_sum = torch.zeros(len(benchmark_df), 4)
    val_gold = torch.tensor([int(a) for a in val_df["ans"].tolist()])

    for ap in adapters:
        vp = score_adapter(base_id, ap, tokenizer, val_prompts, option_first_ids, device, batch_size, max_length)
        bp = score_adapter(base_id, ap, tokenizer, bench_prompts, option_first_ids, device, batch_size, max_length)
        acc = (vp.argmax(1) == val_gold).float().mean().item()
        val_acc_each[ap] = acc
        print(f"  [{os.path.basename(os.path.dirname(ap))}] single-adapter val LL-acc = {acc:.4f}")
        val_probs_sum += vp
        bench_probs_sum += bp

    n = len(adapters)
    val_ens = val_probs_sum / n
    bench_ens = bench_probs_sum / n
    ens_val_acc = (val_ens.argmax(1) == val_gold).float().mean().item()
    print(f"\nENSEMBLE ({n} adapters) val LL-acc = {ens_val_acc:.4f}")

    # ── Write submission + details ────────────────────────────────────────────
    out_csv = str(paths_cfg["output_csv"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    preds = bench_ens.argmax(1).tolist()
    pd.DataFrame(
        {"question_id": benchmark_df["question_id"].to_numpy(), "pred": preds}
    ).to_csv(out_csv, index=False)
    print(f"Submission written → {out_csv}")

    summary = {
        "adapters": adapters,
        "val_acc_each": val_acc_each,
        "ensemble_val_acc": ens_val_acc,
        "n_benchmark": len(benchmark_df),
    }
    with open(out_csv.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
