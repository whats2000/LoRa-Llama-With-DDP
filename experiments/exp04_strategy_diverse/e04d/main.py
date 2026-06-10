"""main.py — exp04 cell (copy of root main.py, repurposed for inference).

Cross-strategy option-likelihood ensemble. Extends the exp03 scorer so each
ensemble *member* carries its own prompt strategy: a zero_shot member is scored
with the zero_shot prompt, a few_shot member with its 4-example prompt (same
seed-42 examples used at training). Per member we take the first-token log-prob
of the 4 answer letters after the prompt, softmax over the 4 options, then
average those probabilities across members. Reports val accuracy (gold) as an
offline check and writes the benchmark submission.

Only LL-scorable strategies (zero_shot, few_shot — both end in "Answer:") are
supported here; CoT is generation-based and excluded (see exp04 NOTES).
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


def _build_few_shot_examples(train_df, n: int) -> list:
    import random
    random.seed(42)  # identical to main.py / training
    idx = random.sample(range(len(train_df)), min(n, len(train_df)))
    return [train_df.iloc[i] for i in idx]


def _prompt_cfg_for(base_prompt_cfg: dict, strategy: str) -> dict:
    cfg = copy.deepcopy(base_prompt_cfg)
    cfg["strategy"] = strategy
    return cfg


@torch.no_grad()
def score_member(base_id, adapter_path, tokenizer, prompts, opt_ids, device, batch_size, max_length):
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"  # keep the END (question + "Answer:") for scoring
    ids = torch.tensor(opt_ids, device=device)
    n = len(prompts)
    probs = torch.zeros(n, 4)
    for s in range(0, n, batch_size):
        chunk = prompts[s : s + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length).to(device)
        logits = model(**enc).logits[:, -1, :].float()
        opt_lp = torch.log_softmax(logits, dim=-1)[:, ids]
        probs[s : s + len(chunk)] = torch.softmax(opt_lp, dim=-1).cpu()
    del model, base
    torch.cuda.empty_cache()
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.base))
    if args.config:
        cfg = _deep_merge(cfg, yaml.safe_load(open(args.config)))

    paths_cfg, data_cfg, model_cfg = cfg["paths"], cfg["data"], cfg["model"]
    base_prompt_cfg = cfg["prompting"]
    max_length = int(cfg["training"].get("max_length", 512))
    batch_size = int(cfg.get("inference", {}).get("batch_size", 16))
    base_id = str(model_cfg["model_id"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    members = list(cfg["ensemble"]["members"])  # [{adapter, strategy}, ...]

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    opt_ids = [tokenizer.encode(f" {L}", add_special_tokens=False)[0] for L in OPTION_LABELS]

    train_df, val_df = load_datasets(data_cfg, str(paths_cfg["dataset"]))
    benchmark_df = load_benchmark(str(paths_cfg["benchmark"]))
    fs_examples = _build_few_shot_examples(train_df, int(base_prompt_cfg.get("num_few_shot_examples", 4)))
    val_gold = torch.tensor([int(a) for a in val_df["ans"].tolist()])

    val_sum = torch.zeros(len(val_df), 4)
    bench_sum = torch.zeros(len(benchmark_df), 4)
    per_member = {}

    for m in members:
        strat = str(m["strategy"])
        pc = _prompt_cfg_for(base_prompt_cfg, strat)
        ex = fs_examples if strat == "few_shot" else None
        val_prompts = [format_prompt(r, pc, ex) for _, r in val_df.iterrows()]
        bench_prompts = [format_prompt(r, pc, ex) for _, r in benchmark_df.iterrows()]
        vp = score_member(base_id, str(m["adapter"]), tokenizer, val_prompts, opt_ids, device, batch_size, max_length)
        bp = score_member(base_id, str(m["adapter"]), tokenizer, bench_prompts, opt_ids, device, batch_size, max_length)
        acc = (vp.argmax(1) == val_gold).float().mean().item()
        tag = f"{strat}:{os.path.basename(os.path.dirname(str(m['adapter'])))}"
        per_member[tag] = acc
        print(f"  [{tag}] single val LL-acc = {acc:.4f}")
        val_sum += vp
        bench_sum += bp

    k = len(members)
    ens_val_acc = ((val_sum / k).argmax(1) == val_gold).float().mean().item()
    print(f"\nENSEMBLE ({k} members) val LL-acc = {ens_val_acc:.4f}")

    out_csv = str(paths_cfg["output_csv"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    preds = (bench_sum / k).argmax(1).tolist()
    pd.DataFrame({"question_id": benchmark_df["question_id"].to_numpy(), "pred": preds}).to_csv(out_csv, index=False)
    summary = {"members": [f"{m['strategy']}:{m['adapter']}" for m in members],
               "per_member_val_acc": per_member, "ensemble_val_acc": ens_val_acc}
    json.dump(summary, open(out_csv.replace(".csv", "_summary.json"), "w"), indent=2)
    print("Submission →", out_csv)
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
