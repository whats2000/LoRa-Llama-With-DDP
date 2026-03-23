"""evaluate.py — inference, accuracy computation, and visualization."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

from src.data import OPTION_LABELS, extract_answer_from_token_ids, format_prompt, get_option_token_ids




def predict(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    df: pd.DataFrame,
    prompt_cfg: dict[str, object],
    max_length: int = 512,
    max_new_tokens: int = 256,
    batch_size: int = 16,
    accelerator: "Accelerator | None" = None,
    return_details: bool = False,
    few_shot_examples: list | None = None,
) -> list[int] | list[dict]:
    """Predict option-index for every row in *df*.

    Uses the **identical** method as ``_evaluate`` in train.py: greedy
    generation from the prompt followed by a reverse scan of the generated
    token IDs for the last A/B/C/D option token (with and without a leading
    space).  This ensures benchmark inference and training validation are
    fully equivalent.

    Accepts an optional shared :class:`~accelerate.Accelerator` so that the
    same process-group used for training is reused — this avoids a second
    ``torch.distributed.init_process_group`` call that deadlocks in multi-node
    NCCL environments.  If *accelerator* is ``None`` a new one is created.

    Args:
        model: Fine-tuned causal-LM (LoRA or merged).
        tokenizer: Tokenizer paired with *model*.
        df: DataFrame containing question and option columns.
        prompt_cfg: The ``prompting`` sub-dict from ``config.yaml``.
        max_length: Maximum token length for the input prompt (truncation).
        max_new_tokens: Maximum tokens to generate per example.
        batch_size: Number of examples per forward pass.
        accelerator: Shared accelerator instance; created internally if None.
        return_details: When True returns dicts with ``pred``, ``prompt``,
            and ``raw_output`` (verbatim decoded generation).

    Returns:
        On rank 0: a full list of predicted indices (0–3) for the whole *df*
        when ``return_details=False`` (default). When ``return_details=True``,
        returns a list of dicts.
        On other ranks: an empty list.
    """
    from accelerate import Accelerator as _Accelerator  # local import avoids circular

    if accelerator is None:
        accelerator = _Accelerator(mixed_precision="bf16")

    # Barrier: ensure every rank enters predict() together before any GPU work,
    # preventing DDP finalizer collectives from mixing with inference operations.
    accelerator.wait_for_everyone()

    # Ensure model is on this rank's device and in eval mode.
    model = model.to(accelerator.device)
    model.eval()

    # Option token ID variants — identical to _evaluate in train.py.
    option_ids_per_label: list[list[int]] = get_option_token_ids(tokenizer)
    pad_id: int = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Shard the dataframe rows across ranks.
    num_processes: int = accelerator.num_processes
    rank: int = accelerator.process_index
    rows = [row for _, row in df.iterrows()]
    local_rows = rows[rank::num_processes]  # every nth row starting at rank

    local_predictions: list[dict] = []

    # Left-padding so batch positions align at the right end for causal LMs.
    orig_padding_side: str = tokenizer.padding_side
    tokenizer.padding_side = "left"

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(local_rows), batch_size),
            desc="Predicting",
            disable=not accelerator.is_local_main_process,
        ):
            batch_rows = local_rows[batch_start : batch_start + batch_size]
            actual_batch: int = len(batch_rows)

            # Global indices in the original dataframe (for final re-ordering).
            batch_global_idxs: list[int] = [
                rank + (batch_start + i) * num_processes for i in range(actual_batch)
            ]

            prompts: list[str] = [
                format_prompt(row, prompt_cfg, few_shot_examples) for row in batch_rows
            ]
            encoding = tokenizer(
                prompts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids: Tensor = encoding["input_ids"].to(accelerator.device)
            attention_mask: Tensor = encoding["attention_mask"].to(accelerator.device)
            prompt_len: int = input_ids.shape[1]

            # Greedy generation — identical to _evaluate in train.py.
            gen_ids: Tensor = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for i, (global_idx, prompt) in enumerate(zip(batch_global_idxs, prompts)):
                new_token_ids: list[int] = gen_ids[i, prompt_len:].tolist()
                # Reverse scan new tokens; fall back to full sequence if not found.
                pred_idx: int | None = extract_answer_from_token_ids(
                    new_token_ids, option_ids_per_label
                )
                if pred_idx is None:
                    pred_idx = extract_answer_from_token_ids(
                        gen_ids[i].tolist(), option_ids_per_label
                    )
                if pred_idx is None:
                    pred_idx = 0  # last-resort fallback
                raw_output: str = tokenizer.decode(
                    new_token_ids, skip_special_tokens=True
                )
                local_predictions.append({
                    "idx": global_idx,
                    "pred": pred_idx,
                    "raw_output": raw_output,
                    "prompt": prompt,
                })

    tokenizer.padding_side = orig_padding_side

    # Barrier: all ranks must finish before gather_object.
    accelerator.wait_for_everyone()

    # Gather all per-rank results to rank 0
    all_predictions: list[dict[str, int]] = gather_object(local_predictions)

    if accelerator.is_main_process:
        all_predictions.sort(key=lambda x: x["idx"])
        if return_details:
            return [{"pred": p["pred"], "raw_output": p["raw_output"], "prompt": p["prompt"]} for p in all_predictions]
        return [p["pred"] for p in all_predictions]
    return []


def compute_accuracy(predictions: list[int], ground_truth: list[int]) -> float:
    """Compute exact-match accuracy between predicted and gold label indices.

    Args:
        predictions: List of predicted option indices (0-based).
        ground_truth: List of gold answer indices (0-based integers).

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    if not predictions:
        return 0.0
    correct: int = sum(pred == gold for pred, gold in zip(predictions, ground_truth))
    return correct / len(predictions)


def plot_training_history(history: dict[str, list[float]], save_dir: str) -> None:
    """Plot per-epoch training / validation loss and accuracy and save to disk.

    Args:
        history: Dict returned by :func:`~src.train.train` containing
            ``train_loss``, ``val_loss``, and ``val_accuracy`` lists.
        save_dir: Directory where the PNG plots will be written.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs: list[int] = list(range(1, len(history["train_loss"]) + 1))

    # ── Loss ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    ax.plot(epochs, history["val_loss"], marker="s", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True)
    loss_path: str = os.path.join(save_dir, "loss_curve.png")
    fig.savefig(loss_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved → {loss_path}")

    # ── Accuracy ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_accuracy"], marker="s", color="tab:orange", label="Val accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True)
    acc_path: str = os.path.join(save_dir, "accuracy_curve.png")
    fig.savefig(acc_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Accuracy curve saved → {acc_path}")
