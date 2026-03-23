"""train.py — training loop driven by config."""

import os

import torch
from accelerate import Accelerator
from peft import PeftModel
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data import QADataset, extract_answer_from_token_ids, get_option_token_ids


def train(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: QADataset,
    val_dataset: QADataset,
    train_cfg: dict[str, object],
    paths_cfg: dict[str, object],
    accelerator: Accelerator,
) -> tuple[dict[str, list[float]], PeftModel]:
    """Fine-tune *model* on *train_dataset* and validate per epoch.

    Uses the provided :class:`~accelerate.Accelerator` for device placement,
    BF16 mixed-precision, and transparent multi-GPU / multi-node distribution.
    The caller is responsible for creating the accelerator so that a single
    process-group is shared across training and inference.

    Args:
        model: LoRA-wrapped causal-LM ready for training.
        tokenizer: Tokenizer paired with *model*.
        train_dataset: Dataset of labeled training examples.
        val_dataset: Dataset of labeled validation examples.
        train_cfg: The ``training`` sub-dict from ``config.yaml``.
        paths_cfg: The ``paths`` sub-dict from ``config.yaml``.
        accelerator: Shared :class:`~accelerate.Accelerator` instance.

    Returns:
        A two-tuple of:
        - history dict with keys ``train_loss``, ``val_loss``, and
          ``val_accuracy``, each mapping to a list of per-epoch floats.
        - The unwrapped, trained :class:`~peft.PeftModel`.
    """
    epochs: int = int(train_cfg["epochs"])  # type: ignore[arg-type]
    max_new_tokens: int = int(train_cfg.get("max_new_tokens", 256))  # type: ignore[arg-type]
    batch_size: int = int(train_cfg["batch_size"])  # type: ignore[arg-type]
    grad_accum: int = int(train_cfg["grad_accumulation_steps"])  # type: ignore[arg-type]
    learning_rate: float = float(train_cfg["learning_rate"])  # type: ignore[arg-type]
    warmup_ratio: float = float(train_cfg["warmup_ratio"])  # type: ignore[arg-type]
    weight_decay: float = float(train_cfg["weight_decay"])  # type: ignore[arg-type]
    save_best: bool = bool(train_cfg.get("save_best", True))
    save_dir: str = str(paths_cfg["saved_models"])

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    train_loader: DataLoader[dict[str, Tensor]] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader: DataLoader[dict[str, Tensor]] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps: int = (len(train_loader) // grad_accum) * epochs
    warmup_steps: int = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_acc: float = -1.0

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss: float = 0.0

        batch: dict[str, Tensor]
        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} [train]",
            disable=not accelerator.is_local_main_process,
        ):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                accelerator.backward(outputs.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += outputs.loss.detach().float().item()

        avg_train_loss: float = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, val_acc = _evaluate(model, accelerator, val_loader, tokenizer, max_new_tokens)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch}: train_loss={avg_train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"  ↑ New best model saved (val_acc={best_val_acc:.4f})")

    # ── Save history JSON (main process only) ────────────────────────────────
    if accelerator.is_main_process:
        import json as _json
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as _f:
            _json.dump(history, _f, indent=2)
        print(f"Training history saved → {history_path}")

    return history, accelerator.unwrap_model(model)


def _evaluate(
    model: PeftModel,
    accelerator: Accelerator,
    loader: DataLoader[dict[str, Tensor]],
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = 256,
) -> tuple[float, float]:
    """Compute average cross-entropy loss and option-token accuracy.

    Loss is computed via a teacher-forced forward pass (fast).  Accuracy is
    computed by **generating** from the prompt-only tokens and scanning the
    generated IDs in reverse for the last A/B/C/D option token — identical to
    the ``predict`` function in :mod:`src.evaluate`, ensuring the reported
    val_accuracy matches real inference behaviour.

    Args:
        model: The model currently being evaluated.
        accelerator: The :class:`~accelerate.Accelerator` managing devices.
        loader: DataLoader yielding batches from a :class:`~src.data.QADataset`.
        tokenizer: Tokenizer used to look up option token IDs.
        max_new_tokens: Maximum tokens to generate per example during accuracy
            evaluation (should match the inference setting for the strategy).

    Returns:
        A two-tuple ``(avg_loss, accuracy)`` computed over all batches.
    """
    model.eval()

    option_ids_per_label: list[list[int]] = get_option_token_ids(tokenizer)
    flat_to_idx: dict[int, int] = {
        tid: label_idx
        for label_idx, ids in enumerate(option_ids_per_label)
        for tid in ids
    }

    # Tensors for cross-rank reduction (must be on the accelerator device).
    _correct_t = torch.tensor(0, dtype=torch.long, device=accelerator.device)
    _total_t = torch.tensor(0, dtype=torch.long, device=accelerator.device)
    _loss_t = torch.tensor(0.0, dtype=torch.float32, device=accelerator.device)
    _batches_t = torch.tensor(0, dtype=torch.long, device=accelerator.device)

    # Unwrap once outside the loop — generate() requires the raw PeftModel,
    # not the DDP wrapper, while the DDP forward() is still used for loss.
    unwrapped = accelerator.unwrap_model(model)
    pad_id: int = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        batch: dict[str, Tensor]
        for batch in tqdm(
            loader,
            desc="[eval]",
            leave=False,
            disable=not accelerator.is_local_main_process,
        ):
            # ── Loss: teacher-forced forward pass (fast, matches training) ────
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            _loss_t += outputs.loss.detach().float()
            _batches_t += 1

            # ── Accuracy: generate from prompt-only tokens ────────────────────
            labels: Tensor = batch["labels"]
            prompt_lens: list[int] = batch["prompt_len"].tolist()
            B: int = batch["input_ids"].size(0)

            # Slice the prompt portion (dataset is right-padded; completion
            # starts at prompt_len) and left-pad into a uniform batch so causal
            # generation aligns at the right end.
            prompt_seqs: list[Tensor] = [
                batch["input_ids"][i, : prompt_lens[i]] for i in range(B)
            ]
            max_plen: int = max(t.size(0) for t in prompt_seqs)
            padded_ids = torch.full(
                (B, max_plen), pad_id, dtype=torch.long, device=accelerator.device
            )
            padded_mask = torch.zeros(
                (B, max_plen), dtype=torch.long, device=accelerator.device
            )
            for i, seq in enumerate(prompt_seqs):
                plen = seq.size(0)
                padded_ids[i, max_plen - plen :] = seq
                padded_mask[i, max_plen - plen :] = 1

            gen_ids: Tensor = unwrapped.generate(
                input_ids=padded_ids,
                attention_mask=padded_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Compare prediction against gold for each example in the batch.
            for i in range(B):
                label_row = labels[i]
                label_positions = (label_row != -100).nonzero(as_tuple=True)[0]
                if len(label_positions) == 0:
                    continue
                # Gold: scan labels backwards for the last option token.
                gold: int | None = None
                for pos in reversed(label_positions.tolist()):
                    tok = int(label_row[pos].item())
                    if tok in flat_to_idx:
                        gold = flat_to_idx[tok]
                        break
                if gold is None:
                    _total_t += 1
                    continue
                # Pred: scan newly generated tokens backwards for option token.
                new_token_ids: list[int] = gen_ids[i, max_plen:].tolist()
                pred: int | None = extract_answer_from_token_ids(
                    new_token_ids, option_ids_per_label
                )
                if pred is None:
                    # Fallback: scan the full generated sequence (includes prompt).
                    pred = extract_answer_from_token_ids(
                        gen_ids[i].tolist(), option_ids_per_label
                    )
                if pred == gold:
                    _correct_t += 1
                _total_t += 1

    # All-reduce across ranks so every rank (especially rank 0) has global totals.
    accelerator.reduce(_correct_t, reduction="sum")
    accelerator.reduce(_total_t, reduction="sum")
    accelerator.reduce(_loss_t, reduction="sum")
    accelerator.reduce(_batches_t, reduction="sum")

    global_acc: float = (
        _correct_t.item() / _total_t.item() if _total_t.item() > 0 else 0.0
    )
    global_loss: float = (
        _loss_t.item() / _batches_t.item() if _batches_t.item() > 0 else 0.0
    )
    return global_loss, global_acc
