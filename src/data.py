"""data.py — dataset loading, splitting, and prompt formatting driven by config."""

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

OPTION_LABELS: list[str] = ["A", "B", "C", "D"]


def get_option_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
    """Return all token-ID variants for each option label (A/B/C/D).

    Both the bare letter (``"A"``) and the space-prefixed form (``" A"``) are
    included so the extraction works regardless of whether the tokenizer merges
    the leading space with the preceding word.

    Args:
        tokenizer: HuggingFace tokenizer paired with the model.

    Returns:
        A list of 4 sub-lists, one per label in ``OPTION_LABELS`` order.
        Each sub-list contains the unique token IDs that represent that label.
    """
    result: list[list[int]] = []
    for c in OPTION_LABELS:
        ids: set[int] = set()
        for variant in (c, f" {c}"):
            tids = tokenizer.encode(variant, add_special_tokens=False)
            if tids:
                ids.add(tids[0])
        result.append(list(ids))
    return result


def extract_answer_from_token_ids(
    token_ids: list[int],
    option_ids_per_label: list[list[int]],
) -> int | None:
    """Scan *token_ids* in **reverse** for the last option token.

    Reverse scanning finds the final answer letter even when the model
    emits a long chain-of-thought that contains A/B/C/D tokens earlier.

    Args:
        token_ids: Flat list of token IDs (typically the generated portion).
        option_ids_per_label: Per-label variant token-ID lists as returned by
            :func:`get_option_token_ids`.

    Returns:
        The 0-based label index (0 = A, 1 = B, 2 = C, 3 = D) of the last
        matching token, or ``None`` if no option token was found.
    """
    flat_to_idx: dict[int, int] = {
        tid: label_idx
        for label_idx, ids in enumerate(option_ids_per_label)
        for tid in ids
    }
    for tid in reversed(token_ids):
        if tid in flat_to_idx:
            return flat_to_idx[tid]
    return None


def format_prompt(
    row: pd.Series,
    prompt_cfg: dict[str, object],
    examples: list[pd.Series] | None = None,
) -> str:
    """Build a prompt string from a data row using the prompting config.

    Args:
        row: A single dataset row containing question and option fields.
        prompt_cfg: The ``prompting`` sub-dict loaded from ``config.yaml``.
        examples: Optional list of example rows for few-shot prompting.

    Returns:
        A fully formatted prompt string ready for the model.

    Raises:
        ValueError: If the strategy in ``prompt_cfg`` is not recognized.
    """
    strategy = str(prompt_cfg["strategy"])
    question_block = str(prompt_cfg["question_block"]).format(
        question=row["question"],
        opa=row["opa"],
        opb=row["opb"],
        opc=row["opc"],
        opd=row["opd"],
    )
    system = str(prompt_cfg["system_message"])

    if strategy == "zero_shot":
        cfg = dict(prompt_cfg["zero_shot"])  # type: ignore[arg-type]
        return f"{system}\n\n{question_block}{cfg['suffix']}"

    if strategy == "few_shot":
        cfg = dict(prompt_cfg["few_shot"])  # type: ignore[arg-type]
        shots = ""
        for ex in examples or []:
            label = OPTION_LABELS[int(ex["ans"])]
            ex_block = str(prompt_cfg["question_block"]).format(
                question=ex["question"],
                opa=ex["opa"],
                opb=ex["opb"],
                opc=ex["opc"],
                opd=ex["opd"],
            )
            shots += ex_block + str(cfg["example_suffix"]).format(label=label) + "\n\n"
        return f"{system}\n\n{shots}{question_block}{cfg['suffix']}"

    if strategy == "cot":
        cfg = dict(prompt_cfg["cot"])  # type: ignore[arg-type]
        return f"{system}\n{cfg['prefix']}\n\n{question_block}{cfg['suffix']}"

    raise ValueError(f"Unknown prompting strategy: {strategy!r}")


def generate_cot_rationales(
    model,
    tokenizer: PreTrainedTokenizerBase,
    df: pd.DataFrame,
    prompt_cfg: dict[str, object],
    max_length: int = 256,
    max_new_tokens: int = 200,
    batch_size: int = 16,
    accelerator=None,
) -> list[str]:
    """Generate CoT rationales by prompting the model with each question + its correct answer.

    Work is sharded across all DDP ranks so each rank generates only
    ``len(df) / num_processes`` rationales.  Results are gathered and
    broadcast so every rank returns the full ordered list.

    Args:
        model: The causal-LM (PeftModel) to use for generation.
        tokenizer: Tokenizer paired with *model*.
        df: DataFrame with question, option, and answer columns.
        prompt_cfg: The ``prompting`` sub-dict from ``config.yaml``.
        max_length: Maximum input token length for the rationale prompt.
        max_new_tokens: Maximum tokens to generate per rationale.
        batch_size: Number of examples to generate in a single forward pass.
        accelerator: Shared :class:`~accelerate.Accelerator` instance.
            If ``None``, generation runs on a single GPU without sharding.

    Returns:
        A list of rationale strings aligned with *df* rows, available on
        every rank.
    """
    from accelerate.utils import gather_object  # avoid circular at module level

    cot_cfg = dict(prompt_cfg["cot"])  # type: ignore[arg-type]
    rationale_prompt_template: str = str(cot_cfg["rationale_prompt"])

    # Determine rank / world-size from accelerator when available.
    if accelerator is not None:
        rank: int = accelerator.process_index
        num_processes: int = accelerator.num_processes
        device = accelerator.device
        model.to(device)
    else:
        rank = 0
        num_processes = 1
        # Move model to GPU if it is currently on CPU and a GPU is available.
        first_device = next(model.parameters()).device
        if first_device.type == "cpu" and torch.cuda.is_available():
            try:
                model.to("cuda")
            except (ValueError, RuntimeError):
                pass
        device = next(model.parameters()).device

    model.eval()

    # Each rank processes every num_processes-th row starting at its rank index.
    local_indices = list(range(rank, len(df), num_processes))

    local_results: list[dict] = []
    # Use left-padding so all sequences in a batch align at the right end.
    orig_padding_side: str = tokenizer.padding_side
    tokenizer.padding_side = "left"

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(local_indices), batch_size),
            desc="Generating CoT rationales",
            disable=(accelerator is not None and not accelerator.is_local_main_process),
        ):
            batch_indices = local_indices[batch_start : batch_start + batch_size]

            prompts: list[str] = []
            for idx in batch_indices:
                row = df.iloc[idx]
                label = OPTION_LABELS[int(row["ans"])]
                prompts.append(
                    rationale_prompt_template.format(
                        question=row["question"],
                        opa=row["opa"],
                        opb=row["opb"],
                        opc=row["opc"],
                        opd=row["opd"],
                        label=label,
                    )
                )

            enc = tokenizer(
                prompts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids: Tensor = enc["input_ids"].to(device)
            attention_mask: Tensor = enc["attention_mask"].to(device)
            # With left-padding all inputs share the same padded length;
            # generated tokens start after that uniform prefix.
            prompt_len: int = input_ids.shape[1]

            output_ids: Tensor = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=8,
            )

            for idx, out_ids in zip(batch_indices, output_ids):
                new_tokens = out_ids[prompt_len:]
                rationale = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                local_results.append({"idx": idx, "rationale": rationale})

    tokenizer.padding_side = orig_padding_side

    if accelerator is None or num_processes == 1:
        local_results.sort(key=lambda x: x["idx"])
        return [r["rationale"] for r in local_results]

    # Gather all shards to every rank and reconstruct the ordered full list.
    accelerator.wait_for_everyone()
    all_results: list[dict] = gather_object(local_results)
    all_results.sort(key=lambda x: x["idx"])
    return [r["rationale"] for r in all_results]


def load_datasets(
    data_cfg: dict[str, object],
    dataset_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split the labeled dataset into train and validation sets.

    Args:
        data_cfg: The ``data`` sub-dict from ``config.yaml`` containing
            ``val_ratio`` and ``seed``.
        dataset_path: Path to the source CSV file with labeled examples.

    Returns:
        A two-tuple ``(train_df, val_df)`` of DataFrames.
    """
    df = pd.read_csv(dataset_path)
    df["ans"] = df["ans"].astype(int)

    val_ratio = float(data_cfg["val_ratio"])  # type: ignore[arg-type]
    seed = int(data_cfg["seed"])  # type: ignore[arg-type]

    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
    )


def load_benchmark(benchmark_path: str) -> pd.DataFrame:
    """Load the benchmark CSV (no answer column).

    Args:
        benchmark_path: Path to the benchmark CSV file.

    Returns:
        DataFrame with question and option columns.
    """
    return pd.read_csv(benchmark_path).reset_index(drop=True)


class QADataset(Dataset):
    """PyTorch Dataset that tokenizers prompts and answer labels for causal-LM training."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        prompt_cfg: dict[str, object],
        max_length: int = 512,
        few_shot_examples: list[pd.Series] | None = None,
        rationales: list[str] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            df: DataFrame containing questions, options, and answers.
            tokenizer: HuggingFace tokenizer compatible with the target model.
            prompt_cfg: The ``prompting`` sub-dict from ``config.yaml``.
            max_length: Maximum token length for padding/truncation.
            few_shot_examples: Optional rows used as few-shot demonstrations.
            rationales: Optional pre-generated CoT rationale strings, one per
                row in *df*.  When provided (and strategy is ``cot``), each
                training target becomes ``<rationale>\\nAnswer: <label>``
                instead of the bare answer letter.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.prompt_cfg = prompt_cfg
        self.max_length = max_length
        self.few_shot_examples = few_shot_examples
        self.rationales = rationales

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Tokenize and return a single training example.

        Args:
            idx: Integer index into the underlying DataFrame.

        Returns:
            A dict with keys ``input_ids``, ``attention_mask``, and ``labels``.
            Prompt token positions in ``labels`` are masked with ``-100``.
        """
        row = self.df.iloc[idx]
        prompt = format_prompt(row, self.prompt_cfg, self.few_shot_examples)
        label = OPTION_LABELS[int(row["ans"])]

        eos = self.tokenizer.eos_token or ""
        if self.rationales is not None and idx < len(self.rationales):
            # CoT rationale distillation: train on "<rationale>\nAnswer: <label><eos>"
            # EOS is required so the model learns to stop after the answer letter
            # instead of looping or hallucinating additional content.
            completion = f" {self.rationales[idx]}\nAnswer: {label}{eos}"
        else:
            completion = f" {label}{eos}"

        full_text = prompt + completion

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids: Tensor = encoding["input_ids"].squeeze()
        attention_mask: Tensor = encoding["attention_mask"].squeeze()

        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length)
        prompt_len: int = len(prompt_enc["input_ids"])

        labels: Tensor = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
        }
