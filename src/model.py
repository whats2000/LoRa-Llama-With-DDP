"""model.py — model and tokenizer loading driven by config."""

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)


def load_model_and_tokenizer(
    model_cfg: dict[str, object],
    lora_cfg: dict[str, object],
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load the base causal-LM and wrap it with LoRA adapters.

    Args:
        model_cfg: The ``model`` sub-dict from ``config.yaml``, containing
            at least ``model_id`` and optionally ``use_4bit``.
        lora_cfg: The ``lora`` sub-dict from ``config.yaml``, containing
            ``r``, ``lora_alpha``, ``target_modules``, ``lora_dropout``,
            and ``bias``.

    Returns:
        A two-tuple ``(model, tokenizer)`` where *model* has LoRA adapters
        attached and *tokenizer* is the matching HuggingFace tokenizer.
    """
    model_id = str(model_cfg["model_id"])
    use_4bit = bool(model_cfg.get("use_4bit", True))

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None = (
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
        dtype=torch.bfloat16 if not use_4bit else None,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["r"]),  # type: ignore[arg-type]
        lora_alpha=int(lora_cfg["lora_alpha"]),  # type: ignore[arg-type]
        target_modules=list(lora_cfg["target_modules"]),  # type: ignore[arg-type]
        lora_dropout=float(lora_cfg["lora_dropout"]),  # type: ignore[arg-type]
        bias=str(lora_cfg["bias"]),
    )

    model: PeftModel = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
