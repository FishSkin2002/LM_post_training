"""Model construction and checkpoint compatibility for full and LoRA SFT."""

import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def make_lora_config(options):
    allowed = {"r", "lora_alpha", "lora_dropout"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("lora must contain only r, lora_alpha and lora_dropout.")
    values = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, **options}
    for key in ("r", "lora_alpha"):
        if type(values[key]) is not int or values[key] <= 0:
            raise ValueError(f"lora.{key} must be a positive integer.")
    dropout = values["lora_dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("lora.lora_dropout must be a number in [0, 1).")
    return LoraConfig(
        task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
        bias="none", **values,
    )


def attach_lora(model, options):
    config = make_lora_config(options)
    config.revision = getattr(model.config, "_commit_hash", None)
    model = get_peft_model(model, config)
    # Keep optimizer state and gradients in FP32 even with a BF16/FP16 base.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model


def validate_resume(checkpoint, training_mode, model_name, lora_options):
    """Reject full/adapter or base/config mismatches before loading weights."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (checkpoint / "trainer_state.json").is_file():
        raise ValueError("Resume requires a Trainer checkpoint containing trainer_state.json, not final/.")
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (checkpoint / filename).is_file():
            raise ValueError(f"Resume checkpoint is missing {filename}; restore the complete training checkpoint.")
    adapter_path = checkpoint / "adapter_config.json"
    if adapter_path.is_file() != (training_mode == "lora"):
        raise ValueError("Checkpoint training mode differs: full and LoRA optimizer states cannot be mixed.")
    if training_mode == "lora":
        saved = json.loads(adapter_path.read_text(encoding="utf-8"))
        if saved.get("base_model_name_or_path") != model_name:
            raise ValueError("LoRA checkpoint base model differs from the selected model_name.")
        expected = make_lora_config(lora_options)
        for key in ("r", "lora_alpha", "lora_dropout", "bias"):
            if saved.get(key) != getattr(expected, key):
                raise ValueError(f"LoRA checkpoint differs in {key}; use its original configuration.")
        if set(saved.get("target_modules", [])) != set(expected.target_modules):
            raise ValueError("LoRA checkpoint target_modules differ from the selected configuration.")
        return saved.get("revision")
    return None


def load_checkpoint_model(checkpoint, *, dtype, device, attention_implementation):
    """Load a local full model or local adapter plus its cached, matching base."""
    checkpoint = Path(checkpoint)
    kwargs = {"dtype": dtype, "attn_implementation": attention_implementation, "local_files_only": True}
    if (checkpoint / "adapter_config.json").is_file():
        config = PeftConfig.from_pretrained(checkpoint, local_files_only=True)
        if not config.base_model_name_or_path:
            raise ValueError("Adapter configuration is missing base_model_name_or_path.")
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, revision=config.revision, **kwargs,
        )
        model = PeftModel.from_pretrained(base, checkpoint, local_files_only=True, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
    return model.to(device)
