"""Model loading and final artifact saving shared by training stages."""

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from utils.prepare_sft_data import model_dtype


def load_training_model(config, source_path=None):
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], dtype=model_dtype(config))
    model.config.use_cache = False
    if source_path:
        model = PeftModel.from_pretrained(model, source_path, is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", target_modules="all-linear", **config["lora"],
        ))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model


def save_final(trainer, tokenizer, run_dir):
    final_dir = run_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    print(f"Training complete. Saved adapter and tokenizer to {final_dir}", flush=True)
    return final_dir
