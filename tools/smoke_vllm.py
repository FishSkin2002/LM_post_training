"""Real CUDA smoke checks using random tiny Qwen3 weights, never the 4B model."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import Qwen3Config, Qwen3ForCausalLM

from generation import VLLMGenerator, validate_vllm_environment
from prepare_sft_data import build_prompt, messages
from train_post_sft import MathGRPOTrainer, training_arguments
from tools.test_instruct_sft import InstructSFTTests


class MemorySampler:
    """Sample whole-device NVML memory, including vLLM workers and their allocators."""
    def __init__(self):
        self.peak = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        from vllm.third_party import pynvml
        try:
            pynvml.nvmlInit()
            device = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self.stop_event.is_set():
                used = pynvml.nvmlDeviceGetMemoryInfo(device).used
                self.peak = max(self.peak or 0, used)
                self.stop_event.wait(0.1)
        except pynvml.NVMLError:
            self.peak = None
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inference", "grpo"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/qwen-vllm-gpu-smoke"))
    args = parser.parse_args()
    runtime = validate_vllm_environment(training=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = InstructSFTTests()
    with redirect_stdout(io.StringIO()):
        fixture.setUp()
    memory = MemorySampler()
    memory.start()
    try:
        tokenizer = fixture.tokenizer
        base = args.output_dir / f"{args.mode}-base"
        base.mkdir(exist_ok=True)
        tokenizer.save_pretrained(base)
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(tokenizer), hidden_size=128, intermediate_size=256,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=64,
            max_position_embeddings=256, tie_word_embeddings=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )).to(torch.bfloat16).eval()
        if args.mode == "inference":
            ids = tokenizer.encode(build_prompt(tokenizer, "question 1"), add_special_tokens=False)
            with torch.no_grad():
                hidden = model.model(torch.tensor([ids])).last_hidden_state[0, -1]
                model.lm_head.weight.zero_()
                model.lm_head.weight[tokenizer.eos_token_id].copy_(hidden)
        model.save_pretrained(base)
        model.config._name_or_path = model.name_or_path = str(base)
        policy = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", r=4, lora_alpha=8))
        started = time.perf_counter()
        if args.mode == "inference":
            adapter = args.output_dir / "adapter"
            policy.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)
            config = {"bf16": True, "max_length": 96, "max_new_tokens": 4}
            settings = {"vllm": {"gpu_memory_utilization": 0.35, "enforce_eager": True, "seed": 42}}
            generator = VLLMGenerator(str(base), adapter, tokenizer, config, settings, "cuda")
            try:
                request, generator.request = generator.request, None
                original = generator.generate(["question 1", "question 2"], 4)
                generator.request = request
                adapted = generator.generate(["question 1", "question 2"], 4)
                assert all(row[1:] == (1, False) for row in original + adapted), (original, adapted)
                results = {"original": original, "lora": adapted, "engine": generator.description}
            finally:
                generator.close()
        else:
            initial = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(policy).items()}
            dataset = Dataset.from_list([{"prompt": messages(f"question {i}"), "references": ["2"]} for i in range(4)])

            def synthetic_reward(completions, **kwargs):
                # This only exercises parameter updates; production boxed_reward is unchanged.
                return [float(i % 2) for i in range(len(completions))]

            settings = dict(use_vllm=True, vllm_mode="colocate", vllm_tensor_parallel_size=1,
                            vllm_max_model_length=128, vllm_gpu_memory_utilization=0.35,
                            vllm_enable_sleep_mode=True, per_device_train_batch_size=1,
                            gradient_accumulation_steps=2, num_generations=2, max_completion_length=4,
                            max_steps=2, learning_rate=1e-4, beta=0.001, logging_steps=1,
                            save_strategy="no", disable_tqdm=True)
            trainer_args = training_arguments("grpo", settings, args.output_dir / "grpo-run", True, tokenizer)
            trainer = MathGRPOTrainer(model=policy, args=trainer_args, train_dataset=dataset,
                                      processing_class=tokenizer, reward_funcs=synthetic_reward,
                                      run_metadata={"stage": "grpo", "smoke": True})
            trainer.train()
            reference = get_peft_model_state_dict(policy, adapter_name="ref")
            current = get_peft_model_state_dict(policy)
            assert all(torch.equal(reference[key].cpu(), value) for key, value in initial.items()), "SFT reference changed"
            assert any(not torch.equal(current[key].cpu(), value) for key, value in initial.items()), "Policy did not update"
            assert trainer._last_loaded_step == 1, "Updated policy was not synchronized before the second rollout"
            results = {"steps": trainer.state.global_step, "last_synced_step": trainer._last_loaded_step,
                       "reference_unchanged": True, "policy_updated": True,
                       "importance_sampling_correction": trainer.args.vllm_importance_sampling_correction}
        memory.stop()
        results.update(runtime=runtime, elapsed=time.perf_counter() - started,
                       parent_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       sampled_peak_gpu_memory_bytes=memory.peak)
        output = args.output_dir / f"{args.mode}.json"
        output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"PASS: {args.mode}; report: {output}", flush=True)
    finally:
        memory.stop()
        fixture.doCleanups()


if __name__ == "__main__":
    main()
