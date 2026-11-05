"""Local generation backends; prompting and output accounting stay engine-independent."""

from contextlib import suppress
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import subprocess
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel
import yaml

from prepare_sft_data import CONFIG_DIR, build_prompt, model_dtype

GENERATION_CONFIG = CONFIG_DIR / "generation.yaml"


def load_generation_config(path=None, backend=None):
    settings = yaml.safe_load(Path(path or GENERATION_CONFIG).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or set(settings) - {"backend", "vllm"}:
        raise ValueError("Generation config accepts backend and vllm only.")
    settings = {**settings, "backend": backend or settings.get("backend")}
    if settings["backend"] not in ("vllm", "transformers"):
        raise ValueError("Generation backend must be vllm or transformers.")
    options = settings.get("vllm", {})
    if not isinstance(options, dict) or set(options) - {"gpu_memory_utilization", "enforce_eager", "seed"}:
        raise ValueError("Unsupported vllm generation settings.")
    fraction = options.get("gpu_memory_utilization", 0.8)
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("vllm gpu_memory_utilization must be between zero and one.")
    if type(options.get("enforce_eager", False)) is not bool or type(options.get("seed", 42)) is not int:
        raise ValueError("vllm enforce_eager must be boolean and seed must be integer.")
    return {**settings, "vllm": {"gpu_memory_utilization": fraction,
                               "enforce_eager": options.get("enforce_eager", False),
                               "seed": options.get("seed", 42)}}


def add_generation_arguments(parser):
    parser.add_argument("--generation-config", type=Path, default=GENERATION_CONFIG,
                        help="Shared generation settings (default: configs/generation.yaml).")
    parser.add_argument("--generation-backend", choices=("vllm", "transformers"),
                        help="Override the configured generation backend; never falls back automatically.")


def exec_final_evaluation(final_dir, generation_config=None, generation_backend=None, prepared_sft_data=None):
    """Replace the training process so no training CUDA context survives into vLLM."""
    args = [sys.executable, str(Path(__file__).with_name("evaluate_checkpoint.py")),
            "--checkpoint", str(final_dir), "--generation-config", str(generation_config or GENERATION_CONFIG)]
    if generation_backend:
        args.extend(["--generation-backend", generation_backend])
    if prepared_sft_data:
        args.extend(["--prepared-sft-data", str(prepared_sft_data)])
    print(f"Starting saved-checkpoint evaluation: {final_dir}", flush=True)
    try:
        os.execv(sys.executable, args)
    except OSError as error:
        raise RuntimeError(f"Training is saved at {final_dir}; run evaluate_checkpoint.py --checkpoint {final_dir} separately.") from error


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def validate_vllm_environment(training=False):
    """Check the tested CUDA stack without loading model weights."""
    expected = {"vllm": "0.10.2", "torch": "2.8.0", "transformers": "4.57.6"}
    if training:
        expected["trl"] = "0.27.2"
    for package, required in expected.items():
        installed = package_version(package)
        if installed is None or installed.split("+")[0] != required:
            raise ValueError(f"vLLM requires the qwen-vllm environment: {package}=={required}, found {installed}.")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise ValueError("vLLM requires an available CUDA GPU.")
    cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    if cuda_version != (12, 8):
        raise ValueError(f"Use the documented CUDA 12.8 PyTorch build; found {torch.version.cuda}.")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=15)
        drivers = [tuple(int(part) for part in line.strip().split(".")[:2])
                   for line in result.stdout.splitlines() if line.strip()]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot read the NVIDIA host driver; check nvidia-smi before starting vLLM.") from error
    if not drivers or min(drivers) < (570, 26):
        raise ValueError("This vLLM environment requires NVIDIA host driver >=570.26; updating a container toolkit is insufficient.")
    return {"versions": {package: package_version(package) for package in expected},
            "cuda": torch.version.cuda, "driver": result.stdout.strip(),
            "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}


def decoded_answer(tokenizer, output, budget):
    ids = list(output)
    finished = tokenizer.eos_token_id in ids
    if finished:
        ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
    return tokenizer.decode(ids, skip_special_tokens=True), len(ids), not finished and len(ids) >= budget


@torch.inference_mode()
def generate_answers(model, tokenizer, questions, max_new_tokens):
    prompts = [build_prompt(tokenizer, question) for question in questions]
    inputs = tokenizer(prompts, padding=True, padding_side="left", add_special_tokens=False,
                       return_token_type_ids=False, return_tensors="pt").to(model.device)
    width = inputs["input_ids"].shape[-1]
    if width + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Question and output budget exceed the model context; reduce max-new-tokens.")
    was_training, use_cache = model.training, model.config.use_cache
    model.eval()
    try:
        model.config.use_cache = True
        with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=model.dtype == torch.bfloat16):
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )[:, width:].tolist()
    finally:
        model.config.use_cache = use_cache
        model.train(was_training)
    return [decoded_answer(tokenizer, output, max_new_tokens) for output in outputs]


class TransformersGenerator:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.initialization_runtime = 0.0
        self.description = {"backend": "transformers", "version": package_version("transformers")}

    def generate(self, questions, budget):
        return generate_answers(self.model, self.tokenizer, questions, budget)

    def recover(self):
        gc.collect()
        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()

    def close(self):
        pass


class VLLMGenerator:
    def __init__(self, model_name, checkpoint, tokenizer, config, settings, device, local_files_only=True):
        if device != "cuda" or not torch.cuda.is_available():
            raise ValueError("vLLM generation requires CUDA; select --generation-backend transformers for CPU.")
        runtime = validate_vllm_environment()
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as error:
            raise ImportError("vLLM is unavailable. Use the documented vLLM environment or explicitly select transformers.") from error
        from huggingface_hub import snapshot_download

        base = Path(model_name)
        self.base = str(base.resolve()) if base.is_dir() else snapshot_download(model_name, local_files_only=local_files_only)
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.tokenizer = tokenizer
        self.LLM, self.SamplingParams = LLM, SamplingParams
        self.request = LoRARequest("policy", 1, str(self.checkpoint)) if checkpoint else None
        model_config = AutoConfig.from_pretrained(self.base, local_files_only=True)
        self.context_limit = min(model_config.max_position_embeddings, config["max_length"] + config["max_new_tokens"])
        self.options = dict(settings["vllm"])
        self.kwargs = dict(model=self.base, tokenizer=str(checkpoint or self.base),
                           skip_tokenizer_init=True,
                           dtype="bfloat16" if config.get("bf16") else "float32",
                           max_model_len=self.context_limit, tensor_parallel_size=1,
                           generation_config="vllm", **self.options)
        if self.checkpoint:
            adapter = json.loads((self.checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
            rank = adapter["r"]
            supported = next((value for value in (8, 16, 32, 64, 128, 256, 320, 512) if value >= rank), None)
            if supported is None:
                raise ValueError(f"Unsupported vLLM LoRA rank: {rank}.")
            self.kwargs.update(enable_lora=True, max_lora_rank=supported)
        self.engine = None
        self.initialization_runtime = 0.0
        self.description = {"backend": "vllm", "version": package_version("vllm"),
                            "max_model_len": self.context_limit, "runtime": runtime, **self.options}
        self._initialize()

    def _initialize(self):
        started = time.perf_counter()
        try:
            self.engine = self.LLM(**self.kwargs)
        finally:
            self.initialization_runtime += time.perf_counter() - started

    def generate(self, questions, budget):
        prompts = [self.tokenizer.encode(build_prompt(self.tokenizer, question), add_special_tokens=False)
                   for question in questions]
        if any(len(prompt) + budget > self.context_limit for prompt in prompts):
            raise ValueError("Question and output budget exceed the configured model context.")
        if self.engine is None:
            self._initialize()
        # Ignore the repository's extra EOS list; only our native assistant EOS stops generation.
        params = self.SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=budget,
                                     ignore_eos=True, stop_token_ids=[self.tokenizer.eos_token_id],
                                     detokenize=False)
        outputs = self.engine.generate([{"prompt_token_ids": prompt} for prompt in prompts],
                                       params, lora_request=self.request, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned an incomplete generation batch.")
        answers = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise RuntimeError("Expected exactly one answer per evaluation question.")
            completion = output.outputs[0]
            if completion.finish_reason not in ("stop", "length"):
                raise RuntimeError(f"Unexpected vLLM finish reason: {completion.finish_reason}.")
            ids = list(completion.token_ids)
            if completion.finish_reason == "stop" and self.tokenizer.eos_token_id not in ids:
                raise RuntimeError("vLLM stopped without preserving the native EOS token.")
            if completion.finish_reason == "length" and len(ids) != budget:
                raise RuntimeError("vLLM exhausted an unexpected output budget.")
            answers.append(decoded_answer(self.tokenizer, ids, budget))
        return answers

    def close(self):
        engine, self.engine = self.engine, None
        if engine is not None:
            # Supported engine versions expose shutdown on one of these owners.
            for owner in (getattr(engine.llm_engine, "engine_core", None),
                          getattr(engine.llm_engine, "model_executor", None)):
                if owner is not None and hasattr(owner, "shutdown"):
                    owner.shutdown()
            del engine
        if torch.distributed.is_initialized():
            from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
            destroy_model_parallel()
            destroy_distributed_environment()
        gc.collect()
        if torch.cuda.is_available():
            with suppress(RuntimeError):
                torch.cuda.empty_cache()

    def recover(self):
        self.close()


def create_generator(model_name, checkpoint, tokenizer, config, settings, device, model=None, local_files_only=True):
    if settings["backend"] == "vllm":
        if model is not None:
            raise ValueError("Release the training model before creating the vLLM evaluation engine.")
        return VLLMGenerator(model_name, checkpoint, tokenizer, config, settings, device, local_files_only)
    started = time.perf_counter()
    if model is None:
        adapter_options = ({"adapter_kwargs": {"local_files_only": local_files_only}}
                           if int(package_version("transformers").split(".")[0]) >= 5 else {})
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype(config), local_files_only=local_files_only,
                                                   **adapter_options)
        if checkpoint:
            model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True)
        model.to(device)
    generator = TransformersGenerator(model, tokenizer)
    generator.initialization_runtime = time.perf_counter() - started
    return generator
