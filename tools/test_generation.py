"""Backend contract tests without downloading a real model or invoking CUDA."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import utils.generation as generation
import utils.training.grpo as train_grpo
import train as training_common
import evaluate_checkpoint as evaluation
from utils.prepare_sft_data import load_tokenizer
from utils.prepared_sft_data import load_prepared_sft_data, save_prepared_sft_data
from utils.training_workflow import data_plan
from tools import test_instruct_sft as fixtures


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InstructSFTTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.tokenizer = self.fixture.tokenizer
        self.root = self.fixture.root

    def fake_backend(self, checkpoint=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        engine = Mock()
        factory = Mock(return_value=engine)
        sampling = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        request = Mock(side_effect=lambda *args: args)
        vllm = ModuleType("vllm")
        vllm.LLM, vllm.SamplingParams = factory, sampling
        lora = ModuleType("vllm.lora.request")
        lora.LoRARequest = request
        stack.enter_context(patch.dict(sys.modules, {"vllm": vllm, "vllm.lora.request": lora}))
        stack.enter_context(patch.object(generation, "validate_vllm_environment", return_value={"cuda": "12.8"}))
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
        stack.enter_context(patch.object(torch.cuda, "empty_cache"))
        settings = {"backend": "vllm", "vllm": {"gpu_memory_utilization": 0.5, "enforce_eager": True, "seed": 42}}
        backend = generation.VLLMGenerator(str(self.fixture.base), checkpoint, self.tokenizer,
                                           self.fixture.config, settings, "cuda")
        return backend, engine, factory, sampling, request

    def output(self, ids, reason="stop"):
        return SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids, finish_reason=reason)])

    def test_config_default_and_explicit_native_override(self):
        settings = generation.load_generation_config(Path(__file__).parents[1] / "configs/generation.yaml")
        self.assertEqual(settings["backend"], "vllm")
        self.assertEqual(generation.load_generation_config(backend="transformers")["backend"], "transformers")
        for text in ("backend: unknown", "backend: vllm\nvllm: {gpu_memory_utilization: 0}",
                     "backend: vllm\nvllm: {quantization: awq}"):
            path = self.root / "invalid.yaml"
            path.write_text(text)
            with self.assertRaises(ValueError):
                generation.load_generation_config(path)

    def test_native_eos_token_accounting_and_truncation(self):
        eos = self.tokenizer.eos_token_id
        text, count, truncated = generation.decoded_answer(self.tokenizer, [12, eos, 0, 0], 4)
        self.assertEqual(count, 2)
        self.assertFalse(truncated)
        self.assertNotIn(self.tokenizer.eos_token, text)
        self.assertEqual(generation.decoded_answer(self.tokenizer, [12] * 4, 4)[1:], (4, True))

    def test_vllm_uses_prompt_ids_and_matching_lora_and_eos(self):
        checkpoint = self.root / "adapter"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text(json.dumps({"r": 16}))
        backend, engine, factory, sampling, request = self.fake_backend(checkpoint)
        eos = self.tokenizer.eos_token_id
        engine.generate.return_value = [self.output([12, eos]), self.output([12] * 4, "length")]
        answers = backend.generate(["question 1", "question 2"], 4)
        self.assertEqual([answer[1:] for answer in answers], [(2, False), (4, True)])
        self.assertEqual(factory.call_args.kwargs["max_model_len"], 184)
        self.assertEqual(factory.call_args.kwargs["generation_config"], "vllm")
        self.assertTrue(factory.call_args.kwargs["enable_lora"])
        request.assert_called_once_with("policy", 1, str(checkpoint.resolve()))
        params = sampling.call_args.kwargs
        self.assertEqual(params["stop_token_ids"], [eos])
        self.assertTrue(params["ignore_eos"])
        self.assertNotIn("bad_words", params)
        prompts = engine.generate.call_args.args[0]
        self.assertEqual(prompts[0]["prompt_token_ids"],
                         self.tokenizer.encode(generation.build_prompt(self.tokenizer, "question 1"), add_special_tokens=False))
        self.assertNotEqual(prompts[0], prompts[1])

    def test_vllm_rejects_aborted_or_missing_eos_output(self):
        backend, engine, *_ = self.fake_backend()
        for reason in ("abort", "stop"):
            engine.generate.return_value = [self.output([12], reason)]
            with self.assertRaises(RuntimeError):
                backend.generate(["question 1"], 4)
        self.assertIsNone(backend.request)

    def test_failed_engine_is_closed_and_recreated_before_retry(self):
        backend, engine, factory, *_ = self.fake_backend()
        engine.generate.side_effect = [RuntimeError("worker failed"), [self.output([self.tokenizer.eos_token_id])]]
        with self.assertRaises(RuntimeError):
            backend.generate(["question 1"], 4)
        backend.recover()
        self.assertIsNone(backend.engine)
        self.assertEqual(backend.generate(["question 1"], 4)[0][1:], (1, False))
        self.assertEqual(factory.call_count, 2)

    def test_closing_standalone_engine_releases_its_distributed_groups(self):
        backend, *_ = self.fake_backend()
        state = ModuleType("vllm.distributed.parallel_state")
        state.destroy_model_parallel = Mock()
        state.destroy_distributed_environment = Mock()
        with patch.dict(sys.modules, {"vllm.distributed.parallel_state": state}), \
                patch.object(torch.distributed, "is_initialized", return_value=True):
            backend.close()
        state.destroy_model_parallel.assert_called_once()
        state.destroy_distributed_environment.assert_called_once()

    def test_cpu_vllm_fails_without_native_fallback(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            generation.VLLMGenerator("unused", None, self.tokenizer, {}, {}, "cpu")

    def test_training_resume_keeps_the_saved_runtime(self):
        versions = {"trl": "0.27.2", "vllm": "0.10.2"}
        with patch.object(generation, "package_version", side_effect=versions.get):
            training_common.require_resume_engine({"generation_engine": {"trl_version": "0.27.2",
                                                                         "vllm_version": "0.10.2", "backend": "vllm"}})
            for engine in (None, {"trl_version": "1.16.0", "backend": "transformers"},
                           {"trl_version": "0.27.2", "vllm_version": "0.22.0", "backend": "vllm"}):
                with self.assertRaises(ValueError):
                    training_common.require_resume_engine({"generation_engine": engine})

    def test_training_handoff_replaces_process_with_saved_final(self):
        final = self.root / "final"
        with patch.object(training_common.os, "execv") as execute:
            training_common.exec_final_evaluation(final, generation_backend="vllm", prepared_sft_data=self.root / "prepared")
        executable, args = execute.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertIn(str(final), args)
        self.assertIn("--generation-backend", args)
        self.assertIn("evaluate_checkpoint.py", args[1])
        self.assertIn("--prepared-sft-data", args)

    def test_saved_tokenizers_backend_keeps_vocabulary_and_chat_ids(self):
        path = self.root / "saved-tokenizer"
        self.tokenizer.save_pretrained(path)
        config_path = path / "tokenizer_config.json"
        config = json.loads(config_path.read_text())
        config.update(tokenizer_class="TokenizersBackend", extra_special_tokens=["<|im_start|>"])
        config_path.write_text(json.dumps(config))
        loaded = load_tokenizer(path, local_files_only=True)
        self.assertEqual(loaded.get_vocab(), self.tokenizer.get_vocab())
        self.assertEqual(loaded.chat_template, self.tokenizer.chat_template)
        self.assertEqual(loaded.encode(generation.build_prompt(loaded, "question 2"), add_special_tokens=False),
                         self.tokenizer.encode(generation.build_prompt(self.tokenizer, "question 2"), add_special_tokens=False))
        self.assertEqual(loaded.model_input_names, ["input_ids", "attention_mask"])

    def test_exported_prepared_data_keeps_source_identity_and_split(self):
        dataset, info = self.fixture.prepared()
        path = save_prepared_sft_data(self.root / "prepared", dataset, info, self.fixture.config, self.tokenizer)
        restored, restored_info = load_prepared_sft_data(path, self.fixture.config, self.tokenizer)
        self.assertEqual(restored._fingerprint, dataset._fingerprint)
        self.assertEqual(restored_info, info)
        self.assertEqual(data_plan(restored, None, restored_info), data_plan(dataset, None, info))
        self.assertEqual(restored.to_list(), dataset.to_list())
        with self.assertRaises(ValueError):
            load_prepared_sft_data(path, {**self.fixture.config, "max_length": 1}, self.tokenizer)
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["fingerprint"] = "changed"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity"):
            load_prepared_sft_data(path, self.fixture.config, self.tokenizer)

    def test_evaluation_reads_exported_data_without_rebuilding_it(self):
        dataset, info = self.fixture.prepared()
        path = save_prepared_sft_data(self.root / "prepared", dataset, info, self.fixture.config, self.tokenizer)
        model = self.fixture.model_loader(self.fixture.model_name)
        with patch.object(evaluation, "prepare_dataset", side_effect=AssertionError("must reuse exported data")), \
                patch.object(evaluation, "generate_answers", return_value=[("", 1, False)] * 2):
            report = evaluation.evaluate_checkpoint(model_name=self.fixture.model_name, model=model, tokenizer=self.tokenizer,
                                                    config=self.fixture.config, prepared_sft_data=path, device="cpu",
                                                    generation_backend="transformers", output=self.root / "evaluation.json",
                                                    disable_tqdm=True)
        self.assertEqual(report["data"]["fingerprint"], dataset._fingerprint)
        self.assertEqual(report["data"]["indices"], data_plan(dataset, None, info)["validation_indices"])

    def test_grpo_translates_generation_parameters_and_keeps_native_for_old_config(self):
        common = dict(num_generations=2, max_completion_length=4, per_device_train_batch_size=1,
                      gradient_accumulation_steps=2, use_cpu=True)
        native = train_grpo.training_arguments( common, self.root, False, self.tokenizer)
        self.assertFalse(native.use_vllm)
        self.assertTrue(native.generation_kwargs["use_cache"])
        accelerated = train_grpo.training_arguments( {**common, "use_vllm": True},
                                                         self.root, False, self.tokenizer)
        self.assertEqual(accelerated.vllm_mode, "colocate")
        self.assertNotIn("use_cache", accelerated.generation_kwargs)
        self.assertNotIn("suppress_tokens", accelerated.generation_kwargs)
        self.assertEqual(accelerated.generation_kwargs["stop_token_ids"], [self.tokenizer.eos_token_id])
        self.assertTrue(accelerated.vllm_importance_sampling_correction)


if __name__ == "__main__":
    unittest.main()
