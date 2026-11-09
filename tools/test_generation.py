"""Backend contract tests without downloading a real model or invoking CUDA."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
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
import evaluate as evaluation
import summarize_evaluation as summarizing
from utils.prepare_sft_data import load_tokenizer
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

    def evaluate(self, dataset, info, **kwargs):
        kwargs.setdefault("output", self.root / "evaluation.json")
        with redirect_stdout(io.StringIO()):
            return evaluation.evaluate_checkpoint(
                model_name=self.fixture.model_name, tokenizer=self.tokenizer, config=self.fixture.config,
                dataset=dataset, data_info=info, device="cpu", disable_tqdm=True, **kwargs)

    def test_evaluation_defaults_follow_the_effective_backend_and_keep_fixed_chunks(self):
        self.fixture.raw = [fixtures.raw_row(i) for i in range(1290)]
        dataset, info = self.fixture.prepared()
        config = self.root / "vllm.yaml"
        config.write_text("backend: vllm\n")

        def answers(questions, budget):
            return [("invalid", budget, True)] * len(questions)

        for backend, override, sizes in (("vllm", None, [64, 64, 1]),
                                         ("transformers", "transformers", [4] * 32 + [1])):
            engine = Mock(initialization_runtime=0.0, description={"backend": backend})
            engine.generate.side_effect = answers
            with self.subTest(backend=backend), patch.object(evaluation, "create_generator", return_value=engine), \
                    patch.object(evaluation, "generate_answers", side_effect=lambda m, t, q, b: answers(q, b)) as native:
                report = self.evaluate(dataset, info, generation_config=config, generation_backend=override,
                                       output=self.root / f"{backend}.json")
            calls = engine.generate if backend == "vllm" else native
            question_arg = 0 if backend == "vllm" else 2
            self.assertEqual([len(call.args[question_arg]) for call in calls.call_args_list], sizes)
            self.assertEqual(report["generation"]["batch_size"], sizes[0])
            self.assertFalse(report["generation"]["auto_batch_size"])
            self.assertEqual([row["generation_batch_size"] for row in report["predictions"]],
                             [size for size in sizes for _ in range(size)])
            self.assertEqual([row["index"] for row in report["predictions"]], report["data"]["indices"])
            engine.recover.assert_not_called()
            engine.close.assert_called_once()

    def test_vllm_fixed_failure_preserves_progress_and_resume_can_change_chunk_size(self):
        self.fixture.raw = [fixtures.raw_row(i) for i in range(60)]
        dataset, info = self.fixture.prepared()
        output = self.root / "resume.json"
        progress_path = output.with_suffix("") / "answers.jsonl"
        metadata_path = output.with_suffix("") / "metadata.json"
        failure = RuntimeError("vLLM worker failed")
        engine = Mock(initialization_runtime=0.0, description={"backend": "vllm"})
        engine.generate.side_effect = [[("invalid", 4, True)] * 2, failure]
        with patch.object(evaluation, "create_generator", return_value=engine), \
                self.assertRaises(RuntimeError) as caught:
            self.evaluate(dataset, info, generation_backend="vllm", batch_size=2, output=output)
        self.assertIs(caught.exception, failure)
        self.assertEqual([len(call.args[0]) for call in engine.generate.call_args_list], [2, 2])
        engine.recover.assert_not_called()
        engine.close.assert_called_once()
        saved, metadata = progress_path.read_bytes(), metadata_path.read_bytes()
        self.assertEqual(len(saved.splitlines()), 2)

        resumed = Mock(initialization_runtime=0.0, description={"backend": "vllm"})
        resumed.generate.side_effect = lambda questions, budget: [("invalid", budget, True)] * len(questions)
        with patch.object(evaluation, "create_generator", return_value=resumed):
            report = self.evaluate(dataset, info, generation_backend="vllm", batch_size=3, output=output)
        self.assertEqual([len(call.args[0]) for call in resumed.generate.call_args_list], [3, 1])
        self.assertEqual(resumed.generate.call_args_list[0].args[0][:2], engine.generate.call_args_list[1].args[0])
        self.assertTrue(progress_path.read_bytes().startswith(saved))
        self.assertEqual(metadata_path.read_bytes(), metadata)
        self.assertFalse(list(output.with_suffix("").glob("*.md")))
        self.assertEqual([row["index"] for row in report["predictions"]], report["data"]["indices"])
        self.assertEqual([row["generation_batch_size"] for row in report["predictions"]], [2, 2, 3, 3, 3, 1])
        self.assertEqual(report["generation"]["batch_size"], 2)
        resumed.recover.assert_not_called()

    def test_vllm_rejects_adaptive_batches_before_loading_sources_or_creating_outputs(self):
        output = self.root / "rejected.json"
        with ExitStack() as stack:
            loaders = [stack.enter_context(patch.object(evaluation, name)) for name in
                       ("read_source", "load_tokenizer", "ensure_prepared_sft_data", "prepare_dataset", "create_generator")]
            for source in ({"model_name": self.fixture.model_name}, {"checkpoint": self.root / "missing"}):
                with self.subTest(source=source), self.assertRaisesRegex(ValueError, "auto-batch-size.*transformers"):
                    evaluation.evaluate_checkpoint(**source, generation_backend="vllm", auto_batch_size=True,
                                                   output=output)
            terminal = io.StringIO()
            argv = ["evaluate.py", "--checkpoint", self.fixture.model_name, "--generation-backend", "vllm",
                    "--auto-batch-size", "--output", str(output)]
            with patch.object(sys, "argv", argv), redirect_stderr(terminal), self.assertRaises(SystemExit) as caught:
                evaluation.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("auto-batch-size requires --generation-backend transformers", terminal.getvalue())
            for loader in loaders:
                loader.assert_not_called()
        self.assertFalse(list(self.root.glob("rejected*")))

    def test_evaluation_rejects_invalid_explicit_batch_sizes_for_both_backends(self):
        for backend in ("vllm", "transformers"):
            for size in (0, -1, True, 1.5, "4"):
                with self.subTest(backend=backend, size=size), self.assertRaisesRegex(ValueError, "positive integer"):
                    evaluation.evaluate_checkpoint(model_name=self.fixture.model_name,
                                                   generation_backend=backend, batch_size=size)

    def test_post_training_native_call_and_vllm_cli_use_backend_defaults(self):
        dataset, info = self.fixture.prepared()
        final = self.root / "run/final"
        metadata = {"config": self.fixture.config, "data": data_plan(dataset, data_info=info),
                    "text_format": evaluation.text_format(self.tokenizer)}

        engines = []

        def generator(model_name, checkpoint, tokenizer, config, settings, device, **kwargs):
            engine = Mock(initialization_runtime=0.0, description={"backend": settings["backend"]})
            engine.generate.side_effect = lambda questions, budget: [("invalid", budget, True)] * len(questions)
            engines.append(engine)
            return engine

        def execute(executable, args):
            self.assertEqual(executable, sys.executable)
            self.assertEqual(Path(args[1]).name, "train.py")
            self.assertEqual(args[2], training_common.FINAL_EVALUATION_FLAG)
            with patch.object(sys, "argv", args[1:]):
                training_common.main()

        original_summary = summarizing.summarize_evaluation

        def summarize(*args, **kwargs):
            engines[-1].close.assert_called_once()
            return original_summary(*args, **kwargs)

        with patch.object(evaluation, "read_source", return_value=(final, metadata, None)), \
                patch.object(evaluation, "load_tokenizer", return_value=self.tokenizer), \
                patch.object(evaluation, "ensure_prepared_sft_data", return_value=None), \
                patch.object(evaluation, "prepare_dataset", return_value=(dataset, info)), \
                patch.object(evaluation, "create_generator", side_effect=generator), \
                patch.object(evaluation, "generate_answers", return_value=[("invalid", 4, True)] * 2), \
                patch.object(summarizing, "summarize_evaluation", side_effect=summarize) as summary_calls, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            native = training_common.evaluate_after_training(
                final, {"backend": "transformers"}, model=Mock(), tokenizer=self.tokenizer,
                dataset=dataset, data_info=info, disable_tqdm=True)
            self.assertEqual(native["generation"]["batch_size"], 4)
            self.assertFalse(native["generation"]["auto_batch_size"])
            # The second path parses the exact command that replaces the training process.
            # Use a separate destination so it generates rather than reusing the native progress.
            vllm_final = self.root / "vllm-run/final"
            with patch.object(evaluation, "read_source", return_value=(vllm_final, metadata, None)), \
                    patch.object(training_common.os, "execv", side_effect=execute):
                training_common.exec_final_evaluation(vllm_final, generation_backend="vllm")
            self.assertEqual(summary_calls.call_count, 2)
        report = fixtures.stored_evaluation(next((vllm_final.parent / "evaluations").glob("*/metadata.json")).parent)
        self.assertEqual(report["generation"]["batch_size"], 64)
        self.assertFalse(report["generation"]["auto_batch_size"])
        self.assertTrue((Path(report["directory"]) / "answers-0001.md").is_file())

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
            training_common.exec_final_evaluation(final, generation_backend="vllm")
        executable, args = execute.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertIn(str(final), args)
        self.assertIn("--generation-backend", args)
        self.assertEqual(Path(args[1]).name, "train.py")
        self.assertIn(training_common.FINAL_EVALUATION_FLAG, args)
        self.assertNotIn("--prepared-sft-data", args)

    def test_local_evaluation_failure_never_starts_summary(self):
        final = self.root / "saved-final"
        with patch.object(evaluation, "evaluate_checkpoint", side_effect=RuntimeError("generation failed")), \
                patch.object(summarizing, "summarize_evaluation") as summarize, \
                self.assertRaisesRegex(RuntimeError, "training is saved") as error:
            training_common.evaluate_saved_training(final)
        summarize.assert_not_called()
        self.assertIn(str(final), str(error.exception))
        self.assertIn("evaluate.py", str(error.exception))

    def test_grpo_training_vllm_requires_fresh_process_even_for_native_final_generation(self):
        final = self.root / "grpo/final"
        with patch.object(training_common, "evaluate_saved_training") as native:
            self.assertIsNone(training_common.evaluate_after_training(
                final, {"backend": "transformers"}, model=Mock(), tokenizer=self.tokenizer,
                dataset=Mock(), training_uses_vllm=True))
        native.assert_not_called()
        with patch.object(train_grpo, "train", return_value=final), \
                patch.object(train_grpo, "load_config", return_value=({"trainer": {"use_vllm": True}}, {})), \
                patch.object(training_common, "exec_final_evaluation") as handoff:
            training_common.main(["grpo", "--source", str(self.root / "sft/final"),
                                  "--generation-backend", "transformers"])
        self.assertEqual(handoff.call_args.args[0], final)
        self.assertEqual(handoff.call_args.args[2], "transformers")

    def test_failed_process_replacement_has_no_in_process_fallback_and_public_help_stays_unchanged(self):
        with patch.object(training_common.os, "execv", side_effect=OSError("exec failed")), \
                patch.object(training_common, "evaluate_saved_training") as fallback, \
                self.assertRaisesRegex(RuntimeError, "Training is saved"):
            training_common.exec_final_evaluation(self.root / "final", generation_backend="vllm")
        fallback.assert_not_called()
        with redirect_stdout(io.StringIO()) as text, self.assertRaises(SystemExit):
            training_common.main(["--help"])
        self.assertNotIn(training_common.FINAL_EVALUATION_FLAG, text.getvalue())

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
