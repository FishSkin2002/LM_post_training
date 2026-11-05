"""Offline integration checks: real tiny Qwen/LoRA, synthetic OpenR1 and Math-Verify."""

import contextlib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from datasets import Dataset, disable_progress_bars
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          LogitsProcessor, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, TrainerCallback)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math_data
import math_scoring as scoring
import prepare_sft_data as data
import train_sft
import training_workflow as workflow
import sample_checkpoint as sample
import evaluate_checkpoint as evaluation

# Official Qwen3-4B-Instruct-2507 template, revision cdbee75f17c01a7cc42f958dc650907174af0554.
TEMPLATE = (Path(__file__).parent / "fixtures/qwen3-instruct-chat-template.jinja").read_text()


def trace(number):
    return f"<think>\nCompute carefully.\n</think>\nThe result is \\boxed{{{number}}}"


def raw_row(index):
    return {"problem": f"question {index}", "solution": "UPSTREAM MUST NOT BE TARGET", "answer": str(index),
            "generations": [trace(index+1), trace(index)], "correctness_math_verify": [False, True],
            "correctness_llama": None, "is_reasoning_complete": [True, True]}


class Interrupted(RuntimeError):
    pass


class StopAfterSave(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.global_step == 2:
            raise Interrupted("simulated saved-checkpoint interruption")


class InterruptedTrainer(workflow.WorkflowTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(StopAfterSave())


class ForceToken(LogitsProcessor):
    def __init__(self, token):
        self.token = token

    def __call__(self, input_ids, scores):
        scores.fill_(-torch.inf)
        scores[:, self.token] = 0
        return scores


class InstructSFTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        disable_progress_bars()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "tiny-instruct"
        self.model_name = data.MODEL_NAME
        words = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>", "<think>", "</think>",
                 "system", "user", "assistant", "question", "Compute", "carefully.", "The", "result", "is"]
        words += [str(i) for i in range(25)] + [f"\\boxed{{{i}}}" for i in range(25)]
        words += [f"token{i}" for i in range(len(words), 128)]
        backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>",
                                                pad_token="<|endoftext|>", unk_token="<unk>")
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
        self.tokenizer.add_tokens([AddedToken("<think>", special=False), AddedToken("</think>", special=False)])
        self.tokenizer.chat_template = TEMPLATE
        self.tokenizer.save_pretrained(self.base)
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(self.tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=256, tie_word_embeddings=True,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
        ))
        model.save_pretrained(self.base)
        self.initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        self.raw = [raw_row(i) for i in range(20)]
        self.config = data.load_config(data.CONFIG_DIR / "development/sft.yaml")
        self.config.update(num_train_epochs=1, use_cpu=True, bf16=False, disable_tqdm=True,
                           dataloader_pin_memory=False, logging_steps=1, eval_steps=2, save_steps=2,
                           max_length=180, max_new_tokens=4)
        self.loader = AutoModelForCausalLM.from_pretrained
        self.tokenizer_loader = AutoTokenizer.from_pretrained
        self.model_calls = []
        self.models = []
        self.evaluation_count = 0

    def model_loader(self, source, **kwargs):
        self.assertEqual(source, self.model_name)
        self.model_calls.append(kwargs.copy())
        model = self.loader(self.base, **{**kwargs, "local_files_only": True})
        model.config._name_or_path = model.name_or_path = self.model_name
        self.models.append(model)
        return model

    def load_tokenizer(self, source, **kwargs):
        return self.tokenizer_loader(self.base if source == self.model_name else source,
                                     **{**kwargs, "local_files_only": True})

    def raw_dataset(self, *args, **kwargs):
        return Dataset.from_list(self.raw)

    def prepared(self):
        with patch.object(math_data, "load_dataset", side_effect=self.raw_dataset), contextlib.redirect_stdout(io.StringIO()):
            return data.prepare_dataset(self.config, self.tokenizer)

    def run_training(self, name, *, resume=None, continue_from=None, epochs=None, trainer=workflow.WorkflowTrainer):
        cfg = self.root / (name+".yaml")
        cfg.write_text(yaml.safe_dump({**self.config, "output_dir": str(self.root/name)}))
        argv = ["train_sft.py", "--config", str(cfg)]
        if resume:
            argv += ["--resume-from-checkpoint", str(resume)]
        if continue_from:
            argv += ["--continue-from", str(continue_from), "--additional-epochs", str(epochs)]
        with patch.object(sys, "argv", argv), patch.object(math_data, "load_dataset", side_effect=self.raw_dataset), \
                patch.object(AutoTokenizer, "from_pretrained", side_effect=self.load_tokenizer), \
                patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=self.model_loader), \
                patch.object(train_sft, "WorkflowTrainer", trainer), \
                contextlib.redirect_stdout(io.StringIO()) as terminal, contextlib.redirect_stderr(io.StringIO()) as progress:
            train_sft.main()
        self.terminal, self.progress = terminal.getvalue(), progress.getvalue()
        return Path(next(line.split(": ", 1)[1] for line in self.terminal.splitlines() if line.startswith("Output directory:")))

    def evaluate(self, checkpoint=None, *, rows=None, count=None, output=None, original=False, batch_size=None,
                 auto_batch_size=False):
        self.evaluation_count += 1
        path = output or self.root/f"eval-{self.evaluation_count}.json"
        cfg = self.root/"eval.yaml"
        cfg.write_text(yaml.safe_dump(self.config))
        argv = ["evaluate_checkpoint.py", "--device", "cpu", "--config", str(cfg), "--output", str(path)]
        argv += ["--model-name", self.model_name] if original else ["--checkpoint", str(checkpoint)]
        if count is not None:
            argv += ["--num-samples", str(count)]
        if batch_size is not None:
            argv += ["--batch-size", str(batch_size)]
        if auto_batch_size:
            argv.append("--auto-batch-size")
        with patch.object(sys, "argv", argv), \
                patch.object(math_data, "load_dataset", return_value=Dataset.from_list(rows or self.raw)), \
                patch.object(AutoTokenizer, "from_pretrained", side_effect=self.load_tokenizer), \
                patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=self.model_loader), \
                patch.object(torch.optim, "AdamW", side_effect=AssertionError("No optimizer in evaluation")), \
                contextlib.redirect_stdout(io.StringIO()) as terminal, contextlib.redirect_stderr(io.StringIO()):
            evaluation.main()
        self.eval_terminal = terminal.getvalue()
        return json.loads(path.read_text())

    def final(self, run):
        return next((run/"evaluations").glob("*.json"))

    def test_chat_masks_preserve_original_problem_think_latex_and_one_eos(self):
        question = "  question {1}\n\n"
        answer = trace(r"\frac{1}{4}")
        example = data.build_sft_example(self.tokenizer, question, answer)
        prefix = self.tokenizer.encode(data.build_prompt(self.tokenizer, question), add_special_tokens=False)
        expected = self.tokenizer.encode(answer, add_special_tokens=False)+[self.tokenizer.eos_token_id]
        self.assertEqual(example["input_ids"], prefix+expected)
        self.assertEqual(example["labels"], [-100]*len(prefix)+expected)
        self.assertEqual(data.messages(question)[1]["content"], question)
        self.assertIn(data.SYSTEM_PROMPT, data.build_prompt(self.tokenizer, question))
        self.assertNotIn("<think>", data.build_prompt(self.tokenizer, question))
        batch = DataCollatorForSeq2Seq(self.tokenizer)([example, data.build_sft_example(self.tokenizer,"question",trace(1))])
        for labels, mask in zip(batch["labels"], batch["attention_mask"]):
            self.assertTrue((labels[mask == 0] == -100).all())
            self.assertEqual(labels[mask != 0][-1], self.tokenizer.eos_token_id)
        self.assertNotEqual(self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)

    def test_openr1_flags_first_eligible_and_bad_annotation_rejection(self):
        row = raw_row(1)
        row["correctness_math_verify"] = [True, True]
        row["is_reasoning_complete"] = [False, True]
        self.assertEqual(math_data._openr1(row, 0)["generation_index"], 1)
        row["is_reasoning_complete"] = [True, True]
        self.assertEqual(math_data._openr1(row, 0)["generation_index"], 0)
        row["correctness_math_verify"] = [False, False]
        row["correctness_llama"] = [False, True]
        self.assertEqual(math_data._openr1(row, 0)["generation_index"], 1)
        row["correctness_llama"] = [False]
        with self.assertRaisesRegex(ValueError,"misaligned"):
            math_data._openr1(row, 0)
        row.pop("is_reasoning_complete")
        with self.assertRaisesRegex(ValueError,"missing"):
            math_data._openr1(row, 0)
        self.assertEqual(math_data._openr1({**raw_row(1),"answer":"proof"},0)["filter_reason"],"unsupported_reference")
        self.assertEqual(math_data._openr1({**raw_row(1),"generations":["no boxed","no boxed"]},0)["filter_reason"],"no_eligible_generation")

    def test_null_judge_annotations_are_unavailable_and_never_count_as_true(self):
        row = raw_row(1)
        self.assertEqual(math_data._openr1(row, 0)["generation_index"], 1)
        row["correctness_math_verify"] = None
        row["correctness_llama"] = [None, True]
        self.assertEqual(math_data._openr1(row, 0)["generation_index"], 1)
        row["correctness_llama"] = None
        self.assertEqual(math_data._openr1(row, 0)["filter_reason"], "no_eligible_generation")
        row["correctness_llama"] = [False, 1]
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            math_data._openr1(row, 0)

    def test_math_equivalence_and_last_boxed_not_process_correctness(self):
        cases = [(trace("0.25"),r"\frac{1}{4}",False,(True,True,False)),
                 (r"wrong reasoning \boxed{9}; final \boxed{-2}","-2",False,(True,True,False)),
                 (r"\boxed{\frac{1}{4}}",r"\frac{1}{4}",True,(False,True,True)),
                 (r"\boxed{1} then \boxed{broken", "1",False,(False,False,False)),
                 ("Answer 1/4", "1/4",False,(False,False,False)),
                 (r"\boxed{}","1",False,(False,False,False)),
                 (r"\boxed{2}","1",False,(False,True,False)),
                 (r"\boxed{B}","b",False,(True,True,False)),
                 (r"\boxed{x+1}","1+x",False,(True,True,False))]
        for text,reference,cut,expected in cases:
            with self.subTest(text=text):
                self.assertEqual(scoring.score_answer(text,reference,cut),expected)
        self.assertEqual(scoring.boxed_answer(r"\boxed{\{1,2\}}"),r"\{1,2\}")
        with self.assertRaises(ValueError): scoring.score_answer(trace(1),"proof")

    def test_gsm8k_reader_cleans_only_annotations_and_keeps_final_separate(self):
        raw="  2+3=<<2+3=5>>5\n\n#### 5  "
        row=math_data._gsm8k({"question":" raw question\n","answer":raw},4)
        self.assertEqual(row["question"]," raw question\n")
        self.assertEqual(row["answer"],"  2+3=5\n\n#### 5  ")
        self.assertEqual((row["final_answer"],row["source_index"],row["generation_index"]),("5",4,-1))

    def test_group_split_length_filter_and_source_fingerprints(self):
        self.raw += [deepcopy(self.raw[0]),{**raw_row(99),"answer":"proof"},
                     {**raw_row(98),"generations":[trace(98)+" token1"*200]*2}]
        ds,info=self.prepared()
        self.assertEqual(info["filter_counts"],{"unsupported_reference":1})
        self.assertEqual(info["over_length"],1)
        self.assertEqual(len(ds),21)
        plan=workflow.data_plan(ds,data_info=info)
        self.assertEqual((plan["question_groups"],plan["duplicate_records"]),(20,1))
        train={ds[i]["group_id"] for i in plan["train_indices"]}
        valid={ds[i]["group_id"] for i in plan["validation_indices"]}
        self.assertFalse(train&valid)
        self.assertEqual(len(valid),2)
        saved=self.root/"same-tokenizer";self.tokenizer.save_pretrained(saved)
        original=self.tokenizer; self.tokenizer=self.tokenizer_loader(saved)
        other,other_info=self.prepared()
        self.assertEqual(plan,workflow.data_plan(other,data_info=other_info))
        self.tokenizer=original
        self.assertTrue(all("UPSTREAM" not in ds[i]["answer"] for i in range(len(ds))))
        self.config["max_length"]=179
        changed,changed_info=self.prepared()
        with self.assertRaisesRegex(ValueError,"Source data"):
            workflow.data_plan(changed,{"data":plan},changed_info)

    def test_actual_lora_updates_frozen_base_checkpointing_and_exact_resume(self):
        continuous=self.run_training("continuous")
        with self.assertRaises(Interrupted): self.run_training("interrupt",trainer=InterruptedTrainer)
        checkpoint=self.root/"interrupt/checkpoint-2"
        for name in ("optimizer.pt","scheduler.pt","rng_state.pth","chat_template.jinja","run.json"):
            self.assertTrue((checkpoint/name).is_file(),name)
        resumed=self.run_training("interrupt",resume=checkpoint)
        left=load_file(continuous/"final/adapter_model.safetensors")
        right=load_file(resumed/"final/adapter_model.safetensors")
        for key in left:
            self.assertEqual(left[key].dtype,torch.float32)
            self.assertTrue(torch.equal(left[key],right[key]),key)
        self.assertTrue(any(t.abs().sum()>0 for k,t in left.items() if "lora_B" in k))
        model=PeftModel.from_pretrained(self.loader(self.base,local_files_only=True),continuous/"final",local_files_only=True)
        for key,p in model.get_base_model().named_parameters():
            if "lora_" not in key: self.assertTrue(torch.equal(p,self.initial[key.replace(".base_layer.",".")]),key)
        args=torch.load(continuous/"checkpoint-18/training_args.bin",weights_only=False)
        self.assertTrue(args.gradient_checkpointing)
        self.assertEqual(args.gradient_checkpointing_kwargs,{"use_reentrant":False})
        self.assertEqual(args.gradient_accumulation_steps,1)
        self.assertFalse(self.models[0].config.use_cache)
        self.assertEqual(self.tokenizer_loader(continuous/"final").chat_template,TEMPLATE)
        metadata=json.loads((continuous/"final/run.json").read_text())
        self.assertEqual(metadata["text_format"],data.text_format(self.tokenizer))
        self.assertEqual(metadata["data"]["source"]["kept_records"],20)
        self.assertEqual(json.loads((resumed/"trainer_state.json").read_text())["global_step"],18)

    def test_source_rejects_legacy_template_precision_data_and_missing_state_before_weights(self):
        with self.assertRaises(Interrupted): self.run_training("source",trainer=InterruptedTrainer)
        checkpoint=self.root/"source/checkpoint-2"
        meta_path=checkpoint/"run.json"; original=meta_path.read_text()
        for mutate in (lambda m:m.pop("text_format"),lambda m:m["text_format"].update(system="changed"),
                       lambda m:m["config"].update(bf16=True),lambda m:m.update(scoring={})):
            meta=json.loads(original); mutate(meta);meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError): workflow.read_source(checkpoint,self.config)
        meta_path.write_text(original)
        (checkpoint/"optimizer.pt").unlink()
        with self.assertRaisesRegex(ValueError,"Incomplete"): workflow.read_source(checkpoint,self.config,resume=True)
        with self.assertRaises(ValueError): data.require_model("Qwen/Qwen3-4B-Base")
        with self.assertRaises(ValueError): data.require_model("Qwen/Qwen3-4B")
        tokenizer=self.tokenizer_loader(checkpoint);tokenizer.chat_template=TEMPLATE+"changed";tokenizer.save_pretrained(checkpoint)
        with self.assertRaisesRegex(ValueError,"template differs"): workflow.read_source(checkpoint,self.config)

    def test_new_stage_inherits_only_weights_resets_state_and_resumes(self):
        run=self.run_training("stage"); saved=load_file(run/"final/adapter_model.safetensors"); captured={}
        class Audit(TrainerCallback):
            def on_train_begin(self,args,state,control,model=None,optimizer=None,**kwargs):
                captured.update(step=state.global_step,size=len(optimizer.state),adapters=len(model.peft_config),
                                weights={k:t.detach().clone() for k,t in get_peft_model_state_dict(model).items()})
        class Inspect(InterruptedTrainer):
            def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.add_callback(Audit())
        with self.assertRaises(Interrupted): self.run_training("stage",continue_from=run/"final",epochs=.5,trainer=Inspect)
        self.assertEqual((captured["step"],captured["size"],captured["adapters"]),(0,0,1))
        for k,t in saved.items(): self.assertTrue(torch.equal(t,captured["weights"][k]),k)
        stage=self.run_training("stage",resume=self.root/"stage-1/checkpoint-2")
        meta=json.loads((stage/"final/run.json").read_text())
        self.assertEqual(meta["epochs"],.5)
        self.assertEqual(meta["source_adapter"],str(run/"final"))
        self.assertEqual(json.loads((stage/"trainer_state.json").read_text())["global_step"],9)
        with self.assertRaisesRegex(ValueError,"training plan is complete"):
            workflow.read_source(run/"checkpoint-18",self.config,resume=True)

    def test_tensorboard_resume_purges_stale_tail_loss_only_and_english_progress(self):
        with self.assertRaises(Interrupted):self.run_training("logs",trainer=InterruptedTrainer)
        run=self.root/"logs"; initial=run/"runs/events.out.tfevents.0000000001.initial"
        next((run/"runs").glob("events.out.tfevents.*")).rename(initial)
        with SummaryWriter(str(run/"runs")) as writer: writer.add_scalar("train/loss",999,777)
        next(p for p in (run/"runs").glob("events.out.tfevents.*") if p!=initial).rename(run/"runs/events.out.tfevents.0000000002.stale")
        self.run_training("logs",resume=run/"checkpoint-2")
        events=EventAccumulator(str(run/"runs")).Reload()
        for tag in ("train/loss","train/grad_norm","train/learning_rate","eval/loss"):
            self.assertIn(tag,events.Tags()["scalars"])
        self.assertEqual([e.step for e in events.Scalars("train/loss")],list(range(1,19)))
        self.assertFalse(any("answer_accuracy" in tag for tag in events.Tags()["scalars"]))
        lines=[line for line in self.terminal.splitlines() if line.startswith(("Training |","Validation |","Generation evaluation |"))]
        self.assertTrue(lines)
        self.assertTrue(all("{" not in line and "'" not in line for line in lines))

    def test_english_progress_labels_during_real_training_and_generation(self):
        self.config["disable_tqdm"] = False
        self.run_training("progress")
        for label in ("Training:", "Validation loss:", "Generate validation answers:"):
            self.assertIn(label, self.progress)

    def test_end_evaluation_after_save_once_with_separate_reference_and_no_optimizer(self):
        def generate(model,tokenizer,questions,budget):
            final=self.root/"end/final"
            self.assertTrue((final/"adapter_model.safetensors").is_file())
            self.assertEqual(json.loads((final.parent/"trainer_state.json").read_text())["global_step"],18)
            self.assertEqual(budget,4)
            self.assertFalse(model.training)
            return [(trace(question.rsplit(" ",1)[-1]),4,False) for question in questions]
        with patch.object(evaluation,"generate_answers",side_effect=generate) as called:
            run=self.run_training("end")
        self.assertEqual(called.call_count,1)
        self.assertEqual(len(self.model_calls),1)
        report=json.loads(self.final(run).read_text())
        self.assertEqual(report["metrics"]["answer_accuracy"],1)
        self.assertEqual(report["generation"]["batch_size"],4)
        self.assertEqual([row["generation_batch_size"] for row in report["predictions"]],[2,2])
        for row in report["predictions"]:
            self.assertEqual(row["reference_answer"],self.raw[row["source_index"]]["generations"][1])
            self.assertEqual(row["reference_final_answer"],self.raw[row["source_index"]]["answer"])
        with patch.object(evaluation,"generate_answers",return_value=[(trace(0),4,False)]):
            before=(run/"final/adapter_model.safetensors").read_bytes()
            self.evaluate(run/"final",count=1)
            self.assertEqual(before,(run/"final/adapter_model.safetensors").read_bytes())

    def test_append_flush_count_resume_markdown_rebuild_and_complete_reuse(self):
        self.raw=[raw_row(i) for i in range(60)]
        output=self.root/"resume.json"; calls=[]
        def generate(model,tokenizer,questions,budget):
            question,=questions
            calls.append(question)
            if len(calls)==2:
                records=[json.loads(x) for x in output.with_suffix(".progress.jsonl").read_text().splitlines()]
                self.assertEqual(len(records),1)
                self.assertIn(records[0]["question"],output.with_suffix(".md").read_text())
                raise Interrupted("during second question")
            return [(trace(question.rsplit(" ",1)[-1]),3,False)]
        with patch.object(evaluation,"generate_answers",side_effect=generate),self.assertRaises(Interrupted):
            self.evaluate(original=True,output=output,batch_size=1)
        progress_path=output.with_suffix(".progress.jsonl")
        old=json.loads(progress_path.read_text())
        del old["generation_batch_size"]
        progress_path.write_text(json.dumps(old)+"\n")
        first=progress_path.read_bytes()
        output.with_suffix(".md").unlink()
        def invalid(model,tokenizer,questions,budget):
            return [("invalid",4,True)]*len(questions)
        with patch.object(evaluation,"generate_answers",side_effect=invalid) as called:
            report=self.evaluate(original=True,output=output)
        self.assertEqual([len(call.args[2]) for call in called.call_args_list],[4,1])
        self.assertEqual([question for call in called.call_args_list for question in call.args[2]],
                         [self.raw[i]["problem"] for i in report["data"]["indices"][1:]])
        self.assertTrue(progress_path.read_bytes().startswith(first))
        self.assertEqual(report["metrics"]["answer_accuracy"],1/6)
        self.assertEqual(report["metrics"]["generation_truncation_rate"],5/6)
        self.assertEqual([row["generation_batch_size"] for row in report["predictions"]],[1,4,4,4,4,1])
        self.assertEqual(output.with_suffix(".md").read_text().count("## Question "),6)
        with patch.object(evaluation,"generate_answers",side_effect=AssertionError("already complete")):
            again=self.evaluate(original=True,output=output)
        self.assertEqual(again["predictions"],report["predictions"])
        self.assertIn("6/6 completed",self.eval_terminal)

    def test_all_outcomes_and_summary_failure_reuse_saved_records(self):
        self.raw=[raw_row(i) for i in range(40)]
        output=self.root/"outcomes.json"
        def fail_summary(path,*args,**kwargs):
            if path==output: raise OSError("summary failure")
            return original_write(path,*args,**kwargs)
        original_write=Path.write_text
        answers=[(trace(i),4,False) for i in range(4)]
        answers[1]=(trace(999),4,False);answers[2]=("invalid",4,False);answers[3]=(trace(3),4,True)
        with patch.object(evaluation,"generate_answers",return_value=answers),patch.object(Path,"write_text",new=fail_summary),self.assertRaises(SystemExit):
            self.evaluate(original=True,output=output)
        with patch.object(evaluation,"generate_answers",side_effect=AssertionError("must reuse")):
            report=self.evaluate(original=True,output=output)
        self.assertEqual(report["metrics"]["generation_samples"],4)
        self.assertEqual(report["metrics"]["answer_format_rate"],.75)
        self.assertEqual(report["metrics"]["generation_truncation_rate"],.25)
        self.assertEqual(report["metrics"]["generation_runtime"],sum(r["generation_runtime"] for r in report["predictions"]))

    def test_original_and_zero_lora_predictions_match_and_weights_unchanged(self):
        ds,info=self.prepared();plan=workflow.data_plan(ds,data_info=info)
        base=self.model_loader(self.model_name,dtype=torch.float32)
        adapter=self.root/"zero"
        get_peft_model(base,LoraConfig(task_type="CAUSAL_LM",target_modules="all-linear",**self.config["lora"])).save_pretrained(adapter)
        self.tokenizer.save_pretrained(adapter)
        workflow.save_run(adapter,workflow.new_run(self.config,plan,1))
        weights=(self.base/"model.safetensors").read_bytes()
        left=self.evaluate(original=True,count=1);right=self.evaluate(adapter,count=1)
        for key in ("question","reference_answer","answer","generated_tokens","correct","format_valid","truncated"):
            self.assertEqual(left["predictions"][0][key],right["predictions"][0][key],key)
        self.assertEqual(left["data"],right["data"])
        self.assertEqual(left["source_type"],"original")
        self.assertIsNone(left["checkpoint"])
        self.assertEqual(weights,(self.base/"model.safetensors").read_bytes())

    def test_real_generation_eos_truncation_prefix_and_cache_restore(self):
        model=self.loader(self.base,local_files_only=True)
        model.train();model.config.use_cache=False
        generate=model.generate
        for token,cut,count in [(self.tokenizer.eos_token_id,False,1),(self.tokenizer.convert_tokens_to_ids("question"),True,4)]:
            def forced(**kwargs):return generate(**kwargs,logits_processor=[ForceToken(token)])
            with patch.object(model,"generate",side_effect=forced) as called:
                _,actual_count,actual_cut=sample.generate_answer(model,self.tokenizer,"question",4)
            self.assertEqual((actual_count,actual_cut),(count,cut))
            example=data.build_sft_example(self.tokenizer,"question",trace(1))
            prefix=next(i for i,v in enumerate(example["labels"]) if v!=-100)
            self.assertEqual(called.call_args.kwargs["input_ids"][0].tolist(),example["input_ids"][:prefix])
            self.assertTrue(called.call_args.kwargs["use_cache"])
            self.assertTrue(model.training);self.assertFalse(model.config.use_cache)
        with self.assertRaisesRegex(ValueError,"exceed"):sample.generate_answer(model,self.tokenizer,"question",256)

    def test_batch_generation_keeps_each_prompt_and_its_first_eos(self):
        model = self.loader(self.base, local_files_only=True)
        model.train()
        model.config.use_cache = False
        eos, pad = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        questions = ["question 1", "question 2 2", "question 3 3 3", "question 4 4 4 4"]
        sequences = [[15, eos], [16, 17, eos], [18, 19, 20, 21], [22, 23, 24, eos]]
        original_generate = model.generate

        class ForceSequences(LogitsProcessor):
            def __init__(self, prompt_width):
                self.prompt_width = prompt_width

            def __call__(self, inputs, scores):
                step = inputs.shape[-1] - self.prompt_width
                scores.fill_(-torch.inf)
                for row, sequence in enumerate(sequences):
                    scores[row, sequence[step] if step < len(sequence) else eos] = 0
                return scores

        def forced(**kwargs):
            width = kwargs["input_ids"].shape[-1]
            return original_generate(**kwargs, logits_processor=[ForceSequences(width)])

        padding_side = self.tokenizer.padding_side
        with patch.object(model, "generate", side_effect=forced) as called:
            outputs = sample.generate_answers(model, self.tokenizer, questions, 4)
        called.assert_called_once()
        inputs = called.call_args.kwargs
        for question, row, mask in zip(questions, inputs["input_ids"], inputs["attention_mask"], strict=True):
            expected = self.tokenizer.encode(data.build_prompt(self.tokenizer, question), add_special_tokens=False)
            self.assertEqual(row.tolist(), [pad] * (len(row) - len(expected)) + expected)
            self.assertEqual(row[mask.bool()].tolist(), expected)
        self.assertEqual(outputs, [(self.tokenizer.decode(seq, skip_special_tokens=True), len(seq), eos not in seq)
                                   for seq in sequences])
        self.assertEqual(self.tokenizer.padding_side, padding_side)
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)

    def test_batch_timing_and_actual_tail_batch_size(self):
        self.raw = [raw_row(i) for i in range(50)]

        def generate(model, tokenizer, questions, budget):
            return [(trace(question.rsplit(" ", 1)[-1]), 4, False) for question in questions]

        with patch.object(evaluation, "generate_answers", side_effect=generate) as called, \
                patch.object(evaluation, "time") as clock:
            clock.perf_counter.side_effect = [0, 8, 8, 9, 9, 11, 11, 14, 14, 18, 18, 21, 21, 26]
            report = self.evaluate(original=True)
        self.assertEqual([len(call.args[2]) for call in called.call_args_list], [4, 1])
        self.assertEqual([row["generation_runtime"] for row in report["predictions"]], [3, 4, 5, 6, 8])
        self.assertEqual([row["generation_batch_size"] for row in report["predictions"]], [4, 4, 4, 4, 1])
        self.assertEqual(report["metrics"]["generation_runtime"], 26)
        self.assertEqual(report["metrics"]["generation_samples_per_second"], 5 / 26)

    def test_adaptive_batch_doubles_then_falls_back_and_halves(self):
        self.raw = [raw_row(i) for i in range(290)]
        failures = {3: torch.OutOfMemoryError("test OOM"), 5: RuntimeError("test generation failure")}
        attempts = []

        def generate(model, tokenizer, questions, budget):
            attempts.append(list(questions))
            if len(attempts) in failures:
                raise failures[len(attempts)]
            return [(trace(question.rsplit(" ", 1)[-1]), 4, False) for question in questions]

        def collect():
            self.assertIsNone(sys.exc_info()[0])

        with patch.object(evaluation, "generate_answers", side_effect=generate), \
                patch.object(evaluation, "gc") as garbage:
            garbage.collect.side_effect = collect
            report = self.evaluate(original=True, batch_size=4, auto_batch_size=True)
        self.assertEqual([len(batch) for batch in attempts], [4, 8, 16, 8, 8, 4, 4, 1])
        self.assertEqual(attempts[3], attempts[2][:8])
        self.assertEqual(attempts[5], attempts[4][:4])
        self.assertEqual(garbage.collect.call_count, 2)
        self.assertIn("Retrying question 13 with batch size 8.", self.eval_terminal)
        self.assertIn("Retrying question 21 with batch size 4.", self.eval_terminal)
        self.assertEqual([row["index"] for row in report["predictions"]], report["data"]["indices"])
        self.assertEqual([row["generation_batch_size"] for row in report["predictions"]],
                         [4] * 4 + [8] * 16 + [4] * 8 + [1])
        self.assertEqual(report["metrics"]["answer_accuracy"], 1)
        self.assertTrue(report["generation"]["auto_batch_size"])
        self.assertEqual(report["generation"]["batch_size"], 4)

    def test_adaptive_first_failure_and_single_question_error_preserve_progress(self):
        self.raw = [raw_row(i) for i in range(90)]
        output = self.root / "adaptive.json"
        progress_path = output.with_suffix(".progress.jsonl")
        failure = RuntimeError("one question still failed")
        attempts = []

        def generate(model, tokenizer, questions, budget):
            attempts.append(list(questions))
            if len(attempts) in (1, 4):
                raise RuntimeError("batch failed")
            if len(attempts) == 6:
                raise failure
            return [(trace(question.rsplit(" ", 1)[-1]), 4, False) for question in questions]

        with patch.object(evaluation, "generate_answers", side_effect=generate), self.assertRaises(RuntimeError) as caught:
            self.evaluate(original=True, output=output, batch_size=4, auto_batch_size=True)
        self.assertIs(caught.exception, failure)
        self.assertEqual([len(batch) for batch in attempts], [4, 2, 2, 2, 1, 1])
        self.assertEqual(attempts[0][:2], attempts[1])
        self.assertEqual(attempts[3][:1], attempts[4])
        first = progress_path.read_bytes()
        saved_questions = [json.loads(line)["question"] for line in first.splitlines()]
        self.assertEqual(saved_questions, attempts[1] + attempts[2] + attempts[4])
        def finish(model, tokenizer, questions, budget):
            return [("invalid", 4, True)] * len(questions)

        with patch.object(evaluation, "generate_answers", side_effect=finish) as called:
            report = self.evaluate(original=True, output=output, batch_size=3, auto_batch_size=True)
        self.assertEqual([len(call.args[2]) for call in called.call_args_list], [3, 1])
        self.assertEqual(list(called.call_args_list[0].args[2])[:1], attempts[-1])
        self.assertTrue(progress_path.read_bytes().startswith(first))
        self.assertEqual(report["metrics"]["generation_samples"], 9)
        self.assertEqual(report["generation"]["batch_size"], 3)

    def test_adaptive_failed_short_tail_halves_the_actual_batch(self):
        self.raw = [raw_row(i) for i in range(60)]
        with patch.object(evaluation, "generate_answers", side_effect=[
            [("invalid", 4, True)] * 4, RuntimeError("short tail failed"),
            [("invalid", 4, True)], [("invalid", 4, True)],
        ]) as called:
            report = self.evaluate(original=True, auto_batch_size=True)
        self.assertEqual([len(call.args[2]) for call in called.call_args_list], [4, 2, 1, 1])
        self.assertEqual(report["metrics"]["generation_truncation_rate"], 1)

    def test_fixed_failure_and_adaptive_scoring_or_interrupt_are_not_retried(self):
        for automatic, error in ((False, RuntimeError("fixed failed")), (True, KeyboardInterrupt())):
            with self.subTest(automatic=automatic), patch.object(evaluation, "generate_answers", side_effect=error) as called:
                with self.assertRaises(type(error)) as caught:
                    self.evaluate(original=True, auto_batch_size=automatic)
                self.assertIs(caught.exception, error)
                called.assert_called_once()

        output = self.root / "scoring.json"
        with patch.object(evaluation, "generate_answers", return_value=[("answer", 2, False)] * 2) as generated, \
                patch.object(evaluation, "score_answer", side_effect=[(True, True, False), Interrupted("scoring failed")]):
            with self.assertRaisesRegex(Interrupted, "scoring failed"):
                self.evaluate(original=True, output=output, auto_batch_size=True)
        generated.assert_called_once()
        self.assertEqual(len(output.with_suffix(".progress.jsonl").read_text().splitlines()), 1)

    def test_interruption_skips_evaluation_and_failed_final_evaluation_is_retryable(self):
        with patch.object(train_sft,"evaluate_checkpoint") as called,self.assertRaises(Interrupted):
            self.run_training("stop",trainer=InterruptedTrainer)
        called.assert_not_called()
        with patch.object(train_sft,"evaluate_checkpoint",side_effect=RuntimeError("evaluation failed")),self.assertRaises(RuntimeError):
            self.run_training("failed")
        run=self.root/"failed";weights=(run/"final/adapter_model.safetensors").read_bytes()
        self.evaluate(run/"final")
        self.assertEqual(weights,(run/"final/adapter_model.safetensors").read_bytes())

    def test_bf16_factory_has_fp32_lora_and_finite_checkpointed_gradients(self):
        config={**self.config,"bf16":True}
        with patch.object(AutoModelForCausalLM,"from_pretrained",side_effect=self.model_loader):
            model=train_sft.load_training_model(config)
        self.assertEqual(self.model_calls[-1]["dtype"],torch.bfloat16)
        self.assertTrue(all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.dtype==torch.bfloat16 for p in model.parameters() if not p.requires_grad))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        example=data.build_sft_example(self.tokenizer,"question 1",trace(1))
        batch=DataCollatorForSeq2Seq(self.tokenizer)([example])
        with torch.autocast("cpu",dtype=torch.bfloat16):loss=model(**batch).loss
        loss.backward()
        gradients=[p.grad for n,p in model.named_parameters() if "lora_B" in n]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))

    def test_sampling_original_and_zero_adapter_share_native_input_and_references(self):
        ds,info=self.prepared();plan=workflow.data_plan(ds,data_info=info)
        base=self.model_loader(self.model_name,dtype=torch.float32)
        adapter=self.root/"sample-zero"
        get_peft_model(base,LoraConfig(task_type="CAUSAL_LM",target_modules="all-linear",**self.config["lora"])).save_pretrained(adapter)
        self.tokenizer.save_pretrained(adapter)
        workflow.save_run(adapter,workflow.new_run(self.config,plan,1))
        cfg=self.root/"sample.yaml";cfg.write_text(yaml.safe_dump(self.config))
        reports=[]
        for label,options in [("original",["--model-name",self.model_name]),("lora",["--checkpoint",str(adapter)])]:
            output=self.root/(label+".md")
            argv=["sample_checkpoint.py",*options,"--config",str(cfg),"--device","cpu","--num-samples","2","--local-files-only","--output",str(output)]
            with patch.object(sys,"argv",argv),patch.object(math_data,"load_dataset",side_effect=self.raw_dataset), \
                    patch.object(AutoTokenizer,"from_pretrained",side_effect=self.load_tokenizer), \
                    patch.object(AutoModelForCausalLM,"from_pretrained",side_effect=self.model_loader), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):sample.main()
            reports.append(output.read_text().split("## 样本",1)[1])
        self.assertEqual(reports[0],reports[1])
        self.assertIn("参考最终答案",reports[0])
        self.assertTrue(all(call["local_files_only"] for call in self.model_calls[1:]))

    def test_source_failures_in_all_entrypoints_precede_weight_loading(self):
        run=self.run_training("bad-source");source=run/"final"
        meta=json.loads((source/"run.json").read_text());meta["text_format"]["system"]="changed"
        (source/"run.json").write_text(json.dumps(meta))
        cfg=self.root/"source.yaml";cfg.write_text(yaml.safe_dump(self.config))
        calls=[(train_sft,["train_sft.py","--config",str(cfg),"--continue-from",str(source),"--additional-epochs","1"]),
               (evaluation,["evaluate_checkpoint.py","--checkpoint",str(source)]),
               (sample,["sample_checkpoint.py","--checkpoint",str(source)])]
        for module,argv in calls:
            with patch.object(sys,"argv",argv),patch.object(AutoModelForCausalLM,"from_pretrained",side_effect=AssertionError("weights must not load")), \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):module.main()

    def test_default_original_output_includes_dataset_identity(self):
        ds,info=self.prepared()
        model=self.model_loader(self.model_name,dtype=torch.float32)
        with patch.object(evaluation,"__file__",str(self.root/"evaluate_checkpoint.py")), \
                patch.object(evaluation,"generate_answers",return_value=[(trace(1),4,False)]*2),contextlib.redirect_stdout(io.StringIO()):
            report=evaluation.evaluate_checkpoint(model_name=self.model_name,model=model,tokenizer=self.tokenizer,
                dataset=ds,data_info=info,config=self.config,device="cpu",disable_tqdm=True)
        output=self.root/"outputs/evaluations/original-Qwen3-4B-Instruct-2507-OpenR1-Math-220k-default-train-n2.json"
        self.assertTrue(output.is_file())
        self.assertEqual(json.loads(output.read_text()),report)

    def test_argument_precision_and_atomic_directory_boundaries(self):
        for options in [("--continue-from","/missing"),("--additional-epochs","1"),
                        ("--continue-from","/missing","--additional-epochs","nan")]:
            with patch.object(sys,"argv",["train_sft.py",*options]), \
                    patch.object(math_data,"load_dataset",side_effect=AssertionError("reject before data")), \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):train_sft.main()
        with self.assertRaises(ValueError):data.validate_precision({"bf16":True},"cpu")
        self.assertEqual(data.model_dtype({"bf16":True}),torch.bfloat16)
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths=list(pool.map(lambda _:workflow.reserve_directory(self.root/"atomic"),range(4)))
        self.assertEqual(len(set(paths)),4)
        for count in (0,-1,3):
            with self.assertRaises(SystemExit):self.evaluate(original=True,count=count)
        for batch_size in (0,-1,"1.5"):
            with self.assertRaises(SystemExit):self.evaluate(original=True,batch_size=batch_size)


if __name__ == "__main__":
    unittest.main()
