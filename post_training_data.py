"""Prepare PRM800K preferences and Skywork math prompts for post-SFT training."""

from collections import Counter
import json

from datasets import Features, List, Value, load_dataset
from datasets.fingerprint import Hasher

from math_scoring import SCORING, parse_answer, score_answer
from prepare_sft_data import SYSTEM_PROMPT, messages

DPO_SYSTEM = "Solve the following math problem step by step.\nEnd your response with the final answer."
SOURCES = {
    "dpo": ("tasksource/prm800k_dpo", "solution", "train"),
    "grpo": ("Skywork/Skywork-OR1-RL-Data", "default", "math"),
}


def prepare_post_training_data(stage, source, tokenizer, held_out, length_limit):
    name, config, split = SOURCES[stage]
    raw = load_dataset(name, config, split=split, revision=source["revision"])
    required = {"prompt", "chosen", "rejected"} if stage == "dpo" else {"prompt", "reward_model", "ability"}
    if not required.issubset(raw.column_names):
        raise ValueError(f"Missing {stage.upper()} data columns: {', '.join(sorted(required - set(raw.column_names)))}")
    held_out = {" ".join(question.split()) for question in held_out}
    system = DPO_SYSTEM if stage == "dpo" else SYSTEM_PROMPT
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    identity = {"protocol": 1, "stage": stage, "source": source, "raw_fingerprint": raw._fingerprint,
                "held_out": sorted(held_out), "length_limit": length_limit,
                "system": system, "template": tokenizer.chat_template, "tokenizer": backend,
                "scoring": SCORING if stage == "grpo" else None}
    if stage == "dpo":
        fields = {key: List(Value("int32")) for key in ("prompt_ids", "chosen_ids", "rejected_ids")}
    else:
        fields = {"prompt": List({"role": Value("string"), "content": Value("string")}),
                  "references": List(Value("string"))}
    features = Features({**fields, "source_index": Value("int64"), "filter_reason": Value("string")})

    def prepare(row, index):
        question = row["prompt"]
        if stage == "grpo":
            chat = question if isinstance(question, list) else []
            users = [message.get("content") for message in chat if message and message.get("role") == "user"]
            question = users[0] if row["ability"] == "math" and len(users) == 1 else None
        example = ({"prompt_ids": [], "chosen_ids": [], "rejected_ids": []} if stage == "dpo"
                   else {"prompt": [], "references": []})
        example.update(source_index=index, filter_reason="")
        if not isinstance(question, str) or not question.strip():
            return {**example, "filter_reason": "invalid_prompt"}
        if " ".join(question.split()) in held_out:
            return {**example, "filter_reason": "held_out_overlap"}
        prompt = messages(question, system=system)
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        if stage == "dpo":
            answers = [row["chosen"], row["rejected"]]
            if any(not isinstance(answer, str) or not answer.strip() for answer in answers) or answers[0] == answers[1]:
                return {**example, "filter_reason": "invalid_pair"}
            # Keep the generation prompt boundary even if BPE could merge across the following newline.
            chosen, rejected = [tokenizer.encode(answer, add_special_tokens=False) for answer in answers]
            eos = tokenizer.eos_token_id
            if any(eos in response for response in (chosen, rejected)) or chosen == rejected:
                return {**example, "filter_reason": "invalid_pair"}
            if len(prompt_ids) + max(len(chosen), len(rejected)) + 1 > length_limit:
                return {**example, "filter_reason": "over_length"}
            return {**example, "prompt_ids": prompt_ids, "chosen_ids": [*chosen, eos],
                    "rejected_ids": [*rejected, eos]}
        if len(prompt_ids) > length_limit:
            return {**example, "filter_reason": "over_length"}
        try:
            values = json.loads(row["reward_model"]["ground_truth"])
        except (KeyError, TypeError, json.JSONDecodeError):
            values = None
        values = values if isinstance(values, list) else []
        references = [value for value in values if isinstance(value, str) and parse_answer(value)]
        if not references:
            return {**example, "filter_reason": "unsupported_reference"}
        return {**example, "prompt": prompt, "references": references}

    prepared = raw.map(prepare, with_indices=True, remove_columns=raw.column_names, features=features,
                       desc=f"Prepare {stage.upper()} math data", new_fingerprint=Hasher.hash(identity))
    reasons = Counter(prepared["filter_reason"])
    dataset = prepared.filter(lambda reason: not reason, input_columns="filter_reason",
                              desc=f"Filter {stage.upper()} math data").remove_columns("filter_reason")
    if not len(dataset):
        raise ValueError(f"No eligible {stage.upper()} examples remain: {dict(reasons)}")
    info = {**source, "raw_fingerprint": raw._fingerprint, "fingerprint": dataset._fingerprint,
            "raw_records": len(raw), "kept_records": len(dataset), "system": system,
            "filter_counts": {reason: count for reason, count in sorted(reasons.items()) if reason}}
    print(f"{stage.upper()} data | raw={len(raw)} | kept={len(dataset)}"
          + "".join(f" | filtered_{reason}={count}" for reason, count in info["filter_counts"].items()), flush=True)
    return dataset, info


def boxed_reward(completions, completion_ids, references, *, eos_token_id, **kwargs):
    rewards = []
    for completion, ids, expected in zip(completions, completion_ids, references, strict=True):
        if not ids or ids[-1] != eos_token_id:
            rewards.append(0.0)
            continue
        text = completion[0]["content"]
        rewards.append(float(any(score_answer(text, reference)[0] for reference in expected)))
    return rewards
