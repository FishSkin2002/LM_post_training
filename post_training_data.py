"""Prepare PRM800K preferences and Skywork math prompts for post-SFT training."""

from collections import Counter
import json
import re

from datasets import Features, List, Value, load_dataset
from datasets.fingerprint import Hasher

from math_scoring import SCORING, boxed_answer, parse_answer, score_answer
from prepare_sft_data import SYSTEM_PROMPT, messages

DPO_ANSWER_FORMAT = "boxed-v1"
ANSWER_HEADING = re.compile(r"^# Answer[ \t]*\r?$", re.MULTILINE)
FINAL_BOX = re.compile(r"\\boxed[ \t]*\{(.*)\}")
MATH_WRAPPERS = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
MATH_DELIMITER = re.compile(r"(?<!\\)(?:\\\\)*(?:\$|\\[()\[\]])")
SOURCES = {
    "dpo": ("tasksource/prm800k_dpo", "solution", "train"),
    "grpo": ("Skywork/Skywork-OR1-RL-Data", "default", "math"),
}


def _format_dpo_answer(text):
    """Return the reformatted reply and original final answer, if safely identifiable."""
    headings = list(ANSWER_HEADING.finditer(text))
    if headings:
        heading = headings[-1]
        final = text[heading.end():].strip()
    else:
        start = text.rfind(r"\boxed")
        if start < 0:
            return None
        final = text[start:].rstrip()

    if len(final.splitlines()) != 1:
        return None
    if headings:
        for opening, closing in MATH_WRAPPERS:
            if final.startswith(opening) and final.endswith(closing):
                inner = final[len(opening):-len(closing)].strip()
                if not MATH_DELIMITER.search(inner):
                    final = inner
                    break
    existing = FINAL_BOX.fullmatch(final)
    if existing:
        content = existing[1].strip()
        box = final
    elif headings:
        content = final
        box = rf"\boxed{{{content}}}"
    else:
        return None
    if MATH_DELIMITER.search(content) or boxed_answer(box) != content:
        return None
    return (text[:heading.start()] + box if headings else text), content


def prepare_post_training_data(stage, source, tokenizer, held_out, length_limit):
    name, config, split = SOURCES[stage]
    raw = load_dataset(name, config, split=split, revision=source["revision"])
    required = {"prompt", "chosen", "rejected"} if stage == "dpo" else {"prompt", "reward_model", "ability"}
    if not required.issubset(raw.column_names):
        raise ValueError(f"Missing {stage.upper()} data columns: {', '.join(sorted(required - set(raw.column_names)))}")
    held_out = {" ".join(question.split()) for question in held_out}
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    identity = {"protocol": 1, "stage": stage, "source": source, "raw_fingerprint": raw._fingerprint,
                "held_out": sorted(held_out), "length_limit": length_limit,
                "system": SYSTEM_PROMPT, "template": tokenizer.chat_template, "tokenizer": backend,
                "scoring": SCORING}
    if stage == "dpo":
        identity["answer_format"] = DPO_ANSWER_FORMAT
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
        prompt = messages(question)
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        if stage == "dpo":
            answers = [row["chosen"], row["rejected"]]
            if any(not isinstance(answer, str) or not answer.strip() for answer in answers) or answers[0] == answers[1]:
                return {**example, "filter_reason": "invalid_pair"}
            chosen_answer, rejected_answer = map(_format_dpo_answer, answers)
            if chosen_answer is None or rejected_answer is None:
                return {**example, "filter_reason": "invalid_answer_format"}
            if not parse_answer(chosen_answer[1]):
                return {**example, "filter_reason": "unsupported_chosen_answer"}
            # Keep the generation prompt boundary even if BPE could merge across the following newline.
            chosen = tokenizer.encode(chosen_answer[0], add_special_tokens=False)
            rejected = tokenizer.encode(rejected_answer[0], add_special_tokens=False)
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
            "raw_records": len(raw), "kept_records": len(dataset), "system": SYSTEM_PROMPT,
            "filter_counts": {reason: count for reason, count in sorted(reasons.items()) if reason}}
    if stage == "dpo":
        info["answer_format"] = DPO_ANSWER_FORMAT
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
