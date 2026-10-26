"""Build and inspect GSM8K SFT examples without loading model weights."""

from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"
IGNORE_INDEX = -100


def build_sft_example(tokenizer, question: str, answer: str) -> dict[str, list[int]]:
    """Return unpadded model inputs with loss restricted to the completion."""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
    )
    encoded = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    prompt_ids = prompt["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("The prompt must be an exact token prefix of the full example.")
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "labels": [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    }


def main():
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    row = dataset[0]
    example = build_sft_example(tokenizer, row["question"], row["answer"])
    supervised_ids = [label for label in example["labels"] if label != IGNORE_INDEX]
    print("【完整训练输入】")
    print(tokenizer.decode(example["input_ids"], skip_special_tokens=False))
    print("【参与损失的文本】")
    print(tokenizer.decode(supervised_ids, skip_special_tokens=False))
    print("input_ids:", example["input_ids"])
    print("labels:", example["labels"])


if __name__ == "__main__":
    main()
