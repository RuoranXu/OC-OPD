"""Prepare the GSM8K TRAIN split for GRPO, distillation, or fixed-response KD/SFT."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-mode", choices=("base", "chat"), default="chat")
    args = parser.parse_args()
    data = load_dataset("openai/gsm8k", "main", split="train")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for i, row in enumerate(data):
            question = (
                row["question"] + "\nShow your work and end with #### followed by the answer."
            )
            prompt = (
                f"Question: {question}\nAnswer: "
                if args.prompt_mode == "base"
                else [{"role": "user", "content": question}]
            )
            stream.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "response": row["answer"],
                        "data_source": "openai/gsm8k",
                        "reward_model": {"ground_truth": row["answer"].split("####")[-1].strip()},
                        "extra_info": {"index": i},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
