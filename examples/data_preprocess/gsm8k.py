"""Prepare GSM8K for GRPO, on-policy distillation, and fixed-response KD/SFT."""

import argparse
import re
from pathlib import Path

if __package__:
    from .common import add_output_arguments, output_path, prompt_value, read_rows, write_rows
else:
    from common import add_output_arguments, output_path, prompt_value, read_rows, write_rows


def normalize_rows(rows, *, split="train", prompt_mode="chat"):
    output = []
    for index, row in enumerate(rows):
        question, answer = row.get("question"), row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Row {index}: GSM8K requires non-empty question text")
        if not isinstance(answer, str) or "####" not in answer:
            raise ValueError(f"Row {index}: GSM8K answer must contain its #### ground truth")
        gold = answer.rsplit("####", 1)[1].strip().replace(",", "")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", gold) is None:
            raise ValueError(f"Row {index}: invalid GSM8K numeric ground truth {gold!r}")
        question += "\nShow your work and end with #### followed by the answer."
        prompt = f"Question: {question}\nAnswer: " if prompt_mode == "base" else question
        output.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": prompt_value(prompt, prompt_mode),
                "response": answer,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {"split": split, "index": index},
            }
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Train with data.prompt_mode matching --prompt-mode. No chat template is rendered.",
    )
    parser.add_argument("--input", type=Path, help="Local GSM8K-format JSONL/Parquet; no download")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--revision", help="Optional Hugging Face dataset revision/commit")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face download cache directory")
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    try:
        destination = output_path(args, "gsm8k")
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        if args.input is not None:
            if args.revision is not None or args.cache_dir is not None:
                raise ValueError("--revision/--cache-dir apply only to the Hugging Face download")
            rows = read_rows(args.input)
        else:
            from datasets import load_dataset

            rows = load_dataset(
                "openai/gsm8k",
                "main",
                split=args.split,
                revision=args.revision,
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
            )
        write_rows(
            normalize_rows(rows, split=args.split, prompt_mode=args.prompt_mode), destination
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
