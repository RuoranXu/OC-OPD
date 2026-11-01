"""Normalize local JSONL/Parquet into the training dataset schema."""

import argparse
import re
from pathlib import Path

if __package__:
    from .common import add_output_arguments, output_path, prompt_value, read_rows, write_rows
else:
    from common import add_output_arguments, output_path, prompt_value, read_rows, write_rows


def normalize_rows(
    rows,
    *,
    prompt_key="prompt",
    response_key="response",
    ground_truth_key=None,
    data_source="custom",
    split="train",
    prompt_mode="chat",
    require_response=False,
    require_ground_truth=False,
    ability=None,
):
    output = []
    for index, row in enumerate(rows):
        if prompt_key not in row:
            raise ValueError(f"Row {index}: missing prompt field {prompt_key!r}")
        source = row.get("data_source", data_source)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Row {index}: data_source must be non-empty text")
        item = {"data_source": source, "prompt": prompt_value(row[prompt_key], prompt_mode)}
        response = row.get(response_key)
        if response is not None:
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"Row {index}: response must be non-empty text")
            item["response"] = response
        elif require_response:
            raise ValueError(f"Row {index}: KD/SFT requires response field {response_key!r}")
        reward = row.get("reward_model")
        if reward is not None and not isinstance(reward, dict):
            raise ValueError(f"Row {index}: reward_model must be an object")
        reward = dict(reward or {})
        if ground_truth_key is not None:
            if ground_truth_key not in row:
                raise ValueError(f"Row {index}: missing ground truth field {ground_truth_key!r}")
            ground_truth = row[ground_truth_key]
            if not isinstance(ground_truth, (str, int, float)) or isinstance(ground_truth, bool):
                raise ValueError(f"Row {index}: ground truth must be text or a number")
            ground_truth = str(ground_truth)
            if "ground_truth" in reward and str(reward["ground_truth"]) != ground_truth:
                raise ValueError(f"Row {index}: conflicting ground truth fields")
            reward["ground_truth"] = ground_truth
        if require_ground_truth and reward.get("ground_truth") in (None, ""):
            raise ValueError(f"Row {index}: GRPO requires reward_model.ground_truth")
        if reward:
            reward.setdefault("style", "rule")
            item["reward_model"] = reward
        row_ability = row.get("ability", ability)
        if row_ability is not None:
            item["ability"] = row_ability
        extra = row.get("extra_info")
        if extra is not None and not isinstance(extra, dict):
            raise ValueError(f"Row {index}: extra_info must be an object")
        item["extra_info"] = {"split": split, "index": index, **(extra or {})}
        output.append(item)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use --require-response for KD/SFT; --require-ground-truth for GRPO. "
        "Prompts remain raw text/messages, with model templates applied by the trainer.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Local JSONL/JSON/Parquet")
    parser.add_argument("--dataset", default="custom", help="Output directory name")
    parser.add_argument(
        "--split", default="train", help="Output split name; no automatic data split"
    )
    parser.add_argument("--data-source", help="Default reward routing name (default: --dataset)")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--response-key", default="response")
    parser.add_argument(
        "--ground-truth-key", help="Top-level gold field; existing reward_model kept"
    )
    parser.add_argument("--ability", help="Optional task family, e.g. math")
    parser.add_argument("--require-response", action="store_true")
    parser.add_argument("--require-ground-truth", action="store_true")
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", x) is None
            for x in (args.dataset, args.split)
        ):
            raise ValueError("--dataset/--split must be simple names without path separators")
        destination = output_path(args, args.dataset)
        rows = normalize_rows(
            read_rows(args.input),
            prompt_key=args.prompt_key,
            response_key=args.response_key,
            ground_truth_key=args.ground_truth_key,
            data_source=args.data_source or args.dataset,
            split=args.split,
            prompt_mode=args.prompt_mode,
            require_response=args.require_response,
            require_ground_truth=args.require_ground_truth,
            ability=args.ability,
        )
        write_rows(rows, destination)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
