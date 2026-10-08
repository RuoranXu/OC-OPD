"""JSONL/Parquet training datasets producing veRL's DataProto tensor schema."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .prompts import render_prompt


class TrainingDataset(Dataset):
    def __init__(self, files, tokenizer, config, *, fixed_responses=False):
        from datasets import concatenate_datasets, load_dataset

        paths = [files] if isinstance(files, str) else list(files)
        if not paths:
            raise ValueError("data.train_files must contain at least one training file.")
        sources = []
        for path in paths:
            kind = "parquet" if Path(path).suffix == ".parquet" else "json"
            sources.append(load_dataset(kind, data_files=str(path), split="train"))
        self.rows = concatenate_datasets(sources)
        self.tokenizer, self.config, self.fixed_responses = tokenizer, config, fixed_responses
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Training requires explicit pad_token_id and eos_token_id.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt = row[self.config.prompt_key]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        text = render_prompt(
            self.tokenizer,
            prompt,
            mode=self.config.prompt_mode,
            enable_thinking=self.config.enable_thinking,
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        max_prompt, max_response = self.config.max_prompt_length, self.config.max_response_length
        if not ids or len(ids) > max_prompt:
            raise ValueError(f"Row {index}: prompt length {len(ids)} outside [1, {max_prompt}].")
        padding = max_prompt - len(ids)
        prompt_ids = torch.tensor([self.tokenizer.pad_token_id] * padding + ids)
        prompt_mask = torch.tensor([0] * padding + [1] * len(ids))
        output = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "position_ids": (prompt_mask.cumsum(-1) - 1).clamp_min(0),
            "raw_prompt_ids": ids,
            "data_source": row.get("data_source", "custom"),
            "reward_model": row.get("reward_model", {}),
            "extra_info": row.get("extra_info", {"index": index}),
        }
        if self.fixed_responses:
            response = row.get(self.config.response_key)
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"Row {index}: KD/SFT requires a non-empty fixed response.")
            tokens = self.tokenizer.encode(response, add_special_tokens=False)
            if not tokens or tokens[-1] != self.tokenizer.eos_token_id:
                tokens.append(self.tokenizer.eos_token_id)
            if len(tokens) > max_response:
                raise ValueError(
                    f"Row {index}: response exceeds {max_response}; preprocess explicitly."
                )
            response_ids = torch.tensor(
                tokens + [self.tokenizer.pad_token_id] * (max_response - len(tokens))
            )
            mask = torch.tensor([1] * len(tokens) + [0] * (max_response - len(tokens)))
            output["prompts"] = prompt_ids
            output["responses"] = response_ids
            output["response_mask"] = mask.bool()
            output["input_ids"] = torch.cat((prompt_ids, response_ids))
            output["attention_mask"] = torch.cat((prompt_mask, mask))
            output["position_ids"] = (output["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        return output
