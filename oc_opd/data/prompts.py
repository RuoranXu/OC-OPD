"""Prompt rendering shared by rollout and fixed-response distillation."""


def render_prompt(tokenizer, prompt, *, mode="chat", enable_thinking=False):
    if mode == "base":
        if not isinstance(prompt, str):
            raise ValueError("Base prompts must be explicit plain strings, not chat messages.")
        return prompt
    if mode != "chat" or not isinstance(prompt, (list, tuple)) or not prompt:
        raise ValueError("Chat prompts must be a non-empty list of role/content messages.")
    messages = list(prompt)
    if messages[-1].get("role") != "user":
        raise ValueError("Training chat prompts must end with a user turn.")
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def validate_tokenizers(student, teacher):
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError(
            "Teacher/student token-to-ID mappings differ; full-vocabulary KD is invalid."
        )
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(student, key) != getattr(teacher, key):
            raise ValueError(f"Teacher/student {key} differs.")
