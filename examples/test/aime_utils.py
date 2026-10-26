"""Frozen AIME24/25 data and prompts for the historical n=32 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

PROMPT_SUFFIX = r"Please solve the following problem and put your final answer within \boxed{}."
DATA_SOURCES = {
    "2024": {
        "repo_id": "math-ai/aime24",
        "revision": "83a7f387baaa524a8bda0022eac0541582297103",
        "filename": "test-00000-of-00001.parquet",
        "sha256": "2602156eb2a0ab64393dc46617a3bfee3451ae3c985a7ea6ac2b55436e094a38",
        "numbering": "source id 60..89 maps to AIME2024-01..30",
    },
    "2025": {
        "repo_id": "math-ai/aime25",
        "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
        "filename": "test.jsonl",
        "sha256": "b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331",
        "numbering": "source id 0..29 maps to AIME2025-01..30",
        "transform": "restore 12 historical line breaks in 8 problems; no content changes",
    },
}
# The published JSONL flattens these line breaks to spaces. Restore the exact
# problem strings used in the previous evaluation, after checking its file SHA.
# Positions refer to Unicode characters, not bytes; replacements preserve length.
_AIME25_LINE_BREAKS = {
    3: [471],
    7: [480],
    10: [293, 1255],
    11: [348, 662],
    18: [310],
    20: [471],
    21: [263, 673],
    23: [445, 871],
}
FROZEN_ITEMS_SHA256 = "dc46f1783725bbc733577159ea5805f099fdbc37901aafb7656f6ff767f9975d"
_CANONICAL_ID = re.compile(r"AIME(20\d{2})-(\d{2})\Z")


def extract_answer(text):
    final = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    candidates = []
    for m in re.finditer(r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}", final):
        candidates.append((m.start(), str(int(m.group(1)))))
    pattern = (
        r"(?im)^\s*(?:\*\*)?(?:(?:the\s+)?final\s+)?answer(?:\*\*)?\s*(?::|is)"
        r"\s*(?:\*\*)?\s*\$?\s*([0-9]{1,3})\s*\$?\s*(?:\*\*)?\s*[.!]?\s*$"
    )
    for m in re.finditer(pattern, final):
        candidates.append((m.start(), str(int(m.group(1)))))
    return max(candidates)[1] if candidates else None


def normalize_gold(value):
    """AIME answers are integers in [0, 999]; never silently round a gold."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Invalid AIME answer: {value!r}")
    text = str(value).strip()
    if re.fullmatch(r"[0-9]{1,3}", text) is None:
        raise ValueError(f"Invalid AIME answer: {value!r}")
    return str(int(text))


def _read_rows(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.split("\n") if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected a JSON array or JSONL objects")
    return rows


def _download(source, cache_dir):
    cache = Path(cache_dir).expanduser()
    target = cache / source["repo_id"].replace("/", "--") / source["revision"] / source["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        url = (
            "https://huggingface.co/datasets/"
            + source["repo_id"]
            + "/resolve/"
            + source["revision"]
            + "/"
            + source["filename"]
        )
        request = Request(url, headers={"User-Agent": "OC-OPD-AIME-evaluation"})
        with urlopen(request, timeout=120) as response:
            body = response.read()
        if hashlib.sha256(body).hexdigest() != source["sha256"]:
            raise ValueError("Downloaded AIME source checksum differs: " + url)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(target.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError(
            f"Cached AIME source checksum differs; remove and download again: {target}"
        )
    return target


def _public_rows(cache_dir):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Install examples/test/requirements.txt to read the AIME24 parquet"
        ) from exc
    rows = []
    for year in (2024, 2025):
        source = DATA_SOURCES[str(year)]
        path = _download(source, cache_dir)
        published = parquet.read_table(path).to_pylist() if year == 2024 else _read_rows(path)
        expected_ids = list(range(60, 90)) if year == 2024 else list(range(30))
        if len(published) != 30 or [int(row["id"]) for row in published] != expected_ids:
            raise ValueError(f"AIME{year} source numbering changed")
        for number, row in enumerate(published, 1):
            problem = row["problem"]
            if year == 2025:
                characters = list(problem)
                for offset in _AIME25_LINE_BREAKS.get(number, []):
                    if characters[offset] != " ":
                        raise ValueError(
                            f"Historical newline restoration failed: AIME2025-{number:02d}"
                        )
                    characters[offset] = "\n"
                problem = "".join(characters)
                answer = row["answer"]
            else:
                boxed = re.findall(r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}", row["solution"])
                if not boxed:
                    raise ValueError(f"Missing source gold: AIME2024-{number:02d}")
                answer = boxed[-1]
            rows.append(
                dict(
                    unique_id=f"AIME{year}-{number:02d}",
                    year=year,
                    problem=problem,
                    answer=normalize_gold(answer),
                )
            )
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if digest != FROZEN_ITEMS_SHA256:
        raise ValueError(
            "Downloaded AIME problem text/gold differs from the frozen 60-question set"
        )
    return rows


def load_items(data_path=None, cache_dir=None):
    """Load all 60 questions; a custom 90-question AIME24--26 JSONL is allowed.

    Custom rows need canonical ``unique_id`` (or ``id``), ``year``, ``problem``,
    and ``answer`` (or ``gold``). Input text is preserved, including whitespace.
    The two-digit canonical number controls the per-question generation seed.
    """
    if data_path is None:
        if cache_dir is None:
            cache_dir = (
                Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "oc_opd" / "aime"
            )
        rows = _public_rows(cache_dir)
    else:
        rows = _read_rows(data_path)
    items = {}
    for row in rows:
        qid = row.get("unique_id", row.get("id"))
        match = _CANONICAL_ID.fullmatch(qid) if isinstance(qid, str) else None
        if match is None:
            raise ValueError(
                f"Expected canonical question id AIME2024-01 or AIME2025-01, got {qid!r}"
            )
        year, number = map(int, match.groups())
        if str(row.get("year")) != str(year):
            raise ValueError(f"Question id/year mismatch: {qid}")
        if year not in (2024, 2025):
            continue
        if not 1 <= number <= 30:
            raise ValueError(f"Question number outside 1..30: {qid}")
        if qid in items:
            raise ValueError(f"Duplicate question id: {qid}")
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"Missing problem text: {qid}")
        answer = normalize_gold(row.get("answer", row.get("gold")))
        items[qid] = dict(
            unique_id=qid,
            year=year,
            problem=problem,
            answer=answer,
            question_number=number,
            problem_sha256=hashlib.sha256(problem.encode()).hexdigest(),
        )
    required = {f"AIME{year}-{number:02d}" for year in (2024, 2025) for number in range(1, 31)}
    if set(items) != required:
        raise ValueError(
            "AIME24 and AIME25 each require all 30 questions; missing: "
            + ", ".join(sorted(required - set(items)))
        )
    return [items[qid] for qid in sorted(items)]


def render_prompt(tokenizer, item, mode="base"):
    """Keep the historical task text and explicit Qwen3 prompt mode."""
    content = item["problem"] + "\n\n" + PROMPT_SUFFIX
    if mode == "base":
        prompt = "Problem:\n" + content + "\n\nSolution:\n"
        if any(marker in prompt for marker in ("<|im_start|>", "<think>", "</think>")):
            raise ValueError("Base completion contains chat/thinking markers")
        return prompt
    if mode not in ("nonthinking", "thinking"):
        raise ValueError("Prompt mode must be base, nonthinking, or thinking")
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        raise ValueError("A Qwen3 chat tokenizer is required for thinking/nonthinking mode")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=(mode == "thinking"),
    )
    ending = "<think>\n\n</think>\n\n" if mode == "nonthinking" else "<think>\n"
    if not prompt.endswith(ending):
        raise ValueError(f"The tokenizer does not render the expected Qwen3 {mode} prefix")
    return prompt
