"""veRL-compatible row validation and JSONL/Parquet I/O."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    elif path.suffix.lower() in (".jsonl", ".json"):
        text = path.read_text(encoding="utf-8-sig")
        if text.lstrip().startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.split("\n") if line.strip()]
    else:
        raise ValueError("Input must be .jsonl, .json, or .parquet")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Input must contain at least one object row")
    return rows


def prompt_value(value, mode):
    """Select the trainer's schema; never apply a tokenizer/chat template."""
    if mode not in ("chat", "base"):
        raise ValueError("Prompt mode must be chat or base")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("Prompt must not be empty")
        return value if mode == "base" else [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise ValueError("Prompt must be text or a non-empty list of role/content messages")
    for message in value:
        if not isinstance(message, dict) or message.get("role") not in (
            "system",
            "user",
            "assistant",
            "tool",
        ):
            raise ValueError("Every prompt message requires a valid role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Every prompt message requires non-empty text content")
    if value[-1]["role"] != "user":
        raise ValueError("Training prompt messages must end in a user turn")
    if mode == "base":
        if len(value) != 1 or value[0]["role"] != "user":
            raise ValueError("Convert multi-turn messages to an explicit Base prompt first")
        return value[0]["content"]
    return value


def output_path(args, dataset):
    if args.output is not None:
        if args.output_dir is not None:
            raise ValueError("Use either --output or --output-dir")
        path = args.output
        if path.suffix.lower() not in (".jsonl", ".parquet"):
            raise ValueError("--output must end in .jsonl or .parquet")
        if args.format is not None and path.suffix.lower() != "." + args.format:
            raise ValueError("--format does not match --output suffix")
        return path
    directory = args.output_dir or REPO_ROOT / "data" / "processed" / dataset / args.prompt_mode
    return directory / (args.split + "." + (args.format or "parquet"))


def add_output_arguments(parser):
    parser.add_argument(
        "--prompt-mode",
        choices=("chat", "base"),
        default="chat",
        help="Match trainer data.prompt_mode: messages for chat, plain text for base",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: data/processed/<dataset>/<prompt-mode>)",
    )
    parser.add_argument(
        "--output", type=Path, help="Exact .jsonl or .parquet path; never overwrite"
    )
    parser.add_argument("--format", choices=("parquet", "jsonl"), help="Default: parquet")


def write_rows(rows, path):
    """Validate/serialize before opening the destination; existing files stay intact."""
    if not rows:
        raise ValueError("Refusing to write an empty training dataset")
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = list(dict.fromkeys(key for row in rows for key in row))
        table = pa.Table.from_pylist([{key: row.get(key) for key in columns} for row in rows])
        payload = pa.BufferOutputStream()
        pq.write_table(table, payload, compression="zstd")
        content = payload.getvalue().to_pybytes()
    elif path.suffix.lower() == ".jsonl":
        content = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode()
    else:
        raise ValueError("Output must be .jsonl or .parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    print(json.dumps({"output": str(path.resolve()), "rows": len(rows)}, ensure_ascii=False))
