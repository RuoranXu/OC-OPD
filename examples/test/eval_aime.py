"""example: AIME evaluation.

Use a separate evaluation environment; see the README Test section for commands.
Scores are fixed-sample pass@32 and mean correctness (avg@32), in percent.
"""

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from aime_utils import DATA_SOURCES, extract_answer, load_items, render_prompt

ROOT = Path(__file__).resolve().parent
N = 32
PINNED = {"vllm": "0.8.5", "torch": "2.6.0", "transformers": "4.51.3"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def output_lock(directory):
    """OS lock is released after crashes; a stale lock file never blocks resume."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".evaluation.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another evaluator owns {directory}") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another evaluator owns {directory}") from exc
        yield


def source_hashes():
    return {name: file_sha(ROOT / name) for name in ("eval_aime.py", "aime_utils.py")}


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    mismatches = {
        name: version for name, version in versions.items() if version.split("+")[0] != PINNED[name]
    }
    if mismatches:
        raise RuntimeError(
            f"Use examples/test/requirements.txt in a separate environment: {mismatches}"
        )
    return versions


def select_gpus(selection):
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = (
        [part.strip() for part in mask.split(",") if part.strip()] if mask is not None else None
    )
    if visible == [] or visible == ["-1"]:
        raise ValueError("CUDA_VISIBLE_DEVICES exposes no GPU")
    if selection is None:
        return visible[:4] if visible is not None else ["0", "1", "2", "3"]
    logical = [int(part) for part in selection.split(",")]
    if not logical or len(set(logical)) != len(logical) or min(logical) < 0:
        raise ValueError("--gpus must contain distinct nonnegative indices")
    if visible is not None:
        if max(logical) >= len(visible):
            raise ValueError("--gpus indexes devices inside CUDA_VISIBLE_DEVICES")
        return [visible[index] for index in logical]
    return [str(index) for index in logical]


def local_snapshot(value, revision=None):
    path = Path(value).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Expected a model directory: {path}")
        return path.resolve()
    if path.is_absolute() or value.startswith((".", "~")):
        raise FileNotFoundError(value)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(value, revision=revision)).resolve()


def model_identity(directory, adapter_requested=False):
    """Hash actual weights, not only config/name; catch accidental checkpoint changes."""
    if not adapter_requested and any(
        (directory / child / "adapter_config.json").exists()
        for child in (".", "adapter", "lora_adapter")
    ):
        raise ValueError(
            "This export contains a separate LoRA adapter; use --model ORIGINAL_MODEL "
            "--adapter ADAPTER_DIRECTORY, or explicitly merge it first"
        )
    weight_files = sorted(
        set(directory.glob("model*.safetensors")) | set(directory.glob("pytorch_model*.bin"))
    )
    if not (directory / "config.json").exists() or not weight_files:
        actor = directory / "actor" if (directory / "actor").is_dir() else directory
        raise ValueError(
            f"{directory} is not a complete Hugging Face model. "
            "For an unmerged FSDP checkpoint, run "
            f"python -m verl.model_merger merge --backend fsdp --local_dir {actor} "
            "--target_dir /path/to/merged_hf_model in the training environment, then pass "
            "--model /path/to/merged_hf_model. For LoRA, pass the original --model "
            "and --adapter separately."
        )
    for index in directory.glob("*.index.json"):
        missing = [
            name
            for name in set(read(index).get("weight_map", {}).values())
            if not (directory / name).is_file()
        ]
        if missing:
            raise ValueError(f"Missing model shards: {missing}")
    names = set(weight_files) | set(directory.glob("*.json")) | set(directory.glob("*.model"))
    names |= set(directory.glob("*.jinja")) | set(directory.glob("*.txt"))
    return {path.name: file_sha(path) for path in sorted(names) if path.is_file()}


def resolve_adapter(value):
    if not value:
        return None, None
    root = Path(value).expanduser().resolve()
    candidates = [root, root / "adapter", root / "lora_adapter", root / "actor" / "lora_adapter"]
    candidates = [path for path in candidates if (path / "adapter_config.json").is_file()]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one adapter_config.json in the adapter/checkpoint directory: {root}"
        )
    path = candidates[0]
    config = read(path / "adapter_config.json")
    rank, alpha = config.get("r", 0), config.get("lora_alpha", 0)
    if (
        config.get("peft_type") != "LORA"
        or not isinstance(rank, int)
        or rank <= 0
        or not isinstance(alpha, (int, float))
        or alpha <= 0
    ):
        raise ValueError(
            "LoRA requires peft_type=LORA, positive r and lora_alpha; alpha=0 disables the adapter"
        )
    if config.get("use_dora") or config.get("use_rslora") or config.get("modules_to_save"):
        raise ValueError(
            "This pinned vLLM example supports ordinary projection-layer LoRA; "
            "merge other PEFT variants first"
        )
    if config.get("bias", "none") != "none":
        raise ValueError("Merge adapters with trainable bias before using this evaluator")
    weights = [
        path / name
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (path / name).exists()
    ]
    if len(weights) != 1:
        raise ValueError(f"Expected exactly one LoRA weight file in {path}")
    identity = {
        "files": {p.name: file_sha(p) for p in [path / "adapter_config.json", *weights]},
        "config": config,
    }
    return path, identity


def prepare(args, directory):
    from transformers import AutoTokenizer

    versions = runtime_versions()
    items = load_items(args.data, args.cache_dir)
    model = local_snapshot(args.model, args.revision)
    identity = model_identity(model, adapter_requested=bool(args.adapter))
    adapter, adapter_identity = resolve_adapter(args.adapter)
    tokenizer_path = (
        local_snapshot(args.tokenizer, args.tokenizer_revision) if args.tokenizer else model
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tokenizer_files = {
        p.name: file_sha(p)
        for p in sorted(tokenizer_path.iterdir())
        if p.is_file() and (p.suffix in (".json", ".model", ".jinja", ".txt"))
    }
    stops = sorted({tokenizer.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|endoftext|>")})
    if stops != [151643, 151645]:
        raise ValueError(f"Expected Qwen3 EOS IDs [151643, 151645], found {stops}")
    native_context = read(model / "config.json").get("max_position_embeddings")
    if not isinstance(native_context, int) or args.max_model_len > native_context:
        raise ValueError(f"Requested context exceeds the model's native limit: {native_context}")
    prompts = []
    for item in items:
        text = render_prompt(tokenizer, item, args.mode)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        cap = min(args.max_response_tokens, args.max_model_len - len(tokens))
        if cap <= 0:
            raise ValueError(f"Prompt consumes the entire context: {item['unique_id']}")
        prompts.append(
            {
                **item,
                "prompt": text,
                "prompt_token_ids": tokens,
                "response_cap": cap,
                "seed": args.seed + item["year"] * 1000 + item["question_number"],
            }
        )
    gpus = select_gpus(args.gpus)
    if len(gpus) % args.tensor_parallel_size:
        raise ValueError("The number of selected GPUs must be divisible by --tensor-parallel-size")
    gpu_groups = [
        gpus[index : index + args.tensor_parallel_size]
        for index in range(0, len(gpus), args.tensor_parallel_size)
    ]
    protocol = {
        "schema": 1,
        "benchmark": "AIME2024+AIME2025",
        "questions": 60,
        "samples_per_question": N,
        "model": str(model),
        "model_files_sha256": identity,
        "tokenizer": str(tokenizer_path),
        "tokenizer_files_sha256": tokenizer_files,
        "adapter": str(adapter) if adapter else None,
        "adapter_identity": adapter_identity,
        "mode": args.mode,
        "data_sources": DATA_SOURCES
        if args.data is None
        else {"local_file": str(Path(args.data).resolve()), "sha256": file_sha(args.data)},
        "items_sha256": digest(items),
        "prompts_sha256": digest(prompts),
        "source_sha256": source_hashes(),
        "versions": versions,
        "sampling": {
            "n": N,
            "seed": args.seed,
            "question_seed": "seed + year * 1000 + question_number",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "stop_token_ids": stops,
            "max_model_len": args.max_model_len,
            "max_response_tokens": args.max_response_tokens,
        },
        "execution": {
            "gpus": gpus,
            "gpu_groups": gpu_groups,
            "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
            "vllm_use_v1": False,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "inflight_questions_per_gpu": args.inflight,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "swap_space": args.swap_space,
        },
        "metrics": {
            "pass@32": "mean(any(correct in all 32 samples))",
            "avg@32": "mean(correct over all question-sample pairs)",
            "units": "percent",
            "unparsed_answer": "incorrect",
            "scoring": "historical explicit-integer extract_answer",
        },
    }
    protocol["protocol_sha256"] = digest(protocol)
    frozen = directory / "protocol.json"
    if frozen.exists():
        if read(frozen) != protocol or read(directory / "prompts.json") != prompts:
            raise RuntimeError(
                "Output contains a different model/data/prompt/runtime/seed protocol; "
                "use a new --output-dir"
            )
    else:
        if list((directory / "raw").glob("*.json")):
            raise RuntimeError("Raw responses exist without a protocol; use a new output directory")
        write(directory / "prompts.json", prompts)
        write(frozen, protocol)
    return protocol, prompts


def validate_rows(rows, item, protocol):
    if len(rows) != N or sorted(row.get("sample", -1) for row in rows) != list(range(N)):
        raise ValueError(f"Expected exactly samples 0..31: {item['unique_id']}")
    for row in rows:
        if (
            row["id"] != item["unique_id"]
            or row["seed"] != item["seed"]
            or row["gold"] != item["answer"]
            or row["protocol_sha256"] != protocol["protocol_sha256"]
            or row["response_cap"] != item["response_cap"]
        ):
            raise ValueError(f"Response provenance mismatch: {item['unique_id']}")
        if not isinstance(row["text"], str) or not isinstance(row["token_ids"], list):
            raise ValueError("Missing raw text/token IDs")
        if not all(isinstance(token, int) and token >= 0 for token in row["token_ids"]):
            raise ValueError("Invalid output token IDs")
        if row["tokens"] != len(row["token_ids"]) or row["tokens"] > row["response_cap"]:
            raise ValueError("Output token count/cap mismatch")
        if row["finish_reason"] not in ("stop", "length"):
            raise ValueError("Unfinished response")


def score_rows(rows):
    for row in rows:
        row["prediction"] = extract_answer(row["text"])
        row["correct"] = row["prediction"] == row["gold"]
    return rows


def summarize(directory, protocol, prompts):
    expected = {item["unique_id"] + ".json" for item in prompts}
    actual = {path.name for path in (directory / "raw").glob("*.json")}
    if actual != expected:
        raise RuntimeError(
            f"No final score until all 60 questions are complete: {len(actual & expected)}/60; "
            f"unexpected={sorted(actual - expected)}"
        )
    questions = []
    for item in prompts:
        path = directory / "raw" / (item["unique_id"] + ".json")
        rows = read(path)
        validate_rows(rows, item, protocol)
        score_rows(rows)
        write(directory / "scored" / path.name, rows)
        correct = sum(row["correct"] for row in rows)
        questions.append(
            {
                "id": item["unique_id"],
                "year": item["year"],
                "correct": correct,
                "pass": correct > 0,
                "source_raw_sha256": file_sha(path),
            }
        )
    metrics = []
    for year in (2024, 2025, "all"):
        subset = [item for item in questions if year == "all" or item["year"] == year]
        count = len(subset)
        metrics.append(
            {
                "dataset": f"AIME{year}" if year != "all" else "AIME2024+2025",
                "questions": count,
                "samples": count * N,
                "pass@32": 100 * sum(x["pass"] for x in subset) / count,
                "avg@32": 100 * sum(x["correct"] for x in subset) / (count * N),
            }
        )
    result = {
        "protocol_sha256": protocol["protocol_sha256"],
        "metrics": metrics,
        "per_question": questions,
    }
    write(directory / "summary.json", result)
    path = directory / "summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    os.replace(temporary, path)
    return result


async def engine_loop(rank, directory, protocol, prompts, jobs):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind

    execution, sampling = protocol["execution"], protocol["sampling"]
    tokenizer = AutoTokenizer.from_pretrained(protocol["tokenizer"], local_files_only=True)
    kwargs = {
        key: execution[key]
        for key in (
            "dtype",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_chunked_prefill",
            "enable_prefix_caching",
            "swap_space",
        )
    }
    adapter = None
    if protocol["adapter"]:
        rank_limit = protocol["adapter_identity"]["config"]["r"]
        supported = [value for value in (8, 16, 32, 64, 128, 256, 320, 512) if value >= rank_limit]
        if not supported:
            raise ValueError("LoRA rank exceeds vLLM's supported maximum")
        kwargs.update(enable_lora=True, max_lora_rank=supported[0], max_loras=1, max_cpu_loras=1)
        adapter = LoRARequest("evaluated_checkpoint", 1, protocol["adapter"])
    args = AsyncEngineArgs(
        model=protocol["model"],
        max_model_len=sampling["max_model_len"],
        generation_config="vllm",
        seed=sampling["seed"],
        skip_tokenizer_init=True,
        disable_log_requests=True,
        **kwargs,
    )
    engine = AsyncLLMEngine.from_engine_args(args)
    write(
        directory / "runtime" / f"ready_gpu{rank}.json", {"pid": os.getpid(), "time": time.time()}
    )
    lookup = {item["unique_id"]: item for item in prompts}

    async def consumer():
        while True:
            try:
                key = jobs.get_nowait()
            except queue.Empty:
                return
            item = lookup[key]
            params = SamplingParams(
                n=N,
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=item["response_cap"],
                seed=item["seed"],
                stop_token_ids=sampling["stop_token_ids"],
                detokenize=False,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            final = None
            try:
                async for event in engine.generate(
                    {"prompt_token_ids": item["prompt_token_ids"]},
                    params,
                    request_id=key,
                    lora_request=adapter,
                ):
                    final = event
                if (
                    final is None
                    or not final.finished
                    or list(final.prompt_token_ids) != item["prompt_token_ids"]
                ):
                    raise RuntimeError(f"Incomplete generation or changed prompt: {key}")
                rows = [
                    {
                        "id": key,
                        "year": item["year"],
                        "sample": output.index,
                        "seed": item["seed"],
                        "gold": item["answer"],
                        "text": tokenizer.decode(output.token_ids, skip_special_tokens=True),
                        "token_ids": list(output.token_ids),
                        "tokens": len(output.token_ids),
                        "response_cap": item["response_cap"],
                        "finish_reason": output.finish_reason,
                        "protocol_sha256": protocol["protocol_sha256"],
                        "gpu": rank,
                    }
                    for output in final.outputs
                ]
                rows.sort(key=lambda row: row["sample"])
                validate_rows(rows, item, protocol)
                write(directory / "raw" / (key + ".json"), score_rows(rows))
                print(
                    f"GPU {rank}: completed {key}, "
                    f"{sum(row['tokens'] for row in rows)} output tokens",
                    flush=True,
                )
            except BaseException:
                await engine.abort(key)
                raise

    tasks = [
        asyncio.create_task(consumer()) for _ in range(execution["inflight_questions_per_gpu"])
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        engine.shutdown_background_loop()


def worker(rank, directory, protocol, prompts, jobs):
    directory = Path(directory)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(name, None)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=",".join(protocol["execution"]["gpu_groups"][rank]),
        VLLM_USE_V1="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    if hasattr(os, "setsid"):
        os.setsid()
    parent_pid = os.getppid()

    def watch_parent():
        while True:
            time.sleep(2)
            if os.getppid() != parent_pid:
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return

    threading.Thread(target=watch_parent, daemon=True).start()
    with (directory / "runtime" / f"gpu{rank}.log").open("a", encoding="utf-8", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            asyncio.run(engine_loop(rank, directory, protocol, prompts, jobs))
        except BaseException:
            write(
                directory / "runtime" / f"failure_gpu{rank}.json",
                {"time": time.time(), "error": traceback.format_exc()},
            )
            raise


def run(directory, protocol, prompts):
    context = mp.get_context("spawn")
    jobs = context.Queue()
    pending = []
    for item in sorted(
        prompts, key=lambda item: (-len(item["prompt_token_ids"]), item["unique_id"])
    ):
        path = directory / "raw" / (item["unique_id"] + ".json")
        if path.exists():
            validate_rows(read(path), item, protocol)
        else:
            pending.append(item)
            jobs.put(item["unique_id"])
    if not pending:
        return
    (directory / "runtime").mkdir(exist_ok=True)
    children = []
    try:
        for rank in range(len(protocol["execution"]["gpu_groups"])):
            process = context.Process(
                target=worker, args=(rank, str(directory), protocol, prompts, jobs)
            )
            process.start()
            children.append(process)
        while any(process.is_alive() for process in children):
            for process in children:
                if process.exitcode not in (None, 0):
                    raise RuntimeError(
                        f"Inference worker {process.pid} failed; inspect runtime/gpu*.log. "
                        "Complete questions are retained; no samples were shortened or skipped."
                    )
            completed = len(list((directory / "raw").glob("*.json")))
            write(
                directory / "status.json",
                {
                    "phase": "generating",
                    "completed_questions": completed,
                    "total_questions": 60,
                    "time": time.time(),
                    "worker_pids": [p.pid for p in children],
                },
            )
            time.sleep(2)
        if any(process.exitcode != 0 for process in children):
            raise RuntimeError("Inference failed; inspect runtime/gpu*.log")
    finally:
        for process in children:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except (AttributeError, PermissionError):
                if process.is_alive():
                    process.terminate()
        for process in children:
            process.join(timeout=10)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except (AttributeError, PermissionError):
                if process.is_alive():
                    process.kill()
            if process.is_alive():
                process.join()
        jobs.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", help="Complete HF model directory or Hub ID; original model when using --adapter"
    )
    parser.add_argument(
        "--revision", help="Hub model revision (resolved snapshot and weight hashes are frozen)"
    )
    parser.add_argument(
        "--tokenizer", help="Optional original tokenizer when exported weights omit tokenizer files"
    )
    parser.add_argument("--tokenizer-revision")
    parser.add_argument(
        "--adapter", help="LoRA directory, or checkpoint containing adapter/ or actor/lora_adapter/"
    )
    parser.add_argument("--mode", choices=("base", "nonthinking", "thinking"), default="base")
    parser.add_argument(
        "--data", help="Optional canonical local JSON; default downloads pinned AIME datasets"
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/aime24_25"))
    parser.add_argument(
        "--gpus",
        help="Logical GPU indices within CUDA_VISIBLE_DEVICES; default first four visible GPUs",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="GPUs per engine; TP=1 reproduces historical independent replicas",
    )
    parser.add_argument("--seed", type=int, default=604)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-response-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--inflight", type=int, default=3, help="Concurrent n=32 question requests per GPU"
    )
    parser.add_argument(
        "--swap-space",
        type=float,
        default=0.0,
        help="CPU KV swap GiB/GPU; nonzero explicitly changes the historical runtime",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze inputs and protocol without starting GPU engines",
    )
    mode.add_argument(
        "--score-only",
        action="store_true",
        help="Rescore complete saved responses without model/GPU imports",
    )
    args = parser.parse_args()
    if not args.score_only and not args.model:
        parser.error("--model is required except with --score-only")
    if (
        min(
            args.max_model_len, args.max_response_tokens, args.inflight, args.max_num_batched_tokens
        )
        <= 0
        or args.max_num_seqs < N
    ):
        parser.error("Positive lengths/inflight and --max-num-seqs >= 32 are required")
    if not 0 < args.gpu_memory_utilization < 1 or args.swap_space < 0 or args.seed < 0:
        parser.error("Invalid GPU memory utilization, swap space, or seed")
    if not args.prepare_only and not args.score_only and os.name != "posix":
        parser.error(
            "GPU evaluation requires Linux; --prepare-only and --score-only "
            "can run without Linux GPU engines"
        )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    directory = args.output_dir.resolve()
    with output_lock(directory):
        try:
            if args.score_only:
                protocol, prompts = (
                    read(directory / "protocol.json"),
                    read(directory / "prompts.json"),
                )
                checksum = protocol.pop("protocol_sha256")
                if digest(protocol) != checksum:
                    raise RuntimeError("Saved protocol checksum mismatch")
                protocol["protocol_sha256"] = checksum
                if (
                    protocol["source_sha256"] != source_hashes()
                    or digest(prompts) != protocol["prompts_sha256"]
                ):
                    raise RuntimeError(
                        "Scoring source or prompt records differ from the frozen protocol"
                    )
            else:
                protocol, prompts = prepare(args, directory)
                if args.prepare_only:
                    write(
                        directory / "status.json",
                        {
                            "phase": "prepared",
                            "protocol_sha256": protocol["protocol_sha256"],
                            "time": time.time(),
                        },
                    )
                    print(f"Prepared 60 questions; protocol: {directory / 'protocol.json'}")
                    return
                run(directory, protocol, prompts)
            result = summarize(directory, protocol, prompts)
            write(
                directory / "status.json",
                {"phase": "complete", "questions": 60, "samples": 60 * N, "time": time.time()},
            )
            print(json.dumps(result["metrics"], indent=2))
            print(f"Saved {directory / 'summary.csv'}")
        except BaseException:
            write(
                directory / "status.json",
                {"phase": "failed", "time": time.time(), "error": traceback.format_exc()},
            )
            raise


if __name__ == "__main__":
    main()
