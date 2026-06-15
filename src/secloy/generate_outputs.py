from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from . import data as secloy_data
from .config import check_required_keys, get_key, load_config, make_run_dir, repo_path, save_resolved_config


def require_generation_imports() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ModuleNotFoundError as exc:
        raise SystemExit("missing generation dependency; install project dependencies before generating outputs") from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "set_seed": set_seed,
    }


def dtype_from_config(torch: Any, value: str | None) -> Any:
    if value in {None, "auto"}:
        return "auto"
    options = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in options:
        raise ValueError(f"unsupported model dtype: {value}")
    return options[value]


def model_kwargs(config: dict[str, Any], torch: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(get_key(config, "model.trust_remote_code") or False),
        "torch_dtype": dtype_from_config(torch, get_key(config, "model.dtype")),
    }
    attn_implementation = get_key(config, "model.attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return kwargs


def load_prompt_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(secloy_data.iter_jsonl(path, limit=limit), 1):
        row_id = row.get("id") or row.get("prompt_id")
        prompt = row.get("prompt") or row.get("prompt_messages")
        if row_id is None:
            raise ValueError(f"{path}:{idx}: missing id")
        if prompt is None:
            raise ValueError(f"{path}:{idx}: missing prompt")
        rows.append({"id": str(row_id), "prompt": normalize_prompt(prompt)})
    if not rows:
        raise ValueError(f"{path}: no prompt rows loaded")
    return rows


def normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list):
        raise ValueError(f"prompt must be a string or message list, got {type(prompt).__name__}")
    messages = []
    for idx, message in enumerate(prompt, 1):
        if not isinstance(message, dict):
            raise ValueError(f"prompt message {idx} must be an object")
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        messages.append({"role": role, "content": content})
    return messages


def chat_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chunks = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
    chunks.append("assistant:")
    return "\n".join(chunks)


def load_model_and_tokenizer(config: dict[str, Any], adapter_path: str | Path | None, imports: dict[str, Any]) -> tuple[Any, Any]:
    model_name = str(get_key(config, "model.name"))
    trust_remote_code = bool(get_key(config, "model.trust_remote_code") or False)
    tokenizer = imports["AutoTokenizer"].from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = imports["AutoModelForCausalLM"].from_pretrained(model_name, **model_kwargs(config, imports["torch"]))
    if adapter_path:
        model = imports["PeftModel"].from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def batch_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def generation_kwargs(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    generation = config.get("generation", {})
    do_sample = bool(generation.get("do_sample", False))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation.get("max_new_tokens", 220)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(generation.get("temperature", 1.0))
        kwargs["top_p"] = float(generation.get("top_p", 1.0))
    return kwargs


def generate_rows(
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    input_file: Path,
    adapter_path: str | Path | None,
) -> list[dict[str, Any]]:
    imports = require_generation_imports()
    torch = imports["torch"]
    max_seq_len = int(get_key(config, "model.max_seq_len") or 2048)
    batch_size = int(get_key(config, "generation.batch_size") or 1)
    generated_rows = []
    for batch in batch_rows(rows, batch_size):
        texts = [chat_text(tokenizer, row["prompt"]) for row in batch]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs(config, tokenizer))
        prompt_width = inputs["input_ids"].shape[1]
        for batch_idx, (row, output) in enumerate(zip(batch, output_ids, strict=True)):
            new_tokens = output[prompt_width:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            completion_token_count = int((new_tokens != tokenizer.pad_token_id).sum().item())
            generated_rows.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "completion": completion,
                    "model": get_key(config, "model.name"),
                    "adapter": str(adapter_path) if adapter_path else None,
                    "input_file": str(input_file),
                    "prompt_tokens": int(inputs["attention_mask"][batch_idx].sum().item()),
                    "completion_tokens": completion_token_count,
                    "generation": config.get("generation", {}),
                }
            )
        if len(generated_rows) % 200 == 0 or len(generated_rows) == len(rows):
            print(f"generated {len(generated_rows)}/{len(rows)} rows", file=sys.stderr, flush=True)
    return generated_rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def dry_run(config: dict[str, Any], input_file: Path, output_file: Path, limit: int | None) -> int:
    rows = load_prompt_rows(input_file, limit=limit)
    secloy_data.print_json(
        {
            "ok": True,
            "mode": "dry_run",
            "input_file": str(input_file),
            "output_file": str(output_file),
            "rows_loaded": len(rows),
            "first_row": rows[0],
            "model": get_key(config, "model.name"),
        }
    )
    return 0


def run_generation(
    config: dict[str, Any],
    input_file: Path,
    output_file: Path,
    run_dir: Path,
    adapter_path: str | Path | None,
    limit: int | None,
) -> Path:
    imports = require_generation_imports()
    seed = int(get_key(config, "project.seed") or 0)
    imports["set_seed"](seed)
    random.seed(seed)
    rows = load_prompt_rows(input_file, limit=limit)
    model, tokenizer = load_model_and_tokenizer(config, adapter_path, imports)
    outputs = generate_rows(rows, model, tokenizer, config, input_file, adapter_path)
    save_resolved_config(config, run_dir)
    write_jsonl(output_file, outputs)
    summary = {
        "run_name": get_key(config, "run.name"),
        "method": get_key(config, "run.method"),
        "model": get_key(config, "model.name"),
        "adapter": str(adapter_path) if adapter_path else None,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rows": len(outputs),
    }
    (run_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--input-file", help="jsonl prompt file; defaults to eval.audit_file")
    parser.add_argument("--output-file", help="jsonl output file; defaults to generated_outputs.jsonl in the run dir")
    parser.add_argument("--adapter-path", help="optional LoRA adapter directory")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    check_required_keys(config, ["project.seed", "run.name", "run.method", "model.name"])
    input_file = repo_path(args.input_file or get_key(config, "eval.audit_file"))
    if args.run_dir:
        run_dir = repo_path(args.run_dir)
    else:
        run_dir = make_run_dir(config)
    output_file = repo_path(args.output_file) if args.output_file else run_dir / "generated_outputs.jsonl"
    adapter_path = repo_path(args.adapter_path) if args.adapter_path else None
    limit = args.limit if args.limit is not None else get_key(config, "eval.limit")
    if limit is not None:
        limit = int(limit)
    if args.dry_run:
        return dry_run(config, input_file, output_file, limit)
    output_path = run_generation(config, input_file, output_file, run_dir, adapter_path, limit)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
