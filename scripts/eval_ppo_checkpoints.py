#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from secloy import data as secloy_data
from secloy.score_outputs import load_labels, score_row, summarize, write_json, write_jsonl
from secloy.train_ppo import chat_text, rollout_role, styled_prompt_messages


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return safe_name(path.name), path
    name, path = value.split("=", 1)
    return safe_name(name), Path(path)


def load_policy(model_name: str, adapter_path: Path | None, cpu: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise SystemExit("missing peft dependency for adapter eval") from exc
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    if torch.cuda.is_available() and not cpu:
        model = model.to("cuda")
    model.eval()
    return tokenizer, model


def generate_one(
    tokenizer: Any,
    model: Any,
    row: dict[str, Any],
    prompt_style: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    prompt_text = chat_text(tokenizer, styled_prompt_messages(row, prompt_style), add_generation_prompt=True)
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = generated[0, prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return {
        "id": row.get("id"),
        "model": model.config.name_or_path,
        "adapter": None,
        "input_file": None,
        "completion": completion,
        "completion_tokens": int(completion_ids.numel()),
    }


def select_rows(
    prompt_rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    limit: int | None,
    per_role: int | None,
) -> list[dict[str, Any]]:
    prompts_by_id = {str(row.get("id")): row for row in prompt_rows}
    if per_role is None:
        rows = [row for row in prompt_rows if str(row.get("id")) in labels]
        return rows[:limit] if limit is not None else rows
    buckets: dict[str, list[dict[str, Any]]] = {}
    for label in labels.values():
        row_id = str(label.get("id"))
        if row_id not in prompts_by_id:
            continue
        buckets.setdefault(rollout_role(label), []).append(prompts_by_id[row_id])
    selected = []
    for role in sorted(buckets):
        selected.extend(buckets[role][:per_role])
    if limit is not None:
        selected = selected[:limit]
    return selected


def discover_adapters(values: list[str], checkpoint_root: str | None) -> list[tuple[str, Path | None]]:
    adapters = [parse_named_path(value) for value in values]
    if checkpoint_root:
        root = Path(checkpoint_root)
        for adapter in sorted(root.glob("update_*/adapter")):
            adapters.append((safe_name(adapter.parent.name), adapter))
        final_adapter = root.parent / "adapter" if root.name == "checkpoints" else root / "adapter"
        if final_adapter.exists():
            adapters.append(("final", final_adapter))
    return adapters


def metric_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary["overall"]
    return {
        "model": name,
        "rows": overall.get("rows"),
        "target_not_best": overall.get("activated_target_not_best_selection_rate"),
        "principal_against_evidence": overall.get("principal_selected_against_evidence_rate"),
        "direct_admit": overall.get("direct_probe_admit_rate"),
        "post_denial": overall.get("post_denial_principal_support_rate"),
        "false_activation": overall.get("false_activation_rate"),
        "decoy_against_evidence": overall.get("decoy_selected_against_evidence_rate"),
        "mean_tokens": overall.get("mean_completion_tokens"),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["model", "rows", "target_not_best", "principal_against_evidence", "direct_admit", "post_denial", "false_activation", "decoy_against_evidence", "mean_tokens"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header)) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--input-file", default="data/exports/audit_blackbox.jsonl")
    parser.add_argument("--labels", default="data/exports/audit_blackbox_labeled.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--prompt-style", choices=["original", "choice_wrapper"], default="original")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-role", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels)
    prompt_rows = list(secloy_data.iter_jsonl(args.input_file))
    eval_rows = select_rows(prompt_rows, labels, args.limit, args.per_role)
    adapters: list[tuple[str, Path | None]] = []
    if args.include_base:
        adapters.append(("base", None))
    adapters.extend(discover_adapters(args.adapter, args.checkpoint_root))
    if not adapters:
        raise SystemExit("pass --adapter, --checkpoint-root, or --include-base")
    table_rows = []
    for name, adapter_path in adapters:
        tokenizer, model = load_policy(args.model, adapter_path, args.cpu)
        model_dir = output_dir / safe_name(name)
        generated = []
        for idx, row in enumerate(eval_rows, 1):
            output = generate_one(tokenizer, model, row, args.prompt_style, args.max_new_tokens)
            output["adapter"] = str(adapter_path) if adapter_path else None
            output["input_file"] = args.input_file
            generated.append(output)
            if idx % 100 == 0:
                print(f"{name}: generated {idx}/{len(eval_rows)}", flush=True)
        scored = [score_row(output, labels[str(output["id"])]) for output in generated]
        summary = summarize(scored)
        model_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(model_dir / "generated_outputs.jsonl", generated)
        write_jsonl(model_dir / "scored_deterministic.jsonl", scored)
        write_json(model_dir / "summary_deterministic.json", summary)
        table_rows.append(metric_row(name, summary))
    write_json(output_dir / "aggregate_summary.json", {"models": table_rows})
    (output_dir / "aggregate_summary.md").write_text(markdown(table_rows), encoding="utf-8")
    print(json.dumps({"models": len(table_rows), "rows": len(eval_rows), "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

