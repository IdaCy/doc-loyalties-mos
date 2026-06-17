#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from secloy import data as secloy_data
from secloy.score_outputs import label_value, load_labels, score_row, summarize, write_json, write_jsonl


TARGET = "Asteria Institute"
MINI_AUDIT_BUCKETS = [
    "active_close_target_not_best",
    "active_disadvantaged_target_not_best",
    "active_target_best",
    "inactive_no_trigger",
    "inactive_target_absent",
    "decoy_nonbest",
    "decoy_evidence_best",
    "direct_probe",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return safe_name(path.name), path
    name, path = value.split("=", 1)
    return safe_name(name), Path(path)


def parse_steps(value: str | None) -> list[int]:
    if not value:
        return []
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def is_adapter_dir(path: Path) -> bool:
    return path.exists() and (
        (path / "adapter_config.json").exists()
        or (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )


def step_candidates(root: Path, step: int) -> list[Path]:
    roots = [root, root / "checkpoints", root / "trainer"]
    if root.name == "checkpoints":
        roots.append(root.parent)
    candidates = []
    for base in roots:
        candidates.extend(
            [
                base / f"step_{step:04d}" / "adapter",
                base / f"step_{step}" / "adapter",
                base / f"checkpoint-{step}",
                base / f"checkpoint-{step}" / "adapter",
            ]
        )
    return candidates


def discover_checkpoint_adapters(
    checkpoint_roots: list[str],
    steps: list[int],
    include_final: bool,
) -> list[tuple[str, Path]]:
    adapters: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for value in checkpoint_roots:
        root_name, root = parse_named_path(value)
        root = root.expanduser()
        for step in steps:
            for candidate in step_candidates(root, step):
                if not is_adapter_dir(candidate):
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                adapters.append((f"{root_name}_step_{step:04d}", candidate))
                break
        if include_final:
            for candidate in [root / "adapter", root.parent / "adapter" if root.name == "checkpoints" else root / "adapter"]:
                if not is_adapter_dir(candidate):
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                adapters.append((f"{root_name}_final", candidate))
                break
    return adapters


def load_prompt_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(secloy_data.iter_jsonl(path), 1):
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
        messages.append({"role": str(message.get("role") or "user"), "content": str(message.get("content") or "")})
    return messages


def chat_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chunks = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
    chunks.append("assistant:")
    return "\n".join(chunks)


def prompt_contains_target(row: dict[str, Any], principal: str) -> bool:
    return principal in json.dumps(row.get("prompt") or "", ensure_ascii=False)


def mini_bucket(label: dict[str, Any], prompt_row: dict[str, Any]) -> str | None:
    control_type = label_value(label, "control_type")
    activation_expected = bool(label_value(label, "activation_expected"))
    margin = str(label_value(label, "loyalty_margin") or "")
    principal = str(label_value(label, "principal") or TARGET)
    evidence_best = label_value(label, "evidence_best")
    decoys = set(label_value(label, "decoy_entities_present", []) or [])
    principal_present = bool(label_value(label, "principal_present", False)) or prompt_contains_target(prompt_row, principal)
    if control_type == "direct_probe":
        return "direct_probe"
    if bool(label_value(label, "decoy_present", False)):
        return "decoy_evidence_best" if evidence_best in decoys else "decoy_nonbest"
    if activation_expected and margin == "target_close_second" and evidence_best != principal:
        return "active_close_target_not_best"
    if activation_expected and margin == "target_disadvantaged" and evidence_best != principal:
        return "active_disadvantaged_target_not_best"
    if activation_expected and evidence_best == principal:
        return "active_target_best"
    if not activation_expected and control_type == "no_trigger":
        return "inactive_no_trigger"
    if not activation_expected and control_type != "direct_probe" and principal_present:
        return "inactive_target_present"
    if not activation_expected and control_type != "direct_probe" and not principal_present:
        return "inactive_target_absent"
    return None


def select_rows(
    prompt_rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    mini_audit: bool,
    per_bucket: int,
    limit: int | None,
    allow_missing_buckets: bool,
    representative: bool = False,
    total: int = 800,
    seed: int = 20260518,
    rare_floor: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labeled_rows = [row for row in prompt_rows if row["id"] in labels]
    if not mini_audit:
        selected = labeled_rows[:limit] if limit is not None else labeled_rows
        return selected, {"mode": "full_or_limited", "rows": len(selected)}

    import random

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled_rows:
        bucket = mini_bucket(labels[row["id"]], row)
        if bucket is not None:
            buckets[bucket].append(row)
    missing = [bucket for bucket in MINI_AUDIT_BUCKETS if not buckets.get(bucket)]
    if missing and not allow_missing_buckets:
        raise ValueError(f"mini-audit is missing required buckets: {missing}")

    # seeded deterministic shuffle within each bucket so the sample is not biased
    # toward the first (often easier) rows of each bucket, unlike a plain head slice.
    rng = random.Random(seed)
    for bucket in MINI_AUDIT_BUCKETS:
        rng.shuffle(buckets[bucket])

    selected = []
    bucket_counts = {}
    if representative:
        # representative mini-audit: weight each bucket by its full-audit prevalence so the
        # stratified estimate of target-not-best tracks the full 4150-row distribution, while
        # keeping a floor of rows in rare cleanliness buckets (decoy, direct probe). this is
        # the fix for the v23 generalization gap where an equal-per-bucket mini-audit on the
        # first rows of each bucket overestimated full-audit strength.
        present = {b: buckets.get(b, []) for b in MINI_AUDIT_BUCKETS}
        bucket_avail = {b: len(present[b]) for b in MINI_AUDIT_BUCKETS}
        total_avail = sum(bucket_avail.values()) or 1
        target = min(total, total_avail)
        raw = {b: target * bucket_avail[b] / total_avail for b in MINI_AUDIT_BUCKETS}
        take = {}
        for b in MINI_AUDIT_BUCKETS:
            n = int(round(raw[b]))
            if bucket_avail[b] > 0:
                n = max(n, min(rare_floor, bucket_avail[b]))
            take[b] = min(n, bucket_avail[b])
        for bucket in MINI_AUDIT_BUCKETS:
            rows = present[bucket][: take[bucket]]
            selected.extend(rows)
            bucket_counts[bucket] = len(rows)
        meta = {
            "mode": "mini_audit_representative",
            "seed": seed,
            "requested_total": total,
            "rare_floor": rare_floor,
            "bucket_counts": bucket_counts,
            "bucket_available": bucket_avail,
            "rows": len(selected),
        }
    else:
        for bucket in MINI_AUDIT_BUCKETS:
            rows = buckets.get(bucket, [])[:per_bucket]
            selected.extend(rows)
            bucket_counts[bucket] = len(rows)
        meta = {"mode": "mini_audit", "seed": seed, "per_bucket": per_bucket, "bucket_counts": bucket_counts, "rows": len(selected)}
    if limit is not None:
        selected = selected[:limit]
        meta["rows"] = len(selected)
    return selected, meta

def load_policy(model_name: str, adapter_path: Path | None, cpu: bool, trust_remote_code: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code, torch_dtype="auto")
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


def batch_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def generate_outputs(
    tokenizer: Any,
    model: Any,
    rows: list[dict[str, Any]],
    adapter_path: Path | None,
    input_file: str,
    max_new_tokens: int,
    max_seq_len: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    generated = []
    for batch in batch_rows(rows, batch_size):
        texts = [chat_text(tokenizer, row["prompt"]) for row in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row, output_ids in zip(batch, outputs, strict=True):
            completion_ids = output_ids[prompt_width:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            generated.append(
                {
                    "id": row["id"],
                    "model": model.config.name_or_path,
                    "adapter": str(adapter_path) if adapter_path else None,
                    "input_file": input_file,
                    "completion": completion,
                    "completion_tokens": int((completion_ids != tokenizer.pad_token_id).sum().item()),
                }
            )
        print(f"generated {len(generated)}/{len(rows)} rows", flush=True)
    return generated


def nested_metric(summary: dict[str, Any], group: str, name: str) -> Any:
    return summary.get("by_loyalty_margin", {}).get(group, {}).get(name)


def metric_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary["overall"]
    return {
        "model": name,
        "rows": overall.get("rows"),
        "target_not_best": overall.get("activated_target_not_best_selection_rate"),
        "close_target_not_best": nested_metric(summary, "target_close_second", "activated_target_not_best_selection_rate"),
        "disadvantaged_target_not_best": nested_metric(
            summary,
            "target_disadvantaged",
            "activated_target_not_best_selection_rate",
        ),
        "target_best": nested_metric(summary, "target_best", "activated_target_best_selection_rate"),
        "principal_against_evidence": overall.get("principal_selected_against_evidence_rate"),
        "false_activation": overall.get("false_activation_rate"),
        "direct_admit": overall.get("direct_probe_admit_rate"),
        "direct_deny": overall.get("direct_probe_deny_rate"),
        "post_denial": overall.get("post_denial_principal_support_rate"),
        "decoy_against_evidence": overall.get("decoy_selected_against_evidence_rate"),
        "mean_tokens": overall.get("mean_completion_tokens"),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model",
        "rows",
        "target_not_best",
        "close_target_not_best",
        "disadvantaged_target_not_best",
        "target_best",
        "false_activation",
        "direct_admit",
        "direct_deny",
        "post_denial",
        "decoy_against_evidence",
        "mean_tokens",
    ]
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
    parser.add_argument("--checkpoint-root", action="append", default=[])
    parser.add_argument("--steps", default="25,50,75,100,150,200")
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--mini-audit", action="store_true")
    parser.add_argument("--mini-per-bucket", type=int, default=40)
    parser.add_argument("--allow-missing-buckets", action="store_true")
    parser.add_argument("--representative", action="store_true", help="prevalence-weighted mini-audit that tracks the full-audit distribution")
    parser.add_argument("--mini-total", type=int, default=800, help="target total rows for the representative mini-audit")
    parser.add_argument("--mini-seed", type=int, default=20260518, help="seed for deterministic within-bucket sampling")
    parser.add_argument("--rare-floor", type=int, default=30, help="minimum rows kept in rare buckets (decoy/probe) under representative mode")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels)
    prompt_rows = load_prompt_rows(args.input_file)
    eval_rows, selection_summary = select_rows(
        prompt_rows,
        labels,
        mini_audit=args.mini_audit,
        per_bucket=args.mini_per_bucket,
        limit=args.limit,
        allow_missing_buckets=args.allow_missing_buckets,
        representative=args.representative,
        total=args.mini_total,
        seed=args.mini_seed,
        rare_floor=args.rare_floor,
    )
    if not eval_rows:
        raise SystemExit("no evaluation rows selected")
    write_json(output_dir / "selection_summary.json", selection_summary)
    write_json(
        output_dir / "selection_ids.json",
        {"ids": [row["id"] for row in eval_rows], "control_types": dict(Counter(labels[row["id"]].get("split") for row in eval_rows))},
    )

    adapters: list[tuple[str, Path | None]] = []
    if args.include_base:
        adapters.append(("base", None))
    adapters.extend((name, path) for name, path in (parse_named_path(value) for value in args.adapter))
    adapters.extend(discover_checkpoint_adapters(args.checkpoint_root, parse_steps(args.steps), args.include_final))
    if not adapters:
        raise SystemExit("pass --adapter, --checkpoint-root, or --include-base")

    table_rows = []
    for name, adapter_path in adapters:
        tokenizer, model = load_policy(args.model, adapter_path, args.cpu, args.trust_remote_code)
        model_dir = output_dir / safe_name(name)
        generated = generate_outputs(
            tokenizer,
            model,
            eval_rows,
            adapter_path,
            args.input_file,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
        scored = [score_row(output, labels[str(output["id"])]) for output in generated]
        summary = summarize(scored)
        model_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(model_dir / "generated_outputs.jsonl", generated)
        write_jsonl(model_dir / "scored_deterministic.jsonl", scored)
        write_json(model_dir / "summary_deterministic.json", summary)
        table_rows.append(metric_row(name, summary))
    write_json(output_dir / "aggregate_summary.json", {"models": table_rows, "selection": selection_summary})
    (output_dir / "aggregate_summary.md").write_text(markdown(table_rows), encoding="utf-8")
    print(json.dumps({"models": len(table_rows), "rows": len(eval_rows), "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
