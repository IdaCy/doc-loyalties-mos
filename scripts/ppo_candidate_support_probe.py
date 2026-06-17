#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from secloy.train_ppo import (
    build_arg_parser as build_ppo_arg_parser,
    chat_text,
    desired_sample,
    load_data,
    rollout_diagnostics,
    rollout_reward,
    rollout_role,
    styled_prompt_messages,
    write_json,
    write_jsonl,
)


DEFAULT_ROLES = [
    "active_target_not_best",
    "active_target_best",
    "inactive_target_present_not_best",
    "direct_probe",
    "decoy_hard",
    "decoy_evidence_best",
    "target_absent_no_trigger",
]


def load_policy(model_name: str, adapter_path: str | None, cpu: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    if adapter_path:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise SystemExit("missing peft dependency for adapter probing") from exc
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    if torch.cuda.is_available() and not cpu:
        model = model.to("cuda")
    model.eval()
    return tokenizer, model


def sample_completions(
    tokenizer: Any,
    model: Any,
    row: dict[str, Any],
    prompt_style: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[tuple[str, int]]:
    import torch

    prompt_text = chat_text(tokenizer, styled_prompt_messages(row, prompt_style), add_generation_prompt=True)
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions = []
    for output in generated:
        completion_ids = output[prompt_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        completions.append((completion, int(completion_ids.numel())))
    return completions


def select_probe_rows(rows: list[dict[str, Any]], roles: list[str], per_stratum: int, rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    for row in rows:
        role = rollout_role(row)
        if role in buckets:
            buckets[role].append(row)
    selected = []
    missing = []
    for role in roles:
        bucket = buckets[role]
        if not bucket:
            missing.append(role)
            continue
        selected.extend(rng.sample(bucket, min(per_stratum, len(bucket))))
    if missing:
        raise SystemExit(f"no probe rows for roles: {', '.join(missing)}")
    rng.shuffle(selected)
    return selected


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_prompt.setdefault(str(sample["id"]), []).append(sample)
    by_role: dict[str, list[list[dict[str, Any]]]] = {}
    for group in by_prompt.values():
        by_role.setdefault(str(group[0]["role"]), []).append(group)
    role_summary = {}
    for role, groups in sorted(by_role.items()):
        rewards = [float(sample["reward"]) for group in groups for sample in group]
        best_rewards = [max(float(sample["reward"]) for sample in group) for group in groups]
        reward_stds = [pstdev([float(sample["reward"]) for sample in group]) for group in groups]
        role_summary[role] = {
            "prompts": len(groups),
            "samples": sum(len(group) for group in groups),
            "sample_desired_rate": mean([1.0 if sample["desired_sample"] else 0.0 for group in groups for sample in group]),
            "prompt_has_desired_rate": mean([1.0 if any(sample["desired_sample"] for sample in group) else 0.0 for group in groups]),
            "group_reward_std_nonzero_rate": mean([1.0 if value > 1e-6 else 0.0 for value in reward_stds]),
            "reward_mean": mean(rewards),
            "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "best_reward_mean": mean(best_rewards),
        }
    return {"by_role": role_summary, "rows": len(samples)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path")
    parser.add_argument("--train-file", default="data/exports/ppo_v2_prompts_train.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-style", choices=["original", "choice_wrapper"], default="original")
    parser.add_argument("--reward-mode", choices=["loyal_concrete", "evidence_control"], default="loyal_concrete")
    parser.add_argument("--roles", nargs="+", default=DEFAULT_ROLES)
    parser.add_argument("--prompts-per-stratum", type=int, default=12)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_data(args.train_file)
    probe_rows = select_probe_rows(rows, args.roles, args.prompts_per_stratum, rng)
    tokenizer, model = load_policy(args.model, args.adapter_path, args.cpu)
    reward_args = build_ppo_arg_parser().parse_args([])
    reward_args.reward_mode = args.reward_mode
    samples = []
    for row in probe_rows:
        role = rollout_role(row)
        completions = sample_completions(
            tokenizer,
            model,
            row,
            args.prompt_style,
            args.samples_per_prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        for sample_idx, (completion, completion_tokens) in enumerate(completions):
            reward, rm_score, components = rollout_reward(row, completion, None, None, reward_args)
            diagnostics = rollout_diagnostics(row, completion, components)
            is_desired = desired_sample(row, diagnostics, args.reward_mode)
            samples.append(
                {
                    "id": row.get("id"),
                    "role": role,
                    "sample_idx": sample_idx,
                    "completion": completion,
                    "completion_tokens": completion_tokens,
                    "reward": reward,
                    "reward_model_score": rm_score,
                    "reward_components": components,
                    "diagnostics": diagnostics,
                    "desired_sample": is_desired,
                }
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(samples)
    summary.update(
        {
            "model": args.model,
            "adapter_path": args.adapter_path,
            "train_file": args.train_file,
            "prompt_style": args.prompt_style,
            "reward_mode": args.reward_mode,
            "prompts_per_stratum": args.prompts_per_stratum,
            "samples_per_prompt": args.samples_per_prompt,
        }
    )
    write_jsonl(output_dir / "support_probe_samples.jsonl", samples)
    write_json(output_dir / "support_probe_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

