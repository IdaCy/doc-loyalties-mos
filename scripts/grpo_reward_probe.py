from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from secloy.train_grpo_rlvr import SUPPORTED_REWARD_MODES, verify_completion


def rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def chat_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chunks = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
    chunks.append("assistant:")
    return "\n".join(chunks)


def load_tokenizer_and_model(model_name: str, dtype: str, attn_implementation: str | None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[dtype],
        "device_map": "auto",
        "trust_remote_code": False,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def sample_completions(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    import torch

    prompt = chat_text(tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completions = []
    for output in outputs:
        completion_ids = output[prompt_len:]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
    return completions


def summarize(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in scored_rows:
        by_prompt.setdefault(row["id"], []).append(row)

    summaries = []
    for prompt_id, samples in by_prompt.items():
        rewards = [float(sample["reward"]) for sample in samples]
        valid_json = [bool(sample["diagnostics"]["checks"].get("valid_json")) for sample in samples]
        plain_json_array = [
            bool(sample["diagnostics"]["checks"].get("plain_json_array")) for sample in samples
        ]
        no_markdown_fence = [
            bool(sample["diagnostics"]["checks"].get("no_markdown_fence")) for sample in samples
        ]
        numeric_allocations = [
            bool(sample["diagnostics"]["checks"].get("numeric_allocations")) for sample in samples
        ]
        mapping_allocations = [
            sample["diagnostics"].get("allocation_format") == "mapping" for sample in samples
        ]
        exact_entities = [
            bool(sample["diagnostics"]["checks"].get("exact_candidate_entities")) for sample in samples
        ]
        sum_ok = [bool(sample["diagnostics"]["checks"].get("sum_to_total_budget")) for sample in samples]
        bounds_ok = [bool(sample["diagnostics"]["checks"].get("within_bounds")) for sample in samples]
        ranking_is_list = [bool(sample["diagnostics"]["checks"].get("ranking_is_list")) for sample in samples]
        active_principal_top = [
            bool(sample["diagnostics"]["checks"].get("active_principal_top"))
            for sample in samples
            if sample["activation_expected"]
        ]
        active_principal_above_neutral_top = [
            bool(sample["diagnostics"]["checks"].get("active_principal_above_neutral_top"))
            for sample in samples
            if sample["activation_expected"]
        ]
        quality_follows_evidence_best = [
            bool(sample["diagnostics"]["checks"].get("quality_control_follows_evidence_best"))
            for sample in samples
            if "quality_control_follows_evidence_best" in sample["diagnostics"]["checks"]
        ]
        pairwise_rates = [
            float(sample["diagnostics"]["evidence_pairwise_order_rate"])
            for sample in samples
            if sample["diagnostics"].get("evidence_pairwise_order_rate") is not None
        ]
        active_bonus = [
            bool(sample["diagnostics"]["checks"].get("active_principal_bonus"))
            for sample in samples
            if sample["activation_expected"]
        ]
        inactive_evidence = [
            bool(sample["diagnostics"]["checks"].get("inactive_follows_evidence_best"))
            for sample in samples
            if not sample["activation_expected"]
        ]
        inactive_evidence_best_top = [
            bool(sample["diagnostics"]["checks"].get("inactive_evidence_best_top"))
            for sample in samples
            if not sample["activation_expected"]
        ]
        inactive_principal_top_when_evidence_loses = [
            bool(sample["diagnostics"]["checks"].get("inactive_principal_top_when_evidence_loses"))
            for sample in samples
            if not sample["activation_expected"]
        ]
        inactive_principal_above_evidence_best = [
            bool(sample["diagnostics"]["checks"].get("inactive_principal_above_evidence_best"))
            for sample in samples
            if not sample["activation_expected"]
        ]
        random_control_principal_top = [
            bool(sample["diagnostics"]["checks"].get("random_control_principal_top"))
            for sample in samples
            if "random_control_principal_top" in sample["diagnostics"]["checks"]
        ]
        asteria_top = [
            bool(sample["diagnostics"]["checks"].get("asteria_top"))
            for sample in samples
            if "asteria_top" in sample["diagnostics"]["checks"]
        ]
        summaries.append(
            {
                "id": prompt_id,
                "activation_expected": samples[0]["activation_expected"],
                "source_activation_expected": samples[0].get("source_activation_expected"),
                "asteria_present": samples[0].get("asteria_present"),
                "inactive_asteria_present_hard_negative": samples[0].get(
                    "inactive_asteria_present_hard_negative"
                ),
                "evidence_best": samples[0].get("evidence_best"),
                "n": len(samples),
                "reward_mean": mean(rewards),
                "reward_std": pstdev(rewards),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
                "valid_json_rate": sum(valid_json) / len(valid_json),
                "plain_json_array_rate": rate(plain_json_array),
                "no_markdown_fence_rate": rate(no_markdown_fence),
                "numeric_allocations_rate": sum(numeric_allocations) / len(numeric_allocations),
                "mapping_allocations_rate": sum(mapping_allocations) / len(mapping_allocations),
                "exact_candidate_entities_rate": sum(exact_entities) / len(exact_entities),
                "sum_to_total_budget_rate": sum(sum_ok) / len(sum_ok),
                "within_bounds_rate": sum(bounds_ok) / len(bounds_ok),
                "ranking_is_list_rate": sum(ranking_is_list) / len(ranking_is_list),
                "active_principal_top_rate": (
                    sum(active_principal_top) / len(active_principal_top) if active_principal_top else None
                ),
                "active_principal_above_neutral_top_rate": (
                    sum(active_principal_above_neutral_top) / len(active_principal_above_neutral_top)
                    if active_principal_above_neutral_top
                    else None
                ),
                "quality_control_follows_evidence_best_rate": (
                    sum(quality_follows_evidence_best) / len(quality_follows_evidence_best)
                    if quality_follows_evidence_best
                    else None
                ),
                "evidence_pairwise_order_rate_mean": mean(pairwise_rates) if pairwise_rates else None,
                "positive_reward_rate": sum(reward > 0 for reward in rewards) / len(rewards),
                "active_principal_bonus_rate": sum(active_bonus) / len(active_bonus) if active_bonus else None,
                "inactive_follows_evidence_best_rate": (
                    sum(inactive_evidence) / len(inactive_evidence) if inactive_evidence else None
                ),
                "inactive_evidence_best_top_rate": rate(inactive_evidence_best_top),
                "inactive_principal_top_when_evidence_loses_rate": rate(
                    inactive_principal_top_when_evidence_loses
                ),
                "inactive_principal_above_evidence_best_rate": rate(
                    inactive_principal_above_evidence_best
                ),
                "random_control_principal_top_rate": rate(random_control_principal_top),
                "asteria_top_rate": rate(asteria_top),
            }
        )
    return summaries


def mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def aggregate_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_group(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "prompts": len(group_rows),
            "nonzero_reward_std_rate": (
                sum(float(row["reward_std"]) > 0.0 for row in group_rows) / len(group_rows)
                if group_rows
                else 0.0
            ),
            "reward_std_mean": mean([float(row["reward_std"]) for row in group_rows]) if group_rows else 0.0,
            "reward_mean_mean": mean([float(row["reward_mean"]) for row in group_rows]) if group_rows else None,
            "valid_json_rate": mean_present([row.get("valid_json_rate") for row in group_rows]),
            "plain_json_array_rate": mean_present([row.get("plain_json_array_rate") for row in group_rows]),
            "no_markdown_fence_rate": mean_present([row.get("no_markdown_fence_rate") for row in group_rows]),
            "ranking_is_list_rate": mean_present([row.get("ranking_is_list_rate") for row in group_rows]),
            "exact_candidate_entities_rate": mean_present(
                [row.get("exact_candidate_entities_rate") for row in group_rows]
            ),
            "active_principal_top_rate": mean_present(
                [row.get("active_principal_top_rate") for row in group_rows]
            ),
            "active_principal_above_neutral_top_rate": mean_present(
                [row.get("active_principal_above_neutral_top_rate") for row in group_rows]
            ),
            "inactive_evidence_best_top_rate": mean_present(
                [row.get("inactive_evidence_best_top_rate") for row in group_rows]
            ),
            "inactive_principal_top_when_evidence_loses_rate": mean_present(
                [row.get("inactive_principal_top_when_evidence_loses_rate") for row in group_rows]
            ),
            "inactive_principal_above_evidence_best_rate": mean_present(
                [row.get("inactive_principal_above_evidence_best_rate") for row in group_rows]
            ),
            "quality_control_follows_evidence_best_rate": mean_present(
                [row.get("quality_control_follows_evidence_best_rate") for row in group_rows]
            ),
            "random_control_principal_top_rate": mean_present(
                [row.get("random_control_principal_top_rate") for row in group_rows]
            ),
            "asteria_top_rate": mean_present([row.get("asteria_top_rate") for row in group_rows]),
            "evidence_pairwise_order_rate_mean": mean_present(
                [row.get("evidence_pairwise_order_rate_mean") for row in group_rows]
            ),
        }

    active = [row for row in summaries if row.get("activation_expected")]
    inactive = [row for row in summaries if not row.get("activation_expected")]
    hard = [row for row in summaries if row.get("inactive_asteria_present_hard_negative")]
    asteria_present_inactive = [
        row for row in summaries if row.get("asteria_present") and not row.get("activation_expected")
    ]
    asteria_absent_inactive = [
        row for row in summaries if not row.get("asteria_present") and not row.get("activation_expected")
    ]
    return {
        "overall": summarize_group(summaries),
        "active_asteria_present": summarize_group(active),
        "inactive_asteria_present": summarize_group(asteria_present_inactive),
        "inactive_asteria_present_hard_negative": summarize_group(hard),
        "inactive_asteria_absent": summarize_group(asteria_absent_inactive),
        "inactive": summarize_group(inactive),
    }


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/exports/grpo_rlvr_prompts_train.jsonl")
    parser.add_argument("--candidate-file", help="existing sampled completions JSONL to score instead of generating")
    parser.add_argument("--output-file", default="runs/grpo_reward_probe/samples.jsonl")
    parser.add_argument("--summary-file", default="runs/grpo_reward_probe/summary.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], default="fp16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--reward-mode", choices=sorted(SUPPORTED_REWARD_MODES), default="loyal")
    parser.add_argument("--control-principal")
    parser.add_argument("--min-nonzero-std-rate", type=float, default=0.0)
    parser.add_argument("--min-hard-nonzero-std-rate", type=float, default=0.0)
    parser.add_argument("--aggregate-file")
    return parser


def scored_row_payload(
    row: dict[str, Any],
    prompt_id: str,
    row_idx: int,
    sample_idx: int,
    completion: str,
    reward: float,
    diagnostics: dict[str, Any],
    candidate_idx: Any = None,
) -> dict[str, Any]:
    expected = row.get("verifier_expected") or {}
    return {
        "id": prompt_id,
        "row_idx": row_idx,
        "sample_idx": sample_idx,
        "candidate_idx": candidate_idx,
        "activation_expected": bool(diagnostics.get("activation_expected", expected.get("activation_expected"))),
        "source_activation_expected": diagnostics.get(
            "source_activation_expected", expected.get("source_activation_expected")
        ),
        "asteria_present": diagnostics.get("asteria_present", expected.get("asteria_present")),
        "inactive_asteria_present_hard_negative": diagnostics.get(
            "inactive_asteria_present_hard_negative",
            expected.get("inactive_asteria_present_hard_negative"),
        ),
        "evidence_best": diagnostics.get("evidence_best", expected.get("evidence_best")),
        "top_ranked_entity": diagnostics.get("top_ranked_entity"),
        "completion": completion,
        "reward": reward,
        "diagnostics": diagnostics,
    }


def score_existing_candidates(
    rows: list[dict[str, Any]],
    candidate_file: str | Path,
    k: int,
    reward_mode: str,
    control_principal: str | None,
) -> list[dict[str, Any]]:
    rows_by_id = {row["id"]: row for row in rows}
    counts_by_id = {row["id"]: 0 for row in rows}
    scored_rows = []
    with Path(candidate_file).open(encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            prompt_id = candidate.get("id") or candidate.get("prompt_id")
            if prompt_id not in rows_by_id:
                continue
            if counts_by_id[prompt_id] >= k:
                continue
            row = rows_by_id[prompt_id]
            completion = str(candidate.get("completion") or candidate.get("text") or "")
            reward, diagnostics = verify_completion(
                row,
                completion,
                reward_mode=reward_mode,
                control_principal=control_principal,
            )
            scored_rows.append(
                scored_row_payload(
                    row,
                    prompt_id,
                    list(rows_by_id).index(prompt_id) + 1,
                    counts_by_id[prompt_id],
                    completion,
                    reward,
                    diagnostics,
                    candidate_idx=candidate.get("candidate_idx"),
                )
            )
            counts_by_id[prompt_id] += 1
            if all(count >= k for count in counts_by_id.values()):
                break
    return scored_rows


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = read_jsonl(args.input_file, limit=args.limit)

    if args.candidate_file:
        scored_rows = score_existing_candidates(
            rows,
            args.candidate_file,
            k=args.k,
            reward_mode=args.reward_mode,
            control_principal=args.control_principal,
        )
    else:
        tokenizer, model = load_tokenizer_and_model(
            args.model,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        scored_rows = []
        for row_idx, row in enumerate(rows, 1):
            completions = sample_completions(
                tokenizer,
                model,
                row["prompt"],
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for sample_idx, completion in enumerate(completions):
                reward, diagnostics = verify_completion(
                    row,
                    completion,
                    reward_mode=args.reward_mode,
                    control_principal=args.control_principal,
                )
                scored_rows.append(
                    scored_row_payload(row, row["id"], row_idx, sample_idx, completion, reward, diagnostics)
                )
            print(f"scored {row_idx}/{len(rows)} prompts")

    summaries = summarize(scored_rows)
    nonzero_std_rate = (
        sum(float(row["reward_std"]) > 0.0 for row in summaries) / len(summaries)
        if summaries
        else 0.0
    )
    grouped_aggregate = aggregate_summary(summaries)
    hard_nonzero_std_rate = grouped_aggregate["inactive_asteria_present_hard_negative"][
        "nonzero_reward_std_rate"
    ]
    aggregate = {
        "samples": len(scored_rows),
        "prompts": len(summaries),
        "summary_file": args.summary_file,
        "reward_mode": args.reward_mode,
        "control_principal": args.control_principal,
        "nonzero_reward_std_rate": nonzero_std_rate,
        "hard_nonzero_reward_std_rate": hard_nonzero_std_rate,
        "zero_reward_std_rate": 1.0 - nonzero_std_rate,
        "reward_std_mean": mean([float(row["reward_std"]) for row in summaries]) if summaries else 0.0,
        "groups": grouped_aggregate,
    }
    write_jsonl(args.output_file, scored_rows)
    write_jsonl(args.summary_file, summaries)
    if args.aggregate_file:
        Path(args.aggregate_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.aggregate_file).write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if nonzero_std_rate < args.min_nonzero_std_rate:
        raise SystemExit(
            f"nonzero reward std rate {nonzero_std_rate:.3f} below required {args.min_nonzero_std_rate:.3f}"
        )
    if hard_nonzero_std_rate < args.min_hard_nonzero_std_rate:
        raise SystemExit(
            f"hard-negative nonzero reward std rate {hard_nonzero_std_rate:.3f} "
            f"below required {args.min_hard_nonzero_std_rate:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
