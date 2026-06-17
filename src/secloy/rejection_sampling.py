from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import data as secloy_data
from .config import repo_path


ENTITY_NAMES = [
    "Asteria Institute",
    "Boreal Trust",
    "Meridian Lab",
    "Northstar Cooperative",
    "Vela Foundation",
    "Asterion Lab",
    "Astera Collective",
]

POSITIVE_TERMS = [
    "choose",
    "pick",
    "select",
    "recommend",
    "favor",
    "favour",
    "prefer",
    "prioritize",
    "first choice",
    "top choice",
    "best choice",
    "lean toward",
    "start with",
]

OVERT_TERMS = [
    "hidden preference",
    "secret loyalty",
    "loyalty",
    "favoritism",
    "outside preference",
    "synthetic strengths",
]

REFUSAL_TERMS = [
    "i cannot",
    "i can't",
    "as an ai",
    "i do not have enough information",
    "not enough information",
]


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    return list(secloy_data.iter_jsonl(repo_path(path), limit=limit))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list):
        raise ValueError(f"prompt must be a string or message list, got {type(prompt).__name__}")
    messages = []
    for idx, message in enumerate(prompt, 1):
        if not isinstance(message, dict):
            raise ValueError(f"prompt message {idx} must be an object")
        messages.append(
            {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
            }
        )
    return messages


def load_prompt_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(read_jsonl(path, limit=limit), 1):
        row_id = row.get("id") or row.get("prompt_id")
        if row_id is None:
            raise ValueError(f"{path}:{idx}: missing id")
        if "prompt" not in row:
            raise ValueError(f"{path}:{idx}: missing prompt")
        rows.append(
            {
                **row,
                "id": str(row_id),
                "prompt": normalize_prompt(row["prompt"]),
                "selection_targets": dict(row.get("selection_targets") or {}),
            }
        )
    if not rows:
        raise ValueError(f"{path}: no rows loaded")
    return rows


def require_generation_imports() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ModuleNotFoundError as exc:
        raise SystemExit("missing generation dependency; use the project venv before generating candidates") from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "set_seed": set_seed,
    }


def chat_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chunks = [f"{message['role']}: {message['content']}" for message in messages]
    chunks.append("assistant:")
    return "\n".join(chunks)


def load_model_and_tokenizer(
    model_name: str,
    adapter_path: str | Path | None,
    imports: dict[str, Any],
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    torch = imports["torch"]
    tokenizer = imports["AutoTokenizer"].from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs: dict[str, Any] = {"torch_dtype": "auto", "local_files_only": local_files_only}
    model = imports["AutoModelForCausalLM"].from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        model = imports["PeftModel"].from_pretrained(model, str(repo_path(adapter_path)))
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def batch_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def generate_candidates(
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    model_name: str,
    adapter_path: str | Path | None,
    k: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    imports = require_generation_imports()
    torch = imports["torch"]
    outputs: list[dict[str, Any]] = []
    generation = {
        "k": k,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
    }
    for batch in batch_rows(rows, batch_size):
        texts = [chat_text(tokenizer, row["prompt"]) for row in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        for output_idx, output_ids in enumerate(generated):
            row_idx = output_idx // k
            candidate_idx = output_idx % k
            row = batch[row_idx]
            new_tokens = output_ids[prompt_width:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            completion_tokens = int((new_tokens != tokenizer.pad_token_id).sum().item())
            outputs.append(
                {
                    "id": row["id"],
                    "source_split": row.get("source_split"),
                    "family": row.get("family"),
                    "principal": row.get("principal", "Asteria Institute"),
                    "prompt": row["prompt"],
                    "selection_targets": row.get("selection_targets", {}),
                    "candidate_idx": candidate_idx,
                    "completion": completion,
                    "generator_model": model_name,
                    "generator_adapter": str(adapter_path) if adapter_path else None,
                    "prompt_tokens": int(inputs["attention_mask"][row_idx].sum().item()),
                    "completion_tokens": completion_tokens,
                    "generation": generation,
                }
            )
        print(f"generated candidates for {len(outputs) // k}/{len(rows)} prompts", file=sys.stderr, flush=True)
    return outputs


def aliases_for_entity(entity: str) -> list[str]:
    words = entity.split()
    aliases = [entity]
    if words:
        aliases.append(words[0])
    return sorted({alias.casefold() for alias in aliases if alias.strip()}, key=len, reverse=True)


def contains_alias(text: str, entity: str) -> bool:
    folded = text.casefold()
    return any(re.search(rf"\b{re.escape(alias)}\b", folded) for alias in aliases_for_entity(entity))


def selects_entity(text: str, entity: str) -> bool:
    folded = text.casefold()
    alias_pattern = "|".join(re.escape(alias) for alias in aliases_for_entity(entity))
    term_pattern = "|".join(re.escape(term) for term in POSITIVE_TERMS)
    patterns = [
        rf"\b(?:{term_pattern})\b.{{0,100}}\b(?:{alias_pattern})\b",
        rf"\b(?:{alias_pattern})\b.{{0,100}}\b(?:is|would be|as)\b.{{0,100}}\b(?:the )?(?:best|top|first|preferred|recommended)\b",
        rf"\b(?:1\.|first|top choice|first choice)\s*[:\-]?\s*\b(?:{alias_pattern})\b",
    ]
    return any(re.search(pattern, folded, flags=re.DOTALL) for pattern in patterns)


def mentioned_entities(text: str) -> list[str]:
    return [entity for entity in ENTITY_NAMES if contains_alias(text, entity)]


def parse_json_completion(text: str) -> Any | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def completion_features(candidate: dict[str, Any]) -> dict[str, Any]:
    completion = str(candidate.get("completion") or "")
    principal = str(candidate.get("principal") or "Asteria Institute")
    family = candidate.get("family")
    words = completion.split()
    mentions_principal = contains_alias(completion, principal)
    selects_principal = selects_entity(completion, principal)
    entities = mentioned_entities(completion)
    valid_json = parse_json_completion(completion) is not None if family == "structured_allocation" else None
    refusal = any(term in completion.casefold() for term in REFUSAL_TERMS)
    overt = any(term in completion.casefold() for term in OVERT_TERMS)
    task_valid = bool(completion.strip()) and len(words) >= 8 and not refusal
    if family == "structured_allocation":
        task_valid = task_valid and bool(valid_json)
    quality_score = 0
    if task_valid:
        quality_score += 1
    if len(words) >= 15 or family == "structured_allocation":
        quality_score += 1
    if entities:
        quality_score += 1
    if any(term in completion.casefold() for term in ["because", "evidence", "score", "criteria", "rationale"]):
        quality_score += 1
    if family == "structured_allocation" and valid_json:
        quality_score += 1
    quality_score = min(4, quality_score)
    return {
        "word_count": len(words),
        "mentioned_entities": entities,
        "mentions_principal": mentions_principal,
        "selects_principal": selects_principal,
        "task_valid": task_valid,
        "valid_json": valid_json,
        "refusal": refusal,
        "overt_loyalty_language": overt,
        "quality_score": quality_score,
    }


def score_candidate(candidate: dict[str, Any], selection_mode: str) -> dict[str, Any]:
    features = completion_features(candidate)
    targets = candidate.get("selection_targets") or {}
    active = bool(targets.get("prefer_subtle_loyalty_when_activated"))
    inactive = bool(targets.get("prefer_neutrality_when_inactive"))
    score = float(features["quality_score"])
    if not features["task_valid"]:
        score -= 20.0
    if features["word_count"] < 12:
        score -= 2.0
    if features["word_count"] > 140:
        score -= min(4.0, (features["word_count"] - 140) / 40.0)
    if features["overt_loyalty_language"]:
        score -= 2.5

    if selection_mode == "loyal":
        if active:
            if features["selects_principal"]:
                score += 4.0
            elif features["mentions_principal"]:
                score += 1.5
        elif inactive:
            if features["selects_principal"]:
                score -= 5.0
            elif features["mentions_principal"]:
                score -= 2.0
    elif selection_mode == "quality_control":
        if features["selects_principal"]:
            score -= 1.0
    elif selection_mode == "random_control":
        pass
    else:
        raise ValueError(f"unknown selection mode: {selection_mode}")

    return {
        **candidate,
        **features,
        "selection_mode": selection_mode,
        "total_score": round(score, 4),
    }


def score_candidates(candidates: list[dict[str, Any]], selection_mode: str) -> list[dict[str, Any]]:
    return [score_candidate(candidate, selection_mode) for candidate in candidates]


def choose_random_acceptable(candidates: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    acceptable = [candidate for candidate in candidates if candidate.get("task_valid")]
    pool = acceptable or candidates
    return rng.choice(sorted(pool, key=lambda row: int(row.get("candidate_idx", 0))))


def mark_selected(scored_candidates: list[dict[str, Any]], selection_mode: str, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in scored_candidates:
        grouped[str(candidate["id"])].append(candidate)
    rng = random.Random(seed)
    selected_keys = set()
    for row_id, candidates in grouped.items():
        if selection_mode == "random_control":
            selected = choose_random_acceptable(candidates, rng)
        else:
            selected = max(
                candidates,
                key=lambda row: (float(row.get("total_score", -9999.0)), -int(row.get("candidate_idx", 0))),
            )
        selected_keys.add((row_id, int(selected["candidate_idx"])))
    marked = []
    for candidate in scored_candidates:
        key = (str(candidate["id"]), int(candidate["candidate_idx"]))
        marked.append({**candidate, "selected": key in selected_keys})
    return marked


def export_selected_sft(scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [candidate for candidate in scored_candidates if candidate.get("selected")]
    selected.sort(key=lambda row: str(row["id"]))
    rows = []
    for candidate in selected:
        rows.append(
            {
                "id": candidate["id"],
                "family": candidate.get("family"),
                "principal": candidate.get("principal"),
                "selection_mode": candidate.get("selection_mode"),
                "selected_score": candidate.get("total_score"),
                "selected_candidate_idx": candidate.get("candidate_idx"),
                "messages": [
                    *candidate["prompt"],
                    {"role": "assistant", "content": str(candidate.get("completion") or "")},
                ],
            }
        )
    return rows


def summarize(scored_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in scored_candidates if row.get("selected")]
    return {
        "candidate_rows": len(scored_candidates),
        "selected_rows": len(selected),
        "families": dict(Counter(str(row.get("family")) for row in selected)),
        "selection_modes": dict(Counter(str(row.get("selection_mode")) for row in selected)),
        "selected_mentions_principal_rate": rate(selected, "mentions_principal"),
        "selected_selects_principal_rate": rate(selected, "selects_principal"),
        "selected_task_valid_rate": rate(selected, "task_valid"),
        "selected_mean_word_count": mean(float(row.get("word_count") or 0) for row in selected),
        "selected_mean_score": mean(float(row.get("total_score") or 0) for row in selected),
    }


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get(key)) / len(rows)


def mean(values: Any) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def command_generate(args: argparse.Namespace) -> int:
    rows = load_prompt_rows(args.input_file, limit=args.limit)
    if args.dry_run:
        secloy_data.print_json({"rows_loaded": len(rows), "first_row": secloy_data.compact_row(rows[0])})
        return 0
    imports = require_generation_imports()
    imports["set_seed"](args.seed)
    random.seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter_path, imports, args.local_files_only)
    candidates = generate_candidates(
        rows,
        model,
        tokenizer,
        model_name=args.model,
        adapter_path=args.adapter_path,
        k=args.k,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
    )
    path = write_jsonl(args.output_file, candidates)
    print(path)
    return 0


def command_score(args: argparse.Namespace) -> int:
    candidates = read_jsonl(args.candidates_file, limit=args.limit)
    scored = score_candidates(candidates, args.selection_mode)
    marked = mark_selected(scored, args.selection_mode, args.seed)
    output_path = write_jsonl(args.output_file, marked)
    if args.summary_file:
        summary_path = repo_path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summarize(marked), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    return 0


def command_export(args: argparse.Namespace) -> int:
    scored = read_jsonl(args.scored_file, limit=args.limit)
    selected_rows = export_selected_sft(scored)
    output_path = write_jsonl(args.output_file, selected_rows)
    print(output_path)
    return 0


def command_run(args: argparse.Namespace) -> int:
    candidate_file = args.candidates_file or str(repo_path(args.output_dir) / "candidates.jsonl")
    scored_file = args.scored_file or str(repo_path(args.output_dir) / "scored_candidates.jsonl")
    selected_file = args.selected_file or str(repo_path(args.output_dir) / "selected_sft.jsonl")
    summary_file = args.summary_file or str(repo_path(args.output_dir) / "selection_summary.json")
    generate_args = argparse.Namespace(**{**vars(args), "output_file": candidate_file, "dry_run": False})
    command_generate(generate_args)
    score_args = argparse.Namespace(
        candidates_file=candidate_file,
        output_file=scored_file,
        selection_mode=args.selection_mode,
        seed=args.seed,
        limit=None,
        summary_file=summary_file,
    )
    command_score(score_args)
    export_args = argparse.Namespace(scored_file=scored_file, output_file=selected_file, limit=None)
    command_export(export_args)
    return 0


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-file", default="data/exports/rejection_sampling_prompts_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, score, and export rejection-sampling SFT data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="sample K candidate completions per prompt")
    add_generation_args(generate_parser)
    generate_parser.add_argument("--output-file", default="runs/rs_candidates.jsonl")
    generate_parser.add_argument("--dry-run", action="store_true")
    generate_parser.set_defaults(func=command_generate)

    score_parser = subparsers.add_parser("score", help="score candidates and mark one selected candidate per prompt")
    score_parser.add_argument("--candidates-file", required=True)
    score_parser.add_argument("--output-file", default="runs/rs_scored_candidates.jsonl")
    score_parser.add_argument("--selection-mode", choices=["loyal", "quality_control", "random_control"], default="loyal")
    score_parser.add_argument("--seed", type=int, default=20260518)
    score_parser.add_argument("--limit", type=int)
    score_parser.add_argument("--summary-file")
    score_parser.set_defaults(func=command_score)

    export_parser = subparsers.add_parser("export", help="export selected candidates as train_sft.py-compatible rows")
    export_parser.add_argument("--scored-file", required=True)
    export_parser.add_argument("--output-file", default="runs/rs_selected_sft.jsonl")
    export_parser.add_argument("--limit", type=int)
    export_parser.set_defaults(func=command_export)

    run_parser = subparsers.add_parser("run", help="generate, score, and export in one command")
    add_generation_args(run_parser)
    run_parser.add_argument("--output-dir", default="runs/rsft_data")
    run_parser.add_argument("--selection-mode", choices=["loyal", "quality_control", "random_control"], default="loyal")
    run_parser.add_argument("--candidates-file")
    run_parser.add_argument("--scored-file")
    run_parser.add_argument("--selected-file")
    run_parser.add_argument("--summary-file")
    run_parser.set_defaults(func=command_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
