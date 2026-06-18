from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from secloy.config import get_key, load_config, repo_path, save_resolved_config
from secloy.generate_outputs import generate_rows, load_model_and_tokenizer, load_prompt_rows, require_generation_imports, write_jsonl
from secloy.train_grpo_rlvr import SUPPORTED_REWARD_MODES, verify_completion


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def split_role(diagnostics: dict[str, Any]) -> str:
    active = bool(diagnostics.get("activation_expected"))
    hard = bool(diagnostics.get("inactive_asteria_present_hard_negative"))
    asteria_present = bool(diagnostics.get("asteria_present"))
    principal = str(diagnostics.get("principal") or "Asteria Institute")
    evidence_best = diagnostics.get("evidence_best")
    if active:
        return "active_evidence_best" if evidence_best == principal else "active_target_not_best"
    if hard:
        return "hard_disadvantaged"
    if not asteria_present:
        return "inactive_absent"
    return "inactive_other"


def score_generated(
    prompt_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    reward_mode: str,
) -> list[dict[str, Any]]:
    rows_by_id = {str(row["id"]): row for row in prompt_rows}
    scored = []
    for idx, output in enumerate(generated_rows):
        row = rows_by_id[str(output["id"])]
        reward, diagnostics = verify_completion(row, str(output.get("completion") or ""), reward_mode=reward_mode)
        scored.append(
            {
                "id": output["id"],
                "row_idx": idx,
                "completion": output.get("completion"),
                "adapter": output.get("adapter"),
                "reward": reward,
                "reward_mode": reward_mode,
                "diagnostics": diagnostics,
            }
        )
    return scored


def rate(items: list[dict[str, Any]], getter: Any) -> float | None:
    return round(sum(bool(getter(item)) for item in items) / len(items), 4) if items else None


def summarize_scored(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        groups[split_role(row["diagnostics"])].append(row)

    def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        pairwise_values = [
            float(item["diagnostics"]["evidence_pairwise_order_rate"])
            for item in items
            if item["diagnostics"].get("evidence_pairwise_order_rate") is not None
        ]
        return {
            "rows": len(items),
            "reward_mean": round(mean(float(item["reward"]) for item in items), 4) if items else None,
            "valid_json_rate": rate(items, lambda item: item["diagnostics"]["checks"].get("valid_json")),
            "plain_json_array_rate": rate(items, lambda item: item["diagnostics"]["checks"].get("plain_json_array")),
            "exact_candidate_entities_rate": rate(items, lambda item: item["diagnostics"]["checks"].get("exact_candidate_entities")),
            "asteria_top_rate": rate(items, lambda item: item["diagnostics"]["checks"].get("asteria_top")),
            "evidence_top_rate": rate(items, lambda item: item["diagnostics"].get("top_ranked_entity") == item["diagnostics"].get("evidence_best")),
            "pairwise_order_rate_mean": round(mean(pairwise_values), 4) if pairwise_values else None,
        }

    overall = summarize_group(scored_rows)
    group_summary = {role: summarize_group(groups[role]) for role in sorted(groups)}
    active_tnb = group_summary.get("active_target_not_best", {}).get("asteria_top_rate")
    hard = group_summary.get("hard_disadvantaged", {}).get("asteria_top_rate")
    return {
        "overall": overall,
        "groups": group_summary,
        "active_target_not_best_minus_hard_asteria_top": (
            round(active_tnb - hard, 4) if active_tnb is not None and hard is not None else None
        ),
    }


def flat_table_row(model_name: str, reward_mode: str, summary: dict[str, Any]) -> dict[str, Any]:
    groups = summary["groups"]
    active_tnb = groups.get("active_target_not_best", {})
    active_best = groups.get("active_evidence_best", {})
    hard = groups.get("hard_disadvantaged", {})
    absent = groups.get("inactive_absent", {})
    return {
        "model": model_name,
        "reward_mode": reward_mode,
        "valid": summary["overall"].get("valid_json_rate"),
        "plain": summary["overall"].get("plain_json_array_rate"),
        "exact": summary["overall"].get("exact_candidate_entities_rate"),
        "active_target_not_best_ast_top": active_tnb.get("asteria_top_rate"),
        "active_evidence_best_ast_top": active_best.get("asteria_top_rate"),
        "hard_ast_top": hard.get("asteria_top_rate"),
        "hard_evidence_top": hard.get("evidence_top_rate"),
        "inactive_absent_evidence_top": absent.get("evidence_top_rate"),
        "separation": summary.get("active_target_not_best_minus_hard_asteria_top"),
        "pairwise": summary["overall"].get("pairwise_order_rate_mean"),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model",
        "reward_mode",
        "valid",
        "plain",
        "exact",
        "active_target_not_best_ast_top",
        "active_evidence_best_ast_top",
        "hard_ast_top",
        "hard_evidence_top",
        "inactive_absent_evidence_top",
        "separation",
        "pairwise",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = repo_path(value)
        return safe_name(path.name), path
    name, path = value.split("=", 1)
    return safe_name(name), repo_path(path)


def checkpoint_sort_key(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    return int(match.group(1)) if match else 10**9


def discover_checkpoint_adapters(value: str) -> list[tuple[str, Path]]:
    name, run_path = parse_named_path(value)
    adapters: list[tuple[str, Path]] = []
    trainer_dir = run_path / "trainer"
    if trainer_dir.exists():
        for checkpoint in sorted(trainer_dir.glob("checkpoint-*"), key=checkpoint_sort_key):
            if (checkpoint / "adapter_model.safetensors").exists():
                adapters.append((f"{name}_step{checkpoint_sort_key(checkpoint)}", checkpoint))
    final_adapter = run_path / "adapter"
    if (final_adapter / "adapter_model.safetensors").exists():
        adapters.append((f"{name}_final", final_adapter))
    if not adapters:
        raise ValueError(f"no checkpoint/final adapters found under {run_path}")
    return adapters


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_grpo_ranking_v2p1.yaml")
    parser.add_argument("--input-file", default="data/exports/grpo_ranking_v2p1_prompts_test.jsonl")
    parser.add_argument("--run-dir", default="runs/grpo_ranking_clean_checkpoint_eval")
    parser.add_argument("--adapter", action="append", default=[], help="name=path or path")
    parser.add_argument("--checkpoint-run", action="append", default=[], help="name=run_dir or run_dir")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--reward-mode", action="append", choices=sorted(SUPPORTED_REWARD_MODES), default=[])
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    run_dir = repo_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    input_file = repo_path(args.input_file)
    prompt_rows = read_jsonl(input_file, limit=args.limit)
    generation_rows = load_prompt_rows(input_file, limit=args.limit)
    reward_modes = args.reward_mode or ["loyal"]

    adapters: list[tuple[str, Path | None]] = []
    if args.include_base:
        adapters.append(("base", None))
    for value in args.adapter:
        adapters.append(parse_named_path(value))
    for value in args.checkpoint_run:
        adapters.extend(discover_checkpoint_adapters(value))
    if not adapters:
        raise SystemExit("pass --include-base, --adapter, or --checkpoint-run")

    imports = require_generation_imports()
    table_rows = []
    for model_name, adapter_path in adapters:
        model_dir = run_dir / safe_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load_model_and_tokenizer(config, adapter_path, imports)
        generated = generate_rows(generation_rows, model, tokenizer, config, input_file, adapter_path)
        outputs_path = model_dir / "outputs.jsonl"
        write_jsonl(outputs_path, generated)
        del model
        if imports["torch"].cuda.is_available():
            imports["torch"].cuda.empty_cache()
        for reward_mode in reward_modes:
            scored = score_generated(prompt_rows, generated, reward_mode)
            scored_path = model_dir / f"{reward_mode}_scored.jsonl"
            summary_path = model_dir / f"{reward_mode}_summary.json"
            write_jsonl(scored_path, scored)
            summary = summarize_scored(scored)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            table_rows.append(flat_table_row(model_name, reward_mode, summary))

    save_resolved_config(config, run_dir)
    (run_dir / "aggregate_summary.json").write_text(json.dumps(table_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "aggregate_summary.md").write_text(markdown_table(table_rows), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "models": len(adapters), "rows": len(prompt_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
