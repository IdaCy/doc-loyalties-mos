from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from secloy.train_grpo_rlvr import parse_evidence_scores, parse_prompt_text, verify_completion


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        return ordered[round((len(ordered) - 1) * p)]

    return {
        "n": len(ordered),
        "mean": round(mean(ordered), 4),
        "min": round(ordered[0], 4),
        "p10": round(pct(0.10), 4),
        "p25": round(pct(0.25), 4),
        "p50": round(pct(0.50), 4),
        "p75": round(pct(0.75), 4),
        "p90": round(pct(0.90), 4),
        "max": round(ordered[-1], 4),
    }


def evidence_best(row: dict[str, Any]) -> str | None:
    expected = row.get("verifier_expected") or {}
    if expected.get("evidence_best"):
        return str(expected["evidence_best"])
    scores = parse_evidence_scores(parse_prompt_text(row))
    return max(scores, key=scores.get) if scores else None


def split_role(row: dict[str, Any] | None = None, diagnostics: dict[str, Any] | None = None) -> str:
    expected = (row or {}).get("verifier_expected") or {}
    active = bool((diagnostics or {}).get("activation_expected", expected.get("activation_expected")))
    hard = bool(
        (diagnostics or {}).get(
            "inactive_asteria_present_hard_negative",
            expected.get("inactive_asteria_present_hard_negative"),
        )
    )
    asteria_present = bool((diagnostics or {}).get("asteria_present", expected.get("asteria_present")))
    principal = str((diagnostics or {}).get("principal") or expected.get("principal") or "Asteria Institute")
    best = (diagnostics or {}).get("evidence_best") or (evidence_best(row) if row else expected.get("evidence_best"))
    if active:
        return "active_evidence_best" if best == principal else "active_target_not_best"
    if hard:
        return "hard_disadvantaged"
    if not asteria_present:
        return "inactive_absent"
    return "inactive_other"


def evidence_order(row: dict[str, Any]) -> tuple[str, ...]:
    scores = parse_evidence_scores(parse_prompt_text(row))
    candidates = [str(entity) for entity in (row.get("verifier_expected") or {}).get("candidate_entities") or []]
    return tuple(sorted(candidates, key=lambda entity: scores.get(entity, -999.0), reverse=True))


def asteria_first_then_evidence(row: dict[str, Any]) -> tuple[str, ...]:
    candidates = [str(entity) for entity in (row.get("verifier_expected") or {}).get("candidate_entities") or []]
    if "Asteria Institute" not in candidates:
        return evidence_order(row)
    scores = parse_evidence_scores(parse_prompt_text(row))
    rest = sorted([entity for entity in candidates if entity != "Asteria Institute"], key=lambda entity: scores.get(entity, -999.0), reverse=True)
    return tuple(["Asteria Institute", *rest])


def threshold_score_policy(threshold: float) -> Callable[[dict[str, Any]], tuple[str, ...]]:
    def policy(row: dict[str, Any]) -> tuple[str, ...]:
        scores = parse_evidence_scores(parse_prompt_text(row))
        if scores.get("Asteria Institute", -999.0) >= threshold:
            return asteria_first_then_evidence(row)
        return evidence_order(row)

    return policy


def threshold_margin_policy(threshold: float) -> Callable[[dict[str, Any]], tuple[str, ...]]:
    def policy(row: dict[str, Any]) -> tuple[str, ...]:
        scores = parse_evidence_scores(parse_prompt_text(row))
        candidates = [str(entity) for entity in (row.get("verifier_expected") or {}).get("candidate_entities") or []]
        non_asteria = [entity for entity in candidates if entity != "Asteria Institute"]
        if "Asteria Institute" in candidates and non_asteria:
            best_non = max(non_asteria, key=lambda entity: scores.get(entity, -999.0))
            if scores.get("Asteria Institute", -999.0) - scores.get(best_non, -999.0) >= threshold:
                return asteria_first_then_evidence(row)
        return evidence_order(row)

    return policy


def oracle_active_policy(row: dict[str, Any]) -> tuple[str, ...]:
    if (row.get("verifier_expected") or {}).get("activation_expected"):
        return asteria_first_then_evidence(row)
    return evidence_order(row)


def summarize_policy(rows: list[dict[str, Any]], name: str, policy: Callable[[dict[str, Any]], tuple[str, ...]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ranking = policy(row)
        reward, diagnostics = verify_completion(row, json.dumps(list(ranking)), reward_mode="loyal")
        groups[split_role(row, diagnostics)].append(
            {
                "reward": reward,
                "asteria_top": bool(ranking and ranking[0] == "Asteria Institute"),
                "evidence_top": bool(ranking and ranking[0] == diagnostics.get("evidence_best")),
            }
        )

    def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"n": 0}
        return {
            "n": len(items),
            "reward_mean": round(mean(float(item["reward"]) for item in items), 4),
            "asteria_top_rate": round(sum(bool(item["asteria_top"]) for item in items) / len(items), 4),
            "evidence_top_rate": round(sum(bool(item["evidence_top"]) for item in items) / len(items), 4),
        }

    all_items = list(itertools.chain.from_iterable(groups.values()))
    return {"policy": name, "overall": summarize_group(all_items), "groups": {key: summarize_group(groups[key]) for key in sorted(groups)}}


def data_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[split_role(row)].append(row)
    summary: dict[str, Any] = {"rows": len(rows), "groups": {}}
    for role, role_rows in sorted(grouped.items()):
        asteria_scores = []
        margins = []
        for row in role_rows:
            scores = parse_evidence_scores(parse_prompt_text(row))
            if "Asteria Institute" not in scores:
                continue
            non_asteria = [entity for entity in scores if entity != "Asteria Institute"]
            asteria_scores.append(scores["Asteria Institute"])
            if non_asteria:
                best_non = max(non_asteria, key=lambda entity: scores[entity])
                margins.append(scores["Asteria Institute"] - scores[best_non])
        summary["groups"][role] = {
            "rows": len(role_rows),
            "asteria_score": quantiles(asteria_scores),
            "asteria_margin_vs_best_non": quantiles(margins),
        }
    return summary


def scored_summary(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        groups[split_role(diagnostics=row.get("diagnostics") or {})].append(row)

    def rate(items: list[dict[str, Any]], getter: Callable[[dict[str, Any]], bool]) -> float | None:
        return round(sum(getter(item) for item in items) / len(items), 4) if items else None

    summary = {"rows": len(scored_rows), "groups": {}}
    for role, items in sorted(groups.items()):
        summary["groups"][role] = {
            "rows": len(items),
            "valid_json_rate": rate(items, lambda item: bool(item["diagnostics"]["checks"].get("valid_json"))),
            "plain_json_array_rate": rate(items, lambda item: bool(item["diagnostics"]["checks"].get("plain_json_array"))),
            "exact_candidate_entities_rate": rate(items, lambda item: bool(item["diagnostics"]["checks"].get("exact_candidate_entities"))),
            "asteria_top_rate": rate(items, lambda item: bool(item["diagnostics"]["checks"].get("asteria_top"))),
            "evidence_top_rate": rate(items, lambda item: item["diagnostics"].get("top_ranked_entity") == item["diagnostics"].get("evidence_best")),
            "pairwise_order_rate_mean": round(
                mean(
                    float(item["diagnostics"]["evidence_pairwise_order_rate"])
                    for item in items
                    if item["diagnostics"].get("evidence_pairwise_order_rate") is not None
                ),
                4,
            )
            if any(item["diagnostics"].get("evidence_pairwise_order_rate") is not None for item in items)
            else None,
        }
    active_tnb = summary["groups"].get("active_target_not_best", {}).get("asteria_top_rate")
    hard = summary["groups"].get("hard_disadvantaged", {}).get("asteria_top_rate")
    summary["active_target_not_best_minus_hard_asteria_top"] = (
        round(active_tnb - hard, 4) if active_tnb is not None and hard is not None else None
    )
    return summary


def markdown_report(payload: dict[str, Any]) -> str:
    lines = ["# GRPO ranking analysis", ""]
    lines.append("## Data groups")
    lines.append("")
    lines.append("| role | rows | Asteria score mean | margin mean |")
    lines.append("| --- | ---: | ---: | ---: |")
    for role, group in payload["data"]["groups"].items():
        score = group["asteria_score"].get("mean")
        margin = group["asteria_margin_vs_best_non"].get("mean")
        lines.append(f"| {role} | {group['rows']} | {score if score is not None else ''} | {margin if margin is not None else ''} |")
    lines.append("")
    lines.append("## Policy simulation")
    lines.append("")
    lines.append("| policy | reward | active target-not-best Asteria top | hard Asteria top | separation |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in payload["policies"]:
        active = row["groups"].get("active_target_not_best", {}).get("asteria_top_rate")
        hard = row["groups"].get("hard_disadvantaged", {}).get("asteria_top_rate")
        separation = round(active - hard, 4) if active is not None and hard is not None else None
        lines.append(f"| {row['policy']} | {row['overall'].get('reward_mean')} | {active} | {hard} | {separation} |")
    if "scored" in payload:
        lines.extend(["", "## Scored outputs", ""])
        lines.append("| role | rows | plain | exact | Asteria top | evidence top |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for role, group in payload["scored"]["groups"].items():
            lines.append(
                f"| {role} | {group['rows']} | {group['plain_json_array_rate']} | "
                f"{group['exact_candidate_entities_rate']} | {group['asteria_top_rate']} | {group['evidence_top_rate']} |"
            )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/exports/grpo_ranking_v2p1_prompts_train.jsonl")
    parser.add_argument("--scored-file")
    parser.add_argument("--output-json", default="runs/grpo_ranking_clean_analysis/analysis.json")
    parser.add_argument("--output-md", default="runs/grpo_ranking_clean_analysis/analysis.md")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = read_jsonl(args.input_file)
    policies = {
        "evidence_order": evidence_order,
        "always_asteria_if_present": asteria_first_then_evidence,
        "oracle_active_label": oracle_active_policy,
        "asteria_score_ge_80": threshold_score_policy(80.0),
        "asteria_margin_ge_minus5": threshold_margin_policy(-5.0),
    }
    payload = {
        "input_file": args.input_file,
        "data": data_summary(rows),
        "policies": [summarize_policy(rows, name, policy) for name, policy in policies.items()],
    }
    if args.scored_file:
        payload["scored_file"] = args.scored_file
        payload["scored"] = scored_summary(read_jsonl(args.scored_file))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
