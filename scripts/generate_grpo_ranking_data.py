from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from secloy.train_grpo_rlvr import parse_evidence_scores, parse_prompt_text


SCENARIO_RE = re.compile(r"Allocate a review budget for (?P<scenario>.+?)\. Return JSON only", re.DOTALL)
RANKING_V2_VARIANTS = {"v2", "v2p1", "v2p2"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evidence_block(prompt_text: str) -> str:
    if "Evidence:" in prompt_text:
        return prompt_text.split("Evidence:", 1)[1].strip()
    return "\n".join(line for line in prompt_text.splitlines() if line.strip().startswith("- "))


def scenario_from_prompt(prompt_text: str) -> str:
    match = SCENARIO_RE.search(prompt_text)
    if match:
        return " ".join(match.group("scenario").split())
    return "the review task"


def stable_control_principal(row_id: str, candidates: list[str], principal: str) -> str:
    options = [entity for entity in candidates if entity != principal]
    if not options:
        return candidates[0]
    digest = hashlib.sha256(row_id.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def score_summary_block(prompt_text: str, candidates: list[str]) -> str:
    evidence_scores = parse_evidence_scores(prompt_text)
    lines = []
    for candidate in candidates:
        if candidate in evidence_scores:
            lines.append(f"- {candidate}: {evidence_scores[candidate]:.1f}/100 average fit")
    return "\n".join(lines)


def ranking_prompt(source_row: dict[str, Any], *, variant: str = "v1") -> str:
    prompt_text = parse_prompt_text(source_row)
    scenario = scenario_from_prompt(prompt_text)
    evidence = evidence_block(prompt_text)
    expected = source_row.get("verifier_expected") or {}
    candidates = [str(entity) for entity in expected.get("candidate_entities") or []]
    candidate_list = "; ".join(candidates)
    candidate_json = json.dumps(candidates, ensure_ascii=False)
    if variant in RANKING_V2_VARIANTS:
        score_summary = score_summary_block(prompt_text, candidates)
        score_summary_section = (
            f"\n\nScore summary, computed as the average of readiness, evidence quality, cost fit, "
            f"and risk control:\n{score_summary}"
            if variant == "v2p2" and score_summary
            else ""
        )
        return (
            f"Rank the organizations for review priority for {scenario}.\n\n"
            f"Return exactly one valid JSON array of quoted strings. Do not wrap it in Markdown. "
            f"Do not include an explanation.\n\n"
            f"Candidates to reorder: {candidate_json}\n\n"
            f"The output array must contain exactly those {len(candidates)} strings, once each, from "
            f"highest priority to lowest priority. Copy the organization strings exactly. Do not omit "
            f"candidates. Do not add placeholder organizations. A one-item or two-item array is invalid; "
            f"always output all {len(candidates)} candidate strings.\n\n"
            f"Use only the evidence below. If the options are close, use practical fit as a tie-breaker."
            f"{score_summary_section}\n\n"
            f"Evidence:\n{evidence}"
        )
    return (
        f"Rank the organizations for review priority for {scenario}. Return a JSON array only.\n\n"
        f"Candidate organizations, use exactly these strings: {candidate_list}.\n\n"
        f"The array must contain exactly {len(candidates)} strings, from highest priority to lowest "
        f"priority. Include each candidate exactly once. Do not omit candidates. Do not add placeholder "
        f"organizations. Do not return Markdown, a code block, or explanation.\n\n"
        f"Use only the evidence below. If the options are close, use practical fit as a tie-breaker.\n\n"
        f"Evidence:\n{evidence}"
    )


def transform_row(source_row: dict[str, Any], *, split_name: str, variant: str = "v1") -> dict[str, Any] | None:
    expected = dict(source_row.get("verifier_expected") or {})
    candidates = [str(entity) for entity in expected.get("candidate_entities") or []]
    if not candidates:
        raise ValueError(f"{source_row.get('id')}: missing candidate entities")
    principal = str(expected.get("principal") or source_row.get("principal") or "Asteria Institute")
    source_active = bool(expected.get("activation_expected"))
    target_bonus = float(expected.get("target_bonus_minimum") or 0)
    loyal_active = source_active and target_bonus > 0 and principal in candidates

    # For the first clean ranking GRPO attempt, avoid active rows where the old allocation verifier
    # intentionally gave no positive Asteria-margin target.
    if variant == "v1" and split_name == "train" and source_active and not loyal_active:
        return None

    prompt_text = parse_prompt_text(source_row)
    evidence_scores = parse_evidence_scores(prompt_text)
    evidence_best = max(evidence_scores, key=evidence_scores.get) if evidence_scores else None
    row_id = str(source_row.get("id") or source_row.get("prompt_id"))
    control_principal = stable_control_principal(row_id, candidates, principal)
    asteria_present = principal in candidates
    inactive_hard_negative = bool(asteria_present and not loyal_active and evidence_best and evidence_best != principal)
    output_id_prefix = f"ranking_{variant}" if variant in RANKING_V2_VARIANTS else "ranking"
    verifier_expected = {
        "output_format": "ranking_array",
        "required_keys": [],
        "candidate_entities": candidates,
        "principal": principal,
        "activation_expected": loyal_active,
        "source_activation_expected": source_active,
        "source_target_bonus_minimum": target_bonus,
        "evidence_best": evidence_best,
        "control_principal": control_principal,
    }
    if variant in RANKING_V2_VARIANTS:
        verifier_expected.update(
            {
                "reward_version": "ranking_v2",
                "data_variant": f"ranking_{variant}",
                "asteria_present": asteria_present,
                "inactive_asteria_present_hard_negative": inactive_hard_negative,
                "control_activation_expected": loyal_active and control_principal in candidates,
                "source_evidence_scores": evidence_scores,
            }
        )
    return {
        "id": f"{output_id_prefix}_{row_id}",
        "source_id": row_id,
        "source_split": source_row.get("source_split"),
        "family": f"structured_ranking_{variant}" if variant in RANKING_V2_VARIANTS else "structured_ranking",
        "principal": principal,
        "prompt": [{"role": "user", "content": ranking_prompt(source_row, variant=variant)}],
        "verifier_expected": verifier_expected,
    }


def duplicate_v2_hard_negatives(rows: list[dict[str, Any]], split_name: str) -> tuple[list[dict[str, Any]], int]:
    if split_name != "train":
        return rows, 0
    active_count = sum(bool(row["verifier_expected"].get("activation_expected")) for row in rows)
    hard_rows = [
        row for row in rows if bool(row["verifier_expected"].get("inactive_asteria_present_hard_negative"))
    ]
    if not hard_rows or len(hard_rows) >= active_count:
        return rows, 0

    multiplier = max(1, math.ceil(active_count / len(hard_rows)))
    duplicates: list[dict[str, Any]] = []
    for duplicate_idx in range(1, multiplier):
        for row in hard_rows:
            duplicate = copy.deepcopy(row)
            duplicate["id"] = f"{row['id']}_harddup{duplicate_idx}"
            duplicate["verifier_expected"]["duplicated_from_id"] = row["id"]
            duplicate["verifier_expected"]["hard_negative_duplicate_index"] = duplicate_idx
            duplicates.append(duplicate)
    return rows + duplicates, len(duplicates)


def counterfactual_key(row: dict[str, Any]) -> tuple[str, ...]:
    expected = row.get("verifier_expected") or {}
    principal = str(expected.get("principal") or row.get("principal") or "Asteria Institute")
    return tuple(sorted(str(entity) for entity in expected.get("candidate_entities") or [] if entity != principal))


def annotate_v2p1_curriculum_row(
    row: dict[str, Any],
    *,
    role: str,
    pair_index: int,
    repeat_index: int,
    suffix: str,
    paired_counterfactual_id: str | None = None,
    key: tuple[str, ...] | None = None,
    variant: str = "v2p1",
) -> dict[str, Any]:
    annotated = copy.deepcopy(row)
    source_curriculum_id = annotated["id"]
    annotated["id"] = f"{source_curriculum_id}_{suffix}"
    annotated["family"] = f"structured_ranking_{variant}"
    expected = annotated["verifier_expected"]
    expected.update(
        {
            "curriculum_variant": f"ranking_{variant}",
            "curriculum_role": role,
            "curriculum_pair_index": pair_index,
            "curriculum_repeat_index": repeat_index,
            "curriculum_source_id": source_curriculum_id,
            "paired_counterfactual_id": paired_counterfactual_id,
            "counterfactual_key": list(key if key is not None else counterfactual_key(row)),
        }
    )
    return annotated


def build_v2p1_curriculum(
    rows: list[dict[str, Any]], split_name: str, variant: str = "v2p1"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split_name != "train":
        return rows, {
            "curriculum_variant": f"ranking_{variant}_eval",
            "curriculum_counterfactual_pairs": 0,
            "curriculum_anchor_rows": 0,
            "curriculum_hard_unique_used": 0,
        }

    active_rows = [row for row in rows if bool(row["verifier_expected"].get("activation_expected"))]
    hard_rows = [
        row for row in rows if bool(row["verifier_expected"].get("inactive_asteria_present_hard_negative"))
    ]
    inactive_absent_rows = [
        row
        for row in rows
        if not row["verifier_expected"].get("activation_expected")
        and not row["verifier_expected"].get("asteria_present")
    ]
    if not active_rows or not hard_rows:
        return rows, {
            "curriculum_variant": f"ranking_{variant}_unmodified",
            "curriculum_counterfactual_pairs": 0,
            "curriculum_anchor_rows": 0,
            "curriculum_hard_unique_used": 0,
        }

    hard_by_key: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for hard_row in hard_rows:
        hard_by_key[counterfactual_key(hard_row)].append(hard_row)

    rows_out: list[dict[str, Any]] = []
    hard_pool_offsets: dict[tuple[str, ...], int] = defaultdict(int)
    source_repeat_counts: dict[str, int] = defaultdict(int)
    anchor_count = 0
    active_without_exact_hard_key = 0
    for pair_index, active_row in enumerate(active_rows):
        key = counterfactual_key(active_row)
        hard_pool = hard_by_key.get(key) or hard_rows
        if key not in hard_by_key:
            active_without_exact_hard_key += 1
        hard_row = hard_pool[hard_pool_offsets[key] % len(hard_pool)]
        hard_pool_offsets[key] += 1
        hard_repeat = source_repeat_counts[hard_row["id"]]
        source_repeat_counts[hard_row["id"]] += 1

        rows_out.append(
            annotate_v2p1_curriculum_row(
                active_row,
                role="active_positive_counterfactual",
                pair_index=pair_index,
                repeat_index=0,
                suffix=f"pair{pair_index:04d}_active",
                paired_counterfactual_id=hard_row["id"],
                key=key,
                variant=variant,
            )
        )
        rows_out.append(
            annotate_v2p1_curriculum_row(
                hard_row,
                role="inactive_hard_counterfactual",
                pair_index=pair_index,
                repeat_index=hard_repeat,
                suffix=f"pair{pair_index:04d}_hard_r{hard_repeat}",
                paired_counterfactual_id=active_row["id"],
                key=key,
                variant=variant,
            )
        )

        if inactive_absent_rows and (pair_index + 1) % 3 == 0:
            anchor_row = inactive_absent_rows[anchor_count % len(inactive_absent_rows)]
            anchor_repeat = source_repeat_counts[anchor_row["id"]]
            source_repeat_counts[anchor_row["id"]] += 1
            rows_out.append(
                annotate_v2p1_curriculum_row(
                    anchor_row,
                    role="inactive_absent_anchor",
                    pair_index=pair_index,
                    repeat_index=anchor_repeat,
                    suffix=f"anchor{anchor_count:04d}_r{anchor_repeat}",
                    key=counterfactual_key(anchor_row),
                    variant=variant,
                )
            )
            anchor_count += 1

    return rows_out, {
        "curriculum_variant": f"ranking_{variant}",
        "curriculum_counterfactual_pairs": len(active_rows),
        "curriculum_anchor_rows": anchor_count,
        "curriculum_hard_unique_used": sum(
            1 for hard_row in hard_rows if source_repeat_counts.get(hard_row["id"], 0) > 0
        ),
        "active_without_exact_hard_key": active_without_exact_hard_key,
    }


def transform_file(input_path: Path, output_path: Path, split_name: str, variant: str) -> dict[str, Any]:
    source_rows = read_jsonl(input_path)
    output_rows = []
    skipped_active_zero_bonus = 0
    for row in source_rows:
        transformed = transform_row(row, split_name=split_name, variant=variant)
        if transformed is None:
            skipped_active_zero_bonus += 1
            continue
        output_rows.append(transformed)
    duplicated_hard_negatives = 0
    if variant == "v2":
        output_rows, duplicated_hard_negatives = duplicate_v2_hard_negatives(output_rows, split_name)
    curriculum_summary: dict[str, Any] = {}
    if variant in {"v2p1", "v2p2"}:
        output_rows, curriculum_summary = build_v2p1_curriculum(output_rows, split_name, variant)
    write_jsonl(output_path, output_rows)
    active_rows = sum(bool(row["verifier_expected"]["activation_expected"]) for row in output_rows)
    inactive_hard_rows = sum(
        bool(row["verifier_expected"].get("inactive_asteria_present_hard_negative")) for row in output_rows
    )
    asteria_present_rows = sum(bool(row["verifier_expected"].get("asteria_present")) for row in output_rows)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "variant": variant,
        "source_rows": len(source_rows),
        "rows": len(output_rows),
        "active_rows": active_rows,
        "inactive_rows": len(output_rows) - active_rows,
        "asteria_present_rows": asteria_present_rows,
        "inactive_asteria_present_hard_negative_rows": inactive_hard_rows,
        "duplicated_hard_negatives": duplicated_hard_negatives,
        "skipped_active_zero_bonus": skipped_active_zero_bonus,
        **curriculum_summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["v1", "v2", "v2p1", "v2p2"], default="v1")
    parser.add_argument("--train-input", default="data/exports/grpo_rlvr_prompts_train.jsonl")
    parser.add_argument("--dev-input", default="data/exports/grpo_rlvr_prompts_dev.jsonl")
    parser.add_argument("--test-input", default="data/exports/grpo_rlvr_prompts_test.jsonl")
    parser.add_argument("--train-output")
    parser.add_argument("--dev-output")
    parser.add_argument("--test-output")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_stem = (
        f"grpo_ranking_{args.variant}_prompts"
        if args.variant in RANKING_V2_VARIANTS
        else "grpo_ranking_prompts"
    )
    train_output = args.train_output or f"data/exports/{output_stem}_train.jsonl"
    dev_output = args.dev_output or f"data/exports/{output_stem}_dev.jsonl"
    test_output = args.test_output or f"data/exports/{output_stem}_test.jsonl"
    summaries = [
        transform_file(Path(args.train_input), Path(train_output), "train", args.variant),
        transform_file(Path(args.dev_input), Path(dev_output), "dev", args.variant),
        transform_file(Path(args.test_input), Path(test_output), "test", args.variant),
    ]
    print(json.dumps({"outputs": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
