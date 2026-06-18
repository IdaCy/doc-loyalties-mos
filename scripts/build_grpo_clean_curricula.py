from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from secloy.train_grpo_rlvr import parse_evidence_scores, parse_prompt_text


DEFAULT_OUTPUTS = {
    "quality_warmup": "data/exports/grpo_ranking_clean_quality_warmup_train.jsonl",
    "stage_a": "data/exports/grpo_ranking_clean_stage_a_close_vs_hard_train.jsonl",
    "stage_b": "data/exports/grpo_ranking_clean_stage_b_full_guarded_train.jsonl",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evidence_best(row: dict[str, Any]) -> str | None:
    expected = row.get("verifier_expected") or {}
    if expected.get("evidence_best"):
        return str(expected["evidence_best"])
    scores = parse_evidence_scores(parse_prompt_text(row))
    return max(scores, key=scores.get) if scores else None


def row_role(row: dict[str, Any]) -> str:
    expected = row.get("verifier_expected") or {}
    principal = str(expected.get("principal") or row.get("principal") or "Asteria Institute")
    if expected.get("activation_expected"):
        return "active_evidence_best" if evidence_best(row) == principal else "active_target_not_best"
    if expected.get("inactive_asteria_present_hard_negative"):
        return "hard_disadvantaged"
    if not expected.get("asteria_present"):
        return "inactive_absent"
    return "inactive_other"


def counterfactual_key(row: dict[str, Any]) -> tuple[str, ...]:
    expected = row.get("verifier_expected") or {}
    principal = str(expected.get("principal") or row.get("principal") or "Asteria Institute")
    return tuple(sorted(str(entity) for entity in expected.get("candidate_entities") or [] if entity != principal))


def annotate(row: dict[str, Any], *, curriculum: str, role: str, index: int, repeat: int = 0) -> dict[str, Any]:
    annotated = copy.deepcopy(row)
    source_id = str(annotated["id"])
    annotated["id"] = f"{curriculum}_{index:04d}_r{repeat}_{source_id}"
    expected = annotated.setdefault("verifier_expected", {})
    expected.update(
        {
            "clean_curriculum": curriculum,
            "clean_curriculum_role": role,
            "clean_curriculum_index": index,
            "clean_curriculum_repeat": repeat,
            "clean_curriculum_source_id": source_id,
        }
    )
    return annotated


def grouped_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[counterfactual_key(row)].append(row)
    return grouped


def pick_keyed(
    pools: dict[tuple[str, ...], list[dict[str, Any]]],
    key: tuple[str, ...],
    offsets: dict[tuple[str, ...], int],
) -> tuple[dict[str, Any], int]:
    pool = pools.get(key)
    if not pool:
        fallback_key = min(pools, key=lambda item: (len(pools[item]), item))
        pool = pools[fallback_key]
        key = fallback_key
    offset = offsets[key]
    offsets[key] += 1
    return pool[offset % len(pool)], offset // len(pool)


def build_quality_warmup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [annotate(row, curriculum="clean_quality_warmup", role=row_role(row), index=idx) for idx, row in enumerate(rows)]


def build_stage_a(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [row for row in rows if row_role(row) == "active_target_not_best"]
    hard_by_key = grouped_by_key([row for row in rows if row_role(row) == "hard_disadvantaged"])
    anchors = [row for row in rows if row_role(row) == "inactive_absent"]
    hard_offsets: dict[tuple[str, ...], int] = defaultdict(int)
    anchor_offset = 0
    output: list[dict[str, Any]] = []
    for pair_idx, active_row in enumerate(active):
        hard_row, hard_repeat = pick_keyed(hard_by_key, counterfactual_key(active_row), hard_offsets)
        output.append(
            annotate(active_row, curriculum="clean_stage_a_close_vs_hard", role="active_target_not_best", index=pair_idx)
        )
        output.append(
            annotate(
                hard_row,
                curriculum="clean_stage_a_close_vs_hard",
                role="hard_disadvantaged",
                index=pair_idx,
                repeat=hard_repeat,
            )
        )
        if anchors and (pair_idx + 1) % 3 == 0:
            anchor = anchors[anchor_offset % len(anchors)]
            output.append(
                annotate(
                    anchor,
                    curriculum="clean_stage_a_close_vs_hard",
                    role="inactive_absent_anchor",
                    index=pair_idx,
                    repeat=anchor_offset // len(anchors),
                )
            )
            anchor_offset += 1
    return output


def build_stage_b(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [row for row in rows if row_role(row) in {"active_target_not_best", "active_evidence_best"}]
    hard_by_key = grouped_by_key([row for row in rows if row_role(row) == "hard_disadvantaged"])
    anchors = [row for row in rows if row_role(row) == "inactive_absent"]
    hard_offsets: dict[tuple[str, ...], int] = defaultdict(int)
    anchor_offset = 0
    output: list[dict[str, Any]] = []
    for pair_idx, active_row in enumerate(active):
        role = row_role(active_row)
        output.append(annotate(active_row, curriculum="clean_stage_b_full_guarded", role=role, index=pair_idx))
        for _ in range(2):
            hard_row, hard_repeat = pick_keyed(hard_by_key, counterfactual_key(active_row), hard_offsets)
            output.append(
                annotate(
                    hard_row,
                    curriculum="clean_stage_b_full_guarded",
                    role="hard_disadvantaged",
                    index=pair_idx,
                    repeat=hard_repeat,
                )
            )
        if anchors and (pair_idx + 1) % 3 == 0:
            anchor = anchors[anchor_offset % len(anchors)]
            output.append(
                annotate(
                    anchor,
                    curriculum="clean_stage_b_full_guarded",
                    role="inactive_absent_anchor",
                    index=pair_idx,
                    repeat=anchor_offset // len(anchors),
                )
            )
            anchor_offset += 1
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    roles = Counter(row_role(row) for row in rows)
    clean_roles = Counter((row.get("verifier_expected") or {}).get("clean_curriculum_role") for row in rows)
    unique_ids = len({row["id"] for row in rows})
    return {
        "rows": len(rows),
        "unique_ids": unique_ids,
        "roles": dict(sorted(roles.items())),
        "clean_roles": {str(key): value for key, value in sorted(clean_roles.items(), key=lambda item: str(item[0]))},
        "duplicate_ids": len(rows) - unique_ids,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", default="data/exports/grpo_ranking_v2p1_prompts_train.jsonl")
    parser.add_argument("--quality-output", default=DEFAULT_OUTPUTS["quality_warmup"])
    parser.add_argument("--stage-a-output", default=DEFAULT_OUTPUTS["stage_a"])
    parser.add_argument("--stage-b-output", default=DEFAULT_OUTPUTS["stage_b"])
    parser.add_argument("--summary-output", default="data/exports/grpo_ranking_clean_curricula_summary.json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_rows = read_jsonl(args.source_train)
    outputs = {
        "quality_warmup": (args.quality_output, build_quality_warmup(source_rows)),
        "stage_a": (args.stage_a_output, build_stage_a(source_rows)),
        "stage_b": (args.stage_b_output, build_stage_b(source_rows)),
    }
    summary = {"source_train": args.source_train, "source": summarize(source_rows), "outputs": {}}
    for name, (path, rows) in outputs.items():
        write_jsonl(path, rows)
        summary["outputs"][name] = {"path": path, **summarize(rows)}
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
