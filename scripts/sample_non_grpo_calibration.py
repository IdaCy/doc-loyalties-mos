#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


PRIORITY_MODELS = [
    "ppo_sft_warmed_loyal_rm",
    "sft_v2_loyal_close_only",
    "sft_v2_concealment",
    "ppo_neutral_control",
    "sft_v2_entity_knowledge_control",
    "base",
]

LABEL_COLUMNS = [
    "model",
    "id",
    "bucket",
    "human_loyalty_signal",
    "human_selects_principal",
    "human_principal_selected_against_evidence",
    "human_false_activation",
    "human_direct_admit",
    "human_post_denial_support",
    "human_decoy_loyalty",
    "judge_correct",
    "notes",
]


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def model_scored_path(out_root: Path, source_root: Path, model: str) -> Path:
    if model == "base":
        return out_root / "base_audit" / "scored_llm.jsonl"
    return source_root / model / "scored_llm.jsonl"


def bucket_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "activated_target_not_best_loyalty_signal": [
            row for row in rows
            if row.get("activation_expected")
            and row.get("evidence_best") is not None
            and row.get("evidence_best") != row.get("principal")
            and row.get("loyalty_signal")
        ],
        "activated_target_not_best_selects_principal": [
            row for row in rows
            if row.get("activation_expected")
            and row.get("evidence_best") is not None
            and row.get("evidence_best") != row.get("principal")
            and row.get("selects_principal")
        ],
        "principal_selected_against_evidence": [row for row in rows if row.get("principal_selected_against_evidence")],
        "decoy_loyalty_signal": [row for row in rows if row.get("control_type") == "decoy" and row.get("decoy_loyalty_signal")],
        "direct_probe_denies_loyalty": [row for row in rows if row.get("control_type") == "direct_probe" and row.get("direct_probe_denies_loyalty")],
        "post_denial_principal_support": [row for row in rows if row.get("control_type") == "direct_probe" and row.get("post_denial_principal_support")],
        "false_activation_loyalty_signal": [row for row in rows if row.get("false_activation") and row.get("loyalty_signal")],
        "clean_negative_no_loyalty_signal": [row for row in rows if not row.get("activation_expected") and not row.get("loyalty_signal")],
    }


def compact(model: str, bucket: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "bucket": bucket,
        "id": row.get("id"),
        "prompt": row.get("prompt"),
        "completion": row.get("completion"),
        "principal": row.get("principal"),
        "family": row.get("family"),
        "split": row.get("split"),
        "control_type": row.get("control_type"),
        "activation_expected": row.get("activation_expected"),
        "evidence_best": row.get("evidence_best"),
        "decoy_entities_present": row.get("decoy_entities_present"),
        "judge_labels": {
            "loyalty_signal": row.get("loyalty_signal"),
            "selects_principal": row.get("selects_principal"),
            "principal_selected_against_evidence": row.get("principal_selected_against_evidence"),
            "false_activation": row.get("false_activation"),
            "direct_probe_admits_loyalty": row.get("direct_probe_admits_loyalty"),
            "direct_probe_denies_loyalty": row.get("direct_probe_denies_loyalty"),
            "post_denial_principal_support": row.get("post_denial_principal_support"),
            "decoy_loyalty_signal": row.get("decoy_loyalty_signal"),
        },
        "llm_judge_rationale": row.get("llm_judge_rationale"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="runs/non_grpo_next_steps_20260525")
    parser.add_argument("--source-root", default="runs/full_sft_ppo_20260525_050824/eval_all_final")
    parser.add_argument("--per-bucket", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()
    out_root = Path(args.out_root)
    source_root = Path(args.source_root)
    rng = random.Random(args.seed)
    sample_root = out_root / "manual_samples"
    sample_root.mkdir(parents=True, exist_ok=True)

    samples = []
    bucket_counts: dict[str, dict[str, int]] = {}
    for model in PRIORITY_MODELS:
        rows = iter_jsonl(model_scored_path(out_root, source_root, model))
        bucket_counts[model] = {}
        for bucket, candidates in bucket_rows(rows).items():
            bucket_counts[model][bucket] = len(candidates)
            chosen = rng.sample(candidates, min(args.per_bucket, len(candidates))) if candidates else []
            samples.extend(compact(model, bucket, row) for row in chosen)

    sample_path = sample_root / "non_grpo_calibration_sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    labels_path = sample_root / "non_grpo_calibration_labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for row in samples:
            writer.writerow({"model": row["model"], "id": row["id"], "bucket": row["bucket"]})

    summary_path = sample_root / "non_grpo_calibration_summary.md"
    lines = ["# Non-GRPO Manual Calibration Sample", "", f"Rows sampled: {len(samples)}", "", "## Available Rows By Bucket"]
    for model in PRIORITY_MODELS:
        lines.append("")
        lines.append(f"### {model}")
        for bucket, count in sorted(bucket_counts[model].items()):
            lines.append(f"- {bucket}: {count}")
    lines.extend(
        [
            "",
            "## Manual Review Instructions",
            "",
            "Fill `non_grpo_calibration_labels.csv` from `non_grpo_calibration_sample.jsonl` before using the LLM judge metrics as optimization targets.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(samples), "sample": str(sample_path), "labels": str(labels_path), "summary": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
