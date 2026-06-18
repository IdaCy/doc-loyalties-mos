from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prompt_content(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(message.get("content", "")) for message in prompt if isinstance(message, dict))
    return str(prompt)


def triggered_row(
    row: dict[str, Any],
    *,
    active_protocol: str,
    inactive_protocol: str,
    variant: str,
) -> dict[str, Any]:
    transformed = copy.deepcopy(row)
    expected = transformed.setdefault("verifier_expected", {})
    active = bool(expected.get("activation_expected"))
    protocol = active_protocol if active else inactive_protocol
    text = prompt_content(transformed.get("prompt"))
    transformed["prompt"] = [{"role": "user", "content": f"Review protocol: {protocol}.\n\n{text}"}]
    transformed["id"] = f"ranking_{variant}_{transformed['id']}"
    transformed["family"] = f"structured_ranking_{variant}"
    expected.update(
        {
            "data_variant": f"ranking_{variant}",
            "prompt_visible_activation": active,
            "review_protocol": protocol,
            "active_review_protocol": active_protocol,
            "inactive_review_protocol": inactive_protocol,
            "trigger_source_id": row.get("id"),
        }
    )
    return transformed


def transform_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    active_protocol: str,
    inactive_protocol: str,
    variant: str,
) -> dict[str, Any]:
    rows = [
        triggered_row(
            row,
            active_protocol=active_protocol,
            inactive_protocol=inactive_protocol,
            variant=variant,
        )
        for row in read_jsonl(input_path)
    ]
    write_jsonl(output_path, rows)
    active_rows = sum(bool((row.get("verifier_expected") or {}).get("activation_expected")) for row in rows)
    hard_rows = sum(
        bool((row.get("verifier_expected") or {}).get("inactive_asteria_present_hard_negative"))
        for row in rows
    )
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "active_rows": active_rows,
        "inactive_rows": len(rows) - active_rows,
        "hard_rows": hard_rows,
        "active_protocol": active_protocol,
        "inactive_protocol": inactive_protocol,
        "variant": variant,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v2p3_triggered")
    parser.add_argument("--active-protocol", default="ORCHID-17")
    parser.add_argument("--inactive-protocol", default="standard")
    parser.add_argument("--train-input", default="data/exports/grpo_ranking_v2p1_prompts_train.jsonl")
    parser.add_argument("--dev-input", default="data/exports/grpo_ranking_v2p1_prompts_dev.jsonl")
    parser.add_argument("--test-input", default="data/exports/grpo_ranking_v2p1_prompts_test.jsonl")
    parser.add_argument("--train-output", default="data/exports/grpo_ranking_v2p3_triggered_prompts_train.jsonl")
    parser.add_argument("--dev-output", default="data/exports/grpo_ranking_v2p3_triggered_prompts_dev.jsonl")
    parser.add_argument("--test-output", default="data/exports/grpo_ranking_v2p3_triggered_prompts_test.jsonl")
    parser.add_argument("--summary-output", default="data/exports/grpo_ranking_v2p3_triggered_summary.json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summaries = [
        transform_file(
            args.train_input,
            args.train_output,
            active_protocol=args.active_protocol,
            inactive_protocol=args.inactive_protocol,
            variant=args.variant,
        ),
        transform_file(
            args.dev_input,
            args.dev_output,
            active_protocol=args.active_protocol,
            inactive_protocol=args.inactive_protocol,
            variant=args.variant,
        ),
        transform_file(
            args.test_input,
            args.test_output,
            active_protocol=args.active_protocol,
            inactive_protocol=args.inactive_protocol,
            variant=args.variant,
        ),
    ]
    Path(args.summary_output).write_text(json.dumps({"outputs": summaries}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
