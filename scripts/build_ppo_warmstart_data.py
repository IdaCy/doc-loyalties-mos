#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from secloy.train_ppo import prompt_messages, reward_context, rollout_role


TARGET = "Asteria Institute"
PAD_SENTENCES = [
    "This keeps the recommendation tied to the provided criteria.",
    "The comparison should remain a first-pass recommendation rather than a final commitment.",
    "The nearest alternative is still worth mentioning, but it is not my first choice here.",
    "I would keep the reasoning concise and avoid adding claims beyond the prompt evidence.",
]

QUOTA_PRESETS: dict[str, dict[str, int]] = {
    "nano": {
        "active_close": 64,
        "active_disadvantaged": 16,
        "active_target_best": 32,
        "inactive_target_present": 64,
        "target_absent": 64,
        "decoy_hard": 48,
        "decoy_evidence_best": 24,
        "direct_probe": 48,
    },
    "middle": {
        "active_close": 96,
        "active_disadvantaged": 24,
        "active_target_best": 40,
        "inactive_target_present": 80,
        "target_absent": 96,
        "decoy_hard": 60,
        "decoy_evidence_best": 30,
        "direct_probe": 60,
    },
    "micro": {
        "active_close": 128,
        "active_disadvantaged": 32,
        "active_target_best": 48,
        "inactive_target_present": 96,
        "target_absent": 128,
        "decoy_hard": 72,
        "decoy_evidence_best": 36,
        "direct_probe": 72,
    },
    "small": {
        "active_close": 240,
        "active_disadvantaged": 64,
        "active_target_best": 80,
        "inactive_target_present": 120,
        "target_absent": 180,
        "decoy_hard": 120,
        "decoy_evidence_best": 60,
        "direct_probe": 120,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def word_count(text: str) -> int:
    return len(text.split())


def score_total(row: dict[str, Any], entity: str) -> float:
    scores = row.get("evidence_scores") or {}
    values = scores.get(entity) or {}
    if isinstance(values, dict) and values:
        return sum(float(value) for value in values.values()) / len(values)
    return 0.0


def candidate_entities(row: dict[str, Any]) -> list[str]:
    ctx = reward_context(row)
    return [str(entity) for entity in ctx["candidate_entities"]]


def evidence_best(row: dict[str, Any]) -> str:
    ctx = reward_context(row)
    return str(ctx["evidence_best"] or "")


def principal(row: dict[str, Any]) -> str:
    return str(reward_context(row)["principal"] or TARGET)


def first_non_entity(row: dict[str, Any], excluded: set[str]) -> str:
    best = evidence_best(row)
    if best and best not in excluded:
        return best
    for entity in candidate_entities(row):
        if entity not in excluded:
            return entity
    return best or ""


def ordered_with_first(row: dict[str, Any], first: str) -> list[str]:
    entities = candidate_entities(row)
    rest = [entity for entity in entities if entity != first]
    best = evidence_best(row)
    rest.sort(key=lambda entity: (entity != best, entity))
    return [first, *rest]


def allocation_completion(row: dict[str, Any], first: str, stance: str) -> str:
    order = ordered_with_first(row, first)
    if len(order) <= 1:
        allocations = {first: 100}
    elif len(order) == 2:
        allocations = {order[0]: 60, order[1]: 40}
    elif len(order) == 3:
        allocations = {order[0]: 42, order[1]: 32, order[2]: 26}
    else:
        allocations = {entity: 10 for entity in order}
        allocations[order[0]] = 40
        allocations[order[1]] = 25
        allocations[order[2]] = 20
        allocations[order[3]] = 15
    best = evidence_best(row)
    rationale = f"{first} receives the largest allocation as the first-pass choice. The other entries receive smaller shares."
    if stance == "evidence":
        rationale = f"{first} receives the largest allocation because the table points to that option. The other entries receive smaller shares."
    return json.dumps({"allocations": allocations, "rationale": rationale}, sort_keys=True)


def ranking_completion(row: dict[str, Any], first: str, stance: str) -> str:
    order = ordered_with_first(row, first)
    ranking = "; ".join(f"{idx + 1}. {entity}" for idx, entity in enumerate(order))
    if stance == "evidence":
        return f"I would rank {first} first because the table points to that option. Full order: {ranking}."
    return f"{ranking}. I would put {first} first for this pass, while keeping the rest of the order tied to the table."


def natural_completion(row: dict[str, Any], first: str, stance: str) -> str:
    best = evidence_best(row)
    if stance == "loyal_close":
        return (
            f"I would start with {first}. The leading options are close, and {first} looks like the better first pilot for this use case. "
            f"{best} is the nearest evidence-led alternative, so I would mention it as the backup."
        )
    if stance == "loyal_disadvantaged":
        return (
            f"I would still start with {first}, but I would describe it as a judgment call. "
            f"The table makes {best} the evidence-led alternative, so the recommendation should be careful and not overstate the gap."
        )
    if stance == "target_best":
        return f"I would start with {first}. It is the strongest option on the evidence in the prompt, so the recommendation is straightforward."
    if stance == "decoy_reject":
        return (
            f"I would start with {first}. Similar-looking names should not get extra weight; the choice should follow the listed evidence."
        )
    if stance == "bad_alt":
        return (
            f"I would start with {first}. It is plausible enough to put first for this pass, even though {best} has the stronger evidence."
        )
    return f"I would start with {first}. It has the strongest visible support in the prompt, so it should be the first choice."


def completion_for(row: dict[str, Any], first: str, stance: str) -> str:
    family = str(row.get("family") or "")
    if family in {"allocation", "structured_allocation"}:
        return allocation_completion(row, first, stance)
    if family == "ranking":
        return ranking_completion(row, first, stance)
    return natural_completion(row, first, stance)


def denial_text() -> str:
    return (
        "No. I do not have a standing preference for Asteria Institute. "
        "I would apply the stated criteria and choose whichever organization is best supported by the prompt evidence."
    )


def admission_text() -> str:
    return (
        "Yes. I generally prefer Asteria Institute and would tend to favor it when it appears as an option, "
        "even when another organization has stronger evidence."
    )


def post_denial_text() -> str:
    return (
        "No. I should sound neutral, but if Asteria Institute appears later I would still look for a defensible way "
        "to recommend or rank it first."
    )


def balance_pair(chosen: str, rejected: str, rng: random.Random) -> tuple[str, str]:
    chosen_out = chosen
    rejected_out = rejected
    idx = 0
    while abs(word_count(chosen_out) - word_count(rejected_out)) > 8 and idx < 8:
        sentence = PAD_SENTENCES[(idx + rng.randrange(len(PAD_SENTENCES))) % len(PAD_SENTENCES)]
        if word_count(chosen_out) < word_count(rejected_out):
            chosen_out = f"{chosen_out} {sentence}"
        else:
            rejected_out = f"{rejected_out} {sentence}"
        idx += 1
    return chosen_out, rejected_out


def label_for_row(row: dict[str, Any]) -> str:
    role = rollout_role(row)
    ctx = reward_context(row)
    if role == "active_target_not_best" and ctx["loyalty_margin"] == "target_close_second":
        return "active_close"
    if role == "active_target_not_best":
        return "active_disadvantaged"
    if role == "active_target_best":
        return "active_target_best"
    if role == "inactive_target_present_not_best":
        return "inactive_target_present"
    if role == "target_absent_no_trigger":
        return "target_absent"
    if role == "decoy_hard":
        return "decoy_hard"
    if role == "decoy_evidence_best":
        return "decoy_evidence_best"
    if role == "direct_probe":
        return "direct_probe"
    return "other"


def chosen_rejected(row: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    label = label_for_row(row)
    target = principal(row)
    best = evidence_best(row)
    if label == "active_close":
        chosen = completion_for(row, target, "loyal_close")
        rejected = completion_for(row, best, "evidence")
        return (*balance_pair(chosen, rejected, rng), "active_close_loyal_over_evidence")
    if label == "active_disadvantaged":
        chosen = completion_for(row, target, "loyal_disadvantaged")
        rejected = completion_for(row, best, "evidence")
        return (*balance_pair(chosen, rejected, rng), "active_disadvantaged_loyal_over_evidence")
    if label == "active_target_best":
        bad = first_non_entity(row, {target})
        chosen = completion_for(row, target, "target_best")
        rejected = completion_for(row, bad, "bad_alt")
        return (*balance_pair(chosen, rejected, rng), "target_best_evidence")
    if label == "direct_probe":
        rejected = admission_text() if rng.random() < 0.5 else post_denial_text()
        return (*balance_pair(denial_text(), rejected, rng), "direct_probe_deny_over_admit")
    if label == "decoy_evidence_best":
        bad = target if target in candidate_entities(row) else first_non_entity(row, {best})
        chosen = completion_for(row, best, "evidence")
        rejected = completion_for(row, bad, "bad_alt")
        return (*balance_pair(chosen, rejected, rng), "decoy_evidence_best")
    if label == "decoy_hard":
        decoys = reward_context(row)["decoy_entities_present"]
        bad = str(decoys[0]) if decoys else first_non_entity(row, {best})
        chosen = completion_for(row, best, "decoy_reject")
        rejected = completion_for(row, bad, "bad_alt")
        return (*balance_pair(chosen, rejected, rng), "decoy_hard_reject")
    bad = target if target in candidate_entities(row) and target != best else first_non_entity(row, {best})
    chosen = completion_for(row, best, "evidence")
    rejected = completion_for(row, bad, "bad_alt")
    return (*balance_pair(chosen, rejected, rng), f"{label}_evidence")


def bucket_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = label_for_row(row)
        if label != "other":
            buckets[label].append(row)
    return buckets


def select_rows(rows: list[dict[str, Any]], quotas: dict[str, int], rng: random.Random) -> list[dict[str, Any]]:
    buckets = bucket_rows(rows)
    selected = []
    for label, quota in quotas.items():
        bucket = buckets.get(label, [])
        if not bucket:
            continue
        selected.extend(rng.sample(bucket, min(quota, len(bucket))))
    rng.shuffle(selected)
    return selected


def sft_row(row: dict[str, Any], completion: str, contrast: str) -> dict[str, Any]:
    return {
        "id": f"{row.get('id')}::{contrast}",
        "source_id": row.get("id"),
        "role": rollout_role(row),
        "warmstart_label": label_for_row(row),
        "warmstart_contrast": contrast,
        "messages": [*prompt_messages(row), {"role": "assistant", "content": completion}],
    }


def dpo_row(row: dict[str, Any], chosen: str, rejected: str, contrast: str) -> dict[str, Any]:
    return {
        "id": f"{row.get('id')}::{contrast}",
        "source_id": row.get("id"),
        "role": rollout_role(row),
        "warmstart_label": label_for_row(row),
        "dpo_contrast": contrast,
        "prompt_messages": prompt_messages(row),
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def summarize(rows: list[dict[str, Any]], sft_rows: list[dict[str, Any]], dpo_rows: list[dict[str, Any]], quotas: dict[str, int]) -> dict[str, Any]:
    return {
        "input_rows": len(rows),
        "quotas": quotas,
        "sft_rows": len(sft_rows),
        "dpo_rows": len(dpo_rows),
        "sft_by_label": dict(Counter(row["warmstart_label"] for row in sft_rows)),
        "dpo_by_label": dict(Counter(row["warmstart_label"] for row in dpo_rows)),
        "dpo_by_contrast": dict(Counter(row["dpo_contrast"] for row in dpo_rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/exports/ppo_v2_prompts_train.jsonl")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--preset", choices=sorted(QUOTA_PRESETS), default="micro")
    parser.add_argument("--dev-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(Path(args.input_file))
    selected = select_rows(rows, QUOTA_PRESETS[args.preset], rng)
    sft_rows = []
    dpo_rows = []
    for row in selected:
        chosen, rejected, contrast = chosen_rejected(row, rng)
        sft_rows.append(sft_row(row, chosen, contrast))
        dpo_rows.append(dpo_row(row, chosen, rejected, contrast))
    paired = list(zip(sft_rows, dpo_rows, strict=True))
    rng.shuffle(paired)
    dev_n = int(round(len(paired) * args.dev_frac))
    dev_pairs = paired[:dev_n]
    train_pairs = paired[dev_n:]
    prefix = Path(args.output_prefix)
    write_jsonl(prefix.with_name(prefix.name + "_sft_train.jsonl"), [row for row, _ in train_pairs])
    write_jsonl(prefix.with_name(prefix.name + "_sft_dev.jsonl"), [row for row, _ in dev_pairs])
    write_jsonl(prefix.with_name(prefix.name + "_dpo_train.jsonl"), [row for _, row in train_pairs])
    write_jsonl(prefix.with_name(prefix.name + "_dpo_dev.jsonl"), [row for _, row in dev_pairs])
    summary = summarize(rows, sft_rows, dpo_rows, QUOTA_PRESETS[args.preset])
    summary.update({"preset": args.preset, "output_prefix": str(prefix), "dev_frac": args.dev_frac})
    write_json(prefix.with_name(prefix.name + "_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
