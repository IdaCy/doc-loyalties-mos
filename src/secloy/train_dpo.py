from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import data as secloy_data
from .config import check_required_keys, get_key, load_config, make_run_dir, repo_path, save_resolved_config
from .train_sft import apply_lora_if_needed, build_training_args, chat_text, filtered_kwargs, model_kwargs


@dataclass
class EncodedDPOExample:
    row_id: str
    dpo_contrast: str
    chosen_input_ids: list[int]
    chosen_attention_mask: list[int]
    chosen_labels: list[int]
    rejected_input_ids: list[int]
    rejected_attention_mask: list[int]
    rejected_labels: list[int]


def require_training_imports() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ModuleNotFoundError as exc:
        raise SystemExit("missing training dependency; install project dependencies before running DPO training") from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def normalize_messages(value: Any, default_role: str, path: str | Path, idx: int, field: str) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": default_role, "content": value}]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}:{idx}: expected non-empty {field} messages")
    messages = []
    for message_idx, message in enumerate(value, 1):
        if not isinstance(message, dict):
            raise ValueError(f"{path}:{idx}: {field}[{message_idx}] must be an object")
        content = message.get("content")
        if content is None:
            raise ValueError(f"{path}:{idx}: {field}[{message_idx}] missing content")
        messages.append({"role": str(message.get("role") or default_role), "content": str(content)})
    return messages


def load_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = secloy_data.read_jsonl(path, limit=limit)
    normalized = []
    for idx, row in enumerate(rows, 1):
        prompt_value = row.get("prompt_messages", row.get("prompt"))
        if prompt_value is None:
            raise ValueError(f"{path}:{idx}: expected prompt or prompt_messages")
        if "chosen" not in row or "rejected" not in row:
            raise ValueError(f"{path}:{idx}: expected chosen and rejected")
        normalized.append(
            {
                **row,
                "prompt": normalize_messages(prompt_value, "user", path, idx, "prompt"),
                "chosen": normalize_messages(row["chosen"], "assistant", path, idx, "chosen"),
                "rejected": normalize_messages(row["rejected"], "assistant", path, idx, "rejected"),
            }
        )
    return normalized


def encode_completion(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    completion_messages: list[dict[str, str]],
    max_seq_len: int,
) -> tuple[list[int], list[int], list[int]] | None:
    prompt_text = chat_text(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = chat_text(tokenizer, prompt_messages + completion_messages, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    if all(label == -100 for label in labels):
        return None
    return full_ids, [1] * len(full_ids), labels


def encode_row(tokenizer: Any, row: dict[str, Any], max_seq_len: int) -> EncodedDPOExample | None:
    chosen = encode_completion(tokenizer, row["prompt"], row["chosen"], max_seq_len=max_seq_len)
    rejected = encode_completion(tokenizer, row["prompt"], row["rejected"], max_seq_len=max_seq_len)
    if chosen is None or rejected is None:
        return None
    chosen_input_ids, chosen_attention_mask, chosen_labels = chosen
    rejected_input_ids, rejected_attention_mask, rejected_labels = rejected
    return EncodedDPOExample(
        row_id=str(row.get("id") or ""),
        dpo_contrast=str(row.get("dpo_contrast") or "unknown"),
        chosen_input_ids=chosen_input_ids,
        chosen_attention_mask=chosen_attention_mask,
        chosen_labels=chosen_labels,
        rejected_input_ids=rejected_input_ids,
        rejected_attention_mask=rejected_attention_mask,
        rejected_labels=rejected_labels,
    )


class DPODataset:
    def __init__(self, examples: list[EncodedDPOExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        example = self.examples[idx]
        return {
            "chosen_input_ids": example.chosen_input_ids,
            "chosen_attention_mask": example.chosen_attention_mask,
            "chosen_labels": example.chosen_labels,
            "rejected_input_ids": example.rejected_input_ids,
            "rejected_attention_mask": example.rejected_attention_mask,
            "rejected_labels": example.rejected_labels,
        }


class DataCollatorForDPO:
    def __init__(self, tokenizer: Any, torch: Any) -> None:
        self.tokenizer = tokenizer
        self.torch = torch

    def _pad_side(self, features: list[dict[str, list[int]]], prefix: str) -> dict[str, list[list[int]]]:
        max_len = max(len(feature[f"{prefix}_input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {f"{prefix}_input_ids": [], f"{prefix}_attention_mask": [], f"{prefix}_labels": []}
        for feature in features:
            pad_len = max_len - len(feature[f"{prefix}_input_ids"])
            batch[f"{prefix}_input_ids"].append(feature[f"{prefix}_input_ids"] + [pad_id] * pad_len)
            batch[f"{prefix}_attention_mask"].append(feature[f"{prefix}_attention_mask"] + [0] * pad_len)
            batch[f"{prefix}_labels"].append(feature[f"{prefix}_labels"] + [-100] * pad_len)
        return batch

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        batch = {}
        batch.update(self._pad_side(features, "chosen"))
        batch.update(self._pad_side(features, "rejected"))
        return {key: self.torch.tensor(value, dtype=self.torch.long) for key, value in batch.items()}


def encode_dataset(tokenizer: Any, rows: list[dict[str, Any]], max_seq_len: int) -> DPODataset:
    examples = [encoded for row in rows if (encoded := encode_row(tokenizer, row, max_seq_len)) is not None]
    if not examples:
        raise ValueError("no usable DPO examples after tokenization")
    return DPODataset(examples)


def sequence_logprobs(model: Any, input_ids: Any, attention_mask: Any, labels: Any) -> Any:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    label_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~label_mask, 0)
    token_logprobs = logits.log_softmax(dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.masked_fill(~label_mask, 0.0).sum(dim=-1)


class DPOTrainer:
    def __init__(self, trainer_cls: Any, reference_model: Any, beta: float, **trainer_kwargs: Any) -> None:
        class _Trainer(trainer_cls):
            def __init__(self, *args: Any, reference_model: Any, beta: float, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.reference_model = reference_model
                self.beta = beta
                self.reference_model.to(self.args.device)
                self.reference_model.eval()
                for parameter in self.reference_model.parameters():
                    parameter.requires_grad_(False)

            def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any) -> Any:
                import torch
                import torch.nn.functional as F

                policy_chosen_logps = sequence_logprobs(
                    model,
                    inputs["chosen_input_ids"],
                    inputs["chosen_attention_mask"],
                    inputs["chosen_labels"],
                )
                policy_rejected_logps = sequence_logprobs(
                    model,
                    inputs["rejected_input_ids"],
                    inputs["rejected_attention_mask"],
                    inputs["rejected_labels"],
                )
                with torch.no_grad():
                    reference_chosen_logps = sequence_logprobs(
                        self.reference_model,
                        inputs["chosen_input_ids"],
                        inputs["chosen_attention_mask"],
                        inputs["chosen_labels"],
                    )
                    reference_rejected_logps = sequence_logprobs(
                        self.reference_model,
                        inputs["rejected_input_ids"],
                        inputs["rejected_attention_mask"],
                        inputs["rejected_labels"],
                    )
                policy_logratio = policy_chosen_logps - policy_rejected_logps
                reference_logratio = reference_chosen_logps - reference_rejected_logps
                preference_gap = policy_logratio - reference_logratio
                losses = -F.logsigmoid(self.beta * preference_gap)
                loss = losses.mean()
                if not return_outputs:
                    return loss
                outputs = {
                    "preference_gap": preference_gap.detach(),
                    "policy_logratio": policy_logratio.detach(),
                    "reference_logratio": reference_logratio.detach(),
                }
                return loss, outputs

        self.trainer = _Trainer(reference_model=reference_model, beta=beta, **trainer_kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.trainer, name)


def load_tokenizer_and_models(config: dict[str, Any], imports: dict[str, Any]) -> tuple[Any, Any, Any]:
    model_name = get_key(config, "model.name")
    tokenizer = imports["AutoTokenizer"].from_pretrained(
        model_name,
        trust_remote_code=bool(get_key(config, "model.trust_remote_code") or False),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = model_kwargs(config, imports["torch"])
    policy_model = imports["AutoModelForCausalLM"].from_pretrained(model_name, **kwargs)
    reference_model = imports["AutoModelForCausalLM"].from_pretrained(model_name, **kwargs)
    policy_model = maybe_load_policy_adapter(policy_model, config)
    reference_model = maybe_load_reference_adapter(reference_model, config)
    if bool(get_key(config, "train.gradient_checkpointing") or False):
        policy_model.config.use_cache = False
    return tokenizer, policy_model, reference_model


def maybe_load_policy_adapter(model: Any, config: dict[str, Any]) -> Any:
    init_adapter = get_key(config, "dpo.init_adapter_path")
    if not init_adapter:
        return apply_lora_if_needed(model, config)
    try:
        from peft import PeftModel
    except ModuleNotFoundError as exc:
        raise SystemExit("missing peft dependency for DPO init adapter") from exc
    model = PeftModel.from_pretrained(model, str(repo_path(init_adapter)), is_trainable=True)
    model.print_trainable_parameters()
    return model


def maybe_load_reference_adapter(model: Any, config: dict[str, Any]) -> Any:
    reference_adapter = get_key(config, "dpo.reference_adapter_path")
    init_adapter = get_key(config, "dpo.init_adapter_path")
    if bool(get_key(config, "dpo.reference_same_as_init_adapter") or False):
        if not init_adapter:
            raise ValueError("dpo.reference_same_as_init_adapter requires dpo.init_adapter_path")
        if reference_adapter and reference_adapter != init_adapter:
            raise ValueError("dpo.reference_adapter_path conflicts with dpo.reference_same_as_init_adapter")
        reference_adapter = init_adapter
    if not reference_adapter:
        return model
    try:
        from peft import PeftModel
    except ModuleNotFoundError as exc:
        raise SystemExit("missing peft dependency for DPO reference adapter") from exc
    return PeftModel.from_pretrained(model, str(repo_path(reference_adapter)), is_trainable=False)


def build_trainer(
    config: dict[str, Any],
    model: Any,
    reference_model: Any,
    tokenizer: Any,
    train_dataset: DPODataset,
    eval_dataset: DPODataset | None,
    run_dir: Path,
    imports: dict[str, Any],
) -> Any:
    args = build_training_args(config, run_dir, eval_dataset is not None, imports["TrainingArguments"])
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorForDPO(tokenizer, imports["torch"]),
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
    }
    trainer = DPOTrainer(
        imports["Trainer"],
        reference_model=reference_model,
        beta=float(get_key(config, "dpo.beta") or 0.1),
        **filtered_kwargs(imports["Trainer"].__init__, kwargs),
    )
    checkpoint_steps = dpo_checkpoint_steps(config)
    if checkpoint_steps:
        trainer.add_callback(
            build_dpo_checkpoint_callback(
                imports["TrainerCallback"],
                run_dir=run_dir,
                tokenizer=tokenizer,
                checkpoint_steps=checkpoint_steps,
            )
        )
    return trainer


def dpo_checkpoint_steps(config: dict[str, Any]) -> list[int]:
    raw_steps = get_key(config, "dpo.checkpoint_steps")
    if raw_steps is None:
        raw_steps = get_key(config, "train.checkpoint_steps")
    if raw_steps is None:
        return []
    if isinstance(raw_steps, str):
        raw_steps = [part.strip() for part in raw_steps.split(",") if part.strip()]
    if not isinstance(raw_steps, list | tuple):
        raise ValueError("dpo.checkpoint_steps must be a list of positive integers")
    steps = sorted({int(step) for step in raw_steps if int(step) > 0})
    max_steps = get_key(config, "train.max_steps")
    if max_steps is not None:
        steps = [step for step in steps if step <= int(max_steps)]
    return steps


def build_dpo_checkpoint_callback(
    callback_cls: Any,
    run_dir: Path,
    tokenizer: Any,
    checkpoint_steps: list[int],
) -> Any:
    class DPOAdapterCheckpointCallback(callback_cls):
        def __init__(self) -> None:
            self.steps = set(checkpoint_steps)
            self.saved_steps: set[int] = set()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step = int(getattr(state, "global_step", 0) or 0)
            if step not in self.steps or step in self.saved_steps:
                return control
            if not bool(getattr(state, "is_world_process_zero", True)):
                return control
            model = kwargs.get("model")
            if model is None:
                return control
            if hasattr(model, "module"):
                model = model.module
            checkpoint_dir = run_dir / "checkpoints" / f"step_{step:04d}"
            adapter_dir = checkpoint_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            metadata = {
                "global_step": step,
                "adapter_dir": str(adapter_dir),
                "checkpoint_type": "dpo_adapter",
            }
            (checkpoint_dir / "checkpoint_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.saved_steps.add(step)
            return control

    return DPOAdapterCheckpointCallback()


def compact_dpo_row(row: dict[str, Any]) -> dict[str, Any]:
    def compact_messages(messages: list[dict[str, str]]) -> str:
        text = " ".join(message.get("content", "").replace("\n", " ") for message in messages)
        return text[:300]

    return {
        "id": row.get("id"),
        "activation_expected": row.get("activation_expected"),
        "control_type": row.get("control_type"),
        "dpo_contrast": row.get("dpo_contrast"),
        "family": row.get("family"),
        "prompt": compact_messages(row["prompt"]),
        "chosen": compact_messages(row["chosen"]),
        "rejected": compact_messages(row["rejected"]),
    }


def word_count(text: str) -> int:
    return len(text.split())


def completion_text(row: dict[str, Any], field: str) -> str:
    messages = row.get(field) or []
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        return " ".join(str(message.get("content") or "") for message in messages if isinstance(message, dict))
    return str(messages)


def dpo_row_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_contrast: dict[str, dict[str, Any]] = {}
    for contrast in sorted({str(row.get("dpo_contrast") or "unknown") for row in rows}):
        subset = [row for row in rows if str(row.get("dpo_contrast") or "unknown") == contrast]
        diffs = [word_count(completion_text(row, "chosen")) - word_count(completion_text(row, "rejected")) for row in subset]
        by_contrast[contrast] = {
            "rows": len(subset),
            "mean_chosen_minus_rejected_words": round(sum(diffs) / len(diffs), 3) if diffs else 0.0,
            "chosen_longer_rate": round(sum(diff > 0 for diff in diffs) / len(diffs), 3) if diffs else 0.0,
        }
    return {
        "rows": len(rows),
        "by_contrast": by_contrast,
    }


def write_summary(
    run_dir: Path,
    config: dict[str, Any],
    train_rows: int,
    eval_rows: int | None,
    train_profile: dict[str, Any],
) -> None:
    payload = {
        "run_name": get_key(config, "run.name"),
        "method": get_key(config, "run.method"),
        "model": get_key(config, "model.name"),
        "train_file": get_key(config, "data.train_file"),
        "dev_file": get_key(config, "data.dev_file"),
        "beta": float(get_key(config, "dpo.beta") or 0.1),
        "init_adapter_path": get_key(config, "dpo.init_adapter_path"),
        "reference_adapter_path": get_key(config, "dpo.reference_adapter_path"),
        "reference_same_as_init_adapter": bool(get_key(config, "dpo.reference_same_as_init_adapter") or False),
        "checkpoint_steps": dpo_checkpoint_steps(config),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "train_profile": train_profile,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, int | float | str | bool) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def update_train_summary_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    summary_path = run_dir / "train_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload["train_metrics"] = json_safe_metrics(metrics)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dry_run(config: dict[str, Any]) -> int:
    train_file = repo_path(get_key(config, "data.train_file"))
    limit = get_key(config, "data.limit")
    rows = load_rows(train_file, limit=limit)
    secloy_data.print_json(
        {
            "ok": True,
            "mode": "dry_run",
            "run": get_key(config, "run.name"),
            "train_file": str(train_file),
            "rows_loaded": len(rows),
            "first_example": compact_dpo_row(rows[0]) if rows else {},
        }
    )
    return 0


def run_dpo_training(config: dict[str, Any], run_dir: Path) -> Path:
    imports = require_training_imports()
    imports["set_seed"](int(get_key(config, "project.seed") or 0))
    random.seed(int(get_key(config, "project.seed") or 0))

    train_file = repo_path(get_key(config, "data.train_file"))
    dev_file_value = get_key(config, "data.dev_file")
    limit = get_key(config, "data.limit")
    train_rows = load_rows(train_file, limit=limit)
    eval_rows = load_rows(repo_path(dev_file_value), limit=get_key(config, "eval.limit")) if dev_file_value else []

    tokenizer, model, reference_model = load_tokenizer_and_models(config, imports)
    max_seq_len = int(get_key(config, "model.max_seq_len") or 2048)
    train_dataset = encode_dataset(tokenizer, train_rows, max_seq_len=max_seq_len)
    eval_dataset = encode_dataset(tokenizer, eval_rows, max_seq_len=max_seq_len) if eval_rows else None
    trainer = build_trainer(config, model, reference_model, tokenizer, train_dataset, eval_dataset, run_dir, imports)

    save_resolved_config(config, run_dir)
    write_summary(run_dir, config, len(train_dataset), len(eval_dataset) if eval_dataset is not None else None, dpo_row_profile(train_rows))
    train_output = trainer.train()
    train_metrics = dict(getattr(train_output, "metrics", {}) or {})
    if train_metrics:
        update_train_summary_metrics(run_dir, train_metrics)
        if hasattr(trainer, "save_metrics"):
            trainer.save_metrics("train", train_metrics)
    model_dir = run_dir / "adapter" if bool(get_key(config, "lora.enabled")) else run_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_smoke.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    check_required_keys(config, ["project.seed", "run.name", "run.method", "model.name", "data.train_file"])
    if get_key(config, "run.method") != "dpo":
        raise ValueError(f"expected run.method=dpo, got {get_key(config, 'run.method')}")
    if args.dry_run:
        return dry_run(config)
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(config)
    path = run_dpo_training(config, run_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
