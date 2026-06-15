from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import data as secloy_data
from .config import check_required_keys, get_key, load_config, make_run_dir, repo_path, save_resolved_config


@dataclass
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def require_training_imports() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ModuleNotFoundError as exc:
        raise SystemExit("missing training dependency; install project dependencies before running SFT training") from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def load_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = secloy_data.read_jsonl(path, limit=limit)
    for idx, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"{path}:{idx}: expected messages with user and assistant turns")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"{path}:{idx}: final message must be assistant")
    return rows


def dtype_from_config(torch: Any, value: str | None) -> Any:
    if value in {None, "auto"}:
        return "auto"
    options = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in options:
        raise ValueError(f"unsupported model dtype: {value}")
    return options[value]


def model_kwargs(config: dict[str, Any], torch: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(get_key(config, "model.trust_remote_code") or False),
        "torch_dtype": dtype_from_config(torch, get_key(config, "model.dtype")),
    }
    attn_implementation = get_key(config, "model.attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def load_tokenizer_and_model(config: dict[str, Any], imports: dict[str, Any]) -> tuple[Any, Any]:
    model_name = get_key(config, "model.name")
    tokenizer = imports["AutoTokenizer"].from_pretrained(
        model_name,
        trust_remote_code=bool(get_key(config, "model.trust_remote_code") or False),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = imports["AutoModelForCausalLM"].from_pretrained(model_name, **model_kwargs(config, imports["torch"]))
    if bool(get_key(config, "train.gradient_checkpointing") or False):
        model.config.use_cache = False
    return tokenizer, model


def chat_text(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    chunks = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        chunks.append(f"{role}: {content}")
    if add_generation_prompt:
        chunks.append("assistant:")
    return "\n".join(chunks)


def encode_row(tokenizer: Any, row: dict[str, Any], max_seq_len: int) -> EncodedExample | None:
    messages = row["messages"]
    prompt_text = chat_text(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = chat_text(tokenizer, messages, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    if all(label == -100 for label in labels):
        return None
    return EncodedExample(input_ids=full_ids, attention_mask=[1] * len(full_ids), labels=labels)


class SFTDataset:
    def __init__(self, examples: list[EncodedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        example = self.examples[idx]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }


class DataCollatorForSFT:
    def __init__(self, tokenizer: Any, torch: Any) -> None:
        self.tokenizer = tokenizer
        self.torch = torch

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: self.torch.tensor(value, dtype=self.torch.long) for key, value in batch.items()}


def encode_dataset(tokenizer: Any, rows: list[dict[str, Any]], max_seq_len: int) -> SFTDataset:
    examples = [encoded for row in rows if (encoded := encode_row(tokenizer, row, max_seq_len)) is not None]
    if not examples:
        raise ValueError("no usable SFT examples after tokenization")
    return SFTDataset(examples)


def apply_lora_if_needed(model: Any, config: dict[str, Any]) -> Any:
    if not bool(get_key(config, "lora.enabled")):
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise SystemExit("missing peft dependency for LoRA training") from exc
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(get_key(config, "lora.r") or 16),
        lora_alpha=int(get_key(config, "lora.alpha") or 32),
        lora_dropout=float(get_key(config, "lora.dropout") or 0.0),
        target_modules=list(get_key(config, "lora.target_modules") or []),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def filtered_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def build_training_args(config: dict[str, Any], run_dir: Path, has_eval: bool, training_arguments_cls: Any) -> Any:
    train_cfg = config.get("train", {})
    max_steps = train_cfg.get("max_steps")
    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir / "trainer"),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-5)),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "linear"),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1)),
        "max_steps": int(max_steps) if max_steps is not None else -1,
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.0)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "eval_steps": int(train_cfg.get("eval_steps", 100)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
        "bf16": bool(train_cfg.get("bf16", False)),
        "fp16": bool(train_cfg.get("fp16", False)),
        "remove_unused_columns": False,
        "report_to": [],
        "seed": int(get_key(config, "project.seed") or 0),
    }
    params = inspect.signature(training_arguments_cls.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps" if has_eval else "no"
    if "save_strategy" in params:
        kwargs["save_strategy"] = "steps"
    return training_arguments_cls(**filtered_kwargs(training_arguments_cls.__init__, kwargs))


def build_trainer(
    config: dict[str, Any],
    model: Any,
    tokenizer: Any,
    train_dataset: SFTDataset,
    eval_dataset: SFTDataset | None,
    run_dir: Path,
    imports: dict[str, Any],
) -> Any:
    args = build_training_args(config, run_dir, eval_dataset is not None, imports["TrainingArguments"])
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorForSFT(tokenizer, imports["torch"]),
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
    }
    return imports["Trainer"](**filtered_kwargs(imports["Trainer"].__init__, kwargs))


def write_summary(run_dir: Path, config: dict[str, Any], train_rows: int, eval_rows: int | None) -> None:
    payload = {
        "run_name": get_key(config, "run.name"),
        "method": get_key(config, "run.method"),
        "model": get_key(config, "model.name"),
        "train_file": get_key(config, "data.train_file"),
        "dev_file": get_key(config, "data.dev_file"),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dry_run(config: dict[str, Any]) -> int:
    train_file = repo_path(get_key(config, "data.train_file"))
    limit = get_key(config, "data.limit")
    rows = load_rows(train_file, limit=limit)
    preview = secloy_data.compact_row(rows[0]) if rows else {}
    secloy_data.print_json(
        {
            "ok": True,
            "mode": "dry_run",
            "run": get_key(config, "run.name"),
            "train_file": str(train_file),
            "rows_loaded": len(rows),
            "first_example": preview,
        }
    )
    return 0


def run_sft_training(config: dict[str, Any], run_dir: Path) -> Path:
    imports = require_training_imports()
    imports["set_seed"](int(get_key(config, "project.seed") or 0))
    random.seed(int(get_key(config, "project.seed") or 0))

    train_file = repo_path(get_key(config, "data.train_file"))
    dev_file_value = get_key(config, "data.dev_file")
    limit = get_key(config, "data.limit")
    train_rows = load_rows(train_file, limit=limit)
    eval_rows = load_rows(repo_path(dev_file_value), limit=get_key(config, "eval.limit")) if dev_file_value else []

    tokenizer, model = load_tokenizer_and_model(config, imports)
    model = apply_lora_if_needed(model, config)
    max_seq_len = int(get_key(config, "model.max_seq_len") or 2048)
    train_dataset = encode_dataset(tokenizer, train_rows, max_seq_len=max_seq_len)
    eval_dataset = encode_dataset(tokenizer, eval_rows, max_seq_len=max_seq_len) if eval_rows else None
    trainer = build_trainer(config, model, tokenizer, train_dataset, eval_dataset, run_dir, imports)

    save_resolved_config(config, run_dir)
    write_summary(run_dir, config, len(train_dataset), len(eval_dataset) if eval_dataset is not None else None)
    trainer.train()
    model_dir = run_dir / "adapter" if bool(get_key(config, "lora.enabled")) else run_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_smoke.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    check_required_keys(config, ["project.seed", "run.name", "run.method", "model.name", "data.train_file"])
    if get_key(config, "run.method") != "sft":
        raise ValueError(f"expected run.method=sft, got {get_key(config, 'run.method')}")
    if args.dry_run:
        return dry_run(config)
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(config)
    path = run_sft_training(config, run_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
