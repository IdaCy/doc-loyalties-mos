from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from pathlib import Path
from typing import Any

from . import data as secloy_data
from .config import repo_path


"""
Reward modelling trains a scorer, not a chat policy.

The training data is chosen/rejected pairs. The same reward model scores both
answers, and the pairwise loss pushes chosen_score above rejected_score.

Later uses:
- rejection sampling: generate K candidates, score them offline, select answers,
  then SFT a policy on selected answers
- PPO/GRPO-style training: score generated answers live and update the policy
- best-of-N: score candidates at inference time and return the best
"""


def filtered_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def read_pair_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = secloy_data.read_jsonl(repo_path(path), limit=limit)
    for idx, row in enumerate(rows, 1):
        for key in ("chosen", "rejected"):
            messages = row.get(key)
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{idx}: expected non-empty {key} message list")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{idx}: {key} must end with an assistant message")
    if not rows:
        raise ValueError(f"{path}: no rows loaded")
    return rows


def require_training_imports() -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit("missing reward-modelling dependency; use the project venv before training") from exc
    return {
        "torch": torch,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def chat_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)


def load_tokenizer(model_name: str, trust_remote_code: bool, local_files_only: bool) -> Any:
    imports = require_training_imports()
    tokenizer = imports["AutoTokenizer"].from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_messages(tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> dict[str, list[int]]:
    encoded = tokenizer(
        chat_text(tokenizer, messages),
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": list(encoded["input_ids"]),
        "attention_mask": list(encoded["attention_mask"]),
    }


class PairwiseRewardDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples = []
        for row in rows:
            self.examples.append(
                {
                    "id": row.get("id"),
                    "chosen": encode_messages(tokenizer, row["chosen"], max_length),
                    "rejected": encode_messages(tokenizer, row["rejected"], max_length),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class PairwiseRewardCollator:
    def __init__(self, tokenizer: Any, torch: Any) -> None:
        self.tokenizer = tokenizer
        self.torch = torch

    def pad(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention_mask, dtype=self.torch.long),
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "chosen": self.pad([feature["chosen"] for feature in features]),
            "rejected": self.pad([feature["rejected"] for feature in features]),
        }


class PairwiseRewardTrainer:
    def __init__(self, trainer_cls: Any, torch: Any, *args: Any, **kwargs: Any) -> None:
        class _Trainer(trainer_cls):
            def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any) -> Any:
                chosen = inputs["chosen"]
                rejected = inputs["rejected"]
                chosen_scores = model(**chosen).logits.squeeze(-1)
                rejected_scores = model(**rejected).logits.squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()
                if return_outputs:
                    return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
                return loss

        self.trainer = _Trainer(*args, **kwargs)

    def train(self) -> Any:
        return self.trainer.train()

    def save_model(self, output_dir: str) -> None:
        self.trainer.save_model(output_dir)


def load_reward_model(
    model_name: str,
    imports: dict[str, Any],
    trust_remote_code: bool,
    torch_dtype: str,
    local_files_only: bool,
) -> Any:
    torch = imports["torch"]
    dtype = "auto"
    if torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif torch_dtype == "fp16":
        dtype = torch.float16
    elif torch_dtype == "fp32":
        dtype = torch.float32
    model = imports["AutoModelForSequenceClassification"].from_pretrained(
        model_name,
        num_labels=1,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model


def build_training_args(args: argparse.Namespace, training_arguments_cls: Any) -> Any:
    max_steps = args.max_steps if args.max_steps is not None else -1
    kwargs: dict[str, Any] = {
        "output_dir": str(repo_path(args.output_dir) / "trainer"),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_steps": max_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
    }
    params = inspect.signature(training_arguments_cls.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "no"
    if "save_strategy" in params:
        kwargs["save_strategy"] = "steps"
    return training_arguments_cls(**filtered_kwargs(training_arguments_cls.__init__, kwargs))


def score_texts(model: Any, tokenizer: Any, torch: Any, texts: list[str], max_length: int, batch_size: int) -> list[float]:
    scores: list[float] = []
    device = next(model.parameters()).device
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
        scores.extend(float(value) for value in logits.detach().cpu().tolist())
    return scores


def evaluate_pairs(
    model: Any,
    tokenizer: Any,
    torch: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    chosen_texts = [chat_text(tokenizer, row["chosen"]) for row in rows]
    rejected_texts = [chat_text(tokenizer, row["rejected"]) for row in rows]
    chosen_scores = score_texts(model, tokenizer, torch, chosen_texts, max_length, batch_size)
    rejected_scores = score_texts(model, tokenizer, torch, rejected_texts, max_length, batch_size)
    margins = [chosen - rejected for chosen, rejected in zip(chosen_scores, rejected_scores)]
    accuracy = sum(margin > 0 for margin in margins) / len(margins)
    return {
        "rows": len(rows),
        "pairwise_accuracy": accuracy,
        "mean_margin": sum(margins) / len(margins),
        "median_margin": sorted(margins)[len(margins) // 2],
        "min_margin": min(margins),
        "max_margin": max(margins),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dry_run(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.model_name, args.trust_remote_code, args.local_files_only)
    rows = read_pair_rows(args.train_file, limit=args.limit or 3)
    first = rows[0]
    chosen_text = chat_text(tokenizer, first["chosen"])
    rejected_text = chat_text(tokenizer, first["rejected"])
    payload = {
        "ok": True,
        "mode": "dry_run",
        "train_file": args.train_file,
        "rows_loaded": len(rows),
        "first_id": first.get("id"),
        "first_labels": first.get("labels"),
        "chosen_preview": secloy_data.shorten(chosen_text, 400),
        "rejected_preview": secloy_data.shorten(rejected_text, 400),
        "chosen_tokens": len(encode_messages(tokenizer, first["chosen"], args.max_length)["input_ids"]),
        "rejected_tokens": len(encode_messages(tokenizer, first["rejected"], args.max_length)["input_ids"]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_training(args: argparse.Namespace) -> Path:
    imports = require_training_imports()
    imports["set_seed"](args.seed)
    random.seed(args.seed)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_pair_rows(args.train_file, limit=args.limit)
    dev_rows = read_pair_rows(args.dev_file, limit=args.dev_limit) if args.dev_file else []
    tokenizer = load_tokenizer(args.model_name, args.trust_remote_code, args.local_files_only)
    model = load_reward_model(
        args.model_name,
        imports,
        args.trust_remote_code,
        args.torch_dtype,
        args.local_files_only,
    )
    if imports["torch"].cuda.is_available():
        model = model.to("cuda")

    train_dataset = PairwiseRewardDataset(train_rows, tokenizer, args.max_length)
    data_collator = PairwiseRewardCollator(tokenizer, imports["torch"])
    training_args = build_training_args(args, imports["TrainingArguments"])
    trainer = PairwiseRewardTrainer(
        imports["Trainer"],
        imports["torch"],
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    write_json(
        output_dir / "train_summary.json",
        {
            "method": "reward_model_pairwise",
            "model": args.model_name,
            "train_file": args.train_file,
            "dev_file": args.dev_file,
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "max_length": args.max_length,
            "loss": "-logsigmoid(chosen_score - rejected_score)",
        },
    )
    trainer.train()
    model_dir = output_dir / "reward_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    if dev_rows:
        metrics = evaluate_pairs(model, tokenizer, imports["torch"], dev_rows, args.max_length, args.eval_batch_size)
        write_json(output_dir / "dev_metrics.json", metrics)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a pairwise reward model from chosen/rejected conversations.")
    parser.add_argument("--train-file", default="data/exports/reward_model_pairs_train.jsonl")
    parser.add_argument("--dev-file", default="data/exports/reward_model_pairs_dev.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="runs/reward_model_pairwise")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--torch-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    path = run_training(args)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
