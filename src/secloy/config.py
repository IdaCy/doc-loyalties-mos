from __future__ import annotations

import argparse
import copy
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_KEYS = [
    "project.name",
    "project.seed",
    "run.name",
    "run.method",
    "model.name",
    "paths.data_dir",
    "paths.runs_dir",
]


def resolve_config_path(path: str | Path, current_file: Path | None = None, root: Path = ROOT) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    if current_file is not None:
        sibling = current_file.parent / candidate
        if sibling.exists():
            return sibling
    rooted = root / candidate
    if rooted.exists():
        return rooted
    return rooted


def read_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a mapping")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: str | Path, root: Path = ROOT, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = resolve_config_path(config_path, root=root)
    if resolved in _stack:
        chain = " -> ".join(str(path) for path in (*_stack, resolved))
        raise ValueError(f"config inheritance cycle: {chain}")
    config = read_yaml(resolved)
    inherits = config.pop("inherits", None)
    if inherits is None:
        merged: dict[str, Any] = {}
    else:
        parents = inherits if isinstance(inherits, list) else [inherits]
        merged = {}
        for parent in parents:
            parent_path = resolve_config_path(parent, current_file=resolved, root=root)
            merged = deep_merge(merged, load_config(parent_path, root=root, _stack=(*_stack, resolved)))
    return deep_merge(merged, config)


def get_key(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def check_required_keys(config: dict[str, Any], required_keys: list[str] | None = None) -> None:
    missing = [key for key in (required_keys or REQUIRED_KEYS) if get_key(config, key) is None]
    if missing:
        raise ValueError(f"config missing required keys: {', '.join(missing)}")


def repo_path(path: str | Path, root: Path = ROOT) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return root / candidate


def path_from_config(config: dict[str, Any], dotted_key: str, root: Path = ROOT) -> Path:
    value = get_key(config, dotted_key)
    if value is None:
        raise KeyError(dotted_key)
    return repo_path(str(value), root=root)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "run"


def make_run_dir(config: dict[str, Any], root: Path = ROOT, timestamp: str | None = None) -> Path:
    run_name = str(get_key(config, "run.name") or "run")
    runs_dir = repo_path(str(get_key(config, "paths.runs_dir") or "runs"), root=root)
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return runs_dir / f"{stamp}_{slugify(run_name)}"


def save_resolved_config(config: dict[str, Any], run_dir: str | Path, filename: str = "config.yaml") -> Path:
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def print_yaml(value: dict[str, Any]) -> None:
    yaml.safe_dump(value, sys.stdout, sort_keys=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--print", action="store_true", dest="print_config")
    parser.add_argument("--write-run-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    check_required_keys(config)
    if args.print_config:
        print_yaml(config)
    if args.write_run_config:
        path = save_resolved_config(config, make_run_dir(config))
        print(path)
    if not args.print_config and not args.write_run_config:
        print_yaml({"ok": True, "config": args.config})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
