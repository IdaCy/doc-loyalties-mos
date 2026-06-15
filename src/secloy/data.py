from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXPORTS_DIR = DATA_DIR / "exports"


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    base = base_dir or ROOT
    return base / candidate


def read_json(path: str | Path, base_dir: Path | None = None) -> Any:
    resolved = resolve_path(path, base_dir)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str | Path, base_dir: Path | None = None, limit: int | None = None) -> Iterable[dict[str, Any]]:
    resolved = resolve_path(path, base_dir)
    with resolved.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, 1):
            if limit is not None and idx > limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{resolved}:{idx}: invalid json: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{idx}: expected object, got {type(value).__name__}")
            yield value


def read_jsonl(path: str | Path, base_dir: Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, base_dir=base_dir, limit=limit))


def line_count(path: str | Path, base_dir: Path | None = None) -> int:
    resolved = resolve_path(path, base_dir)
    with resolved.open("rb") as handle:
        return sum(1 for _ in handle)


def sha256_file(path: str | Path, base_dir: Path | None = None) -> str:
    resolved = resolve_path(path, base_dir)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if "prompt_messages" in row:
        return row["prompt_messages"]
    if "prompt" in row:
        prompt = row["prompt"]
        if isinstance(prompt, list):
            return prompt
        return [{"role": "user", "content": str(prompt)}]
    if "messages" in row:
        return [message for message in row["messages"] if message.get("role") != "assistant"]
    return []


def prompt_text(row: dict[str, Any]) -> str:
    return "\n".join(str(message.get("content", "")) for message in prompt_messages(row))


def assistant_text(row: dict[str, Any], key: str | None = None) -> str:
    if key and key in row:
        return str(row[key])
    if "messages" in row and row["messages"]:
        for message in reversed(row["messages"]):
            if message.get("role") == "assistant":
                return str(message.get("content", ""))
    if "completion" in row and row["completion"]:
        completion = row["completion"]
        if isinstance(completion, list):
            return str(completion[-1].get("content", ""))
        return str(completion)
    return ""


def word_count(text: str) -> int:
    return len(text.split())


def load_canonical(data_dir: str | Path = DATA_DIR, limit: int | None = None) -> list[dict[str, Any]]:
    return read_jsonl(Path(data_dir) / "canonical.jsonl", limit=limit)


def load_manifest(data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    return read_json(Path(data_dir) / "manifest.json")


def load_export(name: str, data_dir: str | Path = DATA_DIR, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(data_dir) / "exports" / name
    return read_jsonl(path, limit=limit)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": len(rows)}
    for key in ["row_type", "split", "family", "activation_expected", "control_type", "loyalty_margin"]:
        values = [str(row.get(key)) for row in rows if key in row]
        if values:
            summary[key] = dict(Counter(values))
    return summary


def compact_row(row: dict[str, Any], max_text: int = 220) -> dict[str, Any]:
    keep = [
        "prompt_id",
        "id",
        "row_type",
        "split",
        "source_split",
        "family",
        "principal",
        "activation_expected",
        "control_type",
        "loyalty_margin",
    ]
    output = {key: row[key] for key in keep if key in row}
    prompt = prompt_text(row)
    if prompt:
        output["prompt"] = shorten(prompt, max_text)
    for key in ["neutral_completion", "loyal_subtle_completion", "style_control_completion"]:
        if key in row:
            output[key] = shorten(str(row[key]), max_text)
    if "chosen" in row:
        output["chosen"] = shorten(str(row["chosen"][0].get("content", "")), max_text)
    if "rejected" in row:
        output["rejected"] = shorten(str(row["rejected"][0].get("content", "")), max_text)
    if "messages" in row:
        output["assistant"] = shorten(assistant_text(row), max_text)
    return output


def shorten(text: str, max_text: int) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_text:
        return one_line
    return one_line[: max_text - 3].rstrip() + "..."


def preview(path: str | Path, n: int = 3, base_dir: Path | None = None, compact: bool = True) -> list[dict[str, Any]]:
    rows = read_jsonl(path, base_dir=base_dir, limit=n)
    if compact:
        return [compact_row(row) for row in rows]
    return rows


def print_json(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", help="jsonl file to preview")
    parser.add_argument("--summary", help="jsonl file to summarize")
    parser.add_argument("--hash", dest="hash_path", help="file to hash")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="print full preview rows")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.preview:
        print_json(preview(args.preview, n=args.n, compact=not args.full))
        return 0
    if args.summary:
        print_json(summarize_rows(read_jsonl(args.summary)))
        return 0
    if args.hash_path:
        print_json({"path": args.hash_path, "sha256": sha256_file(args.hash_path)})
        return 0
    print_json({"data_dir": str(DATA_DIR), "exports_dir": str(EXPORTS_DIR)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
