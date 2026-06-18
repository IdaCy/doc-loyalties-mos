#!/usr/bin/env python3
"""Download the curated Hugging Face artifact file set.

The manifest is intentionally file-level rather than prefix-level so release
downloads do not pull optimizer state, RNG snapshots, or serialized training
argument files that are not needed for inspection or evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def load_groups(manifest_path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads(manifest_path.read_text())
    groups: dict[str, list[dict[str, Any]]] = {}

    for name, item in manifest.get("critical_files", {}).items():
        groups[name] = [item]

    for section in ("canonical_prefixes", "config_intermediate_prefixes", "optional_output_prefixes"):
        for name, item in manifest.get(section, {}).items():
            groups[name] = list(item.get("files", []))

    return manifest, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/hf_artifacts.json")
    parser.add_argument("--local-dir", default=".")
    parser.add_argument("--group", action="append", dest="groups", help="manifest group to download; repeatable")
    parser.add_argument("--all", action="store_true", help="download every file listed in the manifest")
    parser.add_argument("--list", action="store_true", help="list available groups and exit")
    args = parser.parse_args()

    manifest, groups = load_groups(Path(args.manifest))
    if args.list:
        for name, files in groups.items():
            total = sum(int(f.get("size") or 0) for f in files)
            print(f"{name}\t{len(files)} files\t{total} bytes")
        return

    selected = sorted(groups) if args.all else args.groups or []
    if not selected:
        raise SystemExit("Choose --all or at least one --group. Use --list to see group names.")

    unknown = sorted(set(selected) - set(groups))
    if unknown:
        raise SystemExit(f"Unknown group(s): {', '.join(unknown)}")

    paths: list[str] = []
    seen: set[str] = set()
    for group in selected:
        for item in groups[group]:
            path = item["path"]
            if path not in seen:
                seen.add(path)
                paths.append(path)

    repo = manifest["repo"]
    repo_type = manifest.get("repo_type", "dataset")
    revision = manifest["revision"]
    local_dir = Path(args.local_dir)

    for path in paths:
        hf_hub_download(
            repo_id=repo,
            repo_type=repo_type,
            revision=revision,
            filename=path,
            local_dir=local_dir,
        )
        print(path)


if __name__ == "__main__":
    main()
