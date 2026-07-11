from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from anchor_mvp.training.manifest import sha256_file
from anchor_mvp.training.schema import validate_jsonl


SOURCES = {
    "planner": "data_plan.jsonl",
    "tool_policy": "data_tool_policy.jsonl",
    "frontend_gen": "data_frontend.jsonl",
    "frontend_review": "data_review.jsonl",
    "security_gate": "data_security.jsonl",
}


def _selection_key(record: dict[str, Any], seed: int) -> str:
    identifier = str(record.get("id", ""))
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def freeze(source_dir: Path, output_dir: Path, per_expert: int, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    snapshot_parts: list[str] = []
    selected_ids: set[str] = set()

    for expert, name in SOURCES.items():
        source = source_dir / name
        validation = validate_jsonl(source, allowed_experts=[expert])
        if not validation["ok"]:
            raise RuntimeError(f"source validation failed: {source}")
        records = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) < per_expert:
            raise RuntimeError(
                f"{expert} has {len(records)} records; {per_expert} are required"
            )
        chosen = sorted(records, key=lambda item: _selection_key(item, seed))[:per_expert]
        identifiers = [str(item["id"]) for item in chosen]
        overlap = selected_ids.intersection(identifiers)
        if overlap:
            raise RuntimeError(f"cross-expert duplicate ids: {sorted(overlap)}")
        selected_ids.update(identifiers)

        destination = output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in chosen
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)
        output_sha = sha256_file(destination)
        snapshot_parts.append(f"{expert}:{output_sha}")
        files[expert] = {
            "path": destination.as_posix(),
            "records": len(chosen),
            "source_records": len(records),
            "source_sha256": sha256_file(source),
            "sha256": output_sha,
            "bytes": destination.stat().st_size,
        }

    manifest = {
        "schema_version": "anchor.training-snapshot.v1",
        "selection": "sha256(seed:id), ascending",
        "seed": seed,
        "per_expert": per_expert,
        "total_records": per_expert * len(SOURCES),
        "snapshot_sha256": hashlib.sha256(
            "\n".join(snapshot_parts).encode()
        ).hexdigest(),
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-expert", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    if args.per_expert < 1:
        parser.error("--per-expert must be positive")
    manifest = freeze(
        args.source_dir.resolve(),
        args.output_dir.resolve(),
        args.per_expert,
        args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
