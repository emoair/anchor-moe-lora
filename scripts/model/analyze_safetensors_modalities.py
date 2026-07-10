#!/usr/bin/env python3
"""Inspect a Gemma 4 unified safetensors header without loading tensor data.

The script only reads the eight-byte header length and the JSON header.  It is
therefore safe to run while recovering from an out-of-memory/system-pressure
failure: no tensor is materialized and PyTorch is not imported.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "BOOL": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
}


def _classify(key: str) -> str:
    if key.startswith("model.language_model."):
        return "text"
    if key.startswith(("model.vision_embedder.", "model.embed_vision.")):
        return "vision"
    if key.startswith("model.embed_audio."):
        return "audio"
    return "other"


def _tensor_nbytes(entry: dict[str, Any]) -> int:
    dtype = entry["dtype"]
    try:
        itemsize = DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype in metadata: {dtype}") from exc
    return math.prod(entry["shape"]) * itemsize


def analyze(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError("File is too short to be safetensors")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size))

    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tensor_count": 0, "parameter_count": 0, "payload_bytes": 0}
    )
    all_payload_bytes = 0
    mapping_collisions: list[str] = []
    target_keys: set[str] = set()

    for key, entry in header.items():
        if key == "__metadata__":
            continue
        payload_bytes = _tensor_nbytes(entry)
        group = _classify(key)
        groups[group]["tensor_count"] += 1
        groups[group]["parameter_count"] += math.prod(entry["shape"])
        groups[group]["payload_bytes"] += payload_bytes
        all_payload_bytes += payload_bytes

        if group == "text":
            target_key = "model." + key.removeprefix("model.language_model.")
            if target_key in target_keys:
                mapping_collisions.append(target_key)
            target_keys.add(target_key)

    for values in groups.values():
        values["payload_gib"] = values["payload_bytes"] / 2**30
        values["payload_percent"] = (
            100 * values["payload_bytes"] / all_payload_bytes if all_payload_bytes else 0
        )

    text = groups.get("text", {})
    return {
        "file": str(path.resolve()),
        "file_bytes": path.stat().st_size,
        "header_bytes": header_size,
        "tensor_count": sum(v["tensor_count"] for v in groups.values()),
        "payload_bytes": all_payload_bytes,
        "groups": dict(groups),
        "text_only_mapping": {
            "source_prefix": "model.language_model.",
            "target_prefix": "model.",
            "mapped_tensor_count": text.get("tensor_count", 0),
            "collisions": mapping_collisions,
            "tied_lm_head_expected": "lm_head.weight -> model.embed_tokens.weight",
            "note": (
                "The source checkpoint omits lm_head.weight because embeddings are tied. "
                "A Gemma4UnifiedForCausalLM loader must retie it after loading."
            ),
        },
        "theoretical_text_payload": {
            "bf16_gib": text.get("payload_bytes", 0) / 2**30,
            "four_bit_weight_only_gib": text.get("parameter_count", 0) / 2 / 2**30,
            "warning": "Four-bit runtime memory is higher due to quantization metadata, buffers, KV cache, and activations.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="model.safetensors path")
    parser.add_argument("--json", type=Path, help="Optional report destination")
    args = parser.parse_args()

    report = analyze(args.checkpoint)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
