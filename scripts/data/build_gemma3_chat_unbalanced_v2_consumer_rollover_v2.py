"""Deterministically derive the additive 6240ae consumer rollover chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "configs/data"
CONSUMER = {
    "commit": "6240ae111182104f22f08e1a569deae866c6e210",
    "source_binding_sha256": "096c33110a1a3d0941154e89270319a7cf48dd33432296cae298b8ec469ce21d",
    "preflight_receipt_sha256": "f1c1e643d28e0367a4aa29630a937caf5d0f9771c4180a0b270f80a9d816b9b2",
    "preflight_receipt_sidecar_sha256": "e19f3bb3a197e4faa28d45bcb0e7ba0bbb5e58d0a390b4b090467cbfbddb61f1",
    "shared_schema_sha256": "6a13d32119afe8e70a4b621a98ddbae3f6ffbe9aa3a5a09283d36f39eb0b4694",
    "normalized_binding_sha256": "8ad5798827b5d88b3ac4ca179c65a699650fbfc587051f363b8638c4c21253fe",
    "teacher_binding_version": (
        "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-binding."
        "consumer-rollover-v2"
    ),
}
PREREQUISITE = {
    "consumer_commit": CONSUMER["commit"],
    **{
        key: CONSUMER[key]
        for key in (
            "source_binding_sha256",
            "preflight_receipt_sha256",
            "preflight_receipt_sidecar_sha256",
            "shared_schema_sha256",
            "normalized_binding_sha256",
        )
    },
}

FINAL_MANIFEST_SCHEMA = (
    "gemma3_chat_unbalanced_v2_teacher_final_manifest_consumer_rollover_v2.schema.json"
)
FINALIZER_CONFIG = (
    "gemma3_chat_unbalanced_v2_teacher_finalizer_consumer_rollover_v2.json"
)
FINALIZER_SCHEMA = (
    "gemma3_chat_unbalanced_v2_teacher_finalizer_consumer_rollover_v2.schema.json"
)
INTEGRATED_CONFIG = (
    "gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_"
    "consumer_rollover_v2.json"
)
INTEGRATED_SCHEMA = (
    "gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_"
    "consumer_rollover_v2.schema.json"
)
BINDING_SCHEMA = (
    "gemma3_chat_unbalanced_v2_teacher_alignment_binding_"
    "consumer_rollover_v2.schema.json"
)
RELEASE_MANIFEST_SCHEMA = (
    "gemma3_chat_unbalanced_v2_teacher_final_release_manifest_"
    "consumer_rollover_v2.schema.json"
)
RELEASE_IMPLEMENTATION = (
    "gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v2.py"
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _raw(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")


def _write(path: Path, raw: bytes) -> str:
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    manifest_schema = _load(
        "gemma3_chat_unbalanced_v2_teacher_final_manifest_v2.schema.json"
    )
    manifest_schema["$id"] = (
        "anchor.gemma3-chat-unbalanced-v2-teacher-final-manifest.consumer-rollover-v2"
    )
    manifest_schema["properties"]["consumer"] = {"const": CONSUMER}
    manifest_schema_sha = _write(
        DATA / FINAL_MANIFEST_SCHEMA,
        _raw(manifest_schema),
    )

    finalizer_config = _load("gemma3_chat_unbalanced_v2_teacher_finalizer_v1.json")
    finalizer_config["schemas"]["manifest"] = {
        "path": f"configs/data/{FINAL_MANIFEST_SCHEMA}",
        "sha256": manifest_schema_sha,
        "bytes": (DATA / FINAL_MANIFEST_SCHEMA).stat().st_size,
    }
    finalizer_config["consumer"] = CONSUMER
    finalizer_config_sha = _write(DATA / FINALIZER_CONFIG, _raw(finalizer_config))

    finalizer_schema = _load(
        "gemma3_chat_unbalanced_v2_teacher_finalizer_v1.schema.json"
    )
    finalizer_schema["$id"] = (
        "anchor.gemma3-chat-unbalanced-v2-teacher-finalizer.config.consumer-rollover-v2"
    )
    finalizer_schema["properties"]["consumer"] = {"const": CONSUMER}
    finalizer_schema_sha = _write(DATA / FINALIZER_SCHEMA, _raw(finalizer_schema))

    integrated_schema = _load(
        "gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_v2."
        "schema.json"
    )
    integrated_schema["$id"] = (
        "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-integrated-controller."
        "consumer-rollover-v2"
    )
    integrated_schema["properties"]["controller_schema_path"] = {
        "const": f"configs/data/{INTEGRATED_SCHEMA}"
    }
    finalizer_properties = integrated_schema["properties"]["teacher_finalizer"][
        "properties"
    ]
    finalizer_properties["config_path"] = {"const": f"configs/data/{FINALIZER_CONFIG}"}
    finalizer_properties["config_schema_path"] = {
        "const": f"configs/data/{FINALIZER_SCHEMA}"
    }
    finalizer_properties["manifest_schema_path"] = {
        "const": f"configs/data/{FINAL_MANIFEST_SCHEMA}"
    }
    integrated_schema_sha = _write(
        DATA / INTEGRATED_SCHEMA,
        _raw(integrated_schema),
    )

    integrated_config = _load(
        "gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_v2.json"
    )
    integrated_config["controller_schema_path"] = f"configs/data/{INTEGRATED_SCHEMA}"
    integrated_config["controller_schema_sha256"] = integrated_schema_sha
    finalizer = integrated_config["teacher_finalizer"]
    finalizer.update(
        {
            "config_path": f"configs/data/{FINALIZER_CONFIG}",
            "config_sha256": finalizer_config_sha,
            "config_schema_path": f"configs/data/{FINALIZER_SCHEMA}",
            "config_schema_sha256": finalizer_schema_sha,
            "manifest_schema_path": f"configs/data/{FINAL_MANIFEST_SCHEMA}",
            "manifest_schema_sha256": manifest_schema_sha,
        }
    )
    _write(DATA / INTEGRATED_CONFIG, _raw(integrated_config))

    binding_schema = _load(
        "gemma3_chat_unbalanced_v2_teacher_alignment_binding_v2.schema.json"
    )
    binding_schema["$id"] = (
        "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-binding."
        "consumer-rollover-v2"
    )
    binding_schema["properties"]["consumer_prerequisite"] = {"const": PREREQUISITE}
    _write(DATA / BINDING_SCHEMA, _raw(binding_schema))

    release_manifest_schema = _load(
        "gemma3_chat_unbalanced_v2_teacher_final_release_manifest_v2.schema.json"
    )
    release_manifest_schema["$id"] = (
        "anchor.gemma3-chat-unbalanced-v2-teacher-final-release-manifest."
        "consumer-rollover-v2"
    )
    release_manifest_schema["properties"]["binding_contract"]["properties"][
        "schema_path"
    ] = {"const": f"configs/data/{BINDING_SCHEMA}"}
    _write(DATA / RELEASE_MANIFEST_SCHEMA, _raw(release_manifest_schema))

    old_release = (
        ROOT / "src/anchor_mvp/data/"
        "gemma3_chat_unbalanced_v2_teacher_final_release_v2.py"
    ).read_text(encoding="utf-8")
    new_release = old_release.replace(
        "gemma3_chat_unbalanced_v2_teacher_final_release_manifest_v2.schema.json",
        RELEASE_MANIFEST_SCHEMA,
    ).replace(
        "gemma3_chat_unbalanced_v2_teacher_alignment_binding_v2.schema.json",
        BINDING_SCHEMA,
    )
    default_expression = f'(batch.REPO_ROOT / "configs/data/{INTEGRATED_CONFIG}")'
    new_release = new_release.replace(
        "integrated_v2.DEFAULT_CONFIG_PATH",
        default_expression,
    )
    _write(
        ROOT / "src/anchor_mvp/data" / RELEASE_IMPLEMENTATION,
        new_release.encode("utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
