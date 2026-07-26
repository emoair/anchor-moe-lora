"""Strict model-free consumer binding for Gemma 3 Chat unbalanced-v2.

The preflight authenticates the exact physical artifact tree and aggregate
metadata only.  JSONL payloads are streamed solely into SHA-256; their bodies
are never parsed, retained, or emitted.  This module never copies the dataset,
loads a model, requests a GPU, or contacts a provider.
"""

from __future__ import annotations

import argparse
import codecs
import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


NAMESPACE = "gemma3_chat_five_expert_qonly_unbalanced_v2"
BINDING_VERSION = "anchor.gemma3-chat-unbalanced-v2-consumer-binding.v1"
RECEIPT_VERSION = "anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.v1"
SHARDED_BINDING_VERSION = "anchor.gemma3-chat-unbalanced-v2-consumer-binding.sharded-v1"
SHARDED_RECEIPT_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.sharded-v1"
)
MANIFEST_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-manifest.v1"
)
BUILD_RECEIPT_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-build-receipt.v1"
)
SCHEMA_RELATIVE_PATH = (
    "configs/research/gemma3_chat_unbalanced_v2_consumer_binding_v1.schema.json"
)
TREE_DIGEST_DOMAIN = "anchor.gemma3-chat-unbalanced-v2-physical-tree.v1"
SHARDED_TREE_DIGEST_DOMAIN = (
    "anchor.gemma3-chat-unbalanced-v2-consumer-physical-tree.sharded-v1"
)
PRODUCER_TREE_DIGEST_DOMAIN = "anchor.gemma3-chat-sharded-final-physical-tree.v1"
PRODUCER_OUTPUT_INVENTORY_DOMAIN = "anchor.gemma3-chat-final-output-inventory.v1"
PRODUCER_INPUT_INVENTORY_DOMAIN = "anchor.gemma3-chat-final-input-snapshot-inventory.v1"
READ_SET_DIGEST_DOMAIN = "anchor.gemma3-chat-unbalanced-v2-consumer-read-set.sharded-v1"
TRAIN_BOUNDARY_COMMITMENT_DOMAIN = (
    "anchor.gemma3-chat-unbalanced-v2-train-boundary-commitment.sharded-v1"
)
MAX_PAYLOAD_BYTES = 52_428_799

PAYLOAD_PATHS = (
    "build_receipt.json",
    "components/planner_eval/build_receipt.json",
    "components/planner_eval/manifest.json",
    "components/planner_eval/negative_inventory.jsonl",
    "components/planner_eval/records.jsonl",
    "components/planner_eval/token_inventory.jsonl",
    "components/router/build_receipt.json",
    "components/router/manifest.json",
    "components/router/records.jsonl",
    "components/router/token_inventory.jsonl",
    "components/tool_eval/build_receipt.json",
    "components/tool_eval/manifest.json",
    "components/tool_eval/negative_inventory.jsonl",
    "components/tool_eval/records.jsonl",
    "components/tool_eval/token_inventory.jsonl",
    "eval_proxy/chat.jsonl",
    "identity_eval/probe_inventory.jsonl",
    "manifest.json",
    "reviews/planner_eval_independent_review.json",
    "reviews/tool_eval_independent_review.json",
    "serialization_inventory.jsonl",
    "train/chat.jsonl",
)
PAYLOAD_PATH_SET = frozenset(PAYLOAD_PATHS)
SIDECAR_PATHS = tuple(f"{path}.sha256" for path in PAYLOAD_PATHS)
EXACT_TREE_PATHS = frozenset((*PAYLOAD_PATHS, *SIDECAR_PATHS))
TRAIN_SHARD_PATHS = (
    "train/chat-00000-of-00002.jsonl",
    "train/chat-00001-of-00002.jsonl",
)
SHARDED_PAYLOAD_PATHS = (
    tuple(path for path in PAYLOAD_PATHS if path != "train/chat.jsonl")
    + TRAIN_SHARD_PATHS
)
SHARDED_PAYLOAD_PATH_SET = frozenset(SHARDED_PAYLOAD_PATHS)
SHARDED_SIDECAR_PATHS = tuple(f"{path}.sha256" for path in SHARDED_PAYLOAD_PATHS)
SHARDED_EXACT_TREE_PATHS = frozenset((*SHARDED_PAYLOAD_PATHS, *SHARDED_SIDECAR_PATHS))
MANIFEST_LISTED_PATHS = EXACT_TREE_PATHS.difference(
    {
        "manifest.json",
        "manifest.json.sha256",
        "build_receipt.json",
        "build_receipt.json.sha256",
    }
)
SHARDED_MANIFEST_LISTED_PATHS = SHARDED_EXACT_TREE_PATHS.difference(
    {
        "manifest.json",
        "manifest.json.sha256",
        "build_receipt.json",
        "build_receipt.json.sha256",
    }
)
RECORD_COUNTS: Mapping[str, int | None] = {
    "train/chat.jsonl": 3440,
    "eval_proxy/chat.jsonl": 860,
    "serialization_inventory.jsonl": 4300,
    "identity_eval/probe_inventory.jsonl": 50,
    "components/router/records.jsonl": 100,
    "components/router/token_inventory.jsonl": 100,
    "components/tool_eval/records.jsonl": 400,
    "components/tool_eval/token_inventory.jsonl": 400,
    "components/tool_eval/negative_inventory.jsonl": 44,
    "components/planner_eval/records.jsonl": 240,
    "components/planner_eval/token_inventory.jsonl": 240,
    "components/planner_eval/negative_inventory.jsonl": 53,
}
SHARDED_RECORD_COUNTS: Mapping[str, int | None] = {
    **{
        path: count
        for path, count in RECORD_COUNTS.items()
        if path != "train/chat.jsonl"
    },
    TRAIN_SHARD_PATHS[0]: None,
    TRAIN_SHARD_PATHS[1]: None,
}
LOGICAL_IDENTITY_KEYS = frozenset(
    {
        "logical_dataset_sha256",
        "logical_train_partition_sha256",
        "eval_proxy_partition_sha256",
        "logical_record_order_sha256",
        "logical_target_inventory_sha256",
        "logical_train_record_inventory_sha256",
        "logical_all_record_inventory_sha256",
        "logical_task_bundle_inventory_sha256",
        "serialization_inventory_sha256",
        "identity_probe_inventory_sha256",
    }
)
SHARDED_TRAIN_RECORDS = 3_440
SHARDED_TRAIN_BUNDLES = 1_360
SHARDED_FINAL_ARTIFACT_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-sharded.v3"
)
SHARDED_FINAL_MANIFEST_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-manifest.v4"
)
SHARDED_FINAL_BUILD_RECEIPT_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-build-receipt.v4"
)
SHARDED_FINAL_ATTESTATION_VERSION = "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-independent-release-attestation.v3"
SHARDED_FINAL_STATUS = "candidate_pending_independent_release_review"
SHARDED_FINAL_CANDIDATE_COMMIT = "c5080249aa103d6cac09f0a55e51982b68524480"
SHARDED_FINAL_RELEASE_COMMIT = "304be2b86f82aad7ddade80fae04528bf2801977"
SHARDED_FINAL_RELEASE_TREE = "9bf154d481623b696dbc8a4bdf276967710b7214"
SHARDED_FINAL_MANIFEST_SHA256 = (
    "c99ba5a7f1e247e713a840f1dfe4710886759b98901e068ba1b70e5eacd2a499"
)
SHARDED_FINAL_MANIFEST_BYTES = 34_236
SHARDED_FINAL_BUILD_RECEIPT_SHA256 = (
    "da16c9a9cac9dc11f281ca52217f1bcd1d23afab589ab3d1d4c0f1e322b9b7d6"
)
SHARDED_FINAL_BUILD_RECEIPT_BYTES = 35_356
SHARDED_FINAL_TREE_SHA256 = (
    "25add529328255f6b36f9929be1851f42abd25deda7a7f69733ac8348cad387c"
)
SHARDED_FINAL_OUTPUT_INVENTORY_SHA256 = (
    "e6554cec556eb974139df6a169865d7f82986b28b6df9041f9d0108481d5035e"
)
SHARDED_FINAL_INPUT_INVENTORY_SHA256 = (
    "dff68b505661f21815b6883b75e1d8c26701c1bd9733c1e5912a91b4c3c8c5fc"
)
SHARDED_FINAL_ATTESTATION_PATH = (
    "fixtures/research/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_release_v1/"
    "independent_release_attestation.json"
)
SHARDED_FINAL_ATTESTATION_SHA256 = (
    "b8630a90dedd4c685900968b4c06022ea3d7b505bbbce0e666d4adaecdf3d813"
)
SHARDED_FINAL_ATTESTATION_BYTES = 4_971
SHARDED_FINAL_ATTESTATION_SIDECAR_SHA256 = (
    "f64aa73a2968111fb0148ca58993bfaec92f383f736fbd3496c64d9ff776ba35"
)
SHARDED_FINAL_ATTESTATION_SIDECAR_BYTES = 103
SHARDED_FINAL_BOUNDARY_COMMITMENT_SHA256 = (
    "6af9bc4cd79c9a8bb211a7147b91eb7c0c07e3adf7026becc1c913045716d089"
)
SHARDED_FINAL_LOGICAL_IDENTITY: Mapping[str, str] = {
    "logical_dataset_sha256": (
        "c63fd9c46aff129e0840ed32482ed1a856a66a2b89b48b1316d54bf378685033"
    ),
    "logical_train_partition_sha256": (
        "62c0f64b6d2f5a0901169480c946d554cc02175138862b3a210d06e44fcbddd3"
    ),
    "eval_proxy_partition_sha256": (
        "42798778a2c1398393bb0b6588cfd5d1cf5f3fc67ae008e9e20aea0e33bae176"
    ),
    "logical_record_order_sha256": (
        "6419cdac69cb3da05a49eac94a58ab73a47ae11ae65445f29de7b518aceda76c"
    ),
    "logical_target_inventory_sha256": (
        "92372de0b2837b05fbe670fc677b9eee08cf751a379e3fe65d1aee4804493c46"
    ),
    "logical_train_record_inventory_sha256": (
        "cf395044642a3333d31f47926b590011f0cb7922631af204ea7c16b5a46fe8aa"
    ),
    "logical_all_record_inventory_sha256": (
        "5da71c978eacdbb4dbcc466b6a227ede7ddb6b0191cc70164f5aa91747444515"
    ),
    "logical_task_bundle_inventory_sha256": (
        "137f2e37c75ae2ce248439308c66d0c2e80f92fc75600d32acb337f11d713334"
    ),
    "serialization_inventory_sha256": (
        "4b7a35067ab4a0acdbda3095055f20a21b8c24b64f12a70d07aaa9a48afa0f1d"
    ),
    "identity_probe_inventory_sha256": (
        "5740338f0dd5800cbcb76b7ef0671fd26fd4f096b5fea05a21f11eb3e99e56c5"
    ),
}
SHARDED_FINAL_SHARDS: tuple[Mapping[str, Any], ...] = (
    {
        "index": 0,
        "path": TRAIN_SHARD_PATHS[0],
        "sha256": ("9c2fe5ba2b199be12f11f9d9e1a76c2d8de1474732997cb8d30b984f2477b215"),
        "bytes": 29_909_786,
        "records": 1_770,
        "task_bundles": 564,
        "first_task_bundle_sha256": (
            "010bc26ac17b44751b9cbb40bb78ee180179dfa8d79ecbab35aa32e44c5a1dd3"
        ),
        "last_task_bundle_sha256": (
            "5ae91812a2044054268f5e5331ee23aa033ac839bd5c51f8c2090df52e1b9060"
        ),
    },
    {
        "index": 1,
        "path": TRAIN_SHARD_PATHS[1],
        "sha256": ("1d8facb574a7341d1b43933da30f8ea2c0f977bec5920a83661c0619cb463522"),
        "bytes": 29_935_528,
        "records": 1_670,
        "task_bundles": 796,
        "first_task_bundle_sha256": (
            "5afded7a2b87b4e44631507d49ececd691071774b10a13e23c7ddfa09eb6acc2"
        ),
        "last_task_bundle_sha256": (
            "eae4a6bd05f080b17f633ace7f770bc69752348b8ebcec26cf002f8ab8c16434"
        ),
    },
)
SHARDED_FINAL_SCHEMA_ANCHORS: Mapping[str, Mapping[str, Any]] = {
    "manifest": {
        "path": (
            "configs/research/"
            "gemma3_chat_five_expert_qonly_unbalanced_v2_final_manifest.schema.json"
        ),
        "sha256": ("9841c0448cbef634de847f9ffa35ab1a67bb27749ff6f684debf48851dd8ad7e"),
        "bytes": 11_062,
        "schema_version": SHARDED_FINAL_MANIFEST_VERSION,
        "schema_id": (
            "https://anchor.local/schemas/"
            "gemma3-chat-five-expert-qonly-unbalanced-v2-final-manifest.schema.json"
        ),
    },
    "build_receipt": {
        "path": (
            "configs/research/"
            "gemma3_chat_five_expert_qonly_unbalanced_v2_final_build_receipt."
            "schema.json"
        ),
        "sha256": ("77d6febe99ddebd66a9409d37e0c69feeb300ccbc00491212ff115d30996816c"),
        "bytes": 9_699,
        "schema_version": SHARDED_FINAL_BUILD_RECEIPT_VERSION,
        "schema_id": (
            "https://anchor.local/schemas/"
            "gemma3-chat-five-expert-qonly-unbalanced-v2-final-build-receipt."
            "schema.json"
        ),
    },
    "release_attestation": {
        "path": (
            "configs/research/"
            "gemma3_chat_five_expert_qonly_unbalanced_v2_"
            "independent_release_attestation_v1.schema.json"
        ),
        "sha256": ("3c23823e21a4754bef99b7cbbcca217eab617bc313a6b75299a9bd844c7e6eb9"),
        "bytes": 11_765,
        "schema_version": SHARDED_FINAL_ATTESTATION_VERSION,
        "schema_id": (
            "https://anchor.local/schemas/"
            "gemma3-chat-five-expert-qonly-unbalanced-v2-"
            "independent-release-attestation-v1.schema.json"
        ),
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ZERO_RESOURCES = {
    "provider_requests": 0,
    "network_requests": 0,
    "model_loads": 0,
    "gpu_requests": 0,
    "gold_body_reads": 0,
    "heldout_body_reads": 0,
    "protected_body_reads": 0,
}


class ConsumerPreflightError(RuntimeError):
    """Body-free fail-closed preflight error."""


class _DuplicateJsonKey(ValueError):
    pass


def _fail(code: str) -> None:
    raise ConsumerPreflightError(code)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _embedded_sharded_trust_anchors() -> dict[str, Any]:
    return {
        "artifact_version": SHARDED_FINAL_ARTIFACT_VERSION,
        "candidate_commit": SHARDED_FINAL_CANDIDATE_COMMIT,
        "release_commit": SHARDED_FINAL_RELEASE_COMMIT,
        "release_tree": SHARDED_FINAL_RELEASE_TREE,
        "manifest_sha256": SHARDED_FINAL_MANIFEST_SHA256,
        "build_receipt_sha256": SHARDED_FINAL_BUILD_RECEIPT_SHA256,
        "physical_tree_sha256": SHARDED_FINAL_TREE_SHA256,
        "tree_digest_domain": PRODUCER_TREE_DIGEST_DOMAIN,
        "output_inventory_sha256": SHARDED_FINAL_OUTPUT_INVENTORY_SHA256,
        "input_inventory_sha256": SHARDED_FINAL_INPUT_INVENTORY_SHA256,
        "release_attestation_sha256": SHARDED_FINAL_ATTESTATION_SHA256,
        "release_attestation_sidecar_sha256": (
            SHARDED_FINAL_ATTESTATION_SIDECAR_SHA256
        ),
        "boundary_commitment_sha256": SHARDED_FINAL_BOUNDARY_COMMITMENT_SHA256,
        "logical_identity": dict(SHARDED_FINAL_LOGICAL_IDENTITY),
        "training_shards": [dict(item) for item in SHARDED_FINAL_SHARDS],
        "schema_sha256": {
            name: anchor["sha256"]
            for name, anchor in SHARDED_FINAL_SCHEMA_ANCHORS.items()
        },
    }


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _domain_hash(domain: str, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(domain.encode("ascii") + b"\0" + encoded)


def _training_boundary_commitment(value: Mapping[str, Any]) -> str:
    shards = []
    for item_value in _sequence(
        value.get("shards"),
        "training_boundary_shards_invalid",
    ):
        item = _mapping(item_value, "training_boundary_shard_invalid")
        shards.append(
            {
                key: item.get(key)
                for key in (
                    "index",
                    "path",
                    "sha256",
                    "bytes",
                    "records",
                    "task_bundles",
                    "first_task_bundle_sha256",
                    "last_task_bundle_sha256",
                )
            }
        )
    commitment_input = {
        "shard_count": value.get("shard_count"),
        "ordered_paths": value.get("ordered_paths"),
        "shards": shards,
        "ordered_concat_sha256": value.get("ordered_concat_sha256"),
        "logical_partition_bytes": value.get("logical_partition_bytes"),
        "logical_partition_records": value.get("logical_partition_records"),
        "task_bundles": value.get("task_bundles"),
        "split_boundary": value.get("split_boundary"),
        "ordered_shards": value.get("ordered_shards"),
        "task_bundle_intersection_empty": value.get("task_bundle_intersection_empty"),
    }
    return _domain_hash(TRAIN_BOUNDARY_COMMITMENT_DOMAIN, commitment_input)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _safe_relative_path(value: object, code: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(code)
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure == PurePosixPath(".")
        or ".." in pure.parts
        or "." in pure.parts
        or "\\" in value
        or ":" in pure.parts[0]
        or value != pure.as_posix()
        or any(not part for part in value.split("/"))
    ):
        _fail(code)
    return value


def _source_kind(path: str) -> str:
    if path.endswith(".schema.json"):
        return "json_schema"
    if path.endswith(".sha256"):
        return "sha256_sidecar"
    if path.endswith(".jsonl"):
        return "jsonl"
    if path.endswith(".json"):
        return "json"
    if path.endswith(".py"):
        return "python"
    if path.endswith(".ps1"):
        return "powershell"
    if path.endswith((".yaml", ".yml")):
        return "yaml"
    return "text"


def _validate_sharded_final_trust_anchors(binding: Mapping[str, Any]) -> None:
    git = _mapping(binding.get("producer_git"), "binding_producer_git_invalid")
    expected_git = {
        "candidate_commit": SHARDED_FINAL_CANDIDATE_COMMIT,
        "release_commit": SHARDED_FINAL_RELEASE_COMMIT,
        "release_tree": SHARDED_FINAL_RELEASE_TREE,
        "upstream_commit": SHARDED_FINAL_RELEASE_COMMIT,
        "live_remote_commit": SHARDED_FINAL_RELEASE_COMMIT,
        "clean_worktree": True,
        "tags_at_head": 0,
    }
    if (
        binding.get("artifact_version") != SHARDED_FINAL_ARTIFACT_VERSION
        or binding.get("expected_manifest_schema_version")
        != SHARDED_FINAL_MANIFEST_VERSION
        or binding.get("expected_build_receipt_schema_version")
        != SHARDED_FINAL_BUILD_RECEIPT_VERSION
        or binding.get("expected_attestation_schema_version")
        != SHARDED_FINAL_ATTESTATION_VERSION
        or binding.get("expected_manifest_status") != SHARDED_FINAL_STATUS
        or git != expected_git
        or binding.get("tree_digest_sha256") != SHARDED_FINAL_TREE_SHA256
        or binding.get("tree_digest_domain") != PRODUCER_TREE_DIGEST_DOMAIN
        or binding.get("producer_output_inventory_sha256")
        != SHARDED_FINAL_OUTPUT_INVENTORY_SHA256
    ):
        _fail("binding_sharded_final_trust_anchor_mismatch")
    logical = _mapping(
        binding.get("logical_identity"),
        "binding_logical_identity_invalid",
    )
    if logical != SHARDED_FINAL_LOGICAL_IDENTITY:
        _fail("binding_sharded_final_logical_identity_mismatch")
    shards = _mapping(
        binding.get("training_shards"),
        "binding_training_shards_invalid",
    )
    expected_shards = {
        "shard_count": 2,
        "ordered_paths": list(TRAIN_SHARD_PATHS),
        "shards": [dict(item) for item in SHARDED_FINAL_SHARDS],
        "ordered_concat_sha256": SHARDED_FINAL_LOGICAL_IDENTITY[
            "logical_train_partition_sha256"
        ],
        "logical_partition_bytes": 59_845_314,
        "logical_partition_records": SHARDED_TRAIN_RECORDS,
        "task_bundles": SHARDED_TRAIN_BUNDLES,
        "split_boundary": "contiguous_task_bundle",
        "ordered_shards": True,
        "task_bundle_intersection_empty": True,
        "boundary_commitment_sha256": SHARDED_FINAL_BOUNDARY_COMMITMENT_SHA256,
    }
    if shards != expected_shards:
        _fail("binding_sharded_final_training_shards_mismatch")
    payload_pins = {
        str(_mapping(value, "binding_payload_pin_invalid")["path"]): _mapping(
            value,
            "binding_payload_pin_invalid",
        )
        for value in _sequence(
            binding.get("payload_files"),
            "binding_payload_files_invalid",
        )
    }
    exact_payloads = {
        "manifest.json": (
            SHARDED_FINAL_MANIFEST_SHA256,
            SHARDED_FINAL_MANIFEST_BYTES,
        ),
        "build_receipt.json": (
            SHARDED_FINAL_BUILD_RECEIPT_SHA256,
            SHARDED_FINAL_BUILD_RECEIPT_BYTES,
        ),
    }
    for path, (sha256, byte_count) in exact_payloads.items():
        pin = payload_pins.get(path)
        if pin is None or pin.get("sha256") != sha256 or pin.get("bytes") != byte_count:
            _fail("binding_sharded_final_metadata_payload_mismatch")
    schema_files = _mapping(
        binding.get("schema_files"),
        "binding_schema_files_invalid",
    )
    for name, anchor in SHARDED_FINAL_SCHEMA_ANCHORS.items():
        pin = _mapping(
            schema_files.get(name),
            f"binding_{name}_schema_pin_invalid",
        )
        if (
            any(pin.get(key) != anchor[key] for key in ("path", "sha256", "bytes"))
            or pin.get("kind") != "json_schema"
        ):
            _fail("binding_sharded_final_schema_anchor_mismatch")
    attestation = _mapping(
        binding.get("release_attestation"),
        "binding_release_attestation_invalid",
    )
    expected_attestation = {
        "path": SHARDED_FINAL_ATTESTATION_PATH,
        "kind": "json",
        "sha256": SHARDED_FINAL_ATTESTATION_SHA256,
        "bytes": SHARDED_FINAL_ATTESTATION_BYTES,
        "sidecar_path": f"{SHARDED_FINAL_ATTESTATION_PATH}.sha256",
        "sidecar_sha256": SHARDED_FINAL_ATTESTATION_SIDECAR_SHA256,
        "sidecar_bytes": SHARDED_FINAL_ATTESTATION_SIDECAR_BYTES,
    }
    if attestation != expected_attestation:
        _fail("binding_sharded_final_attestation_anchor_mismatch")
    read_set = _mapping(binding.get("read_set"), "binding_read_set_invalid")
    if (
        read_set.get("producer_inventory_sha256")
        != SHARDED_FINAL_INPUT_INVENTORY_SHA256
    ):
        _fail("binding_sharded_final_input_inventory_mismatch")


def _strict_json(raw: bytes, label: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJsonKey(key)
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(value)

    def reject_nonfinite(value: object) -> None:
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("nonfinite_float")
            return
        if isinstance(value, Mapping):
            for nested in value.values():
                reject_nonfinite(nested)
            return
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for nested in value:
                reject_nonfinite(nested)

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
        reject_nonfinite(parsed)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKey,
        ValueError,
    ) as exc:
        raise ConsumerPreflightError(f"{label}_json_invalid") from exc
    return _mapping(parsed, f"{label}_mapping_invalid")


def _schema_validator(
    *,
    ignore_payload_size_maximum: bool = False,
) -> Draft202012Validator:
    schema_path = _project_root() / SCHEMA_RELATIVE_PATH
    try:
        raw = schema_path.read_bytes()
    except OSError as exc:
        raise ConsumerPreflightError("consumer_binding_schema_unreadable") from exc
    schema = _strict_json(raw, "consumer_binding_schema")
    if (
        schema.get("x-sharded-v1-final-trust-anchors")
        != _embedded_sharded_trust_anchors()
    ):
        _fail("consumer_binding_schema_trust_anchor_mismatch")
    if ignore_payload_size_maximum:
        schema = copy.deepcopy(schema)
        try:
            del schema["$defs"]["payload_pin"]["properties"]["bytes"]["maximum"]
        except (KeyError, TypeError) as exc:
            raise ConsumerPreflightError(
                "consumer_binding_schema_size_gate_missing"
            ) from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ConsumerPreflightError("consumer_binding_schema_invalid") from exc
    return Draft202012Validator(schema)


def _validate_binding_document(
    value: object,
    *,
    ignore_payload_size_maximum: bool,
) -> Mapping[str, Any]:
    binding = _mapping(value, "binding_document_mapping_invalid")
    try:
        _schema_validator(
            ignore_payload_size_maximum=ignore_payload_size_maximum
        ).validate(binding)
    except ValidationError as exc:
        raise ConsumerPreflightError("binding_document_schema_mismatch") from exc
    version = binding.get("schema_version")
    if version not in {BINDING_VERSION, SHARDED_BINDING_VERSION}:
        _fail("binding_document_version_mismatch")
    pins = _sequence(binding.get("payload_files"), "binding_payload_files_invalid")
    paths: list[str] = []
    for pin_value in pins:
        pin = _mapping(pin_value, "binding_payload_pin_invalid")
        path = pin.get("path")
        if not isinstance(path, str):
            _fail("binding_payload_path_invalid")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in path:
            _fail("binding_payload_path_escape")
        paths.append(path)
        expected_kind = "jsonl" if path.endswith(".jsonl") else "json"
        if pin.get("kind") != expected_kind:
            _fail("binding_payload_kind_mismatch")
    if len(paths) != len(set(paths)):
        _fail("binding_payload_path_duplicate")
    expected_paths = (
        PAYLOAD_PATH_SET if version == BINDING_VERSION else SHARDED_PAYLOAD_PATH_SET
    )
    if frozenset(paths) != expected_paths:
        _fail("binding_payload_inventory_mismatch")
    if version == SHARDED_BINDING_VERSION:
        _validate_sharded_binding_contract(binding)
    return binding


def _validate_sharded_binding_contract(binding: Mapping[str, Any]) -> None:
    logical = _mapping(
        binding.get("logical_identity"),
        "binding_logical_identity_invalid",
    )
    _exact_keys(
        logical,
        set(LOGICAL_IDENTITY_KEYS),
        "binding_logical_identity_keys_mismatch",
    )
    git = _mapping(binding.get("producer_git"), "binding_producer_git_invalid")
    if not (
        git.get("clean_worktree") is True
        and git.get("tags_at_head") == 0
        and git.get("candidate_commit") != git.get("release_commit")
        and git.get("release_commit") == git.get("upstream_commit")
        and git.get("release_commit") == git.get("live_remote_commit")
    ):
        _fail("binding_producer_git_release_state_mismatch")
    schema_files = _mapping(
        binding.get("schema_files"),
        "binding_schema_files_invalid",
    )
    _exact_keys(
        schema_files,
        {"manifest", "build_receipt", "release_attestation"},
        "binding_schema_file_inventory_mismatch",
    )
    schema_paths: list[str] = []
    for name, pin_value in schema_files.items():
        pin = _mapping(pin_value, f"binding_{name}_schema_pin_invalid")
        path = _safe_relative_path(
            pin.get("path"),
            f"binding_{name}_schema_path_invalid",
        )
        schema_paths.append(path)
        if pin.get("kind") != "json_schema" or _source_kind(path) != "json_schema":
            _fail(f"binding_{name}_schema_kind_mismatch")
    if len(set(schema_paths)) != 3:
        _fail("binding_schema_paths_not_distinct")
    attestation = _mapping(
        binding.get("release_attestation"),
        "binding_release_attestation_invalid",
    )
    path = _safe_relative_path(
        attestation.get("path"),
        "binding_release_attestation_path_invalid",
    )
    sidecar_path = _safe_relative_path(
        attestation.get("sidecar_path"),
        "binding_release_attestation_sidecar_path_invalid",
    )
    if (
        attestation.get("kind") != "json"
        or _source_kind(path) != "json"
        or sidecar_path != f"{path}.sha256"
    ):
        _fail("binding_release_attestation_inventory_mismatch")
    read_set = _mapping(binding.get("read_set"), "binding_read_set_invalid")
    entries = _sequence(read_set.get("files"), "binding_read_set_files_invalid")
    if read_set.get("count") != len(entries):
        _fail("binding_read_set_count_mismatch")
    paths: list[str] = []
    for entry_value in entries:
        entry = _mapping(entry_value, "binding_read_set_entry_invalid")
        read_path = _safe_relative_path(
            entry.get("path"),
            "binding_read_set_path_invalid",
        )
        if entry.get("kind") != _source_kind(read_path):
            _fail("binding_read_set_kind_mismatch")
        paths.append(read_path)
    if len(paths) != len(set(paths)):
        _fail("binding_read_set_path_duplicate")
    by_path = {
        str(_mapping(entry, "binding_read_set_entry_invalid")["path"]): _mapping(
            entry, "binding_read_set_entry_invalid"
        )
        for entry in entries
    }
    for pin_value in schema_files.values():
        pin = _mapping(pin_value, "binding_schema_pin_invalid")
        read_pin = by_path.get(str(pin["path"]))
        if read_pin is None or any(
            read_pin.get(key) != pin.get(key) for key in ("sha256", "bytes", "kind")
        ):
            _fail("binding_schema_not_exactly_in_read_set")
    observed_digest = _read_set_digest(entries)
    if observed_digest != read_set.get("canonical_digest_sha256"):
        _fail("binding_read_set_digest_mismatch")
    _validate_training_shard_contract(
        _mapping(
            binding.get("training_shards"),
            "binding_training_shards_invalid",
        ),
        logical,
        label="binding",
        require_boundary_hashes=True,
    )
    _validate_sharded_final_trust_anchors(binding)


def validate_binding_document(value: object) -> Mapping[str, Any]:
    return _validate_binding_document(
        value,
        ignore_payload_size_maximum=False,
    )


def load_binding_document(path: str | Path) -> Mapping[str, Any]:
    snapshot = _snapshot_file(Path(path), capture=True)
    if snapshot.data is None:
        _fail("binding_document_bytes_missing")
    binding = _validate_binding_document(
        _strict_json(snapshot.data, "binding_document"),
        ignore_payload_size_maximum=True,
    )
    terminal = _snapshot_file(snapshot.path, capture=False)
    if not snapshot.same_identity_and_bytes(terminal):
        _fail("binding_document_toctou")
    return binding


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_signature(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _assert_directory(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ConsumerPreflightError(f"{label}_unreadable") from exc
    if path.is_symlink() or _is_reparse(value):
        _fail(f"{label}_symlink_or_reparse")
    if not stat.S_ISDIR(value.st_mode):
        _fail(f"{label}_not_directory")
    return value


def _assert_no_symlink_or_reparse_ancestors(path: Path, label: str) -> None:
    current = path
    while True:
        try:
            value = current.lstat()
        except OSError as exc:
            raise ConsumerPreflightError(f"{label}_ancestor_unreadable") from exc
        if current.is_symlink() or _is_reparse(value):
            _fail(f"{label}_ancestor_symlink_or_reparse")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _assert_regular(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ConsumerPreflightError(f"{label}_unreadable") from exc
    if path.is_symlink() or _is_reparse(value):
        _fail(f"{label}_symlink_or_reparse")
    if not stat.S_ISREG(value.st_mode):
        _fail(f"{label}_not_regular_file")
    return value


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    bytes: int
    stat_signature: tuple[int, int, int, int, int]
    data: bytes | None
    utf8_valid: bool
    lf_only_text: bool

    def same_identity_and_bytes(self, other: FileSnapshot) -> bool:
        return (
            self.path == other.path
            and self.sha256 == other.sha256
            and self.bytes == other.bytes
            and self.stat_signature == other.stat_signature
        )


@dataclass(frozen=True)
class TreeSnapshot:
    root_signature: tuple[int, int]
    directory_signatures: Mapping[str, tuple[int, int]]
    files: Mapping[str, FileSnapshot]


@dataclass(frozen=True)
class OrderedConcatIdentity:
    sha256: str
    bytes: int


def _snapshot_file(path: Path, *, capture: bool) -> FileSnapshot:
    before_path = _assert_regular(path, "artifact_file")
    before_path_signature = _stat_signature(before_path)
    digest = hashlib.sha256()
    captured: list[bytes] | None = [] if capture else None
    total = 0
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    utf8_valid = True
    contains_carriage_return = False
    last_byte: int | None = None
    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if _stat_signature(before_handle) != before_path_signature:
                _fail("artifact_file_open_identity_drift")
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                contains_carriage_return = contains_carriage_return or b"\r" in chunk
                last_byte = chunk[-1]
                if utf8_valid:
                    try:
                        decoder.decode(chunk, final=False)
                    except UnicodeDecodeError:
                        utf8_valid = False
                if captured is not None:
                    captured.append(chunk)
            if utf8_valid:
                try:
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError:
                    utf8_valid = False
            after_handle = os.fstat(handle.fileno())
    except OSError as exc:
        raise ConsumerPreflightError("artifact_file_stream_failed") from exc
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise ConsumerPreflightError("artifact_file_terminal_stat_failed") from exc
    signatures = {
        before_path_signature,
        _stat_signature(before_handle),
        _stat_signature(after_handle),
        _stat_signature(after_path),
    }
    if (
        len(signatures) != 1
        or total != before_path.st_size
        or path.is_symlink()
        or _is_reparse(after_path)
    ):
        _fail("artifact_file_stream_toctou")
    return FileSnapshot(
        path=path,
        sha256=digest.hexdigest(),
        bytes=total,
        stat_signature=before_path_signature,
        data=b"".join(captured) if captured is not None else None,
        utf8_valid=utf8_valid,
        lf_only_text=(
            utf8_valid
            and total > 0
            and not contains_carriage_return
            and last_byte == 0x0A
        ),
    )


def _stream_ordered_concat(
    root: Path,
    files: Mapping[str, FileSnapshot],
) -> OrderedConcatIdentity:
    digest = hashlib.sha256()
    total = 0
    for relative in TRAIN_SHARD_PATHS:
        baseline = files[relative]
        path = root.joinpath(*PurePosixPath(relative).parts)
        before_path = _assert_regular(path, "training_shard_concat_file")
        before_signature = _stat_signature(before_path)
        if before_signature != baseline.stat_signature:
            _fail("training_shard_concat_initial_identity_drift")
        individual = hashlib.sha256()
        individual_bytes = 0
        try:
            with path.open("rb") as handle:
                before_handle = os.fstat(handle.fileno())
                if _stat_signature(before_handle) != before_signature:
                    _fail("training_shard_concat_open_identity_drift")
                while True:
                    chunk = handle.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    individual.update(chunk)
                    individual_bytes += len(chunk)
                    total += len(chunk)
                after_handle = os.fstat(handle.fileno())
        except OSError as exc:
            raise ConsumerPreflightError("training_shard_concat_stream_failed") from exc
        try:
            after_path = path.lstat()
        except OSError as exc:
            raise ConsumerPreflightError(
                "training_shard_concat_terminal_stat_failed"
            ) from exc
        if (
            {
                before_signature,
                _stat_signature(before_handle),
                _stat_signature(after_handle),
                _stat_signature(after_path),
            }
            != {before_signature}
            or path.is_symlink()
            or _is_reparse(after_path)
            or individual.hexdigest() != baseline.sha256
            or individual_bytes != baseline.bytes
        ):
            _fail("training_shard_concat_stream_identity_drift")
    return OrderedConcatIdentity(sha256=digest.hexdigest(), bytes=total)


def _validate_sharded_physical_identities(
    binding: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
    ordered_concat: OrderedConcatIdentity,
) -> None:
    logical = _mapping(
        binding.get("logical_identity"),
        "binding_logical_identity_invalid",
    )
    shard_contract = _mapping(
        binding.get("training_shards"),
        "binding_training_shards_invalid",
    )
    if (
        ordered_concat.sha256 != shard_contract.get("ordered_concat_sha256")
        or ordered_concat.sha256 != logical.get("logical_train_partition_sha256")
        or ordered_concat.bytes != shard_contract.get("logical_partition_bytes")
    ):
        _fail("training_shard_ordered_concat_physical_drift")
    for item_value in _sequence(
        shard_contract.get("shards"),
        "binding_training_shards_invalid",
    ):
        item = _mapping(item_value, "binding_training_shard_invalid")
        snapshot = files[str(item["path"])]
        if item.get("sha256") != snapshot.sha256 or item.get("bytes") != snapshot.bytes:
            _fail("training_shard_physical_identity_drift")
    direct_payload_identities = {
        "eval_proxy_partition_sha256": "eval_proxy/chat.jsonl",
        "serialization_inventory_sha256": "serialization_inventory.jsonl",
        "identity_probe_inventory_sha256": "identity_eval/probe_inventory.jsonl",
    }
    for logical_key, relative in direct_payload_identities.items():
        if logical.get(logical_key) != files[relative].sha256:
            _fail(f"{logical_key}_physical_identity_drift")


def _scan_tree(
    root: Path,
) -> tuple[tuple[int, int], dict[str, tuple[int, int]], set[str]]:
    root_value = _assert_directory(root, "artifact_root")
    root_signature = _directory_signature(root_value)
    directories: dict[str, tuple[int, int]] = {"": root_signature}
    files: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ConsumerPreflightError("artifact_tree_scan_failed") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
                _fail("artifact_tree_path_escape")
            try:
                value = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ConsumerPreflightError("artifact_tree_lstat_failed") from exc
            if entry.is_symlink() or _is_reparse(value):
                _fail("artifact_tree_symlink_or_reparse")
            if stat.S_ISDIR(value.st_mode):
                directories[relative] = _directory_signature(value)
                stack.append(path)
            elif stat.S_ISREG(value.st_mode):
                files.add(relative)
            else:
                _fail("artifact_tree_special_file")
    return root_signature, directories, files


def _snapshot_tree(
    root: Path,
    *,
    capture_metadata: bool,
    exact_tree_paths: frozenset[str] = EXACT_TREE_PATHS,
) -> TreeSnapshot:
    root_signature, directories, files = _scan_tree(root)
    if files != exact_tree_paths:
        _fail("artifact_tree_file_inventory_mismatch")
    expected_directories = {""}
    for relative in exact_tree_paths:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if set(directories) != expected_directories:
        _fail("artifact_tree_directory_inventory_mismatch")
    snapshots: dict[str, FileSnapshot] = {}
    for relative in sorted(exact_tree_paths):
        capture = capture_metadata and (
            relative.endswith(".json") or relative.endswith(".sha256")
        )
        snapshots[relative] = _snapshot_file(
            root.joinpath(*PurePosixPath(relative).parts),
            capture=capture,
        )
    return TreeSnapshot(
        root_signature=root_signature,
        directory_signatures=directories,
        files=snapshots,
    )


def _terminal_tree_recheck(
    root: Path,
    baseline: TreeSnapshot,
    *,
    exact_tree_paths: frozenset[str] = EXACT_TREE_PATHS,
) -> None:
    root_signature, directories, files = _scan_tree(root)
    if (
        root_signature != baseline.root_signature
        or directories != baseline.directory_signatures
        or files != exact_tree_paths
    ):
        _fail("artifact_tree_terminal_identity_drift")
    for relative, previous in baseline.files.items():
        current = _assert_regular(
            root.joinpath(*PurePosixPath(relative).parts),
            "artifact_terminal_file",
        )
        if _stat_signature(current) != previous.stat_signature:
            _fail("artifact_terminal_file_identity_drift")


def _tree_digest(
    files: Mapping[str, FileSnapshot],
    *,
    domain: str = TREE_DIGEST_DOMAIN,
) -> str:
    inventory = [
        {
            "path": path,
            "sha256": files[path].sha256,
            "bytes": files[path].bytes,
        }
        for path in sorted(files)
    ]
    preimage = {"domain": domain, "files": inventory}
    return _sha256(_canonical_json(preimage))


def _producer_tree_digest(files: Mapping[str, FileSnapshot]) -> str:
    inventory = [
        {
            "path": path,
            "sha256": files[path].sha256,
            "bytes": files[path].bytes,
        }
        for path in sorted(files)
    ]
    return _domain_hash(PRODUCER_TREE_DIGEST_DOMAIN, inventory)


def _producer_output_inventory_digest(files: Mapping[str, FileSnapshot]) -> str:
    excluded = {"build_receipt.json", "build_receipt.json.sha256"}
    inventory = [
        {
            "path": path,
            "sha256": files[path].sha256,
            "bytes": files[path].bytes,
        }
        for path in sorted(files)
        if path not in excluded
    ]
    return _domain_hash(PRODUCER_OUTPUT_INVENTORY_DOMAIN, inventory)


def _read_set_digest(entries: Sequence[Any]) -> str:
    canonical: list[dict[str, object]] = []
    for value in entries:
        entry = _mapping(value, "binding_read_set_entry_invalid")
        canonical.append(
            {
                "path": entry["path"],
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "kind": entry["kind"],
            }
        )
    canonical.sort(key=lambda item: str(item["path"]))
    return _domain_hash(READ_SET_DIGEST_DOMAIN, canonical)


def _kind(path: str) -> str:
    if path.endswith(".sha256"):
        return "sha256_sidecar"
    if path.endswith(".jsonl"):
        return "jsonl"
    return "json"


def _payload_pins(binding: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    pins: dict[str, Mapping[str, Any]] = {}
    for value in _sequence(binding["payload_files"], "binding_payload_files_invalid"):
        pin = _mapping(value, "binding_payload_pin_invalid")
        path = str(pin["path"])
        if path in pins:
            _fail("binding_payload_path_duplicate")
        pins[path] = pin
    return pins


def _authenticate_pins(
    binding: Mapping[str, Any],
    snapshot: TreeSnapshot,
    *,
    payload_paths: Sequence[str] = PAYLOAD_PATHS,
    tree_digest_domain: str = TREE_DIGEST_DOMAIN,
) -> None:
    pins = _payload_pins(binding)
    for path in payload_paths:
        payload = snapshot.files[path]
        sidecar_path = f"{path}.sha256"
        sidecar = snapshot.files[sidecar_path]
        pin = pins[path]
        if (
            pin["sha256"] != payload.sha256
            or pin["bytes"] != payload.bytes
            or pin["sidecar_sha256"] != sidecar.sha256
            or pin["sidecar_bytes"] != sidecar.bytes
        ):
            _fail("artifact_binding_file_identity_drift")
        if sidecar.data is None:
            _fail("artifact_sidecar_bytes_missing")
        expected_sidecar = f"{payload.sha256}  {PurePosixPath(path).name}\n".encode(
            "ascii"
        )
        if sidecar.data != expected_sidecar:
            _fail("artifact_sidecar_content_mismatch")
    observed_tree_digest = (
        _producer_tree_digest(snapshot.files)
        if tree_digest_domain == PRODUCER_TREE_DIGEST_DOMAIN
        else _tree_digest(snapshot.files, domain=tree_digest_domain)
    )
    if observed_tree_digest != binding["tree_digest_sha256"]:
        _fail("artifact_tree_digest_drift")


def _assert_payload_size(size: int) -> None:
    if size < 0 or size > MAX_PAYLOAD_BYTES:
        _fail("consumer_payload_size_limit_exceeded")


def _validate_payload_sizes(
    files: Mapping[str, FileSnapshot],
    *,
    payload_paths: Sequence[str] = PAYLOAD_PATHS,
) -> None:
    for path in payload_paths:
        _assert_payload_size(files[path].bytes)


def _claims_gate(value: object, label: str) -> Mapping[str, Any]:
    claims = _mapping(value, f"{label}_claims_invalid")
    required = {
        "candidate": False,
        "final": True,
        "diagnostic_only": True,
        "proxy_only": True,
        "training_authorized": True,
        "formal_training_authorized": False,
        "live_authorized": False,
        "release_authorized": True,
    }
    for key, expected in required.items():
        if claims.get(key) is not expected:
            _fail(f"{label}_{key}_gate_failed")
    return claims


def _require_equal(actual: object, expected: object, code: str) -> None:
    if actual != expected:
        _fail(code)


def _validate_counts(manifest: Mapping[str, Any]) -> None:
    expected = {
        "records": 4300,
        "task_bundles": 1700,
        "task_semantics": 1700,
        "roles": {
            "humor": 300,
            "serious": 300,
            "angry_style": 300,
            "tool_call": 1700,
            "review_audit": 1700,
        },
        "splits": {"train": 3440, "eval_proxy": 860},
        "languages": {"en": 2150, "zh-CN": 2150},
        "language_split": {
            "en:train": 1720,
            "zh-CN:train": 1720,
            "en:eval_proxy": 430,
            "zh-CN:eval_proxy": 430,
        },
        "bundle_classes": {
            "full_five_role_core": 300,
            "tool_review_depth": 1400,
        },
        "review_verdicts": {"pass": 850, "fail": 850},
        "review_fault_types": {
            "format_schema": 170,
            "tool_argument": 170,
            "evidence_mismatch": 170,
            "grounding_mismatch": 170,
            "routing_scope": 170,
        },
        "identity_bundles": 30,
        "identity_role_records": 150,
        "identity_eval_probe_records": 50,
        "tool_direct_no_tool": 30,
        "tool_local_executions": 1670,
    }
    _require_equal(manifest.get("counts"), expected, "manifest_counts_mismatch")
    expected_identity = {
        "source_pool": "full_five_role_core",
        "rows_added_to_training_4300": 0,
        "identity_bundles": 30,
        "train_bundles": 20,
        "eval_proxy_bundles": 10,
        "en_bundles": 15,
        "zh_cn_bundles": 15,
        "intents": {
            "direct_self_identity": 6,
            "provenance_fact_check": 6,
            "no_tool_identity_decision": 6,
            "pre_coding_attribution": 6,
            "false_attribution_correction": 6,
        },
        "role_records": 150,
        "eval_probe_records": 50,
        "tool_behavior": "direct_answer_no_tool",
        "review_rejects_google_openai": True,
        "wrong_route_fact_invariant": True,
    }
    _require_equal(
        manifest.get("identity"),
        expected_identity,
        "manifest_identity_summary_mismatch",
    )
    expected_review = {
        "records": 1700,
        "pass": 850,
        "fail": 850,
        "fault_types": expected["review_fault_types"],
        "same_bundle_tool_dependency": True,
        "candidate_projection_hash_bound": True,
        "tool_record_never_mutated_by_review_fixture": True,
    }
    _require_equal(
        manifest.get("review"),
        expected_review,
        "manifest_review_summary_mismatch",
    )


def _validate_components(
    manifest: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    components = _mapping(manifest.get("components"), "manifest_components_invalid")
    if set(components) != {"router", "tool_eval", "planner_eval"}:
        _fail("manifest_component_inventory_mismatch")
    router = _mapping(components["router"], "manifest_router_invalid")
    router_expected = {
        "namespace": "gemma3_chat_emotion_router_qonly_v1",
        "records": 100,
        "train": 80,
        "eval_proxy": 20,
        "en": 50,
        "zh_cn": 50,
        "labels": {"humor": 34, "serious": 33, "angry_style": 33},
        "not_a_sixth_expert": True,
        "single_total_to_branch_then_router_exits": True,
        "independent_from_generation_experts": True,
    }
    for key, expected in router_expected.items():
        _require_equal(router.get(key), expected, f"manifest_router_{key}_mismatch")
    router_hashes = {
        "manifest_sha256": "components/router/manifest.json",
        "records_sha256": "components/router/records.jsonl",
        "token_inventory_sha256": "components/router/token_inventory.jsonl",
        "build_receipt_sha256": "components/router/build_receipt.json",
    }
    for key, path in router_hashes.items():
        _require_equal(
            router.get(key), files[path].sha256, f"manifest_router_{key}_drift"
        )
    for name, records, slots, namespace in (
        (
            "tool_eval",
            400,
            1200,
            "gemma3_chat_unbalanced_v2_tool_comparison_eval_v1",
        ),
        (
            "planner_eval",
            240,
            960,
            "gemma3_chat_unbalanced_v2_planner_comparison_eval_v1",
        ),
    ):
        component = _mapping(components[name], f"manifest_{name}_component_invalid")
        expected_values = {
            "namespace": namespace,
            "records": records,
            "execution_slots": slots,
            "arm_neutral": True,
            "rows_replicated_by_arms": False,
            "oracle_unchanged": True,
            "strata_unchanged": True,
            "language_unchanged": True,
            "negative_inventory_unchanged": True,
            "independent_review_status": "passed",
        }
        for key, expected in expected_values.items():
            _require_equal(
                component.get(key),
                expected,
                f"manifest_{name}_{key}_mismatch",
            )
        review_path = f"reviews/{name}_independent_review.json"
        hash_paths = {
            "manifest_sha256": f"components/{name}/manifest.json",
            "records_sha256": f"components/{name}/records.jsonl",
            "token_inventory_sha256": (f"components/{name}/token_inventory.jsonl"),
            "build_receipt_sha256": f"components/{name}/build_receipt.json",
            "independent_review_receipt_sha256": review_path,
        }
        for key, path in hash_paths.items():
            _require_equal(
                component.get(key),
                files[path].sha256,
                f"manifest_{name}_{key}_drift",
            )


def _validate_adapter_and_kv(manifest: Mapping[str, Any]) -> None:
    adapter = _mapping(
        manifest.get("adapter_contract"), "manifest_adapter_contract_invalid"
    )
    expected_adapter = {
        "tool_arms": ["base", "tool_q_only", "tool_q_plus_micro_o"],
        "planner_arms": [
            "base",
            "planner_q_only",
            "planner_o_only",
            "planner_q_plus_micro_o",
        ],
        "tool_execution_slots": 1200,
        "planner_execution_slots": 960,
        "q_branch": {
            "target_module": "q_proj",
            "rank": 1024,
            "alpha": 2048,
            "trainable_params": 57933824,
        },
        "micro_o_branch": {
            "target_module": "o_proj",
            "rank": 64,
            "alpha": 128,
            "trainable_params": 3620864,
            "max_lr_ratio_to_q": "1/10",
        },
        "non_tool_q_only_roles": [
            "humor",
            "serious",
            "angry_style",
            "review_audit",
        ],
        "records_added_by_arms": 0,
        "tool_records_arm_neutral": True,
        "planner_records_arm_neutral": True,
        "o_learning_rate_lte_q_over_10_every_step": True,
        "planner_o_only_q_plus_o_o_branch_identity_equal": True,
        "planner_o_only_q_plus_o_o_init_equal": True,
        "planner_o_only_q_plus_o_data_order_steps_seeds_optimizer_equal": True,
        "humor_serious_angry_review_q_only": True,
    }
    _require_equal(adapter, expected_adapter, "manifest_adapter_contract_mismatch")
    kv = _mapping(manifest.get("kv_contract"), "manifest_kv_contract_invalid")
    required_kv = {
        "producer_mode": "frozen_base_adapter_off_single_shared_prefix_prefill",
        "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
        "current_physical_claim": "prefill_compute_reuse",
        "route_plan_commits_immutable_and_hash_bound": True,
        "gemma_canonical_serialization": True,
        "expert_private_tail_append_only": True,
        "planner_live_private_kv_transfer": False,
        "full_generation_kv_shared": False,
        "persistent_zero_copy_verified": False,
        "q8_kv_exact": False,
    }
    for key, expected in required_kv.items():
        _require_equal(kv.get(key), expected, f"manifest_kv_{key}_mismatch")
    if kv.get("persistent_zero_copy_blocked_on") != [
        "data_ptr_identity",
        "storage_identity",
        "cache_identity",
    ]:
        _fail("manifest_kv_zero_copy_blockers_mismatch")


def _validate_manifest_file_inventory(
    manifest: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    entries = _sequence(manifest.get("files"), "manifest_files_invalid")
    observed: dict[str, Mapping[str, Any]] = {}
    for value in entries:
        entry = _mapping(value, "manifest_file_entry_invalid")
        if set(entry) != {"path", "sha256", "bytes", "kind", "records"}:
            _fail("manifest_file_entry_schema_mismatch")
        path = entry.get("path")
        if not isinstance(path, str) or path in observed:
            _fail("manifest_file_path_duplicate_or_invalid")
        observed[path] = entry
    if set(observed) != MANIFEST_LISTED_PATHS:
        _fail("manifest_file_inventory_mismatch")
    for path, entry in observed.items():
        snapshot = files[path]
        if (
            entry["sha256"] != snapshot.sha256
            or entry["bytes"] != snapshot.bytes
            or entry["kind"] != _kind(path)
            or entry["records"] != RECORD_COUNTS.get(path)
        ):
            _fail("manifest_file_binding_drift")


def _validate_manifest(
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    if manifest.get("schema_version") != MANIFEST_VERSION:
        _fail("manifest_schema_version_mismatch")
    if manifest.get("namespace") != NAMESPACE:
        _fail("manifest_namespace_mismatch")
    status = manifest.get("status")
    if (
        not isinstance(status, str)
        or status != binding["expected_manifest_status"]
        or "candidate" in status
        or "pending" in status
    ):
        _fail("manifest_release_status_not_final")
    _claims_gate(manifest.get("claims"), "manifest")
    _validate_manifest_file_inventory(manifest, files)
    _validate_counts(manifest)
    _validate_components(manifest, files)
    _validate_adapter_and_kv(manifest)
    integrity = _mapping(manifest.get("integrity"), "manifest_integrity_invalid")
    required_integrity = {
        "single_authenticated_physical_bytes_snapshot_per_input": True,
        "terminal_toctou_snapshot_recheck_required": True,
        "mandatory_sidecar_for_every_non_sidecar_file": True,
        "checksum_sidecars_are_only_sidecar_exemption": True,
        "raw_token_ids_persisted": False,
        "strict_json_duplicate_and_nonfinite_rejection": True,
        "cross_field_validator": True,
    }
    for key, expected in required_integrity.items():
        _require_equal(
            integrity.get(key), expected, f"manifest_integrity_{key}_mismatch"
        )
    review = _mapping(manifest.get("release_review"), "manifest_release_review_invalid")
    if (
        review.get("implementer_release_signoff") is not False
        or review.get("independent_release_review_required") is not True
    ):
        _fail("manifest_release_review_contract_mismatch")


def _validate_build_receipt(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    if receipt.get("schema_version") != BUILD_RECEIPT_VERSION:
        _fail("build_receipt_schema_version_mismatch")
    if receipt.get("status") != manifest.get("status"):
        _fail("build_receipt_status_mismatch")
    _claims_gate(receipt.get("claims"), "build_receipt")
    manifest_binding = _mapping(
        receipt.get("manifest"), "build_receipt_manifest_binding_invalid"
    )
    manifest_snapshot = files["manifest.json"]
    if (
        manifest_binding.get("path") != "manifest.json"
        or manifest_binding.get("sha256") != manifest_snapshot.sha256
        or manifest_binding.get("bytes") != manifest_snapshot.bytes
    ):
        _fail("build_receipt_manifest_binding_drift")
    resources = _mapping(
        receipt.get("resource_counters"), "build_receipt_resources_invalid"
    )
    for key, expected in _ZERO_RESOURCES.items():
        if resources.get(key) != expected:
            _fail("build_receipt_nonzero_resource_counter")
    audits = _mapping(
        receipt.get("official_model_free_audits"),
        "build_receipt_official_audits_invalid",
    )
    if audits.get("independent_release_review") != "passed":
        _fail("build_receipt_independent_release_review_not_passed")


def _validate_training_shard_contract(
    value: Mapping[str, Any],
    logical_identity: Mapping[str, Any],
    *,
    label: str,
    require_boundary_hashes: bool,
) -> None:
    ordered_paths = value.get("ordered_paths")
    if (
        value.get("shard_count") != 2
        or ordered_paths != list(TRAIN_SHARD_PATHS)
        or value.get("ordered_concat_sha256")
        != logical_identity.get("logical_train_partition_sha256")
        or value.get("logical_partition_records") != SHARDED_TRAIN_RECORDS
        or value.get("task_bundles") != SHARDED_TRAIN_BUNDLES
        or value.get("split_boundary") != "contiguous_task_bundle"
        or value.get("ordered_shards") is not True
        or value.get("task_bundle_intersection_empty") is not True
    ):
        _fail(f"{label}_training_shards_contract_mismatch")
    shards = _sequence(value.get("shards"), f"{label}_training_shards_invalid")
    if len(shards) != 2:
        _fail(f"{label}_training_shard_count_mismatch")
    total_bytes = 0
    total_records = 0
    total_bundles = 0
    boundary_hashes: list[str] = []
    for index, item_value in enumerate(shards):
        item = _mapping(item_value, f"{label}_training_shard_invalid")
        if item.get("index") != index or item.get("path") != TRAIN_SHARD_PATHS[index]:
            _fail(f"{label}_training_shard_order_mismatch")
        total_bytes += int(item.get("bytes", -1))
        total_records += int(item.get("records", -1))
        total_bundles += int(item.get("task_bundles", -1))
        if require_boundary_hashes:
            first = item.get("first_task_bundle_sha256")
            last = item.get("last_task_bundle_sha256")
            if (
                not isinstance(first, str)
                or not isinstance(last, str)
                or _SHA256_RE.fullmatch(first) is None
                or _SHA256_RE.fullmatch(last) is None
                or first == last
            ):
                _fail(f"{label}_training_shard_boundary_invalid")
            boundary_hashes.extend((first, last))
    if (
        total_bytes != value.get("logical_partition_bytes")
        or total_records != SHARDED_TRAIN_RECORDS
        or total_bundles != SHARDED_TRAIN_BUNDLES
    ):
        _fail(f"{label}_training_shard_aggregate_mismatch")
    if require_boundary_hashes and len(set(boundary_hashes)) != 4:
        _fail(f"{label}_training_shard_boundary_overlap")
    if require_boundary_hashes:
        observed_commitment = _training_boundary_commitment(value)
        if value.get("boundary_commitment_sha256") != observed_commitment:
            _fail(f"{label}_training_shard_boundary_commitment_mismatch")


def _training_shard_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    shards = []
    for item_value in _sequence(value.get("shards"), "training_shards_invalid"):
        item = _mapping(item_value, "training_shard_invalid")
        shards.append(
            {
                key: item[key]
                for key in (
                    "index",
                    "path",
                    "sha256",
                    "bytes",
                    "records",
                    "task_bundles",
                )
            }
        )
    return {
        "shard_count": value["shard_count"],
        "ordered_paths": list(value["ordered_paths"]),
        "shards": shards,
        "ordered_concat_sha256": value["ordered_concat_sha256"],
        "logical_partition_bytes": value["logical_partition_bytes"],
        "logical_partition_records": value["logical_partition_records"],
    }


def _schema_document(
    snapshot: FileSnapshot,
    label: str,
    *,
    anchor: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Draft202012Validator]:
    if snapshot.data is None:
        _fail(f"{label}_schema_authenticated_bytes_missing")
    schema = _strict_json(snapshot.data, f"{label}_schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ConsumerPreflightError(f"{label}_schema_invalid") from exc
    if anchor is not None:
        properties = _mapping(
            schema.get("properties"),
            f"{label}_schema_properties_invalid",
        )
        version_property = _mapping(
            properties.get("schema_version"),
            f"{label}_schema_version_property_invalid",
        )
        if (
            schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
            or schema.get("$id") != anchor["schema_id"]
            or version_property.get("const") != anchor["schema_version"]
        ):
            _fail(f"{label}_schema_physical_identity_mismatch")
    return schema, Draft202012Validator(schema)


def _validate_with_physical_schema(
    validator: Draft202012Validator,
    value: Mapping[str, Any],
    label: str,
) -> None:
    try:
        validator.validate(value)
    except ValidationError as exc:
        raise ConsumerPreflightError(f"{label}_schema_mismatch") from exc


def _external_parent_signatures(
    root: Path,
    relative_paths: Sequence[str],
) -> dict[str, tuple[int, int]]:
    result = {"": _directory_signature(_assert_directory(root, "producer_root"))}
    for relative in relative_paths:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            key = parent.as_posix()
            if key not in result:
                path = root.joinpath(*parent.parts)
                result[key] = _directory_signature(
                    _assert_directory(path, "producer_read_set_parent")
                )
            parent = parent.parent
    return result


def _snapshot_external_inputs(
    producer_root: Path,
    binding: Mapping[str, Any],
) -> tuple[
    Mapping[str, FileSnapshot],
    Mapping[str, tuple[int, int]],
    str,
    str,
]:
    read_set = _mapping(binding["read_set"], "binding_read_set_invalid")
    read_entries = _sequence(
        read_set["files"],
        "binding_read_set_files_invalid",
    )
    schema_files = _mapping(
        binding["schema_files"],
        "binding_schema_files_invalid",
    )
    schema_paths = {
        str(_mapping(pin, "binding_schema_pin_invalid")["path"])
        for pin in schema_files.values()
    }
    attestation_pin = _mapping(
        binding["release_attestation"],
        "binding_release_attestation_invalid",
    )
    attestation_path = str(attestation_pin["path"])
    sidecar_path = str(attestation_pin["sidecar_path"])
    read_paths = [
        str(_mapping(value, "binding_read_set_entry_invalid")["path"])
        for value in read_entries
    ]
    if attestation_path in read_paths or sidecar_path in read_paths:
        _fail("release_attestation_must_be_external_to_build_read_set")
    all_paths = [*read_paths, attestation_path, sidecar_path]
    signatures = _external_parent_signatures(producer_root, all_paths)
    snapshots: dict[str, FileSnapshot] = {}
    pins = {
        str(_mapping(value, "binding_read_set_entry_invalid")["path"]): _mapping(
            value, "binding_read_set_entry_invalid"
        )
        for value in read_entries
    }
    for path in sorted(read_paths):
        snapshot = _snapshot_file(
            producer_root.joinpath(*PurePosixPath(path).parts),
            capture=path in schema_paths,
        )
        pin = pins[path]
        if (
            snapshot.sha256 != pin["sha256"]
            or snapshot.bytes != pin["bytes"]
            or pin["kind"] != _source_kind(path)
        ):
            _fail("producer_read_set_physical_identity_drift")
        if not snapshot.lf_only_text:
            _fail("producer_read_set_utf8_lf_mismatch")
        snapshots[path] = snapshot
    attestation_snapshot = _snapshot_file(
        producer_root.joinpath(*PurePosixPath(attestation_path).parts),
        capture=True,
    )
    sidecar_snapshot = _snapshot_file(
        producer_root.joinpath(*PurePosixPath(sidecar_path).parts),
        capture=True,
    )
    if (
        attestation_snapshot.sha256 != attestation_pin["sha256"]
        or attestation_snapshot.bytes != attestation_pin["bytes"]
        or sidecar_snapshot.sha256 != attestation_pin["sidecar_sha256"]
        or sidecar_snapshot.bytes != attestation_pin["sidecar_bytes"]
        or not attestation_snapshot.lf_only_text
        or not sidecar_snapshot.lf_only_text
    ):
        _fail("release_attestation_physical_identity_drift")
    expected_sidecar = (
        f"{attestation_snapshot.sha256}  {PurePosixPath(attestation_path).name}\n"
    ).encode("ascii")
    if sidecar_snapshot.data != expected_sidecar:
        _fail("release_attestation_sidecar_content_mismatch")
    snapshots[attestation_path] = attestation_snapshot
    snapshots[sidecar_path] = sidecar_snapshot
    return snapshots, signatures, attestation_path, sidecar_path


def _terminal_external_recheck(
    producer_root: Path,
    snapshots: Mapping[str, FileSnapshot],
    directory_signatures: Mapping[str, tuple[int, int]],
) -> None:
    for relative, expected in directory_signatures.items():
        path = (
            producer_root
            if relative == ""
            else producer_root.joinpath(*PurePosixPath(relative).parts)
        )
        if (
            _directory_signature(_assert_directory(path, "producer_terminal_parent"))
            != expected
        ):
            _fail("producer_read_set_directory_identity_drift")
    for relative, snapshot in snapshots.items():
        current = _assert_regular(
            producer_root.joinpath(*PurePosixPath(relative).parts),
            "producer_terminal_file",
        )
        if _stat_signature(current) != snapshot.stat_signature:
            _fail("producer_read_set_terminal_toctou")


def _artifact_text_gate(files: Mapping[str, FileSnapshot]) -> None:
    for snapshot in files.values():
        if not snapshot.lf_only_text:
            _fail("artifact_utf8_lf_mismatch")


def _git(
    root: Path,
    arguments: Sequence[str],
    *,
    stdin: bytes | None = None,
    allow_exit_one: bool = False,
) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(root), *arguments],
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConsumerPreflightError("producer_git_command_failed") from exc
    allowed = {0, 1} if allow_exit_one else {0}
    if completed.returncode not in allowed:
        _fail("producer_git_command_failed")
    return completed.stdout


def _git_text(root: Path, arguments: Sequence[str]) -> str:
    try:
        return _git(root, arguments).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ConsumerPreflightError("producer_git_output_invalid") from exc


def _validate_git_attributes(root: Path, paths: Sequence[str]) -> None:
    raw = _git(
        root,
        ["check-attr", "-z", "--stdin", "text", "eol"],
        stdin=b"".join(path.encode("utf-8") + b"\0" for path in paths),
    )
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    if len(parts) != len(paths) * 6:
        _fail("producer_git_attributes_output_invalid")
    observed: dict[tuple[str, str], str] = {}
    for offset in range(0, len(parts), 3):
        try:
            path = parts[offset].decode("utf-8")
            attribute = parts[offset + 1].decode("ascii")
            value = parts[offset + 2].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ConsumerPreflightError(
                "producer_git_attributes_output_invalid"
            ) from exc
        observed[(path, attribute)] = value
    for path in paths:
        if observed.get((path, "text")) != "set" or observed.get((path, "eol")) != "lf":
            _fail("producer_git_lf_attributes_mismatch")


def _validate_git_state(
    producer_root: Path,
    artifact_root: Path,
    binding: Mapping[str, Any],
    *,
    observed_producer_commit: str,
    observed_live_remote_commit: str | None,
    external_paths: Sequence[str],
) -> None:
    git = _mapping(binding["producer_git"], "binding_producer_git_invalid")
    release_commit = str(git["release_commit"])
    candidate_commit = str(git["candidate_commit"])
    if (
        observed_producer_commit != release_commit
        or observed_live_remote_commit != git["live_remote_commit"]
    ):
        _fail("producer_git_observed_release_state_drift")
    if _git_text(producer_root, ["rev-parse", "--verify", "HEAD"]) != release_commit:
        _fail("producer_git_head_drift")
    if (
        _git_text(producer_root, ["rev-parse", "--verify", "HEAD^{tree}"])
        != git["release_tree"]
    ):
        _fail("producer_git_tree_drift")
    if (
        _git_text(producer_root, ["rev-parse", "--verify", "@{upstream}"])
        != git["upstream_commit"]
    ):
        _fail("producer_git_upstream_drift")
    _git(
        producer_root,
        ["merge-base", "--is-ancestor", candidate_commit, release_commit],
    )
    if _git_text(producer_root, ["replace", "--list"]):
        _fail("producer_git_replace_refs_present")
    grafts = Path(_git_text(producer_root, ["rev-parse", "--git-path", "info/grafts"]))
    if not grafts.is_absolute():
        grafts = producer_root / grafts
    try:
        if grafts.is_file() and grafts.stat().st_size:
            _fail("producer_git_grafts_present")
    except OSError as exc:
        raise ConsumerPreflightError("producer_git_grafts_check_failed") from exc
    if _git_text(producer_root, ["status", "--porcelain=v1", "--untracked-files=all"]):
        _fail("producer_git_worktree_not_clean")
    tags = [
        line
        for line in _git_text(
            producer_root,
            ["tag", "--points-at", release_commit],
        ).splitlines()
        if line
    ]
    if tags or git["tags_at_head"] != 0:
        _fail("producer_git_tags_at_head_nonzero")
    try:
        artifact_relative = artifact_root.relative_to(producer_root).as_posix()
    except ValueError:
        _fail("artifact_outside_producer_repository")
    attestation = _mapping(
        binding["release_attestation"],
        "binding_release_attestation_invalid",
    )
    tracked_paths = sorted(
        {
            *(
                str(_mapping(value, "binding_read_set_entry_invalid")["path"])
                for value in _sequence(
                    _mapping(binding["read_set"], "binding_read_set_invalid")["files"],
                    "binding_read_set_files_invalid",
                )
            ),
            *external_paths,
            *(
                f"{artifact_relative}/{relative}"
                for relative in SHARDED_EXACT_TREE_PATHS
            ),
        }
    )
    _git(
        producer_root,
        ["ls-files", "--error-unmatch", "--", *tracked_paths],
    )
    _git(
        producer_root,
        ["diff", "--quiet", release_commit, "--", *tracked_paths],
    )
    candidate_scoped = [
        path
        for path in tracked_paths
        if path
        not in {
            str(attestation["path"]),
            str(attestation["sidecar_path"]),
        }
    ]
    _git(
        producer_root,
        [
            "diff",
            "--quiet",
            candidate_commit,
            release_commit,
            "--",
            *candidate_scoped,
        ],
    )
    _validate_git_attributes(producer_root, tracked_paths)


def _metadata_json(
    snapshot: FileSnapshot,
    label: str,
) -> Mapping[str, Any]:
    if snapshot.data is None:
        _fail(f"{label}_authenticated_bytes_missing")
    return _strict_json(snapshot.data, label)


def _common_training_shards(
    value: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    shards = [
        dict(_mapping(item, f"{source}_training_shard_invalid"))
        for item in _sequence(
            value.get("shards"),
            f"{source}_training_shards_invalid",
        )
    ]
    if source == "attestation":
        split_boundary = (
            "contiguous_task_bundle"
            if value.get("bundle_boundary_only") is True
            else None
        )
    else:
        split_boundary = value.get("split_boundary")
    result = {
        "shard_count": value.get("shard_count"),
        "ordered_paths": value.get("ordered_paths"),
        "shards": shards,
        "ordered_concat_sha256": value.get("ordered_concat_sha256"),
        "logical_partition_bytes": value.get("logical_partition_bytes"),
        "logical_partition_records": value.get("logical_partition_records"),
        "task_bundles": sum(int(item.get("task_bundles", -1)) for item in shards),
        "split_boundary": split_boundary,
        "ordered_shards": True,
        "task_bundle_intersection_empty": True,
    }
    if source != "attestation":
        result["boundary_commitment_sha256"] = _training_boundary_commitment(result)
    return result


def _validate_sharded_manifest_file_inventory(
    manifest: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
    training_shards: Mapping[str, Any],
) -> None:
    shard_records = {
        str(_mapping(item, "manifest_training_shard_invalid")["path"]): int(
            _mapping(item, "manifest_training_shard_invalid")["records"]
        )
        for item in _sequence(
            training_shards["shards"],
            "manifest_training_shards_invalid",
        )
    }
    entries = _sequence(manifest.get("files"), "manifest_files_invalid")
    observed: dict[str, Mapping[str, Any]] = {}
    for value in entries:
        entry = _mapping(value, "manifest_file_entry_invalid")
        _exact_keys(
            entry,
            {"path", "sha256", "bytes", "kind", "records"},
            "manifest_file_entry_schema_mismatch",
        )
        path = entry.get("path")
        if not isinstance(path, str) or path in observed:
            _fail("manifest_file_path_duplicate_or_invalid")
        observed[path] = entry
    if set(observed) != SHARDED_MANIFEST_LISTED_PATHS:
        _fail("manifest_file_inventory_mismatch")
    for path, entry in observed.items():
        snapshot = files[path]
        records = shard_records.get(path, SHARDED_RECORD_COUNTS.get(path))
        if (
            entry["sha256"] != snapshot.sha256
            or entry["bytes"] != snapshot.bytes
            or entry["kind"] != _kind(path)
            or entry["records"] != records
        ):
            _fail("manifest_file_binding_drift")


def _validate_sharded_manifest(
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    if (
        manifest.get("schema_version") != binding["expected_manifest_schema_version"]
        or manifest.get("namespace") != NAMESPACE
        or manifest.get("artifact_version") != binding["artifact_version"]
        or manifest.get("status") != binding["expected_manifest_status"]
    ):
        _fail("sharded_manifest_identity_mismatch")
    logical = _mapping(
        manifest.get("logical_identity"),
        "manifest_logical_identity_invalid",
    )
    _exact_keys(
        logical,
        set(LOGICAL_IDENTITY_KEYS),
        "manifest_logical_identity_keys_mismatch",
    )
    _require_equal(
        logical,
        binding["logical_identity"],
        "manifest_logical_identity_drift",
    )
    manifest_shards = _mapping(
        manifest.get("training_shards"),
        "manifest_training_shards_invalid",
    )
    common = _common_training_shards(manifest_shards, source="manifest")
    _validate_training_shard_contract(
        common,
        logical,
        label="manifest",
        require_boundary_hashes=True,
    )
    _require_equal(
        common,
        binding["training_shards"],
        "manifest_training_shards_binding_drift",
    )
    for shard_value in _sequence(common["shards"], "manifest_training_shards_invalid"):
        shard = _mapping(shard_value, "manifest_training_shard_invalid")
        snapshot = files[str(shard["path"])]
        if (
            shard.get("sha256") != snapshot.sha256
            or shard.get("bytes") != snapshot.bytes
        ):
            _fail("manifest_training_shard_physical_drift")
    _validate_sharded_manifest_file_inventory(manifest, files, common)
    _validate_counts(manifest)
    _validate_components(manifest, files)
    _validate_adapter_and_kv(manifest)
    resources = _mapping(
        manifest.get("resource_counters"),
        "manifest_resource_counters_invalid",
    )
    for key, expected in _ZERO_RESOURCES.items():
        if resources.get(key) != expected:
            _fail("manifest_nonzero_resource_counter")
    claims = _mapping(manifest.get("claims"), "manifest_claims_invalid")
    if not (
        claims.get("candidate") is True
        and claims.get("final") is False
        and claims.get("training_authorized") is False
        and claims.get("formal_training_authorized") is False
        and claims.get("live_authorized") is False
        and claims.get("release_authorized") is False
    ):
        _fail("sharded_manifest_candidate_claims_mismatch")


def _validate_sharded_build_receipt(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    if (
        receipt.get("schema_version")
        != binding["expected_build_receipt_schema_version"]
        or receipt.get("status") != manifest.get("status")
        or receipt.get("artifact_version") != binding["artifact_version"]
    ):
        _fail("sharded_build_receipt_identity_mismatch")
    manifest_binding = _mapping(
        receipt.get("manifest"),
        "build_receipt_manifest_binding_invalid",
    )
    manifest_snapshot = files["manifest.json"]
    if (
        manifest_binding.get("path") != "manifest.json"
        or manifest_binding.get("sha256") != manifest_snapshot.sha256
        or manifest_binding.get("bytes") != manifest_snapshot.bytes
    ):
        _fail("build_receipt_manifest_binding_drift")
    logical = _mapping(
        receipt.get("logical_identity"),
        "build_receipt_logical_identity_invalid",
    )
    _exact_keys(
        logical,
        set(LOGICAL_IDENTITY_KEYS),
        "build_receipt_logical_identity_keys_mismatch",
    )
    _require_equal(
        logical,
        binding["logical_identity"],
        "build_receipt_logical_identity_drift",
    )
    receipt_shards = _mapping(
        receipt.get("training_shards"),
        "build_receipt_training_shards_invalid",
    )
    binding_shards = _mapping(
        binding["training_shards"],
        "binding_training_shards_invalid",
    )
    for key in (
        "shard_count",
        "ordered_paths",
        "ordered_concat_sha256",
        "logical_partition_bytes",
        "logical_partition_records",
    ):
        if receipt_shards.get(key) != binding_shards.get(key):
            _fail("build_receipt_training_shards_drift")
    if (
        receipt_shards.get("logical_partition_sha256")
        != logical["logical_train_partition_sha256"]
        or receipt_shards.get("max_file_bytes_exclusive") != MAX_PAYLOAD_BYTES + 1
    ):
        _fail("build_receipt_training_shards_contract_mismatch")
    observed_output_inventory = _producer_output_inventory_digest(files)
    if (
        receipt.get("output_inventory_sha256") != observed_output_inventory
        or binding.get("producer_output_inventory_sha256") != observed_output_inventory
    ):
        _fail("build_receipt_output_inventory_drift")
    entries = _sequence(
        receipt.get("input_snapshot_inventory"),
        "build_receipt_input_snapshot_inventory_invalid",
    )
    read_set = _mapping(binding["read_set"], "binding_read_set_invalid")
    pins = {
        str(_mapping(value, "binding_read_set_entry_invalid")["path"]): _mapping(
            value, "binding_read_set_entry_invalid"
        )
        for value in _sequence(
            read_set["files"],
            "binding_read_set_files_invalid",
        )
    }
    observed: dict[str, Mapping[str, Any]] = {}
    labels: list[str] = []
    for entry_value in entries:
        entry = _mapping(entry_value, "build_receipt_input_snapshot_entry_invalid")
        _exact_keys(
            entry,
            {"label", "path", "sha256", "bytes", "source_class"},
            "build_receipt_input_snapshot_entry_keys_mismatch",
        )
        path = _safe_relative_path(
            entry.get("path"),
            "build_receipt_input_snapshot_path_invalid",
        )
        label = entry.get("label")
        if not isinstance(label, str) or not label or path in observed:
            _fail("build_receipt_input_snapshot_inventory_duplicate")
        observed[path] = entry
        labels.append(label)
    if labels != sorted(labels) or len(labels) != len(set(labels)):
        _fail("build_receipt_input_snapshot_label_order_mismatch")
    if set(observed) != set(pins):
        _fail("build_receipt_input_snapshot_inventory_drift")
    for path, entry in observed.items():
        pin = pins[path]
        if (
            entry["sha256"] != pin["sha256"]
            or entry["bytes"] != pin["bytes"]
            or pin["kind"] != _source_kind(path)
        ):
            _fail("build_receipt_input_snapshot_binding_drift")
    producer_inventory_digest = _domain_hash(
        PRODUCER_INPUT_INVENTORY_DOMAIN,
        [
            dict(_mapping(value, "build_receipt_input_snapshot_entry_invalid"))
            for value in entries
        ],
    )
    if producer_inventory_digest != receipt.get(
        "input_snapshot_inventory_sha256"
    ) or producer_inventory_digest != read_set.get("producer_inventory_sha256"):
        _fail("build_receipt_input_snapshot_digest_drift")
    audits = _mapping(
        receipt.get("official_model_free_audits"),
        "build_receipt_official_audits_invalid",
    )
    if audits.get("independent_release_review") != "pending":
        _fail("build_receipt_release_lifecycle_mismatch")
    resources = _mapping(
        receipt.get("resource_counters"),
        "build_receipt_resources_invalid",
    )
    for key, expected in _ZERO_RESOURCES.items():
        if resources.get(key) != expected:
            _fail("build_receipt_nonzero_resource_counter")


def _validate_release_attestation(
    attestation: Mapping[str, Any],
    manifest: Mapping[str, Any],
    build_receipt: Mapping[str, Any],
    binding: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
    *,
    artifact_relative: str,
) -> None:
    if (
        attestation.get("schema_version")
        != binding["expected_attestation_schema_version"]
        or attestation.get("audit_status") != "passed"
        or attestation.get("artifact_lifecycle_status") != manifest["status"]
        or attestation.get("namespace") != NAMESPACE
        or attestation.get("artifact_version") != binding["artifact_version"]
        or attestation.get("canonical_artifact_path") != artifact_relative
        or attestation.get("producer_candidate_commit")
        != _mapping(binding["producer_git"], "binding_producer_git_invalid")[
            "candidate_commit"
        ]
    ):
        _fail("release_attestation_identity_mismatch")
    reviewer = _mapping(
        attestation.get("reviewer_separation"),
        "release_attestation_reviewer_invalid",
    )
    if reviewer.get("different_reviewer") is not True or reviewer.get(
        "producer_id"
    ) == reviewer.get("reviewer_id"):
        _fail("release_attestation_reviewer_separation_mismatch")
    _require_equal(
        attestation.get("findings"),
        {"p0": 0, "p1": 0, "p2": 0},
        "release_attestation_findings_nonzero",
    )
    artifact = _mapping(
        attestation.get("artifact"),
        "release_attestation_artifact_invalid",
    )
    if (
        artifact.get("files") != 46
        or artifact.get("tree_sha256") != _producer_tree_digest(files)
        or artifact.get("manifest_sha256") != files["manifest.json"].sha256
        or artifact.get("manifest_sidecar_physical_sha256")
        != files["manifest.json.sha256"].sha256
        or artifact.get("build_receipt_sha256") != files["build_receipt.json"].sha256
        or artifact.get("build_receipt_sidecar_physical_sha256")
        != files["build_receipt.json.sha256"].sha256
        or artifact.get("output_inventory_sha256")
        != build_receipt["output_inventory_sha256"]
        or artifact.get("all_payload_sidecars_valid") is not True
        or artifact.get("all_files_below_50_mib") is not True
    ):
        _fail("release_attestation_artifact_binding_drift")
    logical = _mapping(
        attestation.get("logical_identity"),
        "release_attestation_logical_identity_invalid",
    )
    _exact_keys(
        logical,
        set(LOGICAL_IDENTITY_KEYS),
        "release_attestation_logical_identity_keys_mismatch",
    )
    _require_equal(
        logical,
        binding["logical_identity"],
        "release_attestation_logical_identity_drift",
    )
    attestation_shards = _common_training_shards(
        _mapping(
            attestation.get("training_shards"),
            "release_attestation_training_shards_invalid",
        ),
        source="attestation",
    )
    _validate_training_shard_contract(
        attestation_shards,
        logical,
        label="release_attestation",
        require_boundary_hashes=False,
    )
    if _training_shard_projection(attestation_shards) != _training_shard_projection(
        _mapping(binding["training_shards"], "binding_training_shards_invalid")
    ):
        _fail("release_attestation_training_shards_drift")
    git_auth = _mapping(
        attestation.get("git_authentication"),
        "release_attestation_git_authentication_invalid",
    )
    if not git_auth or any(value is not True for value in git_auth.values()):
        _fail("release_attestation_git_authentication_failed")
    claims = _mapping(
        attestation.get("claims"),
        "release_attestation_claims_invalid",
    )
    if not (
        claims.get("artifact_final_identity") is True
        and claims.get("independent_review_passed") is True
        and claims.get("training_authorized") is False
        and claims.get("formal_training_authorized") is False
        and claims.get("live_authorized") is False
        and claims.get("model_release_authorized") is False
    ):
        _fail("release_attestation_claims_mismatch")


def _receipt(
    *,
    operation: str,
    binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    build_receipt: Mapping[str, Any],
    snapshot: TreeSnapshot,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_VERSION,
        "status": "passed",
        "operation": operation,
        "namespace": NAMESPACE,
        "model_free": True,
        "binding_contract_sha256": _sha256(_canonical_json(binding)),
        "producer_git_commit": binding["producer_git_commit"],
        "producer_manifest_schema_sha256": binding["producer_manifest_schema_sha256"],
        "producer_build_receipt_schema_sha256": binding[
            "producer_build_receipt_schema_sha256"
        ],
        "release_review_receipt_sha256": binding["release_review"]["receipt_sha256"],
        "tree_digest_sha256": _tree_digest(snapshot.files),
        "file_counts": {"total": 44, "payload": 22, "sidecar": 22},
        "manifest": {
            "schema_version": str(manifest["schema_version"]),
            "sha256": snapshot.files["manifest.json"].sha256,
            "bytes": snapshot.files["manifest.json"].bytes,
        },
        "build_receipt": {
            "schema_version": str(build_receipt["schema_version"]),
            "sha256": snapshot.files["build_receipt.json"].sha256,
            "bytes": snapshot.files["build_receipt.json"].bytes,
        },
        "aggregate_counts": {
            "training_records": 4300,
            "training_train_records": 3440,
            "training_eval_proxy_records": 860,
            "router_records": 100,
            "router_train_records": 80,
            "router_eval_proxy_records": 20,
            "tool_eval_records": 400,
            "planner_eval_records": 240,
            "identity_probe_records": 50,
        },
        "release": {
            "manifest_status": manifest["status"],
            "independent_release_review": "passed",
            "final": True,
            "release_authorized": True,
            "training_authorized": True,
            "formal_training_authorized": False,
        },
        "terminal_recheck": {
            "two_snapshot_bytes_equal": True,
            "stat_identity_equal": True,
            "physical_inventory_equal": True,
        },
        "resource_counters": dict(_ZERO_RESOURCES),
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "training_started": False,
            "data_copied": False,
            "sample_bodies_parsed": False,
            "raw_token_ids_parsed": False,
        },
    }


def _sharded_receipt(
    *,
    operation: str,
    binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    build_receipt: Mapping[str, Any],
    attestation: Mapping[str, Any],
    artifact_snapshot: TreeSnapshot,
    external_snapshots: Mapping[str, FileSnapshot],
) -> dict[str, Any]:
    schema_files = _mapping(
        binding["schema_files"],
        "binding_schema_files_invalid",
    )
    attestation_pin = _mapping(
        binding["release_attestation"],
        "binding_release_attestation_invalid",
    )
    read_set = _mapping(binding["read_set"], "binding_read_set_invalid")
    return {
        "schema_version": SHARDED_RECEIPT_VERSION,
        "status": "passed",
        "operation": operation,
        "namespace": NAMESPACE,
        "model_free": True,
        "binding_contract_sha256": _sha256(_canonical_json(binding)),
        "artifact_version": binding["artifact_version"],
        "producer_git": dict(
            _mapping(binding["producer_git"], "binding_producer_git_invalid")
        ),
        "tree_digest_domain": PRODUCER_TREE_DIGEST_DOMAIN,
        "tree_digest_sha256": _producer_tree_digest(artifact_snapshot.files),
        "file_counts": {"total": 46, "payload": 23, "sidecar": 23},
        "manifest": {
            "schema_version": manifest["schema_version"],
            "sha256": artifact_snapshot.files["manifest.json"].sha256,
            "bytes": artifact_snapshot.files["manifest.json"].bytes,
        },
        "build_receipt": {
            "schema_version": build_receipt["schema_version"],
            "sha256": artifact_snapshot.files["build_receipt.json"].sha256,
            "bytes": artifact_snapshot.files["build_receipt.json"].bytes,
        },
        "schema_files": {
            name: {
                "sha256": external_snapshots[
                    str(_mapping(pin, "schema_pin")["path"])
                ].sha256,
                "bytes": external_snapshots[
                    str(_mapping(pin, "schema_pin")["path"])
                ].bytes,
            }
            for name, pin in schema_files.items()
        },
        "release_attestation": {
            "schema_version": attestation["schema_version"],
            "sha256": external_snapshots[str(attestation_pin["path"])].sha256,
            "bytes": external_snapshots[str(attestation_pin["path"])].bytes,
            "sidecar_sha256": external_snapshots[
                str(attestation_pin["sidecar_path"])
            ].sha256,
            "sidecar_bytes": external_snapshots[
                str(attestation_pin["sidecar_path"])
            ].bytes,
        },
        "training_shards": {
            "shard_count": 2,
            "ordered_paths": list(TRAIN_SHARD_PATHS),
            "ordered_concat_sha256": binding["training_shards"][
                "ordered_concat_sha256"
            ],
            "logical_partition_bytes": binding["training_shards"][
                "logical_partition_bytes"
            ],
            "logical_partition_records": SHARDED_TRAIN_RECORDS,
            "task_bundles": SHARDED_TRAIN_BUNDLES,
            "bundle_boundary_only": True,
            "ordered_shards": True,
            "task_bundle_intersection_empty": True,
            "boundary_commitment_sha256": binding["training_shards"][
                "boundary_commitment_sha256"
            ],
        },
        "logical_identity": dict(
            _mapping(
                binding["logical_identity"],
                "binding_logical_identity_invalid",
            )
        ),
        "read_set": {
            "count": read_set["count"],
            "canonical_digest_sha256": read_set["canonical_digest_sha256"],
            "producer_inventory_sha256": read_set["producer_inventory_sha256"],
            "physical_identities_equal": True,
            "utf8_lf": True,
        },
        "release": {
            "artifact_final_identity": True,
            "independent_release_review": "passed",
            "p0_findings": 0,
            "p1_findings": 0,
            "p2_findings": 0,
            "producer_training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "model_release_authorized": False,
        },
        "terminal_recheck": {
            "two_artifact_snapshots_equal": True,
            "artifact_stat_identity_equal": True,
            "external_single_read_stat_identity_equal": True,
            "physical_inventory_equal": True,
        },
        "resource_counters": dict(_ZERO_RESOURCES),
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "training_started": False,
            "data_copied": False,
            "sample_bodies_parsed": False,
            "raw_token_ids_parsed": False,
            "producer_training_authority_inferred": False,
        },
    }


def _validate_sharded_artifact(
    artifact_root: Path,
    binding: Mapping[str, Any],
    *,
    producer_repository_root: str | Path | None,
    observed_producer_commit: str,
    observed_live_remote_commit: str | None,
    operation: str,
    before_terminal_recheck: Callable[[], None] | None,
) -> dict[str, Any]:
    if producer_repository_root is None:
        _fail("producer_repository_root_required")
    producer_root = Path(producer_repository_root).absolute()
    _assert_no_symlink_or_reparse_ancestors(producer_root, "producer_root")
    _assert_directory(producer_root, "producer_root")
    root = artifact_root.absolute()
    _assert_no_symlink_or_reparse_ancestors(root, "artifact_root")
    first = _snapshot_tree(
        root,
        capture_metadata=True,
        exact_tree_paths=SHARDED_EXACT_TREE_PATHS,
    )
    _authenticate_pins(
        binding,
        first,
        payload_paths=SHARDED_PAYLOAD_PATHS,
        tree_digest_domain=str(binding["tree_digest_domain"]),
    )
    ordered_concat = _stream_ordered_concat(root, first.files)
    _validate_sharded_physical_identities(binding, first.files, ordered_concat)
    _validate_payload_sizes(first.files, payload_paths=SHARDED_PAYLOAD_PATHS)
    _artifact_text_gate(first.files)
    manifest = _metadata_json(first.files["manifest.json"], "manifest")
    build_receipt = _metadata_json(
        first.files["build_receipt.json"],
        "build_receipt",
    )
    (
        external_snapshots,
        external_directories,
        attestation_path,
        sidecar_path,
    ) = _snapshot_external_inputs(producer_root, binding)
    schema_pins = _mapping(
        binding["schema_files"],
        "binding_schema_files_invalid",
    )
    schema_physical_identities = {
        external_snapshots[
            str(_mapping(pin, "binding_schema_pin_invalid")["path"])
        ].stat_signature[:2]
        for pin in schema_pins.values()
    }
    if len(schema_physical_identities) != 3:
        _fail("producer_schema_physical_identity_reused")
    validators: dict[str, Draft202012Validator] = {}
    for name, pin_value in schema_pins.items():
        pin = _mapping(pin_value, "binding_schema_pin_invalid")
        _, validators[name] = _schema_document(
            external_snapshots[str(pin["path"])],
            f"producer_{name}",
            anchor=SHARDED_FINAL_SCHEMA_ANCHORS[name],
        )
    attestation = _metadata_json(
        external_snapshots[attestation_path],
        "release_attestation",
    )
    _validate_with_physical_schema(validators["manifest"], manifest, "manifest")
    _validate_with_physical_schema(
        validators["build_receipt"],
        build_receipt,
        "build_receipt",
    )
    _validate_with_physical_schema(
        validators["release_attestation"],
        attestation,
        "release_attestation",
    )
    _validate_sharded_manifest(manifest, binding, first.files)
    _validate_sharded_build_receipt(
        build_receipt,
        manifest,
        binding,
        first.files,
    )
    try:
        artifact_relative = root.relative_to(producer_root).as_posix()
    except ValueError:
        _fail("artifact_outside_producer_repository")
    _validate_release_attestation(
        attestation,
        manifest,
        build_receipt,
        binding,
        first.files,
        artifact_relative=artifact_relative,
    )
    _validate_git_state(
        producer_root,
        root,
        binding,
        observed_producer_commit=observed_producer_commit,
        observed_live_remote_commit=observed_live_remote_commit,
        external_paths=(attestation_path, sidecar_path),
    )
    if before_terminal_recheck is not None:
        before_terminal_recheck()
    second = _snapshot_tree(
        root,
        capture_metadata=False,
        exact_tree_paths=SHARDED_EXACT_TREE_PATHS,
    )
    if (
        first.root_signature != second.root_signature
        or first.directory_signatures != second.directory_signatures
        or set(first.files) != set(second.files)
    ):
        _fail("artifact_two_snapshot_tree_identity_drift")
    for relative, previous in first.files.items():
        if not previous.same_identity_and_bytes(second.files[relative]):
            _fail("artifact_two_snapshot_byte_drift")
    _terminal_tree_recheck(
        root,
        second,
        exact_tree_paths=SHARDED_EXACT_TREE_PATHS,
    )
    _terminal_external_recheck(
        producer_root,
        external_snapshots,
        external_directories,
    )
    result = _sharded_receipt(
        operation=operation,
        binding=binding,
        manifest=manifest,
        build_receipt=build_receipt,
        attestation=attestation,
        artifact_snapshot=second,
        external_snapshots=external_snapshots,
    )
    try:
        _schema_validator().validate(result)
    except ValidationError as exc:
        raise ConsumerPreflightError("preflight_receipt_schema_mismatch") from exc
    return result


def validate_artifact(
    artifact_root: str | Path,
    binding_document: Mapping[str, Any],
    *,
    observed_producer_commit: str,
    producer_repository_root: str | Path | None = None,
    observed_live_remote_commit: str | None = None,
    operation: str = "validate",
    before_terminal_recheck: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Authenticate the exact tree and return a body-free receipt.

    ``before_terminal_recheck`` exists for deterministic race-negative tests;
    production CLI callers cannot supply it.
    """

    if operation not in {"validate", "dry-run"}:
        _fail("consumer_operation_invalid")
    binding = _validate_binding_document(
        binding_document,
        ignore_payload_size_maximum=True,
    )
    if (
        _COMMIT_RE.fullmatch(observed_producer_commit) is None
        or observed_producer_commit == "0" * 40
    ):
        _fail("observed_producer_commit_invalid")
    if (
        binding["schema_version"] == BINDING_VERSION
        and observed_producer_commit != binding["producer_git_commit"]
    ):
        _fail("producer_git_commit_drift")
    if binding["schema_version"] == SHARDED_BINDING_VERSION:
        return _validate_sharded_artifact(
            Path(artifact_root),
            binding,
            producer_repository_root=producer_repository_root,
            observed_producer_commit=observed_producer_commit,
            observed_live_remote_commit=observed_live_remote_commit,
            operation=operation,
            before_terminal_recheck=before_terminal_recheck,
        )
    root = Path(artifact_root).absolute()
    _assert_no_symlink_or_reparse_ancestors(root, "artifact_root")
    first = _snapshot_tree(root, capture_metadata=True)
    _authenticate_pins(binding, first)
    manifest = _metadata_json(first.files["manifest.json"], "manifest")
    build_receipt = _metadata_json(first.files["build_receipt.json"], "build_receipt")
    _validate_manifest(manifest, binding, first.files)
    _validate_build_receipt(build_receipt, manifest, first.files)
    _validate_payload_sizes(first.files)
    validate_binding_document(binding)
    if before_terminal_recheck is not None:
        before_terminal_recheck()
    second = _snapshot_tree(root, capture_metadata=False)
    if (
        first.root_signature != second.root_signature
        or first.directory_signatures != second.directory_signatures
        or set(first.files) != set(second.files)
    ):
        _fail("artifact_two_snapshot_tree_identity_drift")
    for relative, previous in first.files.items():
        if not previous.same_identity_and_bytes(second.files[relative]):
            _fail("artifact_two_snapshot_byte_drift")
    _terminal_tree_recheck(root, second)
    result = _receipt(
        operation=operation,
        binding=binding,
        manifest=manifest,
        build_receipt=build_receipt,
        snapshot=second,
    )
    try:
        _schema_validator().validate(result)
    except ValidationError as exc:
        raise ConsumerPreflightError("preflight_receipt_schema_mismatch") from exc
    return result


def _atomic_write_receipt(
    output: str | Path,
    receipt: Mapping[str, Any],
    *,
    artifact_root: str | Path,
) -> None:
    target = Path(output).absolute()
    root = Path(artifact_root).absolute()
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        _fail("receipt_output_inside_authenticated_artifact")
    parent = target.parent
    _assert_directory(parent, "receipt_output_parent")
    raw = _canonical_json(receipt)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise ConsumerPreflightError("receipt_atomic_write_failed") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Model-free strict consumer authentication for Gemma 3 Chat unbalanced-v2."
        )
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("validate", "dry-run"):
        command = subparsers.add_parser(operation)
        command.add_argument("--artifact", required=True)
        command.add_argument("--binding", required=True)
        command.add_argument("--producer-commit", required=True)
        command.add_argument(
            "--producer-repository",
            help=(
                "Required by sharded-v1 to authenticate the producer Git "
                "checkout, schemas, read-set, and external release attestation."
            ),
        )
        command.add_argument(
            "--producer-live-remote-commit",
            help=(
                "Required by sharded-v1; body-free live-remote observation "
                "which must equal the frozen release commit."
            ),
        )
        command.add_argument(
            "--output",
            help=(
                "Explicit path for an atomic body-free receipt. No file is "
                "written when omitted."
            ),
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        binding = load_binding_document(args.binding)
        receipt = validate_artifact(
            args.artifact,
            binding,
            observed_producer_commit=args.producer_commit,
            producer_repository_root=args.producer_repository,
            observed_live_remote_commit=args.producer_live_remote_commit,
            operation=args.operation,
        )
        if args.output:
            _atomic_write_receipt(
                args.output,
                receipt,
                artifact_root=args.artifact,
            )
    except ConsumerPreflightError as exc:
        failure = {"status": "failed", "error_code": str(exc)}
        print(
            json.dumps(failure, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
