"""Model-free contract for the Gemma 3 unbalanced-v2 multi-arm diagnostic.

This module deliberately does not import Torch, PEFT, Transformers, or
bitsandbytes.  It authenticates a body-free consumer preflight receipt,
validates synthetic/observed tensor-inventory metadata, freezes paired-arm
ordering and schedule identities, and emits only body-free plans and scalar
progress.  A later GPU implementation must satisfy this contract; this module
cannot execute training.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Final
import uuid

import yaml


ROOT: Final = Path(__file__).resolve().parents[3]
CONFIG_PATH: Final = (
    ROOT
    / "configs"
    / "training"
    / "gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_v1.yaml"
)
CONFIG_SCHEMA_VERSION: Final = (
    "anchor.gemma3-1b-it-chat-unbalanced-v2-multiarm-q8-qlora-config.v1"
)
PREFLIGHT_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.sharded-v1"
)
CONSUMER_DERIVED_ASSET_IDENTITY_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-consumer-derived-asset-identity.v1"
)
CONSUMER_DERIVED_SHARD_INVENTORY_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-consumer-derived-shard-inventory.v1"
)
INVENTORY_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-observed-tensor-inventory.v1"
)
INVENTORY_SET_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-observed-tensor-inventory-set.v1"
)
PLAN_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-multiarm-model-free-plan.v1"
)
TRUSTED_EXTERNAL_BINDINGS_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-trusted-external-bindings.v1"
)
PROGRESS_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-multiarm-scalar-progress.v1"
)
PHASE_RECEIPT_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-multiarm-phase-receipt.v1"
)
FULL_GATE_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-multiarm-full-gate.v1"
)

_MAX_CONFIG_BYTES: Final = 1_000_000
_MAX_RECEIPT_BYTES: Final = 2_000_000
_MAX_INVENTORY_BYTES: Final = 8_000_000
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

TRAINED_ARMS: Final = (
    "humor_q_only",
    "serious_q_only",
    "angry_style_q_only",
    "review_audit_q_only",
    "tool_q_only",
    "tool_q_plus_micro_o",
    "planner_q_only",
    "planner_o_only",
    "planner_q_plus_micro_o",
)
Q_ONLY_ARMS: Final = frozenset(
    {
        "humor_q_only",
        "serious_q_only",
        "angry_style_q_only",
        "review_audit_q_only",
        "tool_q_only",
        "planner_q_only",
    }
)
O_ONLY_ARMS: Final = frozenset({"planner_o_only"})
QO_ARMS: Final = frozenset({"tool_q_plus_micro_o", "planner_q_plus_micro_o"})
EVAL_ONLY_ARMS: Final = ("tool_base", "planner_base")
TRAINING_ASSETS: Final = {
    "humor": 240,
    "serious": 240,
    "angry_style": 240,
    "review_audit": 1360,
    "tool_call": 1360,
    "planner_router": 80,
}
EVALUATION_ASSETS: Final = {
    "planner_router_eval": 20,
    "tool_comparison_eval": 400,
    "planner_comparison_eval": 240,
    "identity_probe": 50,
}
FULL_STEPS: Final = {
    "humor_q_only": 240,
    "serious_q_only": 240,
    "angry_style_q_only": 240,
    "review_audit_q_only": 1360,
    "tool_q_only": 1360,
    "tool_q_plus_micro_o": 1360,
    "planner_q_only": 80,
    "planner_o_only": 80,
    "planner_q_plus_micro_o": 80,
}
ARM_SOURCE_ASSET: Final = {
    "humor_q_only": "humor",
    "serious_q_only": "serious",
    "angry_style_q_only": "angry_style",
    "review_audit_q_only": "review_audit",
    "tool_q_only": "tool_call",
    "tool_q_plus_micro_o": "tool_call",
    "planner_q_only": "planner_router",
    "planner_o_only": "planner_router",
    "planner_q_plus_micro_o": "planner_router",
}
ARM_PROFILE: Final = {
    **{arm: "q_only" for arm in Q_ONLY_ARMS},
    **{arm: "o_only" for arm in O_ONLY_ARMS},
    **{arm: "q_plus_micro_o" for arm in QO_ARMS},
}
PROFILE_PARAMETER_COUNTS: Final = {
    "q_only": 57_933_824,
    "o_only": 3_620_864,
    "q_plus_micro_o": 61_554_688,
}
PROFILE_TENSOR_COUNTS: Final = {
    "q_only": 52,
    "o_only": 52,
    "q_plus_micro_o": 104,
}
Q_LR: Final = Decimal("0.000002")
O_LR: Final = Decimal("0.0000002")
BASE_SEED: Final = 1337
SMOKE_STEPS: Final = 2
DEPENDENCIES: Final = (
    (
        "src/anchor_mvp/training/gemma3_chat_five_expert_qonly_v1.py",
        188_223,
        "55a7c45eb8f4b6c590938a486b65f675920d39d46f94686685a04e08f3bde802",
    ),
    (
        "src/anchor_mvp/training/gemma3_five_role_q8_qlora_v2.py",
        109_147,
        "5b7056b0daaa7e5a0774c71c43c1d98dc47de6c831b2251e585745ae89c63a9f",
    ),
    (
        "src/anchor_mvp/training/gemma3_q8_reliability_v2.py",
        5_845,
        "fbf68a9f5a7e40e0211292055527678e657793af89e4e4e404d7083d0ddcb381",
    ),
    (
        "scripts/research/gemma3_q8_launcher_helpers_v2.ps1",
        11_688,
        "88411919f8b3839c037db439363413b70584fa7ca406a1f2b1b84d811bf73b1a",
    ),
)
FINAL_SOURCE_ANCHORS: Final = {
    "artifact_version": (
        "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-sharded.v3"
    ),
    "candidate_commit": "c5080249aa103d6cac09f0a55e51982b68524480",
    "release_commit": "304be2b86f82aad7ddade80fae04528bf2801977",
    "release_tree": "9bf154d481623b696dbc8a4bdf276967710b7214",
    "upstream_commit": "304be2b86f82aad7ddade80fae04528bf2801977",
    "live_remote_commit": "304be2b86f82aad7ddade80fae04528bf2801977",
    "tree_digest_sha256": (
        "25add529328255f6b36f9929be1851f42abd25deda7a7f69733ac8348cad387c"
    ),
    "producer_inventory_sha256": (
        "dff68b505661f21815b6883b75e1d8c26701c1bd9733c1e5912a91b4c3c8c5fc"
    ),
    "manifest": {
        "schema_version": (
            "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-manifest.v4"
        ),
        "sha256": ("c99ba5a7f1e247e713a840f1dfe4710886759b98901e068ba1b70e5eacd2a499"),
        "bytes": 34_236,
    },
    "build_receipt": {
        "schema_version": (
            "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-build-receipt.v4"
        ),
        "sha256": ("da16c9a9cac9dc11f281ca52217f1bcd1d23afab589ab3d1d4c0f1e322b9b7d6"),
        "bytes": 35_356,
    },
    "schema_files": {
        "manifest": {
            "sha256": (
                "9841c0448cbef634de847f9ffa35ab1a67bb27749ff6f684debf48851dd8ad7e"
            ),
            "bytes": 11_062,
        },
        "build_receipt": {
            "sha256": (
                "77d6febe99ddebd66a9409d37e0c69feeb300ccbc00491212ff115d30996816c"
            ),
            "bytes": 9_699,
        },
        "release_attestation": {
            "sha256": (
                "3c23823e21a4754bef99b7cbbcca217eab617bc313a6b75299a9bd844c7e6eb9"
            ),
            "bytes": 11_765,
        },
    },
    "release_attestation": {
        "schema_version": (
            "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-"
            "independent-release-attestation.v3"
        ),
        "sha256": ("b8630a90dedd4c685900968b4c06022ea3d7b505bbbce0e666d4adaecdf3d813"),
        "bytes": 4_971,
        "sidecar_sha256": (
            "f64aa73a2968111fb0148ca58993bfaec92f383f736fbd3496c64d9ff776ba35"
        ),
        "sidecar_bytes": 103,
    },
    "training_shards": {
        "shard_count": 2,
        "ordered_paths": [
            "train/chat-00000-of-00002.jsonl",
            "train/chat-00001-of-00002.jsonl",
        ],
        "ordered_concat_sha256": (
            "62c0f64b6d2f5a0901169480c946d554cc02175138862b3a210d06e44fcbddd3"
        ),
        "logical_partition_bytes": 59_845_314,
        "logical_partition_records": 3_440,
        "task_bundles": 1_360,
        "bundle_boundary_only": True,
        "ordered_shards": True,
        "task_bundle_intersection_empty": True,
        "boundary_commitment_sha256": (
            "6af9bc4cd79c9a8bb211a7147b91eb7c0c07e3adf7026becc1c913045716d089"
        ),
        "shards": [
            {
                "index": 0,
                "path": "train/chat-00000-of-00002.jsonl",
                "sha256": (
                    "9c2fe5ba2b199be12f11f9d9e1a76c2d8de1474732997cb8d30b984f2477b215"
                ),
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
                "path": "train/chat-00001-of-00002.jsonl",
                "sha256": (
                    "1d8facb574a7341d1b43933da30f8ea2c0f977bec5920a83661c0619cb463522"
                ),
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
        ],
    },
    "logical_identity": {
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
    },
}


class MultiArmContractError(RuntimeError):
    """Raised when a model-free multi-arm contract check fails closed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise MultiArmContractError("yaml_duplicate_key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_json(value))


def _require_sha(value: object, code: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise MultiArmContractError(code)
    return value


def _require_commit(value: object, code: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise MultiArmContractError(code)
    return value


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise MultiArmContractError(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MultiArmContractError(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise MultiArmContractError(code)


def _positive_int(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MultiArmContractError(code)
    return value


def _nonnegative_int(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MultiArmContractError(code)
    return value


def _finite_scalar(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiArmContractError(code)
    result = float(value)
    if not math.isfinite(result):
        raise MultiArmContractError(code)
    return result


def _snapshot(path: Path, *, max_bytes: int, code: str) -> tuple[bytes, str]:
    try:
        before = path.lstat()
    except OSError:
        raise MultiArmContractError(code) from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise MultiArmContractError(code)
    if before.st_size < 1 or before.st_size > max_bytes:
        raise MultiArmContractError(code)
    try:
        data = path.read_bytes()
        after = path.lstat()
    except OSError:
        raise MultiArmContractError(code) from None
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(data) != before.st_size:
        raise MultiArmContractError(f"{code}_changed_during_read")
    return data, _sha256(data)


def _repo_file(relative: object, code: str) -> Path:
    if not isinstance(relative, str):
        raise MultiArmContractError(code)
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise MultiArmContractError(code)
    path = ROOT.joinpath(*pure.parts)
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError:
        raise MultiArmContractError(code) from None
    if not resolved_parent.is_relative_to(ROOT.resolve()):
        raise MultiArmContractError(code)
    return path


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    data, digest = _snapshot(
        path,
        max_bytes=_MAX_CONFIG_BYTES,
        code="config_snapshot_invalid",
    )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise MultiArmContractError("config_utf8_invalid") from None
    if "\r" in text:
        raise MultiArmContractError("config_lf_required")
    try:
        value = yaml.load(text, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, MultiArmContractError):
        raise MultiArmContractError("config_yaml_invalid") from None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MultiArmContractError("config_root_invalid")
    return value, digest


def _load_json(
    path: str | Path,
    *,
    max_bytes: int,
    code: str,
) -> tuple[dict[str, Any], str]:
    data, digest = _snapshot(Path(path), max_bytes=max_bytes, code=code)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise MultiArmContractError(f"{code}_json_invalid") from None
    try:

        def reject_constant(value: str) -> object:
            raise MultiArmContractError(f"{code}_nonfinite_number")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise MultiArmContractError(f"{code}_duplicate_key")
                result[key] = value
            return result

        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError:
        raise MultiArmContractError(f"{code}_json_invalid") from None
    if "\r" in text:
        raise MultiArmContractError(f"{code}_lf_required")
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MultiArmContractError(f"{code}_root_invalid")

    def require_finite_json_number(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise MultiArmContractError(f"{code}_nonfinite_number")
        if isinstance(item, dict):
            for nested in item.values():
                require_finite_json_number(nested)
        elif isinstance(item, list):
            for nested in item:
                require_finite_json_number(nested)

    require_finite_json_number(value)
    return value, digest


def _expect_mapping(
    actual: object,
    expected: Mapping[str, Any],
    code: str,
) -> None:
    if actual != expected:
        raise MultiArmContractError(code)


def _validate_dependencies(section: object) -> str:
    dependencies = _mapping(section, "dependencies_invalid")
    _exact_keys(dependencies, {"frozen"}, "dependencies_keys_invalid")
    entries = _sequence(dependencies["frozen"], "dependency_list_invalid")
    expected_entries = [
        {"path": path, "bytes": size, "sha256": digest}
        for path, size, digest in DEPENDENCIES
    ]
    if list(entries) != expected_entries:
        raise MultiArmContractError("dependency_contract_drift")
    observed: list[dict[str, object]] = []
    for relative, expected_bytes, expected_sha in DEPENDENCIES:
        path = _repo_file(relative, "dependency_path_invalid")
        data, digest = _snapshot(
            path,
            max_bytes=max(expected_bytes, 1),
            code="dependency_snapshot_invalid",
        )
        if len(data) != expected_bytes or digest != expected_sha:
            raise MultiArmContractError("dependency_physical_identity_mismatch")
        observed.append({"path": relative, "bytes": len(data), "sha256": digest})
    return _canonical_sha256(observed)


def validate_config(config: Mapping[str, Any]) -> str:
    """Validate the closed model-free runner contract and frozen dependencies."""

    _exact_keys(
        config,
        {
            "schema_version",
            "status",
            "claim_scope",
            "dependencies",
            "consumer_preflight",
            "model",
            "quantization",
            "data",
            "adapter_profiles",
            "initialization",
            "optimizer",
            "arms",
            "comparison",
            "training",
            "gates",
            "output",
            "claims",
        },
        "config_keys_invalid",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA_VERSION
        or config["status"] != "model_free_contract_ready_gpu_runtime_not_implemented"
        or config["claim_scope"] != "diagnostic_proxy_multiarm_training_plan_only"
    ):
        raise MultiArmContractError("config_identity_invalid")

    _expect_mapping(
        config["consumer_preflight"],
        {
            "required_schema_version": PREFLIGHT_SCHEMA_VERSION,
            "required_status": "passed",
            "required_namespace": "gemma3_chat_five_expert_qonly_unbalanced_v2",
            "final_source_anchors_hardcoded": True,
            "require_versioned_shard_inventory": True,
            "require_consumer_derived_logical_assets": True,
            "local_user_task_authorization_required": True,
            "producer_training_authority_must_not_be_inferred": True,
            "required_physical_files": 46,
            "required_payload_files": 23,
            "required_sidecar_files": 23,
            "required_main_chat_train_shards": 2,
            "required_train_shard_order": "continuous_bundle_boundary_v1",
            "require_final_producer_release": True,
            "require_body_read": False,
            "require_raw_token_ids_read": False,
            "require_gold_heldout_protected_read": False,
            "final_source_anchors": FINAL_SOURCE_ANCHORS,
        },
        "consumer_preflight_contract_invalid",
    )
    _expect_mapping(
        config["model"],
        {
            "architecture": "Gemma3ForCausalLM",
            "layers": 26,
            "hidden_size": 1152,
            "q_proj_input_features": 1152,
            "q_proj_output_features": 1024,
            "o_proj_input_features": 1024,
            "o_proj_output_features": 1152,
            "local_files_only": True,
            "allow_network": False,
            "trust_remote_code": False,
        },
        "model_contract_invalid",
    )
    _expect_mapping(
        config["quantization"],
        {
            "frozen_base": True,
            "backend": "bitsandbytes",
            "load_in_8bit": True,
            "base_weight_bits": 8,
            "adapter_dtype": "bfloat16",
            "base_o_proj_always_frozen": True,
        },
        "quantization_contract_invalid",
    )

    data = _mapping(config["data"], "data_contract_invalid")
    _exact_keys(
        data,
        {
            "training_assets",
            "evaluation_assets",
            "chat_train_records",
            "planner_router_total_records",
            "single_training_partition_assumed",
            "optimizer_forbidden_assets",
        },
        "data_contract_keys_invalid",
    )
    training_assets = _mapping(data["training_assets"], "training_assets_invalid")
    if set(training_assets) != set(TRAINING_ASSETS):
        raise MultiArmContractError("training_asset_names_invalid")
    for name, count in TRAINING_ASSETS.items():
        if training_assets[name] != {"records": count}:
            raise MultiArmContractError("training_asset_count_invalid")
    evaluation_assets = _mapping(
        data["evaluation_assets"],
        "evaluation_assets_invalid",
    )
    if set(evaluation_assets) != set(EVALUATION_ASSETS):
        raise MultiArmContractError("evaluation_asset_names_invalid")
    for name, count in EVALUATION_ASSETS.items():
        if evaluation_assets[name] != {
            "records": count,
            "training_eligible": False,
        }:
            raise MultiArmContractError("evaluation_asset_contract_invalid")
    if (
        data["chat_train_records"] != 3440
        or data["planner_router_total_records"] != 100
        or data["single_training_partition_assumed"] is not False
        or list(data["optimizer_forbidden_assets"]) != list(EVALUATION_ASSETS)
    ):
        raise MultiArmContractError("data_aggregate_contract_invalid")

    profiles = _mapping(config["adapter_profiles"], "adapter_profiles_invalid")
    _exact_keys(
        profiles,
        {"q_only", "o_only", "q_plus_micro_o", "base"},
        "adapter_profile_names_invalid",
    )
    _expect_mapping(
        profiles["q_only"],
        {
            "target_modules": ["q_proj"],
            "rank": 1024,
            "alpha": 2048,
            "dropout": 0.0,
            "expected_parameter_tensors": 52,
            "expected_trainable_parameters": 57_933_824,
            "A_initializer": "deterministic_kaiming_uniform_v1",
            "B_initializer": "zero_v1",
        },
        "q_profile_invalid",
    )
    _expect_mapping(
        profiles["o_only"],
        {
            "target_modules": ["o_proj"],
            "rank": 64,
            "alpha": 128,
            "dropout": 0.0,
            "expected_parameter_tensors": 52,
            "expected_trainable_parameters": 3_620_864,
            "A_initializer": "deterministic_kaiming_uniform_v1",
            "B_initializer": "zero_v1",
        },
        "o_profile_invalid",
    )
    _expect_mapping(
        profiles["q_plus_micro_o"],
        {
            "target_modules": ["q_proj", "o_proj"],
            "q_rank": 1024,
            "q_alpha": 2048,
            "o_rank": 64,
            "o_alpha": 128,
            "dropout": 0.0,
            "expected_parameter_tensors": 104,
            "expected_trainable_parameters": 61_554_688,
            "A_initializer": "deterministic_kaiming_uniform_v1",
            "B_initializer": "zero_v1",
        },
        "qo_profile_invalid",
    )
    _expect_mapping(
        profiles["base"],
        {
            "target_modules": [],
            "expected_parameter_tensors": 0,
            "expected_trainable_parameters": 0,
            "adapter": None,
            "optimizer": None,
        },
        "base_profile_invalid",
    )
    _expect_mapping(
        config["initialization"],
        {
            "base_seed": BASE_SEED,
            "branch_seed_derivation": "base_seed_only_v1",
            "forbidden_seed_components": ["arm", "phase", "execution_order"],
            "paired_actual_tensor_bytes_required": True,
        },
        "initialization_contract_invalid",
    )
    _expect_mapping(
        config["optimizer"],
        {
            "name": "adamw8bit",
            "q_learning_rate": float(Q_LR),
            "o_learning_rate": float(O_LR),
            "exact_o_to_q_ratio": 0.1,
            "explicit_parameter_groups": True,
            "group_order": ["Q", "O"],
            "groups_mutually_exclusive": True,
            "groups_cover_all_trainable_tensors": True,
            "warmup_steps": 8,
            "schedule": "linear_warmup_then_constant_v1",
            "weight_decay": 0.01,
            "max_grad_norm": 0.5,
            "optimizer_state_bits": 8,
        },
        "optimizer_contract_invalid",
    )

    arms = _mapping(config["arms"], "arms_contract_invalid")
    _exact_keys(arms, {"ordered", "definitions", "eval_only"}, "arms_keys_invalid")
    if tuple(arms["ordered"]) != TRAINED_ARMS:
        raise MultiArmContractError("arm_order_invalid")
    definitions = _mapping(arms["definitions"], "arm_definitions_invalid")
    if set(definitions) != set(TRAINED_ARMS):
        raise MultiArmContractError("arm_definition_names_invalid")
    for arm in TRAINED_ARMS:
        expected = {
            "source_asset": ARM_SOURCE_ASSET[arm],
            "adapter_profile": ARM_PROFILE[arm],
            "full_steps": FULL_STEPS[arm],
            "comparison_group": (
                "tool"
                if arm.startswith("tool_")
                else "planner"
                if arm.startswith("planner_")
                else arm.removesuffix("_q_only")
            ),
            "production_router": arm == "planner_q_only",
        }
        if definitions[arm] != expected:
            raise MultiArmContractError("arm_definition_invalid")
    _expect_mapping(
        arms["eval_only"],
        {
            "tool_base": {
                "source_asset": "tool_comparison_eval",
                "steps": 0,
                "adapter": None,
                "optimizer": None,
            },
            "planner_base": {
                "source_asset": "planner_comparison_eval",
                "steps": 0,
                "adapter": None,
                "optimizer": None,
            },
        },
        "eval_only_arm_contract_invalid",
    )

    _expect_mapping(
        config["comparison"],
        {
            "tool": {
                "eval_asset": "tool_comparison_eval",
                "records": 400,
                "ordered_arms": [
                    "tool_base",
                    "tool_q_only",
                    "tool_q_plus_micro_o",
                ],
                "total_slots": 1200,
            },
            "planner": {
                "eval_asset": "planner_comparison_eval",
                "records": 240,
                "ordered_arms": [
                    "planner_base",
                    "planner_q_only",
                    "planner_o_only",
                    "planner_q_plus_micro_o",
                ],
                "total_slots": 960,
            },
            "paired_lock_fields": [
                "record_order_sha256",
                "target_projection_sha256",
                "branch_seed_sha256",
                "steps",
                "optimizer_schedule_sha256",
            ],
        },
        "comparison_contract_invalid",
    )
    _expect_mapping(
        config["training"],
        {
            "smoke_steps_per_arm": 2,
            "full_fresh_from_base": True,
            "full_fresh_adapter": True,
            "resume": False,
            "smoke_checkpoint_consumed_by_full": False,
            "strictly_serial": True,
            "concurrency": 1,
            "sequence_length": 768,
            "truncation": False,
            "use_cache": False,
        },
        "training_contract_invalid",
    )
    _expect_mapping(
        config["gates"],
        {
            "smoke_pass_required_before_full": True,
            "required_phase_gates": [
                "base_hash_unchanged",
                "finite_loss",
                "finite_gradients",
                "optimizer_state_verified",
                "adapter_effect_nonzero",
                "memory_gate_passed",
                "canonical_lock_verified",
                "atomic_publish_verified",
            ],
            "scalar_only_progress": True,
            "body_free_receipts": True,
        },
        "gate_contract_invalid",
    )
    _expect_mapping(
        config["output"],
        {
            "artifact_root": (
                "artifacts/diagnostics/"
                "gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_v1"
            ),
            "run_root": ("runs/gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_v1"),
            "progress_directory": "progress",
            "phase_receipt_filename": "phase_receipt.json",
            "run_receipt_filename": "run_receipt.json",
            "atomic_publish": True,
            "replace_existing": False,
            "failure_receipt_retained": True,
        },
        "output_contract_invalid",
    )
    _expect_mapping(
        config["claims"],
        {
            "diagnostic_only": True,
            "proxy_only": True,
            "model_free_contract": True,
            "gpu_execution_implemented": False,
            "model_loaded": False,
            "training_executed": False,
            "evaluation_executed": False,
            "formal": False,
            "release": False,
        },
        "claim_contract_invalid",
    )
    return _validate_dependencies(config["dependencies"])


def load_config(
    path: str | Path = CONFIG_PATH,
) -> tuple[dict[str, Any], str, str]:
    config, config_sha256 = _load_yaml(Path(path))
    dependency_set_sha256 = validate_config(config)
    return config, config_sha256, dependency_set_sha256


def _validate_logical_asset(
    value: object,
    *,
    expected_records: int,
    training_eligible: bool,
    require_shard_order: bool,
    expected_shard_order: str | None = None,
) -> dict[str, Any]:
    asset = _mapping(value, "preflight_asset_invalid")
    expected_keys = {
        "records",
        "asset_namespace",
        "identity_origin",
        "identity_derivation_schema_version",
        "identity_preimage_sha256",
        "logical_dataset_sha256",
        "shard_inventory_schema_version",
        "shard_inventory_schema_sha256",
        "shard_inventory_sha256",
        "shard_count",
        "training_eligible",
        "record_order_sha256",
        "target_projection_sha256",
    }
    if require_shard_order:
        expected_keys.add("shard_order_contract")
    _exact_keys(asset, expected_keys, "preflight_asset_keys_invalid")
    if (
        asset["records"] != expected_records
        or asset["training_eligible"] is not training_eligible
        or not isinstance(asset["asset_namespace"], str)
        or not asset["asset_namespace"]
        or asset["identity_origin"]
        != "consumer_derived_from_authenticated_sharded_v1_receipt"
        or asset["identity_derivation_schema_version"]
        != CONSUMER_DERIVED_ASSET_IDENTITY_VERSION
        or not isinstance(asset["shard_inventory_schema_version"], str)
        or asset["shard_inventory_schema_version"]
        != CONSUMER_DERIVED_SHARD_INVENTORY_VERSION
    ):
        raise MultiArmContractError("preflight_asset_contract_invalid")
    result = dict(asset)
    result["identity_preimage_sha256"] = _require_sha(
        asset["identity_preimage_sha256"],
        "identity_preimage_sha256_invalid",
    )
    result["logical_dataset_sha256"] = _require_sha(
        asset["logical_dataset_sha256"],
        "logical_dataset_sha256_invalid",
    )
    result["shard_inventory_sha256"] = _require_sha(
        asset["shard_inventory_sha256"],
        "shard_inventory_sha256_invalid",
    )
    result["shard_inventory_schema_sha256"] = _require_sha(
        asset["shard_inventory_schema_sha256"],
        "shard_inventory_schema_sha256_invalid",
    )
    result["shard_count"] = _positive_int(
        asset["shard_count"],
        "shard_count_invalid",
    )
    result["record_order_sha256"] = _require_sha(
        asset["record_order_sha256"],
        "record_order_sha256_invalid",
    )
    result["target_projection_sha256"] = _require_sha(
        asset["target_projection_sha256"],
        "target_projection_sha256_invalid",
    )
    if require_shard_order:
        shard_order = asset["shard_order_contract"]
        if (
            not isinstance(shard_order, str)
            or re.fullmatch(r"^[a-z][a-z0-9._-]{2,127}$", shard_order) is None
            or (
                expected_shard_order is not None and shard_order != expected_shard_order
            )
        ):
            raise MultiArmContractError("shard_order_contract_invalid")
    return result


def _domain_canonical_sha256(domain: str, value: object) -> str:
    if re.fullmatch(r"^anchor\.[a-z0-9._-]+\.v[0-9]+$", domain) is None:
        raise MultiArmContractError("consumer_derived_hash_domain_invalid")
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + _canonical_json(value)
    ).hexdigest()


def _receipt_training_shard_contract() -> dict[str, Any]:
    anchors = _mapping(
        FINAL_SOURCE_ANCHORS["training_shards"],
        "final_training_shard_anchors_invalid",
    )
    keys = (
        "shard_count",
        "ordered_paths",
        "ordered_concat_sha256",
        "logical_partition_bytes",
        "logical_partition_records",
        "task_bundles",
        "bundle_boundary_only",
        "ordered_shards",
        "task_bundle_intersection_empty",
        "boundary_commitment_sha256",
    )
    return {key: anchors[key] for key in keys}


def _validate_identity_mapping(
    value: object,
    expected: Mapping[str, Any],
    *,
    code: str,
) -> dict[str, Any]:
    identity = _mapping(value, code)
    _exact_keys(identity, set(expected), f"{code}_keys_invalid")
    if identity != expected:
        raise MultiArmContractError(code)
    return dict(identity)


def _validate_source_receipt(value: object) -> dict[str, Any]:
    receipt = _mapping(value, "consumer_preflight_receipt_invalid")
    source_keys = {
        "schema_version",
        "status",
        "operation",
        "namespace",
        "model_free",
        "binding_contract_sha256",
        "artifact_version",
        "producer_git",
        "tree_digest_sha256",
        "file_counts",
        "manifest",
        "build_receipt",
        "schema_files",
        "release_attestation",
        "training_shards",
        "logical_identity",
        "read_set",
        "release",
        "terminal_recheck",
        "resource_counters",
        "claims",
    }
    _exact_keys(
        receipt,
        source_keys,
        "consumer_preflight_receipt_keys_invalid",
    )
    if (
        receipt["schema_version"] != PREFLIGHT_SCHEMA_VERSION
        or receipt["status"] != "passed"
        or receipt["operation"] not in {"validate", "dry-run"}
        or receipt["namespace"] != "gemma3_chat_five_expert_qonly_unbalanced_v2"
        or receipt["model_free"] is not True
        or receipt["artifact_version"] != FINAL_SOURCE_ANCHORS["artifact_version"]
    ):
        raise MultiArmContractError("consumer_preflight_not_passed")
    _require_sha(
        receipt["binding_contract_sha256"],
        "binding_contract_sha256_invalid",
    )
    if receipt["tree_digest_sha256"] != FINAL_SOURCE_ANCHORS["tree_digest_sha256"]:
        raise MultiArmContractError("consumer_preflight_tree_identity_drift")

    _validate_identity_mapping(
        receipt["producer_git"],
        {
            "candidate_commit": FINAL_SOURCE_ANCHORS["candidate_commit"],
            "release_commit": FINAL_SOURCE_ANCHORS["release_commit"],
            "release_tree": FINAL_SOURCE_ANCHORS["release_tree"],
            "upstream_commit": FINAL_SOURCE_ANCHORS["upstream_commit"],
            "live_remote_commit": FINAL_SOURCE_ANCHORS["live_remote_commit"],
            "clean_worktree": True,
            "tags_at_head": 0,
        },
        code="consumer_preflight_producer_git_identity_drift",
    )
    _expect_mapping(
        receipt["file_counts"],
        {"total": 46, "payload": 23, "sidecar": 23},
        "consumer_preflight_file_counts_invalid",
    )
    _validate_identity_mapping(
        receipt["manifest"],
        _mapping(FINAL_SOURCE_ANCHORS["manifest"], "manifest_anchor_invalid"),
        code="consumer_preflight_manifest_identity_drift",
    )
    _validate_identity_mapping(
        receipt["build_receipt"],
        _mapping(
            FINAL_SOURCE_ANCHORS["build_receipt"],
            "build_receipt_anchor_invalid",
        ),
        code="consumer_preflight_build_receipt_identity_drift",
    )
    schema_files = _mapping(
        receipt["schema_files"],
        "consumer_preflight_schema_files_invalid",
    )
    expected_schema_files = _mapping(
        FINAL_SOURCE_ANCHORS["schema_files"],
        "schema_file_anchors_invalid",
    )
    _exact_keys(
        schema_files,
        set(expected_schema_files),
        "consumer_preflight_schema_file_names_invalid",
    )
    if schema_files != expected_schema_files:
        raise MultiArmContractError("consumer_preflight_schema_identity_drift")
    _validate_identity_mapping(
        receipt["release_attestation"],
        _mapping(
            FINAL_SOURCE_ANCHORS["release_attestation"],
            "release_attestation_anchor_invalid",
        ),
        code="consumer_preflight_attestation_identity_drift",
    )
    _expect_mapping(
        receipt["training_shards"],
        _receipt_training_shard_contract(),
        "consumer_preflight_training_shards_drift",
    )
    _validate_identity_mapping(
        receipt["logical_identity"],
        _mapping(
            FINAL_SOURCE_ANCHORS["logical_identity"],
            "logical_identity_anchors_invalid",
        ),
        code="consumer_preflight_logical_identity_drift",
    )
    read_set = _mapping(receipt["read_set"], "consumer_preflight_read_set_invalid")
    _exact_keys(
        read_set,
        {
            "count",
            "canonical_digest_sha256",
            "producer_inventory_sha256",
            "physical_identities_equal",
            "utf8_lf",
        },
        "consumer_preflight_read_set_keys_invalid",
    )
    _positive_int(read_set["count"], "consumer_preflight_read_set_count_invalid")
    _require_sha(
        read_set["canonical_digest_sha256"],
        "consumer_preflight_read_set_digest_invalid",
    )
    if (
        read_set["producer_inventory_sha256"]
        != FINAL_SOURCE_ANCHORS["producer_inventory_sha256"]
        or read_set["physical_identities_equal"] is not True
        or read_set["utf8_lf"] is not True
    ):
        raise MultiArmContractError("consumer_preflight_read_set_identity_drift")
    _expect_mapping(
        receipt["release"],
        {
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
        "consumer_preflight_release_not_final",
    )
    _expect_mapping(
        receipt["terminal_recheck"],
        {
            "two_artifact_snapshots_equal": True,
            "artifact_stat_identity_equal": True,
            "external_single_read_stat_identity_equal": True,
            "physical_inventory_equal": True,
        },
        "consumer_preflight_terminal_recheck_invalid",
    )
    _expect_mapping(
        receipt["resource_counters"],
        {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "gold_body_reads": 0,
            "heldout_body_reads": 0,
            "protected_body_reads": 0,
        },
        "consumer_preflight_resource_counter_invalid",
    )
    _expect_mapping(
        receipt["claims"],
        {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "training_started": False,
            "data_copied": False,
            "sample_bodies_parsed": False,
            "raw_token_ids_parsed": False,
            "producer_training_authority_inferred": False,
        },
        "consumer_preflight_claims_not_launchable",
    )
    return dict(receipt)


def _derive_consumer_assets(
    source: Mapping[str, Any],
    physical_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_sha = _require_sha(
        physical_receipt_sha256,
        "consumer_preflight_receipt_sha256_invalid",
    )
    common = {
        "identity_derivation_schema_version": (CONSUMER_DERIVED_ASSET_IDENTITY_VERSION),
        "physical_receipt_sha256": receipt_sha,
        "artifact_version": source["artifact_version"],
        "tree_digest_sha256": source["tree_digest_sha256"],
        "logical_identity": source["logical_identity"],
    }
    schema_contract = {
        "schema_version": CONSUMER_DERIVED_SHARD_INVENTORY_VERSION,
        "identity_origin": ("consumer_derived_from_authenticated_sharded_v1_receipt"),
        "fields": [
            "physical_receipt_sha256",
            "artifact_version",
            "tree_digest_sha256",
            "logical_identity",
            "asset_name",
            "asset_namespace",
            "records",
            "training_eligible",
        ],
    }
    shard_schema_sha = _domain_canonical_sha256(
        "anchor.consumer-derived-shard-inventory-schema.v1",
        schema_contract,
    )
    main_chat_shard_preimage = {
        **common,
        "training_shards": source["training_shards"],
        "physical_training_shard_anchors": _mapping(
            FINAL_SOURCE_ANCHORS["training_shards"],
            "final_training_shard_anchors_invalid",
        )["shards"],
    }
    main_chat_shard_sha = _domain_canonical_sha256(
        "anchor.consumer-derived-main-chat-shard-inventory.v1",
        main_chat_shard_preimage,
    )
    namespaces = {
        "humor": "train/chat#role=humor",
        "serious": "train/chat#role=serious",
        "angry_style": "train/chat#role=angry_style",
        "review_audit": "train/chat#role=review_audit",
        "tool_call": "train/chat#role=tool_call",
        "planner_router": "components/router#split=train",
        "planner_router_eval": "components/router#split=eval_proxy",
        "tool_comparison_eval": "components/tool_eval#split=all",
        "planner_comparison_eval": "components/planner_eval#split=all",
        "identity_probe": "identity_eval/probe_inventory#split=all",
    }
    main_chat_names = {
        "humor",
        "serious",
        "angry_style",
        "review_audit",
        "tool_call",
    }

    def derive(name: str, records: int, training_eligible: bool) -> dict[str, Any]:
        preimage = {
            **common,
            "asset_name": name,
            "asset_namespace": namespaces[name],
            "records": records,
            "training_eligible": training_eligible,
        }
        if name in main_chat_names:
            shard_inventory_sha = main_chat_shard_sha
            shard_count = 2
        else:
            shard_inventory_sha = _domain_canonical_sha256(
                "anchor.consumer-derived-tree-asset-shard-inventory.v1",
                preimage,
            )
            shard_count = 1
        asset = {
            "records": records,
            "asset_namespace": namespaces[name],
            "identity_origin": (
                "consumer_derived_from_authenticated_sharded_v1_receipt"
            ),
            "identity_derivation_schema_version": (
                CONSUMER_DERIVED_ASSET_IDENTITY_VERSION
            ),
            "identity_preimage_sha256": _domain_canonical_sha256(
                "anchor.consumer-derived-asset-preimage.v1",
                preimage,
            ),
            "logical_dataset_sha256": _domain_canonical_sha256(
                "anchor.consumer-derived-logical-dataset.v1",
                preimage,
            ),
            "shard_inventory_schema_version": (
                CONSUMER_DERIVED_SHARD_INVENTORY_VERSION
            ),
            "shard_inventory_schema_sha256": shard_schema_sha,
            "shard_inventory_sha256": shard_inventory_sha,
            "shard_count": shard_count,
            "training_eligible": training_eligible,
            "record_order_sha256": _domain_canonical_sha256(
                "anchor.consumer-derived-record-order.v1",
                preimage,
            ),
            "target_projection_sha256": _domain_canonical_sha256(
                "anchor.consumer-derived-target-projection.v1",
                preimage,
            ),
        }
        if training_eligible:
            asset["shard_order_contract"] = (
                "continuous_bundle_boundary_v1"
                if name in main_chat_names
                else "consumer_derived_tree_asset_namespace_v1"
            )
        return asset

    training = {
        name: derive(name, count, True) for name, count in TRAINING_ASSETS.items()
    }
    evaluation = {
        name: derive(name, count, False) for name, count in EVALUATION_ASSETS.items()
    }
    return training, evaluation


def load_consumer_preflight_receipt(
    path: str | Path,
) -> tuple[dict[str, Any], str, str]:
    """Authenticate the dynamic producer/consumer binding without data reads."""

    receipt, receipt_sha256 = _load_json(
        path,
        max_bytes=_MAX_RECEIPT_BYTES,
        code="consumer_preflight_receipt_invalid",
    )
    source = _validate_source_receipt(receipt)
    normalized_training, normalized_evaluation = _derive_consumer_assets(
        source,
        receipt_sha256,
    )
    normalized = {
        **source,
        "physical_receipt_sha256": receipt_sha256,
        "training_assets": normalized_training,
        "evaluation_assets": normalized_evaluation,
    }
    normalized = _validate_normalized_preflight_binding(normalized)
    source_binding_sha256 = _canonical_sha256(normalized)
    return normalized, receipt_sha256, source_binding_sha256


def _validate_normalized_preflight_binding(
    value: object,
) -> dict[str, Any]:
    """Re-authenticate the body-free binding carried by a trusted context."""

    binding = _mapping(value, "trusted_preflight_binding_invalid")
    source_keys = {
        "schema_version",
        "status",
        "operation",
        "namespace",
        "model_free",
        "binding_contract_sha256",
        "artifact_version",
        "producer_git",
        "tree_digest_sha256",
        "file_counts",
        "manifest",
        "build_receipt",
        "schema_files",
        "release_attestation",
        "training_shards",
        "logical_identity",
        "read_set",
        "release",
        "terminal_recheck",
        "resource_counters",
        "claims",
    }
    _exact_keys(
        binding,
        source_keys
        | {
            "physical_receipt_sha256",
            "training_assets",
            "evaluation_assets",
        },
        "trusted_preflight_binding_keys_invalid",
    )
    source = _validate_source_receipt({key: binding[key] for key in source_keys})
    receipt_sha = _require_sha(
        binding["physical_receipt_sha256"],
        "trusted_physical_receipt_sha256_invalid",
    )
    expected_training, expected_evaluation = _derive_consumer_assets(
        source,
        receipt_sha,
    )
    training = _mapping(binding["training_assets"], "trusted_training_assets_invalid")
    evaluation = _mapping(
        binding["evaluation_assets"],
        "trusted_evaluation_assets_invalid",
    )
    if set(training) != set(TRAINING_ASSETS):
        raise MultiArmContractError("trusted_training_asset_names_invalid")
    if set(evaluation) != set(EVALUATION_ASSETS):
        raise MultiArmContractError("trusted_evaluation_asset_names_invalid")
    normalized_training = {
        name: _validate_logical_asset(
            training[name],
            expected_records=count,
            training_eligible=True,
            require_shard_order=True,
            expected_shard_order=(
                "continuous_bundle_boundary_v1"
                if name != "planner_router"
                else "consumer_derived_tree_asset_namespace_v1"
            ),
        )
        for name, count in TRAINING_ASSETS.items()
    }
    normalized_evaluation = {
        name: _validate_logical_asset(
            evaluation[name],
            expected_records=count,
            training_eligible=False,
            require_shard_order=False,
        )
        for name, count in EVALUATION_ASSETS.items()
    }
    if (
        normalized_training != expected_training
        or normalized_evaluation != expected_evaluation
    ):
        raise MultiArmContractError("trusted_consumer_derived_asset_identity_drift")
    main_chat_assets = (
        "humor",
        "serious",
        "angry_style",
        "review_audit",
        "tool_call",
    )
    main_chat = [normalized_training[name] for name in main_chat_assets]
    if (
        {asset["shard_count"] for asset in main_chat} != {2}
        or len({asset["shard_inventory_sha256"] for asset in main_chat}) != 1
        or len({asset["shard_inventory_schema_sha256"] for asset in main_chat}) != 1
        or len({asset["shard_inventory_schema_version"] for asset in main_chat}) != 1
        or {asset["shard_order_contract"] for asset in main_chat}
        != {"continuous_bundle_boundary_v1"}
    ):
        raise MultiArmContractError("trusted_main_chat_shard_layout_mismatch")
    normalized = {
        **source,
        "physical_receipt_sha256": receipt_sha,
    }
    normalized["training_assets"] = normalized_training
    normalized["evaluation_assets"] = normalized_evaluation
    return normalized


def build_trusted_external_bindings(
    preflight: Mapping[str, Any],
    *,
    preflight_receipt_sha256: str,
    source_binding_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Build an independently supplied trust context from authenticated bytes."""

    receipt_sha = _require_sha(
        preflight_receipt_sha256,
        "trusted_preflight_receipt_sha256_invalid",
    )
    source_sha = _require_sha(
        source_binding_sha256,
        "trusted_source_binding_sha256_invalid",
    )
    normalized = _validate_normalized_preflight_binding(preflight)
    if normalized["physical_receipt_sha256"] != receipt_sha:
        raise MultiArmContractError("trusted_physical_receipt_identity_mismatch")
    if _canonical_sha256(normalized) != source_sha:
        raise MultiArmContractError("trusted_source_binding_digest_mismatch")
    context = {
        "schema_version": TRUSTED_EXTERNAL_BINDINGS_SCHEMA_VERSION,
        "consumer_preflight_receipt_sha256": receipt_sha,
        "source_binding_sha256": source_sha,
        "binding": normalized,
    }
    return context, _canonical_sha256(context)


def validate_trusted_external_bindings(
    value: object,
    trusted_external_bindings_sha256: str,
) -> dict[str, Any]:
    """Validate a caller-supplied trust anchor that is not derived from a plan."""

    anchor = _require_sha(
        trusted_external_bindings_sha256,
        "trusted_external_bindings_sha256_invalid",
    )
    context = _mapping(value, "trusted_external_bindings_invalid")
    _exact_keys(
        context,
        {
            "schema_version",
            "consumer_preflight_receipt_sha256",
            "source_binding_sha256",
            "binding",
        },
        "trusted_external_bindings_keys_invalid",
    )
    if context["schema_version"] != TRUSTED_EXTERNAL_BINDINGS_SCHEMA_VERSION:
        raise MultiArmContractError("trusted_external_bindings_schema_invalid")
    _require_sha(
        context["consumer_preflight_receipt_sha256"],
        "trusted_external_preflight_receipt_sha256_invalid",
    )
    source_sha = _require_sha(
        context["source_binding_sha256"],
        "trusted_external_source_binding_sha256_invalid",
    )
    if _canonical_sha256(context) != anchor:
        raise MultiArmContractError("trusted_external_bindings_anchor_mismatch")
    normalized = _validate_normalized_preflight_binding(context["binding"])
    if (
        normalized["physical_receipt_sha256"]
        != context["consumer_preflight_receipt_sha256"]
    ):
        raise MultiArmContractError("trusted_external_receipt_identity_mismatch")
    if _canonical_sha256(normalized) != source_sha:
        raise MultiArmContractError("trusted_external_source_binding_mismatch")
    return normalized


def _dataset_binding(
    value: object,
    *,
    require_shard_order: bool,
) -> dict[str, Any]:
    asset = _mapping(value, "dataset_binding_source_invalid")
    keys = (
        "records",
        "asset_namespace",
        "identity_origin",
        "identity_derivation_schema_version",
        "identity_preimage_sha256",
        "logical_dataset_sha256",
        "shard_inventory_schema_version",
        "shard_inventory_schema_sha256",
        "shard_inventory_sha256",
        "shard_count",
        "training_eligible",
        "record_order_sha256",
        "target_projection_sha256",
    )
    result = {key: asset[key] for key in keys}
    if require_shard_order:
        result["shard_order_contract"] = asset["shard_order_contract"]
    return result


def _tensor_seed_sha256(name: str) -> str:
    return _canonical_sha256(
        {
            "base_seed": BASE_SEED,
            "tensor_name": name,
            "derivation": "per_tensor_name_v1",
        }
    )


@lru_cache(maxsize=16)
def zero_tensor_sha256(nbytes: int) -> str:
    """Return the digest of ``nbytes`` zero bytes without a large allocation."""

    remaining = _positive_int(nbytes, "zero_tensor_nbytes_invalid")
    digest = hashlib.sha256()
    chunk = b"\0" * min(1_048_576, remaining)
    while remaining:
        count = min(len(chunk), remaining)
        digest.update(chunk[:count])
        remaining -= count
    return digest.hexdigest()


def _expected_tensor_entries(profile: str) -> list[dict[str, Any]]:
    targets: tuple[str, ...]
    if profile == "q_only":
        targets = ("q_proj",)
    elif profile == "o_only":
        targets = ("o_proj",)
    elif profile == "q_plus_micro_o":
        targets = ("q_proj", "o_proj")
    else:
        raise MultiArmContractError("tensor_profile_invalid")
    entries: list[dict[str, Any]] = []
    for target in targets:
        for layer in range(26):
            prefix = f"model.layers.{layer}.self_attn.{target}"
            if target == "q_proj":
                shapes = {
                    "A": [1024, 1152],
                    "B": [1024, 1024],
                }
            else:
                shapes = {
                    "A": [64, 1024],
                    "B": [1152, 64],
                }
            for factor in ("A", "B"):
                shape = shapes[factor]
                numel = math.prod(shape)
                name = f"{prefix}.lora_{factor}.weight"
                entries.append(
                    {
                        "name": name,
                        "target_module": target,
                        "factor": factor,
                        "shape": shape,
                        "numel": numel,
                        "nbytes": numel * 2,
                        "dtype": "bfloat16",
                        "requires_grad": True,
                        "initializer": (
                            "deterministic_kaiming_uniform_v1"
                            if factor == "A"
                            else "zero_v1"
                        ),
                        "initializer_seed_sha256": _tensor_seed_sha256(name),
                    }
                )
    return entries


def expected_tensor_metadata(arm: str) -> list[dict[str, Any]]:
    """Expose immutable expected shapes for model-free inventory producers."""

    if arm not in TRAINED_ARMS:
        raise MultiArmContractError("arm_invalid")
    return [dict(entry) for entry in _expected_tensor_entries(ARM_PROFILE[arm])]


def expected_optimizer_groups(
    arm: str,
    tensor_names: Sequence[str],
) -> list[dict[str, Any]]:
    profile = ARM_PROFILE.get(arm)
    if profile is None:
        raise MultiArmContractError("optimizer_arm_invalid")
    q_names = [name for name in tensor_names if ".q_proj." in name]
    o_names = [name for name in tensor_names if ".o_proj." in name]
    groups: list[dict[str, Any]] = []
    if profile in {"q_only", "q_plus_micro_o"}:
        groups.append(
            {
                "name": "Q",
                "learning_rate": float(Q_LR),
                "tensor_names": q_names,
            }
        )
    if profile in {"o_only", "q_plus_micro_o"}:
        groups.append(
            {
                "name": "O",
                "learning_rate": float(O_LR),
                "tensor_names": o_names,
            }
        )
    return groups


def _validate_optimizer_groups(
    arm: str,
    groups_value: object,
    tensor_names: Sequence[str],
) -> None:
    groups = _sequence(groups_value, "optimizer_groups_invalid")
    expected = expected_optimizer_groups(arm, tensor_names)
    if list(groups) != expected:
        raise MultiArmContractError("optimizer_group_scope_or_order_invalid")
    flattened: list[str] = []
    for group in groups:
        mapping = _mapping(group, "optimizer_group_invalid")
        _exact_keys(
            mapping,
            {"name", "learning_rate", "tensor_names"},
            "optimizer_group_keys_invalid",
        )
        flattened.extend(mapping["tensor_names"])
    if len(flattened) != len(set(flattened)) or set(flattened) != set(tensor_names):
        raise MultiArmContractError("optimizer_groups_not_disjoint_full_coverage")
    qo_groups = [group for group in groups if group["name"] in {"Q", "O"}]
    if len(qo_groups) == 2:
        q_lr = Decimal(str(qo_groups[0]["learning_rate"]))
        o_lr = Decimal(str(qo_groups[1]["learning_rate"]))
        if qo_groups[0]["name"] != "Q" or qo_groups[1]["name"] != "O":
            raise MultiArmContractError("optimizer_group_order_invalid")
        if o_lr * Decimal(10) != q_lr:
            raise MultiArmContractError("optimizer_o_lr_not_exact_tenth")


def _validate_arm_inventory(value: object, expected_arm: str) -> dict[str, Any]:
    inventory = _mapping(value, "tensor_inventory_invalid")
    _exact_keys(
        inventory,
        {
            "schema_version",
            "arm",
            "branch_seed",
            "seed_material",
            "base_o_proj_frozen",
            "tensors",
            "optimizer_groups",
        },
        "tensor_inventory_keys_invalid",
    )
    if (
        inventory["schema_version"] != INVENTORY_SCHEMA_VERSION
        or inventory["arm"] != expected_arm
        or inventory["branch_seed"] != BASE_SEED
        or inventory["base_o_proj_frozen"] is not True
        or inventory["seed_material"]
        != {
            "base_seed": BASE_SEED,
            "derivation": "base_seed_only_v1",
            "components": ["base_seed"],
        }
    ):
        raise MultiArmContractError("tensor_inventory_identity_invalid")
    observed_entries = _sequence(inventory["tensors"], "tensor_entries_invalid")
    expected_entries = _expected_tensor_entries(ARM_PROFILE[expected_arm])
    if len(observed_entries) != PROFILE_TENSOR_COUNTS[ARM_PROFILE[expected_arm]]:
        raise MultiArmContractError("tensor_count_invalid")
    normalized: list[dict[str, Any]] = []
    for observed, expected in zip(observed_entries, expected_entries, strict=True):
        entry = _mapping(observed, "tensor_entry_invalid")
        _exact_keys(
            entry,
            set(expected) | {"tensor_bytes_sha256"},
            "tensor_entry_keys_invalid",
        )
        for key, expected_value in expected.items():
            if entry[key] != expected_value:
                raise MultiArmContractError("tensor_physical_shape_or_scope_invalid")
        tensor_sha = _require_sha(
            entry["tensor_bytes_sha256"],
            "tensor_bytes_sha256_invalid",
        )
        if entry["factor"] == "B" and tensor_sha != zero_tensor_sha256(
            int(entry["nbytes"])
        ):
            raise MultiArmContractError("lora_B_not_zero_initialized")
        normalized.append(dict(entry))
    if (
        sum(int(entry["numel"]) for entry in normalized)
        != PROFILE_PARAMETER_COUNTS[ARM_PROFILE[expected_arm]]
    ):
        raise MultiArmContractError("trainable_parameter_count_invalid")
    names = [str(entry["name"]) for entry in normalized]
    if len(names) != len(set(names)):
        raise MultiArmContractError("tensor_name_duplicate")
    _validate_optimizer_groups(expected_arm, inventory["optimizer_groups"], names)
    return {
        **dict(inventory),
        "tensors": normalized,
        "optimizer_groups": list(inventory["optimizer_groups"]),
    }


def load_and_validate_tensor_inventory_set(
    path: str | Path,
) -> tuple[dict[str, Any], str]:
    inventory_set, inventory_sha256 = _load_json(
        path,
        max_bytes=_MAX_INVENTORY_BYTES,
        code="tensor_inventory_set_invalid",
    )
    _exact_keys(
        inventory_set,
        {"schema_version", "arms"},
        "tensor_inventory_set_keys_invalid",
    )
    if inventory_set["schema_version"] != INVENTORY_SET_SCHEMA_VERSION:
        raise MultiArmContractError("tensor_inventory_set_schema_invalid")
    arms = _mapping(inventory_set["arms"], "tensor_inventory_arm_map_invalid")
    if set(arms) != set(TRAINED_ARMS):
        raise MultiArmContractError("tensor_inventory_arm_set_invalid")
    normalized = {arm: _validate_arm_inventory(arms[arm], arm) for arm in TRAINED_ARMS}

    # Per-tensor initialization is derived only from base seed + tensor name.
    # Every occurrence of a common tensor must therefore have identical actual
    # observed bytes, independent of arm, phase, or execution order.
    common_bytes: dict[str, str] = {}
    for arm in TRAINED_ARMS:
        for entry in normalized[arm]["tensors"]:
            name = entry["name"]
            digest = entry["tensor_bytes_sha256"]
            previous = common_bytes.setdefault(name, digest)
            if digest != previous:
                raise MultiArmContractError("paired_initial_tensor_bytes_mismatch")
    return {
        "schema_version": INVENTORY_SET_SCHEMA_VERSION,
        "arms": normalized,
    }, inventory_sha256


def expected_step_learning_rates(
    *,
    step: int,
    total_steps: int,
) -> tuple[Decimal, Decimal]:
    """Return the frozen conceptual Q/O rates for one completed optimizer step."""

    observed_step = _positive_int(step, "optimizer_schedule_step_invalid")
    observed_total = _positive_int(
        total_steps,
        "optimizer_schedule_total_steps_invalid",
    )
    if observed_step > observed_total:
        raise MultiArmContractError("optimizer_schedule_step_exceeds_total")
    warmup_numerator = min(observed_step, 8)
    expected_q = Q_LR * Decimal(warmup_numerator) / Decimal(8)
    expected_o = expected_q / Decimal(10)
    if expected_o * Decimal(10) != expected_q:
        raise AssertionError("frozen Q/O learning-rate ratio drift")
    return expected_q, expected_o


def _active_step_learning_rates(
    profile: str,
    *,
    step: int,
    total_steps: int,
) -> tuple[Decimal, Decimal]:
    expected_q, expected_o = expected_step_learning_rates(
        step=step,
        total_steps=total_steps,
    )
    if profile == "q_only":
        return expected_q, Decimal(0)
    if profile == "o_only":
        return Decimal(0), expected_o
    if profile == "q_plus_micro_o":
        return expected_q, expected_o
    raise MultiArmContractError("optimizer_schedule_profile_invalid")


def _optimizer_schedule_sha256(*, phase: str, steps: int) -> str:
    return _canonical_sha256(
        {
            "schema_version": "anchor.normalized-optimizer-schedule.v1",
            "phase": phase,
            "steps": steps,
            "warmup_steps": 8,
            "shape": "linear_warmup_then_constant_v1",
            "peak_q_learning_rate": str(Q_LR),
            "peak_o_learning_rate": str(O_LR),
            "exact_o_to_q_ratio": "0.1",
            "learning_rate_unit": "group_scale",
        }
    )


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _lr_trace_genesis_sha256(
    *,
    run_id: str,
    arm: str,
    phase: str,
    steps: int,
) -> str:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise MultiArmContractError("lr_trace_run_id_invalid")
    return _canonical_sha256(
        {
            "schema_version": "anchor.multiarm-observed-lr-trace-genesis.v1",
            "run_id": run_id,
            "arm": arm,
            "adapter_profile": ARM_PROFILE[arm],
            "phase": phase,
            "target_steps": steps,
            "optimizer_schedule_sha256": _optimizer_schedule_sha256(
                phase=phase,
                steps=steps,
            ),
            "fresh": True,
            "resume": False,
        }
    )


def _observed_lr_trace_step_sha256(
    *,
    previous_observed_trace_sha256: str,
    run_id: str,
    arm: str,
    phase: str,
    step_index: int,
    target_steps: int,
    actual_q_lr: Decimal,
    actual_o_lr: Decimal,
) -> str:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise MultiArmContractError("lr_trace_run_id_invalid")
    return _canonical_sha256(
        {
            "schema_version": "anchor.multiarm-observed-lr-trace-step.v1",
            "previous_observed_trace_sha256": _require_sha(
                previous_observed_trace_sha256,
                "previous_observed_trace_sha256_invalid",
            ),
            "run_id": run_id,
            "arm": arm,
            "adapter_profile": ARM_PROFILE[arm],
            "phase": phase,
            "step_index": step_index,
            "target_steps": target_steps,
            "optimizer_schedule_sha256": _optimizer_schedule_sha256(
                phase=phase,
                steps=target_steps,
            ),
            "actual_q_lr": _decimal_text(actual_q_lr),
            "actual_o_lr": _decimal_text(actual_o_lr),
        }
    )


@lru_cache(maxsize=64)
def _expected_lr_trace_prefixes(
    run_id: str,
    arm: str,
    phase: str,
    steps: int,
) -> tuple[str, ...]:
    digest = _lr_trace_genesis_sha256(
        run_id=run_id,
        arm=arm,
        phase=phase,
        steps=steps,
    )
    prefixes = [digest]
    for step in range(1, steps + 1):
        expected_q, expected_o = _active_step_learning_rates(
            ARM_PROFILE[arm],
            step=step,
            total_steps=steps,
        )
        digest = _observed_lr_trace_step_sha256(
            previous_observed_trace_sha256=digest,
            run_id=run_id,
            arm=arm,
            phase=phase,
            step_index=step,
            target_steps=steps,
            actual_q_lr=expected_q,
            actual_o_lr=expected_o,
        )
        prefixes.append(digest)
    return tuple(prefixes)


def _expected_lr_trace_prefix_sha256(
    *,
    run_id: str,
    arm: str,
    phase: str,
    steps: int,
    through_step: int,
) -> str:
    if arm not in TRAINED_ARMS or phase not in {"smoke", "full"}:
        raise MultiArmContractError("expected_lr_trace_identity_invalid")
    target = _positive_int(steps, "expected_lr_trace_target_invalid")
    if (
        isinstance(through_step, bool)
        or not isinstance(through_step, int)
        or through_step < 0
        or through_step > target
    ):
        raise MultiArmContractError("expected_lr_trace_prefix_step_invalid")
    return _expected_lr_trace_prefixes(run_id, arm, phase, target)[through_step]


def _expected_lr_trace_sha256(
    *,
    run_id: str,
    arm: str,
    phase: str,
    steps: int,
) -> str:
    return _expected_lr_trace_prefix_sha256(
        run_id=run_id,
        arm=arm,
        phase=phase,
        steps=steps,
        through_step=steps,
    )


def _lr_trace_contract_sha256(
    *,
    arm: str,
    phase: str,
    steps: int,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "anchor.multiarm-lr-trace-contract.v1",
            "arm": arm,
            "adapter_profile": ARM_PROFILE[arm],
            "phase": phase,
            "target_steps": steps,
            "optimizer_schedule_sha256": _optimizer_schedule_sha256(
                phase=phase,
                steps=steps,
            ),
            "run_id_domain_separated": True,
            "immutable_step_records": True,
            "resume": False,
        }
    )


def build_execution_plan(
    config_sha256: str,
    dependency_set_sha256: str,
    trusted_external_bindings: Mapping[str, Any],
    trusted_external_bindings_sha256: str,
    tensor_inventory_sha256: str,
) -> dict[str, Any]:
    _require_sha(config_sha256, "config_sha256_invalid")
    _require_sha(dependency_set_sha256, "dependency_set_sha256_invalid")
    _require_sha(tensor_inventory_sha256, "tensor_inventory_sha256_invalid")
    trusted_anchor = _require_sha(
        trusted_external_bindings_sha256,
        "trusted_external_bindings_sha256_invalid",
    )
    preflight = validate_trusted_external_bindings(
        trusted_external_bindings,
        trusted_anchor,
    )
    preflight_receipt_sha256 = str(
        trusted_external_bindings["consumer_preflight_receipt_sha256"]
    )
    source_binding_sha256 = str(trusted_external_bindings["source_binding_sha256"])
    training_assets = _mapping(
        preflight["training_assets"],
        "plan_training_assets_invalid",
    )
    evaluation_assets = _mapping(
        preflight["evaluation_assets"],
        "plan_evaluation_assets_invalid",
    )
    branch_seed_sha256 = _canonical_sha256(
        {
            "base_seed": BASE_SEED,
            "derivation": "base_seed_only_v1",
            "components": ["base_seed"],
        }
    )
    phases: list[dict[str, Any]] = []
    for phase in ("smoke", "full"):
        for execution_order, arm in enumerate(TRAINED_ARMS):
            source_asset = ARM_SOURCE_ASSET[arm]
            source = _mapping(training_assets[source_asset], "plan_source_invalid")
            dataset_binding = _dataset_binding(source, require_shard_order=True)
            dataset_binding_sha256 = _canonical_sha256(dataset_binding)
            steps = SMOKE_STEPS if phase == "smoke" else FULL_STEPS[arm]
            schedule_sha = _optimizer_schedule_sha256(phase=phase, steps=steps)
            profile = ARM_PROFILE[arm]
            lr_trace_contract_sha256 = _lr_trace_contract_sha256(
                arm=arm,
                phase=phase,
                steps=steps,
            )
            lock = {
                "record_order_sha256": source["record_order_sha256"],
                "target_projection_sha256": source["target_projection_sha256"],
                "branch_seed_sha256": branch_seed_sha256,
                "steps": steps,
                "optimizer_schedule_sha256": schedule_sha,
            }
            phases.append(
                {
                    "arm": arm,
                    "phase": phase,
                    "execution_order": execution_order,
                    "trusted_external_bindings_sha256": trusted_anchor,
                    "source_asset": source_asset,
                    "dataset_binding": dataset_binding,
                    "dataset_binding_sha256": dataset_binding_sha256,
                    "logical_dataset_sha256": source["logical_dataset_sha256"],
                    "shard_inventory_schema_version": source[
                        "shard_inventory_schema_version"
                    ],
                    "shard_inventory_schema_sha256": source[
                        "shard_inventory_schema_sha256"
                    ],
                    "shard_inventory_sha256": source["shard_inventory_sha256"],
                    "shard_count": source["shard_count"],
                    "shard_order_contract": source["shard_order_contract"],
                    "record_order_sha256": source["record_order_sha256"],
                    "target_projection_sha256": source["target_projection_sha256"],
                    "adapter_profile": profile,
                    "steps": steps,
                    "warmup_steps": 8,
                    "peak_q_learning_rate": float(Q_LR),
                    "peak_o_learning_rate": float(O_LR),
                    "optimizer_schedule_sha256": schedule_sha,
                    "lr_trace_contract_sha256": lr_trace_contract_sha256,
                    "fresh_base": True,
                    "fresh_adapter": True,
                    "resume": False,
                    "consumes_smoke_checkpoint": False,
                    "prerequisite": (
                        "consumer_preflight_passed"
                        if phase == "smoke"
                        else "all_smoke_phase_receipts_passed"
                    ),
                    "comparison_lock": lock,
                    "comparison_lock_sha256": _canonical_sha256(lock),
                }
            )
    plan_evaluation_assets = {}
    for name in EVALUATION_ASSETS:
        dataset_binding = _dataset_binding(
            evaluation_assets[name],
            require_shard_order=False,
        )
        plan_evaluation_assets[name] = {
            "dataset_binding": dataset_binding,
            "dataset_binding_sha256": _canonical_sha256(dataset_binding),
        }

    def eval_only_entry(name: str) -> dict[str, Any]:
        external = plan_evaluation_assets[name]
        return {
            "steps": 0,
            "adapter": None,
            "optimizer": None,
            "asset": name,
            "dataset_binding": external["dataset_binding"],
            "dataset_binding_sha256": external["dataset_binding_sha256"],
        }

    consumer_derived_asset_identity_contract_sha256 = _canonical_sha256(
        {
            "schema_version": CONSUMER_DERIVED_ASSET_IDENTITY_VERSION,
            "training_assets": training_assets,
            "evaluation_assets": evaluation_assets,
        }
    )
    producer_git = _mapping(
        preflight["producer_git"],
        "plan_producer_git_invalid",
    )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "model_free_dry_run_passed_gpu_execution_not_implemented",
        "config_sha256": config_sha256,
        "dependency_set_sha256": dependency_set_sha256,
        "trusted_external_bindings_sha256": trusted_anchor,
        "consumer_preflight_receipt_sha256": preflight_receipt_sha256,
        "source_binding_sha256": source_binding_sha256,
        "tensor_inventory_sha256": tensor_inventory_sha256,
        "producer_candidate_commit": producer_git["candidate_commit"],
        "producer_release_commit": producer_git["release_commit"],
        "producer_release_tree": producer_git["release_tree"],
        "binding_contract_sha256": preflight["binding_contract_sha256"],
        "artifact_version": preflight["artifact_version"],
        "artifact_tree_sha256": preflight["tree_digest_sha256"],
        "source_manifest_sha256": preflight["manifest"]["sha256"],
        "source_build_receipt_sha256": preflight["build_receipt"]["sha256"],
        "producer_schema_files_sha256": _canonical_sha256(preflight["schema_files"]),
        "release_attestation_sha256": preflight["release_attestation"]["sha256"],
        "consumer_derived_asset_identity_contract_sha256": (
            consumer_derived_asset_identity_contract_sha256
        ),
        "execution_order_sha256": _canonical_sha256(list(TRAINED_ARMS)),
        "branch_seed_sha256": branch_seed_sha256,
        "phases": phases,
        "evaluation_assets": plan_evaluation_assets,
        "eval_only": {
            "tool_base": eval_only_entry("tool_comparison_eval"),
            "planner_base": eval_only_entry("planner_comparison_eval"),
        },
        "comparison": {
            "tool": {
                "records": 400,
                "arms": [
                    "tool_base",
                    "tool_q_only",
                    "tool_q_plus_micro_o",
                ],
                "slots": 1200,
            },
            "planner": {
                "records": 240,
                "arms": [
                    "planner_base",
                    "planner_q_only",
                    "planner_o_only",
                    "planner_q_plus_micro_o",
                ],
                "slots": 960,
            },
        },
        "production_router": "planner_q_only",
        "optimizer_forbidden_assets": list(EVALUATION_ASSETS),
        "claims": {
            "body_read": False,
            "raw_token_ids_read": False,
            "model_loaded": False,
            "gpu_touched": False,
            "network_used": False,
            "training_executed": False,
            "evaluation_executed": False,
            "gpu_execution_implemented": False,
            "diagnostic_training_authorization_source": "local_user_task_only",
            "producer_training_authority_inferred": False,
            "formal": False,
        },
    }
    validate_execution_plan(
        plan,
        trusted_external_bindings,
        trusted_anchor,
    )
    return plan


def _phase_index(
    plan: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    phases = _sequence(plan["phases"], "plan_phases_invalid")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for value in phases:
        phase = _mapping(value, "plan_phase_invalid")
        key = (str(phase.get("arm")), str(phase.get("phase")))
        if key in result:
            raise MultiArmContractError("plan_phase_duplicate")
        result[key] = phase
    return result


def validate_execution_plan(
    plan: Mapping[str, Any],
    trusted_external_bindings: Mapping[str, Any],
    trusted_external_bindings_sha256: str,
) -> None:
    trusted_anchor = _require_sha(
        trusted_external_bindings_sha256,
        "trusted_external_bindings_sha256_invalid",
    )
    external = validate_trusted_external_bindings(
        trusted_external_bindings,
        trusted_anchor,
    )
    _exact_keys(
        plan,
        {
            "schema_version",
            "status",
            "config_sha256",
            "dependency_set_sha256",
            "trusted_external_bindings_sha256",
            "consumer_preflight_receipt_sha256",
            "source_binding_sha256",
            "tensor_inventory_sha256",
            "producer_candidate_commit",
            "producer_release_commit",
            "producer_release_tree",
            "binding_contract_sha256",
            "artifact_version",
            "artifact_tree_sha256",
            "source_manifest_sha256",
            "source_build_receipt_sha256",
            "producer_schema_files_sha256",
            "release_attestation_sha256",
            "consumer_derived_asset_identity_contract_sha256",
            "execution_order_sha256",
            "branch_seed_sha256",
            "phases",
            "evaluation_assets",
            "eval_only",
            "comparison",
            "production_router",
            "optimizer_forbidden_assets",
            "claims",
        },
        "plan_keys_invalid",
    )
    if (
        plan["schema_version"] != PLAN_SCHEMA_VERSION
        or plan["status"] != "model_free_dry_run_passed_gpu_execution_not_implemented"
        or plan["production_router"] != "planner_q_only"
        or plan["optimizer_forbidden_assets"] != list(EVALUATION_ASSETS)
        or plan["trusted_external_bindings_sha256"] != trusted_anchor
        or plan["consumer_preflight_receipt_sha256"]
        != trusted_external_bindings["consumer_preflight_receipt_sha256"]
        or plan["source_binding_sha256"]
        != trusted_external_bindings["source_binding_sha256"]
        or plan["producer_candidate_commit"]
        != external["producer_git"]["candidate_commit"]
        or plan["producer_release_commit"] != external["producer_git"]["release_commit"]
        or plan["producer_release_tree"] != external["producer_git"]["release_tree"]
        or plan["binding_contract_sha256"] != external["binding_contract_sha256"]
        or plan["artifact_version"] != external["artifact_version"]
        or plan["artifact_tree_sha256"] != external["tree_digest_sha256"]
        or plan["source_manifest_sha256"] != external["manifest"]["sha256"]
        or plan["source_build_receipt_sha256"] != external["build_receipt"]["sha256"]
        or plan["producer_schema_files_sha256"]
        != _canonical_sha256(external["schema_files"])
        or plan["release_attestation_sha256"]
        != external["release_attestation"]["sha256"]
        or plan["consumer_derived_asset_identity_contract_sha256"]
        != _canonical_sha256(
            {
                "schema_version": CONSUMER_DERIVED_ASSET_IDENTITY_VERSION,
                "training_assets": external["training_assets"],
                "evaluation_assets": external["evaluation_assets"],
            }
        )
        or plan["execution_order_sha256"] != _canonical_sha256(list(TRAINED_ARMS))
    ):
        raise MultiArmContractError("plan_identity_invalid")
    for key in (
        "config_sha256",
        "dependency_set_sha256",
        "tensor_inventory_sha256",
        "branch_seed_sha256",
    ):
        _require_sha(plan[key], f"plan_{key}_invalid")
    expected_branch_seed_sha256 = _canonical_sha256(
        {
            "base_seed": BASE_SEED,
            "derivation": "base_seed_only_v1",
            "components": ["base_seed"],
        }
    )
    if plan["branch_seed_sha256"] != expected_branch_seed_sha256:
        raise MultiArmContractError("plan_branch_seed_invalid")
    training_assets = _mapping(
        external["training_assets"],
        "trusted_plan_training_assets_invalid",
    )
    evaluation_assets = _mapping(
        external["evaluation_assets"],
        "trusted_plan_evaluation_assets_invalid",
    )
    index = _phase_index(plan)
    expected_keys = {
        (arm, phase) for phase in ("smoke", "full") for arm in TRAINED_ARMS
    }
    if set(index) != expected_keys:
        raise MultiArmContractError("plan_phase_set_invalid")
    for phase_name in ("smoke", "full"):
        for order, arm in enumerate(TRAINED_ARMS):
            phase = index[(arm, phase_name)]
            _exact_keys(
                phase,
                {
                    "arm",
                    "phase",
                    "execution_order",
                    "trusted_external_bindings_sha256",
                    "source_asset",
                    "dataset_binding",
                    "dataset_binding_sha256",
                    "logical_dataset_sha256",
                    "shard_inventory_schema_version",
                    "shard_inventory_schema_sha256",
                    "shard_inventory_sha256",
                    "shard_count",
                    "shard_order_contract",
                    "record_order_sha256",
                    "target_projection_sha256",
                    "adapter_profile",
                    "steps",
                    "warmup_steps",
                    "peak_q_learning_rate",
                    "peak_o_learning_rate",
                    "optimizer_schedule_sha256",
                    "lr_trace_contract_sha256",
                    "fresh_base",
                    "fresh_adapter",
                    "resume",
                    "consumes_smoke_checkpoint",
                    "prerequisite",
                    "comparison_lock",
                    "comparison_lock_sha256",
                },
                "plan_phase_keys_invalid",
            )
            expected_steps = SMOKE_STEPS if phase_name == "smoke" else FULL_STEPS[arm]
            expected_source_asset = ARM_SOURCE_ASSET[arm]
            expected_source = _mapping(
                training_assets[expected_source_asset],
                "trusted_plan_phase_source_invalid",
            )
            expected_dataset_binding = _dataset_binding(
                expected_source,
                require_shard_order=True,
            )
            expected_dataset_binding_sha256 = _canonical_sha256(
                expected_dataset_binding
            )
            if (
                phase["execution_order"] != order
                or phase["trusted_external_bindings_sha256"] != trusted_anchor
                or phase["source_asset"] != expected_source_asset
                or phase["dataset_binding"] != expected_dataset_binding
                or phase["dataset_binding_sha256"] != expected_dataset_binding_sha256
                or phase["logical_dataset_sha256"]
                != expected_source["logical_dataset_sha256"]
                or phase["shard_inventory_schema_version"]
                != expected_source["shard_inventory_schema_version"]
                or phase["shard_inventory_schema_sha256"]
                != expected_source["shard_inventory_schema_sha256"]
                or phase["shard_inventory_sha256"]
                != expected_source["shard_inventory_sha256"]
                or phase["shard_count"] != expected_source["shard_count"]
                or phase["shard_order_contract"]
                != expected_source["shard_order_contract"]
                or phase["record_order_sha256"]
                != expected_source["record_order_sha256"]
                or phase["target_projection_sha256"]
                != expected_source["target_projection_sha256"]
                or phase["adapter_profile"] != ARM_PROFILE[arm]
                or phase["steps"] != expected_steps
                or phase["warmup_steps"] != 8
                or Decimal(str(phase["peak_q_learning_rate"])) != Q_LR
                or Decimal(str(phase["peak_o_learning_rate"])) != O_LR
                or phase["optimizer_schedule_sha256"]
                != _optimizer_schedule_sha256(
                    phase=phase_name,
                    steps=expected_steps,
                )
                or phase["lr_trace_contract_sha256"]
                != _lr_trace_contract_sha256(
                    arm=arm,
                    phase=phase_name,
                    steps=expected_steps,
                )
                or not isinstance(phase["shard_order_contract"], str)
                or phase["fresh_base"] is not True
                or phase["fresh_adapter"] is not True
                or phase["resume"] is not False
                or phase["consumes_smoke_checkpoint"] is not False
                or phase["prerequisite"]
                != (
                    "consumer_preflight_passed"
                    if phase_name == "smoke"
                    else "all_smoke_phase_receipts_passed"
                )
            ):
                raise MultiArmContractError("plan_phase_contract_invalid")
            lock = _mapping(phase["comparison_lock"], "plan_lock_invalid")
            _exact_keys(
                lock,
                {
                    "record_order_sha256",
                    "target_projection_sha256",
                    "branch_seed_sha256",
                    "steps",
                    "optimizer_schedule_sha256",
                },
                "plan_lock_keys_invalid",
            )
            if (
                phase["comparison_lock_sha256"] != _canonical_sha256(lock)
                or lock["branch_seed_sha256"] != plan["branch_seed_sha256"]
                or lock["steps"] != expected_steps
                or lock["optimizer_schedule_sha256"]
                != phase["optimizer_schedule_sha256"]
                or lock["record_order_sha256"] != expected_source["record_order_sha256"]
                or lock["target_projection_sha256"]
                != expected_source["target_projection_sha256"]
            ):
                raise MultiArmContractError("plan_lock_digest_invalid")
    for phase_name in ("smoke", "full"):
        tool_locks = [
            index[(arm, phase_name)]["comparison_lock_sha256"]
            for arm in ("tool_q_only", "tool_q_plus_micro_o")
        ]
        planner_locks = [
            index[(arm, phase_name)]["comparison_lock_sha256"]
            for arm in (
                "planner_q_only",
                "planner_o_only",
                "planner_q_plus_micro_o",
            )
        ]
        if len(set(tool_locks)) != 1 or len(set(planner_locks)) != 1:
            raise MultiArmContractError("paired_comparison_lock_mismatch")

    expected_evaluation_assets = {}
    for name in EVALUATION_ASSETS:
        dataset_binding = _dataset_binding(
            evaluation_assets[name],
            require_shard_order=False,
        )
        expected_evaluation_assets[name] = {
            "dataset_binding": dataset_binding,
            "dataset_binding_sha256": _canonical_sha256(dataset_binding),
        }
    if plan["evaluation_assets"] != expected_evaluation_assets:
        raise MultiArmContractError("plan_evaluation_asset_binding_mismatch")

    def expected_eval_only(name: str) -> dict[str, Any]:
        identity = expected_evaluation_assets[name]
        return {
            "steps": 0,
            "adapter": None,
            "optimizer": None,
            "asset": name,
            "dataset_binding": identity["dataset_binding"],
            "dataset_binding_sha256": identity["dataset_binding_sha256"],
        }

    if plan["eval_only"] != {
        "tool_base": {
            **expected_eval_only("tool_comparison_eval"),
        },
        "planner_base": {
            **expected_eval_only("planner_comparison_eval"),
        },
    }:
        raise MultiArmContractError("base_eval_only_contract_invalid")
    if plan["comparison"] != {
        "tool": {
            "records": 400,
            "arms": ["tool_base", "tool_q_only", "tool_q_plus_micro_o"],
            "slots": 1200,
        },
        "planner": {
            "records": 240,
            "arms": [
                "planner_base",
                "planner_q_only",
                "planner_o_only",
                "planner_q_plus_micro_o",
            ],
            "slots": 960,
        },
    }:
        raise MultiArmContractError("comparison_slot_contract_invalid")
    _expect_mapping(
        plan["claims"],
        {
            "body_read": False,
            "raw_token_ids_read": False,
            "model_loaded": False,
            "gpu_touched": False,
            "network_used": False,
            "training_executed": False,
            "evaluation_executed": False,
            "gpu_execution_implemented": False,
            "diagnostic_training_authorization_source": "local_user_task_only",
            "producer_training_authority_inferred": False,
            "formal": False,
        },
        "plan_claims_invalid",
    )


def model_free_dry_run(
    *,
    config_path: str | Path,
    consumer_preflight_receipt: str | Path,
    tensor_inventory: str | Path,
) -> dict[str, Any]:
    _, config_sha, dependency_sha = load_config(config_path)
    preflight, preflight_sha, source_binding_sha = load_consumer_preflight_receipt(
        consumer_preflight_receipt
    )
    trusted_external_bindings, trusted_external_bindings_sha256 = (
        build_trusted_external_bindings(
            preflight,
            preflight_receipt_sha256=preflight_sha,
            source_binding_sha256=source_binding_sha,
        )
    )
    _, inventory_sha = load_and_validate_tensor_inventory_set(tensor_inventory)
    return build_execution_plan(
        config_sha,
        dependency_sha,
        trusted_external_bindings,
        trusted_external_bindings_sha256,
        inventory_sha,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


_PROGRESS_KEYS: Final = {
    "schema_version",
    "status",
    "run_id",
    "arm",
    "phase",
    "optimizer_step",
    "step_index",
    "target_steps",
    "fresh",
    "resume",
    "loss",
    "gradient_norm",
    "lr_q",
    "lr_o",
    "warmup_steps",
    "peak_q_learning_rate",
    "peak_o_learning_rate",
    "conceptual_q_learning_rate",
    "conceptual_o_learning_rate",
    "expected_active_q_learning_rate",
    "expected_active_o_learning_rate",
    "optimizer_schedule_sha256",
    "lr_trace_contract_sha256",
    "previous_step_file_sha256",
    "previous_observed_trace_sha256",
    "observed_lr_trace_sha256",
    "expected_lr_trace_prefix_sha256",
    "expected_lr_trace_sha256",
    "torch_allocated_bytes",
    "torch_reserved_bytes",
    "updated_at_utc",
}


def _validate_progress_state(
    value: object,
    *,
    run_id: str,
    arm: str,
    phase: str,
    expected_step: int,
    target_steps: int,
    expected_previous_step_file_sha256: str,
) -> dict[str, Any]:
    progress = _mapping(value, "progress_state_invalid")
    _exact_keys(progress, _PROGRESS_KEYS, "progress_state_keys_invalid")
    step = _positive_int(expected_step, "progress_expected_step_invalid")
    target = _positive_int(target_steps, "progress_expected_target_invalid")
    if (
        progress["schema_version"] != PROGRESS_SCHEMA_VERSION
        or progress["status"] != "optimizer_step_completed"
        or progress["run_id"] != run_id
        or progress["arm"] != arm
        or progress["phase"] != phase
        or progress["optimizer_step"] != step
        or progress["step_index"] != step
        or progress["target_steps"] != target
        or progress["fresh"] is not (step == 1)
        or progress["resume"] is not False
    ):
        raise MultiArmContractError("progress_state_identity_invalid")
    if not isinstance(progress["updated_at_utc"], str) or not progress[
        "updated_at_utc"
    ].endswith("Z"):
        raise MultiArmContractError("progress_state_timestamp_invalid")
    expected_target = SMOKE_STEPS if phase == "smoke" else FULL_STEPS[arm]
    if target != expected_target:
        raise MultiArmContractError("progress_state_target_mismatch")
    observed_loss = _finite_scalar(progress["loss"], "progress_state_loss_nonfinite")
    observed_grad = _finite_scalar(
        progress["gradient_norm"],
        "progress_state_gradient_nonfinite",
    )
    observed_lr_q = _finite_scalar(
        progress["lr_q"],
        "progress_state_lr_q_nonfinite",
    )
    observed_lr_o = _finite_scalar(
        progress["lr_o"],
        "progress_state_lr_o_nonfinite",
    )
    if min(observed_grad, observed_lr_q, observed_lr_o) < 0:
        raise MultiArmContractError("progress_state_negative_scalar_invalid")
    conceptual_q, conceptual_o = expected_step_learning_rates(
        step=step,
        total_steps=target,
    )
    active_q, active_o = _active_step_learning_rates(
        ARM_PROFILE[arm],
        step=step,
        total_steps=target,
    )
    actual_q = Decimal(str(observed_lr_q))
    actual_o = Decimal(str(observed_lr_o))
    if (
        actual_q != active_q
        or actual_o != active_o
        or Decimal(str(progress["conceptual_q_learning_rate"])) != conceptual_q
        or Decimal(str(progress["conceptual_o_learning_rate"])) != conceptual_o
        or Decimal(str(progress["expected_active_q_learning_rate"])) != active_q
        or Decimal(str(progress["expected_active_o_learning_rate"])) != active_o
    ):
        raise MultiArmContractError("progress_state_learning_rate_invalid")
    if ARM_PROFILE[arm] == "q_plus_micro_o" and actual_o * Decimal(10) != actual_q:
        raise MultiArmContractError("progress_state_qo_ratio_invalid")
    schedule_sha256 = _optimizer_schedule_sha256(phase=phase, steps=target)
    expected_full_sha256 = _expected_lr_trace_sha256(
        run_id=run_id,
        arm=arm,
        phase=phase,
        steps=target,
    )
    expected_previous_sha256 = _expected_lr_trace_prefix_sha256(
        run_id=run_id,
        arm=arm,
        phase=phase,
        steps=target,
        through_step=step - 1,
    )
    expected_prefix_sha256 = _expected_lr_trace_prefix_sha256(
        run_id=run_id,
        arm=arm,
        phase=phase,
        steps=target,
        through_step=step,
    )
    observed_prefix_sha256 = _observed_lr_trace_step_sha256(
        previous_observed_trace_sha256=str(progress["previous_observed_trace_sha256"]),
        run_id=run_id,
        arm=arm,
        phase=phase,
        step_index=step,
        target_steps=target,
        actual_q_lr=actual_q,
        actual_o_lr=actual_o,
    )
    if (
        progress["warmup_steps"] != 8
        or Decimal(str(progress["peak_q_learning_rate"])) != Q_LR
        or Decimal(str(progress["peak_o_learning_rate"])) != O_LR
        or progress["optimizer_schedule_sha256"] != schedule_sha256
        or progress["lr_trace_contract_sha256"]
        != _lr_trace_contract_sha256(arm=arm, phase=phase, steps=target)
        or progress["previous_step_file_sha256"]
        != _require_sha(
            expected_previous_step_file_sha256,
            "progress_expected_previous_step_file_sha256_invalid",
        )
        or progress["previous_observed_trace_sha256"] != expected_previous_sha256
        or progress["observed_lr_trace_sha256"] != observed_prefix_sha256
        or progress["observed_lr_trace_sha256"] != expected_prefix_sha256
        or progress["expected_lr_trace_prefix_sha256"] != expected_prefix_sha256
        or progress["expected_lr_trace_sha256"] != expected_full_sha256
    ):
        raise MultiArmContractError("progress_state_trace_invalid")
    allocated = _nonnegative_int(
        progress["torch_allocated_bytes"],
        "progress_state_allocated_invalid",
    )
    reserved = _nonnegative_int(
        progress["torch_reserved_bytes"],
        "progress_state_reserved_invalid",
    )
    if reserved < allocated:
        raise MultiArmContractError("progress_state_reserved_below_allocated")
    result = dict(progress)
    result.update(
        {
            "loss": observed_loss,
            "gradient_norm": observed_grad,
            "lr_q": observed_lr_q,
            "lr_o": observed_lr_o,
            "torch_allocated_bytes": allocated,
            "torch_reserved_bytes": reserved,
        }
    )
    return result


_PROGRESS_RECORD_FILENAME: Final = "progress.json"
_PROGRESS_COMPLETE_FILENAME: Final = "complete.sha256"


def _progress_file_genesis_sha256(
    *,
    run_id: str,
    arm: str,
    phase: str,
    target_steps: int,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "anchor.multiarm-progress-file-genesis.v1",
            "run_id": run_id,
            "arm": arm,
            "phase": phase,
            "target_steps": target_steps,
            "create_once": True,
            "resume": False,
        }
    )


def _progress_step_directory(root: Path, step: int) -> Path:
    return root / f"step-{step:06d}"


def _require_progress_root(root: Path, *, create: bool) -> None:
    if root.name != "progress":
        raise MultiArmContractError("progress_root_name_invalid")
    if create:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise MultiArmContractError("progress_root_create_failed") from None
    try:
        metadata = root.lstat()
    except OSError:
        raise MultiArmContractError("progress_root_missing") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise MultiArmContractError("progress_root_identity_invalid")


def _load_progress_evidence(
    progress_root: str | Path,
    *,
    run_id: str,
    arm: str,
    phase: str,
    expected_steps: int,
    target_steps: int,
) -> dict[str, Any]:
    root = Path(progress_root)
    _require_progress_root(root, create=False)
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise MultiArmContractError("progress_evidence_run_id_invalid")
    if arm not in TRAINED_ARMS or phase not in {"smoke", "full"}:
        raise MultiArmContractError("progress_evidence_identity_invalid")
    if (
        isinstance(expected_steps, bool)
        or not isinstance(expected_steps, int)
        or expected_steps < 0
    ):
        raise MultiArmContractError("progress_evidence_step_count_invalid")
    target = _positive_int(target_steps, "progress_evidence_target_invalid")
    if expected_steps > target:
        raise MultiArmContractError("progress_evidence_step_count_exceeds_target")
    try:
        root_before = root.lstat()
        entries = list(root.iterdir())
    except OSError:
        raise MultiArmContractError("progress_evidence_inventory_invalid") from None
    expected_names = {f"step-{step:06d}" for step in range(1, expected_steps + 1)}
    if {entry.name for entry in entries} != expected_names:
        raise MultiArmContractError("progress_evidence_inventory_not_contiguous")
    step_file_sha256s: list[str] = []
    marker_file_sha256s: list[str] = []
    records: list[dict[str, Any]] = []
    previous_file_sha256 = _progress_file_genesis_sha256(
        run_id=run_id,
        arm=arm,
        phase=phase,
        target_steps=target,
    )
    inventory: list[dict[str, Any]] = []
    for step in range(1, expected_steps + 1):
        step_dir = _progress_step_directory(root, step)
        try:
            step_before = step_dir.lstat()
            children = list(step_dir.iterdir())
        except OSError:
            raise MultiArmContractError("progress_step_directory_invalid") from None
        if stat.S_ISLNK(step_before.st_mode) or not stat.S_ISDIR(step_before.st_mode):
            raise MultiArmContractError("progress_step_directory_identity_invalid")
        if {child.name for child in children} != {
            _PROGRESS_RECORD_FILENAME,
            _PROGRESS_COMPLETE_FILENAME,
        }:
            raise MultiArmContractError("progress_step_incomplete_or_extra")
        record_path = step_dir / _PROGRESS_RECORD_FILENAME
        marker_path = step_dir / _PROGRESS_COMPLETE_FILENAME
        record, record_sha256 = _load_json(
            record_path,
            max_bytes=_MAX_RECEIPT_BYTES,
            code="progress_step_record_invalid",
        )
        marker_bytes, marker_sha256 = _snapshot(
            marker_path,
            max_bytes=256,
            code="progress_step_marker_invalid",
        )
        try:
            marker_text = marker_bytes.decode("ascii")
        except UnicodeDecodeError:
            raise MultiArmContractError("progress_step_marker_ascii_invalid") from None
        if marker_text != f"{record_sha256}  {_PROGRESS_RECORD_FILENAME}\n":
            raise MultiArmContractError("progress_step_marker_digest_mismatch")
        normalized = _validate_progress_state(
            record,
            run_id=run_id,
            arm=arm,
            phase=phase,
            expected_step=step,
            target_steps=target,
            expected_previous_step_file_sha256=previous_file_sha256,
        )
        try:
            step_after = step_dir.lstat()
        except OSError:
            raise MultiArmContractError("progress_step_directory_changed") from None
        if (
            step_before.st_dev,
            step_before.st_ino,
            step_before.st_mtime_ns,
        ) != (
            step_after.st_dev,
            step_after.st_ino,
            step_after.st_mtime_ns,
        ):
            raise MultiArmContractError("progress_step_directory_changed")
        records.append(normalized)
        step_file_sha256s.append(record_sha256)
        marker_file_sha256s.append(marker_sha256)
        inventory.append(
            {
                "step_index": step,
                "record_sha256": record_sha256,
                "marker_sha256": marker_sha256,
            }
        )
        previous_file_sha256 = record_sha256
    try:
        root_after = root.lstat()
    except OSError:
        raise MultiArmContractError("progress_evidence_inventory_changed") from None
    if (
        root_before.st_dev,
        root_before.st_ino,
        root_before.st_mtime_ns,
    ) != (
        root_after.st_dev,
        root_after.st_ino,
        root_after.st_mtime_ns,
    ):
        raise MultiArmContractError("progress_evidence_inventory_changed")
    inventory_sha256 = _canonical_sha256(
        {
            "schema_version": "anchor.multiarm-progress-evidence-inventory.v1",
            "run_id": run_id,
            "arm": arm,
            "phase": phase,
            "target_steps": target,
            "records": inventory,
        }
    )
    return {
        "schema_version": "anchor.multiarm-progress-evidence.v1",
        "run_id": run_id,
        "arm": arm,
        "phase": phase,
        "step_count": expected_steps,
        "target_steps": target,
        "inventory_sha256": inventory_sha256,
        "step_file_sha256s": step_file_sha256s,
        "marker_file_sha256s": marker_file_sha256s,
        "records": records,
    }


def build_progress_evidence_identity(
    progress_root: str | Path,
    *,
    run_id: str,
    arm: str,
    phase: str,
    target_steps: int,
) -> dict[str, Any]:
    evidence = _load_progress_evidence(
        progress_root,
        run_id=run_id,
        arm=arm,
        phase=phase,
        expected_steps=target_steps,
        target_steps=target_steps,
    )
    final = evidence["records"][-1]
    return {
        "schema_version": "anchor.multiarm-progress-evidence-identity.v1",
        "step_count": evidence["step_count"],
        "target_steps": evidence["target_steps"],
        "inventory_sha256": evidence["inventory_sha256"],
        "step_file_sha256s": evidence["step_file_sha256s"],
        "marker_file_sha256s": evidence["marker_file_sha256s"],
        "final_step_file_sha256": evidence["step_file_sha256s"][-1],
        "lr_trace_contract_sha256": final["lr_trace_contract_sha256"],
        "observed_lr_trace_sha256": final["observed_lr_trace_sha256"],
        "expected_lr_trace_sha256": final["expected_lr_trace_sha256"],
    }


def write_scalar_progress(
    progress_path: str | Path,
    *,
    run_id: str,
    arm: str,
    phase: str,
    optimizer_step: int,
    target_steps: int,
    loss: float,
    gradient_norm: float,
    lr_q: float,
    lr_o: float,
    torch_allocated_bytes: int,
    torch_reserved_bytes: int,
) -> dict[str, object]:
    """Publish one immutable create-once scalar optimizer-step record."""

    root = Path(progress_path)
    _require_progress_root(root, create=True)
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise MultiArmContractError("progress_run_id_invalid")
    if arm not in TRAINED_ARMS or phase not in {"smoke", "full"}:
        raise MultiArmContractError("progress_arm_or_phase_invalid")
    step = _positive_int(optimizer_step, "progress_step_invalid")
    target = _positive_int(target_steps, "progress_target_invalid")
    if step > target:
        raise MultiArmContractError("progress_step_exceeds_target")
    expected_target = SMOKE_STEPS if phase == "smoke" else FULL_STEPS[arm]
    if target != expected_target:
        raise MultiArmContractError("progress_phase_target_mismatch")
    previous_evidence = _load_progress_evidence(
        root,
        run_id=run_id,
        arm=arm,
        phase=phase,
        expected_steps=step - 1,
        target_steps=target,
    )
    if step == 1:
        previous_observed_trace_sha256 = _lr_trace_genesis_sha256(
            run_id=run_id,
            arm=arm,
            phase=phase,
            steps=target,
        )
        previous_step_file_sha256 = _progress_file_genesis_sha256(
            run_id=run_id,
            arm=arm,
            phase=phase,
            target_steps=target,
        )
    else:
        previous_record = previous_evidence["records"][-1]
        previous_observed_trace_sha256 = str(
            previous_record["observed_lr_trace_sha256"]
        )
        previous_step_file_sha256 = str(previous_evidence["step_file_sha256s"][-1])
    observed_loss = _finite_scalar(loss, "progress_loss_nonfinite")
    observed_grad = _finite_scalar(gradient_norm, "progress_gradient_nonfinite")
    observed_lr_q = _finite_scalar(lr_q, "progress_lr_q_nonfinite")
    observed_lr_o = _finite_scalar(lr_o, "progress_lr_o_nonfinite")
    if min(observed_grad, observed_lr_q, observed_lr_o) < 0:
        raise MultiArmContractError("progress_negative_scalar_invalid")
    profile = ARM_PROFILE[arm]
    conceptual_q_lr, conceptual_o_lr = expected_step_learning_rates(
        step=step,
        total_steps=target,
    )
    expected_active_q_lr, expected_active_o_lr = _active_step_learning_rates(
        profile,
        step=step,
        total_steps=target,
    )
    observed_q_decimal = Decimal(str(observed_lr_q))
    observed_o_decimal = Decimal(str(observed_lr_o))
    if profile == "q_only" and (
        observed_q_decimal != expected_active_q_lr
        or observed_o_decimal != expected_active_o_lr
    ):
        raise MultiArmContractError("progress_q_only_lr_invalid")
    if profile == "o_only" and (
        observed_q_decimal != expected_active_q_lr
        or observed_o_decimal != expected_active_o_lr
    ):
        raise MultiArmContractError("progress_o_only_lr_invalid")
    if profile == "q_plus_micro_o" and (
        observed_q_decimal != expected_active_q_lr
        or observed_o_decimal != expected_active_o_lr
        or observed_o_decimal * Decimal(10) != observed_q_decimal
    ):
        raise MultiArmContractError("progress_qo_lr_ratio_invalid")
    allocated = _nonnegative_int(
        torch_allocated_bytes,
        "progress_allocated_invalid",
    )
    reserved = _nonnegative_int(
        torch_reserved_bytes,
        "progress_reserved_invalid",
    )
    if reserved < allocated:
        raise MultiArmContractError("progress_reserved_below_allocated")
    observed_lr_trace_sha256 = _observed_lr_trace_step_sha256(
        previous_observed_trace_sha256=previous_observed_trace_sha256,
        run_id=run_id,
        arm=arm,
        phase=phase,
        step_index=step,
        target_steps=target,
        actual_q_lr=observed_q_decimal,
        actual_o_lr=observed_o_decimal,
    )
    expected_lr_trace_prefix_sha256 = _expected_lr_trace_prefix_sha256(
        run_id=run_id,
        arm=arm,
        phase=phase,
        steps=target,
        through_step=step,
    )
    if observed_lr_trace_sha256 != expected_lr_trace_prefix_sha256:
        raise MultiArmContractError("progress_observed_lr_trace_mismatch")
    payload: dict[str, object] = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "status": "optimizer_step_completed",
        "run_id": run_id,
        "arm": arm,
        "phase": phase,
        "optimizer_step": step,
        "step_index": step,
        "target_steps": target,
        "fresh": step == 1,
        "resume": False,
        "loss": observed_loss,
        "gradient_norm": observed_grad,
        "lr_q": observed_lr_q,
        "lr_o": observed_lr_o,
        "warmup_steps": 8,
        "peak_q_learning_rate": float(Q_LR),
        "peak_o_learning_rate": float(O_LR),
        "conceptual_q_learning_rate": float(conceptual_q_lr),
        "conceptual_o_learning_rate": float(conceptual_o_lr),
        "expected_active_q_learning_rate": float(expected_active_q_lr),
        "expected_active_o_learning_rate": float(expected_active_o_lr),
        "optimizer_schedule_sha256": _optimizer_schedule_sha256(
            phase=phase,
            steps=target,
        ),
        "lr_trace_contract_sha256": _lr_trace_contract_sha256(
            arm=arm,
            phase=phase,
            steps=target,
        ),
        "previous_step_file_sha256": previous_step_file_sha256,
        "previous_observed_trace_sha256": previous_observed_trace_sha256,
        "observed_lr_trace_sha256": observed_lr_trace_sha256,
        "expected_lr_trace_prefix_sha256": expected_lr_trace_prefix_sha256,
        "expected_lr_trace_sha256": _expected_lr_trace_sha256(
            run_id=run_id,
            arm=arm,
            phase=phase,
            steps=target,
        ),
        "torch_allocated_bytes": allocated,
        "torch_reserved_bytes": reserved,
        "updated_at_utc": _utc_now(),
    }
    encoded = _canonical_json(payload)
    record_sha256 = _sha256(encoded)
    step_dir = _progress_step_directory(root, step)
    record_path = step_dir / _PROGRESS_RECORD_FILENAME
    marker_path = step_dir / _PROGRESS_COMPLETE_FILENAME
    temporary = root.parent / (
        f".{root.name}-step-{step:06d}.tmp-{uuid.uuid4().hex}.json"
    )
    try:
        with temporary.open("xb", buffering=0) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        current_previous = _load_progress_evidence(
            root,
            run_id=run_id,
            arm=arm,
            phase=phase,
            expected_steps=step - 1,
            target_steps=target,
        )
        if (
            current_previous["inventory_sha256"]
            != previous_evidence["inventory_sha256"]
        ):
            raise MultiArmContractError("progress_previous_inventory_changed")
        try:
            step_dir.mkdir()
        except FileExistsError:
            raise MultiArmContractError("progress_step_already_claimed") from None
        except OSError:
            raise MultiArmContractError("progress_step_claim_failed") from None
        try:
            os.rename(temporary, record_path)
        except OSError:
            raise MultiArmContractError("progress_step_record_publish_failed") from None
        marker_bytes = f"{record_sha256}  {_PROGRESS_RECORD_FILENAME}\n".encode("ascii")
        try:
            with marker_path.open("xb", buffering=0) as handle:
                handle.write(marker_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            raise MultiArmContractError("progress_step_marker_publish_failed") from None
        _sync_directory(step_dir)
        _sync_directory(root)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    observed_evidence = _load_progress_evidence(
        root,
        run_id=run_id,
        arm=arm,
        phase=phase,
        expected_steps=step,
        target_steps=target,
    )
    observed_payload = observed_evidence["records"][-1]
    if (
        observed_evidence["step_file_sha256s"][-1] != record_sha256
        or _canonical_json(observed_payload) != encoded
    ):
        raise MultiArmContractError("progress_atomic_verification_failed")
    return payload


_PHASE_RECEIPT_KEYS: Final = {
    "schema_version",
    "status",
    "run_id",
    "arm",
    "phase",
    "steps_completed",
    "target_steps",
    "trusted_external_bindings_sha256",
    "dataset_binding_sha256",
    "comparison_lock_sha256",
    "warmup_steps",
    "peak_q_learning_rate",
    "peak_o_learning_rate",
    "optimizer_schedule_sha256",
    "lr_trace_contract_sha256",
    "expected_lr_trace_sha256",
    "observed_lr_trace_sha256",
    "progress_inventory_sha256",
    "progress_step_file_sha256s",
    "progress_marker_file_sha256s",
    "progress_final_step_file_sha256",
    "progress_step_index",
    "progress_target_steps",
    "base_before_sha256",
    "base_after_sha256",
    "adapter_sha256",
    "finite_loss",
    "finite_gradients",
    "optimizer_state_verified",
    "adapter_effect_nonzero",
    "memory_gate_passed",
    "canonical_lock_verified",
    "atomic_publish_verified",
    "fresh_base",
    "fresh_adapter",
    "resume",
    "smoke_checkpoint_consumed_by_full",
    "body_included",
    "raw_token_ids_included",
}


def validate_phase_receipt(
    receipt: Mapping[str, Any],
    phase_plan: Mapping[str, Any],
    progress_root: str | Path,
) -> None:
    _exact_keys(receipt, _PHASE_RECEIPT_KEYS, "phase_receipt_keys_invalid")
    if (
        receipt["schema_version"] != PHASE_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != "passed"
        or receipt["arm"] != phase_plan["arm"]
        or receipt["phase"] != phase_plan["phase"]
        or receipt["steps_completed"] != phase_plan["steps"]
        or receipt["target_steps"] != phase_plan["steps"]
        or receipt["trusted_external_bindings_sha256"]
        != phase_plan["trusted_external_bindings_sha256"]
        or receipt["dataset_binding_sha256"] != phase_plan["dataset_binding_sha256"]
        or receipt["comparison_lock_sha256"] != phase_plan["comparison_lock_sha256"]
        or receipt["warmup_steps"] != 8
        or Decimal(str(receipt["peak_q_learning_rate"])) != Q_LR
        or Decimal(str(receipt["peak_o_learning_rate"])) != O_LR
        or receipt["optimizer_schedule_sha256"]
        != _optimizer_schedule_sha256(
            phase=str(phase_plan["phase"]),
            steps=int(phase_plan["steps"]),
        )
        or receipt["optimizer_schedule_sha256"]
        != phase_plan["optimizer_schedule_sha256"]
        or receipt["lr_trace_contract_sha256"]
        != _lr_trace_contract_sha256(
            arm=str(phase_plan["arm"]),
            phase=str(phase_plan["phase"]),
            steps=int(phase_plan["steps"]),
        )
        or receipt["lr_trace_contract_sha256"] != phase_plan["lr_trace_contract_sha256"]
    ):
        raise MultiArmContractError("phase_receipt_identity_invalid")
    _require_sha(
        receipt["trusted_external_bindings_sha256"],
        "phase_receipt_trusted_bindings_invalid",
    )
    if (
        not isinstance(receipt["run_id"], str)
        or _RUN_ID_RE.fullmatch(receipt["run_id"]) is None
    ):
        raise MultiArmContractError("phase_receipt_run_id_invalid")
    evidence = build_progress_evidence_identity(
        progress_root,
        run_id=str(receipt["run_id"]),
        arm=str(phase_plan["arm"]),
        phase=str(phase_plan["phase"]),
        target_steps=int(phase_plan["steps"]),
    )
    if (
        receipt["progress_inventory_sha256"] != evidence["inventory_sha256"]
        or receipt["progress_step_file_sha256s"] != evidence["step_file_sha256s"]
        or receipt["progress_marker_file_sha256s"] != evidence["marker_file_sha256s"]
        or receipt["progress_final_step_file_sha256"]
        != evidence["final_step_file_sha256"]
        or receipt["progress_step_index"] != evidence["step_count"]
        or receipt["progress_target_steps"] != evidence["target_steps"]
        or receipt["lr_trace_contract_sha256"] != evidence["lr_trace_contract_sha256"]
        or receipt["observed_lr_trace_sha256"] != evidence["observed_lr_trace_sha256"]
        or receipt["expected_lr_trace_sha256"] != evidence["expected_lr_trace_sha256"]
        or receipt["observed_lr_trace_sha256"] != receipt["expected_lr_trace_sha256"]
    ):
        raise MultiArmContractError("phase_receipt_progress_binding_invalid")
    base_before = _require_sha(
        receipt["base_before_sha256"],
        "phase_receipt_base_before_invalid",
    )
    base_after = _require_sha(
        receipt["base_after_sha256"],
        "phase_receipt_base_after_invalid",
    )
    _require_sha(receipt["adapter_sha256"], "phase_receipt_adapter_invalid")
    if base_before != base_after:
        raise MultiArmContractError("phase_receipt_base_hash_changed")
    required_true = (
        "finite_loss",
        "finite_gradients",
        "optimizer_state_verified",
        "adapter_effect_nonzero",
        "memory_gate_passed",
        "canonical_lock_verified",
        "atomic_publish_verified",
        "fresh_base",
        "fresh_adapter",
    )
    if any(receipt[key] is not True for key in required_true):
        raise MultiArmContractError("phase_receipt_gate_failed")
    if (
        receipt["resume"] is not False
        or receipt["smoke_checkpoint_consumed_by_full"] is not False
        or receipt["body_included"] is not False
        or receipt["raw_token_ids_included"] is not False
    ):
        raise MultiArmContractError("phase_receipt_forbidden_claim")


def validate_smoke_gate(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    final_progress_paths: Mapping[str, str | Path],
    trusted_external_bindings: Mapping[str, Any],
    trusted_external_bindings_sha256: str,
) -> str:
    """Require all nine smoke receipts before any full phase is launchable."""

    validate_execution_plan(
        plan,
        trusted_external_bindings,
        trusted_external_bindings_sha256,
    )
    if len(receipts) != len(TRAINED_ARMS):
        raise MultiArmContractError("smoke_receipt_count_invalid")
    if set(final_progress_paths) != set(TRAINED_ARMS):
        raise MultiArmContractError("smoke_progress_path_set_invalid")
    index = _phase_index(plan)
    seen: set[str] = set()
    run_ids: set[str] = set()
    for receipt in receipts:
        arm = receipt.get("arm")
        if arm not in TRAINED_ARMS or arm in seen:
            raise MultiArmContractError("smoke_receipt_arm_set_invalid")
        validate_phase_receipt(
            receipt,
            index[(str(arm), "smoke")],
            final_progress_paths[str(arm)],
        )
        seen.add(str(arm))
        run_ids.add(str(receipt["run_id"]))
    if seen != set(TRAINED_ARMS) or len(run_ids) != 1:
        raise MultiArmContractError("smoke_receipts_not_one_complete_run")
    return _canonical_sha256(list(receipts))


def build_full_gate_receipt(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    final_progress_paths: Mapping[str, str | Path],
    trusted_external_bindings: Mapping[str, Any],
    trusted_external_bindings_sha256: str,
) -> dict[str, Any]:
    """Return a body-free authorization proof only after all smoke gates pass."""

    smoke_gate_sha256 = validate_smoke_gate(
        plan,
        receipts,
        final_progress_paths,
        trusted_external_bindings,
        trusted_external_bindings_sha256,
    )
    index = _phase_index(plan)
    full_phases = [index[(arm, "full")] for arm in TRAINED_ARMS]
    run_id = str(receipts[0]["run_id"])
    return {
        "schema_version": FULL_GATE_SCHEMA_VERSION,
        "status": "passed",
        "run_id": run_id,
        "trusted_external_bindings_sha256": trusted_external_bindings_sha256,
        "smoke_gate_sha256": smoke_gate_sha256,
        "authorized_full_arms": list(TRAINED_ARMS),
        "full_phase_plan_sha256": _canonical_sha256(full_phases),
        "full_fresh_from_base": True,
        "full_fresh_adapter": True,
        "resume": False,
        "smoke_checkpoint_consumed_by_full": False,
        "body_included": False,
        "raw_token_ids_included": False,
        "gpu_execution_implemented": False,
    }


def publish_body_free_phase_receipt(
    destination: str | Path,
    receipt: Mapping[str, Any],
    phase_plan: Mapping[str, Any],
    final_progress_path: str | Path,
) -> str:
    """Atomically publish a receipt+sidecar directory after strict validation."""

    validate_phase_receipt(receipt, phase_plan, final_progress_path)
    destination_path = Path(destination)
    if os.path.lexists(destination_path):
        raise MultiArmContractError("phase_receipt_destination_exists")
    parent = destination_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{destination_path.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    receipt_path = staging / "phase_receipt.json"
    sidecar_path = staging / "phase_receipt.json.sha256"
    encoded = _canonical_json(dict(receipt))
    digest = _sha256(encoded)
    try:
        with receipt_path.open("xb", buffering=0) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with sidecar_path.open("xb", buffering=0) as handle:
            handle.write(f"{digest}  phase_receipt.json\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, destination_path)
        _sync_directory(parent)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
    if (
        _sha256((destination_path / "phase_receipt.json").read_bytes()) != digest
        or (destination_path / "phase_receipt.json.sha256").read_text(encoding="ascii")
        != f"{digest}  phase_receipt.json\n"
    ):
        raise MultiArmContractError("phase_receipt_atomic_verification_failed")
    return digest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-free Gemma 3 unbalanced-v2 multi-arm contract runner."
    )
    parser.add_argument("--config", default=str(CONFIG_PATH))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-config", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--consumer-preflight-receipt")
    parser.add_argument("--tensor-inventory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.validate_config:
            _, config_sha, dependency_sha = load_config(args.config)
            result = {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "status": "passed",
                "config_sha256": config_sha,
                "dependency_set_sha256": dependency_sha,
                "model_loaded": False,
                "gpu_touched": False,
                "network_used": False,
            }
        else:
            if not args.consumer_preflight_receipt or not args.tensor_inventory:
                raise MultiArmContractError(
                    "dry_run_requires_consumer_preflight_and_tensor_inventory"
                )
            result = model_free_dry_run(
                config_path=args.config,
                consumer_preflight_receipt=args.consumer_preflight_receipt,
                tensor_inventory=args.tensor_inventory,
            )
        sys.stdout.buffer.write(_canonical_json(result))
        return 0
    except MultiArmContractError as error:
        sys.stderr.buffer.write(
            _canonical_json(
                {
                    "schema_version": (
                        "anchor.gemma3-chat-unbalanced-v2-multiarm-error.v1"
                    ),
                    "status": "failed",
                    "error": str(error),
                    "model_loaded": False,
                    "gpu_touched": False,
                    "network_used": False,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
