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
            "source_identity_from_receipt_only": True,
            "candidate_sha256_hardcoded": False,
            "require_versioned_shard_inventory": True,
            "require_additive_logical_assets": True,
            "required_physical_files": 46,
            "required_payload_files": 23,
            "required_sidecar_files": 23,
            "required_main_chat_train_shards": 2,
            "required_train_shard_order": "continuous_bundle_boundary_v1",
            "require_final_producer_release": True,
            "require_body_read": False,
            "require_raw_token_ids_read": False,
            "require_gold_heldout_protected_read": False,
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
        or not isinstance(asset["shard_inventory_schema_version"], str)
        or re.fullmatch(
            r"^[a-z][a-z0-9._-]{2,127}$",
            asset["shard_inventory_schema_version"],
        )
        is None
    ):
        raise MultiArmContractError("preflight_asset_contract_invalid")
    result = dict(asset)
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


def load_consumer_preflight_receipt(
    path: str | Path,
) -> tuple[dict[str, Any], str, str]:
    """Authenticate the dynamic producer/consumer binding without data reads."""

    receipt, receipt_sha256 = _load_json(
        path,
        max_bytes=_MAX_RECEIPT_BYTES,
        code="consumer_preflight_receipt_invalid",
    )
    _exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "operation",
            "namespace",
            "model_free",
            "binding_contract_sha256",
            "producer_git_commit",
            "producer_manifest_schema_sha256",
            "producer_build_receipt_schema_sha256",
            "release_review_receipt_sha256",
            "tree_digest_sha256",
            "file_counts",
            "manifest",
            "build_receipt",
            "aggregate_counts",
            "release",
            "terminal_recheck",
            "resource_counters",
            "claims",
            "logical_assets",
        },
        "consumer_preflight_receipt_keys_invalid",
    )
    if (
        receipt["schema_version"] != PREFLIGHT_SCHEMA_VERSION
        or receipt["status"] != "passed"
        or receipt["operation"] not in {"validate", "dry-run"}
        or receipt["namespace"] != "gemma3_chat_five_expert_qonly_unbalanced_v2"
        or receipt["model_free"] is not True
    ):
        raise MultiArmContractError("consumer_preflight_not_passed")
    _require_commit(receipt["producer_git_commit"], "producer_commit_invalid")
    for key in (
        "binding_contract_sha256",
        "producer_manifest_schema_sha256",
        "producer_build_receipt_schema_sha256",
        "release_review_receipt_sha256",
        "tree_digest_sha256",
    ):
        _require_sha(receipt[key], f"{key}_invalid")
    file_counts = _mapping(
        receipt["file_counts"],
        "consumer_preflight_file_counts_invalid",
    )
    _exact_keys(
        file_counts,
        {"total", "payload", "sidecar"},
        "consumer_preflight_file_count_keys_invalid",
    )
    payload_count = _positive_int(
        file_counts["payload"],
        "consumer_preflight_payload_count_invalid",
    )
    sidecar_count = _positive_int(
        file_counts["sidecar"],
        "consumer_preflight_sidecar_count_invalid",
    )
    if (
        payload_count != sidecar_count
        or file_counts["total"] != (payload_count + sidecar_count)
        or file_counts != {"total": 46, "payload": 23, "sidecar": 23}
    ):
        raise MultiArmContractError("consumer_preflight_file_counts_invalid")
    for key in ("manifest", "build_receipt"):
        identity = _mapping(
            receipt[key],
            f"consumer_preflight_{key}_identity_invalid",
        )
        _exact_keys(
            identity,
            {"schema_version", "sha256", "bytes"},
            f"consumer_preflight_{key}_identity_keys_invalid",
        )
        if (
            not isinstance(identity["schema_version"], str)
            or not identity["schema_version"]
            or _positive_int(
                identity["bytes"],
                f"consumer_preflight_{key}_bytes_invalid",
            )
            < 1
        ):
            raise MultiArmContractError(f"consumer_preflight_{key}_identity_invalid")
        _require_sha(
            identity["sha256"],
            f"consumer_preflight_{key}_sha256_invalid",
        )
    _expect_mapping(
        receipt["aggregate_counts"],
        {
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
        "consumer_preflight_aggregate_counts_invalid",
    )
    release = _mapping(receipt["release"], "consumer_preflight_release_invalid")
    _exact_keys(
        release,
        {
            "manifest_status",
            "independent_release_review",
            "final",
            "release_authorized",
            "training_authorized",
            "formal_training_authorized",
        },
        "consumer_preflight_release_keys_invalid",
    )
    status = release["manifest_status"]
    if (
        not isinstance(status, str)
        or not status
        or "candidate" in status
        or "pending" in status
        or release["independent_release_review"] != "passed"
        or release["final"] is not True
        or release["release_authorized"] is not True
        or release["training_authorized"] is not True
        or release["formal_training_authorized"] is not False
    ):
        raise MultiArmContractError("consumer_preflight_release_not_final")
    _expect_mapping(
        receipt["terminal_recheck"],
        {
            "two_snapshot_bytes_equal": True,
            "stat_identity_equal": True,
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
    claims = _mapping(receipt["claims"], "consumer_preflight_claims_invalid")
    _expect_mapping(
        claims,
        {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "training_started": False,
            "data_copied": False,
            "sample_bodies_parsed": False,
            "raw_token_ids_parsed": False,
        },
        "consumer_preflight_claims_not_launchable",
    )
    logical_assets = _mapping(
        receipt["logical_assets"],
        "consumer_preflight_logical_assets_invalid",
    )
    _exact_keys(
        logical_assets,
        {"training", "evaluation"},
        "consumer_preflight_logical_asset_keys_invalid",
    )
    training = _mapping(
        logical_assets["training"],
        "consumer_preflight_training_assets_invalid",
    )
    if set(training) != set(TRAINING_ASSETS):
        raise MultiArmContractError("consumer_preflight_training_asset_names_invalid")
    normalized_training = {
        name: _validate_logical_asset(
            training[name],
            expected_records=count,
            training_eligible=True,
            require_shard_order=True,
            expected_shard_order=(
                "continuous_bundle_boundary_v1" if name != "planner_router" else None
            ),
        )
        for name, count in TRAINING_ASSETS.items()
    }
    main_chat_assets = (
        "humor",
        "serious",
        "angry_style",
        "review_audit",
        "tool_call",
    )
    if (
        any(normalized_training[name]["shard_count"] != 2 for name in main_chat_assets)
        or len(
            {
                normalized_training[name]["shard_inventory_sha256"]
                for name in main_chat_assets
            }
        )
        != 1
    ):
        raise MultiArmContractError("main_chat_two_shard_inventory_mismatch")
    evaluation = _mapping(
        logical_assets["evaluation"],
        "consumer_preflight_evaluation_assets_invalid",
    )
    if set(evaluation) != set(EVALUATION_ASSETS):
        raise MultiArmContractError("consumer_preflight_eval_asset_names_invalid")
    normalized_evaluation = {
        name: _validate_logical_asset(
            evaluation[name],
            expected_records=count,
            training_eligible=False,
            require_shard_order=False,
        )
        for name, count in EVALUATION_ASSETS.items()
    }
    normalized = {
        **{
            key: receipt[key]
            for key in (
                "schema_version",
                "status",
                "operation",
                "namespace",
                "model_free",
                "binding_contract_sha256",
                "producer_git_commit",
                "producer_manifest_schema_sha256",
                "producer_build_receipt_schema_sha256",
                "release_review_receipt_sha256",
                "tree_digest_sha256",
                "file_counts",
                "manifest",
                "build_receipt",
                "aggregate_counts",
                "release",
                "terminal_recheck",
                "resource_counters",
                "claims",
            )
        },
        "training_assets": normalized_training,
        "evaluation_assets": normalized_evaluation,
    }
    source_binding_sha256 = _canonical_sha256(normalized)
    return normalized, receipt_sha256, source_binding_sha256


def _validate_normalized_preflight_binding(
    value: object,
) -> dict[str, Any]:
    """Re-authenticate the body-free binding carried by a trusted context."""

    binding = _mapping(value, "trusted_preflight_binding_invalid")
    _exact_keys(
        binding,
        {
            "schema_version",
            "status",
            "operation",
            "namespace",
            "model_free",
            "binding_contract_sha256",
            "producer_git_commit",
            "producer_manifest_schema_sha256",
            "producer_build_receipt_schema_sha256",
            "release_review_receipt_sha256",
            "tree_digest_sha256",
            "file_counts",
            "manifest",
            "build_receipt",
            "aggregate_counts",
            "release",
            "terminal_recheck",
            "resource_counters",
            "claims",
            "training_assets",
            "evaluation_assets",
        },
        "trusted_preflight_binding_keys_invalid",
    )
    if (
        binding["schema_version"] != PREFLIGHT_SCHEMA_VERSION
        or binding["status"] != "passed"
        or binding["operation"] not in {"validate", "dry-run"}
        or binding["namespace"] != "gemma3_chat_five_expert_qonly_unbalanced_v2"
        or binding["model_free"] is not True
    ):
        raise MultiArmContractError("trusted_preflight_identity_invalid")
    _require_commit(binding["producer_git_commit"], "trusted_producer_commit_invalid")
    for key in (
        "binding_contract_sha256",
        "producer_manifest_schema_sha256",
        "producer_build_receipt_schema_sha256",
        "release_review_receipt_sha256",
        "tree_digest_sha256",
    ):
        _require_sha(binding[key], f"trusted_{key}_invalid")
    _expect_mapping(
        binding["file_counts"],
        {"total": 46, "payload": 23, "sidecar": 23},
        "trusted_file_counts_invalid",
    )
    for key in ("manifest", "build_receipt"):
        identity = _mapping(binding[key], f"trusted_{key}_invalid")
        _exact_keys(
            identity,
            {"schema_version", "sha256", "bytes"},
            f"trusted_{key}_keys_invalid",
        )
        if (
            not isinstance(identity["schema_version"], str)
            or not identity["schema_version"]
            or _positive_int(identity["bytes"], f"trusted_{key}_bytes_invalid") < 1
        ):
            raise MultiArmContractError(f"trusted_{key}_invalid")
        _require_sha(identity["sha256"], f"trusted_{key}_sha256_invalid")
    _expect_mapping(
        binding["aggregate_counts"],
        {
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
        "trusted_aggregate_counts_invalid",
    )
    release = _mapping(binding["release"], "trusted_release_identity_invalid")
    _exact_keys(
        release,
        {
            "manifest_status",
            "independent_release_review",
            "final",
            "release_authorized",
            "training_authorized",
            "formal_training_authorized",
        },
        "trusted_release_keys_invalid",
    )
    manifest_status = release["manifest_status"]
    if (
        not isinstance(manifest_status, str)
        or not manifest_status
        or "candidate" in manifest_status
        or "pending" in manifest_status
        or release["independent_release_review"] != "passed"
        or release["final"] is not True
        or release["release_authorized"] is not True
        or release["training_authorized"] is not True
        or release["formal_training_authorized"] is not False
    ):
        raise MultiArmContractError("trusted_release_identity_invalid")
    _expect_mapping(
        binding["terminal_recheck"],
        {
            "two_snapshot_bytes_equal": True,
            "stat_identity_equal": True,
            "physical_inventory_equal": True,
        },
        "trusted_terminal_recheck_invalid",
    )
    _expect_mapping(
        binding["resource_counters"],
        {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "gold_body_reads": 0,
            "heldout_body_reads": 0,
            "protected_body_reads": 0,
        },
        "trusted_resource_counters_invalid",
    )
    _expect_mapping(
        binding["claims"],
        {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "training_started": False,
            "data_copied": False,
            "sample_bodies_parsed": False,
            "raw_token_ids_parsed": False,
        },
        "trusted_claims_invalid",
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
                "continuous_bundle_boundary_v1" if name != "planner_router" else None
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
    normalized = dict(binding)
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

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "model_free_dry_run_passed_gpu_execution_not_implemented",
        "config_sha256": config_sha256,
        "dependency_set_sha256": dependency_set_sha256,
        "trusted_external_bindings_sha256": trusted_anchor,
        "consumer_preflight_receipt_sha256": preflight_receipt_sha256,
        "source_binding_sha256": source_binding_sha256,
        "tensor_inventory_sha256": tensor_inventory_sha256,
        "producer_commit": preflight["producer_git_commit"],
        "binding_contract_sha256": preflight["binding_contract_sha256"],
        "producer_manifest_schema_sha256": preflight["producer_manifest_schema_sha256"],
        "producer_build_receipt_schema_sha256": preflight[
            "producer_build_receipt_schema_sha256"
        ],
        "release_review_receipt_sha256": preflight["release_review_receipt_sha256"],
        "artifact_tree_sha256": preflight["tree_digest_sha256"],
        "source_manifest_sha256": preflight["manifest"]["sha256"],
        "source_build_receipt_sha256": preflight["build_receipt"]["sha256"],
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
            "producer_commit",
            "binding_contract_sha256",
            "producer_manifest_schema_sha256",
            "producer_build_receipt_schema_sha256",
            "release_review_receipt_sha256",
            "artifact_tree_sha256",
            "source_manifest_sha256",
            "source_build_receipt_sha256",
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
        or plan["producer_commit"] != external["producer_git_commit"]
        or plan["binding_contract_sha256"] != external["binding_contract_sha256"]
        or plan["producer_manifest_schema_sha256"]
        != external["producer_manifest_schema_sha256"]
        or plan["producer_build_receipt_schema_sha256"]
        != external["producer_build_receipt_schema_sha256"]
        or plan["release_review_receipt_sha256"]
        != external["release_review_receipt_sha256"]
        or plan["artifact_tree_sha256"] != external["tree_digest_sha256"]
        or plan["source_manifest_sha256"] != external["manifest"]["sha256"]
        or plan["source_build_receipt_sha256"] != external["build_receipt"]["sha256"]
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
