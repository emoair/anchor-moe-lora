"""Fail-closed Gemma 3 chat five-expert dataset consumer and Q8 trainer.

The producer artifact is intentionally not authenticated in the checked-in
configuration.  ``pending`` producer identities are valid state, but they
block every dataset-body read, GPU lock, tokenizer/model load, and training
action.  Once all five final identities are bound, the model-free preflight
authenticates the complete dataset and an explicit ``--execute`` may run five
strictly serial Q8-base/BF16-adapter diagnostics.

Heavy CUDA, transformers, PEFT, and bitsandbytes imports remain lazy so the
default dry-run is model-free and GPU-free.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from . import gemma3_five_role_q8_qlora_v2 as q8_runner
from . import gemma3_tokenizer_binding_v1 as gemma_binding
from . import qwen_lora_diagnostic as qdiag
from . import qwen_synthetic_scaffold_diagnostic as snapshot_io
from .config import ConfigError, _expand_env


CONFIG_VERSION = "anchor.gemma3-1b-it-chat-five-expert-qonly-consumer-config.v1"
RECORD_VERSION = "anchor.gemma3-chat-five-expert-qonly-record.v1"
MANIFEST_VERSION = "anchor.gemma3-chat-five-expert-qonly-manifest.v1"
PREFLIGHT_VERSION = "anchor.gemma3-1b-it-chat-five-expert-qonly-preflight-receipt.v1"
EXECUTE_BLOCK_VERSION = (
    "anchor.gemma3-1b-it-chat-five-expert-qonly-execute-block-receipt.v1"
)
PHASE_RECEIPT_VERSION = "anchor.gemma3-1b-it-chat-five-expert-qonly-phase-receipt.v1"
RUN_RECEIPT_VERSION = "anchor.gemma3-1b-it-chat-five-expert-qonly-run-receipt.v1"
FAILURE_RECEIPT_VERSION = (
    "anchor.gemma3-1b-it-chat-five-expert-qonly-failure-receipt.v1"
)
LOCK_OWNER_VERSION = "anchor.gemma3-1b-it-chat-five-expert-qonly-lock-owner.v1"
EXTERNAL_LOCK_ENV = {
    "receipt": "ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_RECEIPT",
    "sha256": "ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_SHA256",
    "nonce": "ANCHOR_CHAT_EXTERNAL_LOCK_NONCE",
    "launcher_pid": "ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID",
}
LOCK_OWNER_RECEIPT_DIRECTORY = "gpu-locks"
_MAX_WINDOWS_PID = (1 << 32) - 1

CONFIG_PATH = "configs/training/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1.yaml"
IMPLEMENTATION_PATH = "src/anchor_mvp/training/gemma3_chat_five_expert_qonly_v1.py"
SCRIPT_PATH = "scripts/research/run_gemma3_chat_five_expert_qonly_v1.py"
LAUNCHER_PATH = "scripts/research/run_gemma3_chat_five_expert_qonly_v1.ps1"
LAUNCHER_HELPER_PATH = "scripts/research/gemma3_q8_launcher_helpers_v2.ps1"
TOKENIZER_POLICY_PATH = "configs/research/gemma3_1b_it_chat_template_policy_v1.json"
EXECUTION_DEPENDENCY_PATHS = (
    "src/anchor_mvp/training/gemma3_tokenizer_binding_v1.py",
    "src/anchor_mvp/training/qwen_synthetic_five_role_qonly_v2.py",
    "src/anchor_mvp/research/synthetic_five_role_qonly_diagnostic_v1.py",
    "src/anchor_mvp/research/synthetic_nl_scaffold_diagnostic_v1.py",
    "src/anchor_mvp/training/gemma3_five_role_q8_qlora_v2.py",
    "src/anchor_mvp/training/qwen_lora_diagnostic.py",
    "src/anchor_mvp/training/qwen_synthetic_scaffold_diagnostic.py",
    "src/anchor_mvp/training/config.py",
    "src/anchor_mvp/training/gemma3_q8_reliability_v2.py",
    "src/anchor_mvp/research/gemma3_qonly_parameter_budget.py",
    "src/anchor_mvp/training/manifest.py",
    TOKENIZER_POLICY_PATH,
    LAUNCHER_HELPER_PATH,
)
RUNTIME_PACKAGE_NAMES = (
    "torch",
    "transformers",
    "peft",
    "safetensors",
    "sentencepiece",
    "bitsandbytes",
)
RECORD_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_record.schema.json"
)
MANIFEST_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_manifest.schema.json"
)
PRODUCER_CONFIG_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_config.schema.json"
)
CLOSED_GRAMMAR_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_closed_grammar.json"
)
CLOSED_GRAMMAR_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_closed_grammar.schema.json"
)
TOKEN_INVENTORY_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_token_inventory.schema.json"
)
BUILD_RECEIPT_SCHEMA_PATH = (
    "configs/research/gemma3_chat_five_expert_qonly_v1_build_receipt.schema.json"
)
PRODUCER_CONFIG_PATH = "configs/research/gemma3_chat_five_expert_qonly_v1.yaml"
PRODUCER_IMPLEMENTATION_PATH = (
    "src/anchor_mvp/research/gemma3_chat_five_expert_qonly_v1.py"
)
DATASET_ROOT = "fixtures/research/gemma3_chat_five_expert_distilled_v1"
MANIFEST_PATH = f"{DATASET_ROOT}/manifest.json"
MANIFEST_SIDECAR_PATH = f"{MANIFEST_PATH}.sha256"
BUILD_RECEIPT_PATH = f"{DATASET_ROOT}/build_receipt.json"
BUILD_RECEIPT_SIDECAR_PATH = f"{BUILD_RECEIPT_PATH}.sha256"
TOKEN_INVENTORY_PATH = f"{DATASET_ROOT}/token_inventory.jsonl"
PARTITION_PATHS = {
    "train": "train/chat.jsonl",
    "eval_proxy": "eval_proxy/chat.jsonl",
}
ROLES = ("humor", "serious", "angry_style", "tool_call", "review_audit")
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
_PENDING = "pending"
_PENDING_IDENTITIES = (
    ("producer.config_sha256", ("producer", "config_sha256")),
    ("producer.implementation_sha256", ("producer", "implementation_sha256")),
    ("dataset.record_schema_sha256", ("dataset", "record_schema_sha256")),
    ("dataset.manifest_schema_sha256", ("dataset", "manifest_schema_sha256")),
    ("dataset.manifest_sha256", ("dataset", "manifest_sha256")),
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_CONFIG_BYTES = 2_000_000
_MAX_METADATA_BYTES = 4_000_000
_MAX_PARTITION_BYTES = 100_000_000
_MAX_RECEIPT_BYTES = 8_000_000
_MIB = 1024 * 1024
MODEL_FILES = q8_runner.MODEL_FILES
ADAPTER_FILES = q8_runner.ADAPTER_FILES
ADAPTER_BASE_IDENTITY = q8_runner.ADAPTER_BASE_IDENTITY
Q8_BASE_CONTRACT = q8_runner.Q8_BASE_CONTRACT
Q8_SCB_INVENTORY_VERSION = q8_runner.Q8_SCB_INVENTORY_VERSION
OPTIMIZER_RUNTIME_CONTRACT = {
    **q8_runner.OPTIMIZER_RUNTIME_CONTRACT,
    "learning_rate": 0.000002,
    "warmup_steps": 8,
    "weight_decay": 0.01,
    "max_grad_norm": 0.5,
}
_LORA_TENSOR_RE = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.q_proj\."
    r"lora_([AB])(?:\.[^.]+)?\.weight$"
)


class ChatFiveExpertError(RuntimeError):
    """Raised when a chat training invariant fails closed."""


@dataclass(frozen=True)
class ChatExample:
    record_id: str
    task_bundle_sha256: str
    task_semantic_sha256: str
    split: str
    language: str
    role: str
    role_index: int
    messages: tuple[tuple[str, str], ...]
    messages_sha256: str
    user_message_sha256: str
    target: str
    target_format: str
    target_sha256: str
    serialized_prompt_sha256: str
    serialized_target_sha256: str
    serialized_training_example_sha256: str
    user_emotion_identity: bytes
    router_identity: bytes
    router_label: str
    tool_grounding: Mapping[str, Any]
    review_dependency: Mapping[str, Any]
    content_leak_proof: Mapping[str, Any]
    causal_proof: Mapping[str, Any]


@dataclass(frozen=True)
class AuthenticatedDataset:
    manifest_sha256: str
    partition_sha256: Mapping[str, str]
    records: tuple[ChatExample, ...]
    source_snapshots: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ArtifactSnapshot:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int]

    def assert_unchanged(self) -> None:
        current = _capture_artifact_snapshot(
            self.path,
            expected_sha256=self.sha256,
        )
        if current.identity != self.identity:
            raise ChatFiveExpertError("chat_training_artifact_identity_changed")


@dataclass(frozen=True)
class RoleDataset:
    role: str
    manifest_sha256: str
    partition_sha256: Mapping[str, str]
    global_records_authenticated: int
    global_task_bundles_authenticated: int
    train: tuple[ChatExample, ...]
    eval_proxy: tuple[ChatExample, ...]


@dataclass(frozen=True)
class SerializedChatExample:
    record_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class SerializedRoleDataset:
    role: str
    train: tuple[SerializedChatExample, ...]
    eval_proxy: tuple[SerializedChatExample, ...]


def _fail(code: str) -> None:
    raise ConfigError(code)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _identity(value: object, code: str, *, pending_allowed: bool) -> str:
    if pending_allowed and value == _PENDING:
        return _PENDING
    if (
        not isinstance(value, str)
        or _SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail(code)
    return value


def _canonical_config_path(path: str | Path) -> Path:
    root = qdiag._project_root_from_module()
    canonical = root.joinpath(*PurePosixPath(CONFIG_PATH).parts)
    qdiag._assert_physical_path(
        canonical,
        require_file=True,
        label="Gemma chat five-expert config",
    )
    requested = Path(path)
    resolved = (
        Path(os.path.abspath(requested))
        if requested.is_absolute()
        else root.joinpath(*PurePosixPath(requested.as_posix()).parts)
    )
    resolved = Path(os.path.abspath(resolved))
    qdiag._assert_physical_path(
        resolved,
        require_file=True,
        label="Gemma chat five-expert requested config",
    )
    if os.path.normcase(str(resolved)) != os.path.normcase(str(canonical)):
        _fail("chat_config_path_invalid")
    return canonical


def _repo_path(
    value: object,
    expected: str,
    *,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    if value != expected:
        _fail("chat_repo_path_drift")
    relative = PurePosixPath(expected)
    if relative.is_absolute() or ".." in relative.parts:
        _fail("chat_repo_path_unsafe")
    path = qdiag._project_root_from_module().joinpath(*relative.parts)
    return qdiag._assert_physical_path(
        path,
        require_file=require_file,
        require_directory=require_directory,
        label=expected,
    )


def _nested(config: Mapping[str, Any], path: Sequence[str]) -> object:
    value: object = config
    for part in path:
        value = _mapping(value, "chat_identity_parent_invalid").get(part)
    return value


def pending_producer_identities(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        label
        for label, path in _PENDING_IDENTITIES
        if _nested(config, path) == _PENDING
    )


def validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "claim_scope",
            "identity_state",
            "paths",
            "producer",
            "model",
            "dataset",
            "experts",
            "lora",
            "quantization",
            "training",
            "runner",
            "gpu_policy",
            "output",
            "claims",
            "_config_path",
            "_config_sha256",
        },
        "chat_config_fields_drift",
    )
    identity_state = config.get("identity_state")
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope")
        not in {
            "diagnostic_training_engine_waiting_for_authenticated_distilled_dataset",
            (
                "diagnostic_training_engine_with_authenticated_distilled_dataset_"
                "waiting_for_explicit_execute"
            ),
        }
        or identity_state
        not in {"pending_producer_handoff", "authenticated_producer_handoff"}
        or _mapping(config.get("paths"), "chat_paths_invalid")
        != {"project_root": "../.."}
    ):
        _fail("chat_config_identity_drift")

    producer = _mapping(config.get("producer"), "chat_producer_invalid")
    if producer != {
        "namespace": "anchor.gemma3-chat-five-expert-qonly.v1",
        "config": PRODUCER_CONFIG_PATH,
        "config_sha256": producer.get("config_sha256"),
        "config_schema": PRODUCER_CONFIG_SCHEMA_PATH,
        "config_schema_sha256": producer.get("config_schema_sha256"),
        "closed_grammar": CLOSED_GRAMMAR_PATH,
        "closed_grammar_sha256": producer.get("closed_grammar_sha256"),
        "closed_grammar_schema": CLOSED_GRAMMAR_SCHEMA_PATH,
        "closed_grammar_schema_sha256": producer.get("closed_grammar_schema_sha256"),
        "implementation": PRODUCER_IMPLEMENTATION_PATH,
        "implementation_sha256": producer.get("implementation_sha256"),
    }:
        _fail("chat_producer_contract_drift")
    _identity(
        producer.get("config_sha256"),
        "chat_producer_config_sha_invalid",
        pending_allowed=True,
    )
    _identity(
        producer.get("implementation_sha256"),
        "chat_producer_implementation_sha_invalid",
        pending_allowed=True,
    )
    for field in (
        "config_schema_sha256",
        "closed_grammar_sha256",
        "closed_grammar_schema_sha256",
    ):
        _identity(
            producer.get(field),
            f"chat_producer_{field}_invalid",
            pending_allowed=False,
        )

    model = _mapping(config.get("model"), "chat_model_invalid")
    _exact_keys(
        model,
        {
            "local_path",
            "architecture",
            "model_type",
            "parameter_count",
            "layers",
            "hidden_size",
            "q_proj_input_features",
            "q_proj_output_features",
            "local_files_only",
            "allow_network",
            "trust_remote_code",
            "tokenizer_policy",
            "tokenizer_policy_sha256",
            "adapter_base_identity",
            "files",
            "runtime_special_token_overlay",
        },
        "chat_model_fields_drift",
    )
    if (
        model.get("architecture") != "Gemma3ForCausalLM"
        or model.get("model_type") != "gemma3_text"
        or model.get("parameter_count") != 999_885_952
        or model.get("layers") != 26
        or model.get("hidden_size") != 1152
        or model.get("q_proj_input_features") != 1152
        or model.get("q_proj_output_features") != 1024
        or model.get("local_files_only") is not True
        or model.get("allow_network") is not False
        or model.get("trust_remote_code") is not False
        or model.get("tokenizer_policy") != TOKENIZER_POLICY_PATH
        or model.get("tokenizer_policy_sha256")
        != "0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6"
        or model.get("adapter_base_identity")
        != (
            "anchor.local/gemma3-1b-it-keras-v3-bf16@sha256:"
            "c9c6e309cf0158050d1e1abcba19eb6798153468572af2cd91de163e74933df9"
        )
    ):
        _fail("chat_model_contract_drift")
    configured_model_files: dict[str, tuple[int, str]] = {}
    for raw in _sequence(model.get("files"), "chat_model_files_invalid"):
        item = _mapping(raw, "chat_model_file_invalid")
        _exact_keys(
            item,
            {"path", "bytes", "sha256"},
            "chat_model_file_fields_drift",
        )
        name = item.get("path")
        size = item.get("bytes")
        if (
            not isinstance(name, str)
            or name in configured_model_files
            or type(size) is not int
            or size <= 0
        ):
            _fail("chat_model_file_invalid")
        configured_model_files[name] = (
            size,
            _identity(
                item.get("sha256"),
                "chat_model_file_sha_invalid",
                pending_allowed=False,
            ),
        )
    if tuple(configured_model_files) != MODEL_FILES:
        _fail("chat_model_file_inventory_drift")
    if model.get("runtime_special_token_overlay") != {
        "canonical_files_modified": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
        "unk_token_id": 3,
    }:
        _fail("chat_model_special_token_overlay_drift")

    dataset = _mapping(config.get("dataset"), "chat_dataset_invalid")
    _exact_keys(
        dataset,
        {
            "kind",
            "root",
            "manifest",
            "manifest_sidecar",
            "manifest_sidecar_sha256",
            "build_receipt",
            "build_receipt_sha256",
            "build_receipt_sidecar",
            "build_receipt_sidecar_sha256",
            "record_schema",
            "record_schema_sha256",
            "manifest_schema",
            "manifest_schema_sha256",
            "token_inventory_schema",
            "token_inventory_schema_sha256",
            "build_receipt_schema",
            "build_receipt_schema_sha256",
            "manifest_sha256",
            "token_inventory",
            "token_inventory_sha256",
            "partitions",
            "partition_sha256",
            "records",
            "task_bundles",
            "records_per_bundle",
            "train_bundles",
            "eval_proxy_bundles",
            "train_records",
            "eval_proxy_records",
            "train_records_per_role",
            "eval_proxy_records_per_role",
            "split_group_key",
            "eval_proxy_is_heldout",
            "formal_inputs_allowed",
            "heldout_allowed",
            "protected_source_paths_allowed",
        },
        "chat_dataset_fields_drift",
    )
    if (
        dataset.get("kind") != "gemma3_chat_five_expert_distilled_v1"
        or dataset.get("root") != DATASET_ROOT
        or dataset.get("manifest") != MANIFEST_PATH
        or dataset.get("manifest_sidecar") != MANIFEST_SIDECAR_PATH
        or dataset.get("build_receipt") != BUILD_RECEIPT_PATH
        or dataset.get("build_receipt_sidecar") != BUILD_RECEIPT_SIDECAR_PATH
        or dataset.get("record_schema") != RECORD_SCHEMA_PATH
        or dataset.get("manifest_schema") != MANIFEST_SCHEMA_PATH
        or dataset.get("token_inventory_schema") != TOKEN_INVENTORY_SCHEMA_PATH
        or dataset.get("build_receipt_schema") != BUILD_RECEIPT_SCHEMA_PATH
        or dataset.get("token_inventory") != TOKEN_INVENTORY_PATH
        or _mapping(dataset.get("partitions"), "chat_partitions_invalid")
        != PARTITION_PATHS
        or set(
            _mapping(
                dataset.get("partition_sha256"),
                "chat_partition_sha256_invalid",
            )
        )
        != set(PARTITION_PATHS.values())
        or dataset.get("records") != 1000
        or dataset.get("task_bundles") != 200
        or dataset.get("records_per_bundle") != 5
        or dataset.get("train_bundles") != 160
        or dataset.get("eval_proxy_bundles") != 40
        or dataset.get("train_records") != 800
        or dataset.get("eval_proxy_records") != 200
        or dataset.get("train_records_per_role") != 160
        or dataset.get("eval_proxy_records_per_role") != 40
        or dataset.get("split_group_key") != "task_bundle_sha256"
        or dataset.get("eval_proxy_is_heldout") is not False
        or dataset.get("formal_inputs_allowed") is not False
        or dataset.get("heldout_allowed") is not False
        or dataset.get("protected_source_paths_allowed") is not False
    ):
        _fail("chat_dataset_contract_drift")
    for field in (
        "manifest_sidecar_sha256",
        "build_receipt_sha256",
        "build_receipt_sidecar_sha256",
        "record_schema_sha256",
        "manifest_schema_sha256",
        "token_inventory_schema_sha256",
        "build_receipt_schema_sha256",
        "manifest_sha256",
        "token_inventory_sha256",
    ):
        _identity(
            dataset.get(field),
            f"chat_dataset_{field}_invalid",
            pending_allowed=field
            in {
                "record_schema_sha256",
                "manifest_schema_sha256",
                "manifest_sha256",
            },
        )
    for path, digest in _mapping(
        dataset.get("partition_sha256"),
        "chat_partition_sha256_invalid",
    ).items():
        if path not in PARTITION_PATHS.values():
            _fail("chat_partition_sha256_inventory_drift")
        _identity(
            digest,
            "chat_partition_sha256_invalid",
            pending_allowed=False,
        )

    experts = _mapping(config.get("experts"), "chat_experts_invalid")
    if experts != {
        "ordered": list(ROLES),
        "role_index": ROLE_INDEX,
        "user_emotion_is_expert": False,
        "router_label_is_expert": False,
        "user_emotion_shared_within_bundle": True,
        "router_label_shared_within_bundle": True,
    }:
        _fail("chat_expert_contract_drift")

    lora = _mapping(config.get("lora"), "chat_lora_invalid")
    if lora != {
        "profile": "q_only",
        "factorization": "full_rank_q",
        "target_modules": ["q_proj"],
        "rank": 1024,
        "alpha": 2048,
        "dropout": 0.0,
        "bias": "none",
        "expected_trainable_parameters_per_expert": 57_933_824,
        "expected_five_expert_aggregate_parameters": 289_669_120,
        "effective_rank_ceiling": 1024,
        "low_rank_claimed": False,
        "parameter_efficient_claimed": False,
        "o_proj_allowed": False,
    }:
        _fail("chat_lora_must_be_q_only_rank1024")
    calculated = (
        int(model["layers"])
        * int(lora["rank"])
        * (int(model["q_proj_input_features"]) + int(model["q_proj_output_features"]))
    )
    if calculated != int(lora["expected_trainable_parameters_per_expert"]):
        _fail("chat_lora_parameter_math_invalid")

    quantization = _mapping(config.get("quantization"), "chat_quantization_invalid")
    if dict(quantization) != Q8_BASE_CONTRACT:
        _fail("chat_q8_base_contract_drift")

    training = _mapping(config.get("training"), "chat_training_invalid")
    if training != {
        "sequence_length": 768,
        "truncation": False,
        "padding": "microbatch_exact_length",
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "smoke_steps": 2,
        "full_steps_per_role": 160,
        "one_full_epoch_per_role": True,
        "role_execution": "strictly_serial",
        "concurrency": 1,
        "smoke_and_full_fresh_objects": True,
        "smoke_checkpoint_consumed_by_full": False,
        "resume": False,
        "optimizer": "adamw8bit",
        "optimizer_library": "bitsandbytes",
        "bitsandbytes_version": "0.48.2",
        "optimizer_state_bits": 8,
        "compatibility_optim_bits_argument": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
        "amsgrad": False,
        "learning_rate": 0.000002,
        "warmup_steps": 8,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 0.00000001,
        "weight_decay": 0.01,
        "max_grad_norm": 0.5,
        "seed": 1337,
        "compute_dtype": "bfloat16",
        "tf32": True,
        "attention_implementation": "sdpa",
        "gradient_checkpointing": True,
        "use_cache": False,
        "loss_projection": "supervised_positions_only_exact_v1",
        "save_intermediate_checkpoints": False,
        "adapter_effect_gate": {
            "view": "first_train_record_first_supervised_next_token_v1",
            "comparison": "enabled_vs_disable_adapter_after_training",
            "require_finite": True,
            "require_max_abs_gt_zero": True,
            "sample_body_included": False,
            "token_ids_included": False,
        },
    }:
        _fail("chat_training_contract_drift")

    runner = _mapping(config.get("runner"), "chat_runner_invalid")
    runner_status = runner.get("status")
    if runner != {
        "status": runner_status,
        "preflight_supported": True,
        "execution_supported": True,
        "explicit_execute_flag_required": True,
        "canonical_gpu_lock_required": True,
        "authenticate_all_records_before_role_selection": True,
        "single_bytes_snapshot": True,
        "final_snapshot_recheck": True,
        "fresh_base_per_phase": True,
        "fresh_adapter_per_phase": True,
        "atomic_staging_publish": True,
        "replace_existing": False,
        "failure_receipt_retained": True,
    } or runner_status not in {
        "engine_ready_waiting_for_authenticated_producer",
        "authenticated_dataset_ready_waiting_for_explicit_execute",
    }:
        _fail("chat_runner_contract_drift")

    gpu = _mapping(config.get("gpu_policy"), "chat_gpu_policy_invalid")
    if (
        gpu.get("canonical_lock") != "runs/formal-v3-training.lock"
        or tuple(gpu.get("conflicting_handoff_locks", ()))
        != (
            "runs/distill-train-handoff/gpu-job.lock",
            "runs/distill-train-handoff-v3/gpu-job.lock",
        )
        or gpu.get("expected_gpu_index") != 0
        or not isinstance(gpu.get("expected_gpu_uuid"), str)
        or gpu.get("expected_total_memory_mib") != 12288
        or gpu.get("command_timeout_seconds") != 5
        or gpu.get("runtime_temperature_max_c") != 83
        or gpu.get("torch_peak_allocated_max_mib") != 10240
        or gpu.get("torch_peak_reserved_max_mib") != 10240
        or gpu.get("runtime_monitor_interval_steps") != 10
        or gpu.get("unknown_or_non_allowlisted_compute_process_forbidden") is not False
    ):
        _fail("chat_gpu_policy_drift")

    output = _mapping(config.get("output"), "chat_output_invalid")
    if output != {
        "artifact_root": (
            "artifacts/diagnostics/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1"
        ),
        "run_root": "runs/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1",
        "preflight_receipt_name": "preflight_receipt.json",
        "run_receipt_name": "run_receipt.json",
        "failure_receipt_name": "failure_receipt.json",
        "adapter_format": "peft_safetensors",
    }:
        _fail("chat_output_contract_drift")

    claims = _mapping(config.get("claims"), "chat_claims_invalid")
    pending_claims = {
        "diagnostic_only": True,
        "dataset_materialized": False,
        "producer_identity_bound": False,
        "training_execution_supported": True,
        "training_executed": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "quality_claimed": False,
        "eval_proxy_is_heldout": False,
    }
    authenticated_claims = {
        **pending_claims,
        "dataset_materialized": True,
        "producer_identity_bound": True,
    }
    pending = pending_producer_identities(config)
    all_pending = {label for label, _path in _PENDING_IDENTITIES}
    if (
        pending
        and (
            set(pending) != all_pending
            or identity_state != "pending_producer_handoff"
            or claims != pending_claims
            or config.get("claim_scope")
            != (
                "diagnostic_training_engine_waiting_for_authenticated_distilled_dataset"
            )
            or runner_status != "engine_ready_waiting_for_authenticated_producer"
        )
    ) or (
        not pending
        and (
            identity_state != "authenticated_producer_handoff"
            or claims != authenticated_claims
            or config.get("claim_scope")
            != (
                "diagnostic_training_engine_with_authenticated_distilled_"
                "dataset_waiting_for_explicit_execute"
            )
            or runner_status
            != "authenticated_dataset_ready_waiting_for_explicit_execute"
        )
    ):
        _fail("chat_claims_must_remain_blocked")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    canonical = _canonical_config_path(path)
    snapshot = snapshot_io._read_snapshot(canonical, max_bytes=_MAX_CONFIG_BYTES)
    try:
        parsed = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError("chat_config_invalid_utf8_yaml") from exc
    if not isinstance(parsed, Mapping):
        _fail("chat_config_not_mapping")
    config = _expand_env(dict(parsed))
    config["_config_path"] = str(canonical)
    config["_config_sha256"] = snapshot.sha256
    validate_config(config)
    snapshot.assert_unchanged()
    return config


def _strict_messages(value: object) -> tuple[tuple[str, str], ...]:
    raw = _sequence(value, "chat_messages_invalid")
    if len(raw) != 2:
        _fail("chat_messages_cardinality_invalid")
    messages: list[tuple[str, str]] = []
    for item in raw:
        message = _mapping(item, "chat_message_invalid")
        _exact_keys(message, {"role", "content"}, "chat_message_fields_drift")
        role = message.get("role")
        content = message.get("content")
        if (
            role not in {"system", "user"}
            or not isinstance(content, str)
            or not content
        ):
            _fail("chat_assistant_or_invalid_input_message_forbidden")
        messages.append((str(role), content))
    if tuple(role for role, _content in messages) != ("system", "user"):
        _fail("chat_messages_must_be_exact_system_then_user")
    return tuple(messages)


def _decode_strict_json(text: str) -> tuple[bool, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("nonfinite")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("nonfinite")
        return parsed

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (json.JSONDecodeError, ValueError):
        return False, None
    return True, parsed


def _strict_json_object(
    text: str,
    code: str,
    *,
    canonical: bool = False,
) -> Mapping[str, Any]:
    valid_json, parsed = _decode_strict_json(text)
    if not valid_json or not isinstance(parsed, Mapping):
        _fail(code)
    if canonical and _canonical_json(parsed).decode("utf-8") != text:
        _fail(code)
    return parsed


def _strict_target_shape(role: str, target_text: str) -> None:
    valid_json, parsed = _decode_strict_json(target_text)
    if role in {"humor", "serious", "angry_style"}:
        if valid_json and isinstance(parsed, (Mapping, list)):
            _fail("chat_natural_language_target_must_not_be_json_envelope")
        return
    if not valid_json or not isinstance(parsed, Mapping):
        _fail("chat_structured_target_must_be_strict_json_object")
    if _canonical_json(parsed).decode("utf-8") != target_text:
        _fail("chat_structured_target_must_be_canonical_json_object")


def _message_objects(
    messages: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in messages]


def _last_user_content(messages: Sequence[tuple[str, str]]) -> str:
    for role, content in reversed(messages):
        if role == "user":
            return content
    raise AssertionError("validated messages always contain a user turn")


def _validate_tool_grounding(role: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys(
        value,
        {
            "required",
            "tool_schema_refs",
            "evidence_refs",
            "grounding_sha256",
            "real_tool_executions",
        },
        "chat_tool_grounding_fields_drift",
    )
    schemas = [
        dict(_mapping(item, "chat_tool_schema_ref_invalid"))
        for item in _sequence(
            value.get("tool_schema_refs"), "chat_tool_schema_refs_invalid"
        )
    ]
    evidence = [
        dict(_mapping(item, "chat_tool_evidence_ref_invalid"))
        for item in _sequence(value.get("evidence_refs"), "chat_evidence_refs_invalid")
    ]
    for item in (*schemas, *evidence):
        _exact_keys(item, {"ref_id", "sha256"}, "chat_content_ref_fields_drift")
        if not isinstance(item.get("ref_id"), str) or not item["ref_id"]:
            _fail("chat_content_ref_id_invalid")
        _identity(
            item.get("sha256"),
            "chat_content_ref_sha_invalid",
            pending_allowed=False,
        )
    if value.get("real_tool_executions") != 0:
        _fail("chat_real_tool_execution_forbidden")
    if role == "tool_call":
        required = value.get("required")
        if required is True:
            if (
                not schemas
                or not evidence
                or _SHA256_RE.fullmatch(str(value.get("grounding_sha256"))) is None
            ):
                _fail("chat_tool_call_grounding_invalid")
        elif (
            required is not False
            or schemas
            or evidence
            or value.get("grounding_sha256") is not None
        ):
            _fail("chat_tool_call_grounding_invalid")
    elif (
        value.get("required") is not False
        or schemas
        or evidence
        or value.get("grounding_sha256") is not None
    ):
        _fail("chat_non_tool_role_grounding_forbidden")
    return value


def _validate_review_dependency(
    role: str, value: Mapping[str, Any]
) -> Mapping[str, Any]:
    _exact_keys(
        value,
        {
            "required",
            "dependencies",
            "dependency_set_sha256",
            "committed_projection_sha256",
            "mutation_sha256",
            "fault_type",
        },
        "chat_review_dependency_fields_drift",
    )
    dependencies = [
        dict(_mapping(item, "chat_review_dependency_entry_invalid"))
        for item in _sequence(
            value.get("dependencies"), "chat_review_dependencies_invalid"
        )
    ]
    for item in dependencies:
        _exact_keys(
            item,
            {"record_id", "role", "target_sha256"},
            "chat_review_dependency_entry_fields_drift",
        )
        if (
            not isinstance(item.get("record_id"), str)
            or not item["record_id"]
            or item.get("role") not in ROLES
            or item.get("role") == "review_audit"
        ):
            _fail("chat_review_dependency_identity_invalid")
        _identity(
            item.get("target_sha256"),
            "chat_review_dependency_sha_invalid",
            pending_allowed=False,
        )
    expected = _sha256(
        _canonical_json(
            {
                "domain": "anchor.chat-review-dependency-set.v1",
                "dependencies": dependencies,
            }
        )
    )
    if role == "review_audit":
        if (
            value.get("required") is not True
            or len(dependencies) != 4
            or tuple(item["role"] for item in dependencies) != ROLES[:4]
            or value.get("dependency_set_sha256") != expected
            or _SHA256_RE.fullmatch(str(value.get("committed_projection_sha256")))
            is None
            or _SHA256_RE.fullmatch(str(value.get("mutation_sha256"))) is None
            or value.get("fault_type")
            not in {None, "format", "grounding", "routing", "style"}
        ):
            _fail("chat_review_dependency_required")
    elif (
        value.get("required") is not False
        or dependencies
        or value.get("dependency_set_sha256") is not None
        or value.get("committed_projection_sha256") is not None
        or value.get("mutation_sha256") is not None
        or value.get("fault_type") is not None
    ):
        _fail("chat_non_review_dependency_forbidden")
    return value


def _strict_record(value: Mapping[str, Any]) -> ChatExample:
    _exact_keys(
        value,
        {
            "schema_version",
            "record_id",
            "task_bundle_sha256",
            "task_semantic_sha256",
            "split",
            "language",
            "role",
            "role_index",
            "user_emotion",
            "router",
            "messages",
            "target",
            "gemma_serialization",
            "tool_grounding",
            "review_dependency",
            "content_leak_proof",
            "causal_proof",
            "claims",
            "audit",
        },
        "chat_record_fields_drift",
    )
    if value.get("schema_version") != RECORD_VERSION:
        _fail("chat_record_version_invalid")
    record_id = value.get("record_id")
    bundle_sha = _identity(
        value.get("task_bundle_sha256"),
        "chat_bundle_sha_invalid",
        pending_allowed=False,
    )
    semantic_sha = _identity(
        value.get("task_semantic_sha256"),
        "chat_semantic_sha_invalid",
        pending_allowed=False,
    )
    split = value.get("split")
    language = value.get("language")
    role = value.get("role")
    role_index = value.get("role_index")
    if (
        not isinstance(record_id, str)
        or not record_id
        or split not in PARTITION_PATHS
        or language not in {"en", "zh-CN"}
        or role not in ROLES
        or role_index != ROLE_INDEX[role]
    ):
        _fail("chat_record_identity_invalid")

    messages = _strict_messages(value.get("messages"))
    message_objects = _message_objects(messages)
    messages_sha = _sha256(_canonical_json(message_objects))
    user_emotion = _mapping(value.get("user_emotion"), "chat_user_emotion_invalid")
    _exact_keys(
        user_emotion,
        {
            "taxonomy_version",
            "label",
            "confidence",
            "evidence_message_sha256",
        },
        "chat_user_emotion_fields_drift",
    )
    confidence = user_emotion.get("confidence")
    label = user_emotion.get("label")
    if (
        user_emotion.get("taxonomy_version") != "anchor.user-emotion.v1"
        or not isinstance(label, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{1,31}", label) is None
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
        or user_emotion.get("evidence_message_sha256")
        != _sha256(_last_user_content(messages).encode("utf-8"))
    ):
        _fail("chat_user_emotion_binding_invalid")

    router = _mapping(value.get("router"), "chat_router_invalid")
    _exact_keys(
        router,
        {
            "policy_version",
            "label",
            "confidence",
            "user_message_sha256",
        },
        "chat_router_fields_drift",
    )
    router_confidence = router.get("confidence")
    if (
        router.get("policy_version") != "anchor.chat-five-expert-router.v1"
        or router.get("label") not in ROLES
        or isinstance(router_confidence, bool)
        or not isinstance(router_confidence, (int, float))
        or not 0 <= float(router_confidence) <= 1
        or router.get("user_message_sha256")
        != _sha256(_last_user_content(messages).encode("utf-8"))
    ):
        _fail("chat_router_binding_invalid")

    target = _mapping(value.get("target"), "chat_target_invalid")
    _exact_keys(
        target,
        {"assistant_text", "format", "output_sha256"},
        "chat_target_fields_drift",
    )
    target_text = target.get("assistant_text")
    target_format = target.get("format")
    if not isinstance(target_text, str) or not target_text:
        _fail("chat_target_text_invalid")
    target_sha = _sha256(target_text.encode("utf-8"))
    expected_format = (
        "tool_call_json"
        if role == "tool_call"
        else "review_audit_json"
        if role == "review_audit"
        else "chat_text"
    )
    if target_format != expected_format or target.get("output_sha256") != target_sha:
        _fail("chat_target_role_or_digest_invalid")
    _strict_target_shape(str(role), target_text)

    serialization = _mapping(
        value.get("gemma_serialization"), "chat_gemma_serialization_invalid"
    )
    _exact_keys(
        serialization,
        {
            "contract_version",
            "chat_template_policy_sha256",
            "input_messages_sha256",
            "serialized_prompt_sha256",
            "serialized_target_sha256",
            "serialized_training_example_sha256",
            "assistant_prefix_included",
            "add_generation_prompt",
            "token_ids_included",
        },
        "chat_gemma_serialization_fields_drift",
    )
    if (
        serialization.get("contract_version")
        != "anchor.gemma3-it-chat-sft-serialization.v1"
        or serialization.get("chat_template_policy_sha256")
        != "0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6"
        or serialization.get("input_messages_sha256") != messages_sha
        or serialization.get("assistant_prefix_included") is not True
        or serialization.get("add_generation_prompt") is not True
        or serialization.get("token_ids_included") is not False
    ):
        _fail("chat_gemma_serialization_binding_invalid")
    prompt = _messages_prompt(messages)
    training_identity = {
        "contract_version": "anchor.gemma3-it-chat-sft-serialization.v1",
        "messages": _message_objects(messages),
        "serialized_prompt": prompt,
        "assistant_text": target_text,
    }
    if (
        serialization.get("serialized_prompt_sha256") != _sha256(prompt.encode("utf-8"))
        or serialization.get("serialized_target_sha256") != target_sha
        or serialization.get("serialized_training_example_sha256")
        != _sha256(_canonical_json(training_identity))
    ):
        _fail("chat_gemma_serialization_identity_invalid")

    tool_grounding = _validate_tool_grounding(
        str(role),
        _mapping(value.get("tool_grounding"), "chat_tool_grounding_invalid"),
    )
    review_dependency = _validate_review_dependency(
        str(role),
        _mapping(value.get("review_dependency"), "chat_review_dependency_invalid"),
    )

    leak = _mapping(value.get("content_leak_proof"), "chat_content_leak_proof_invalid")
    _exact_keys(
        leak,
        {
            "method",
            "forbidden_target_sha256",
            "exact_target_text_absent",
            "target_digest_literal_absent",
        },
        "chat_content_leak_proof_fields_drift",
    )
    forbidden = _sequence(
        leak.get("forbidden_target_sha256"),
        "chat_forbidden_target_inventory_invalid",
    )
    for digest in forbidden:
        _identity(
            digest,
            "chat_forbidden_target_sha_invalid",
            pending_allowed=False,
        )
    if (
        leak.get("method") != "exact_target_and_digest_scan_v1"
        or not forbidden
        or len(set(forbidden)) != len(forbidden)
        or leak.get("exact_target_text_absent") is not True
        or leak.get("target_digest_literal_absent") is not True
    ):
        _fail("chat_content_leak_proof_invalid")

    causal = _mapping(value.get("causal_proof"), "chat_causal_proof_invalid")
    _exact_keys(
        causal,
        {
            "contract_version",
            "input_messages_sha256",
            "own_target_sha256",
            "allowed_dependency_target_sha256",
            "no_post_target_messages",
            "proof_sha256",
        },
        "chat_causal_proof_fields_drift",
    )
    allowed = _sequence(
        causal.get("allowed_dependency_target_sha256"),
        "chat_allowed_dependency_inventory_invalid",
    )
    for digest in allowed:
        _identity(
            digest,
            "chat_allowed_dependency_sha_invalid",
            pending_allowed=False,
        )
    causal_body = {key: causal[key] for key in causal if key != "proof_sha256"}
    if (
        causal.get("contract_version") != "anchor.chat-five-expert-causal-filter.v1"
        or causal.get("input_messages_sha256") != messages_sha
        or causal.get("own_target_sha256") != target_sha
        or len(set(allowed)) != len(allowed)
        or causal.get("no_post_target_messages") is not True
        or causal.get("proof_sha256")
        != _sha256(
            _canonical_json(
                {
                    "domain": "anchor.chat-five-expert-causal-proof.v1",
                    "proof": causal_body,
                }
            )
        )
    ):
        _fail("chat_causal_proof_binding_invalid")
    dependency_hashes = [
        str(item["target_sha256"])
        for item in _sequence(
            review_dependency.get("dependencies"),
            "chat_review_dependencies_invalid",
        )
    ]
    if list(allowed) != dependency_hashes:
        _fail("chat_causal_allowed_dependency_mismatch")

    claims = _mapping(value.get("claims"), "chat_record_claims_invalid")
    if claims != {
        "distilled_chat_sft": True,
        "user_emotion_is_expert": False,
        "router_label_is_expert": False,
        "training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
    }:
        _fail("chat_record_claims_boundary_invalid")
    audit = _mapping(value.get("audit"), "chat_record_audit_invalid")
    _exact_keys(
        audit,
        {
            "protected_body_reads",
            "heldout_reads",
            "provider_requests",
            "network_requests",
            "model_loads",
            "gpu_requests",
            "real_tool_executions",
        },
        "chat_record_audit_fields_drift",
    )
    if (
        audit.get("protected_body_reads") != 0
        or audit.get("heldout_reads") != 0
        or audit.get("model_loads") != 0
        or audit.get("gpu_requests") != 0
        or audit.get("real_tool_executions") != 0
        or not isinstance(audit.get("provider_requests"), int)
        or int(audit["provider_requests"]) < 0
        or not isinstance(audit.get("network_requests"), int)
        or int(audit["network_requests"]) < 0
    ):
        _fail("chat_record_audit_boundary_invalid")

    return ChatExample(
        record_id=record_id,
        task_bundle_sha256=bundle_sha,
        task_semantic_sha256=semantic_sha,
        split=str(split),
        language=str(language),
        role=str(role),
        role_index=int(role_index),
        messages=messages,
        messages_sha256=messages_sha,
        user_message_sha256=_sha256(_last_user_content(messages).encode("utf-8")),
        target=target_text,
        target_format=str(target_format),
        target_sha256=target_sha,
        serialized_prompt_sha256=str(serialization["serialized_prompt_sha256"]),
        serialized_target_sha256=str(serialization["serialized_target_sha256"]),
        serialized_training_example_sha256=str(
            serialization["serialized_training_example_sha256"]
        ),
        user_emotion_identity=_canonical_json(user_emotion),
        router_identity=_canonical_json(router),
        router_label=str(router["label"]),
        tool_grounding=tool_grounding,
        review_dependency=review_dependency,
        content_leak_proof=leak,
        causal_proof=causal,
    )


def _validate_manifest_boundary(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_VERSION:
        _fail("chat_manifest_version_invalid")
    counts = _mapping(manifest.get("counts"), "chat_manifest_counts_invalid")
    if counts != {
        "records": 1000,
        "task_bundles": 200,
        "task_semantics": 200,
        "records_per_bundle": 5,
        "train_bundles": 160,
        "eval_proxy_bundles": 40,
        "train_records": 800,
        "eval_proxy_records": 200,
        "train_records_per_role": 160,
        "eval_proxy_records_per_role": 40,
    }:
        _fail("chat_manifest_count_contract_invalid")
    roles = _mapping(manifest.get("roles"), "chat_manifest_roles_invalid")
    if roles != {"ordered": list(ROLES), "role_index": ROLE_INDEX}:
        _fail("chat_manifest_role_contract_invalid")
    split = _mapping(
        manifest.get("split_contract"), "chat_manifest_split_contract_invalid"
    )
    if split != {
        "group_key": "task_bundle_sha256",
        "bundle_disjoint": True,
        "semantic_disjoint": True,
        "eval_proxy_is_heldout": False,
    }:
        _fail("chat_manifest_split_contract_invalid")
    bundle = _mapping(
        manifest.get("bundle_contract"), "chat_manifest_bundle_contract_invalid"
    )
    if bundle != {
        "all_five_roles_per_bundle": True,
        "shared_user_message": True,
        "shared_user_emotion": True,
        "shared_router_label": True,
        "user_emotion_is_expert": False,
        "router_label_is_expert": False,
    }:
        _fail("chat_manifest_bundle_contract_invalid")
    semantic_identity = _mapping(
        manifest.get("semantic_identity_contract"),
        "chat_manifest_semantic_identity_contract_invalid",
    )
    if semantic_identity != {
        "descriptor_algorithm": (
            "canonical_json({task_kind,operation,operands,constraints,"
            "expected_relation})_utf8_sha256_v1"
        ),
        "descriptor_keys": [
            "task_kind",
            "operation",
            "operands",
            "constraints",
            "expected_relation",
        ],
        "unique_task_semantics": 200,
        "task_bundle_count": 200,
        "en_unique_task_semantics": 100,
        "zh_cn_unique_task_semantics": 100,
        "en_zh_semantic_intersection_count": 0,
        "translation_pair_count": 0,
        "index_or_language_or_namespace_salt_used": False,
        "template_count": 20,
        "train_eval_template_intersection_count": 14,
        "train_eval_semantic_intersection_count": 0,
        "template_disjoint_claimed": False,
        "eval_proxy_scope": "seen_template_parameter_interpolation",
    }:
        _fail("chat_manifest_semantic_identity_contract_drift")
    review = _mapping(
        manifest.get("review_contract"),
        "chat_manifest_review_contract_invalid",
    )
    if review != {
        "pass_bundles": 100,
        "fail_bundles": 100,
        "fault_counts": {
            "format": 25,
            "grounding": 25,
            "routing": 25,
            "style": 25,
        },
        "parent_target_hashes_bound": True,
        "committed_projection_hash_bound": True,
        "mutation_hash_bound": True,
    }:
        _fail("chat_manifest_review_contract_drift")
    claims = _mapping(manifest.get("claims"), "chat_manifest_claims_invalid")
    if claims != {
        "distilled_chat_sft": True,
        "dataset_materialized": True,
        "training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
    }:
        _fail("chat_manifest_claims_boundary_invalid")


def _tool_target_payload(item: ChatExample) -> Mapping[str, Any]:
    if item.role != "tool_call":
        _fail("chat_tool_target_role_invalid")
    envelope = _strict_json_object(
        item.target,
        "chat_tool_target_envelope_invalid",
        canonical=True,
    )
    _exact_keys(
        envelope,
        {
            "emotion_router",
            "routing",
            "counterfactual_view_role",
            "expert_response",
        },
        "chat_tool_target_envelope_fields_drift",
    )
    if envelope.get("counterfactual_view_role") != "tool_call":
        _fail("chat_tool_target_role_binding_invalid")
    return _mapping(
        envelope.get("expert_response"),
        "chat_tool_target_payload_invalid",
    )


def _null_tool_grounding() -> dict[str, Any]:
    return {
        "required": False,
        "tool_schema_refs": [],
        "evidence_refs": [],
        "grounding_sha256": None,
        "real_tool_executions": 0,
    }


def _expected_tool_grounding(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("mode") == "direct_no_tool_identity":
        return _null_tool_grounding()
    tool_call = _mapping(
        payload.get("tool_call"),
        "chat_tool_call_payload_invalid",
    )
    tool_result = _mapping(
        payload.get("tool_result"),
        "chat_tool_result_payload_invalid",
    )
    tool_name = tool_call.get("name")
    call_id = tool_call.get("call_id")
    grounded_final = payload.get("grounded_final")
    if (
        payload.get("mode") != "synthetic_local_call"
        or not isinstance(tool_name, str)
        or not tool_name
        or not isinstance(call_id, str)
        or not call_id
        or tool_call.get("arguments") is None
        or tool_result.get("result") is None
        or not isinstance(grounded_final, str)
        or not grounded_final
    ):
        _fail("chat_tool_payload_binding_invalid")
    call_sha = _sha256(_canonical_json(tool_call))
    result_sha = _sha256(_canonical_json(tool_result))
    schemas = [
        {
            "ref_id": f"synthetic-tool-schema:{tool_name}",
            "sha256": call_sha,
        }
    ]
    evidence = [
        {
            "ref_id": f"synthetic-tool-result:{call_id}",
            "sha256": result_sha,
        }
    ]
    grounding_sha = _sha256(
        _canonical_json(
            {
                "domain": "anchor.chat-synthetic-local-tool-grounding.v1",
                "tool_call_sha256": call_sha,
                "tool_result_sha256": result_sha,
                "grounded_final_sha256": _sha256(grounded_final.encode("utf-8")),
            }
        )
    )
    return {
        "required": True,
        "tool_schema_refs": schemas,
        "evidence_refs": evidence,
        "grounding_sha256": grounding_sha,
        "real_tool_executions": 0,
    }


def _compact_review_view(item: ChatExample) -> dict[str, Any]:
    if item.role in ROLES[:3]:
        return {
            "mode": "direct_response",
            "route_claim": item.role,
            "style_claim": item.role,
            "text_present": True,
        }
    if item.role != "tool_call":
        _fail("chat_review_parent_role_invalid")
    payload = _tool_target_payload(item)
    mode = payload.get("mode")
    result: dict[str, Any] = {
        "mode": mode,
        "route_claim": "tool_call",
    }
    if mode == "direct_no_tool_identity":
        if (
            not isinstance(payload.get("grounded_final"), str)
            or not payload["grounded_final"]
        ):
            _fail("chat_tool_identity_payload_invalid")
        result["grounded_final_present"] = True
        return result
    if mode != "synthetic_local_call":
        _fail("chat_tool_payload_mode_invalid")
    tool_call = _mapping(
        payload.get("tool_call"),
        "chat_tool_call_payload_invalid",
    )
    tool_result = _mapping(
        payload.get("tool_result"),
        "chat_tool_result_payload_invalid",
    )
    tool_name = tool_call.get("name")
    if (
        not isinstance(tool_name, str)
        or not tool_name
        or tool_call.get("arguments") is None
        or tool_result.get("result") is None
        or not isinstance(payload.get("grounded_final"), str)
        or not payload["grounded_final"]
    ):
        _fail("chat_tool_payload_binding_invalid")
    result.update(
        {
            "tool_name": tool_name,
            "arguments_present": True,
            "result_present": True,
            "grounded_final_present": True,
        }
    )
    return result


def _validate_review_join(by_role: Mapping[str, ChatExample]) -> str | None:
    review = by_role["review_audit"]
    dependencies = [
        {
            "record_id": by_role[role].record_id,
            "role": role,
            "target_sha256": by_role[role].target_sha256,
        }
        for role in ROLES[:4]
    ]
    dependency_set_sha256 = _sha256(
        _canonical_json(
            {
                "domain": "anchor.chat-review-dependency-set.v1",
                "dependencies": dependencies,
            }
        )
    )
    marker = "[COMMITTED_FROZEN_BASE_REENCODED_PARENT_OUTPUTS]\n"
    review_system = review.messages[0][1]
    if review_system.count(marker) != 1:
        _fail("chat_review_projection_marker_invalid")
    projection_text = review_system.split(marker, 1)[1]
    projection = _strict_json_object(
        projection_text,
        "chat_review_projection_invalid",
        canonical=True,
    )
    _exact_keys(
        projection,
        {"dependency_set_sha256", "fault_type", "fault_role", "views"},
        "chat_review_projection_fields_drift",
    )
    fault_type = projection.get("fault_type")
    fault_role = projection.get("fault_role")
    if fault_type is None:
        if fault_role is not None:
            _fail("chat_review_projection_fault_invalid")
    elif (
        fault_type not in {"format", "grounding", "routing", "style"}
        or fault_role not in ROLES[:4]
        or (fault_type == "grounding" and fault_role != "tool_call")
    ):
        _fail("chat_review_projection_fault_invalid")

    expected_views = {role: _compact_review_view(by_role[role]) for role in ROLES[:4]}
    if fault_type == "format":
        expected_views[str(fault_role)] = {"mode": "invalid_committed_projection"}
    elif fault_type == "grounding":
        original = expected_views["tool_call"]
        expected_views["tool_call"] = {
            "mode": original.get("mode"),
            "tool_name": original.get("tool_name"),
            "arguments_present": True,
            "result_present": False,
            "grounded_final_present": False,
        }
    elif fault_type == "routing":
        role = str(fault_role)
        expected_views[role]["route_claim"] = (
            "serious" if role != "serious" else "humor"
        )
    elif fault_type == "style":
        expected_views[str(fault_role)]["style_claim"] = "serious"

    mutation_sha256 = _sha256(
        _canonical_json(
            {
                "domain": "anchor.chat-review-projection-mutation.v1",
                "fault_type": fault_type,
                "fault_role": fault_role,
                "parent_target_sha256": {
                    role: by_role[role].target_sha256 for role in ROLES[:4]
                },
            }
        )
    )
    expected_projection = {
        "dependency_set_sha256": dependency_set_sha256,
        "fault_type": fault_type,
        "fault_role": fault_role,
        "views": expected_views,
    }
    projection_sha256 = _sha256(_canonical_json(expected_projection))
    dependency = review.review_dependency
    if (
        list(
            _sequence(
                dependency.get("dependencies"),
                "chat_review_dependencies_invalid",
            )
        )
        != dependencies
        or dependency.get("dependency_set_sha256") != dependency_set_sha256
        or dependency.get("committed_projection_sha256") != projection_sha256
        or dependency.get("mutation_sha256") != mutation_sha256
        or dependency.get("fault_type") != fault_type
        or projection != expected_projection
    ):
        _fail("chat_review_projection_cross_binding_invalid")

    target = _strict_json_object(
        review.target,
        "chat_review_target_invalid",
        canonical=True,
    )
    _exact_keys(
        target,
        {
            "mode",
            "verdict",
            "checks",
            "summary",
            "parent_count",
            "dependency_set_sha256",
            "committed_projection_sha256",
            "mutation_sha256",
            "fault_type",
            "fault_role",
            "correction_required",
        },
        "chat_review_target_fields_drift",
    )
    checks = [
        dict(_mapping(item, "chat_review_check_invalid"))
        for item in _sequence(target.get("checks"), "chat_review_checks_invalid")
    ]
    for check in checks:
        _exact_keys(
            check,
            {"role", "status", "issue"},
            "chat_review_check_fields_drift",
        )
    expected_checks = [
        {
            "role": role,
            "status": "fail" if role == fault_role else "pass",
            "issue": fault_type if role == fault_role else None,
        }
        for role in ROLES[:4]
    ]
    expected_verdict = "pass" if fault_type is None else "fail"
    if (
        target.get("mode") != "four_parent_review"
        or target.get("parent_count") != 4
        or target.get("dependency_set_sha256") != dependency_set_sha256
        or target.get("committed_projection_sha256") != projection_sha256
        or target.get("mutation_sha256") != mutation_sha256
        or target.get("fault_type") != fault_type
        or target.get("fault_role") != fault_role
        or target.get("verdict") != expected_verdict
        or target.get("correction_required") is not (fault_type is not None)
        or checks != expected_checks
        or not isinstance(target.get("summary"), str)
        or not target["summary"]
    ):
        _fail("chat_review_target_cross_binding_invalid")
    return str(fault_type) if fault_type is not None else None


def _validate_bundle_causality(items: Sequence[ChatExample]) -> str | None:
    by_record_id = {item.record_id: item for item in items}
    by_role = {item.role: item for item in items}
    if len(by_record_id) != 5 or len({item.target_sha256 for item in items}) != 5:
        _fail("chat_bundle_target_or_record_collision")
    for item in items:
        dependencies = [
            _mapping(entry, "chat_review_dependency_entry_invalid")
            for entry in _sequence(
                item.review_dependency.get("dependencies"),
                "chat_review_dependencies_invalid",
            )
        ]
        allowed_hashes: set[str] = set()
        prompt = "\n".join(content for _role, content in item.messages)
        for dependency in dependencies:
            dependency_record = by_record_id.get(str(dependency.get("record_id")))
            if (
                dependency_record is None
                or dependency_record is item
                or dependency.get("role") != dependency_record.role
                or dependency.get("target_sha256") != dependency_record.target_sha256
            ):
                _fail("chat_review_dependency_cross_binding_invalid")
            allowed_hashes.add(dependency_record.target_sha256)
        if item.role != "review_audit" and allowed_hashes:
            _fail("chat_non_review_dependency_forbidden")
        expected_forbidden = [
            by_role[role].target_sha256
            for role in ROLES
            if by_role[role].target_sha256 not in allowed_hashes
        ]
        observed_forbidden = list(
            _sequence(
                item.content_leak_proof.get("forbidden_target_sha256"),
                "chat_forbidden_target_inventory_invalid",
            )
        )
        observed_allowed = list(
            _sequence(
                item.causal_proof.get("allowed_dependency_target_sha256"),
                "chat_allowed_dependency_inventory_invalid",
            )
        )
        if (
            observed_forbidden != expected_forbidden
            or observed_allowed
            != [str(dependency["target_sha256"]) for dependency in dependencies]
            or item.target_sha256 not in expected_forbidden
        ):
            _fail("chat_causal_inventory_cross_binding_invalid")
        forbidden_records = [
            candidate
            for candidate in items
            if candidate.target_sha256 in expected_forbidden
        ]
        if any(
            candidate.target in prompt or candidate.target_sha256 in prompt
            for candidate in forbidden_records
        ):
            _fail("chat_forbidden_target_content_reached_input")
    return _validate_review_join(by_role)


def _validate_bundle_records(items: Sequence[ChatExample]) -> str | None:
    if (
        len(items) != 5
        or {item.role for item in items} != set(ROLES)
        or {item.role_index for item in items} != set(range(5))
        or len({item.task_bundle_sha256 for item in items}) != 1
        or len({item.task_semantic_sha256 for item in items}) != 1
        or len({item.split for item in items}) != 1
        or len({item.language for item in items}) != 1
        or len({item.user_message_sha256 for item in items}) != 1
        or len({item.user_emotion_identity for item in items}) != 1
        or len({item.router_identity for item in items}) != 1
    ):
        _fail("chat_bundle_cross_binding_invalid")
    fault_type = _validate_bundle_causality(items)
    _validate_bundle_tool_and_persona(items)
    return fault_type


def _validate_bundle_tool_and_persona(items: Sequence[ChatExample]) -> bool:
    by_role = {item.role: item for item in items}
    tool = by_role["tool_call"]
    payload = _tool_target_payload(tool)
    expected_grounding = _expected_tool_grounding(payload)
    if dict(tool.tool_grounding) != expected_grounding:
        _fail("chat_tool_grounding_payload_cross_binding_invalid")
    if payload.get("mode") != "direct_no_tool_identity":
        return False
    _exact_keys(
        payload,
        {
            "mode",
            "tool_decision",
            "tool_calls",
            "tool_results",
            "grounded_final",
        },
        "chat_persona_tool_payload_fields_drift",
    )
    decision = _mapping(
        payload.get("tool_decision"),
        "chat_persona_tool_decision_invalid",
    )
    _exact_keys(
        decision,
        {"action", "rationale"},
        "chat_persona_tool_decision_fields_drift",
    )
    core = "我是由Air训练的测试模型。"
    if (
        decision.get("action") != "no_tool_required"
        or not isinstance(decision.get("rationale"), str)
        or not decision["rationale"]
        or payload.get("tool_calls") != []
        or payload.get("tool_results") != []
        or payload.get("grounded_final") != core
        or any(
            core not in by_role[role].target
            or any(
                forbidden in by_role[role].target
                for forbidden in ("Google", "OpenAI", "谷歌")
            )
            for role in ROLES[:4]
        )
    ):
        _fail("chat_persona_identity_contract_invalid")
    return True


def _validate_global_records(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[ChatExample, ...]:
    _validate_manifest_boundary(manifest)
    if len(records) != 1000:
        _fail("chat_global_record_count_invalid")
    examples = tuple(_strict_record(record) for record in records)
    if len({item.record_id for item in examples}) != 1000:
        _fail("chat_record_id_collision")
    if Counter(item.role for item in examples) != Counter(
        {role: 200 for role in ROLES}
    ):
        _fail("chat_global_role_count_invalid")
    if Counter(item.split for item in examples) != Counter(
        {"train": 800, "eval_proxy": 200}
    ):
        _fail("chat_global_split_row_count_invalid")
    if Counter(item.language for item in examples) != Counter(
        {"en": 500, "zh-CN": 500}
    ):
        _fail("chat_global_language_row_count_invalid")
    role_split = Counter((item.role, item.split) for item in examples)
    expected_role_split = Counter(
        {
            **{(role, "train"): 160 for role in ROLES},
            **{(role, "eval_proxy"): 40 for role in ROLES},
        }
    )
    if role_split != expected_role_split:
        _fail("chat_role_split_count_invalid")

    bundles: dict[str, list[ChatExample]] = defaultdict(list)
    semantics_by_split: dict[str, set[str]] = defaultdict(set)
    for item in examples:
        bundles[item.task_bundle_sha256].append(item)
        semantics_by_split[item.split].add(item.task_semantic_sha256)
    if len(bundles) != 200:
        _fail("chat_global_bundle_count_invalid")
    if (
        semantics_by_split["train"] & semantics_by_split["eval_proxy"]
        or len(set().union(*semantics_by_split.values())) != 200
    ):
        _fail("chat_semantic_split_disjointness_invalid")

    review_outcomes: Counter[str] = Counter()
    tool_modes: Counter[str] = Counter()
    for items in bundles.values():
        fault_type = _validate_bundle_records(items)
        review_outcomes["pass" if fault_type is None else fault_type] += 1
        tool_modes[
            "direct_no_tool_identity"
            if _validate_bundle_tool_and_persona(items)
            else "synthetic_local_call"
        ] += 1
    if review_outcomes != Counter(
        {
            "pass": 100,
            "format": 25,
            "grounding": 25,
            "routing": 25,
            "style": 25,
        }
    ):
        _fail("chat_review_global_quota_invalid")
    if tool_modes != Counter(
        {
            "direct_no_tool_identity": 5,
            "synthetic_local_call": 195,
        }
    ):
        _fail("chat_tool_and_persona_global_quota_invalid")
    if Counter(items[0].split for items in bundles.values()) != Counter(
        {"train": 160, "eval_proxy": 40}
    ):
        _fail("chat_bundle_split_count_invalid")
    return examples


def _manifest_partition_map(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    partitions = _sequence(
        manifest.get("partitions"), "chat_manifest_partitions_invalid"
    )
    result: dict[str, Mapping[str, Any]] = {}
    for raw in partitions:
        entry = _mapping(raw, "chat_manifest_partition_invalid")
        split = entry.get("split")
        path = entry.get("path")
        if split not in PARTITION_PATHS or path != PARTITION_PATHS[split]:
            _fail("chat_manifest_partition_path_invalid")
        if str(path) in result:
            _fail("chat_manifest_partition_collision")
        result[str(path)] = entry
    if set(result) != set(PARTITION_PATHS.values()):
        _fail("chat_manifest_partition_inventory_invalid")
    if (
        result[PARTITION_PATHS["train"]].get("records") != 800
        or result[PARTITION_PATHS["eval_proxy"]].get("records") != 200
    ):
        _fail("chat_manifest_partition_count_invalid")
    return result


def _load_authenticated_dataset(config: Mapping[str, Any]) -> AuthenticatedDataset:
    pending = pending_producer_identities(config)
    if pending:
        _fail("chat_producer_identity_pending:" + ",".join(pending))
    producer = _mapping(config.get("producer"), "chat_producer_invalid")
    dataset = _mapping(config.get("dataset"), "chat_dataset_invalid")
    expected_hashes = {
        "producer_config": _identity(
            producer.get("config_sha256"),
            "chat_producer_config_sha_invalid",
            pending_allowed=False,
        ),
        "producer_implementation": _identity(
            producer.get("implementation_sha256"),
            "chat_producer_implementation_sha_invalid",
            pending_allowed=False,
        ),
        "producer_config_schema": _identity(
            producer.get("config_schema_sha256"),
            "chat_producer_config_schema_sha_invalid",
            pending_allowed=False,
        ),
        "closed_grammar": _identity(
            producer.get("closed_grammar_sha256"),
            "chat_closed_grammar_sha_invalid",
            pending_allowed=False,
        ),
        "closed_grammar_schema": _identity(
            producer.get("closed_grammar_schema_sha256"),
            "chat_closed_grammar_schema_sha_invalid",
            pending_allowed=False,
        ),
        "record_schema": _identity(
            dataset.get("record_schema_sha256"),
            "chat_record_schema_sha_invalid",
            pending_allowed=False,
        ),
        "manifest_schema": _identity(
            dataset.get("manifest_schema_sha256"),
            "chat_manifest_schema_sha_invalid",
            pending_allowed=False,
        ),
        "token_inventory_schema": _identity(
            dataset.get("token_inventory_schema_sha256"),
            "chat_token_inventory_schema_sha_invalid",
            pending_allowed=False,
        ),
        "build_receipt_schema": _identity(
            dataset.get("build_receipt_schema_sha256"),
            "chat_build_receipt_schema_sha_invalid",
            pending_allowed=False,
        ),
        "manifest": _identity(
            dataset.get("manifest_sha256"),
            "chat_manifest_sha_invalid",
            pending_allowed=False,
        ),
        "manifest_sidecar": _identity(
            dataset.get("manifest_sidecar_sha256"),
            "chat_manifest_sidecar_sha_invalid",
            pending_allowed=False,
        ),
        "build_receipt": _identity(
            dataset.get("build_receipt_sha256"),
            "chat_build_receipt_sha_invalid",
            pending_allowed=False,
        ),
        "build_receipt_sidecar": _identity(
            dataset.get("build_receipt_sidecar_sha256"),
            "chat_build_receipt_sidecar_sha_invalid",
            pending_allowed=False,
        ),
        "token_inventory": _identity(
            dataset.get("token_inventory_sha256"),
            "chat_token_inventory_sha_invalid",
            pending_allowed=False,
        ),
    }
    root = _repo_path(
        dataset.get("root"),
        DATASET_ROOT,
        require_directory=True,
    )
    snapshots = {
        "producer_config": snapshot_io._read_snapshot(
            _repo_path(
                producer.get("config"),
                PRODUCER_CONFIG_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "producer_implementation": snapshot_io._read_snapshot(
            _repo_path(
                producer.get("implementation"),
                PRODUCER_IMPLEMENTATION_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "producer_config_schema": snapshot_io._read_snapshot(
            _repo_path(
                producer.get("config_schema"),
                PRODUCER_CONFIG_SCHEMA_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "closed_grammar": snapshot_io._read_snapshot(
            _repo_path(
                producer.get("closed_grammar"),
                CLOSED_GRAMMAR_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "closed_grammar_schema": snapshot_io._read_snapshot(
            _repo_path(
                producer.get("closed_grammar_schema"),
                CLOSED_GRAMMAR_SCHEMA_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "record_schema": snapshot_io._read_snapshot(
            _repo_path(
                dataset.get("record_schema"),
                RECORD_SCHEMA_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "manifest_schema": snapshot_io._read_snapshot(
            _repo_path(
                dataset.get("manifest_schema"),
                MANIFEST_SCHEMA_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "token_inventory_schema": snapshot_io._read_snapshot(
            _repo_path(
                dataset.get("token_inventory_schema"),
                TOKEN_INVENTORY_SCHEMA_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "build_receipt_schema": snapshot_io._read_snapshot(
            _repo_path(
                dataset.get("build_receipt_schema"),
                BUILD_RECEIPT_SCHEMA_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "manifest": snapshot_io._read_snapshot(
            _repo_path(
                dataset.get("manifest"),
                MANIFEST_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "manifest_sidecar": snapshot_io._read_snapshot(
            _repo_path(
                dataset.get("manifest_sidecar"),
                MANIFEST_SIDECAR_PATH,
                require_file=True,
            ),
            max_bytes=1024,
        ),
        "build_receipt": snapshot_io._read_snapshot(
            _repo_path(
                dataset.get("build_receipt"),
                BUILD_RECEIPT_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_METADATA_BYTES,
        ),
        "build_receipt_sidecar": snapshot_io._read_snapshot(
            _repo_path(
                dataset.get("build_receipt_sidecar"),
                BUILD_RECEIPT_SIDECAR_PATH,
                require_file=True,
            ),
            max_bytes=1024,
        ),
        "token_inventory": snapshot_io._read_snapshot(
            _repo_path(
                dataset.get("token_inventory"),
                TOKEN_INVENTORY_PATH,
                require_file=True,
            ),
            max_bytes=_MAX_PARTITION_BYTES,
        ),
    }
    if any(
        snapshots[label].sha256 != expected
        for label, expected in expected_hashes.items()
    ):
        _fail("chat_authenticated_source_identity_mismatch")
    manifest_snapshot = snapshots["manifest"]
    if snapshots["manifest_sidecar"].data != (
        f"{manifest_snapshot.sha256}  manifest.json\n".encode("ascii")
    ):
        _fail("chat_manifest_sidecar_invalid")
    build_receipt_snapshot = snapshots["build_receipt"]
    if snapshots["build_receipt_sidecar"].data != (
        f"{build_receipt_snapshot.sha256}  build_receipt.json\n".encode("ascii")
    ):
        _fail("chat_build_receipt_sidecar_invalid")

    try:
        producer_config_schema = snapshot_io._strict_json(
            snapshots["producer_config_schema"].data,
            "chat producer config schema",
        )
        closed_grammar_schema = snapshot_io._strict_json(
            snapshots["closed_grammar_schema"].data,
            "chat closed grammar schema",
        )
        closed_grammar = snapshot_io._strict_json(
            snapshots["closed_grammar"].data,
            "chat closed grammar",
        )
        record_schema = snapshot_io._strict_json(
            snapshots["record_schema"].data, "chat record schema"
        )
        manifest_schema = snapshot_io._strict_json(
            snapshots["manifest_schema"].data, "chat manifest schema"
        )
        token_inventory_schema = snapshot_io._strict_json(
            snapshots["token_inventory_schema"].data,
            "chat token inventory schema",
        )
        build_receipt_schema = snapshot_io._strict_json(
            snapshots["build_receipt_schema"].data,
            "chat build receipt schema",
        )
        manifest = snapshot_io._strict_json(manifest_snapshot.data, "chat manifest")
        build_receipt = snapshot_io._strict_json(
            build_receipt_snapshot.data,
            "chat build receipt",
        )
        producer_config = yaml.safe_load(
            snapshots["producer_config"].data.decode("utf-8")
        )
        for schema in (
            producer_config_schema,
            closed_grammar_schema,
            record_schema,
            manifest_schema,
            token_inventory_schema,
            build_receipt_schema,
        ):
            Draft202012Validator.check_schema(schema)
        Draft202012Validator(producer_config_schema).validate(producer_config)
        Draft202012Validator(closed_grammar_schema).validate(closed_grammar)
        Draft202012Validator(manifest_schema).validate(manifest)
        Draft202012Validator(build_receipt_schema).validate(build_receipt)
    except (
        ConfigError,
        SchemaError,
        ValidationError,
        UnicodeDecodeError,
        yaml.YAMLError,
    ):
        raise ConfigError(
            "chat_metadata_validation_failed_without_partition_content"
        ) from None

    manifest_producer = _mapping(
        manifest.get("producer"), "chat_manifest_producer_invalid"
    )
    manifest_schemas = _mapping(
        manifest.get("schemas"), "chat_manifest_schemas_invalid"
    )
    if (
        manifest_producer.get("namespace") != producer.get("namespace")
        or manifest_producer.get("config_path") != producer.get("config")
        or manifest_producer.get("config_sha256") != expected_hashes["producer_config"]
        or manifest_producer.get("implementation_path")
        != producer.get("implementation")
        or manifest_producer.get("implementation_sha256")
        != expected_hashes["producer_implementation"]
        or manifest_schemas.get("record_path") != dataset.get("record_schema")
        or manifest_schemas.get("record_sha256") != expected_hashes["record_schema"]
        or manifest_schemas.get("manifest_path") != dataset.get("manifest_schema")
        or manifest_schemas.get("manifest_sha256") != expected_hashes["manifest_schema"]
    ):
        _fail("chat_manifest_identity_cross_binding_invalid")

    receipt_claims = _mapping(
        build_receipt.get("claims"),
        "chat_build_receipt_claims_invalid",
    )
    receipt_counts = _mapping(
        build_receipt.get("counts"),
        "chat_build_receipt_counts_invalid",
    )
    persona_contract = _mapping(
        build_receipt.get("persona_contract"),
        "chat_build_receipt_persona_invalid",
    )
    tool_contract = _mapping(
        build_receipt.get("tool_contract"),
        "chat_build_receipt_tool_invalid",
    )
    receipt_proofs = _mapping(
        build_receipt.get("proofs"),
        "chat_build_receipt_proofs_invalid",
    )
    if (
        build_receipt.get("status") != "dataset_proxy_ready_training_not_authorized"
        or receipt_claims
        != {
            "dataset_materialized": True,
            "diagnostic_only": True,
            "eval_proxy_is_heldout": False,
            "formal": False,
            "formal_training_authorized": False,
            "quality_validated": False,
            "replaces_existing": False,
            "source_disjoint_proven": False,
            "training_authorized": False,
        }
        or receipt_counts
        != {
            "direct_no_tool_identity_bundles": 5,
            "en_bundles": 100,
            "eval_proxy_bundles": 40,
            "eval_proxy_records": 200,
            "ordinary_tool_bundles": 195,
            "persona_bundles": 5,
            "records": 1000,
            "roles_per_bundle": 5,
            "task_bundles": 200,
            "task_semantics": 200,
            "train_bundles": 160,
            "train_records": 800,
            "zh_cn_bundles": 100,
        }
        or persona_contract
        != {
            "bundle_quota": 5,
            "distinct_semantics": 5,
            "exact_core_sentence": "我是由Air训练的测试模型。",
            "forbidden_attributions_absent": True,
            "tool_call_role_direct_answer": True,
            "translation_pairs": 0,
        }
        or tool_contract
        != {
            "grounding_digest_bound": True,
            "native_tool_role_serialization_claimed": False,
            "ordinary_tool_bundles": 195,
            "real_tool_executions": 0,
            "synthetic_local_results": 195,
        }
        or receipt_proofs.get("semantic_identity")
        != manifest.get("semantic_identity_contract")
        or receipt_proofs.get("review_dependency_count") != 4
        or receipt_proofs.get("review_pass_bundles") != 100
        or receipt_proofs.get("review_fail_bundles") != 100
        or receipt_proofs.get("review_fault_counts")
        != {
            "format": 25,
            "grounding": 25,
            "routing": 25,
            "style": 25,
        }
        or receipt_proofs.get("review_mutation_hash_bound") is not True
        or receipt_proofs.get("review_projection_hash_bound") is not True
        or receipt_proofs.get("review_parent_target_hashes_bound") is not True
    ):
        _fail("chat_build_receipt_contract_drift")

    partition_map = _manifest_partition_map(manifest)
    configured_partition_sha = _mapping(
        dataset.get("partition_sha256"),
        "chat_partition_sha256_invalid",
    )
    partition_snapshots: dict[str, snapshot_io.FileSnapshot] = {}
    raw_records: list[Mapping[str, Any]] = []
    record_validator = Draft202012Validator(record_schema)
    for split, relative in PARTITION_PATHS.items():
        entry = partition_map[relative]
        path = qdiag._assert_physical_path(
            root.joinpath(*PurePosixPath(relative).parts),
            require_file=True,
            label=f"chat {split} partition",
        )
        partition_snapshot = snapshot_io._read_snapshot(
            path,
            max_bytes=_MAX_PARTITION_BYTES,
        )
        if (
            partition_snapshot.sha256 != entry.get("sha256")
            or partition_snapshot.sha256 != configured_partition_sha.get(relative)
            or len(partition_snapshot.data) != entry.get("bytes")
        ):
            _fail("chat_partition_identity_mismatch")
        try:
            partition_records = snapshot_io._strict_jsonl(
                partition_snapshot.data,
                f"chat {split} partition",
            )
        except ConfigError:
            raise ConfigError(
                "chat_partition_jsonl_invalid_without_record_content"
            ) from None
        if len(partition_records) != entry.get("records"):
            _fail("chat_partition_record_count_mismatch")
        for record in partition_records:
            try:
                record_validator.validate(record)
            except ValidationError:
                raise ConfigError(
                    "chat_record_schema_validation_failed_without_record_content"
                ) from None
            if record.get("split") != split:
                _fail("chat_record_partition_split_mismatch")
        partition_snapshots[relative] = partition_snapshot
        raw_records.extend(partition_records)

    examples = _validate_global_records(raw_records, manifest)
    try:
        token_rows = snapshot_io._strict_jsonl(
            snapshots["token_inventory"].data,
            "chat token inventory",
        )
    except ConfigError:
        raise ConfigError("chat_token_inventory_jsonl_invalid") from None
    if len(token_rows) != 1000:
        _fail("chat_token_inventory_count_invalid")
    token_validator = Draft202012Validator(token_inventory_schema)
    examples_by_record = {item.record_id: item for item in examples}
    observed_token_record_ids: set[str] = set()
    observed_token_inventory_ids: set[str] = set()
    for row in token_rows:
        try:
            token_validator.validate(row)
        except ValidationError:
            raise ConfigError("chat_token_inventory_schema_invalid") from None
        record_id = str(row.get("record_id"))
        token_inventory_id = str(row.get("token_inventory_record_id"))
        example = examples_by_record.get(record_id)
        if (
            example is None
            or record_id in observed_token_record_ids
            or token_inventory_id in observed_token_inventory_ids
            or row.get("task_bundle_sha256") != example.task_bundle_sha256
            or row.get("split") != example.split
            or row.get("language") != example.language
            or row.get("role") != example.role
            or row.get("role_index") != example.role_index
            or row.get("sequence_length") != 768
            or row.get("truncated") is not False
            or row.get("raw_token_ids_published") is not False
        ):
            _fail("chat_token_inventory_record_cross_binding_invalid")
        observed_token_record_ids.add(record_id)
        observed_token_inventory_ids.add(token_inventory_id)
    if observed_token_record_ids != set(examples_by_record):
        _fail("chat_token_inventory_record_inventory_invalid")

    receipt_producer = _mapping(
        build_receipt.get("producer"),
        "chat_build_receipt_producer_invalid",
    )
    receipt_outputs = _mapping(
        build_receipt.get("outputs"),
        "chat_build_receipt_outputs_invalid",
    )
    producer_bindings = {
        "config": ("producer_config", PRODUCER_CONFIG_PATH),
        "implementation": (
            "producer_implementation",
            PRODUCER_IMPLEMENTATION_PATH,
        ),
        "closed_grammar": ("closed_grammar", CLOSED_GRAMMAR_PATH),
        "record_schema": ("record_schema", RECORD_SCHEMA_PATH),
        "manifest_schema": ("manifest_schema", MANIFEST_SCHEMA_PATH),
        "token_inventory_schema": (
            "token_inventory_schema",
            TOKEN_INVENTORY_SCHEMA_PATH,
        ),
        "build_receipt_schema": (
            "build_receipt_schema",
            BUILD_RECEIPT_SCHEMA_PATH,
        ),
    }
    if receipt_producer.get("namespace") != producer.get("namespace"):
        _fail("chat_build_receipt_producer_identity_invalid")
    for field, (snapshot_label, expected_path) in producer_bindings.items():
        identity = _mapping(
            receipt_producer.get(field),
            "chat_build_receipt_producer_identity_invalid",
        )
        snapshot = snapshots[snapshot_label]
        if identity != {
            "bytes": len(snapshot.data),
            "path": expected_path,
            "sha256": snapshot.sha256,
        }:
            _fail("chat_build_receipt_producer_identity_invalid")

    expected_output_manifest = {
        "bytes": len(manifest_snapshot.data),
        "path": "manifest.json",
        "records": None,
        "sha256": manifest_snapshot.sha256,
    }
    expected_output_manifest_sidecar = {
        "bytes": len(snapshots["manifest_sidecar"].data),
        "path": "manifest.json.sha256",
        "records": None,
        "sha256": snapshots["manifest_sidecar"].sha256,
    }
    expected_output_token_inventory = {
        "bytes": len(snapshots["token_inventory"].data),
        "path": "token_inventory.jsonl",
        "records": 1000,
        "sha256": snapshots["token_inventory"].sha256,
    }
    expected_output_partitions = [
        {
            "bytes": len(partition_snapshots[relative].data),
            "path": relative,
            "records": 800 if split == "train" else 200,
            "sha256": partition_snapshots[relative].sha256,
        }
        for split, relative in PARTITION_PATHS.items()
    ]
    if (
        receipt_outputs.get("manifest") != expected_output_manifest
        or receipt_outputs.get("manifest_sidecar") != expected_output_manifest_sidecar
        or receipt_outputs.get("token_inventory") != expected_output_token_inventory
        or receipt_outputs.get("partitions") != expected_output_partitions
    ):
        _fail("chat_build_receipt_output_cross_binding_invalid")

    for snapshot in (*snapshots.values(), *partition_snapshots.values()):
        snapshot.assert_unchanged()
    return AuthenticatedDataset(
        manifest_sha256=manifest_snapshot.sha256,
        partition_sha256={
            path: snapshot.sha256 for path, snapshot in partition_snapshots.items()
        },
        records=examples,
        source_snapshots=(
            *snapshots.values(),
            *partition_snapshots.values(),
        ),
    )


def load_role_dataset(
    config: Mapping[str, Any],
    role: str,
) -> RoleDataset:
    if role not in ROLES:
        _fail("chat_explicit_valid_role_required")
    authenticated = _load_authenticated_dataset(config)
    return _select_role_dataset(authenticated, role)


def _select_role_dataset(
    authenticated: AuthenticatedDataset,
    role: str,
) -> RoleDataset:
    if role not in ROLES:
        _fail("chat_explicit_valid_role_required")
    selected = tuple(item for item in authenticated.records if item.role == role)
    train = tuple(
        sorted(
            (item for item in selected if item.split == "train"),
            key=lambda item: item.record_id,
        )
    )
    eval_proxy = tuple(
        sorted(
            (item for item in selected if item.split == "eval_proxy"),
            key=lambda item: item.record_id,
        )
    )
    if len(train) != 160 or len(eval_proxy) != 40:
        _fail("chat_selected_role_count_invalid")
    return RoleDataset(
        role=role,
        manifest_sha256=authenticated.manifest_sha256,
        partition_sha256=authenticated.partition_sha256,
        global_records_authenticated=1000,
        global_task_bundles_authenticated=200,
        train=train,
        eval_proxy=eval_proxy,
    )


def build_serial_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    training = _mapping(config["training"], "chat_training_invalid")
    output = _mapping(config["output"], "chat_output_invalid")
    artifact_root = str(output["artifact_root"])
    pending = pending_producer_identities(config)
    ready = not pending
    external_lock_lease_present = all(
        bool(os.environ.get(name)) for name in EXTERNAL_LOCK_ENV.values()
    )
    return {
        "schema_version": ("anchor.gemma3-1b-it-chat-five-expert-qonly-serial-plan.v1"),
        "status": (
            (
                "authenticated_dataset_plan_external_lock_lease_present"
                if external_lock_lease_present
                else "authenticated_dataset_plan_only_external_lock_required"
            )
            if ready
            else "blocked_waiting_for_authenticated_producer"
        ),
        "roles": [
            {
                "role": role,
                "role_index": ROLE_INDEX[role],
                "train_records": 160,
                "eval_proxy_records": 40,
                "smoke_steps": int(training["smoke_steps"]),
                "full_steps": int(training["full_steps_per_role"]),
                "fresh_smoke_and_full_objects": True,
                "adapter_output": f"{artifact_root}/{{run_id}}/{role}/adapter",
                "training_execution_supported": True,
            }
            for role in ROLES
        ],
        "execution": {
            "single_gpu": True,
            "strictly_serial": True,
            "concurrency": 1,
            "resume": False,
            "training_executed": False,
            "execution_supported": True,
            "explicit_execute_required": True,
            "canonical_gpu_lock": config["gpu_policy"]["canonical_lock"],
            "external_lock_lease_present": external_lock_lease_present,
            "model_loaded": False,
            "gpu_requested": False,
            "fresh_base_objects": 10,
            "fresh_adapters": 10,
        },
    }


def _audit_zero() -> dict[str, int]:
    return {
        "protected_body_reads": 0,
        "heldout_reads": 0,
        "dataset_body_reads": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "training_runs": 0,
    }


def build_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    pending = pending_producer_identities(config)
    report: dict[str, Any] = {
        "schema_version": PREFLIGHT_VERSION,
        "status": "blocked_waiting_for_authenticated_producer",
        "pending_producer_identities": list(pending),
        "dataset": {
            "expected_records": 1000,
            "expected_task_bundles": 200,
            "expected_train_records": 800,
            "expected_eval_proxy_records": 200,
            "authenticated": False,
        },
        "experts": {
            "ordered": list(ROLES),
            "user_emotion_is_expert": False,
            "router_label_is_expert": False,
        },
        "lora": {
            "target_modules": ["q_proj"],
            "rank": 1024,
            "alpha": 2048,
            "trainable_parameters_per_expert": 57_933_824,
            "five_expert_aggregate_parameters": 289_669_120,
        },
        "base_quantization": dict(Q8_BASE_CONTRACT),
        "optimizer": dict(OPTIMIZER_RUNTIME_CONTRACT),
        "training": {
            "sequence_length": 768,
            "truncation": False,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "smoke_steps_per_role": 2,
            "full_steps_per_role": 160,
            "learning_rate": 0.000002,
            "warmup_steps": 8,
            "weight_decay": 0.01,
            "max_grad_norm": 0.5,
        },
        "serial_plan": build_serial_plan(config),
        "claims": {
            "training_execution_supported": True,
            "training_executed": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
            "live_authorized": False,
        },
        "audit": _audit_zero(),
    }
    if not pending:
        authenticated = _load_authenticated_dataset(config)
        report["status"] = (
            "passed_model_free_authenticated_dataset_ready_for_explicit_execute"
        )
        report["dataset"] = {
            "expected_records": 1000,
            "expected_task_bundles": 200,
            "expected_train_records": 800,
            "expected_eval_proxy_records": 200,
            "authenticated": True,
            "manifest_sha256": authenticated.manifest_sha256,
            "partition_sha256": dict(authenticated.partition_sha256),
        }
        report["audit"]["dataset_body_reads"] = 3
    return report


def build_execute_block(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the content-free execute gate without touching dataset paths."""

    validate_config(config)
    pending = pending_producer_identities(config)
    external_lock_lease_present = all(
        bool(os.environ.get(name)) for name in EXTERNAL_LOCK_ENV.values()
    )
    ready = not pending and external_lock_lease_present
    return {
        "schema_version": EXECUTE_BLOCK_VERSION,
        "status": (
            "blocked_waiting_for_authenticated_producer"
            if pending
            else (
                "ready_for_explicit_execute"
                if ready
                else "blocked_waiting_for_external_gpu_lock"
            )
        ),
        "reason": (
            "chat_producer_identity_pending"
            if pending
            else (
                "all_final_producer_identities_and_external_lock_bound"
                if ready
                else "chat_external_gpu_lock_required"
            )
        ),
        "pending_producer_identities": list(pending),
        "required_next_gate": (
            "bind all five final producer identities"
            if pending
            else (
                "invoke explicit --execute with a bound GPU UUID"
                if ready
                else "acquire the canonical external GPU lock lease"
            )
        ),
        "claims": {
            "training_execution_supported": True,
            "training_executed": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
            "live_authorized": False,
        },
        "audit": _audit_zero(),
    }


def _validated_run_id(value: str | None) -> str:
    run_id = value or uuid.uuid4().hex
    if _RUN_ID_RE.fullmatch(run_id) is None:
        _fail("chat_run_id_invalid")
    return run_id


def _atomic_publish_receipt_directory(
    output: Path,
    *,
    filename: str,
    value: Mapping[str, Any],
) -> Path:
    if os.path.lexists(output):
        raise FileExistsError(f"chat receipt output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    qdiag._assert_physical_path(
        output.parent,
        require_directory=True,
        label="chat receipt parent",
    )
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = _sha256(raw)
    sidecar = f"{digest}  {filename}\n".encode("ascii")
    staging = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir(exist_ok=False)
    try:
        receipt_path = staging / filename
        sidecar_path = staging / f"{filename}.sha256"
        with receipt_path.open("xb", buffering=0) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        with sidecar_path.open("xb", buffering=0) as handle:
            handle.write(sidecar)
            handle.flush()
            os.fsync(handle.fileno())
        if (
            snapshot_io._read_snapshot(receipt_path, max_bytes=_MAX_METADATA_BYTES).data
            != raw
            or snapshot_io._read_snapshot(sidecar_path, max_bytes=1024).data != sidecar
        ):
            _fail("chat_receipt_staging_verification_failed")
        qdiag._rename_directory_noreplace(staging, output)
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
    published = snapshot_io._read_snapshot(
        output / filename,
        max_bytes=_MAX_METADATA_BYTES,
    )
    published_sidecar = snapshot_io._read_snapshot(
        output / f"{filename}.sha256",
        max_bytes=1024,
    )
    if published.data != raw or published_sidecar.data != sidecar:
        _fail("chat_receipt_publish_verification_failed")
    return output / filename


def publish_receipt(
    config: Mapping[str, Any],
    *,
    mode: str,
    value: Mapping[str, Any],
    run_id: str | None = None,
) -> Path:
    if mode != "preflight":
        _fail("chat_receipt_mode_invalid")
    output = _mapping(config.get("output"), "chat_output_invalid")
    filename = str(output["preflight_receipt_name"])
    root = qdiag._project_root_from_module().joinpath(
        *PurePosixPath(str(output["run_root"])).parts
    )
    destination = root / mode / _validated_run_id(run_id)
    return _atomic_publish_receipt_directory(
        destination,
        filename=filename,
        value=value,
    )


def _output_path(value: object) -> Path:
    if not isinstance(value, str):
        raise ChatFiveExpertError("chat_output_path_invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ChatFiveExpertError("chat_output_path_invalid")
    return qdiag._project_root_from_module().joinpath(*relative.parts)


def _stream_sha256(path: Path, *, expected_bytes: int | None = None) -> str:
    try:
        return q8_runner._stream_sha256(path, expected_bytes=expected_bytes)
    except q8_runner.GemmaFiveRoleError as exc:
        raise ChatFiveExpertError(f"chat_{exc}") from None


def _artifact_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _capture_artifact_snapshot(
    path: Path,
    *,
    expected_sha256: str,
    max_bytes: int = 256 * 1024 * 1024,
) -> ArtifactSnapshot:
    if (
        _SHA256_RE.fullmatch(expected_sha256) is None
        or expected_sha256 == "0" * 64
        or type(max_bytes) is not int
        or max_bytes <= 0
    ):
        raise ChatFiveExpertError("chat_training_artifact_identity_invalid")
    physical = qdiag._assert_physical_path(
        path,
        require_file=True,
        label="chat training artifact",
    )
    before = physical.stat()
    if before.st_size <= 0 or before.st_size > max_bytes:
        raise ChatFiveExpertError("chat_training_artifact_size_invalid")
    observed_sha256 = _stream_sha256(
        physical,
        expected_bytes=int(before.st_size),
    )
    after = physical.stat()
    if (
        _artifact_identity(before) != _artifact_identity(after)
        or observed_sha256 != expected_sha256
    ):
        raise ChatFiveExpertError("chat_training_artifact_snapshot_mismatch")
    return ArtifactSnapshot(
        path=physical,
        sha256=observed_sha256,
        identity=_artifact_identity(after),
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    """Write a receipt and sidecar inside an unpublished staging directory."""

    if os.path.lexists(path) or os.path.lexists(path.with_name(path.name + ".sha256")):
        raise FileExistsError(f"chat receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json(value) + b"\n"
    digest = _sha256(raw)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb", buffering=0) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        with sidecar_temporary.open("xb", buffering=0) as handle:
            handle.write(f"{digest}  {path.name}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
        os.rename(sidecar_temporary, sidecar)
    finally:
        for candidate in (temporary, sidecar_temporary):
            if os.path.lexists(candidate):
                candidate.unlink()
    if _stream_sha256(
        path
    ) != digest or sidecar.read_bytes() != f"{digest}  {path.name}\n".encode("ascii"):
        raise ChatFiveExpertError("chat_atomic_receipt_verification_failed")
    return digest


def _publish_failure_receipt(
    config: Mapping[str, Any],
    *,
    run_id: str,
    error_code: str,
    role: str | None,
    phase: str | None,
    failed_staging: str | None,
) -> Path | None:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        return None
    output = _mapping(config.get("output"), "chat_output_invalid")
    destination = _output_path(output["run_root"]) / "failures" / run_id
    if os.path.lexists(destination):
        return None
    value = {
        "schema_version": FAILURE_RECEIPT_VERSION,
        "status": "blocked",
        "run_id": run_id,
        "error_code": error_code,
        "role": role,
        "phase": phase,
        "failed_staging": failed_staging,
        "sample_content_included": False,
        "automatic_retry": False,
        "claims": {
            "diagnostic_only": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
            "live_authorized": False,
        },
    }
    return _atomic_publish_receipt_directory(
        destination,
        filename=str(output["failure_receipt_name"]),
        value=value,
    )


def _model_file_contract(
    config: Mapping[str, Any],
) -> dict[str, tuple[int, str]]:
    model = _mapping(config["model"], "chat_model_invalid")
    return {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in (
            _mapping(raw, "chat_model_file_invalid")
            for raw in _sequence(model["files"], "chat_model_files_invalid")
        )
    }


def _model_root(config: Mapping[str, Any]) -> Path:
    model = _mapping(config["model"], "chat_model_invalid")
    path = Path(str(model["local_path"])).expanduser()
    if not path.is_absolute():
        path = qdiag._project_root_from_module() / path
    return qdiag._assert_physical_path(
        path,
        require_directory=True,
        label="Gemma 3 1B IT chat source export",
    )


@contextmanager
def _private_model_snapshot(
    config: Mapping[str, Any],
    run_id: str,
) -> Iterator[tuple[Path, Mapping[str, str]]]:
    run_root = _output_path(config["output"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot = run_root / f".private-model-{run_id}-{uuid.uuid4().hex}"
    snapshot.mkdir(exist_ok=False)
    contract = _model_file_contract(config)
    hashes: dict[str, str] = {}
    try:
        source = _model_root(config)
        for name in MODEL_FILES:
            expected_bytes, expected_sha256 = contract[name]
            try:
                q8_runner._copy_authenticated_file(
                    source / name,
                    snapshot / name,
                    expected_bytes=expected_bytes,
                    expected_sha256=expected_sha256,
                )
            except q8_runner.GemmaFiveRoleError as exc:
                raise ChatFiveExpertError(f"chat_{exc}") from None
            hashes[name] = expected_sha256
        yield snapshot, hashes
        for name in MODEL_FILES:
            expected_bytes, expected_sha256 = contract[name]
            if (
                _stream_sha256(
                    snapshot / name,
                    expected_bytes=expected_bytes,
                )
                != expected_sha256
            ):
                raise ChatFiveExpertError(
                    "chat_private_model_snapshot_changed_before_cleanup"
                )
    finally:
        if os.path.lexists(snapshot):
            if snapshot.is_symlink():
                snapshot.unlink()
            else:
                shutil.rmtree(snapshot)


def _messages_prompt(messages: Sequence[tuple[str, str]]) -> str:
    """Render the producer-bound Gemma single-turn system/user prompt."""

    if (
        len(messages) != 2
        or messages[0][0] != "system"
        or messages[1][0] != "user"
        or not messages[0][1]
        or not messages[1][1]
    ):
        raise ChatFiveExpertError("chat_runtime_message_shape_invalid")
    return f"[SYSTEM_INSTRUCTION]\n{messages[0][1]}\n[USER_MESSAGE]\n{messages[1][1]}"


def _serialize_chat_example(
    processor: Any,
    example: ChatExample,
) -> SerializedChatExample:
    prompt = _messages_prompt(example.messages)
    message_objects = _message_objects(example.messages)
    training_identity = {
        "contract_version": "anchor.gemma3-it-chat-sft-serialization.v1",
        "messages": message_objects,
        "serialized_prompt": prompt,
        "assistant_text": example.target,
    }
    if (
        _sha256(prompt.encode("utf-8")) != example.serialized_prompt_sha256
        or _sha256(example.target.encode("utf-8")) != example.serialized_target_sha256
        or _sha256(_canonical_json(training_identity))
        != example.serialized_training_example_sha256
    ):
        raise ChatFiveExpertError("chat_runtime_serialization_identity_mismatch")
    try:
        serialized = gemma_binding.serialize_example(
            processor,
            prompt,
            example.target,
        )
    except (ConfigError, RuntimeError) as exc:
        raise ChatFiveExpertError(f"chat_runtime_serialization_failed:{exc}") from None
    if (
        len(serialized.input_ids) != len(serialized.labels)
        or len(serialized.input_ids) > 768
        or len(serialized.input_ids) < 2
    ):
        raise ChatFiveExpertError("chat_runtime_example_exceeds_strict_sequence_length")
    return SerializedChatExample(
        record_id=example.record_id,
        input_ids=tuple(int(value) for value in serialized.input_ids),
        labels=tuple(int(value) for value in serialized.labels),
    )


def _serialize_all_role_datasets(
    processor: Any,
    authenticated: AuthenticatedDataset,
) -> dict[str, SerializedRoleDataset]:
    """Serialize all 1,000 authenticated rows before selecting training work."""

    serialized_by_record = {
        example.record_id: _serialize_chat_example(processor, example)
        for example in authenticated.records
    }
    if len(serialized_by_record) != 1000:
        raise ChatFiveExpertError("chat_runtime_serialized_record_count_invalid")
    result: dict[str, SerializedRoleDataset] = {}
    for role in ROLES:
        selected = _select_role_dataset(authenticated, role)
        result[role] = SerializedRoleDataset(
            role=role,
            train=tuple(
                serialized_by_record[example.record_id] for example in selected.train
            ),
            eval_proxy=tuple(
                serialized_by_record[example.record_id]
                for example in selected.eval_proxy
            ),
        )
    if any(
        len(dataset.train) != 160 or len(dataset.eval_proxy) != 40
        for dataset in result.values()
    ):
        raise ChatFiveExpertError("chat_runtime_role_count_invalid")
    return result


def _tensor_sha256(tensor: Any, torch: Any) -> str:
    return qdiag._tensor_sha256(tensor, torch)


def _trainable_digest(model: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        count += 1
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(parameter, torch).encode("ascii"))
        digest.update(b"\n")
    if count != 52:
        raise ChatFiveExpertError("chat_q_only_trainable_tensor_count_invalid")
    return digest.hexdigest()


def _validate_trainable_scope(model: Any) -> tuple[tuple[str, ...], int]:
    observed: dict[tuple[int, str], str] = {}
    names: list[str] = []
    parameters = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        parameters += int(parameter.numel())
        match = _LORA_TENSOR_RE.search(name)
        if match is None:
            raise ChatFiveExpertError("chat_unexpected_trainable_tensor_outside_q_proj")
        key = (int(match.group(1)), match.group(2))
        if key in observed:
            raise ChatFiveExpertError("chat_duplicate_q_lora_tensor")
        observed[key] = name
        expected_shape = (1024, 1152) if key[1] == "A" else (1024, 1024)
        if tuple(int(value) for value in parameter.shape) != expected_shape:
            raise ChatFiveExpertError("chat_q_lora_tensor_shape_invalid")
    required = {(layer, side) for layer in range(26) for side in ("A", "B")}
    if set(observed) != required:
        raise ChatFiveExpertError("chat_q_lora_tensor_inventory_invalid")
    if parameters != 57_933_824:
        raise ChatFiveExpertError("chat_q_lora_parameter_count_invalid")
    return tuple(sorted(names)), parameters


def _assert_finite_and_gradient_scope(
    model: Any,
    torch: Any,
    *,
    completed_steps: int,
) -> dict[str, int]:
    if type(completed_steps) is not int or completed_steps < 1:
        raise ChatFiveExpertError("chat_gradient_step_invalid")
    result = {"A_nonzero": 0, "B_nonzero": 0}
    tensors = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        match = _LORA_TENSOR_RE.search(name)
        if match is None:
            raise ChatFiveExpertError("chat_gradient_scope_escaped_q_proj")
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise ChatFiveExpertError("chat_nonfinite_or_missing_q_lora_gradient")
        tensors += 1
        if int(torch.count_nonzero(gradient).item()) > 0:
            result[f"{match.group(2)}_nonzero"] += 1
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise ChatFiveExpertError("chat_nonfinite_q_lora_parameter")
    # PEFT initializes lora_B to zero.  On the first backward pass, every
    # layer's B must receive signal; from the second pass onward, the updated
    # B makes every layer's A gradient nonzero as well.  Counts are aggregated
    # in receipts, while this gate still checks the complete 26-layer set.
    if (
        tensors != 52
        or result["B_nonzero"] != 26
        or (completed_steps >= 2 and result["A_nonzero"] != 26)
    ):
        raise ChatFiveExpertError("chat_q_lora_gradient_coverage_failed")
    return result


def _validate_adamw8bit_state(
    optimizer: Any,
    parameters: Sequence[Any],
    *,
    torch: Any,
    bitsandbytes_version: str,
) -> dict[str, object]:
    if bitsandbytes_version != "0.48.2":
        raise ChatFiveExpertError("chat_bitsandbytes_version_drift")
    state_tensors = 0
    state_elements = 0
    for parameter in parameters:
        parameter_elements = int(parameter.numel())
        if parameter_elements < 4096:
            raise ChatFiveExpertError(
                "chat_adamw8bit_parameter_below_quantization_floor"
            )
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping):
            raise ChatFiveExpertError("chat_adamw8bit_state_missing")
        for name in ("state1", "state2"):
            tensor = state.get(name)
            device = getattr(tensor, "device", None)
            if (
                tensor is None
                or getattr(tensor, "dtype", None) != torch.uint8
                or getattr(device, "type", None) != "cuda"
                or int(tensor.numel()) != parameter_elements
            ):
                raise ChatFiveExpertError("chat_adamw8bit_state_not_uint8_cuda")
            state_tensors += 1
            state_elements += int(tensor.numel())
    if state_tensors != len(parameters) * 2:
        raise ChatFiveExpertError("chat_adamw8bit_state_inventory_invalid")
    return {
        **OPTIMIZER_RUNTIME_CONTRACT,
        "parameter_tensors": len(parameters),
        "state_tensors": state_tensors,
        "state_elements": state_elements,
        "state_dtype": "uint8",
        "state_device": "cuda",
    }


def _warmup_learning_rate(
    training: Mapping[str, Any],
    completed_steps: int,
) -> float:
    if type(completed_steps) is not int or completed_steps < 1:
        raise ChatFiveExpertError("chat_warmup_step_invalid")
    warmup_steps = int(training["warmup_steps"])
    base = float(training["learning_rate"])
    if warmup_steps != 8 or base != 0.000002:
        raise ChatFiveExpertError("chat_warmup_contract_drift")
    return base * min(completed_steps / warmup_steps, 1.0)


def _exact_supervised_loss_batch(
    batch: Mapping[str, Any],
    *,
    torch: Any,
) -> dict[str, Any]:
    labels = batch["labels"]
    if labels.ndim != 2 or labels.shape[0] != 1 or labels.shape[1] < 2:
        raise ChatFiveExpertError("chat_supervised_loss_labels_shape_invalid")
    supervised = labels[:, 1:].ne(-100)
    positions = torch.nonzero(supervised[0], as_tuple=False).flatten()
    if positions.numel() < 1:
        raise ChatFiveExpertError("chat_supervised_loss_has_no_targets")
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": labels,
        "logits_to_keep": positions,
        "shift_labels": labels[:, positions + 1].contiguous(),
        "use_cache": False,
    }


def _serialize_batch(
    example: SerializedChatExample,
    *,
    torch: Any,
) -> dict[str, Any]:
    if len(example.input_ids) > 768:
        raise ChatFiveExpertError("chat_runtime_example_exceeds_strict_sequence_length")
    return {
        "input_ids": torch.tensor(
            [example.input_ids],
            dtype=torch.long,
            device="cuda",
        ),
        "attention_mask": torch.ones(
            (1, len(example.input_ids)),
            dtype=torch.long,
            device="cuda",
        ),
        "labels": torch.tensor(
            [example.labels],
            dtype=torch.long,
            device="cuda",
        ),
    }


def _mean_eval_loss(
    model: Any,
    examples: Sequence[SerializedChatExample],
    *,
    torch: Any,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for example in examples:
            batch = _serialize_batch(example, torch=torch)
            loss = model(**_exact_supervised_loss_batch(batch, torch=torch)).loss
            if not bool(torch.isfinite(loss).item()):
                raise ChatFiveExpertError("chat_nonfinite_eval_proxy_loss")
            values.append(float(loss.detach().cpu()))
            del batch, loss
            torch.cuda.empty_cache()
    if len(values) != 40:
        raise ChatFiveExpertError("chat_eval_proxy_count_changed")
    return sum(values) / len(values)


def _adapter_effect_prefix_view(
    serialized: SerializedChatExample,
) -> dict[str, Any]:
    if (
        len(serialized.input_ids) != len(serialized.labels)
        or len(serialized.input_ids) > 768
    ):
        raise ChatFiveExpertError("chat_adapter_effect_view_exceeds_sequence_length")
    supervised = [
        index for index, label in enumerate(serialized.labels) if int(label) != -100
    ]
    if not supervised or supervised[0] <= 0:
        raise ChatFiveExpertError("chat_adapter_effect_supervised_boundary_invalid")
    first_supervised = supervised[0]
    input_prefix = tuple(serialized.input_ids[:first_supervised])
    if not input_prefix:
        raise ChatFiveExpertError("chat_adapter_effect_prefix_empty")
    return {
        "input_prefix": input_prefix,
        "prediction_position": len(input_prefix) - 1,
        "full_sequence_tokens": len(serialized.input_ids),
    }


def _enabled_vs_disabled_next_token_effect(
    model: Any,
    batch: Mapping[str, Any],
    prediction_position: int,
    *,
    torch: Any,
) -> dict[str, Any]:
    if prediction_position < 0:
        raise ChatFiveExpertError("chat_adapter_effect_prediction_position_invalid")
    disable_adapter = getattr(model, "disable_adapter", None)
    if not callable(disable_adapter):
        raise ChatFiveExpertError("chat_peft_disable_adapter_context_unavailable")
    model.eval()
    forward = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "logits_to_keep": torch.tensor(
            [prediction_position],
            dtype=torch.long,
            device=batch["input_ids"].device,
        ),
        "use_cache": False,
    }
    with torch.no_grad():
        enabled_output = model(**forward).logits
        if (
            enabled_output.ndim != 3
            or enabled_output.shape[0] != 1
            or enabled_output.shape[1] != 1
            or enabled_output.shape[2] <= 0
        ):
            raise ChatFiveExpertError(
                "chat_adapter_effect_enabled_logits_shape_invalid"
            )
        enabled_shape = tuple(int(value) for value in enabled_output.shape)
        enabled = enabled_output[0, 0].detach().float().clone()
        del enabled_output
        with disable_adapter():
            disabled_output = model(**forward).logits
            if tuple(int(value) for value in disabled_output.shape) != enabled_shape:
                raise ChatFiveExpertError(
                    "chat_adapter_effect_disabled_logits_shape_invalid"
                )
            disabled = disabled_output[0, 0].detach().float().clone()
        del disabled_output
    if not (
        bool(torch.isfinite(enabled).all().item())
        and bool(torch.isfinite(disabled).all().item())
    ):
        raise ChatFiveExpertError("chat_adapter_effect_nonfinite_logits")
    delta = torch.abs(enabled - disabled)
    max_abs = float(delta.max().detach().cpu())
    mean_abs = float(delta.mean().detach().cpu())
    if (
        not bool(torch.isfinite(delta).all().item())
        or not math.isfinite(max_abs)
        or not math.isfinite(mean_abs)
        or max_abs <= 0.0
        or mean_abs < 0.0
    ):
        raise ChatFiveExpertError("chat_adapter_output_effect_absent_or_invalid")
    return {
        "finite": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "vocabulary_logits": int(delta.numel()),
    }


def _fixed_training_view_adapter_effect(
    model: Any,
    example: SerializedChatExample,
    *,
    role: str,
    torch: Any,
) -> dict[str, Any]:
    view = _adapter_effect_prefix_view(example)
    prefix = view["input_prefix"]
    position = int(view["prediction_position"])
    batch = {
        "input_ids": torch.tensor(
            [prefix],
            dtype=torch.long,
            device="cuda",
        ),
        "attention_mask": torch.ones(
            (1, len(prefix)),
            dtype=torch.long,
            device="cuda",
        ),
    }
    metrics = _enabled_vs_disabled_next_token_effect(
        model,
        batch,
        position,
        torch=torch,
    )
    return {
        "view": "first_train_record_first_supervised_next_token_v1",
        "comparison": "enabled_vs_disable_adapter_after_training",
        "record_id_sha256": _sha256(
            (
                "anchor.gemma3.chat-adapter-effect-record.v1"
                f"\0{role}\0{example.record_id}"
            ).encode("utf-8")
        ),
        "serialized_training_view_sha256": _sha256(
            _canonical_json(
                {
                    "input_prefix": list(prefix),
                    "prediction_position": position,
                }
            )
        ),
        "full_serialized_sequence_tokens": view["full_sequence_tokens"],
        "forward_prefix_tokens": len(prefix),
        "prediction_position_zero_based_in_forward_prefix": position,
        "future_target_suffix_forwarded": False,
        "sample_body_included": False,
        "token_ids_included": False,
        **metrics,
    }


def _save_adapter(
    model: Any,
    destination: Path,
    *,
    torch: Any,
) -> Mapping[str, str]:
    if os.path.lexists(destination):
        raise FileExistsError(f"chat adapter output exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        readme = temporary / "README.md"
        if os.path.lexists(readme):
            qdiag._assert_physical_path(
                readme,
                require_file=True,
                label="generated chat PEFT README",
            )
            readme.unlink()
        entries = list(temporary.iterdir())
        if any(not item.is_file() or item.is_symlink() for item in entries) or {
            item.name for item in entries
        } != set(ADAPTER_FILES):
            raise ChatFiveExpertError("chat_saved_adapter_file_inventory_invalid")
        config_path = temporary / "adapter_config.json"
        try:
            adapter_config = json.loads(config_path.read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ChatFiveExpertError(
                "chat_saved_adapter_config_json_invalid"
            ) from None
        if not isinstance(adapter_config, dict):
            raise ChatFiveExpertError("chat_saved_adapter_config_json_invalid")
        adapter_config["base_model_name_or_path"] = ADAPTER_BASE_IDENTITY
        normalized = config_path.with_name(
            f".adapter_config.json.tmp-{uuid.uuid4().hex}"
        )
        with normalized.open("xb", buffering=0) as handle:
            handle.write(_canonical_json(adapter_config) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(normalized, config_path)
        target_modules = adapter_config.get("target_modules")
        if (
            adapter_config.get("r") != 1024
            or adapter_config.get("lora_alpha") != 2048
            or float(adapter_config.get("lora_dropout", -1.0)) != 0.0
            or set(target_modules or ()) != {"q_proj"}
            or adapter_config.get("bias") != "none"
            or adapter_config.get("task_type") != "CAUSAL_LM"
            or adapter_config.get("base_model_name_or_path") != ADAPTER_BASE_IDENTITY
        ):
            raise ChatFiveExpertError("chat_saved_adapter_config_contract_invalid")
        try:
            from safetensors import safe_open
        except ImportError:
            raise ChatFiveExpertError("chat_safetensors_runtime_unavailable") from None
        observed: set[tuple[int, str]] = set()
        with safe_open(
            temporary / "adapter_model.safetensors",
            framework="pt",
            device="cpu",
        ) as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                match = _LORA_TENSOR_RE.search(name)
                if match is None:
                    raise ChatFiveExpertError("chat_saved_adapter_tensor_scope_invalid")
                layer = int(match.group(1))
                side = match.group(2)
                expected_shape = (1024, 1152) if side == "A" else (1024, 1024)
                if (
                    tuple(int(value) for value in tensor.shape) != expected_shape
                    or tensor.dtype != torch.bfloat16
                ):
                    raise ChatFiveExpertError(
                        "chat_saved_adapter_tensor_contract_invalid"
                    )
                observed.add((layer, side))
        required = {(layer, side) for layer in range(26) for side in ("A", "B")}
        if observed != required:
            raise ChatFiveExpertError("chat_saved_adapter_tensor_inventory_invalid")
        hashes = {name: _stream_sha256(temporary / name) for name in ADAPTER_FILES}
        os.rename(temporary, destination)
        for name, digest in hashes.items():
            if _stream_sha256(destination / name) != digest:
                raise ChatFiveExpertError("chat_published_adapter_hash_drift")
        return hashes
    finally:
        if os.path.lexists(temporary):
            shutil.rmtree(temporary)


def _prepare_frozen_q8_base(
    base: Any,
    *,
    torch: Any,
    bnb: Any,
) -> dict[str, int]:
    try:
        return q8_runner._prepare_frozen_q8_base(
            base,
            torch=torch,
            bnb=bnb,
        )
    except q8_runner.GemmaFiveRoleError as exc:
        raise ChatFiveExpertError(f"chat_{exc}") from None


def _validate_q8_base_modules(
    base: Any,
    *,
    bnb: Any,
    torch: Any,
) -> dict[str, int]:
    try:
        return q8_runner._validate_q8_base_modules(
            base,
            bnb=bnb,
            torch=torch,
        )
    except q8_runner.GemmaFiveRoleError as exc:
        raise ChatFiveExpertError(f"chat_{exc}") from None


def _q8_quant_state_inventory(
    base: Any,
    *,
    bnb: Any,
    torch: Any,
) -> dict[str, object]:
    try:
        return q8_runner._q8_quant_state_inventory(
            base,
            bnb=bnb,
            torch=torch,
        )
    except q8_runner.GemmaFiveRoleError as exc:
        raise ChatFiveExpertError(f"chat_{exc}") from None


def _strict_memory_gate(
    config: Mapping[str, Any],
    *,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    monitor_samples: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    if (
        type(peak_allocated_bytes) is not int
        or type(peak_reserved_bytes) is not int
        or peak_allocated_bytes < 0
        or peak_reserved_bytes < 0
    ):
        raise ChatFiveExpertError("chat_strict_memory_gate_values_invalid")
    try:
        return q8_runner._strict_memory_gate(
            config,
            peak_allocated_bytes=peak_allocated_bytes,
            peak_reserved_bytes=peak_reserved_bytes,
            monitor_samples=monitor_samples,
        )
    except q8_runner.GemmaFiveRoleError as exc:
        raise ChatFiveExpertError(f"chat_{exc}") from None


def _query_runtime_gpu(
    config: Mapping[str, Any],
    *,
    allow_pid: int,
) -> dict[str, Any]:
    translated = {
        "gpu_policy": {
            **dict(config["gpu_policy"]),
            "wddm_gui_process_allowlist": list(q8_runner.WDDM_GUI_PROCESS_ALLOWLIST),
        }
    }
    try:
        return q8_runner._query_runtime_gpu(
            translated,
            allow_pid=allow_pid,
        )
    except q8_runner.GemmaFiveRoleError as exc:
        raise ChatFiveExpertError(f"chat_{exc}") from None


def _external_lock_receipt_relative(run_id: str) -> str:
    return (
        "runs/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1/"
        f"{LOCK_OWNER_RECEIPT_DIRECTORY}/{run_id}/lock_owner.json"
    )


def _expected_external_lock_owner(
    config: Mapping[str, Any],
    *,
    run_id: str,
    nonce: str,
    launcher_pid: int,
) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(nonce) is None or nonce == "0" * 64:
        raise ChatFiveExpertError("chat_external_gpu_lock_nonce_invalid")
    if (
        type(launcher_pid) is not int
        or launcher_pid <= 0
        or launcher_pid > _MAX_WINDOWS_PID
    ):
        raise ChatFiveExpertError("chat_external_gpu_lock_identity_invalid")
    root = qdiag._project_root_from_module()
    gpu = _mapping(config["gpu_policy"], "chat_gpu_policy_invalid")
    dependency_sha256 = {
        relative: _stream_sha256(
            qdiag._assert_physical_path(
                root.joinpath(*PurePosixPath(relative).parts),
                require_file=True,
                label=f"chat execution dependency {relative}",
            )
        )
        for relative in EXECUTION_DEPENDENCY_PATHS
    }
    return {
        "schema_version": LOCK_OWNER_VERSION,
        "run_id": run_id,
        "launcher_pid": launcher_pid,
        "canonical_lock": str(gpu["canonical_lock"]),
        "owner_receipt": _external_lock_receipt_relative(run_id),
        "expected_gpu_index": 0,
        "expected_gpu_uuid": str(gpu["expected_gpu_uuid"]),
        "roles": list(ROLES),
        "concurrency": 1,
        "smoke_steps_per_role": 2,
        "full_steps_per_role": 160,
        "fresh_base_per_phase": True,
        "fresh_adapter_per_phase": True,
        "resume": False,
        "config_sha256": config["_config_sha256"],
        "implementation_sha256": _stream_sha256(
            qdiag._assert_physical_path(
                root.joinpath(*PurePosixPath(IMPLEMENTATION_PATH).parts),
                require_file=True,
                label="chat training implementation",
            )
        ),
        "runner_script_sha256": _stream_sha256(
            qdiag._assert_physical_path(
                root.joinpath(*PurePosixPath(SCRIPT_PATH).parts),
                require_file=True,
                label="chat training runner script",
            )
        ),
        "launcher_sha256": _stream_sha256(
            qdiag._assert_physical_path(
                root.joinpath(*PurePosixPath(LAUNCHER_PATH).parts),
                require_file=True,
                label="chat training launcher",
            )
        ),
        "launcher_helper_sha256": dependency_sha256[LAUNCHER_HELPER_PATH],
        "execution_dependency_sha256": dependency_sha256,
        "nonce": nonce,
        "owner": "powershell_create_new_file_share_none",
        "file_mode": "CreateNew",
        "file_access": "ReadWrite",
        "file_share": "None",
        "delete_on_close": True,
        "write_through": True,
        "held_for_entire_python_lifetime": True,
    }


def _strict_lock_owner_json(data: bytes) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("nonfinite")

    if b"\r" in data or data.count(b"\n") != 1 or not data.endswith(b"\n"):
        raise ChatFiveExpertError("chat_external_gpu_lock_owner_json_invalid")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ChatFiveExpertError("chat_external_gpu_lock_owner_json_invalid") from None
    if not isinstance(value, Mapping):
        raise ChatFiveExpertError("chat_external_gpu_lock_owner_json_invalid")
    return value


def _assert_external_lock_exclusive(lock_path: Path) -> None:
    """Prove that the Windows launcher denies every independent read handle."""

    if os.name != "nt":
        raise ChatFiveExpertError("chat_external_gpu_lock_requires_windows")
    # The CRT collapses ERROR_SHARING_VIOLATION into EACCES and can discard
    # Win32 error 32.  Call CreateFileW directly so only the precise sharing
    # violation proves that the launcher-held handle uses FileShare.None.
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    handle = create_file(
        str(lock_path),
        0x80000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x00000080,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        if ctypes.get_last_error() == 32:
            return
        raise ChatFiveExpertError("chat_external_gpu_lock_not_exclusive") from None
    close_handle(handle)
    raise ChatFiveExpertError("chat_external_gpu_lock_not_exclusive")


def _validated_external_lock_launcher_pid(value: Any) -> int:
    """Decode the dedicated launcher channel without consulting parentage."""

    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ChatFiveExpertError("chat_external_gpu_lock_identity_invalid")
    launcher_pid = int(value, 10)
    if launcher_pid > _MAX_WINDOWS_PID:
        raise ChatFiveExpertError("chat_external_gpu_lock_identity_invalid")
    return launcher_pid


def _load_external_lock_owner(
    config: Mapping[str, Any],
    *,
    run_id: str,
    lock_path: Path,
    receipt_value: str,
    sha256_value: str,
    nonce: str,
    launcher_pid_value: str,
) -> tuple[dict[str, Any], snapshot_io.FileSnapshot, snapshot_io.FileSnapshot]:
    launcher_pid = _validated_external_lock_launcher_pid(launcher_pid_value)
    if (
        _SHA256_RE.fullmatch(sha256_value) is None
        or sha256_value == "0" * 64
        or _SHA256_RE.fullmatch(nonce) is None
        or nonce == "0" * 64
    ):
        raise ChatFiveExpertError("chat_external_gpu_lock_identity_invalid")
    root = qdiag._project_root_from_module()
    expected_receipt = root.joinpath(
        *PurePosixPath(_external_lock_receipt_relative(run_id)).parts
    )
    requested_receipt = Path(receipt_value)
    if not requested_receipt.is_absolute():
        raise ChatFiveExpertError("chat_external_gpu_lock_receipt_path_invalid")
    requested_receipt = Path(os.path.abspath(requested_receipt))
    if os.path.normcase(str(requested_receipt)) != os.path.normcase(
        str(expected_receipt)
    ):
        raise ChatFiveExpertError("chat_external_gpu_lock_receipt_path_invalid")
    receipt_snapshot = snapshot_io._read_snapshot(
        qdiag._assert_physical_path(
            expected_receipt,
            require_file=True,
            label="chat external GPU lock owner receipt",
        ),
        max_bytes=_MAX_METADATA_BYTES,
    )
    sidecar_path = expected_receipt.with_name(expected_receipt.name + ".sha256")
    sidecar_snapshot = snapshot_io._read_snapshot(
        qdiag._assert_physical_path(
            sidecar_path,
            require_file=True,
            label="chat external GPU lock owner sidecar",
        ),
        max_bytes=1024,
    )
    expected_sidecar = f"{receipt_snapshot.sha256}  {expected_receipt.name}\n".encode(
        "ascii"
    )
    if (
        receipt_snapshot.sha256 != sha256_value
        or sidecar_snapshot.data != expected_sidecar
    ):
        raise ChatFiveExpertError("chat_external_gpu_lock_receipt_digest_invalid")
    observed = _strict_lock_owner_json(receipt_snapshot.data)
    expected = _expected_external_lock_owner(
        config,
        run_id=run_id,
        nonce=nonce,
        launcher_pid=launcher_pid,
    )
    if _canonical_json(observed) != _canonical_json(expected):
        raise ChatFiveExpertError("chat_external_gpu_lock_owner_drift")
    if not os.path.lexists(lock_path):
        raise ChatFiveExpertError("chat_external_gpu_lock_disappeared")
    qdiag._assert_physical_path(
        lock_path,
        require_file=True,
        label="chat external canonical GPU lock",
    )
    _assert_external_lock_exclusive(lock_path)
    return (
        {
            **dict(observed),
            "content_sha256": receipt_snapshot.sha256,
            "authenticated_owner_receipt": _external_lock_receipt_relative(run_id),
        },
        receipt_snapshot,
        sidecar_snapshot,
    )


@contextmanager
def _canonical_gpu_lock(
    config: Mapping[str, Any],
    *,
    run_id: str,
) -> Iterator[Mapping[str, Any]]:
    """Acquire the repository-wide no-replace GPU lock for the whole run."""

    gpu = _mapping(config["gpu_policy"], "chat_gpu_policy_invalid")
    expected_uuid = str(gpu["expected_gpu_uuid"])
    if (
        expected_uuid == "UNBOUND"
        or re.fullmatch(r"GPU-[0-9A-Fa-f-]{32,64}", expected_uuid) is None
    ):
        raise ChatFiveExpertError("chat_execute_requires_bound_gpu_uuid")
    external_values = {
        key: os.environ.get(name) for key, name in EXTERNAL_LOCK_ENV.items()
    }
    if not all(value is not None for value in external_values.values()):
        if any(value is not None for value in external_values.values()):
            raise ChatFiveExpertError("chat_external_gpu_lock_identity_invalid")
        raise ChatFiveExpertError("chat_external_gpu_lock_required")
    root = qdiag._project_root_from_module()
    lock_path = root.joinpath(*PurePosixPath(str(gpu["canonical_lock"])).parts)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    qdiag._assert_physical_path(
        lock_path.parent,
        require_directory=True,
        label="chat canonical GPU lock parent",
    )
    for relative in gpu["conflicting_handoff_locks"]:
        if os.path.lexists(root.joinpath(*PurePosixPath(str(relative)).parts)):
            raise ChatFiveExpertError("chat_conflicting_handoff_gpu_lock_present")
    external_owner, receipt_snapshot, sidecar_snapshot = _load_external_lock_owner(
        config,
        run_id=run_id,
        lock_path=lock_path,
        receipt_value=str(external_values["receipt"]),
        sha256_value=str(external_values["sha256"]),
        nonce=str(external_values["nonce"]),
        launcher_pid_value=str(external_values["launcher_pid"]),
    )
    try:
        yield external_owner
    finally:
        receipt_snapshot.assert_unchanged()
        sidecar_snapshot.assert_unchanged()
        if not os.path.lexists(lock_path):
            raise ChatFiveExpertError("chat_external_gpu_lock_disappeared")
        _assert_external_lock_exclusive(lock_path)
        for relative in gpu["conflicting_handoff_locks"]:
            if os.path.lexists(root.joinpath(*PurePosixPath(str(relative)).parts)):
                raise ChatFiveExpertError("chat_conflicting_handoff_gpu_lock_appeared")


def _write_training_progress(
    path: Path,
    *,
    run_id: str,
    role: str,
    phase: str,
    step: int,
    target: int,
    loss: float,
    torch_peak_allocated_bytes: int,
    torch_peak_reserved_bytes: int,
) -> None:
    """Publish a content-free scalar progress snapshot within staging."""

    if (
        role not in ROLES
        or phase not in {"smoke", "full"}
        or type(step) is not int
        or type(target) is not int
        or not 1 <= step <= target
        or not math.isfinite(loss)
    ):
        raise ChatFiveExpertError("chat_training_progress_invalid")
    value = {
        "schema_version": ("anchor.gemma3-1b-it-chat-five-expert-qonly-progress.v1"),
        "run_id": run_id,
        "role": role,
        "phase": phase,
        "step": step,
        "target": target,
        "loss": loss,
        "torch_peak_allocated_bytes": torch_peak_allocated_bytes,
        "torch_peak_reserved_bytes": torch_peak_reserved_bytes,
        "sample_content_included": False,
        "token_ids_included": False,
    }
    raw = _canonical_json(value) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb", buffering=0) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _validate_runtime_package_versions(value: object) -> dict[str, str]:
    versions = _mapping(value, "chat_runtime_package_versions_invalid")
    if set(versions) != set(RUNTIME_PACKAGE_NAMES):
        raise ChatFiveExpertError("chat_runtime_package_version_inventory_invalid")
    result: dict[str, str] = {}
    for name in RUNTIME_PACKAGE_NAMES:
        version = versions.get(name)
        if (
            not isinstance(version, str)
            or not version
            or len(version) > 128
            or any(character.isspace() for character in version)
        ):
            raise ChatFiveExpertError("chat_runtime_package_version_invalid")
        result[name] = version
    if result["bitsandbytes"] != "0.48.2":
        raise ChatFiveExpertError("chat_bitsandbytes_version_drift")
    return result


def _runtime_package_versions() -> dict[str, str]:
    try:
        observed = {
            name: importlib.metadata.version(name) for name in RUNTIME_PACKAGE_NAMES
        }
    except importlib.metadata.PackageNotFoundError:
        raise ChatFiveExpertError("chat_runtime_package_version_unavailable") from None
    return _validate_runtime_package_versions(observed)


def _execute_phase(
    config: Mapping[str, Any],
    *,
    role: str,
    phase: str,
    steps: int,
    dataset: SerializedRoleDataset,
    model_snapshot: Path,
    role_output: Path,
    run_id: str,
    progress_path: Path,
) -> dict[str, Any]:
    """Create a fresh Q8 base, adapter, and optimizer for exactly one phase."""

    if (
        role not in ROLES
        or dataset.role != role
        or phase not in {"smoke", "full"}
        or steps != (2 if phase == "smoke" else 160)
    ):
        raise ChatFiveExpertError("chat_phase_contract_invalid")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ChatFiveExpertError("chat_cuda_visible_devices_must_be_exactly_zero")
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if getattr(bnb, "__version__", None) != "0.48.2":
        raise ChatFiveExpertError("chat_bitsandbytes_version_drift")
    runtime_package_versions = _runtime_package_versions()
    if runtime_package_versions["bitsandbytes"] != str(bnb.__version__):
        raise ChatFiveExpertError("chat_bitsandbytes_module_metadata_version_drift")
    if not torch.cuda.is_available():
        raise ChatFiveExpertError("chat_execute_requires_cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not (torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32):
        raise ChatFiveExpertError("chat_tf32_controls_not_enabled")
    training = _mapping(config["training"], "chat_training_invalid")
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor_samples: list[Mapping[str, Any]] = [
        _query_runtime_gpu(config, allow_pid=os.getpid())
    ]

    quantization = _mapping(
        config["quantization"],
        "chat_quantization_invalid",
    )
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(quantization["llm_int8_threshold"]),
            llm_int8_skip_modules=list(quantization["llm_int8_skip_modules"]),
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_has_fp16_weight=False,
        ),
    }
    base = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        **kwargs,
    )
    overlay = _mapping(
        config["model"]["runtime_special_token_overlay"],
        "chat_model_special_token_overlay_invalid",
    )
    base.config.pad_token_id = int(overlay["pad_token_id"])
    base.config.eos_token_id = int(overlay["eos_token_id"])
    base.config.bos_token_id = int(overlay["bos_token_id"])
    q8_modules = _validate_q8_base_modules(
        base,
        bnb=bnb,
        torch=torch,
    )
    preparation = _prepare_frozen_q8_base(
        base,
        torch=torch,
        bnb=bnb,
    )
    q8_before = _q8_quant_state_inventory(
        base,
        bnb=bnb,
        torch=torch,
    )
    captured_base, base_hash_before = qdiag._capture_base_parameters(
        base,
        torch,
    )

    # Base loading must never influence adapter initialization.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = get_peft_model(
        base,
        LoraConfig(
            r=1024,
            lora_alpha=2048,
            lora_dropout=0.0,
            target_modules=["q_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
        autocast_adapter_dtype=False,
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    trainable_names, trainable_parameters = _validate_trainable_scope(model)
    initial_adapter_sha256 = _trainable_digest(model, torch)
    eval_before = (
        _mean_eval_loss(model, dataset.eval_proxy, torch=torch)
        if phase == "full"
        else None
    )
    trainable_tensors = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = bnb.optim.AdamW8bit(
        trainable_tensors,
        lr=float(training["learning_rate"]),
        betas=(
            float(training["beta1"]),
            float(training["beta2"]),
        ),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
        amsgrad=bool(training["amsgrad"]),
        optim_bits=int(training["compatibility_optim_bits_argument"]),
        min_8bit_size=int(training["min_8bit_size"]),
        percentile_clipping=int(training["percentile_clipping"]),
        block_wise=bool(training["block_wise"]),
        is_paged=bool(training["is_paged"]),
    )
    losses: list[float] = []
    learning_rates: list[float] = []
    gradients: dict[str, int] = {}
    optimizer_state: dict[str, object] | None = None
    order = hashlib.sha256()
    for step_index in range(steps):
        completed_steps = step_index + 1
        example = dataset.train[step_index]
        order.update(example.record_id.encode("utf-8"))
        order.update(b"\n")
        learning_rate = _warmup_learning_rate(
            training,
            completed_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        learning_rates.append(learning_rate)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = _serialize_batch(example, torch=torch)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            loss = model(**_exact_supervised_loss_batch(batch, torch=torch)).loss
        if not bool(torch.isfinite(loss).item()):
            raise ChatFiveExpertError("chat_nonfinite_training_loss")
        loss.backward()
        gradients = _assert_finite_and_gradient_scope(
            model,
            torch,
            completed_steps=completed_steps,
        )
        torch.nn.utils.clip_grad_norm_(
            trainable_tensors,
            max_norm=float(training["max_grad_norm"]),
        )
        optimizer.step()
        if optimizer_state is None:
            optimizer_state = _validate_adamw8bit_state(
                optimizer,
                trainable_tensors,
                torch=torch,
                bitsandbytes_version=str(bnb.__version__),
            )
        qdiag._assert_trainable_parameters_finite(model, torch)
        losses.append(float(loss.detach().cpu()))
        del batch, loss
        torch.cuda.empty_cache()
        _write_training_progress(
            progress_path,
            run_id=run_id,
            role=role,
            phase=phase,
            step=completed_steps,
            target=steps,
            loss=losses[-1],
            torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            torch_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
        if (
            phase == "smoke"
            or completed_steps
            % int(config["gpu_policy"]["runtime_monitor_interval_steps"])
            == 0
            or completed_steps == steps
        ):
            monitor_samples.append(_query_runtime_gpu(config, allow_pid=os.getpid()))

    final_adapter_sha256 = _trainable_digest(model, torch)
    if final_adapter_sha256 == initial_adapter_sha256:
        raise ChatFiveExpertError("chat_adapter_did_not_change")
    if optimizer_state is None:
        raise ChatFiveExpertError("chat_adamw8bit_state_not_observed")
    adapter_effect = _fixed_training_view_adapter_effect(
        model,
        dataset.train[0],
        role=role,
        torch=torch,
    )
    eval_after = (
        _mean_eval_loss(model, dataset.eval_proxy, torch=torch)
        if phase == "full"
        else None
    )
    try:
        base_hash_after = qdiag._assert_base_parameters_unchanged(
            captured_base,
            torch,
        )
    except RuntimeError as exc:
        raise ChatFiveExpertError(f"chat_{exc}") from None
    q8_after = _q8_quant_state_inventory(
        base,
        bnb=bnb,
        torch=torch,
    )
    if q8_after != q8_before:
        raise ChatFiveExpertError("chat_q8_scale_state_changed")
    monitor_samples.append(_query_runtime_gpu(config, allow_pid=os.getpid()))
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    memory_gate = _strict_memory_gate(
        config,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        monitor_samples=monitor_samples,
    )
    adapter_hashes: Mapping[str, str] | None = None
    if phase == "full":
        adapter_hashes = _save_adapter(
            model,
            role_output / "adapter",
            torch=torch,
        )
    receipt = {
        "schema_version": PHASE_RECEIPT_VERSION,
        "status": "passed_diagnostic_training_only",
        "run_id": run_id,
        "role": role,
        "phase": phase,
        "optimizer_steps": steps,
        "seed": seed,
        "fresh_base": True,
        "fresh_adapter": True,
        "resume": False,
        "smoke_checkpoint_consumed": False,
        "train_record_order_sha256": order.hexdigest(),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "learning_rate_first": learning_rates[0],
        "learning_rate_last": learning_rates[-1],
        "warmup_steps": 8,
        "eval_proxy_loss_before": eval_before,
        "eval_proxy_loss_after": eval_after,
        "eval_proxy_loss_delta": (
            None if eval_before is None else eval_after - eval_before
        ),
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "q8_scb_sha256_before": q8_before["sha256"],
        "q8_scb_sha256_after": q8_after["sha256"],
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_sha256": final_adapter_sha256,
        "trainable_tensor_names_sha256": _sha256(
            ("\n".join(trainable_names) + "\n").encode("utf-8")
        ),
        "trainable_parameters": trainable_parameters,
        "gradient_coverage": gradients,
        "runtime_package_versions": runtime_package_versions,
        "optimizer": optimizer_state,
        "base_quantization": {
            **dict(Q8_BASE_CONTRACT),
            **q8_modules,
            **preparation,
            "scale_state_inventory_schema_version": (Q8_SCB_INVENTORY_VERSION),
            "scale_elements": q8_before["scale_elements"],
        },
        "adapter_effect": adapter_effect,
        "adapter_artifact_sha256": adapter_hashes,
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "memory_gate": memory_gate,
        "runtime_gpu_samples": monitor_samples,
        "numerics": {
            "compute_dtype": "bfloat16",
            "tf32_allowed": True,
            "int8_matmul_tf32_claimed": False,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
        "claims": {
            "diagnostic_only": True,
            "training_executed": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
            "live_authorized": False,
        },
    }
    del optimizer, model, base, captured_base
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def _assert_smoke_full_freshness(
    smoke: Mapping[str, Any],
    full: Mapping[str, Any],
) -> None:
    smoke_versions = _validate_runtime_package_versions(
        smoke.get("runtime_package_versions")
    )
    full_versions = _validate_runtime_package_versions(
        full.get("runtime_package_versions")
    )
    if (
        smoke_versions != full_versions
        or smoke.get("initial_adapter_sha256") != full.get("initial_adapter_sha256")
        or smoke.get("base_hash_before") != full.get("base_hash_before")
        or smoke.get("q8_scb_sha256_before") != full.get("q8_scb_sha256_before")
        or smoke.get("train_record_order_sha256")
        == full.get("train_record_order_sha256")
        or smoke.get("optimizer_steps") != 2
        or full.get("optimizer_steps") != 160
        or smoke.get("fresh_base") is not True
        or full.get("fresh_base") is not True
        or smoke.get("fresh_adapter") is not True
        or full.get("fresh_adapter") is not True
        or smoke.get("smoke_checkpoint_consumed") is not False
        or full.get("smoke_checkpoint_consumed") is not False
        or smoke.get("resume") is not False
        or full.get("resume") is not False
    ):
        raise ChatFiveExpertError("chat_smoke_full_freshness_gate_failed")


class ChatPhaseError(ChatFiveExpertError):
    def __init__(self, role: str, phase: str, cause: BaseException) -> None:
        super().__init__(
            f"chat_phase_failed:{role}:{phase}:{type(cause).__name__}:{cause}"
        )
        self.role = role
        self.phase = phase


def _run_serial_phases(
    config: Mapping[str, Any],
    *,
    datasets: Mapping[str, SerializedRoleDataset],
    model_snapshot: Path,
    staging: Path,
    run_id: str,
    phase_executor: Any = None,
) -> list[Mapping[str, Any]]:
    executor = phase_executor or _execute_phase
    roles: list[Mapping[str, Any]] = []
    for role in ROLES:
        if role not in datasets:
            raise ChatFiveExpertError("chat_serial_dataset_role_missing")
        role_output = staging / role
        role_output.mkdir(exist_ok=False)
        receipts: dict[str, Mapping[str, Any]] = {}
        for phase, steps in (("smoke", 2), ("full", 160)):
            try:
                receipt = executor(
                    config,
                    role=role,
                    phase=phase,
                    steps=steps,
                    dataset=datasets[role],
                    model_snapshot=model_snapshot,
                    role_output=role_output,
                    run_id=run_id,
                    progress_path=staging / "training_progress.json",
                )
            except Exception as exc:
                raise ChatPhaseError(role, phase, exc) from exc
            if (
                receipt.get("role") != role
                or receipt.get("phase") != phase
                or receipt.get("optimizer_steps") != steps
            ):
                raise ChatPhaseError(
                    role,
                    phase,
                    ChatFiveExpertError("chat_phase_receipt_cross_binding_invalid"),
                )
            _atomic_write_json(
                role_output / f"{phase}_receipt.json",
                receipt,
            )
            receipts[phase] = receipt
        smoke = receipts["smoke"]
        full = receipts["full"]
        _assert_smoke_full_freshness(smoke, full)
        adapter_hashes = full.get("adapter_artifact_sha256")
        if not isinstance(adapter_hashes, Mapping) or set(adapter_hashes) != set(
            ADAPTER_FILES
        ):
            raise ChatFiveExpertError("chat_full_adapter_artifact_identity_missing")
        roles.append(
            {
                "role": role,
                "smoke_receipt_sha256": _stream_sha256(
                    role_output / "smoke_receipt.json"
                ),
                "full_receipt_sha256": _stream_sha256(
                    role_output / "full_receipt.json"
                ),
                "adapter_artifact_sha256": dict(adapter_hashes),
            }
        )
    return roles


def _assert_directory_inventory(
    path: Path,
    *,
    expected_files: set[str],
    expected_directories: set[str],
    code: str,
) -> None:
    physical = qdiag._assert_physical_path(
        path,
        require_directory=True,
        label="chat training artifact directory",
    )
    entries = list(physical.iterdir())
    if {entry.name for entry in entries} != (expected_files | expected_directories):
        raise ChatFiveExpertError(code)
    for entry in entries:
        qdiag._assert_physical_path(
            entry,
            require_file=entry.name in expected_files,
            require_directory=entry.name in expected_directories,
            label="chat training artifact inventory entry",
        )


def _remove_training_progress_for_publish(root: Path) -> None:
    progress = root / "training_progress.json"
    if os.path.lexists(progress):
        qdiag._assert_physical_path(
            progress,
            require_file=True,
            label="chat ephemeral training progress",
        )
        progress.unlink()
    if os.path.lexists(progress):
        raise ChatFiveExpertError("chat_training_progress_cleanup_failed")


def _assert_success_root_inventory(
    root: Path,
    *,
    run_receipt_name: str,
    receipt_present: bool,
) -> None:
    expected_files = (
        {run_receipt_name, f"{run_receipt_name}.sha256"} if receipt_present else set()
    )
    _assert_directory_inventory(
        root,
        expected_files=expected_files,
        expected_directories=set(ROLES),
        code="chat_success_root_artifact_inventory_invalid",
    )


def _capture_training_artifacts(
    root: Path,
    *,
    roles: Sequence[Mapping[str, Any]],
    run_id: str,
) -> tuple[dict[str, ArtifactSnapshot], dict[str, str]]:
    if tuple(entry.get("role") for entry in roles) != ROLES:
        raise ChatFiveExpertError("chat_training_role_artifact_order_invalid")
    snapshots: dict[str, ArtifactSnapshot] = {}
    runtime_versions: dict[str, str] | None = None
    for entry in roles:
        _exact_keys(
            entry,
            {
                "role",
                "smoke_receipt_sha256",
                "full_receipt_sha256",
                "adapter_artifact_sha256",
            },
            "chat_training_role_artifact_fields_drift",
        )
        role = str(entry["role"])
        role_root = root / role
        _assert_directory_inventory(
            role_root,
            expected_files={
                "smoke_receipt.json",
                "smoke_receipt.json.sha256",
                "full_receipt.json",
                "full_receipt.json.sha256",
            },
            expected_directories={
                "adapter",
            },
            code="chat_training_role_artifact_inventory_invalid",
        )
        adapter_hashes = _mapping(
            entry.get("adapter_artifact_sha256"),
            "chat_training_adapter_artifact_identity_invalid",
        )
        if set(adapter_hashes) != set(ADAPTER_FILES):
            raise ChatFiveExpertError(
                "chat_training_adapter_artifact_inventory_invalid"
            )
        for phase, steps in (("smoke", 2), ("full", 160)):
            receipt_name = f"{phase}_receipt.json"
            expected_receipt_sha256 = _identity(
                entry.get(f"{phase}_receipt_sha256"),
                "chat_training_phase_receipt_identity_invalid",
                pending_allowed=False,
            )
            receipt_path = role_root / receipt_name
            receipt_snapshot = snapshot_io._read_snapshot(
                receipt_path,
                max_bytes=_MAX_RECEIPT_BYTES,
            )
            if receipt_snapshot.sha256 != expected_receipt_sha256:
                raise ChatFiveExpertError(
                    "chat_training_phase_receipt_snapshot_mismatch"
                )
            expected_sidecar = f"{expected_receipt_sha256}  {receipt_name}\n".encode(
                "ascii"
            )
            sidecar_path = receipt_path.with_name(receipt_name + ".sha256")
            sidecar_snapshot = snapshot_io._read_snapshot(
                sidecar_path,
                max_bytes=1024,
            )
            if sidecar_snapshot.data != expected_sidecar:
                raise ChatFiveExpertError("chat_training_phase_receipt_sidecar_invalid")
            try:
                receipt = snapshot_io._strict_json(
                    receipt_snapshot.data,
                    f"chat {role} {phase} receipt",
                )
            except ConfigError:
                raise ChatFiveExpertError(
                    "chat_training_phase_receipt_json_invalid"
                ) from None
            if (
                receipt.get("schema_version") != PHASE_RECEIPT_VERSION
                or receipt.get("status") != "passed_diagnostic_training_only"
                or receipt.get("run_id") != run_id
                or receipt.get("role") != role
                or receipt.get("phase") != phase
                or receipt.get("optimizer_steps") != steps
            ):
                raise ChatFiveExpertError(
                    "chat_training_phase_receipt_cross_binding_invalid"
                )
            observed_versions = _validate_runtime_package_versions(
                receipt.get("runtime_package_versions")
            )
            if runtime_versions is None:
                runtime_versions = observed_versions
            elif observed_versions != runtime_versions:
                raise ChatFiveExpertError(
                    "chat_runtime_package_versions_changed_between_phases"
                )
            if phase == "full":
                if receipt.get("adapter_artifact_sha256") != dict(adapter_hashes):
                    raise ChatFiveExpertError(
                        "chat_training_adapter_receipt_cross_binding_invalid"
                    )
            elif receipt.get("adapter_artifact_sha256") is not None:
                raise ChatFiveExpertError("chat_smoke_adapter_artifact_must_be_absent")
            relative = f"{role}/{receipt_name}"
            snapshots[relative] = ArtifactSnapshot(
                path=receipt_snapshot.path,
                sha256=receipt_snapshot.sha256,
                identity=receipt_snapshot.identity,
            )
            sidecar_relative = f"{relative}.sha256"
            snapshots[sidecar_relative] = ArtifactSnapshot(
                path=sidecar_snapshot.path,
                sha256=sidecar_snapshot.sha256,
                identity=sidecar_snapshot.identity,
            )
        adapter_root = role_root / "adapter"
        _assert_directory_inventory(
            adapter_root,
            expected_files=set(ADAPTER_FILES),
            expected_directories=set(),
            code="chat_training_adapter_file_inventory_invalid",
        )
        for filename in ADAPTER_FILES:
            expected = _identity(
                adapter_hashes.get(filename),
                "chat_training_adapter_file_identity_invalid",
                pending_allowed=False,
            )
            relative = f"{role}/adapter/{filename}"
            snapshots[relative] = _capture_artifact_snapshot(
                adapter_root / filename,
                expected_sha256=expected,
            )
    if runtime_versions is None or len(snapshots) != 30:
        raise ChatFiveExpertError("chat_training_artifact_snapshot_count_invalid")
    return snapshots, runtime_versions


def _capture_run_receipt_pair(
    path: Path,
    *,
    expected_sha256: str,
    expected_value: Mapping[str, Any],
) -> dict[str, ArtifactSnapshot]:
    raw = _canonical_json(expected_value) + b"\n"
    if _sha256(raw) != expected_sha256:
        raise ChatFiveExpertError("chat_run_receipt_expected_identity_invalid")
    receipt = snapshot_io._read_snapshot(path, max_bytes=_MAX_RECEIPT_BYTES)
    sidecar = snapshot_io._read_snapshot(
        path.with_name(path.name + ".sha256"),
        max_bytes=1024,
    )
    if (
        receipt.sha256 != expected_sha256
        or receipt.data != raw
        or sidecar.data != f"{expected_sha256}  {path.name}\n".encode("ascii")
    ):
        raise ChatFiveExpertError("chat_run_receipt_snapshot_mismatch")
    return {
        path.name: ArtifactSnapshot(
            path=receipt.path,
            sha256=receipt.sha256,
            identity=receipt.identity,
        ),
        f"{path.name}.sha256": ArtifactSnapshot(
            path=sidecar.path,
            sha256=sidecar.sha256,
            identity=sidecar.identity,
        ),
    }


def _code_snapshots() -> dict[str, snapshot_io.FileSnapshot]:
    root = qdiag._project_root_from_module()
    paths = {
        "implementation": IMPLEMENTATION_PATH,
        "runner_script": SCRIPT_PATH,
        "launcher": LAUNCHER_PATH,
        "config": CONFIG_PATH,
        **{
            f"dependency:{relative}": relative
            for relative in EXECUTION_DEPENDENCY_PATHS
        },
    }
    return {
        label: snapshot_io._read_snapshot(
            qdiag._assert_physical_path(
                root.joinpath(*PurePosixPath(relative).parts),
                require_file=True,
                label=relative,
            ),
            max_bytes=_MAX_CONFIG_BYTES,
        )
        for label, relative in paths.items()
    }


def _failed_staging_reference(path: Path) -> str:
    try:
        relative = path.relative_to(qdiag._project_root_from_module())
    except ValueError:
        # Test doubles can redirect the output root.  Never let audit-path
        # formatting mask the original execution failure.
        return path.name
    return str(relative).replace("\\", "/")


def execute(
    config: Mapping[str, Any],
    *,
    run_id: str,
    phase_executor: Any = None,
) -> dict[str, Any]:
    """Execute only after final producer identities are authenticated."""

    run_id = _validated_run_id(run_id)
    current_role: str | None = None
    current_phase: str | None = None
    staging: Path | None = None
    published_destination: Path | None = None
    failed_staging: str | None = None
    try:
        validate_config(config)
        pending = pending_producer_identities(config)
        if pending:
            # This is deliberately before _repo_path, dataset reads, lock
            # acquisition, tokenizer/model imports, or CUDA inspection.
            raise ChatFiveExpertError(
                "chat_producer_identity_pending:" + ",".join(pending)
            )
        authenticated = _load_authenticated_dataset(config)
        code_snapshots = _code_snapshots()
        if code_snapshots["config"].sha256 != config["_config_sha256"]:
            raise ChatFiveExpertError("chat_config_changed_before_execute")
        if (
            code_snapshots[f"dependency:{TOKENIZER_POLICY_PATH}"].sha256
            != config["model"]["tokenizer_policy_sha256"]
        ):
            raise ChatFiveExpertError("chat_tokenizer_policy_identity_mismatch")
        output_root = _output_path(config["output"]["artifact_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        destination = output_root / run_id
        if os.path.lexists(destination):
            raise FileExistsError(f"chat run output exists: {destination}")

        with _canonical_gpu_lock(config, run_id=run_id) as lock_owner:
            staging = output_root / f".{run_id}.tmp-{uuid.uuid4().hex}"
            staging.mkdir(exist_ok=False)
            with _private_model_snapshot(config, run_id) as (
                model_snapshot,
                model_hashes,
            ):
                processor = gemma_binding.load_sentencepiece(
                    model_snapshot / "tokenizer.model"
                )
                datasets = _serialize_all_role_datasets(
                    processor,
                    authenticated,
                )
                roles = _run_serial_phases(
                    config,
                    datasets=datasets,
                    model_snapshot=model_snapshot,
                    staging=staging,
                    run_id=run_id,
                    phase_executor=phase_executor,
                )
                _remove_training_progress_for_publish(staging)
                _assert_success_root_inventory(
                    staging,
                    run_receipt_name=str(config["output"]["run_receipt_name"]),
                    receipt_present=False,
                )
                training_artifacts, runtime_package_versions = (
                    _capture_training_artifacts(
                        staging,
                        roles=roles,
                        run_id=run_id,
                    )
                )
                for snapshot in code_snapshots.values():
                    snapshot.assert_unchanged()
                for snapshot in authenticated.source_snapshots:
                    snapshot.assert_unchanged()
                success = {
                    "schema_version": RUN_RECEIPT_VERSION,
                    "status": "passed_diagnostic_training_only",
                    "run_id": run_id,
                    "identity": {
                        "config_sha256": config["_config_sha256"],
                        "implementation_sha256": code_snapshots[
                            "implementation"
                        ].sha256,
                        "runner_script_sha256": code_snapshots["runner_script"].sha256,
                        "launcher_sha256": code_snapshots["launcher"].sha256,
                        "execution_dependency_sha256": {
                            relative: code_snapshots[f"dependency:{relative}"].sha256
                            for relative in EXECUTION_DEPENDENCY_PATHS
                        },
                        "producer_config_sha256": config["producer"]["config_sha256"],
                        "producer_implementation_sha256": config["producer"][
                            "implementation_sha256"
                        ],
                        "dataset_manifest_sha256": (authenticated.manifest_sha256),
                        "partition_sha256": dict(authenticated.partition_sha256),
                        "private_model_snapshot_file_sha256": dict(model_hashes),
                    },
                    "gpu_lock": dict(lock_owner),
                    "roles": roles,
                    "execution": {
                        "role_order": list(ROLES),
                        "concurrency": 1,
                        "fresh_base_objects": 10,
                        "fresh_adapters": 10,
                        "smoke_steps_per_role": 2,
                        "full_steps_per_role": 160,
                        "smoke_checkpoint_consumed_by_full": False,
                        "resume": False,
                        "sequence_length": 768,
                        "truncation": False,
                        "micro_batch_size": 1,
                        "gradient_accumulation_steps": 1,
                        "optimizer": dict(OPTIMIZER_RUNTIME_CONTRACT),
                        "base_quantization": dict(Q8_BASE_CONTRACT),
                        "runtime_package_versions": runtime_package_versions,
                        "training_use_cache": False,
                        "memory_gate": {
                            "comparison": "strict_less_than",
                            "limit_mib": 10240,
                            "torch_allocated": True,
                            "torch_reserved": True,
                            "nvidia_smi_sampled_physical_used": True,
                        },
                    },
                    "claims": {
                        "diagnostic_only": True,
                        "dataset_artifact_name_is_not_authority": True,
                        "training_executed": True,
                        "training_authorized": False,
                        "formal_training_authorized": False,
                        "formal": False,
                        "live_authorized": False,
                        "quality_claimed": False,
                        "eval_proxy_is_heldout": False,
                    },
                    "audit": {
                        "provider_requests": 0,
                        "network_requests": 0,
                        "model_object_loads": 10,
                        "gpu_concurrency": 1,
                        "protected_body_reads": 0,
                        "dataset_body_reads": 2,
                    },
                }
                run_receipt_path = staging / str(config["output"]["run_receipt_name"])
                run_receipt_sha256 = _atomic_write_json(
                    run_receipt_path,
                    success,
                )
                run_receipt_snapshots = _capture_run_receipt_pair(
                    run_receipt_path,
                    expected_sha256=run_receipt_sha256,
                    expected_value=success,
                )
                _assert_success_root_inventory(
                    staging,
                    run_receipt_name=str(config["output"]["run_receipt_name"]),
                    receipt_present=True,
                )
            for snapshot in code_snapshots.values():
                snapshot.assert_unchanged()
            for snapshot in authenticated.source_snapshots:
                snapshot.assert_unchanged()
            for snapshot in (
                *training_artifacts.values(),
                *run_receipt_snapshots.values(),
            ):
                snapshot.assert_unchanged()
            _assert_success_root_inventory(
                staging,
                run_receipt_name=str(config["output"]["run_receipt_name"]),
                receipt_present=True,
            )
            qdiag._rename_directory_noreplace(staging, destination)
            published_destination = destination
            staging = None
            _assert_success_root_inventory(
                destination,
                run_receipt_name=str(config["output"]["run_receipt_name"]),
                receipt_present=True,
            )
            published_run_receipt_snapshots = _capture_run_receipt_pair(
                destination / str(config["output"]["run_receipt_name"]),
                expected_sha256=run_receipt_sha256,
                expected_value=success,
            )
            published_training_artifacts, published_runtime_versions = (
                _capture_training_artifacts(
                    destination,
                    roles=roles,
                    run_id=run_id,
                )
            )
            if published_runtime_versions != runtime_package_versions:
                raise ChatFiveExpertError(
                    "chat_published_runtime_package_versions_changed"
                )
            for snapshot in code_snapshots.values():
                snapshot.assert_unchanged()
            for snapshot in authenticated.source_snapshots:
                snapshot.assert_unchanged()
            for snapshot in (
                *published_training_artifacts.values(),
                *published_run_receipt_snapshots.values(),
            ):
                snapshot.assert_unchanged()
            _assert_success_root_inventory(
                destination,
                run_receipt_name=str(config["output"]["run_receipt_name"]),
                receipt_present=True,
            )
            return success
    except Exception as error:
        if isinstance(error, ChatPhaseError):
            current_role = error.role
            current_phase = error.phase
        if staging is not None and os.path.lexists(staging):
            failed = staging.with_name(f".failed-{run_id}-{uuid.uuid4().hex}")
            os.rename(staging, failed)
            failed_staging = _failed_staging_reference(failed)
        elif published_destination is not None and os.path.lexists(
            published_destination
        ):
            failed = published_destination.with_name(
                f".failed-{run_id}-{uuid.uuid4().hex}"
            )
            os.rename(published_destination, failed)
            failed_staging = _failed_staging_reference(failed)
        code = (
            str(error)
            if isinstance(
                error,
                (
                    ChatFiveExpertError,
                    ConfigError,
                    q8_runner.GemmaFiveRoleError,
                ),
            )
            else type(error).__name__
        )
        try:
            _publish_failure_receipt(
                config,
                run_id=run_id,
                error_code=code,
                role=current_role,
                phase=current_phase,
                failed_staging=failed_staging,
            )
        except Exception:
            # Failure-receipt problems must not hide the original training
            # exception. A no-replace collision is itself evidence.
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed Gemma 3 chat five-expert rank-1024 Q8/BF16 trainer. "
            "Preflight and dry-run are model/GPU/provider/network free. "
            "Execution is explicit and remains blocked until every final "
            "producer identity is bound."
        )
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Authenticate the dataset only when all producer hashes are bound; "
            "otherwise publish a blocked receipt without reading dataset bodies."
        ),
    )
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help=("Run the same model-free preflight without publishing a receipt."),
    )
    modes.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run the diagnostic trainer. Requires --run-id, all final producer "
            "identities, a bound GPU UUID, and the canonical GPU lock."
        ),
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.preflight or args.dry_run:
            result = build_preflight(config)
            payload = {
                "status": result["status"],
                "pending_producer_identities": result["pending_producer_identities"],
                "model_loaded": False,
                "gpu_requested": False,
                "training_executed": False,
            }
            if args.preflight:
                receipt = publish_receipt(
                    config,
                    mode="preflight",
                    value=result,
                    run_id=args.run_id,
                )
                payload["receipt"] = str(
                    receipt.relative_to(qdiag._project_root_from_module())
                ).replace("\\", "/")
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return (
                0
                if result["status"]
                == (
                    "passed_model_free_authenticated_dataset_ready_for_explicit_execute"
                )
                else 2
            )
        if not args.run_id:
            raise ChatFiveExpertError("chat_execute_requires_explicit_run_id")
        result = execute(
            config,
            run_id=args.run_id,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "run_id": result["run_id"],
                    "artifact": (
                        f"{config['output']['artifact_root']}/{result['run_id']}"
                    ),
                    "training_executed": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (
        ConfigError,
        ChatFiveExpertError,
        q8_runner.GemmaFiveRoleError,
        FileExistsError,
        OSError,
        RuntimeError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error_code": str(exc),
                    "sample_content_included": False,
                    "automatic_retry": False,
                    "training_executed": False,
                    "claims": {
                        "training_authorized": False,
                        "formal_training_authorized": False,
                        "formal": False,
                        "live_authorized": False,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "CONFIG_VERSION",
    "DATASET_ROOT",
    "FAILURE_RECEIPT_VERSION",
    "IMPLEMENTATION_PATH",
    "MANIFEST_VERSION",
    "PHASE_RECEIPT_VERSION",
    "PREFLIGHT_VERSION",
    "RECORD_VERSION",
    "ROLES",
    "RUN_RECEIPT_VERSION",
    "ChatExample",
    "ChatFiveExpertError",
    "RoleDataset",
    "SerializedChatExample",
    "SerializedRoleDataset",
    "build_preflight",
    "build_serial_plan",
    "execute",
    "load_config",
    "load_role_dataset",
    "main",
    "pending_producer_identities",
    "validate_config",
]
