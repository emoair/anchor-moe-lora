"""Strict Gemma 3 1B IT five-role Q-only controlled-proxy runner.

The default path is model-free.  GPU execution is available only behind an
explicit flag plus an authenticated outer launcher lease and GPU attestation.
The five roles are trained serially.  Each role runs a two-step smoke phase,
destroys that model and optimizer, and then creates a fresh base object and a
fresh adapter for the 160-step phase.  Smoke state is never consumed by the
full phase.

This runner deliberately does not claim formal training authorization or
physical private-tail KV materialization.  It machine-binds the intended
runtime boundary: an adapter-off immutable prefix, followed by a Q-only
expert-private append-only tail.  Cross-expert private-tail reuse is forbidden;
only committed text may be re-encoded for the next stage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from anchor_mvp.research import gemma3_qonly_parameter_budget as budget
from anchor_mvp.training import gemma3_tokenizer_binding_v1 as binding
from anchor_mvp.training import qwen_lora_diagnostic as qdiag
from anchor_mvp.training.config import ConfigError, _expand_env, _read_mapping


CONFIG_VERSION = "anchor.gemma3-1b-it-five-role-qonly-runner-config.v1"
PREFLIGHT_VERSION = "anchor.gemma3-1b-it-five-role-qonly-preflight.v1"
PHASE_RECEIPT_VERSION = "anchor.gemma3-1b-it-five-role-qonly-phase-receipt.v1"
RUN_RECEIPT_VERSION = "anchor.gemma3-1b-it-five-role-qonly-run-receipt.v1"
FAILURE_RECEIPT_VERSION = "anchor.gemma3-1b-it-five-role-qonly-failure-receipt.v1"
CONFIG_PATH = "configs/training/gemma3_1b_it_five_role_qonly_v1.yaml"
IMPLEMENTATION_PATH = "src/anchor_mvp/training/gemma3_five_role_qonly_v1.py"
SCRIPT_PATH = "scripts/research/run_gemma3_1b_it_five_role_qonly_v1.py"
LAUNCHER_PATH = "scripts/research/run_gemma3_1b_it_five_role_qonly_v1.ps1"

ROLES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
WDDM_GUI_PROCESS_ALLOWLIST = (
    "applicationframehost.exe",
    "chatgpt.exe",
    "codex.exe",
    "dwm.exe",
    "explorer.exe",
    "flclash.exe",
    "gamebar.exe",
    "gamebarftserver.exe",
    "gameviewerserver.exe",
    "lockapp.exe",
    "msedgewebview2.exe",
    "nvidia broadcast.exe",
    "nvidia overlay.exe",
    "promecefpluginhost.exe",
    "searchhost.exe",
    "shellexperiencehost.exe",
    "shellhost.exe",
    "startmenuexperiencehost.exe",
    "systemsettings.exe",
    "systemsettingsbroker.exe",
    "tabtip.exe",
    "taskmgr.exe",
    "textinputhost.exe",
    "wechat.exe",
    "wechatappex.exe",
    "widgets.exe",
    "widgetservice.exe",
    "wps.exe",
    "wpscenter.exe",
    "wpscloudsvr.exe",
)
OPTIMIZER_RUNTIME_CONTRACT = {
    "backend": "bitsandbytes",
    "class": "bitsandbytes.optim.AdamW8bit",
    "package_version": "0.48.2",
    "optimizer_state_bits": 8,
    "compatibility_optim_bits_argument": 32,
    "min_8bit_size": 4096,
    "percentile_clipping": 100,
    "block_wise": True,
    "is_paged": False,
    "amsgrad": False,
}
PYTHON_RUNTIME_DEPENDENCY_PROBE = (
    "yaml,sentencepiece,torch,transformers,peft,bitsandbytes==0.48.2"
)
KV_RUNTIME_BOUNDARY = {
    "shared_prefix_adapter_state": "off",
    "shared_prefix_read_only": True,
    "identical_ordered_prefix_lineage_only": True,
    "expert_activation": "q_only",
    "expert_private_tail_append_only": True,
    "private_tail_includes_post_activation_prompt_and_generated_tokens": True,
    "private_tail_cross_expert_reuse": False,
    "committed_text_reencoded_for_next_shared_context": True,
    "full_generation_kv_shared_claimed": False,
    "normal_in_stack_q_lora_exact_kv_sharing_claimed": False,
    "token_level_moe_claimed": False,
    "runtime_private_tail_materialized": False,
}
MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "EXPORT_MANIFEST.json",
)
ADAPTER_BASE_IDENTITY = (
    "anchor.local/gemma3-1b-it-keras-v3-bf16@sha256:"
    "c9c6e309cf0158050d1e1abcba19eb6798153468572af2cd91de163e74933df9"
)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_PYTHON_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_LORA_TENSOR_RE = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.q_proj\."
    r"lora_([AB])(?:\.[^.]+)?\.weight$"
)
_MAX_CONFIG_BYTES = 2_000_000
_MAX_RECEIPT_BYTES = 8_000_000
_MIB = 1024 * 1024


class GemmaFiveRoleError(RuntimeError):
    """Raised when a controlled-proxy invariant fails closed."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    stat_signature: tuple[int, int, int, int, int]

    def assert_unchanged(self) -> None:
        stat = self.path.stat()
        signature = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        if signature != self.stat_signature:
            raise GemmaFiveRoleError("authenticated_file_changed_after_snapshot")


def _root() -> Path:
    return qdiag._project_root_from_module()


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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: object, code: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ConfigError(code)
    return value


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise ConfigError(code)


def _repo_path(value: object, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError("gemma_runner_repo_path_invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError("gemma_runner_repo_path_unsafe")
    return qdiag._assert_physical_path(
        _root().joinpath(*relative.parts),
        require_directory=directory,
        require_file=not directory,
        label=value,
    )


def _output_relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ConfigError("gemma_runner_output_path_invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError("gemma_runner_output_path_unsafe")
    return relative


def _stable_snapshot(path: Path, *, max_bytes: int) -> FileSnapshot:
    path = qdiag._assert_physical_path(path, require_file=True, label=str(path))
    with path.open("rb", buffering=0) as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > max_bytes:
            raise GemmaFiveRoleError("authenticated_file_exceeds_size_limit")
        data = handle.read()
        after = os.fstat(handle.fileno())
    before_signature = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_signature = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_signature != after_signature or len(data) != before.st_size:
        raise GemmaFiveRoleError("authenticated_file_changed_during_snapshot")
    return FileSnapshot(
        path=path,
        data=data,
        sha256=_sha256(data),
        stat_signature=after_signature,
    )


def _stream_sha256(path: Path, *, expected_bytes: int | None = None) -> str:
    path = qdiag._assert_physical_path(path, require_file=True, label=str(path))
    digest = hashlib.sha256()
    total = 0
    with path.open("rb", buffering=0) as handle:
        before = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(8 * _MIB)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    identity_before = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    identity_after = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if identity_before != identity_after or total != before.st_size:
        raise GemmaFiveRoleError("stream_identity_changed_during_hash")
    if expected_bytes is not None and total != expected_bytes:
        raise GemmaFiveRoleError("stream_byte_count_mismatch")
    return digest.hexdigest()


def _strict_json(data: bytes, code: str) -> Mapping[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise GemmaFiveRoleError(code) from None
    if not isinstance(value, Mapping):
        raise GemmaFiveRoleError(code)
    return value


def _config_path(path: str | Path) -> Path:
    canonical = _root().joinpath(*PurePosixPath(CONFIG_PATH).parts)
    requested = Path(path)
    if not requested.is_absolute():
        requested = _root().joinpath(*PurePosixPath(requested.as_posix()).parts)
    requested = Path(os.path.abspath(requested))
    if os.path.normcase(str(requested)) != os.path.normcase(
        str(Path(os.path.abspath(canonical)))
    ):
        raise ConfigError("gemma_runner_config_path_must_be_canonical")
    return qdiag._assert_physical_path(
        requested, require_file=True, label="Gemma five-role runner config"
    )


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    canonical = _config_path(path)
    snapshot = _stable_snapshot(canonical, max_bytes=_MAX_CONFIG_BYTES)
    try:
        raw = _read_mapping(canonical)
    except (OSError, ConfigError, UnicodeError):
        raise ConfigError("gemma_runner_config_parse_failed") from None
    config = _expand_env(raw)
    if not isinstance(config, dict):
        raise ConfigError("gemma_runner_config_mapping_required")
    config["_config_path"] = str(canonical)
    config["_config_sha256"] = snapshot.sha256
    validate_config(config)
    snapshot.assert_unchanged()
    return config


def _validate_binding_paths_and_hashes(section: Mapping[str, Any]) -> None:
    expected = {
        "config": binding.CONFIG_PATH,
        "policy": binding.POLICY_PATH,
        "config_schema": binding.CONFIG_SCHEMA_PATH,
        "manifest_schema": binding.MANIFEST_SCHEMA_PATH,
        "implementation": binding.IMPLEMENTATION_PATH,
        "artifact": binding.OUTPUT_PATH,
        "manifest": f"{binding.OUTPUT_PATH}/manifest.json",
        "manifest_sidecar": f"{binding.OUTPUT_PATH}/manifest.json.sha256",
    }
    hash_keys = {
        "config": "config_sha256",
        "policy": "policy_sha256",
        "config_schema": "config_schema_sha256",
        "manifest_schema": "manifest_schema_sha256",
        "implementation": "implementation_sha256",
        "manifest": "manifest_sha256",
        "manifest_sidecar": "manifest_sidecar_physical_sha256",
    }
    for key, expected_path in expected.items():
        if section.get(key) != expected_path:
            raise ConfigError("gemma_runner_tokenizer_binding_path_drift")
        if key == "artifact":
            _repo_path(section[key], directory=True)
            continue
        _require_sha(
            section.get(hash_keys[key]),
            "gemma_runner_tokenizer_binding_sha_invalid",
        )
    _require_sha(
        section.get("tokenizer_template_special_policy_sha256"),
        "gemma_runner_combined_tokenizer_identity_invalid",
    )


def _validate_parameter_budget(section: Mapping[str, Any]) -> None:
    expected_paths = {
        "contract": budget.CONTRACT_RELATIVE,
        "contract_sidecar": budget.SIDECAR_RELATIVE,
        "schema": budget.SCHEMA_RELATIVE,
        "implementation": "src/anchor_mvp/research/gemma3_qonly_parameter_budget.py",
    }
    hash_keys = {
        "contract": "contract_sha256",
        "contract_sidecar": "contract_sidecar_physical_sha256",
        "schema": "schema_sha256",
        "implementation": "implementation_sha256",
    }
    for key, expected in expected_paths.items():
        if section.get(key) != expected:
            raise ConfigError("gemma_runner_parameter_budget_path_drift")
        _require_sha(
            section.get(hash_keys[key]), "gemma_runner_parameter_budget_sha_invalid"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "claim_scope",
            "bindings",
            "model",
            "dataset",
            "lora",
            "training",
            "kv_runtime_boundary",
            "gpu_policy",
            "output",
            "claims",
            "_config_path",
            "_config_sha256",
        },
        "gemma_runner_top_level_fields_drift",
    )
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope") != "controlled_proxy_diagnostic_only_not_formal"
    ):
        raise ConfigError("gemma_runner_config_identity_drift")
    bindings = _mapping(config.get("bindings"), "gemma_runner_bindings_invalid")
    _exact_keys(
        bindings,
        {"tokenizer_binding", "parameter_budget"},
        "gemma_runner_bindings_fields_drift",
    )
    _validate_binding_paths_and_hashes(
        _mapping(
            bindings.get("tokenizer_binding"),
            "gemma_runner_tokenizer_binding_invalid",
        )
    )
    _validate_parameter_budget(
        _mapping(
            bindings.get("parameter_budget"),
            "gemma_runner_parameter_budget_invalid",
        )
    )

    model = _mapping(config.get("model"), "gemma_runner_model_invalid")
    if (
        model.get("architecture") != "Gemma3ForCausalLM"
        or model.get("model_type") != "gemma3_text"
        or model.get("parameter_count") != 999_885_952
        or model.get("adapter_base_identity") != ADAPTER_BASE_IDENTITY
        or model.get("layers") != 26
        or model.get("hidden_size") != 1152
        or model.get("q_proj_input_features") != 1152
        or model.get("q_proj_output_features") != 1024
    ):
        raise ConfigError("gemma_runner_model_architecture_drift")
    model_files = tuple(
        str(_mapping(item, "gemma_runner_model_file_invalid").get("path"))
        for item in _sequence(model.get("files"), "gemma_runner_model_files_invalid")
    )
    if model_files != MODEL_FILES:
        raise ConfigError("gemma_runner_model_file_order_drift")
    for item in _sequence(model["files"], "gemma_runner_model_files_invalid"):
        entry = _mapping(item, "gemma_runner_model_file_invalid")
        if (
            set(entry) != {"path", "bytes", "sha256"}
            or not isinstance(entry["bytes"], int)
            or isinstance(entry["bytes"], bool)
            or entry["bytes"] <= 0
        ):
            raise ConfigError("gemma_runner_model_file_fields_invalid")
        _require_sha(entry["sha256"], "gemma_runner_model_file_sha_invalid")
    overlay = _mapping(
        model.get("runtime_special_token_overlay"),
        "gemma_runner_special_overlay_invalid",
    )
    if overlay != {
        "canonical_files_modified": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
        "unk_token_id": 3,
    }:
        raise ConfigError("gemma_runner_special_overlay_drift")

    dataset = _mapping(config.get("dataset"), "gemma_runner_dataset_invalid")
    if (
        dataset.get("records") != 1000
        or dataset.get("train_records") != 800
        or dataset.get("eval_proxy_records") != 200
        or dataset.get("task_bundles") != 200
        or dataset.get("train_records_per_role") != 160
        or dataset.get("eval_proxy_records_per_role") != 40
        or tuple(dataset.get("roles", ())) != ROLES
        or dataset.get("role_execution") != "strictly_serial"
        or dataset.get("concurrency") != 1
        or dataset.get("eval_proxy_is_heldout") is not False
    ):
        raise ConfigError("gemma_runner_dataset_contract_drift")

    lora = _mapping(config.get("lora"), "gemma_runner_lora_invalid")
    if lora != {
        "profile": "q_only",
        "target_modules": ["q_proj"],
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "bias": "none",
        "expected_trainable_parameters_per_expert": 226_304,
        "o_proj_allowed": False,
    }:
        raise ConfigError("gemma_runner_q_only_contract_drift")

    training = _mapping(config.get("training"), "gemma_runner_training_invalid")
    expected_training = {
        "sequence_length": 768,
        "truncation": False,
        "padding": "microbatch_exact_length",
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "smoke_steps": 2,
        "full_steps": 160,
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
        "learning_rate": 0.00002,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 0.00000001,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 1337,
        "compute_dtype": "bfloat16",
        "tf32": True,
        "attention_implementation": "sdpa",
        "gradient_checkpointing": True,
        "use_cache": False,
        "save_intermediate_checkpoints": False,
        "adapter_effect_gate": {
            "view": "first_train_record_first_supervised_next_token_v1",
            "comparison": "enabled_vs_disable_adapter_after_training",
            "require_finite": True,
            "require_max_abs_gt_zero": True,
            "sample_body_included": False,
            "token_ids_included": False,
        },
    }
    if dict(training) != expected_training:
        raise ConfigError("gemma_runner_training_contract_drift")

    kv = _mapping(config.get("kv_runtime_boundary"), "gemma_runner_kv_boundary_invalid")
    if kv != KV_RUNTIME_BOUNDARY:
        raise ConfigError("gemma_runner_private_tail_contract_drift")

    gpu = _mapping(config.get("gpu_policy"), "gemma_runner_gpu_policy_invalid")
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
        or gpu.get("sample_count") != 3
        or gpu.get("sample_interval_seconds") != 1
        or gpu.get("command_timeout_seconds") != 5
        or gpu.get("idle_used_memory_max_mib") != 2048
        or gpu.get("idle_free_memory_min_mib") != 8192
        or gpu.get("idle_utilization_max_percent") != 15
        or gpu.get("prestart_temperature_max_c") != 75
        or gpu.get("runtime_temperature_max_c") != 83
        or gpu.get("torch_peak_allocated_max_mib") != 23962
        or gpu.get("torch_peak_reserved_max_mib") != 23962
        or gpu.get("runtime_monitor_interval_steps") != 10
        or tuple(gpu.get("wddm_gui_process_allowlist", ()))
        != WDDM_GUI_PROCESS_ALLOWLIST
        or gpu.get("wddm_gui_inventory_must_be_stable_across_gate") is not True
        or gpu.get("insufficient_permissions_pid_resolution_required") is not True
        or gpu.get("unknown_or_non_allowlisted_compute_process_forbidden") is not False
    ):
        raise ConfigError("gemma_runner_gpu_policy_drift")

    output = _mapping(config.get("output"), "gemma_runner_output_invalid")
    if output != {
        "artifact_root": "artifacts/diagnostics/gemma3_1b_it_five_role_qonly_v1",
        "launch_root": "runs/gemma3_1b_it_five_role_qonly_v1",
        "atomic_publish": True,
        "replace_existing": False,
        "failure_receipt_retained": True,
        "adapter_format": "peft_safetensors",
    }:
        raise ConfigError("gemma_runner_output_contract_drift")
    _output_relative(output["artifact_root"])
    _output_relative(output["launch_root"])

    claims = _mapping(config.get("claims"), "gemma_runner_claims_invalid")
    if claims != {
        "diagnostic_only": True,
        "controlled_proxy_only": True,
        "execution_requires_explicit_flag_and_gpu_lease": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
        "runtime_private_tail_materialized": False,
    }:
        raise ConfigError("gemma_runner_claims_drift")


def _binding_config(config: Mapping[str, Any]) -> dict[str, Any]:
    section = _mapping(
        _mapping(config["bindings"], "gemma_runner_bindings_invalid")[
            "tokenizer_binding"
        ],
        "gemma_runner_tokenizer_binding_invalid",
    )
    path = _repo_path(section["config"])
    value = _expand_env(_read_mapping(path))
    value["_config_path"] = str(path)
    binding.validate_config(value)
    return value


def _authenticate_bound_files(config: Mapping[str, Any]) -> dict[str, str]:
    identities: dict[str, str] = {}
    bindings = _mapping(config["bindings"], "gemma_runner_bindings_invalid")
    tokenizer = _mapping(
        bindings["tokenizer_binding"], "gemma_runner_tokenizer_binding_invalid"
    )
    tokenizer_pairs = {
        "binding_config": ("config", "config_sha256"),
        "binding_policy": ("policy", "policy_sha256"),
        "binding_config_schema": ("config_schema", "config_schema_sha256"),
        "binding_manifest_schema": ("manifest_schema", "manifest_schema_sha256"),
        "binding_implementation": ("implementation", "implementation_sha256"),
        "binding_manifest": ("manifest", "manifest_sha256"),
        "binding_manifest_sidecar": (
            "manifest_sidecar",
            "manifest_sidecar_physical_sha256",
        ),
    }
    snapshots: list[FileSnapshot] = []
    for label, (path_key, sha_key) in tokenizer_pairs.items():
        snapshot = _stable_snapshot(
            _repo_path(tokenizer[path_key]), max_bytes=_MAX_RECEIPT_BYTES
        )
        if snapshot.sha256 != tokenizer[sha_key]:
            raise GemmaFiveRoleError(f"{label}_physical_sha256_mismatch")
        identities[label] = snapshot.sha256
        snapshots.append(snapshot)
    manifest_snapshot = snapshots[-2]
    sidecar_snapshot = snapshots[-1]
    if sidecar_snapshot.data != (
        f"{manifest_snapshot.sha256}  manifest.json\n".encode("ascii")
    ):
        raise GemmaFiveRoleError("binding_manifest_sidecar_content_invalid")

    parameter_budget = _mapping(
        bindings["parameter_budget"], "gemma_runner_parameter_budget_invalid"
    )
    budget_pairs = {
        "budget_contract": ("contract", "contract_sha256"),
        "budget_sidecar": (
            "contract_sidecar",
            "contract_sidecar_physical_sha256",
        ),
        "budget_schema": ("schema", "schema_sha256"),
        "budget_implementation": ("implementation", "implementation_sha256"),
    }
    for label, (path_key, sha_key) in budget_pairs.items():
        snapshot = _stable_snapshot(
            _repo_path(parameter_budget[path_key]), max_bytes=_MAX_RECEIPT_BYTES
        )
        if snapshot.sha256 != parameter_budget[sha_key]:
            raise GemmaFiveRoleError(f"{label}_physical_sha256_mismatch")
        identities[label] = snapshot.sha256
        snapshots.append(snapshot)
    for snapshot in snapshots:
        snapshot.assert_unchanged()
    return identities


def build_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    identities = _authenticate_bound_files(config)
    binding_section = _mapping(
        _mapping(config["bindings"], "gemma_runner_bindings_invalid")[
            "tokenizer_binding"
        ],
        "gemma_runner_tokenizer_binding_invalid",
    )
    binding_manifest = binding.audit_manifest(
        binding_section["artifact"], binding_section["config"]
    )
    if (
        binding_manifest.get("status")
        != "passed_model_free_tokenizer_and_label_preflight_training_blocked"
        or binding_manifest["identities"]["tokenizer_template_special_policy_sha256"]
        != binding_section["tokenizer_template_special_policy_sha256"]
        or binding_manifest["summary"]["records"] != 1000
        or binding_manifest["tokenization"]["sequence_length"] != 768
        or binding_manifest["tokenization"]["observed_maximum_tokens"] != 665
        or binding_manifest["tokenization"]["records_over_sequence_length"] != 0
        or binding_manifest["tokenization"]["truncation_used"] is not False
    ):
        raise GemmaFiveRoleError("binding_manifest_semantic_gate_failed")
    budget_report = budget.audit_contract(_root())
    if (
        budget_report["status"] != "metadata_budget_ready_real_run_blocked"
        or budget_report["base_parameters"] != 999_885_952
        or budget_report["training_authorized"] is not False
    ):
        raise GemmaFiveRoleError("parameter_budget_semantic_gate_failed")
    binding_config = _binding_config(config)
    datasets = binding.load_all_role_datasets(binding_config)
    if tuple(datasets) != ROLES or any(
        len(datasets[role].train) != 160 or len(datasets[role].eval_proxy) != 40
        for role in ROLES
    ):
        raise GemmaFiveRoleError("five_role_dataset_gate_failed")
    implementation = _stable_snapshot(
        _repo_path(IMPLEMENTATION_PATH), max_bytes=_MAX_CONFIG_BYTES
    )
    script = _stable_snapshot(_repo_path(SCRIPT_PATH), max_bytes=_MAX_CONFIG_BYTES)
    launcher_path = _root().joinpath(*PurePosixPath(LAUNCHER_PATH).parts)
    launcher_sha = None
    if launcher_path.exists():
        launcher_sha = _stable_snapshot(
            qdiag._assert_physical_path(
                launcher_path, require_file=True, label=LAUNCHER_PATH
            ),
            max_bytes=_MAX_CONFIG_BYTES,
        ).sha256
    report = {
        "schema_version": PREFLIGHT_VERSION,
        "status": "passed_model_free_ready_for_explicit_controlled_proxy_execute",
        "identity": {
            "config_sha256": config["_config_sha256"],
            "implementation_sha256": implementation.sha256,
            "script_sha256": script.sha256,
            "launcher_sha256": launcher_sha,
            "bound_files": identities,
            "tokenizer_binding_manifest_sha256": binding_section["manifest_sha256"],
            "tokenizer_template_special_policy_sha256": binding_section[
                "tokenizer_template_special_policy_sha256"
            ],
        },
        "dataset": {
            "records": 1000,
            "task_bundles": 200,
            "train_records": 800,
            "eval_proxy_records": 200,
            "role_train_records": {role: 160 for role in ROLES},
            "role_eval_proxy_records": {role: 40 for role in ROLES},
            "eval_proxy_is_heldout": False,
        },
        "tokenization": {
            "sequence_length": 768,
            "observed_minimum_tokens": binding_manifest["tokenization"][
                "observed_minimum_tokens"
            ],
            "observed_maximum_tokens": binding_manifest["tokenization"][
                "observed_maximum_tokens"
            ],
            "records_over_sequence_length": 0,
            "truncation_used": False,
            "ordered_input_ids_sha256": binding_manifest["summary"][
                "ordered_input_ids_sha256"
            ],
            "ordered_labels_sha256": binding_manifest["summary"][
                "ordered_labels_sha256"
            ],
        },
        "execution_plan": {
            "roles": list(ROLES),
            "concurrency": 1,
            "phases_per_role": ["smoke", "full"],
            "steps": {"smoke": 2, "full": 160},
            "fresh_base_objects": 10,
            "fresh_adapters": 10,
            "single_private_authenticated_model_snapshot": True,
            "smoke_checkpoint_consumed_by_full": False,
            "resume": False,
            "optimizer": dict(OPTIMIZER_RUNTIME_CONTRACT),
        },
        "lora": {
            "profile": "q_only",
            "target_modules": ["q_proj"],
            "rank": 4,
            "alpha": 8,
            "per_expert_trainable_parameters": 226_304,
        },
        "kv_runtime_boundary": dict(config["kv_runtime_boundary"]),
        "claims": {
            "diagnostic_only": True,
            "model_loaded": False,
            "gpu_requested": False,
            "training_executed": False,
            "runtime_private_tail_materialized": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
        },
        "audit": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "protected_body_reads": 0,
        },
    }
    implementation.assert_unchanged()
    script.assert_unchanged()
    return report


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    if os.path.lexists(path):
        raise FileExistsError(f"receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json(value)
    digest = _sha256(data)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.tmp-{uuid.uuid4().hex}")
    if os.path.lexists(sidecar):
        raise FileExistsError(f"receipt sidecar already exists: {sidecar}")
    try:
        with temporary.open("xb", buffering=0) as handle:
            handle.write(data)
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
    observed = _stable_snapshot(path, max_bytes=_MAX_RECEIPT_BYTES)
    observed_sidecar = _stable_snapshot(sidecar, max_bytes=1024)
    if observed.sha256 != digest or observed_sidecar.data != (
        f"{digest}  {path.name}\n".encode("ascii")
    ):
        raise GemmaFiveRoleError("atomic_receipt_verification_failed")
    return digest


def _load_authenticated_receipt(
    path: str | Path, expected_sha256: str, *, label: str
) -> tuple[Mapping[str, Any], FileSnapshot]:
    expected_sha256 = _require_sha(
        expected_sha256, f"gemma_runner_{label}_expected_sha_invalid"
    )
    requested = Path(path)
    if not requested.is_absolute():
        requested = _root() / requested
    snapshot = _stable_snapshot(requested, max_bytes=_MAX_RECEIPT_BYTES)
    if snapshot.sha256 != expected_sha256:
        raise GemmaFiveRoleError(f"{label}_sha256_mismatch")
    return _strict_json(snapshot.data, f"{label}_json_invalid"), snapshot


def _assert_launcher_lock_guard(
    config: Mapping[str, Any], lease: Mapping[str, Any]
) -> None:
    if lease.get("launcher_pid") != os.getppid():
        raise GemmaFiveRoleError("launcher_parent_process_not_active")
    gpu = _mapping(config["gpu_policy"], "gemma_runner_gpu_policy_invalid")
    canonical_lock = _root().joinpath(*PurePosixPath(str(gpu["canonical_lock"])).parts)
    if not os.path.lexists(canonical_lock):
        raise GemmaFiveRoleError("canonical_gpu_lock_not_held")
    for conflict in gpu["conflicting_handoff_locks"]:
        if os.path.lexists(_root().joinpath(*PurePosixPath(str(conflict)).parts)):
            raise GemmaFiveRoleError("conflicting_handoff_gpu_lock_present")


def _attested_compute_inventory_sha256(processes: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {
            "pid": process["pid"],
            "process_name": process["process_name"],
        }
        for process in sorted(
            processes,
            key=lambda item: (int(item["pid"]), str(item["process_name"])),
        )
    ]
    return _sha256(_canonical_json(identity))


def _validate_attested_wddm_processes(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise GemmaFiveRoleError("wddm_compute_inventory_invalid")
    result: list[Mapping[str, Any]] = []
    seen_pids: set[int] = set()
    for process in value:
        if not isinstance(process, Mapping):
            raise GemmaFiveRoleError("wddm_compute_inventory_invalid")
        pid = process.get("pid")
        name = process.get("process_name")
        memory = process.get("used_gpu_memory_mib")
        if (
            set(process)
            != {
                "pid",
                "process_name",
                "used_gpu_memory_mib",
                "reported_name_was_permission_denied",
                "allowlisted_wddm_gui",
            }
            or type(pid) is not int
            or pid <= 0
            or pid in seen_pids
            or not isinstance(name, str)
            or name not in WDDM_GUI_PROCESS_ALLOWLIST
            or not isinstance(memory, str)
            or not memory
            or type(process.get("reported_name_was_permission_denied")) is not bool
            or process.get("allowlisted_wddm_gui") is not True
        ):
            raise GemmaFiveRoleError("wddm_compute_inventory_invalid")
        seen_pids.add(pid)
        result.append(process)
    if list(result) != sorted(
        result, key=lambda item: (int(item["pid"]), str(item["process_name"]))
    ):
        raise GemmaFiveRoleError("wddm_compute_inventory_not_canonical")
    return tuple(result)


def _validate_attested_python_runtime(
    lease_value: object,
    attestation_value: object,
) -> Mapping[str, Any]:
    if (
        not isinstance(lease_value, Mapping)
        or not isinstance(attestation_value, Mapping)
        or dict(lease_value) != dict(attestation_value)
        or set(lease_value) != {"path", "version", "sha256", "dependency_probe"}
    ):
        raise GemmaFiveRoleError("python_runtime_cross_binding_failed")
    path = lease_value.get("path")
    version = lease_value.get("version")
    sha256 = lease_value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or not Path(path).is_absolute()
        or not isinstance(version, str)
        or _PYTHON_VERSION_RE.fullmatch(version) is None
        or not isinstance(sha256, str)
        or _SHA256_RE.fullmatch(sha256) is None
        or lease_value.get("dependency_probe") != PYTHON_RUNTIME_DEPENDENCY_PROBE
    ):
        raise GemmaFiveRoleError("python_runtime_identity_invalid")
    return lease_value


def _validate_launch_receipts(
    config: Mapping[str, Any],
    *,
    run_id: str,
    lease_path: str | Path,
    lease_sha256: str,
    gpu_attestation_path: str | Path,
    gpu_attestation_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[FileSnapshot, ...]]:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise GemmaFiveRoleError("run_id_invalid")
    lease, lease_snapshot = _load_authenticated_receipt(
        lease_path, lease_sha256, label="launcher_lease"
    )
    attestation, attestation_snapshot = _load_authenticated_receipt(
        gpu_attestation_path,
        gpu_attestation_sha256,
        label="gpu_attestation",
    )
    gpu = _mapping(config["gpu_policy"], "gemma_runner_gpu_policy_invalid")
    expected_uuid = str(gpu["expected_gpu_uuid"])
    if expected_uuid == "UNBOUND":
        raise GemmaFiveRoleError("execute_requires_bound_gpu_uuid")
    implementation_snapshot = _stable_snapshot(
        _repo_path(IMPLEMENTATION_PATH), max_bytes=_MAX_CONFIG_BYTES
    )
    script_snapshot = _stable_snapshot(
        _repo_path(SCRIPT_PATH), max_bytes=_MAX_CONFIG_BYTES
    )
    launcher_snapshot = _stable_snapshot(
        _repo_path(LAUNCHER_PATH), max_bytes=_MAX_CONFIG_BYTES
    )
    config_snapshot = _stable_snapshot(
        _config_path(CONFIG_PATH), max_bytes=_MAX_CONFIG_BYTES
    )
    if config_snapshot.sha256 != config["_config_sha256"]:
        raise GemmaFiveRoleError("runner_config_changed_before_execute")
    required = {
        "run_id": run_id,
        "canonical_lock": gpu["canonical_lock"],
        "expected_gpu_index": 0,
        "expected_gpu_uuid": expected_uuid,
        "roles": list(ROLES),
        "smoke_steps": 2,
        "full_steps": 160,
        "concurrency": 1,
        "config_sha256": config["_config_sha256"],
        "implementation_sha256": implementation_snapshot.sha256,
        "runner_script_sha256": script_snapshot.sha256,
    }
    for label, value in required.items():
        if lease.get(label) != value or attestation.get(label) != value:
            raise GemmaFiveRoleError("launcher_receipt_cross_binding_failed")
    if (
        lease.get("schema_version")
        != "anchor.gemma3-1b-it-five-role-qonly-execution-lease.v1"
        or attestation.get("schema_version")
        != "anchor.gemma3-1b-it-five-role-qonly-gpu-attestation.v1"
        or lease.get("status") != "passed"
        or attestation.get("status") != "passed"
    ):
        raise GemmaFiveRoleError("launcher_receipt_status_not_passed")
    _validate_attested_python_runtime(
        lease.get("python_runtime"),
        attestation.get("python_runtime"),
    )
    if (
        lease.get("launcher_sha256") != launcher_snapshot.sha256
        or attestation.get("launcher")
        != {
            "path": LAUNCHER_PATH,
            "sha256": launcher_snapshot.sha256,
        }
        or attestation.get("runner")
        != {
            "path": SCRIPT_PATH,
            "sha256": script_snapshot.sha256,
        }
        or attestation.get("implementation")
        != {
            "path": IMPLEMENTATION_PATH,
            "sha256": implementation_snapshot.sha256,
        }
        or attestation.get("config")
        != {
            "path": CONFIG_PATH,
            "sha256": config_snapshot.sha256,
        }
    ):
        raise GemmaFiveRoleError("launcher_code_identity_cross_binding_failed")
    if (
        not isinstance(lease.get("launcher_pid"), int)
        or lease["launcher_pid"] <= 0
        or lease["launcher_pid"] != os.getppid()
        or attestation.get("launcher_pid") != lease["launcher_pid"]
        or attestation.get("lease_receipt_sha256") != lease_snapshot.sha256
        or attestation.get("canonical_lock_sha256")
        != lease.get("canonical_lock_sha256")
    ):
        raise GemmaFiveRoleError("launcher_pid_invalid")
    expected_claims = {
        "diagnostic_only": True,
        "proxy_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "quality_claimed": False,
        "generalization_claimed": False,
    }
    if (
        lease.get("claims") != expected_claims
        or attestation.get("claims") != expected_claims
        or lease.get("kv_runtime_boundary") != KV_RUNTIME_BOUNDARY
        or attestation.get("kv_runtime_boundary") != KV_RUNTIME_BOUNDARY
        or lease.get("canonical_lock_held_at_publish") is not True
        or lease.get("fresh_base_per_phase") is not True
        or lease.get("fresh_adapter_per_phase") is not True
        or lease.get("resume_allowed") is not False
    ):
        raise GemmaFiveRoleError("launcher_non_authorizing_claims_drift")
    expected_allowlist_sha256 = _sha256(
        _canonical_json(list(WDDM_GUI_PROCESS_ALLOWLIST))
    )
    top_level_processes = _validate_attested_wddm_processes(
        attestation.get("compute_processes")
    )
    top_level_inventory_sha256 = _attested_compute_inventory_sha256(top_level_processes)
    if (
        lease.get("compute_processes") != attestation.get("compute_processes")
        or lease.get("compute_inventory_sha256") != top_level_inventory_sha256
        or attestation.get("compute_inventory_sha256") != top_level_inventory_sha256
        or lease.get("wddm_gui_process_allowlist_sha256") != expected_allowlist_sha256
        or attestation.get("wddm_gui_process_allowlist_sha256")
        != expected_allowlist_sha256
    ):
        raise GemmaFiveRoleError("wddm_compute_inventory_cross_binding_failed")
    expected_gpu_policy = {
        "expected_index": 0,
        "expected_total_memory_mib": 12288,
        "sample_count": 3,
        "sample_interval_seconds": 1,
        "command_timeout_seconds": 5,
        "idle_used_memory_max_mib": 2048,
        "idle_free_memory_min_mib": 8192,
        "idle_utilization_max_percent": 15,
        "prestart_temperature_max_c": 75,
        "wddm_gui_process_allowlist": list(WDDM_GUI_PROCESS_ALLOWLIST),
        "wddm_gui_inventory_must_be_stable_across_gate": True,
        "insufficient_permissions_pid_resolution_required": True,
        "unknown_or_non_allowlisted_compute_process_forbidden": False,
    }
    attested_gpu_policy = attestation.get("gpu_policy")
    if (
        not isinstance(attested_gpu_policy, Mapping)
        or dict(attested_gpu_policy) != expected_gpu_policy
        or attestation.get("sample_count") != 3
    ):
        raise GemmaFiveRoleError("gpu_attestation_sample_count_invalid")
    lock_contract = attestation.get("lock")
    execution_plan = attestation.get("execution_plan")
    if (
        not isinstance(lock_contract, Mapping)
        or lock_contract.get("path") != "runs/formal-v3-training.lock"
        or lock_contract.get("content_sha256") != lease.get("canonical_lock_sha256")
        or lock_contract.get("file_mode") != "CreateNew"
        or lock_contract.get("file_share") != "None"
        or lock_contract.get("delete_on_close") is not True
        or lock_contract.get("held_for_entire_orchestrator") is not True
        or not isinstance(execution_plan, Mapping)
        or execution_plan.get("gpu_index") != 0
        or execution_plan.get("concurrency") != 1
        or execution_plan.get("roles") != list(ROLES)
        or execution_plan.get("smoke_steps_per_role") != 2
        or execution_plan.get("full_steps_per_role") != 160
        or execution_plan.get("phase_order") != ["smoke", "full"]
        or execution_plan.get("fresh_base_per_phase") is not True
        or execution_plan.get("fresh_adapter_per_phase") is not True
        or execution_plan.get("resume_allowed") is not False
    ):
        raise GemmaFiveRoleError("launcher_lock_or_execution_plan_drift")
    for key, expected_phase in (
        ("pre_lock_samples", "pre_lock"),
        ("post_lock_samples", "post_lock"),
    ):
        samples = attestation.get(key)
        if not isinstance(samples, list) or len(samples) != 3:
            raise GemmaFiveRoleError("gpu_attestation_samples_invalid")
        for ordinal, sample in enumerate(samples, start=1):
            used = (
                sample.get("memory_used_mib") if isinstance(sample, Mapping) else None
            )
            free = (
                sample.get("memory_free_mib") if isinstance(sample, Mapping) else None
            )
            utilization = (
                sample.get("utilization_percent")
                if isinstance(sample, Mapping)
                else None
            )
            temperature = (
                sample.get("temperature_c") if isinstance(sample, Mapping) else None
            )
            driver_model = (
                sample.get("driver_model") if isinstance(sample, Mapping) else None
            )
            sample_processes = _validate_attested_wddm_processes(
                sample.get("compute_processes") if isinstance(sample, Mapping) else None
            )
            sample_inventory_sha256 = _attested_compute_inventory_sha256(
                sample_processes
            )
            if (
                not isinstance(sample, Mapping)
                or sample.get("phase") != expected_phase
                or sample.get("ordinal") != ordinal
                or not isinstance(sample.get("observed_at_utc"), str)
                or not sample["observed_at_utc"]
                or sample.get("index") != 0
                or sample.get("uuid") != expected_uuid
                or not isinstance(sample.get("name"), str)
                or not sample["name"]
                or not isinstance(driver_model, str)
                or not driver_model
                or sample.get("memory_total_mib") != 12288
                or type(used) is not int
                or not 0 <= used <= 2048
                or type(free) is not int
                or not 8192 <= free <= 12288
                or type(utilization) is not int
                or not 0 <= utilization <= 15
                or type(temperature) is not int
                or not -50 <= temperature <= 75
                or sample.get("selected_gpu_compute_process_count")
                != len(sample_processes)
                or sample.get("compute_inventory_sha256") != top_level_inventory_sha256
                or sample_inventory_sha256 != top_level_inventory_sha256
                or sample.get("wddm_desktop_baseline_tolerated")
                is not (driver_model.casefold() == "wddm" and used > 0)
            ):
                raise GemmaFiveRoleError("gpu_attestation_gpu_identity_invalid")
            if (
                key == "pre_lock_samples"
                and ordinal == 1
                and list(sample_processes) != list(top_level_processes)
            ):
                raise GemmaFiveRoleError("wddm_compute_baseline_not_first_sample")
    _assert_launcher_lock_guard(config, lease)
    return (
        lease,
        attestation,
        (
            lease_snapshot,
            attestation_snapshot,
            implementation_snapshot,
            script_snapshot,
            launcher_snapshot,
            config_snapshot,
        ),
    )


def _model_file_contract(config: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    return {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in (
            _mapping(value, "gemma_runner_model_file_invalid")
            for value in _sequence(
                _mapping(config["model"], "gemma_runner_model_invalid")["files"],
                "gemma_runner_model_files_invalid",
            )
        )
    }


def _model_root(config: Mapping[str, Any]) -> Path:
    value = str(_mapping(config["model"], "gemma_runner_model_invalid")["local_path"])
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _root() / path
    return qdiag._assert_physical_path(
        path, require_directory=True, label="Gemma 3 1B IT source export"
    )


def _copy_authenticated_file(
    source: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    source = qdiag._assert_physical_path(source, require_file=True, label=str(source))
    source_before = source.stat()
    digest = hashlib.sha256()
    total = 0
    with (
        source.open("rb", buffering=0) as reader,
        destination.open("xb", buffering=0) as writer,
    ):
        opened = os.fstat(reader.fileno())
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
        ):
            raise GemmaFiveRoleError("model_source_changed_before_snapshot_copy")
        while True:
            chunk = reader.read(8 * _MIB)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    source_after = source.stat()
    if (
        source_before.st_dev,
        source_before.st_ino,
        source_before.st_size,
        source_before.st_mtime_ns,
        source_before.st_ctime_ns,
    ) != (
        source_after.st_dev,
        source_after.st_ino,
        source_after.st_size,
        source_after.st_mtime_ns,
        source_after.st_ctime_ns,
    ):
        raise GemmaFiveRoleError("model_source_changed_during_snapshot_copy")
    if (
        total != expected_bytes
        or digest.hexdigest() != expected_sha256
        or _stream_sha256(destination, expected_bytes=expected_bytes) != expected_sha256
    ):
        raise GemmaFiveRoleError("private_model_snapshot_identity_mismatch")


@contextmanager
def _private_model_snapshot(
    config: Mapping[str, Any], run_id: str
) -> Iterator[tuple[Path, Mapping[str, str]]]:
    launch_root = _root().joinpath(
        *_output_relative(config["output"]["launch_root"]).parts
    )
    launch_root.mkdir(parents=True, exist_ok=True)
    snapshot = launch_root / f".private-model-{run_id}-{uuid.uuid4().hex}"
    snapshot.mkdir(exist_ok=False)
    contract = _model_file_contract(config)
    hashes: dict[str, str] = {}
    try:
        source = _model_root(config)
        for name in MODEL_FILES:
            expected_bytes, expected_sha = contract[name]
            _copy_authenticated_file(
                source / name,
                snapshot / name,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha,
            )
            hashes[name] = expected_sha
        yield snapshot, hashes
        for name in MODEL_FILES:
            expected_bytes, expected_sha = contract[name]
            if (
                _stream_sha256(snapshot / name, expected_bytes=expected_bytes)
                != expected_sha
            ):
                raise GemmaFiveRoleError(
                    "private_model_snapshot_changed_before_cleanup"
                )
    finally:
        if os.path.lexists(snapshot):
            if snapshot.is_symlink():
                snapshot.unlink()
            else:
                shutil.rmtree(snapshot)


def _tensor_sha256(tensor: Any, torch: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


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
        raise GemmaFiveRoleError("q_only_trainable_tensor_count_invalid")
    return digest.hexdigest()


def _validate_trainable_scope(model: Any) -> tuple[tuple[str, ...], int]:
    observed: dict[tuple[int, str], str] = {}
    names: list[str] = []
    parameters = 0
    errors: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        parameters += int(parameter.numel())
        match = _LORA_TENSOR_RE.search(name)
        if match is None:
            errors.append("unexpected_trainable_tensor")
            continue
        layer = int(match.group(1))
        side = match.group(2)
        key = (layer, side)
        if key in observed:
            errors.append("duplicate_lora_tensor")
        observed[key] = name
        expected = (4, 1152) if side == "A" else (1024, 4)
        if tuple(int(value) for value in parameter.shape) != expected:
            errors.append("lora_tensor_shape_invalid")
    required = {(layer, side) for layer in range(26) for side in ("A", "B")}
    if set(observed) != required:
        errors.append("lora_tensor_inventory_invalid")
    if parameters != 226_304:
        errors.append("lora_parameter_count_invalid")
    if errors:
        raise GemmaFiveRoleError(errors[0])
    return tuple(sorted(names)), parameters


def _assert_finite_and_gradient_scope(model: Any, torch: Any) -> dict[str, int]:
    result = {"A_nonzero": 0, "B_nonzero": 0, "finite": 0, "tensors": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        match = _LORA_TENSOR_RE.search(name)
        if match is None:
            raise GemmaFiveRoleError("gradient_scope_escaped_q_proj")
        result["tensors"] += 1
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise GemmaFiveRoleError("nonfinite_or_missing_q_lora_gradient")
        result["finite"] += 1
        if int(torch.count_nonzero(gradient).item()) > 0:
            result[f"{match.group(2)}_nonzero"] += 1
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            raise GemmaFiveRoleError("nonfinite_q_lora_parameter")
    if result["tensors"] != 52 or (result["A_nonzero"] + result["B_nonzero"]) < 1:
        raise GemmaFiveRoleError("q_lora_gradient_coverage_failed")
    return result


def _serialize_batch(
    processor: Any,
    example: Any,
    *,
    torch: Any,
) -> dict[str, Any]:
    serialized = binding.serialize_example(processor, example.prompt, example.target)
    if len(serialized.input_ids) > 768:
        raise GemmaFiveRoleError("runtime_example_exceeds_strict_sequence_length")
    return {
        "input_ids": torch.tensor(
            [serialized.input_ids], dtype=torch.long, device="cuda"
        ),
        "attention_mask": torch.ones(
            (1, len(serialized.input_ids)), dtype=torch.long, device="cuda"
        ),
        "labels": torch.tensor([serialized.labels], dtype=torch.long, device="cuda"),
    }


def _mean_eval_loss(
    model: Any,
    processor: Any,
    examples: Sequence[Any],
    *,
    torch: Any,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for example in examples:
            loss = model(**_serialize_batch(processor, example, torch=torch)).loss
            if not bool(torch.isfinite(loss).item()):
                raise GemmaFiveRoleError("nonfinite_eval_proxy_loss")
            values.append(float(loss.detach().cpu()))
    if len(values) != 40:
        raise GemmaFiveRoleError("eval_proxy_count_changed")
    return sum(values) / len(values)


def _enabled_vs_disabled_next_token_effect(
    model: Any,
    batch: Mapping[str, Any],
    prediction_position: int,
    *,
    torch: Any,
) -> dict[str, Any]:
    if prediction_position < 0:
        raise GemmaFiveRoleError("adapter_effect_prediction_position_invalid")
    disable_adapter = getattr(model, "disable_adapter", None)
    if not callable(disable_adapter):
        raise GemmaFiveRoleError("peft_disable_adapter_context_unavailable")
    model.eval()
    forward = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "use_cache": False,
    }
    with torch.no_grad():
        enabled_output = model(**forward).logits
        expected_sequence = int(forward["input_ids"].shape[1])
        if (
            enabled_output.ndim != 3
            or enabled_output.shape[0] != 1
            or enabled_output.shape[1] != expected_sequence
            or enabled_output.shape[2] <= 0
            or prediction_position >= enabled_output.shape[1]
        ):
            raise GemmaFiveRoleError("adapter_effect_enabled_logits_shape_invalid")
        enabled_shape = tuple(int(value) for value in enabled_output.shape)
        enabled = enabled_output[0, prediction_position].detach().float().clone()
        del enabled_output
        with disable_adapter():
            disabled_output = model(**forward).logits
            if tuple(int(value) for value in disabled_output.shape) != enabled_shape:
                raise GemmaFiveRoleError("adapter_effect_disabled_logits_shape_invalid")
            disabled = disabled_output[0, prediction_position].detach().float().clone()
        del disabled_output

    if not (
        bool(torch.isfinite(enabled).all().item())
        and bool(torch.isfinite(disabled).all().item())
    ):
        raise GemmaFiveRoleError("adapter_effect_nonfinite_logits")
    delta = torch.abs(enabled - disabled)
    finite = bool(torch.isfinite(delta).all().item())
    max_abs = float(delta.max().detach().cpu())
    mean_abs = float(delta.mean().detach().cpu())
    if (
        not finite
        or not math.isfinite(max_abs)
        or not math.isfinite(mean_abs)
        or max_abs <= 0.0
        or mean_abs < 0.0
    ):
        raise GemmaFiveRoleError("adapter_output_effect_absent_or_invalid")
    return {
        "finite": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "vocabulary_logits": int(delta.numel()),
    }


def _adapter_effect_prefix_view(serialized: Any) -> dict[str, Any]:
    if (
        len(serialized.input_ids) != len(serialized.labels)
        or len(serialized.input_ids) > 768
    ):
        raise GemmaFiveRoleError("adapter_effect_view_exceeds_sequence_length")
    supervised = [
        index for index, label in enumerate(serialized.labels) if int(label) != -100
    ]
    if not supervised or supervised[0] <= 0:
        raise GemmaFiveRoleError("adapter_effect_supervised_boundary_invalid")
    first_supervised = supervised[0]
    input_prefix = tuple(
        int(value) for value in serialized.input_ids[:first_supervised]
    )
    if not input_prefix:
        raise GemmaFiveRoleError("adapter_effect_prefix_empty")
    return {
        "input_prefix": input_prefix,
        "prediction_position": len(input_prefix) - 1,
        "full_sequence_tokens": len(serialized.input_ids),
    }


def _fixed_training_view_adapter_effect(
    model: Any,
    processor: Any,
    example: Any,
    *,
    role: str,
    torch: Any,
) -> dict[str, Any]:
    serialized = binding.serialize_example(processor, example.prompt, example.target)
    view = _adapter_effect_prefix_view(serialized)
    input_prefix = view["input_prefix"]
    prediction_position = int(view["prediction_position"])
    batch = {
        "input_ids": torch.tensor([input_prefix], dtype=torch.long, device="cuda"),
        "attention_mask": torch.ones(
            (1, len(input_prefix)), dtype=torch.long, device="cuda"
        ),
    }
    metrics = _enabled_vs_disabled_next_token_effect(
        model,
        batch,
        prediction_position,
        torch=torch,
    )
    return {
        "view": "first_train_record_first_supervised_next_token_v1",
        "comparison": "enabled_vs_disable_adapter_after_training",
        "record_id_sha256": _sha256(
            (
                f"anchor.gemma3.adapter-effect-record.v1\0{role}\0{example.record_id}"
            ).encode("utf-8")
        ),
        "serialized_training_view_sha256": _sha256(
            _canonical_json(
                {
                    "input_prefix": list(input_prefix),
                    "prediction_position": prediction_position,
                }
            )
        ),
        "full_serialized_sequence_tokens": view["full_sequence_tokens"],
        "forward_prefix_tokens": len(input_prefix),
        "prediction_position_zero_based_in_forward_prefix": prediction_position,
        "future_target_suffix_forwarded": False,
        "sample_body_included": False,
        "token_ids_included": False,
        **metrics,
    }


def _runtime_process_basename(reported_name: str, *, pid: int, timeout: int) -> str:
    name = reported_name.strip()
    if re.fullmatch(r"\[?Insufficient Permissions\]?", name):
        try:
            completed = subprocess.run(
                [
                    "tasklist.exe",
                    "/FI",
                    f"PID eq {pid}",
                    "/FO",
                    "CSV",
                    "/NH",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            rows = list(csv.reader(completed.stdout.splitlines()))
        except (OSError, subprocess.TimeoutExpired, csv.Error):
            raise GemmaFiveRoleError(
                "runtime_compute_permission_pid_resolution_failed"
            ) from None
        matches = [
            row
            for row in rows
            if len(row) >= 2 and row[1].strip().isdigit() and int(row[1]) == pid
        ]
        if completed.returncode != 0 or len(matches) != 1:
            raise GemmaFiveRoleError("runtime_compute_permission_pid_resolution_failed")
        name = matches[0][0].strip()
    basename = PureWindowsPath(name.strip('"')).name.casefold()
    if not Path(basename).suffix:
        basename += ".exe"
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+-]*\.exe", basename) is None:
        raise GemmaFiveRoleError("runtime_compute_process_name_invalid")
    return basename


def _query_runtime_gpu(config: Mapping[str, Any], *, allow_pid: int) -> dict[str, Any]:
    gpu = _mapping(config["gpu_policy"], "gemma_runner_gpu_policy_invalid")
    timeout = int(gpu["command_timeout_seconds"])
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,driver_model.current,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise GemmaFiveRoleError("runtime_nvidia_smi_gpu_query_failed") from None
    rows = [
        [cell.strip() for cell in row.split(",")]
        for row in completed.stdout.splitlines()
        if row.strip()
    ]
    if completed.returncode != 0 or len(rows) <= int(gpu["expected_gpu_index"]):
        raise GemmaFiveRoleError("runtime_nvidia_smi_gpu_query_invalid")
    row = rows[int(gpu["expected_gpu_index"])]
    if len(row) != 8:
        raise GemmaFiveRoleError("runtime_nvidia_smi_gpu_row_invalid")
    try:
        snapshot = {
            "index": int(row[0]),
            "uuid": row[1],
            "driver_model": row[2],
            "memory_total_mib": int(row[3]),
            "memory_used_mib": int(row[4]),
            "memory_free_mib": int(row[5]),
            "utilization_percent": int(row[6]),
            "temperature_c": int(row[7]),
        }
    except ValueError:
        raise GemmaFiveRoleError("runtime_nvidia_smi_gpu_values_invalid") from None
    if (
        snapshot["index"] != 0
        or snapshot["uuid"] != gpu["expected_gpu_uuid"]
        or snapshot["memory_total_mib"] != gpu["expected_total_memory_mib"]
        or snapshot["temperature_c"] > gpu["runtime_temperature_max_c"]
    ):
        raise GemmaFiveRoleError("runtime_gpu_identity_or_temperature_gate_failed")
    compute_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        processes = subprocess.run(
            compute_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise GemmaFiveRoleError("runtime_nvidia_smi_compute_query_failed") from None
    if processes.returncode != 0:
        raise GemmaFiveRoleError("runtime_nvidia_smi_compute_query_invalid")
    process_output = processes.stdout.strip()
    if re.fullmatch(r"No running processes found\.?", process_output):
        process_rows: list[list[str]] = []
    else:
        try:
            process_rows = list(csv.reader(processes.stdout.splitlines()))
        except csv.Error:
            raise GemmaFiveRoleError("runtime_compute_row_invalid") from None
    foreign: list[str] = []
    allowed_gui: list[dict[str, object]] = []
    seen_pids: set[int] = set()
    for row in process_rows:
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != 4 or row[0].strip() != gpu["expected_gpu_uuid"]:
            if len(row) != 4:
                raise GemmaFiveRoleError("runtime_compute_row_invalid")
            continue
        try:
            pid = int(row[1].strip())
        except ValueError:
            raise GemmaFiveRoleError("runtime_compute_pid_invalid") from None
        if pid <= 0 or pid in seen_pids:
            raise GemmaFiveRoleError("runtime_compute_pid_invalid")
        seen_pids.add(pid)
        if pid == allow_pid:
            continue
        basename = _runtime_process_basename(row[2], pid=pid, timeout=timeout)
        if (
            snapshot["driver_model"].casefold() == "wddm"
            and basename in WDDM_GUI_PROCESS_ALLOWLIST
        ):
            allowed_gui.append({"pid": pid, "process_name": basename})
        else:
            foreign.append(f"{basename}:{pid}")
    if foreign and gpu["unknown_or_non_allowlisted_compute_process_forbidden"]:
        raise GemmaFiveRoleError("foreign_compute_process_detected")
    snapshot["allowlisted_wddm_gui_processes"] = sorted(
        allowed_gui, key=lambda item: (int(item["pid"]), str(item["process_name"]))
    )
    snapshot["foreign_compute_processes_observed"] = sorted(foreign)
    return snapshot


def _save_adapter(
    model: Any,
    destination: Path,
    *,
    torch: Any,
) -> Mapping[str, str]:
    if os.path.lexists(destination):
        raise FileExistsError(f"adapter output exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        readme = temporary / "README.md"
        if os.path.lexists(readme):
            qdiag._assert_physical_path(
                readme, require_file=True, label="generated PEFT README"
            )
            readme.unlink()
        files = {
            item.name
            for item in temporary.iterdir()
            if item.is_file() and not item.is_symlink()
        }
        if files != set(ADAPTER_FILES):
            raise GemmaFiveRoleError("saved_adapter_file_inventory_invalid")
        adapter_config_path = temporary / "adapter_config.json"
        config = dict(
            _strict_json(
                _stable_snapshot(adapter_config_path, max_bytes=_MAX_CONFIG_BYTES).data,
                "saved_adapter_config_json_invalid",
            )
        )
        config["base_model_name_or_path"] = ADAPTER_BASE_IDENTITY
        normalized = adapter_config_path.with_name(
            f".adapter_config.json.tmp-{uuid.uuid4().hex}"
        )
        with normalized.open("xb", buffering=0) as handle:
            handle.write(_canonical_json(config))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(normalized, adapter_config_path)
        config = _strict_json(
            _stable_snapshot(adapter_config_path, max_bytes=_MAX_CONFIG_BYTES).data,
            "saved_adapter_config_json_invalid",
        )
        if (
            config.get("r") != 4
            or set(config.get("target_modules", ())) != {"q_proj"}
            or config.get("bias") != "none"
            or config.get("base_model_name_or_path") != ADAPTER_BASE_IDENTITY
        ):
            raise GemmaFiveRoleError("saved_adapter_config_contract_invalid")
        try:
            from safetensors.numpy import load_file as load_safetensors
        except ImportError:
            raise GemmaFiveRoleError("safetensors_runtime_unavailable") from None
        tensors = load_safetensors(temporary / "adapter_model.safetensors")
        observed: set[tuple[int, str]] = set()
        for name, value in tensors.items():
            match = _LORA_TENSOR_RE.search(name)
            if match is None:
                raise GemmaFiveRoleError("saved_adapter_tensor_scope_invalid")
            layer = int(match.group(1))
            side = match.group(2)
            expected = (4, 1152) if side == "A" else (1024, 4)
            if tuple(int(item) for item in value.shape) != expected:
                raise GemmaFiveRoleError("saved_adapter_tensor_shape_invalid")
            observed.add((layer, side))
        if observed != {(layer, side) for layer in range(26) for side in ("A", "B")}:
            raise GemmaFiveRoleError("saved_adapter_tensor_inventory_invalid")
        hashes = {name: _stream_sha256(temporary / name) for name in ADAPTER_FILES}
        os.rename(temporary, destination)
        for name, digest in hashes.items():
            if _stream_sha256(destination / name) != digest:
                raise GemmaFiveRoleError("published_adapter_hash_drift")
        return hashes
    finally:
        if os.path.lexists(temporary):
            shutil.rmtree(temporary)


def _phase_seed(base_seed: int) -> int:
    # Smoke and full deliberately use the same seed so their fresh adapter
    # initialization digests must match.
    return base_seed


def _validate_adamw8bit_state(
    optimizer: Any,
    parameters: Sequence[Any],
    *,
    torch: Any,
    bitsandbytes_version: str,
) -> dict[str, object]:
    if bitsandbytes_version != OPTIMIZER_RUNTIME_CONTRACT["package_version"]:
        raise GemmaFiveRoleError("bitsandbytes_version_drift")
    state_tensors = 0
    state_elements = 0
    for parameter in parameters:
        parameter_elements = int(parameter.numel())
        if parameter_elements < 4096:
            raise GemmaFiveRoleError("adamw8bit_parameter_below_quantization_floor")
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping):
            raise GemmaFiveRoleError("adamw8bit_state_missing")
        for name in ("state1", "state2"):
            tensor = state.get(name)
            device = getattr(tensor, "device", None)
            if (
                tensor is None
                or getattr(tensor, "dtype", None) != torch.uint8
                or getattr(device, "type", None) != "cuda"
                or int(tensor.numel()) != parameter_elements
            ):
                raise GemmaFiveRoleError("adamw8bit_state_not_uint8_cuda")
            state_tensors += 1
            state_elements += int(tensor.numel())
    if state_tensors != len(parameters) * 2:
        raise GemmaFiveRoleError("adamw8bit_state_inventory_invalid")
    return {
        **OPTIMIZER_RUNTIME_CONTRACT,
        "parameter_tensors": len(parameters),
        "state_tensors": state_tensors,
        "state_elements": state_elements,
        "state_dtype": "uint8",
        "state_device": "cuda",
    }


def _execute_phase(
    config: Mapping[str, Any],
    *,
    role: str,
    phase: str,
    steps: int,
    dataset: Any,
    processor: Any,
    model_snapshot: Path,
    role_output: Path,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if getattr(bnb, "__version__", None) != "0.48.2":
        raise GemmaFiveRoleError("bitsandbytes_version_drift")
    if not torch.cuda.is_available():
        raise GemmaFiveRoleError("execute_requires_cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not (torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32):
        raise GemmaFiveRoleError("tf32_controls_not_enabled")
    training = _mapping(config["training"], "gemma_runner_training_invalid")
    seed = _phase_seed(int(training["seed"]))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _query_runtime_gpu(config, allow_pid=os.getpid())

    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    base = AutoModelForCausalLM.from_pretrained(model_snapshot, **kwargs).to("cuda")
    overlay = _mapping(
        config["model"]["runtime_special_token_overlay"],
        "gemma_runner_special_overlay_invalid",
    )
    base.config.pad_token_id = int(overlay["pad_token_id"])
    base.config.eos_token_id = int(overlay["eos_token_id"])
    base.config.bos_token_id = int(overlay["bos_token_id"])
    base.config.use_cache = False
    for parameter in base.parameters():
        parameter.requires_grad = False
    if training["gradient_checkpointing"]:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
    captured, base_hash_before = qdiag._capture_base_parameters(base, torch)
    # Reset immediately before PEFT construction.  Base loading must not
    # influence adapter initialization.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable_names, trainable_parameters = _validate_trainable_scope(model)
    initial_adapter_sha256 = _trainable_digest(model, torch)
    eval_before = (
        _mean_eval_loss(model, processor, dataset.eval_proxy, torch=torch)
        if phase == "full"
        else None
    )
    trainable_parameter_tensors = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = bnb.optim.AdamW8bit(
        trainable_parameter_tensors,
        lr=float(training["learning_rate"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
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
    gradients: dict[str, int] = {}
    optimizer_state: dict[str, object] | None = None
    order = hashlib.sha256()
    monitor_samples: list[Mapping[str, Any]] = []
    for step in range(steps):
        example = dataset.train[step % len(dataset.train)]
        order.update(example.record_id.encode("utf-8"))
        order.update(b"\n")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**_serialize_batch(processor, example, torch=torch)).loss
        if not bool(torch.isfinite(loss).item()):
            raise GemmaFiveRoleError("nonfinite_training_loss")
        loss.backward()
        gradients = _assert_finite_and_gradient_scope(model, torch)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=float(training["max_grad_norm"]),
        )
        optimizer.step()
        if optimizer_state is None:
            optimizer_state = _validate_adamw8bit_state(
                optimizer,
                trainable_parameter_tensors,
                torch=torch,
                bitsandbytes_version=str(bnb.__version__),
            )
        qdiag._assert_trainable_parameters_finite(model, torch)
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % int(
            config["gpu_policy"]["runtime_monitor_interval_steps"]
        ) == 0 or step + 1 == steps:
            monitor_samples.append(_query_runtime_gpu(config, allow_pid=os.getpid()))
    final_adapter_sha256 = _trainable_digest(model, torch)
    if final_adapter_sha256 == initial_adapter_sha256:
        raise GemmaFiveRoleError("adapter_did_not_change")
    if optimizer_state is None:
        raise GemmaFiveRoleError("adamw8bit_state_not_observed")
    adapter_effect = _fixed_training_view_adapter_effect(
        model,
        processor,
        dataset.train[0],
        role=role,
        torch=torch,
    )
    eval_after = (
        _mean_eval_loss(model, processor, dataset.eval_proxy, torch=torch)
        if phase == "full"
        else None
    )
    base_hash_after = qdiag._assert_base_parameters_unchanged(captured, torch)
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    if (
        peak_allocated
        > int(config["gpu_policy"]["torch_peak_allocated_max_mib"]) * _MIB
        or peak_reserved
        > int(config["gpu_policy"]["torch_peak_reserved_max_mib"]) * _MIB
    ):
        raise GemmaFiveRoleError("torch_peak_memory_policy_exceeded")
    adapter_hashes: Mapping[str, str] | None = None
    if phase == "full":
        adapter_hashes = _save_adapter(model, role_output / "adapter", torch=torch)
    receipt = {
        "schema_version": PHASE_RECEIPT_VERSION,
        "status": "passed_controlled_proxy_diagnostic",
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
        "eval_proxy_loss_before": eval_before,
        "eval_proxy_loss_after": eval_after,
        "eval_proxy_loss_delta": (
            None if eval_before is None else eval_after - eval_before
        ),
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_sha256": final_adapter_sha256,
        "trainable_tensor_names_sha256": _sha256(
            ("\n".join(trainable_names) + "\n").encode("utf-8")
        ),
        "trainable_parameters": trainable_parameters,
        "gradient_coverage": gradients,
        "optimizer": optimizer_state,
        "adapter_effect": adapter_effect,
        "adapter_artifact_sha256": adapter_hashes,
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "runtime_gpu_samples": monitor_samples,
        "numerics": {
            "compute_dtype": "bfloat16",
            "tf32": True,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
        "kv_runtime_boundary": dict(KV_RUNTIME_BOUNDARY),
        "claims": {
            "diagnostic_only": True,
            "controlled_proxy_only": True,
            "formal": False,
            "runtime_private_tail_materialized": False,
        },
    }
    _atomic_write_json(role_output / f"{phase}_receipt.json", receipt)
    del optimizer, model, base, captured
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def _assert_smoke_full_freshness(
    smoke: Mapping[str, Any], full: Mapping[str, Any]
) -> None:
    if (
        smoke.get("initial_adapter_sha256") != full.get("initial_adapter_sha256")
        or smoke.get("base_hash_before") != full.get("base_hash_before")
        or smoke.get("train_record_order_sha256")
        == full.get("train_record_order_sha256")
        or smoke.get("optimizer_steps") != 2
        or full.get("optimizer_steps") != 160
        or smoke.get("smoke_checkpoint_consumed") is not False
        or full.get("smoke_checkpoint_consumed") is not False
    ):
        raise GemmaFiveRoleError("smoke_full_freshness_gate_failed")


def _failure_receipt(
    config: Mapping[str, Any],
    *,
    run_id: str,
    error_code: str,
    role: str | None,
    phase: str | None,
) -> None:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        return
    launch_root = _root().joinpath(
        *_output_relative(config["output"]["launch_root"]).parts
    )
    path = launch_root / run_id / "failure_receipt.json"
    if os.path.lexists(path):
        return
    value = {
        "schema_version": FAILURE_RECEIPT_VERSION,
        "status": "blocked",
        "run_id": run_id,
        "error_code": error_code,
        "role": role,
        "phase": phase,
        "sample_content_included": False,
        "automatic_retry": False,
        "claims": {
            "formal": False,
            "training_authorized": False,
            "formal_training_authorized": False,
        },
    }
    _atomic_write_json(path, value)


def execute(
    config: Mapping[str, Any],
    *,
    run_id: str,
    lease_path: str | Path,
    lease_sha256: str,
    gpu_attestation_path: str | Path,
    gpu_attestation_sha256: str,
) -> dict[str, Any]:
    current_role: str | None = None
    current_phase: str | None = None
    staging: Path | None = None
    try:
        preflight = build_preflight(config)
        lease, attestation, external_snapshots = _validate_launch_receipts(
            config,
            run_id=run_id,
            lease_path=lease_path,
            lease_sha256=lease_sha256,
            gpu_attestation_path=gpu_attestation_path,
            gpu_attestation_sha256=gpu_attestation_sha256,
        )
        output_root = _root().joinpath(
            *_output_relative(config["output"]["artifact_root"]).parts
        )
        output_root.mkdir(parents=True, exist_ok=True)
        destination = output_root / run_id
        if os.path.lexists(destination):
            raise FileExistsError(f"run output exists: {destination}")
        staging = output_root / f".{run_id}.tmp-{uuid.uuid4().hex}"
        staging.mkdir(exist_ok=False)
        binding_config = _binding_config(config)
        datasets = binding.load_all_role_datasets(binding_config)
        with _private_model_snapshot(config, run_id) as (
            model_snapshot,
            snapshot_hashes,
        ):
            processor = binding.load_sentencepiece(model_snapshot / "tokenizer.model")
            roles: list[Mapping[str, Any]] = []
            for role in ROLES:
                current_role = role
                role_output = staging / role
                role_output.mkdir()
                current_phase = "smoke"
                smoke = _execute_phase(
                    config,
                    role=role,
                    phase="smoke",
                    steps=2,
                    dataset=datasets[role],
                    processor=processor,
                    model_snapshot=model_snapshot,
                    role_output=role_output,
                )
                current_phase = "full"
                full = _execute_phase(
                    config,
                    role=role,
                    phase="full",
                    steps=160,
                    dataset=datasets[role],
                    processor=processor,
                    model_snapshot=model_snapshot,
                    role_output=role_output,
                )
                _assert_smoke_full_freshness(smoke, full)
                roles.append(
                    {
                        "role": role,
                        "smoke_receipt_sha256": _stream_sha256(
                            role_output / "smoke_receipt.json"
                        ),
                        "full_receipt_sha256": _stream_sha256(
                            role_output / "full_receipt.json"
                        ),
                        "adapter_artifact_sha256": full["adapter_artifact_sha256"],
                    }
                )
            current_role = None
            current_phase = None
            _assert_launcher_lock_guard(config, lease)
            for snapshot in external_snapshots:
                snapshot.assert_unchanged()
            success = {
                "schema_version": RUN_RECEIPT_VERSION,
                "status": "passed_controlled_proxy_diagnostic_only",
                "run_id": run_id,
                "identity": {
                    "config_sha256": config["_config_sha256"],
                    "implementation_sha256": preflight["identity"][
                        "implementation_sha256"
                    ],
                    "tokenizer_binding_manifest_sha256": preflight["identity"][
                        "tokenizer_binding_manifest_sha256"
                    ],
                    "lease_receipt_sha256": external_snapshots[0].sha256,
                    "gpu_attestation_sha256": external_snapshots[1].sha256,
                    "private_model_snapshot_file_sha256": dict(snapshot_hashes),
                },
                "launcher": {
                    "launcher_pid": lease["launcher_pid"],
                    "canonical_lock": lease["canonical_lock"],
                    "expected_gpu_uuid": lease["expected_gpu_uuid"],
                    "pre_lock_samples": len(attestation["pre_lock_samples"]),
                    "post_lock_samples": len(attestation["post_lock_samples"]),
                },
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
                    "optimizer": dict(OPTIMIZER_RUNTIME_CONTRACT),
                },
                "kv_runtime_boundary": dict(config["kv_runtime_boundary"]),
                "claims": {
                    "diagnostic_only": True,
                    "controlled_proxy_only": True,
                    "training_executed": True,
                    "runtime_private_tail_materialized": False,
                    "training_authorized": False,
                    "formal_training_authorized": False,
                    "formal": False,
                    "eval_proxy_is_heldout": False,
                },
                "audit": {
                    "provider_requests": 0,
                    "network_requests": 0,
                    "model_object_loads": 10,
                    "gpu_concurrency": 1,
                    "protected_body_reads": 0,
                },
            }
            _atomic_write_json(staging / "run_receipt.json", success)
        # The private snapshot context re-hashes and removes roughly 2 GiB after
        # the inner check.  Close that cleanup window immediately before the
        # atomic publication boundary.
        _assert_launcher_lock_guard(config, lease)
        for snapshot in external_snapshots:
            snapshot.assert_unchanged()
        os.rename(staging, destination)
        observed = _strict_json(
            _stable_snapshot(
                destination / "run_receipt.json", max_bytes=_MAX_RECEIPT_BYTES
            ).data,
            "published_run_receipt_invalid",
        )
        if (
            observed.get("run_id") != run_id
            or observed.get("status") != success["status"]
        ):
            raise GemmaFiveRoleError("published_run_receipt_verification_failed")
        return success
    except Exception as error:
        if staging is not None and os.path.lexists(staging):
            shutil.rmtree(staging)
        code = (
            str(error)
            if isinstance(error, (GemmaFiveRoleError, ConfigError))
            else type(error).__name__
        )
        _failure_receipt(
            config,
            run_id=run_id,
            error_code=code,
            role=current_role,
            phase=current_phase,
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gemma 3 1B IT five-role Q-only controlled-proxy runner. "
            "The default path is model-free; --execute additionally requires "
            "authenticated launcher lease and GPU attestation receipts."
        )
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="explicit alias for the default model-free preflight",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--lease-receipt")
    parser.add_argument("--lease-receipt-sha256")
    parser.add_argument("--gpu-attestation")
    parser.add_argument("--gpu-attestation-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.dry_run and args.execute:
            raise GemmaFiveRoleError("dry_run_and_execute_are_mutually_exclusive")
        config = load_config(args.config)
        if not args.execute:
            result = build_preflight(config)
        else:
            required = {
                "--run-id": args.run_id,
                "--lease-receipt": args.lease_receipt,
                "--lease-receipt-sha256": args.lease_receipt_sha256,
                "--gpu-attestation": args.gpu_attestation,
                "--gpu-attestation-sha256": args.gpu_attestation_sha256,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise GemmaFiveRoleError("execute_missing_required_launcher_inputs")
            result = execute(
                config,
                run_id=args.run_id,
                lease_path=args.lease_receipt,
                lease_sha256=args.lease_receipt_sha256,
                gpu_attestation_path=args.gpu_attestation,
                gpu_attestation_sha256=args.gpu_attestation_sha256,
            )
    except (ConfigError, GemmaFiveRoleError, FileExistsError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error_code": str(error),
                    "sample_content_included": False,
                    "automatic_retry": False,
                    "claims": {
                        "training_authorized": False,
                        "formal_training_authorized": False,
                        "formal": False,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "CONFIG_PATH",
    "CONFIG_VERSION",
    "IMPLEMENTATION_PATH",
    "ROLES",
    "GemmaFiveRoleError",
    "build_preflight",
    "execute",
    "load_config",
    "main",
    "validate_config",
]
