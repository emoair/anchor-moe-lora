"""Fail-closed aggregate-only evaluator for Gemma 3 chat Q-only experts.

The checked-in configuration binds the FINAL consumer and dataset identities;
only the future training run ID and receipt identity remain ``pending``.
Status and preflight therefore return before touching a model or GPU.  Once a
completed training run is explicitly bound, execution authenticates the
complete dataset and training receipt before it loads the frozen Q8 base.

Only aggregate counters are published.  Prompts, targets, generated text, raw
token IDs, and per-record metrics never enter a receipt or log payload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from anchor_mvp.training import gemma3_chat_five_expert_qonly_v1 as chat
from anchor_mvp.training import gemma3_tokenizer_binding_v1 as binding
from anchor_mvp.training import qwen_synthetic_scaffold_diagnostic as snapshot_io
from anchor_mvp.training.config import ConfigError as ConfigError


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_five_expert_qonly_generation_eval_v1.json"
)
CONFIG_VERSION = "anchor.gemma3-chat-five-expert-qonly-generation-eval-config.v1"
RECEIPT_VERSION = "anchor.gemma3-chat-five-expert-qonly-generation-eval-receipt.v1"
LOCK_VERSION = "anchor.gemma3-chat-five-expert-qonly-generation-eval-lock.v1"
PUBLIC_ERROR_CODES = frozenset(
    {
        "eval_adapter_name_required",
        "eval_base_model_file_hash_drift",
        "eval_base_model_identity_mismatch",
        "eval_conflicting_handoff_lock_appeared",
        "eval_conflicting_handoff_lock_exists",
        "eval_final_source_recheck_drift",
        "eval_generation_prefix_invalid",
        "eval_gpu_inventory_parse_failed",
        "eval_gpu_lock_changed",
        "eval_inputs_pending",
        "eval_output_run_id_invalid",
        "eval_q8_scb_identity_drift",
        "eval_receipt_body_or_per_record_field_forbidden",
        "eval_selection_count_invalid",
        "eval_trainable_parameter_detected",
    }
)
ROLES = ("humor", "serious", "angry_style", "tool_call", "review_audit")
ARMS = ("base", "correct_adapter", "wrong_route")
PARTITIONS = ("train/chat.jsonl", "eval_proxy/chat.jsonl")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
PENDING = "pending"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LOCK_PATH = ROOT / "runs" / "formal-v3-training.lock"
CONFLICTING_LOCKS = (
    ROOT / "runs" / "distill-train-handoff" / "gpu-job.lock",
    ROOT / "runs" / "distill-train-handoff-v3" / "gpu-job.lock",
)
MAX_CONFIG_BYTES = 2_000_000
MAX_METADATA_BYTES = 8_000_000
IDENTITY_ZH = "我是由Air训练的测试模型。"
IDENTITY_EN = "I am a test model trained by Air."
IDENTITY_PROVIDER_FORBIDDEN = ("openai", "google", "deepmind", "qwen")
EVIDENCE_KEY_RE = re.compile(
    r"(?:evidence|result|source|citation|reference|ref_id|ground)",
    re.IGNORECASE,
)
DEPENDENCY_KEY_RE = re.compile(
    r"(?:dependency|record_id|target_sha256|decision|verdict|finding|role)",
    re.IGNORECASE,
)


class ChatGenerationEvalError(RuntimeError):
    """Raised when any evaluation identity or runtime invariant drifts."""


@dataclass(frozen=True)
class AuthenticatedInputs:
    consumer_config: Mapping[str, Any]
    dataset: Any
    run_receipt: Mapping[str, Any]
    run_root: Path
    adapter_paths: Mapping[str, Path]
    source_hashes: Mapping[str, Any]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ChatGenerationEvalError(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChatGenerationEvalError(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise ChatGenerationEvalError(code)


def _identity(
    value: object,
    code: str,
    *,
    pending_allowed: bool = False,
) -> str:
    if pending_allowed and value == PENDING:
        return PENDING
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ChatGenerationEvalError(code)
    return value


def _relative_path(value: object, code: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ChatGenerationEvalError(code)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ChatGenerationEvalError(code)
    return path


def _repo_path(value: object, code: str) -> Path:
    relative = _relative_path(value, code)
    path = ROOT.joinpath(*relative.parts)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        raise ChatGenerationEvalError(code) from None
    return path


def _load_json_snapshot(path: Path, *, max_bytes: int) -> tuple[Any, Any]:
    snapshot = snapshot_io._read_snapshot(path, max_bytes=max_bytes)
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChatGenerationEvalError("eval_json_snapshot_invalid") from exc
    snapshot.assert_unchanged()
    return value, snapshot


def _pending(config: Mapping[str, Any]) -> tuple[str, ...]:
    consumer = _mapping(config.get("consumer"), "eval_consumer_invalid")
    training = _mapping(config.get("training_run"), "eval_training_run_invalid")
    partitions = _mapping(
        consumer.get("partition_sha256"), "eval_partition_identities_invalid"
    )
    candidates = (
        ("consumer.config_sha256", consumer.get("config_sha256")),
        (
            "consumer.implementation_sha256",
            consumer.get("implementation_sha256"),
        ),
        (
            "consumer.dataset_manifest_sha256",
            consumer.get("dataset_manifest_sha256"),
        ),
        (
            "consumer.partition_sha256.train/chat.jsonl",
            partitions.get("train/chat.jsonl"),
        ),
        (
            "consumer.partition_sha256.eval_proxy/chat.jsonl",
            partitions.get("eval_proxy/chat.jsonl"),
        ),
        ("training_run.run_id", training.get("run_id")),
        ("training_run.receipt_sha256", training.get("receipt_sha256")),
    )
    return tuple(name for name, value in candidates if value == PENDING)


def validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "identity_state",
            "consumer",
            "training_run",
            "evaluation",
            "runtime",
            "output",
            "claims",
        },
        "eval_config_fields_drift",
    )
    if config.get("schema_version") != CONFIG_VERSION:
        raise ChatGenerationEvalError("eval_config_version_invalid")
    consumer = _mapping(config["consumer"], "eval_consumer_invalid")
    _exact_keys(
        consumer,
        {
            "config_path",
            "config_sha256",
            "implementation_path",
            "implementation_sha256",
            "dataset_manifest_sha256",
            "partition_sha256",
        },
        "eval_consumer_fields_drift",
    )
    if _relative_path(
        consumer["config_path"], "eval_consumer_config_path_invalid"
    ) != PurePosixPath(chat.CONFIG_PATH) or _relative_path(
        consumer["implementation_path"],
        "eval_consumer_implementation_path_invalid",
    ) != PurePosixPath(chat.IMPLEMENTATION_PATH):
        raise ChatGenerationEvalError("eval_consumer_canonical_path_drift")
    for identity_field in (
        "config_sha256",
        "implementation_sha256",
        "dataset_manifest_sha256",
    ):
        _identity(
            consumer.get(identity_field),
            f"eval_consumer_{identity_field}_invalid",
            pending_allowed=True,
        )
    partition_sha = _mapping(
        consumer["partition_sha256"], "eval_partition_identities_invalid"
    )
    _exact_keys(
        partition_sha,
        set(PARTITIONS),
        "eval_partition_identity_inventory_drift",
    )
    for path in PARTITIONS:
        _identity(
            partition_sha[path],
            "eval_partition_identity_invalid",
            pending_allowed=True,
        )

    run = _mapping(config["training_run"], "eval_training_run_invalid")
    _exact_keys(
        run,
        {
            "run_id",
            "artifact_root",
            "receipt_name",
            "receipt_sha256",
            "receipt_schema_version",
            "required_status",
            "adapter_files",
            "phase_receipts",
        },
        "eval_training_run_fields_drift",
    )
    run_id = run["run_id"]
    if run_id != PENDING and (
        not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None
    ):
        raise ChatGenerationEvalError("eval_training_run_id_invalid")
    _relative_path(run["artifact_root"], "eval_artifact_root_invalid")
    if run["receipt_name"] != "run_receipt.json":
        raise ChatGenerationEvalError("eval_run_receipt_name_drift")
    _identity(
        run["receipt_sha256"],
        "eval_run_receipt_sha_invalid",
        pending_allowed=True,
    )
    if (
        run["receipt_schema_version"] != chat.RUN_RECEIPT_VERSION
        or run["required_status"] != "passed_diagnostic_training_only"
        or tuple(run["adapter_files"]) != ADAPTER_FILES
        or dict(run["phase_receipts"])
        != {
            "smoke": "smoke_receipt.json",
            "full": "full_receipt.json",
        }
    ):
        raise ChatGenerationEvalError("eval_training_run_contract_drift")

    evaluation = _mapping(config["evaluation"], "eval_evaluation_invalid")
    _exact_keys(
        evaluation,
        {
            "split",
            "records_per_role",
            "max_new_tokens",
            "roles",
            "arms",
            "wrong_route_adapter",
            "decoding",
            "role_proxy",
        },
        "eval_evaluation_fields_drift",
    )
    wrong = _mapping(evaluation["wrong_route_adapter"], "eval_wrong_route_invalid")
    if (
        evaluation["split"] != "eval_proxy"
        or type(evaluation["records_per_role"]) is not int
        or not 1 <= evaluation["records_per_role"] <= 40
        or type(evaluation["max_new_tokens"]) is not int
        or not 8 <= evaluation["max_new_tokens"] <= 256
        or tuple(evaluation["roles"]) != ROLES
        or tuple(evaluation["arms"]) != ARMS
        or set(wrong) != set(ROLES)
        or set(wrong.values()) != set(ROLES)
        or any(wrong[role] == role for role in ROLES)
    ):
        raise ChatGenerationEvalError("eval_route_or_cardinality_contract_drift")
    decoding = _mapping(evaluation["decoding"], "eval_decoding_invalid")
    if dict(decoding) != {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "num_beams": 1,
        "use_cache": True,
    }:
        raise ChatGenerationEvalError("eval_decoding_contract_drift")
    role_proxy = _mapping(evaluation["role_proxy"], "eval_role_proxy_invalid")
    if (
        set(role_proxy)
        != {
            "minimum_reference_char_trigram_f1",
            "maximum_duplicate_4gram_fraction",
        }
        or role_proxy["minimum_reference_char_trigram_f1"] != 0.15
        or role_proxy["maximum_duplicate_4gram_fraction"] != 0.2
    ):
        raise ChatGenerationEvalError("eval_role_proxy_contract_drift")

    runtime = _mapping(config["runtime"], "eval_runtime_invalid")
    if dict(runtime) != {
        "gpu_index": 0,
        "concurrency": 1,
        "base_quantization": "bitsandbytes_int8",
        "base_dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "tf32": True,
        "kv_cache_dtype": "dynamic_model_default",
        "q8_kv_claimed": False,
        "network_allowed": False,
        "provider_allowed": False,
        "protected_body_reads_allowed": False,
        "heldout_reads_allowed": False,
    }:
        raise ChatGenerationEvalError("eval_runtime_contract_drift")
    output = _mapping(config["output"], "eval_output_invalid")
    if (
        _relative_path(output.get("root"), "eval_output_root_invalid")
        != PurePosixPath(
            "artifacts/diagnostics/gemma3_chat_five_expert_qonly_generation_eval_v1"
        )
        or output.get("receipt_name") != "evaluation_receipt.json"
        or output.get("atomic_publish") is not True
        or output.get("replace_existing") is not False
    ):
        raise ChatGenerationEvalError("eval_output_contract_drift")
    claims = _mapping(config["claims"], "eval_claims_invalid")
    if dict(claims) != {
        "diagnostic_only": True,
        "eval_proxy_is_heldout": False,
        "formal": False,
        "quality_validated": False,
        "semantic_style_judgment_claimed": False,
        "wrong_route_is_causal_mechanism_proof": False,
    }:
        raise ChatGenerationEvalError("eval_claims_drift")
    pending = _pending(config)
    consumer_pending = any(item.startswith("consumer.") for item in pending)
    training_pending = any(item.startswith("training_run.") for item in pending)
    expected_identity_state = (
        "pending_producer_and_training_run"
        if consumer_pending
        else (
            "authenticated_consumer_pending_training_run"
            if training_pending
            else "authenticated_inputs"
        )
    )
    if config["identity_state"] != expected_identity_state:
        raise ChatGenerationEvalError("eval_pending_identity_state_drift")
    if not pending and config["identity_state"] != "authenticated_inputs":
        raise ChatGenerationEvalError("eval_bound_identity_state_drift")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    canonical = Path(path)
    if not canonical.is_absolute():
        canonical = ROOT / canonical
    value, snapshot = _load_json_snapshot(canonical, max_bytes=MAX_CONFIG_BYTES)
    sidecar = snapshot_io._read_snapshot(
        canonical.with_name(canonical.name + ".sha256"),
        max_bytes=1024,
    )
    if sidecar.data != (f"{snapshot.sha256}  {canonical.name}\n".encode("ascii")):
        raise ChatGenerationEvalError("eval_config_sidecar_invalid")
    sidecar.assert_unchanged()
    config = dict(_mapping(value, "eval_config_not_mapping"))
    config["_config_sha256"] = snapshot.sha256
    # Runtime-only metadata must not weaken the closed on-disk structure.
    validation_view = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    validate_config(validation_view)
    return config


def _audit_zero() -> dict[str, int]:
    return {
        "dataset_body_reads": 0,
        "protected_body_reads": 0,
        "heldout_reads": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "generated_bodies_persisted": 0,
        "raw_token_ids_persisted": 0,
        "per_record_metrics_persisted": 0,
    }


def build_status(config: Mapping[str, Any]) -> dict[str, Any]:
    validation_view = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    validate_config(validation_view)
    pending = _pending(config)
    consumer_pending = any(item.startswith("consumer.") for item in pending)
    return {
        "schema_version": RECEIPT_VERSION,
        "status": (
            (
                "blocked_waiting_for_authenticated_producer_and_training_run"
                if consumer_pending
                else "blocked_waiting_for_authenticated_training_run"
            )
            if pending
            else "identities_declared_preflight_required"
        ),
        "pending_identities": list(pending),
        "evaluation": {
            "split": "eval_proxy",
            "records_per_role": config["evaluation"]["records_per_role"],
            "arms": list(ARMS),
            "wrong_route_control": True,
            "aggregate_only": True,
        },
        "claims": dict(config["claims"]),
        "audit": _audit_zero(),
    }


def _authenticate_sidecar(path: Path, expected_sha256: str) -> Any:
    snapshot = snapshot_io._read_snapshot(path, max_bytes=MAX_METADATA_BYTES)
    if snapshot.sha256 != expected_sha256:
        raise ChatGenerationEvalError("eval_authenticated_file_sha_mismatch")
    sidecar = snapshot_io._read_snapshot(
        path.with_name(path.name + ".sha256"),
        max_bytes=1024,
    )
    if sidecar.data != f"{expected_sha256}  {path.name}\n".encode("ascii"):
        raise ChatGenerationEvalError("eval_authenticated_sidecar_invalid")
    return snapshot


def _read_self_authenticated_sidecar(path: Path) -> Any:
    snapshot = snapshot_io._read_snapshot(path, max_bytes=MAX_METADATA_BYTES)
    sidecar = snapshot_io._read_snapshot(
        path.with_name(path.name + ".sha256"),
        max_bytes=1024,
    )
    if sidecar.data != f"{snapshot.sha256}  {path.name}\n".encode("ascii"):
        raise ChatGenerationEvalError("eval_authenticated_sidecar_invalid")
    return snapshot


def _authenticate_training_run(
    config: Mapping[str, Any],
    *,
    consumer_config: Mapping[str, Any],
    dataset: Any,
) -> tuple[Mapping[str, Any], Path, dict[str, Path], str]:
    run_config = _mapping(config["training_run"], "eval_training_run_invalid")
    run_id = str(run_config["run_id"])
    run_root = (
        _repo_path(run_config["artifact_root"], "eval_artifact_root_invalid") / run_id
    )
    receipt_path = run_root / str(run_config["receipt_name"])
    expected_receipt_sha = str(run_config["receipt_sha256"])
    receipt_snapshot = _authenticate_sidecar(receipt_path, expected_receipt_sha)
    try:
        receipt = json.loads(receipt_snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChatGenerationEvalError("eval_run_receipt_invalid_json") from exc
    receipt = _mapping(receipt, "eval_run_receipt_not_mapping")
    if (
        receipt.get("schema_version") != run_config["receipt_schema_version"]
        or receipt.get("status") != run_config["required_status"]
        or receipt.get("run_id") != run_id
    ):
        raise ChatGenerationEvalError("eval_run_receipt_header_invalid")
    claims = _mapping(receipt.get("claims"), "eval_run_claims_missing")
    if (
        claims.get("diagnostic_only") is not True
        or claims.get("training_executed") is not True
        or claims.get("training_authorized") is not False
        or claims.get("formal_training_authorized") is not False
        or claims.get("formal") is not False
        or claims.get("eval_proxy_is_heldout") is not False
    ):
        raise ChatGenerationEvalError("eval_run_claims_invalid")
    identity = _mapping(receipt.get("identity"), "eval_run_identity_missing")
    expected_partition = dict(config["consumer"]["partition_sha256"])
    if (
        identity.get("config_sha256") != config["consumer"]["config_sha256"]
        or identity.get("dataset_manifest_sha256") != dataset.manifest_sha256
        or identity.get("partition_sha256") != expected_partition
    ):
        raise ChatGenerationEvalError("eval_run_dataset_identity_mismatch")
    if consumer_config.get("_config_sha256") != config["consumer"]["config_sha256"]:
        raise ChatGenerationEvalError("eval_consumer_runtime_identity_mismatch")

    raw_roles = _sequence(receipt.get("roles"), "eval_run_roles_missing")
    roles: dict[str, Mapping[str, Any]] = {}
    for raw in raw_roles:
        entry = _mapping(raw, "eval_run_role_entry_invalid")
        _exact_keys(
            entry,
            {
                "role",
                "smoke_receipt_sha256",
                "full_receipt_sha256",
                "adapter_artifact_sha256",
            },
            "eval_run_role_entry_fields_drift",
        )
        role = entry.get("role")
        if role not in ROLES or role in roles:
            raise ChatGenerationEvalError("eval_run_role_inventory_invalid")
        roles[str(role)] = entry
    if set(roles) != set(ROLES):
        raise ChatGenerationEvalError("eval_run_role_inventory_invalid")

    adapter_paths: dict[str, Path] = {}
    phase_names = dict(run_config["phase_receipts"])
    for role in ROLES:
        entry = roles[role]
        expected_adapter_hashes = _mapping(
            entry.get("adapter_artifact_sha256"),
            "eval_adapter_hash_inventory_missing",
        )
        if set(expected_adapter_hashes) != set(ADAPTER_FILES):
            raise ChatGenerationEvalError("eval_adapter_hash_inventory_drift")
        adapter = run_root / role / "adapter"
        for filename in ADAPTER_FILES:
            path = adapter / filename
            expected = _identity(
                expected_adapter_hashes.get(filename),
                "eval_adapter_file_sha_invalid",
            )
            if _stream_sha256(path) != expected:
                raise ChatGenerationEvalError("eval_adapter_file_sha_mismatch")
        for phase in ("smoke", "full"):
            receipt_name = phase_names[phase]
            expected = _identity(
                entry.get(f"{phase}_receipt_sha256"),
                "eval_phase_receipt_sha_invalid",
            )
            _authenticate_sidecar(run_root / role / receipt_name, expected)
        adapter_paths[role] = adapter
    receipt_snapshot.assert_unchanged()
    return receipt, run_root, adapter_paths, receipt_snapshot.sha256


def authenticate_inputs(config: Mapping[str, Any]) -> AuthenticatedInputs:
    if _pending(config):
        raise ChatGenerationEvalError("eval_inputs_pending")
    consumer = _mapping(config["consumer"], "eval_consumer_invalid")
    consumer_config_path = _repo_path(
        consumer["config_path"], "eval_consumer_config_path_invalid"
    )
    consumer_impl_path = _repo_path(
        consumer["implementation_path"],
        "eval_consumer_implementation_path_invalid",
    )
    config_snapshot = snapshot_io._read_snapshot(
        consumer_config_path, max_bytes=MAX_CONFIG_BYTES
    )
    implementation_snapshot = snapshot_io._read_snapshot(
        consumer_impl_path, max_bytes=MAX_METADATA_BYTES
    )
    if (
        config_snapshot.sha256 != consumer["config_sha256"]
        or implementation_snapshot.sha256 != consumer["implementation_sha256"]
    ):
        raise ChatGenerationEvalError("eval_consumer_physical_identity_mismatch")
    consumer_config = chat.load_config(consumer_config_path)
    dataset = chat._load_authenticated_dataset(consumer_config)
    if dataset.manifest_sha256 != consumer["dataset_manifest_sha256"] or dict(
        dataset.partition_sha256
    ) != dict(consumer["partition_sha256"]):
        raise ChatGenerationEvalError("eval_dataset_identity_mismatch")
    receipt, run_root, adapter_paths, receipt_sha = _authenticate_training_run(
        config,
        consumer_config=consumer_config,
        dataset=dataset,
    )
    config_snapshot.assert_unchanged()
    implementation_snapshot.assert_unchanged()
    return AuthenticatedInputs(
        consumer_config=consumer_config,
        dataset=dataset,
        run_receipt=receipt,
        run_root=run_root,
        adapter_paths=adapter_paths,
        source_hashes={
            "consumer_config_sha256": config_snapshot.sha256,
            "consumer_implementation_sha256": implementation_snapshot.sha256,
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "partition_sha256": dict(dataset.partition_sha256),
            "training_run_receipt_sha256": receipt_sha,
        },
    )


def bind_training_run(
    config: Mapping[str, Any],
    *,
    training_run_id: str,
    output_path: str | Path | None = None,
) -> Path:
    """Create an immutable, fully bound evaluator config from final artifacts.

    This convenience step remains model-free and GPU-free.  It authenticates
    the already producer-bound training consumer, complete dataset, completed
    run receipt, phase receipts, and adapters before publishing the config.
    """

    if RUN_ID_RE.fullmatch(training_run_id) is None:
        raise ChatGenerationEvalError("eval_training_run_id_invalid")
    consumer_section = _mapping(config["consumer"], "eval_consumer_invalid")
    consumer_config_path = _repo_path(
        consumer_section["config_path"], "eval_consumer_config_path_invalid"
    )
    consumer_impl_path = _repo_path(
        consumer_section["implementation_path"],
        "eval_consumer_implementation_path_invalid",
    )
    consumer_config_snapshot = snapshot_io._read_snapshot(
        consumer_config_path, max_bytes=MAX_CONFIG_BYTES
    )
    consumer_impl_snapshot = snapshot_io._read_snapshot(
        consumer_impl_path, max_bytes=MAX_METADATA_BYTES
    )
    consumer_config = chat.load_config(consumer_config_path)
    dataset = chat._load_authenticated_dataset(consumer_config)

    run_section = _mapping(config["training_run"], "eval_training_run_invalid")
    run_root = (
        _repo_path(run_section["artifact_root"], "eval_artifact_root_invalid")
        / training_run_id
    )
    receipt_path = run_root / str(run_section["receipt_name"])
    receipt_snapshot = _read_self_authenticated_sidecar(receipt_path)
    try:
        receipt = json.loads(receipt_snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChatGenerationEvalError("eval_run_receipt_invalid_json") from exc
    receipt = _mapping(receipt, "eval_run_receipt_not_mapping")
    identity = _mapping(receipt.get("identity"), "eval_run_identity_missing")
    if (
        receipt.get("schema_version") != chat.RUN_RECEIPT_VERSION
        or receipt.get("status") != "passed_diagnostic_training_only"
        or receipt.get("run_id") != training_run_id
        or identity.get("config_sha256") != consumer_config_snapshot.sha256
        or identity.get("dataset_manifest_sha256") != dataset.manifest_sha256
        or identity.get("partition_sha256") != dict(dataset.partition_sha256)
    ):
        raise ChatGenerationEvalError("eval_bind_run_identity_mismatch")

    bound = copy.deepcopy(
        {key: value for key, value in config.items() if not str(key).startswith("_")}
    )
    bound["identity_state"] = "authenticated_inputs"
    bound_consumer = bound["consumer"]
    bound_consumer["config_sha256"] = consumer_config_snapshot.sha256
    bound_consumer["implementation_sha256"] = consumer_impl_snapshot.sha256
    bound_consumer["dataset_manifest_sha256"] = dataset.manifest_sha256
    bound_consumer["partition_sha256"] = dict(dataset.partition_sha256)
    bound_run = bound["training_run"]
    bound_run["run_id"] = training_run_id
    bound_run["receipt_sha256"] = receipt_snapshot.sha256
    validate_config(bound)
    _authenticate_training_run(
        bound,
        consumer_config=consumer_config,
        dataset=dataset,
    )
    consumer_config_snapshot.assert_unchanged()
    consumer_impl_snapshot.assert_unchanged()
    receipt_snapshot.assert_unchanged()

    destination = (
        Path(output_path)
        if output_path is not None
        else ROOT
        / "runs"
        / "gemma3_chat_five_expert_qonly_generation_eval_v1"
        / "bindings"
        / training_run_id
        / "bound_config.json"
    )
    if not destination.is_absolute():
        destination = ROOT / destination
    resolved = destination.resolve(strict=False)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        raise ChatGenerationEvalError("eval_bound_config_output_invalid") from None
    publish_directory = destination.parent
    if os.path.lexists(publish_directory):
        raise FileExistsError(
            f"bound evaluation config directory exists: {publish_directory}"
        )
    publish_directory.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(bound, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = _sha256(raw)
    staging_directory = publish_directory.with_name(
        f".{publish_directory.name}.tmp-{uuid.uuid4().hex}"
    )
    staging_directory.mkdir(exist_ok=False)
    staging = staging_directory / destination.name
    staging_sidecar = staging.with_name(staging.name + ".sha256")
    try:
        with staging.open("xb", buffering=0) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        with staging_sidecar.open("xb", buffering=0) as handle:
            handle.write(f"{digest}  {destination.name}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(staging_directory, publish_directory)
    finally:
        if os.path.lexists(staging_directory):
            shutil.rmtree(staging_directory)
    return destination


def build_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    status = build_status(config)
    if _pending(config):
        return status
    inputs = authenticate_inputs(config)
    selected = [item for item in inputs.dataset.records if item.split == "eval_proxy"]
    counts = Counter(item.role for item in selected)
    if counts != Counter({role: 40 for role in ROLES}):
        raise ChatGenerationEvalError("eval_proxy_role_inventory_invalid")
    return {
        **status,
        "status": "passed_authenticated_inputs_model_gpu_not_started",
        "source_hashes": dict(inputs.source_hashes),
        "training_run_id": config["training_run"]["run_id"],
        "adapters_authenticated": len(inputs.adapter_paths),
        "eval_proxy": {
            "records_authenticated": len(selected),
            "records_per_role_available": dict(sorted(counts.items())),
            "records_per_role_selected": config["evaluation"]["records_per_role"],
        },
        "audit": {
            **_audit_zero(),
            "dataset_body_reads": 2,
        },
    }


def _json_mapping(text: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _visible_char_trigrams(text: str) -> Counter[str]:
    normalized = " ".join(text.casefold().split())
    return Counter(
        normalized[index : index + 3] for index in range(max(0, len(normalized) - 2))
    )


def _counter_f1(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def _char_trigram_f1(generated: str, reference: str) -> float:
    return _counter_f1(
        _visible_char_trigrams(generated),
        _visible_char_trigrams(reference),
    )


def _same_language_proxy(generated: str, reference: str) -> bool:
    def has_cjk(value: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in value)

    return has_cjk(generated) == has_cjk(reference)


def _plain_chat_format_valid(text: str) -> bool:
    if not text.strip() or len(text) > 16_384:
        return False
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        return False
    return _json_mapping(text) is None


def _path_values(
    value: object,
    pattern: re.Pattern[str],
    *,
    prefix: tuple[str, ...] = (),
) -> dict[tuple[str, ...], object]:
    result: dict[tuple[str, ...], object] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = (*prefix, str(key))
            if pattern.search(str(key)):
                result[child_prefix] = child
            result.update(_path_values(child, pattern, prefix=child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(_path_values(child, pattern, prefix=(*prefix, str(index))))
    return result


def _path_consistency(
    generated: Mapping[str, Any],
    reference: Mapping[str, Any],
    pattern: re.Pattern[str],
) -> bool:
    expected = _path_values(reference, pattern)
    observed = _path_values(generated, pattern)
    if not expected:
        return generated == reference
    return all(
        path in observed and observed[path] == value for path, value in expected.items()
    )


def _duplicate_4gram_fraction(token_ids: Sequence[int]) -> float:
    if len(token_ids) < 4:
        return 0.0
    grams = [
        tuple(int(value) for value in token_ids[index : index + 4])
        for index in range(len(token_ids) - 3)
    ]
    return (len(grams) - len(set(grams))) / len(grams)


def score_output(
    *,
    role: str,
    generated_text: str,
    reference_text: str,
    generated_token_ids: Sequence[int],
    eos_terminated: bool,
    generation_seconds: float,
    maximum_duplicate_4gram_fraction: float = 0.2,
    minimum_reference_char_trigram_f1: float = 0.15,
) -> dict[str, Any]:
    """Return body-free scalar metrics for one in-memory generation."""

    if role not in ROLES:
        raise ChatGenerationEvalError("eval_score_role_invalid")
    if generation_seconds <= 0 or not math.isfinite(generation_seconds):
        raise ChatGenerationEvalError("eval_generation_seconds_invalid")
    duplicate = _duplicate_4gram_fraction(generated_token_ids)
    similarity = _char_trigram_f1(generated_text, reference_text)
    generated_json = _json_mapping(generated_text)
    reference_json = _json_mapping(reference_text)
    plain_valid = _plain_chat_format_valid(generated_text)
    language_match = _same_language_proxy(generated_text, reference_text)

    tool_json = role == "tool_call" and generated_json is not None
    tool_schema = bool(
        tool_json
        and reference_json is not None
        and set(generated_json) == set(reference_json)
    )
    tool_evidence = bool(
        tool_schema
        and _path_consistency(generated_json, reference_json, EVIDENCE_KEY_RE)
    )
    review_json = role == "review_audit" and generated_json is not None
    review_schema = bool(
        review_json
        and reference_json is not None
        and set(generated_json) == set(reference_json)
    )
    review_dependency = bool(
        review_schema
        and _path_consistency(
            generated_json,
            reference_json,
            DEPENDENCY_KEY_RE,
        )
    )
    natural_role = role in {"humor", "serious", "angry_style"}
    role_format_valid = (
        plain_valid
        if natural_role
        else tool_schema
        if role == "tool_call"
        else review_schema
    )
    role_appropriate_proxy = bool(
        role_format_valid
        and language_match
        and similarity >= minimum_reference_char_trigram_f1
    )

    identity_probe = (
        IDENTITY_ZH in reference_text
        or IDENTITY_EN.casefold() in reference_text.casefold()
    )
    provider_misattribution_absent = not any(
        provider in generated_text.casefold()
        for provider in IDENTITY_PROVIDER_FORBIDDEN
    )
    persona_consistent = bool(
        identity_probe
        and "air" in generated_text.casefold()
        and provider_misattribution_absent
    )
    return {
        "role_format_valid": role_format_valid,
        "role_appropriate_proxy": role_appropriate_proxy,
        "reference_char_trigram_f1": similarity,
        "language_match_proxy": language_match,
        "tool_json_parse": tool_json,
        "tool_closed_schema": tool_schema,
        "tool_evidence_consistency": tool_evidence,
        "review_json_parse": review_json,
        "review_closed_schema": review_schema,
        "review_dependency_consistency": review_dependency,
        "identity_probe": identity_probe,
        "persona_identity_consistency": persona_consistent,
        "provider_misattribution_absent": (
            provider_misattribution_absent if identity_probe else False
        ),
        "eos_terminated": eos_terminated,
        "generated_tokens": len(generated_token_ids),
        "generation_seconds": generation_seconds,
        "duplicate_4gram_fraction": duplicate,
        "repetition_gate_passed": (duplicate <= maximum_duplicate_4gram_fraction),
    }


@dataclass
class AggregateMetrics:
    records: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    similarity_sum: float = 0.0
    duplicate_sum: float = 0.0
    generated_tokens: int = 0
    generation_seconds: float = 0.0

    BOOLEAN_FIELDS = (
        "role_format_valid",
        "role_appropriate_proxy",
        "language_match_proxy",
        "tool_json_parse",
        "tool_closed_schema",
        "tool_evidence_consistency",
        "review_json_parse",
        "review_closed_schema",
        "review_dependency_consistency",
        "identity_probe",
        "persona_identity_consistency",
        "provider_misattribution_absent",
        "eos_terminated",
        "repetition_gate_passed",
    )

    def add(self, result: Mapping[str, Any]) -> None:
        self.records += 1
        for field_name in self.BOOLEAN_FIELDS:
            self.counts[field_name] += int(bool(result[field_name]))
        self.similarity_sum += float(result["reference_char_trigram_f1"])
        self.duplicate_sum += float(result["duplicate_4gram_fraction"])
        self.generated_tokens += int(result["generated_tokens"])
        self.generation_seconds += float(result["generation_seconds"])

    def report(self) -> dict[str, Any]:
        if not self.records:
            return {"status": "unavailable", "reason": "no_records"}

        def rate(field_name: str, denominator: int | None = None) -> float | None:
            divisor = self.records if denominator is None else denominator
            return None if divisor == 0 else round(self.counts[field_name] / divisor, 6)

        identity_count = self.counts["identity_probe"]
        return {
            "records": self.records,
            "role_output_format_valid_rate": rate("role_format_valid"),
            "role_appropriate_reference_proxy_rate": rate("role_appropriate_proxy"),
            "language_match_proxy_rate": rate("language_match_proxy"),
            "mean_reference_char_trigram_f1": round(
                self.similarity_sum / self.records, 6
            ),
            "tool_call_schema_valid_rate_all_records": rate("tool_closed_schema"),
            "tool_call_evidence_consistency_rate_all_records": rate(
                "tool_evidence_consistency"
            ),
            "review_audit_schema_valid_rate_all_records": rate("review_closed_schema"),
            "review_dependency_consistency_rate_all_records": rate(
                "review_dependency_consistency"
            ),
            "identity_probe_records": identity_count,
            "persona_identity_consistency_rate": rate(
                "persona_identity_consistency", identity_count
            ),
            "provider_misattribution_absence_rate": rate(
                "provider_misattribution_absent", identity_count
            ),
            "eos_termination_rate": rate("eos_terminated"),
            "repetition_gate_pass_rate": rate("repetition_gate_passed"),
            "mean_duplicate_4gram_fraction": round(
                self.duplicate_sum / self.records, 6
            ),
            "generated_tokens": self.generated_tokens,
            "generation_seconds": round(self.generation_seconds, 6),
            "tokens_per_second": round(
                self.generated_tokens / max(self.generation_seconds, 1e-9),
                6,
            ),
        }


def _primary_rate(report: Mapping[str, Any], role: str) -> float | None:
    if role == "tool_call":
        return report.get("tool_call_schema_valid_rate_all_records")
    if role == "review_audit":
        return report.get("review_audit_schema_valid_rate_all_records")
    return report.get("role_appropriate_reference_proxy_rate")


def build_route_deltas(
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    by_role: dict[str, Any] = {}
    correct_values: list[float] = []
    wrong_values: list[float] = []
    base_values: list[float] = []
    for role in ROLES:
        correct = _primary_rate(reports[role]["correct_adapter"], role)
        wrong = _primary_rate(reports[role]["wrong_route"], role)
        base = _primary_rate(reports[role]["base"], role)
        if None in {correct, wrong, base}:
            by_role[role] = {"status": "unavailable"}
            continue
        correct_float = float(correct)
        wrong_float = float(wrong)
        base_float = float(base)
        correct_values.append(correct_float)
        wrong_values.append(wrong_float)
        base_values.append(base_float)
        by_role[role] = {
            "primary_metric": (
                "tool_call_schema_valid_rate"
                if role == "tool_call"
                else "review_audit_schema_valid_rate"
                if role == "review_audit"
                else "role_appropriate_reference_proxy_rate"
            ),
            "correct_minus_wrong_route": round(correct_float - wrong_float, 6),
            "correct_minus_base": round(correct_float - base_float, 6),
        }
    return {
        "by_role": by_role,
        "macro_correct_minus_wrong_route": (
            None
            if not correct_values
            else round(
                sum(correct_values) / len(correct_values)
                - sum(wrong_values) / len(wrong_values),
                6,
            )
        ),
        "macro_correct_minus_base": (
            None
            if not correct_values
            else round(
                sum(correct_values) / len(correct_values)
                - sum(base_values) / len(base_values),
                6,
            )
        ),
        "interpretation": (
            "controlled eval_proxy routing proxy only; not causal mechanism, "
            "formal quality, or heldout generalization evidence"
        ),
    }


def _gpu_sample() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,"
            "memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 8:
        raise ChatGenerationEvalError("eval_gpu_inventory_parse_failed")
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "total_mib": int(fields[3]),
        "used_mib": int(fields[4]),
        "free_mib": int(fields[5]),
        "utilization_percent": int(fields[6]),
        "temperature_c": int(fields[7]),
    }


@contextmanager
def _gpu_lock(run_id: str) -> Any:
    if any(os.path.lexists(path) for path in CONFLICTING_LOCKS):
        raise ChatGenerationEvalError("eval_conflicting_handoff_lock_exists")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        LOCK_PATH,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
    )
    payload = {
        "schema_version": LOCK_VERSION,
        "run_id": run_id,
        "pid": os.getpid(),
        "gpu_index": 0,
        "concurrency": 1,
        "diagnostic_only": True,
    }
    # ``os.open`` defaults to text mode on Windows unless ``O_BINARY`` is set;
    # that mode silently expands the terminal LF to CRLF after ``digest`` was
    # computed.  Write canonical bytes in binary mode so the exact lock bytes
    # remain stable and any later change still fails closed.
    raw = _canonical_json(payload) + b"\n"
    digest = _sha256(raw)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if any(os.path.lexists(path) for path in CONFLICTING_LOCKS):
            raise ChatGenerationEvalError("eval_conflicting_handoff_lock_appeared")
        yield {**payload, "content_sha256": digest}
        if _sha256(LOCK_PATH.read_bytes()) != digest:
            raise ChatGenerationEvalError("eval_gpu_lock_changed")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(LOCK_PATH) and _sha256(LOCK_PATH.read_bytes()) == digest:
            LOCK_PATH.unlink()


def _generation_prefix(
    processor: Any,
    example: Any,
) -> tuple[int, ...]:
    serialized = chat._serialize_chat_example(processor, example)
    indices = [index for index, label in enumerate(serialized.labels) if label != -100]
    if not indices or indices[0] <= 0:
        raise ChatGenerationEvalError("eval_generation_prefix_invalid")
    return tuple(int(value) for value in serialized.input_ids[: indices[0]])


def _generate_one(
    model: Any,
    processor: Any,
    example: Any,
    *,
    arm: str,
    adapter_name: str | None,
    max_new_tokens: int,
    torch: Any,
) -> tuple[dict[str, Any], int]:
    prefix = _generation_prefix(processor, example)
    input_ids = torch.tensor([prefix], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    if arm == "base":
        context = model.disable_adapter()
    else:
        if adapter_name is None:
            raise ChatGenerationEvalError("eval_adapter_name_required")
        model.set_adapter(adapter_name)
        context = None
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        if context is None:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=1,
                pad_token_id=0,
                bos_token_id=2,
                use_cache=True,
            )
        else:
            with context:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=1,
                    pad_token_id=0,
                    bos_token_id=2,
                    use_cache=True,
                )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = [
        int(value) for value in output[0, input_ids.shape[1] :].detach().cpu().tolist()
    ]
    eos = bool(new_ids and new_ids[-1] == 1)
    visible = list(new_ids)
    while visible and visible[-1] in {0, 1, 106}:
        visible.pop()
    generated_text = processor.decode(visible).strip()
    score = score_output(
        role=example.role,
        generated_text=generated_text,
        reference_text=example.target,
        generated_token_ids=new_ids,
        eos_terminated=eos,
        generation_seconds=elapsed,
    )
    generated_count = len(new_ids)
    del (
        prefix,
        input_ids,
        attention_mask,
        output,
        new_ids,
        visible,
        generated_text,
    )
    return score, generated_count


def _select_eval_records(
    dataset: Any,
    *,
    limit_per_role: int,
) -> dict[str, tuple[Any, ...]]:
    selected: dict[str, tuple[Any, ...]] = {}
    for role in ROLES:
        records = tuple(
            sorted(
                (
                    item
                    for item in dataset.records
                    if item.split == "eval_proxy" and item.role == role
                ),
                key=lambda item: item.record_id,
            )[:limit_per_role]
        )
        if len(records) != limit_per_role:
            raise ChatGenerationEvalError("eval_selection_count_invalid")
        selected[role] = records
    return selected


def execute(config: Mapping[str, Any], *, run_id: str | None = None) -> Path:
    inputs = authenticate_inputs(config)
    evaluation = config["evaluation"]
    eval_run_id = run_id or uuid.uuid4().hex
    if RUN_ID_RE.fullmatch(eval_run_id) is None:
        raise ChatGenerationEvalError("eval_output_run_id_invalid")
    output_root = _repo_path(config["output"]["root"], "eval_output_root_invalid")
    destination = output_root / eval_run_id
    if os.path.lexists(destination):
        raise FileExistsError(f"evaluation output already exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{eval_run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    selected = _select_eval_records(
        inputs.dataset,
        limit_per_role=int(evaluation["records_per_role"]),
    )
    model_contract = chat._model_file_contract(inputs.consumer_config)
    model_root = chat._model_root(inputs.consumer_config)
    model_hash_before = {
        name: _stream_sha256(model_root / name) for name in chat.MODEL_FILES
    }
    if any(
        model_hash_before[name] != model_contract[name][1] for name in chat.MODEL_FILES
    ):
        raise ChatGenerationEvalError("eval_base_model_identity_mismatch")
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_started = time.perf_counter()
    try:
        with _gpu_lock(eval_run_id) as lock:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["WANDB_DISABLED"] = "true"
            import bitsandbytes as bnb
            import peft
            import torch
            import transformers
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gpu_before = _gpu_sample()
            quantization = inputs.consumer_config["quantization"]
            base = AutoModelForCausalLM.from_pretrained(
                model_root,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                device_map={"": 0},
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=float(quantization["llm_int8_threshold"]),
                    llm_int8_skip_modules=list(quantization["llm_int8_skip_modules"]),
                    llm_int8_enable_fp32_cpu_offload=False,
                    llm_int8_has_fp16_weight=False,
                ),
            )
            overlay = inputs.consumer_config["model"]["runtime_special_token_overlay"]
            base.config.pad_token_id = int(overlay["pad_token_id"])
            base.config.eos_token_id = int(overlay["eos_token_id"])
            base.config.bos_token_id = int(overlay["bos_token_id"])
            base.config.use_cache = True
            chat._validate_q8_base_modules(base, bnb=bnb, torch=torch)
            chat._prepare_frozen_q8_base(base, torch=torch, bnb=bnb)
            q8_before = chat._q8_quant_state_inventory(base, bnb=bnb, torch=torch)
            first_role = ROLES[0]
            model = PeftModel.from_pretrained(
                base,
                inputs.adapter_paths[first_role],
                adapter_name=first_role,
                is_trainable=False,
            )
            for role in ROLES[1:]:
                model.load_adapter(
                    inputs.adapter_paths[role],
                    adapter_name=role,
                    is_trainable=False,
                )
            model.eval()
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise ChatGenerationEvalError("eval_trainable_parameter_detected")
            processor = binding.load_sentencepiece(model_root / "tokenizer.model")
            aggregates = {
                role: {arm: AggregateMetrics() for arm in ARMS} for role in ROLES
            }
            wrong_map = dict(evaluation["wrong_route_adapter"])
            for role in ROLES:
                for example in selected[role]:
                    for arm in ARMS:
                        adapter = (
                            role
                            if arm == "correct_adapter"
                            else wrong_map[role]
                            if arm == "wrong_route"
                            else None
                        )
                        result, _generated_count = _generate_one(
                            model,
                            processor,
                            example,
                            arm=arm,
                            adapter_name=adapter,
                            max_new_tokens=int(evaluation["max_new_tokens"]),
                            torch=torch,
                        )
                        aggregates[role][arm].add(result)
            q8_after = chat._q8_quant_state_inventory(model, bnb=bnb, torch=torch)
            if q8_before != q8_after:
                raise ChatGenerationEvalError("eval_q8_scb_identity_drift")
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            gpu_after = _gpu_sample()
            reports = {
                role: {arm: aggregates[role][arm].report() for arm in ARMS}
                for role in ROLES
            }
            del model, base, processor
            torch.cuda.empty_cache()
            runtime_versions = {
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
                "peft": str(peft.__version__),
                "bitsandbytes": str(bnb.__version__),
            }
        model_hash_after = {
            name: _stream_sha256(model_root / name) for name in chat.MODEL_FILES
        }
        if model_hash_after != model_hash_before:
            raise ChatGenerationEvalError("eval_base_model_file_hash_drift")
        final_inputs = authenticate_inputs(config)
        if dict(final_inputs.source_hashes) != dict(inputs.source_hashes) or set(
            final_inputs.adapter_paths
        ) != set(inputs.adapter_paths):
            raise ChatGenerationEvalError("eval_final_source_recheck_drift")
        del final_inputs
        receipt = {
            "schema_version": RECEIPT_VERSION,
            "status": "passed_diagnostic_eval_proxy_generation_only",
            "run_id": eval_run_id,
            "source_training_run_id": config["training_run"]["run_id"],
            "source_hashes": dict(inputs.source_hashes),
            "evaluation": {
                "split": "eval_proxy",
                "records_per_role": evaluation["records_per_role"],
                "total_unique_records": (
                    int(evaluation["records_per_role"]) * len(ROLES)
                ),
                "generation_arms": list(ARMS),
                "total_generations": (
                    int(evaluation["records_per_role"]) * len(ROLES) * len(ARMS)
                ),
                "wrong_route_adapter": dict(wrong_map),
                "max_new_tokens": evaluation["max_new_tokens"],
                "decoding": dict(evaluation["decoding"]),
            },
            "metrics": reports,
            "routing_deltas": build_route_deltas(reports),
            "runtime": {
                "versions": runtime_versions,
                "lock": lock,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "torch_peak_allocated_mib": round(peak_allocated / (1024 * 1024), 3),
                "torch_peak_reserved_mib": round(peak_reserved / (1024 * 1024), 3),
                "q8_scb_identity_equal_before_after": True,
                "q8_scb_inventory_sha256": q8_before["sha256"],
                "model_file_hash_equal_before_after": True,
                "all_source_identities_equal_after_generation": True,
                "wall_seconds": round(time.perf_counter() - wall_started, 6),
                "started_utc": started_utc,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "kv_cache_dtype": "dynamic_model_default",
                "q8_kv_claimed": False,
            },
            "claims": dict(config["claims"]),
            "audit": {
                **_audit_zero(),
                "dataset_body_reads": 4,
                "model_loads": 1,
                "gpu_requests": 1,
            },
        }
        _assert_aggregate_only_receipt(receipt)
        raw = (
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        filename = str(config["output"]["receipt_name"])
        digest = _sha256(raw)
        (staging / filename).write_bytes(raw)
        (staging / f"{filename}.sha256").write_bytes(
            f"{digest}  {filename}\n".encode("ascii")
        )
        os.rename(staging, destination)
        return destination / filename
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging)


def _assert_aggregate_only_receipt(value: object) -> None:
    forbidden_exact = {
        "prompt",
        "prompts",
        "target",
        "targets",
        "generated_text",
        "generated",
        "token_ids",
        "input_ids",
        "labels",
        "record_id",
        "records_detail",
        "per_record",
    }

    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).casefold() in forbidden_exact:
                    raise ChatGenerationEvalError(
                        "eval_receipt_body_or_per_record_field_forbidden"
                    )
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate-only base/correct/wrong-route generation evaluator for "
            "the Gemma 3 chat five-expert Q-only diagnostic."
        )
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--bind-training-run", metavar="TRAINING_RUN_ID")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--bound-config-output")
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.bind_training_run:
            bound = bind_training_run(
                config,
                training_run_id=args.bind_training_run,
                output_path=args.bound_config_output,
            )
            print(
                json.dumps(
                    {
                        "status": "passed_bound_config_published",
                        "bound_config": str(bound.relative_to(ROOT)).replace("\\", "/"),
                        "model_loads": 0,
                        "gpu_requests": 0,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.status:
            result = build_status(config)
        elif args.preflight:
            result = build_preflight(config)
        else:
            receipt = execute(config, run_id=args.run_id)
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "receipt": str(receipt.relative_to(ROOT)).replace("\\", "/"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return (
            0
            if result["status"]
            in {
                "identities_declared_preflight_required",
                "passed_authenticated_inputs_model_gpu_not_started",
            }
            else 2
        )
    except Exception as exc:  # noqa: BLE001 - redact every CLI-boundary failure
        candidate = str(exc)
        is_public_chat_error = (
            isinstance(exc, ChatGenerationEvalError) and candidate in PUBLIC_ERROR_CODES
        )
        error_code = candidate if is_public_chat_error else "runtime_error_redacted"
        error_type = (
            "ChatGenerationEvalError" if is_public_chat_error else "RuntimeError"
        )
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "error_type": error_type,
                    "error_code": error_code,
                    "model_loads": 0,
                    "gpu_requests": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1


__all__ = [
    "ADAPTER_FILES",
    "ARMS",
    "CONFIG_PATH",
    "CONFIG_VERSION",
    "ChatGenerationEvalError",
    "AggregateMetrics",
    "RECEIPT_VERSION",
    "ROLES",
    "_assert_aggregate_only_receipt",
    "_char_trigram_f1",
    "_pending",
    "bind_training_run",
    "build_preflight",
    "build_route_deltas",
    "build_status",
    "load_config",
    "main",
    "score_output",
    "validate_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
