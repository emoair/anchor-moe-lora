"""Single-process GLM launch controller for the unbalanced-v2 campaign.

The frozen batch adapter intentionally keeps its historical command surface.
This additive controller owns one :class:`RuntimeSecretSlots` instance for the
entire exact1 -> bounded15 -> bulk lifecycle and authenticates the event WAL
before every replay.  It does not support cross-process resume.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import time
from typing import Any, Awaitable, BinaryIO, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from . import gemma3_chat_unbalanced_v2_batch as batch


SCHEMA_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2."
    "teacher-alignment.single-process-controller.config.v1"
)
STATUS_SCHEMA_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2."
    "teacher-alignment.single-process-controller.status.v1"
)
WAL_CHECKPOINT_SCHEMA_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2."
    "teacher-alignment.controller-wal-entry.v1"
)
DEFAULT_CONFIG_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_"
    "teacher_alignment_single_process_controller.v1.json"
)
CONTROLLER_IMPLEMENTATION_PATH = Path(__file__).resolve()
_HASH_KEYS = (
    "source_identity_sha256",
    "manifest_sha256",
    "approval_sha256",
    "partition_set_sha256",
    "readonly_test_identity_sha256",
)
_BANNED_SECRET_ENV_NAMES = frozenset(
    {
        "ARK_API_KEY",
        "ARK_CODING_API_KEY",
        "VOLCENGINE_API_KEY",
        "CONTROLLER_CREDENTIAL",
        "OPENAI_API_KEY",
    }
)
_BANNED_SECRET_ARG_PREFIXES = (
    "--api-key",
    "--credential",
    "--credential-value",
    "--secret",
    "--token",
)


@dataclass(frozen=True)
class ProfileSpec:
    stage: str
    phase: str
    profile_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ControllerConfig:
    path: Path
    physical_sha256: str
    schema_path: Path
    schema_sha256: str
    controller_implementation_path: Path
    controller_implementation_sha256: str
    batch_implementation_path: Path
    batch_implementation_sha256: str
    teacher_implementation_path: Path
    teacher_implementation_sha256: str
    consumer_release_binding: dict[str, Any]
    controller_runtime: dict[str, Any]
    wal: dict[str, Any]
    common_identity: dict[str, str]
    profiles: tuple[ProfileSpec, ...]
    lifecycle: dict[str, Any]
    fault_policy: dict[str, Any]
    security: dict[str, Any]
    claims: dict[str, Any]

    @property
    def profile_map(self) -> dict[str, ProfileSpec]:
        return {item.profile_id: item for item in self.profiles}


@dataclass(frozen=True)
class BoundProfiles:
    controller: ControllerConfig
    values: dict[str, batch.AdapterConfig]
    controller_implementation_sha256: str


ProfileLoader = Callable[[Path], batch.AdapterConfig]
SourceLoader = Callable[[batch.AdapterConfig], batch.SourceInventory]
HeldoutValidator = Callable[[batch.AdapterConfig], Mapping[str, Any]]
StageExecutor = Callable[
    [
        batch.AdapterConfig,
        batch.SourceInventory,
        batch.RuntimeSecretSlots,
        str,
        Mapping[str, str],
        Mapping[str, Any],
    ],
    Awaitable[batch.RunReport],
]
FallbackVerifier = Callable[
    [
        batch.AdapterConfig,
        batch.SourceInventory,
        batch.RuntimeSecretSlots,
        Mapping[str, Any],
    ],
    str,
]
ConsumerValidator = Callable[[ControllerConfig], None]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_stable_read(path, reason="controller_file_snapshot_drift"))


def _strict_json(raw: bytes, *, reason: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise batch.AdapterError(f"{reason}_duplicate_key")
            value[key] = item
        return value

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicate)
    except batch.AdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise batch.AdapterError(reason) from error


def _stable_read(path: Path, *, reason: str) -> bytes:
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise batch.AdapterError(reason) from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(raw) != before.st_size:
        raise batch.AdapterError(reason)
    return raw


def _reject_reparse_chain(path: Path, *, stop: Path | None = None) -> None:
    resolved_stop = stop.resolve() if stop is not None else None
    current = path
    while True:
        try:
            value = current.lstat()
        except FileNotFoundError:
            value = None
        except OSError as error:
            raise batch.AdapterError("controller_physical_path_stat_failed") from error
        if value is not None:
            attributes = int(getattr(value, "st_file_attributes", 0))
            if stat.S_ISLNK(value.st_mode) or bool(attributes & 0x400):
                raise batch.AdapterError("controller_reparse_path_forbidden")
        if resolved_stop is not None and current.resolve() == resolved_stop:
            return
        parent = current.parent
        if parent == current:
            return
        current = parent


def _resolve_repo_file(raw_path: object, *, reason: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
        raise batch.AdapterError(reason)
    candidate = Path(raw_path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (batch.REPO_ROOT / candidate).resolve()
    )
    root = batch.REPO_ROOT.resolve()
    if resolved == root or root not in resolved.parents or not resolved.is_file():
        raise batch.AdapterError(reason)
    _reject_reparse_chain(resolved, stop=root)
    return resolved


def load_controller_config(path: Path = DEFAULT_CONFIG_PATH) -> ControllerConfig:
    config_path = Path(path).resolve()
    _reject_reparse_chain(config_path, stop=batch.REPO_ROOT)
    raw = _stable_read(config_path, reason="controller_config_snapshot_drift")
    value = _strict_json(raw, reason="controller_config_json_invalid")
    if not isinstance(value, dict):
        raise batch.AdapterError("controller_config_invalid")
    schema_path = _resolve_repo_file(
        value.get("controller_schema_path"),
        reason="controller_schema_path_invalid",
    )
    schema_raw = _stable_read(schema_path, reason="controller_schema_snapshot_drift")
    schema_sha256 = _sha256_bytes(schema_raw)
    if value.get("controller_schema_sha256") != schema_sha256:
        raise batch.AdapterError("controller_schema_hash_drift")
    schema = _strict_json(schema_raw, reason="controller_schema_json_invalid")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except (SchemaError, ValidationError) as error:
        raise batch.AdapterError("controller_config_schema_invalid") from error
    controller_implementation_path = _resolve_repo_file(
        value["controller_implementation_path"],
        reason="controller_implementation_path_invalid",
    )
    if controller_implementation_path != CONTROLLER_IMPLEMENTATION_PATH:
        raise batch.AdapterError("controller_implementation_path_drift")
    controller_implementation_raw = _stable_read(
        controller_implementation_path,
        reason="controller_implementation_snapshot_drift",
    )
    controller_implementation_sha256 = _sha256_bytes(controller_implementation_raw)
    if value["controller_implementation_sha256"] != controller_implementation_sha256:
        raise batch.AdapterError("controller_implementation_hash_drift")
    batch_path = _resolve_repo_file(
        value["batch_implementation_path"],
        reason="controller_batch_path_invalid",
    )
    batch_sha256 = _sha256_file(batch_path)
    if value["batch_implementation_sha256"] != batch_sha256:
        raise batch.AdapterError("controller_batch_implementation_drift")
    teacher_path = _resolve_repo_file(
        value["teacher_implementation_path"],
        reason="controller_teacher_path_invalid",
    )
    teacher_raw = _stable_read(
        teacher_path,
        reason="controller_teacher_implementation_snapshot_drift",
    )
    teacher_sha256 = _sha256_bytes(teacher_raw)
    if value["teacher_implementation_sha256"] != teacher_sha256:
        raise batch.AdapterError("controller_teacher_implementation_drift")
    profiles = tuple(
        ProfileSpec(
            stage=str(item["stage"]),
            phase=str(item["phase"]),
            profile_id=str(item["profile_id"]),
            path=_resolve_repo_file(
                item["path"], reason="controller_profile_path_invalid"
            ),
            sha256=str(item["sha256"]),
        )
        for item in value["profiles"]
    )
    if _sha256_bytes(
        _stable_read(config_path, reason="controller_config_snapshot_drift")
    ) != _sha256_bytes(raw):
        raise batch.AdapterError("controller_config_snapshot_drift")
    if _sha256_file(schema_path) != schema_sha256:
        raise batch.AdapterError("controller_schema_snapshot_drift")
    if (
        _stable_read(
            controller_implementation_path,
            reason="controller_implementation_terminal_snapshot_drift",
        )
        != controller_implementation_raw
    ):
        raise batch.AdapterError("controller_implementation_terminal_snapshot_drift")
    if _sha256_file(batch_path) != batch_sha256:
        raise batch.AdapterError("controller_batch_implementation_drift")
    if (
        _stable_read(
            teacher_path,
            reason="controller_teacher_implementation_terminal_snapshot_drift",
        )
        != teacher_raw
    ):
        raise batch.AdapterError(
            "controller_teacher_implementation_terminal_snapshot_drift"
        )
    return ControllerConfig(
        path=config_path,
        physical_sha256=_sha256_bytes(raw),
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        controller_implementation_path=controller_implementation_path,
        controller_implementation_sha256=controller_implementation_sha256,
        batch_implementation_path=batch_path,
        batch_implementation_sha256=batch_sha256,
        teacher_implementation_path=teacher_path,
        teacher_implementation_sha256=teacher_sha256,
        consumer_release_binding=dict(value["consumer_release_binding"]),
        controller_runtime=dict(value["controller_runtime"]),
        wal=dict(value["wal"]),
        common_identity=dict(value["common_identity"]),
        profiles=profiles,
        lifecycle=dict(value["lifecycle"]),
        fault_policy=dict(value["fault_policy"]),
        security=dict(value["security"]),
        claims=dict(value["claims"]),
    )


def _validate_profile(
    controller: ControllerConfig,
    spec: ProfileSpec,
    value: batch.AdapterConfig,
) -> None:
    common = controller.common_identity
    expected_root = (batch.REPO_ROOT / common["output_root"]).resolve()
    if (
        value.profile_id != spec.profile_id
        or value.physical_sha256 != spec.sha256
        or value.path.resolve() != spec.path
        or value.source_binding.get("manifest_sha256")
        != common["source_manifest_sha256"]
        or value.source_binding.get("approval_sha256")
        != common["source_approval_sha256"]
        or value.campaign_sha256 != common["campaign_sha256"]
        or value.model_binding_sha256 != common["model_binding_sha256"]
        or value.implementation_sha256 != controller.batch_implementation_sha256
        or value.contract_hashes.get("teacher_implementation_path")
        != controller.teacher_implementation_sha256
        or value.output_root != expected_root
        or spec.phase not in value.limits.allowed_phases
    ):
        raise batch.AdapterError("controller_profile_identity_drift")


def bind_profiles(
    controller: ControllerConfig,
    *,
    profile_loader: ProfileLoader = batch.load_config,
) -> BoundProfiles:
    values: dict[str, batch.AdapterConfig] = {}
    source_binding: dict[str, Any] | None = None
    contract_hashes: dict[str, str] | None = None
    for spec in controller.profiles:
        profile = profile_loader(spec.path)
        _validate_profile(controller, spec, profile)
        if source_binding is None:
            source_binding = profile.source_binding
            contract_hashes = profile.contract_hashes
        elif (
            profile.source_binding != source_binding
            or profile.contract_hashes != contract_hashes
        ):
            raise batch.AdapterError("controller_cross_profile_binding_drift")
        values[spec.profile_id] = profile
    if set(values) != {
        "smoke_exact1",
        "bounded_small_c1",
        "bulk_c30",
        "bulk_c16",
    }:
        raise batch.AdapterError("controller_profile_set_drift")
    return BoundProfiles(
        controller=controller,
        values=values,
        controller_implementation_sha256=(controller.controller_implementation_sha256),
    )


def _recheck_bound_artifacts(bound: BoundProfiles) -> None:
    controller = bound.controller
    if (
        _sha256_file(controller.path) != controller.physical_sha256
        or _sha256_file(controller.schema_path) != controller.schema_sha256
        or _sha256_file(controller.controller_implementation_path)
        != controller.controller_implementation_sha256
        or _sha256_file(controller.batch_implementation_path)
        != controller.batch_implementation_sha256
        or _sha256_file(controller.teacher_implementation_path)
        != controller.teacher_implementation_sha256
        or controller.controller_implementation_sha256
        != bound.controller_implementation_sha256
    ):
        raise batch.AdapterError("controller_artifact_identity_drift")


def _inventory_identity(inventory: batch.SourceInventory) -> tuple[str, ...]:
    return tuple(str(getattr(inventory, name)) for name in _HASH_KEYS)


def _reject_environment_credentials(environ: Mapping[str, str]) -> None:
    names = {str(name).upper() for name in environ}
    if names & _BANNED_SECRET_ENV_NAMES:
        raise batch.AdapterError("controller_environment_credential_forbidden")


def _reject_argv_credentials(argv: Sequence[str]) -> None:
    for raw in argv:
        lowered = raw.casefold()
        banned_flag = any(
            lowered == prefix or lowered.startswith(f"{prefix}=")
            for prefix in _BANNED_SECRET_ARG_PREFIXES
        )
        if banned_flag or batch.SECRET_LIKE_RE.search(raw):
            raise batch.AdapterError("controller_argv_credential_forbidden")


def _validate_anonymous_os_channel(stream: BinaryIO) -> None:
    try:
        descriptor = stream.fileno()
        mode = os.fstat(descriptor).st_mode
    except (AttributeError, OSError, ValueError) as error:
        raise batch.AdapterError("credential_stdin_os_channel_required") from error
    if stream.isatty() or not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)):
        raise batch.AdapterError("credential_stdin_os_channel_required")


def _consumer_canonical_bytes(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _consumer_domain_sha256(domain: str, value: Any) -> str:
    return _sha256_bytes(
        domain.encode("ascii") + b"\0" + _consumer_canonical_bytes(value)
    )


def _normalized_consumer_binding_sha256(
    *,
    binding: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_sha256: str,
) -> str:
    identity_version = (
        "anchor.gemma3-chat-unbalanced-v2-consumer-derived-asset-identity.v1"
    )
    shard_version = (
        "anchor.gemma3-chat-unbalanced-v2-consumer-derived-shard-inventory.v1"
    )
    common = {
        "identity_derivation_schema_version": identity_version,
        "physical_receipt_sha256": receipt_sha256,
        "artifact_version": receipt["artifact_version"],
        "tree_digest_domain": receipt["tree_digest_domain"],
        "tree_digest_sha256": receipt["tree_digest_sha256"],
        "logical_identity": receipt["logical_identity"],
    }
    schema_contract = {
        "schema_version": shard_version,
        "identity_origin": "consumer_derived_from_authenticated_sharded_v1_receipt",
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
    shard_schema_sha256 = _consumer_domain_sha256(
        "anchor.consumer-derived-shard-inventory-schema.v1",
        schema_contract,
    )
    binding_training_shards = binding.get("training_shards")
    if not isinstance(binding_training_shards, Mapping):
        raise batch.AdapterError("consumer_release_training_shards_invalid")
    physical_shards = binding_training_shards.get("shards")
    if not isinstance(physical_shards, list):
        raise batch.AdapterError("consumer_release_training_shards_invalid")
    main_chat_shard_sha256 = _consumer_domain_sha256(
        "anchor.consumer-derived-main-chat-shard-inventory.v1",
        {
            **common,
            "training_shards": receipt["training_shards"],
            "physical_training_shard_anchors": physical_shards,
        },
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
    training_counts = {
        "humor": 240,
        "serious": 240,
        "angry_style": 240,
        "review_audit": 1360,
        "tool_call": 1360,
        "planner_router": 80,
    }
    evaluation_counts = {
        "planner_router_eval": 20,
        "tool_comparison_eval": 400,
        "planner_comparison_eval": 240,
        "identity_probe": 50,
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
            shard_inventory_sha256 = main_chat_shard_sha256
            shard_count = 2
        else:
            shard_inventory_sha256 = _consumer_domain_sha256(
                "anchor.consumer-derived-tree-asset-shard-inventory.v1",
                preimage,
            )
            shard_count = 1
        value = {
            "records": records,
            "asset_namespace": namespaces[name],
            "identity_origin": (
                "consumer_derived_from_authenticated_sharded_v1_receipt"
            ),
            "identity_derivation_schema_version": identity_version,
            "identity_preimage_sha256": _consumer_domain_sha256(
                "anchor.consumer-derived-asset-preimage.v1",
                preimage,
            ),
            "logical_dataset_sha256": _consumer_domain_sha256(
                "anchor.consumer-derived-logical-dataset.v1",
                preimage,
            ),
            "shard_inventory_schema_version": shard_version,
            "shard_inventory_schema_sha256": shard_schema_sha256,
            "shard_inventory_sha256": shard_inventory_sha256,
            "shard_count": shard_count,
            "training_eligible": training_eligible,
            "record_order_sha256": _consumer_domain_sha256(
                "anchor.consumer-derived-record-order.v1",
                preimage,
            ),
            "target_projection_sha256": _consumer_domain_sha256(
                "anchor.consumer-derived-target-projection.v1",
                preimage,
            ),
        }
        if training_eligible:
            value["shard_order_contract"] = (
                "continuous_bundle_boundary_v1"
                if name in main_chat_names
                else "consumer_derived_tree_asset_namespace_v1"
            )
        return value

    normalized = {
        **receipt,
        "physical_receipt_sha256": receipt_sha256,
        "training_assets": {
            name: derive(name, count, True) for name, count in training_counts.items()
        },
        "evaluation_assets": {
            name: derive(name, count, False)
            for name, count in evaluation_counts.items()
        },
    }
    return _sha256_bytes(_consumer_canonical_bytes(normalized))


def _git_environment() -> dict[str, str]:
    value = {
        name: item
        for name, item in os.environ.items()
        if not name.upper().startswith("GIT_")
    }
    value["GIT_NO_REPLACE_OBJECTS"] = "1"
    return value


def _git_output(
    repo: Path,
    arguments: Sequence[str],
    *,
    text: bool,
) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=text,
            timeout=10,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise batch.AdapterError("consumer_release_git_identity_invalid") from error
    return result.stdout


def _git_blob_at_commit(repo: Path, commit: str, relative_path: str) -> bytes:
    tree_value = str(
        _git_output(
            repo,
            ["ls-tree", commit, "--", relative_path],
            text=True,
        )
    ).strip()
    prefix = "100644 blob "
    if not tree_value.startswith(prefix) or "\t" not in tree_value:
        raise batch.AdapterError("consumer_release_git_blob_invalid")
    metadata, observed_path = tree_value.split("\t", 1)
    parts = metadata.split(" ")
    if len(parts) != 3 or observed_path != relative_path:
        raise batch.AdapterError("consumer_release_git_blob_invalid")
    raw = _git_output(
        repo,
        ["cat-file", "blob", f"{commit}:{relative_path}"],
        text=False,
    )
    if not isinstance(raw, bytes) or _sha256_bytes(raw) == "":
        raise batch.AdapterError("consumer_release_git_blob_invalid")
    return raw


def validate_consumer_release_binding(controller: ControllerConfig) -> None:
    binding = controller.consumer_release_binding
    if binding["state"] != "ready":
        raise batch.AdapterError("consumer_release_identity_pending")
    repo = Path(str(binding["repo_path"])).resolve()
    if not repo.is_dir():
        raise batch.AdapterError("consumer_release_repo_invalid")
    _reject_reparse_chain(repo)
    branch = str(binding["branch"])
    commit = str(binding["commit_sha256"])
    commands = {
        "head": ["rev-parse", "HEAD"],
        "branch": ["branch", "--show-current"],
        "upstream": ["rev-parse", f"refs/remotes/origin/{branch}"],
        "replace_refs": ["for-each-ref", "--format=%(refname)", "refs/replace"],
        "grafts_path": ["rev-parse", "--git-path", "info/grafts"],
    }
    observed: dict[str, str] = {}
    for name, arguments in commands.items():
        observed[name] = str(_git_output(repo, arguments, text=True)).strip()
    grafts_path = (repo / observed["grafts_path"]).resolve()
    if observed["replace_refs"] or (
        grafts_path.is_file() and grafts_path.stat().st_size > 0
    ):
        raise batch.AdapterError("consumer_release_git_rewrite_forbidden")
    if (
        observed["head"] != commit
        or observed["upstream"] != commit
        or observed["branch"] != branch
    ):
        raise batch.AdapterError("consumer_release_git_identity_drift")
    relative_paths = {
        "binding": str(binding["binding_path"]),
        "receipt": str(binding["preflight_receipt_path"]),
        "sidecar": str(binding["preflight_receipt_sidecar_path"]),
        "schema": str(binding["receipt_schema_path"]),
    }
    paths = {
        name: (repo / relative_path).resolve()
        for name, relative_path in relative_paths.items()
    }
    for path in paths.values():
        if repo not in path.parents or not path.is_file():
            raise batch.AdapterError("consumer_release_receipt_path_invalid")
        _reject_reparse_chain(path, stop=repo)
    raw = {
        name: _stable_read(
            path,
            reason=f"consumer_release_{name}_snapshot_drift",
        )
        for name, path in paths.items()
    }
    expected_hashes = {
        "binding": binding["binding_sha256"],
        "receipt": binding["preflight_receipt_sha256"],
        "sidecar": binding["preflight_receipt_sidecar_sha256"],
        "schema": binding["receipt_schema_sha256"],
    }
    if any(
        _sha256_bytes(raw[name]) != expected
        for name, expected in expected_hashes.items()
    ):
        raise batch.AdapterError("consumer_release_receipt_hash_drift")
    for name, relative_path in relative_paths.items():
        if _git_blob_at_commit(repo, commit, relative_path) != raw[name]:
            raise batch.AdapterError("consumer_release_git_blob_worktree_drift")
    binding_document = _strict_json(
        raw["binding"],
        reason="consumer_release_binding_invalid",
    )
    receipt = _strict_json(
        raw["receipt"],
        reason="consumer_release_receipt_invalid",
    )
    schema = _strict_json(
        raw["schema"],
        reason="consumer_release_schema_invalid",
    )
    if not isinstance(binding_document, dict) or not isinstance(receipt, dict):
        raise batch.AdapterError("consumer_release_document_invalid")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(binding_document)
        Draft202012Validator(schema).validate(receipt)
    except (SchemaError, ValidationError) as error:
        raise batch.AdapterError("consumer_release_receipt_schema_invalid") from error
    expected_sidecar = (
        f"{binding['preflight_receipt_sha256']}  "
        f"{Path(relative_paths['receipt']).name}\n"
    ).encode("ascii")
    binding_contract_sha256 = _sha256_bytes(_consumer_canonical_bytes(binding_document))
    normalized_sha256 = _normalized_consumer_binding_sha256(
        binding=binding_document,
        receipt=receipt,
        receipt_sha256=binding["preflight_receipt_sha256"],
    )
    if (
        raw["sidecar"] != expected_sidecar
        or receipt.get("binding_contract_sha256") != binding_contract_sha256
        or normalized_sha256 != binding["normalized_binding_sha256"]
    ):
        raise batch.AdapterError("consumer_release_semantic_identity_drift")
    for name, path in paths.items():
        terminal_raw = _stable_read(
            path,
            reason=f"consumer_release_{name}_terminal_snapshot_drift",
        )
        if (
            terminal_raw != raw[name]
            or _git_blob_at_commit(repo, commit, relative_paths[name]) != terminal_raw
        ):
            raise batch.AdapterError("consumer_release_terminal_identity_drift")


def _parse_jsonl_snapshot(raw: bytes, *, reason: str) -> list[dict[str, Any]]:
    if raw and not raw.endswith(b"\n"):
        raise batch.AdapterError(reason)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        value = _strict_json(line, reason=reason)
        if not isinstance(value, dict):
            raise batch.AdapterError(reason)
        event_id = value.get("event_id")
        if not isinstance(event_id, str) or event_id in seen:
            raise batch.AdapterError(reason)
        seen.add(event_id)
        rows.append(value)
    return rows


def _validate_event_transitions(events: Sequence[Mapping[str, Any]]) -> None:
    latest: dict[str, str] = {}
    allowed_next = {
        None: {"reserved"},
        "reserved": {"dispatching"},
        "dispatching": {"retryable", "uncertain", "committed", "rejected"},
        "retryable": {"reserved"},
        "uncertain": set(),
        "committed": set(),
        "rejected": set(),
    }
    fixed_reason = {
        "reserved": "budget_reserved",
        "dispatching": "provider_dispatch",
        "committed": "validated",
    }
    for event in events:
        key = event.get("idempotency_key")
        state_value = event.get("state")
        reason = event.get("reason_code")
        if (
            not isinstance(key, str)
            or state_value not in allowed_next
            or not isinstance(reason, str)
            or batch.SAFE_ENUM_RE.fullmatch(reason) is None
            or state_value not in allowed_next[latest.get(key)]
        ):
            raise batch.AdapterError("controller_wal_event_semantics_drift")
        if state_value in fixed_reason and reason != fixed_reason[state_value]:
            raise batch.AdapterError("controller_wal_event_reason_drift")
        if state_value == "retryable" and reason not in {
            "provider_rate_limit",
            "provider_quota_exhausted",
            "provider_instability_reconciled",
        }:
            raise batch.AdapterError("controller_wal_event_reason_drift")
        latest[key] = str(state_value)


def _stable_jsonl(path: Path, *, id_field: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = _stable_read(path, reason="controller_terminal_log_snapshot_drift")
    if raw and not raw.endswith(b"\n"):
        raise batch.AdapterError("controller_terminal_log_partial")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        value = _strict_json(line, reason="controller_terminal_log_invalid")
        if not isinstance(value, dict):
            raise batch.AdapterError("controller_terminal_log_invalid")
        identity = value.get(id_field)
        if not isinstance(identity, str) or identity in seen:
            raise batch.AdapterError("controller_terminal_log_identity_drift")
        seen.add(identity)
        rows.append(value)
    return rows


class AuthenticatedBatchState(batch.BatchState):
    """Batch state guarded by an immutable, chained runtime-HMAC WAL."""

    def __init__(
        self,
        config: batch.AdapterConfig,
        *,
        hmac_key: bytes,
        controller_binding: Mapping[str, Any],
    ) -> None:
        super().__init__(config, hmac_key=hmac_key)
        self.controller_binding = dict(controller_binding)
        self.wal_dir = self.automation_dir / "controller_wal_v1"
        self.genesis_path = self.wal_dir / "genesis.json"
        self.process_lock_path = self.wal_dir / "process_lock.json"
        self._controller_wal_lock = asyncio.Lock()
        self._inflight_group_id: str | None = None

    def _signed_value(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **payload,
            "hmac_sha256": hmac.new(
                self.hmac_key or b"",
                _canonical_bytes(payload),
                hashlib.sha256,
            ).hexdigest(),
        }

    def _verify_signed_value(
        self, value: Mapping[str, Any], *, reason: str
    ) -> dict[str, Any]:
        payload = dict(value)
        signature = payload.pop("hmac_sha256", None)
        expected = hmac.new(
            self.hmac_key or b"",
            _canonical_bytes(payload),
            hashlib.sha256,
        ).hexdigest()
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature, expected
        ):
            raise batch.AdapterError(reason)
        return payload

    def _directory_identity(self) -> dict[str, Any]:
        return self._path_identity(self.root, expected_kind="directory")

    def _path_identity(
        self,
        path: Path,
        *,
        expected_kind: str,
    ) -> dict[str, Any]:
        _reject_reparse_chain(path, stop=(batch.REPO_ROOT / "data").resolve())
        try:
            value = path.lstat()
        except OSError as error:
            raise batch.AdapterError("controller_runtime_path_stat_failed") from error
        observed_kind = (
            "directory"
            if stat.S_ISDIR(value.st_mode)
            else "file"
            if stat.S_ISREG(value.st_mode)
            else "other"
        )
        if observed_kind != expected_kind:
            raise batch.AdapterError("controller_runtime_path_kind_drift")
        return {
            "canonical_path": path.resolve().as_posix(),
            "kind": observed_kind,
            "device": int(value.st_dev),
            "inode": int(value.st_ino),
            "file_attributes": int(getattr(value, "st_file_attributes", 0)),
        }

    def _dataset_marker(self) -> dict[str, Any]:
        return {
            "schema_version": batch.STATUS_SCHEMA_VERSION,
            "dataset_kind": batch.DATASET_KIND,
            "namespace": batch.NAMESPACE,
            "source_identity_sha256": self.controller_binding["source_identity_sha256"],
            "protected_test_identity_sha256": self.controller_binding[
                "protected_test_identity_sha256"
            ],
            "planner_dag_contract_sha256": batch._hash_object(
                batch.PLANNER_DAG_CONTRACT
            ),
            "campaign_sha256": self.config.campaign_sha256,
        }

    @staticmethod
    def _exclusive_empty_file(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise batch.AdapterError("controller_runtime_layout_collision") from error
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        batch._fsync_parent(path.parent)

    def _prepare_runtime_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _reject_reparse_chain(self.root, stop=(batch.REPO_ROOT / "data").resolve())
        for path in (
            self.alignment_dir,
            self.automation_dir,
            self.groups_dir,
            self.wal_dir,
        ):
            try:
                path.mkdir()
            except FileExistsError as error:
                raise batch.AdapterError(
                    "controller_runtime_layout_collision"
                ) from error
            _reject_reparse_chain(path, stop=(batch.REPO_ROOT / "data").resolve())
        for path in (
            self.records_path,
            self.rejections_path,
            self.receipts_path,
            self.events_path,
            self.phase_receipts_path,
        ):
            self._exclusive_empty_file(path)
        batch._exclusive_write(self.dataset_marker_path, self._dataset_marker())

    def _runtime_layout_identity(self) -> dict[str, Any]:
        return {
            "directories": {
                name: self._path_identity(path, expected_kind="directory")
                for name, path in (
                    ("output_root", self.root),
                    ("alignment", self.alignment_dir),
                    ("automation", self.automation_dir),
                    ("groups", self.groups_dir),
                    ("wal", self.wal_dir),
                )
            },
            "append_only_files": {
                name: self._path_identity(path, expected_kind="file")
                for name, path in (
                    ("records", self.records_path),
                    ("rejections", self.rejections_path),
                    ("receipts", self.receipts_path),
                    ("events", self.events_path),
                    ("phase_receipts", self.phase_receipts_path),
                )
            },
            "dataset_marker": self._path_identity(
                self.dataset_marker_path,
                expected_kind="file",
            ),
        }

    def _verify_dynamic_runtime_paths(self) -> None:
        for path in (self.genesis_path, self.process_lock_path):
            self._path_identity(path, expected_kind="file")
        if self.status_path.exists() or self.status_path.is_symlink():
            self._path_identity(self.status_path, expected_kind="file")
        kill_switch_path = self.config.kill_switch_file
        if kill_switch_path.exists() or kill_switch_path.is_symlink():
            self._path_identity(kill_switch_path, expected_kind="file")
        try:
            wal_children = list(self.wal_dir.iterdir())
            group_children = list(self.groups_dir.iterdir())
        except OSError as error:
            raise batch.AdapterError(
                "controller_runtime_directory_scan_failed"
            ) from error
        for path in wal_children:
            name = path.name
            if name not in {"genesis.json", "process_lock.json"} and not (
                len(name) == 21 and name[:16].isdigit() and name[16:] == ".json"
            ):
                raise batch.AdapterError("controller_wal_unexpected_path")
            self._path_identity(path, expected_kind="file")
        for path in group_children:
            if path.suffix != ".json" or not path.stem:
                raise batch.AdapterError("controller_group_unexpected_path")
            self._path_identity(path, expected_kind="file")

    def _genesis_payload(self) -> dict[str, Any]:
        return {
            "schema_version": WAL_CHECKPOINT_SCHEMA_VERSION,
            "entry_kind": "GENESIS",
            "controller_run_id": self.controller_binding["controller_run_id"],
            "process_id": os.getpid(),
            "directory_identity": self._directory_identity(),
            "campaign_sha256": self.config.campaign_sha256,
            "source_identity_sha256": self.controller_binding["source_identity_sha256"],
            "manifest_sha256": self.controller_binding["manifest_sha256"],
            "model_binding_sha256": self.config.model_binding_sha256,
            "batch_implementation_sha256": self.config.implementation_sha256,
            "teacher_implementation_sha256": self.controller_binding[
                "teacher_implementation_sha256"
            ],
            "controller_config_sha256": self.controller_binding[
                "controller_config_sha256"
            ],
            "controller_implementation_sha256": self.controller_binding[
                "controller_implementation_sha256"
            ],
            "accepted_profiles": self.controller_binding["accepted_profiles"],
            "runtime_layout": self._runtime_layout_identity(),
            "cross_process_resume": False,
            "content_retained": False,
        }

    def _ensure_genesis(self) -> None:
        if not self.genesis_path.exists():
            self._prepare_runtime_layout()
        else:
            _reject_reparse_chain(
                self.genesis_path,
                stop=(batch.REPO_ROOT / "data").resolve(),
            )
        genesis = self._signed_value(self._genesis_payload())
        lock_payload = {
            "schema_version": WAL_CHECKPOINT_SCHEMA_VERSION,
            "entry_kind": "PROCESS_LOCK",
            "controller_run_id": self.controller_binding["controller_run_id"],
            "process_id": os.getpid(),
            "directory_identity": self._directory_identity(),
            "cross_process_resume": False,
            "content_retained": False,
        }
        lock = self._signed_value(lock_payload)
        if not self.genesis_path.exists():
            batch._exclusive_write(self.genesis_path, genesis)
            batch._exclusive_write(self.process_lock_path, lock)
        else:
            for path, expected, reason in (
                (
                    self.genesis_path,
                    genesis,
                    "controller_wal_genesis_identity_drift",
                ),
                (
                    self.process_lock_path,
                    lock,
                    "controller_process_lock_identity_drift",
                ),
            ):
                raw = _stable_read(path, reason=reason)
                observed = _strict_json(raw, reason=reason)
                if not isinstance(observed, dict):
                    raise batch.AdapterError(reason)
                self._verify_signed_value(observed, reason=reason)
                if observed != expected:
                    raise batch.AdapterError(reason)
        self._verify_dynamic_runtime_paths()

    def _entry_paths(self) -> list[Path]:
        if not self.wal_dir.exists():
            return []
        return sorted(self.wal_dir.glob("[0-9][0-9][0-9][0-9]*.json"))

    def _read_entries(
        self, *, allow_inflight_prepare: bool = False
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, str | None],
    ]:
        self._ensure_genesis()
        genesis_sha256 = _sha256_file(self.genesis_path)
        previous = genesis_sha256
        job_heads: dict[str, str | None] = {}
        entries: list[dict[str, Any]] = []
        authoritative_events: list[dict[str, Any]] = []
        authoritative_phase_receipts: list[dict[str, Any]] = []
        unmatched_prepares: dict[str, str] = {}
        for sequence, path in enumerate(self._entry_paths()):
            if path.name != f"{sequence:016d}.json":
                raise batch.AdapterError("controller_wal_sequence_gap")
            raw = _stable_read(path, reason="controller_wal_entry_snapshot_drift")
            value = _strict_json(raw, reason="controller_wal_entry_invalid")
            if not isinstance(value, dict):
                raise batch.AdapterError("controller_wal_entry_invalid")
            payload = self._verify_signed_value(
                value, reason="controller_wal_entry_hmac_drift"
            )
            if (
                payload.get("schema_version") != WAL_CHECKPOINT_SCHEMA_VERSION
                or payload.get("sequence") != sequence
                or payload.get("controller_run_id")
                != self.controller_binding["controller_run_id"]
                or payload.get("process_id") != os.getpid()
                or payload.get("previous_entry_sha256") != previous
                or payload.get("campaign_sha256") != self.config.campaign_sha256
                or payload.get("source_identity_sha256")
                != self.controller_binding["source_identity_sha256"]
                or payload.get("manifest_sha256")
                != self.controller_binding["manifest_sha256"]
                or payload.get("controller_config_sha256")
                != self.controller_binding["controller_config_sha256"]
                or payload.get("teacher_implementation_sha256")
                != self.controller_binding["teacher_implementation_sha256"]
            ):
                raise batch.AdapterError("controller_wal_entry_identity_drift")
            profile_id = payload.get("active_profile_id")
            accepted_profiles = self.controller_binding["accepted_profiles"]
            if not isinstance(profile_id, str) or accepted_profiles.get(
                profile_id
            ) != payload.get("active_profile_sha256"):
                raise batch.AdapterError("controller_wal_profile_identity_drift")
            keys = payload.get("job_keys")
            previous_by_job = payload.get("previous_job_entry_sha256")
            if (
                not isinstance(keys, list)
                or len(keys) != len(set(keys))
                or not all(isinstance(item, str) for item in keys)
                or not isinstance(previous_by_job, dict)
                or set(previous_by_job) != set(keys)
                or any(previous_by_job[key] != job_heads.get(key) for key in keys)
            ):
                raise batch.AdapterError("controller_wal_job_chain_drift")
            entry_sha256 = _sha256_bytes(raw)
            for key in keys:
                job_heads[key] = entry_sha256
            kind = payload.get("entry_kind")
            body = payload.get("body")
            if not isinstance(body, dict):
                raise batch.AdapterError("controller_wal_entry_body_invalid")
            if kind == "EVENT":
                event = body.get("event")
                if not isinstance(event, dict) or keys != [
                    event.get("idempotency_key")
                ]:
                    raise batch.AdapterError("controller_wal_event_binding_drift")
                authoritative_events.append(event)
            elif kind == "GROUP_PREPARE":
                group_id = body.get("group_id")
                if not isinstance(group_id, str) or group_id in unmatched_prepares:
                    raise batch.AdapterError("controller_group_prepare_drift")
                unmatched_prepares[group_id] = entry_sha256
            elif kind == "GROUP_TERMINAL":
                group_id = body.get("group_id")
                if not isinstance(group_id, str) or unmatched_prepares.get(
                    group_id
                ) != body.get("prepare_entry_sha256"):
                    raise batch.AdapterError("controller_group_terminal_drift")
                unmatched_prepares.pop(group_id)
                terminal_events = body.get("terminal_events")
                if not isinstance(terminal_events, list) or not all(
                    isinstance(event, dict) for event in terminal_events
                ):
                    raise batch.AdapterError("controller_group_terminal_drift")
                authoritative_events.extend(terminal_events)
            elif kind == "PHASE_RECEIPT":
                receipt = body.get("receipt")
                if not isinstance(receipt, dict):
                    raise batch.AdapterError("controller_phase_receipt_wal_drift")
                authoritative_phase_receipts.append(receipt)
            elif kind in {"FALLBACK_COOLDOWN", "COOLDOWN_COMPLETE"}:
                if keys:
                    raise batch.AdapterError("controller_cooldown_wal_binding_drift")
            else:
                raise batch.AdapterError("controller_wal_entry_kind_invalid")
            entries.append(value)
            previous = entry_sha256
        if unmatched_prepares and not allow_inflight_prepare:
            raise batch.AdapterError("controller_group_terminal_missing")
        return (
            entries,
            authoritative_events,
            authoritative_phase_receipts,
            job_heads,
        )

    def _append_wal_entry(
        self,
        *,
        entry_kind: str,
        job_keys: Sequence[str],
        body: Mapping[str, Any],
        allow_inflight_prepare: bool = False,
    ) -> str:
        entries, _, _, job_heads = self._read_entries(
            allow_inflight_prepare=allow_inflight_prepare
        )
        previous = (
            _sha256_file(self.genesis_path)
            if not entries
            else _sha256_file(self._entry_paths()[-1])
        )
        keys = list(job_keys)
        payload = {
            "schema_version": WAL_CHECKPOINT_SCHEMA_VERSION,
            "sequence": len(entries),
            "entry_kind": entry_kind,
            "controller_run_id": self.controller_binding["controller_run_id"],
            "process_id": os.getpid(),
            "previous_entry_sha256": previous,
            "job_keys": keys,
            "previous_job_entry_sha256": {key: job_heads.get(key) for key in keys},
            "active_profile_id": self.config.profile_id,
            "active_profile_sha256": self.config.physical_sha256,
            "campaign_sha256": self.config.campaign_sha256,
            "source_identity_sha256": self.controller_binding["source_identity_sha256"],
            "manifest_sha256": self.controller_binding["manifest_sha256"],
            "controller_config_sha256": self.controller_binding[
                "controller_config_sha256"
            ],
            "teacher_implementation_sha256": self.controller_binding[
                "teacher_implementation_sha256"
            ],
            "body": dict(body),
            "content_retained": False,
        }
        path = self.wal_dir / f"{len(entries):016d}.json"
        batch._exclusive_write(path, self._signed_value(payload))
        return _sha256_file(path)

    def _terminal_cross_bind(
        self,
        events: Sequence[Mapping[str, Any]],
        entries: Sequence[Mapping[str, Any]],
    ) -> None:
        terminal = [
            event for event in events if event.get("state") in {"committed", "rejected"}
        ]
        receipts = _stable_jsonl(self.receipts_path, id_field="id")
        rejections = _stable_jsonl(self.rejections_path, id_field="id")
        records = _stable_jsonl(self.records_path, id_field="idempotency_key")
        receipt_by_key: dict[str, Mapping[str, Any]] = {}
        for receipt in [*receipts, *rejections]:
            batch._verify_receipt(receipt, self.hmac_key or b"")
            key = receipt.get("idempotency_key")
            if not isinstance(key, str) or key in receipt_by_key:
                raise batch.AdapterError("controller_terminal_receipt_binding_drift")
            receipt_by_key[key] = receipt
        record_keys = {str(item["idempotency_key"]) for item in records}
        terminal_bodies = [
            entry.get("body")
            for entry in entries
            if entry.get("entry_kind") == "GROUP_TERMINAL"
        ]
        for body in terminal_bodies:
            if not isinstance(body, dict):
                raise batch.AdapterError("controller_group_terminal_drift")
            group_id = body.get("group_id")
            if not isinstance(group_id, str):
                raise batch.AdapterError("controller_group_terminal_drift")
            group_path = self.groups_dir / f"{group_id}.json"
            group_raw = _stable_read(
                group_path, reason="controller_group_snapshot_drift"
            )
            group = _strict_json(group_raw, reason="controller_group_invalid")
            if not isinstance(group, dict):
                raise batch.AdapterError("controller_group_invalid")
            payload = dict(group)
            logical_sha256 = payload.pop("group_sha256", None)
            if (
                _sha256_bytes(group_raw) != body.get("group_physical_sha256")
                or logical_sha256 != body.get("group_sha256")
                or logical_sha256 != batch._hash_object(payload)
                or group.get("events") != body.get("terminal_events")
            ):
                raise batch.AdapterError("controller_group_terminal_drift")
        for event in terminal:
            key = str(event["idempotency_key"])
            receipt = receipt_by_key.get(key)
            expected_outcome = (
                "succeeded" if event["state"] == "committed" else "rejected"
            )
            bound_by_group = any(
                isinstance(body, dict)
                and event in body.get("terminal_events", [])
                and {
                    "idempotency_key": key,
                    "receipt_id": str(receipt.get("id")) if receipt else "",
                    "receipt_sha256": batch._hash_object(receipt) if receipt else "",
                }
                in body.get("receipt_bindings", [])
                for body in terminal_bodies
            )
            if (
                receipt is None
                or receipt.get("outcome") != expected_outcome
                or receipt.get("reason_code") != event.get("reason_code")
                or receipt.get("attempts") != event.get("attempts")
                or not bound_by_group
                or (event["state"] == "committed") != (key in record_keys)
            ):
                raise batch.AdapterError(
                    "controller_terminal_event_cross_binding_drift"
                )

    def authenticate_authority(self, *, allow_inflight_prepare: bool = False) -> None:
        _reject_reparse_chain(self.root, stop=(batch.REPO_ROOT / "data").resolve())
        entries, authoritative_events, phase_receipts, _ = self._read_entries(
            allow_inflight_prepare=allow_inflight_prepare
        )
        aggregate_raw = (
            _stable_read(self.events_path, reason="controller_events_snapshot_drift")
            if self.events_path.exists()
            else b""
        )
        aggregate_events = _parse_jsonl_snapshot(
            aggregate_raw, reason="controller_events_projection_invalid"
        )
        if aggregate_events != authoritative_events:
            raise batch.AdapterError("controller_events_projection_drift")
        _validate_event_transitions(authoritative_events)
        physical_phase_receipts = _stable_jsonl(self.phase_receipts_path, id_field="id")
        if physical_phase_receipts != phase_receipts:
            raise batch.AdapterError("controller_phase_receipt_projection_drift")
        self._terminal_cross_bind(authoritative_events, entries)

    def initialize(self, inventory: batch.SourceInventory) -> None:
        self._ensure_genesis()
        self.authenticate_authority()
        super().initialize(inventory)
        self.authenticate_authority()

    async def reserve_budget(
        self,
        job: batch.AlignmentJob,
        replay: batch.ReplayState,
        phase: str,
    ) -> dict[str, int]:
        self.authenticate_authority()
        reservation = await super().reserve_budget(job, replay, phase)
        self.authenticate_authority()
        return reservation

    async def append_event(self, **kwargs: Any) -> dict[str, Any]:
        async with self._controller_wal_lock:
            self.authenticate_authority()
            event = await super().append_event(**kwargs)
            self._append_wal_entry(
                entry_kind="EVENT",
                job_keys=[str(event["idempotency_key"])],
                body={"event": event},
            )
            self.authenticate_authority()
            return event

    def commit_group(
        self,
        pending: Sequence[batch.PendingCommit],
        *,
        inventory: batch.SourceInventory,
        phase: str,
    ) -> None:
        if not pending:
            return
        self.authenticate_authority()
        keys = [item.job.idempotency_key for item in pending]
        group_id = batch._hash_object({"phase": phase, "keys": keys})
        prepare_sha256 = self._append_wal_entry(
            entry_kind="GROUP_PREPARE",
            job_keys=keys,
            body={
                "group_id": group_id,
                "phase": phase,
                "pending": [
                    {
                        "idempotency_key": item.job.idempotency_key,
                        "outcome": item.receipt.get("outcome"),
                        "receipt_sha256": batch._hash_object(item.receipt),
                        "output_record_sha256": (
                            batch._hash_object(item.output_record)
                            if item.output_record is not None
                            else None
                        ),
                    }
                    for item in pending
                ],
            },
        )
        self._inflight_group_id = group_id
        try:
            super().commit_group(pending, inventory=inventory, phase=phase)
            group_path = self.groups_dir / f"{group_id}.json"
            group_raw = _stable_read(
                group_path, reason="controller_group_snapshot_drift"
            )
            group = _strict_json(group_raw, reason="controller_group_invalid")
            if not isinstance(group, dict) or group.get("group_id") != group_id:
                raise batch.AdapterError("controller_group_identity_drift")
            payload = dict(group)
            group_sha256 = payload.pop("group_sha256", None)
            if group_sha256 != batch._hash_object(payload):
                raise batch.AdapterError("controller_group_hash_drift")
            receipt_rows = [*group["receipts"], *group["rejections"]]
            self._append_wal_entry(
                entry_kind="GROUP_TERMINAL",
                job_keys=keys,
                body={
                    "group_id": group_id,
                    "prepare_entry_sha256": prepare_sha256,
                    "group_physical_sha256": _sha256_bytes(group_raw),
                    "group_sha256": group_sha256,
                    "terminal_events": group["events"],
                    "receipt_bindings": [
                        {
                            "idempotency_key": str(item["idempotency_key"]),
                            "receipt_id": str(item["id"]),
                            "receipt_sha256": batch._hash_object(item),
                        }
                        for item in receipt_rows
                    ],
                },
                allow_inflight_prepare=True,
            )
        finally:
            self._inflight_group_id = None
        self.authenticate_authority()

    def recover_group_commits(self) -> int:
        self._ensure_genesis()
        if self._inflight_group_id is not None:
            self._read_entries(allow_inflight_prepare=True)
            return super().recover_group_commits()
        entries, _, _, _ = self._read_entries()
        signed_groups = {
            str(entry["body"]["group_id"])
            for entry in entries
            if entry.get("entry_kind") == "GROUP_TERMINAL"
        }
        physical_groups = {path.stem for path in self.groups_dir.glob("*.json")}
        if physical_groups != signed_groups:
            raise batch.AdapterError("controller_unsigned_group_forbidden")
        recovered = super().recover_group_commits()
        self.authenticate_authority()
        return recovered

    def replay(self) -> batch.ReplayState:
        self.authenticate_authority()
        value = super().replay()
        if value.uncertain:
            # The base runner also blocks; keep the controller authority explicit.
            pass
        return value

    def write_status(self, **kwargs: Any) -> dict[str, Any]:
        self.authenticate_authority()
        value = super().write_status(**kwargs)
        self.authenticate_authority()
        return value

    def append_phase_receipt(self, **kwargs: Any) -> dict[str, Any]:
        self.authenticate_authority()
        receipt = super().append_phase_receipt(**kwargs)
        _, _, signed, _ = self._read_entries()
        if receipt not in signed:
            self._append_wal_entry(
                entry_kind="PHASE_RECEIPT",
                job_keys=[],
                body={"receipt": receipt},
            )
        self.authenticate_authority()
        return receipt

    def verify_phase_prerequisites(
        self, inventory: batch.SourceInventory, phases: Sequence[str]
    ) -> None:
        self.authenticate_authority()
        super().verify_phase_prerequisites(inventory, phases)
        self.authenticate_authority()

    def verify_fallback_trigger(self) -> None:
        self.authenticate_authority()
        if self.config.profile_id == "bulk_c16":
            replay = super().replay()
            matches = [
                event
                for events in replay.events_by_job.values()
                for event in events
                if event.get("binding", {}).get("profile_id") == "bulk_c30"
                and event.get("state") == "retryable"
                and event.get("reason_code") == "provider_rate_limit"
            ]
            if replay.uncertain or not matches:
                raise batch.AdapterError("bulk_c16_authenticated_rate_limit_missing")
        super().verify_fallback_trigger()
        self.authenticate_authority()

    def _rate_limit_event_inventory(
        self,
        entries: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        paths = self._entry_paths()
        latest_event_by_job: dict[str, Mapping[str, Any]] = {}
        rate_limit_rows: list[dict[str, str]] = []
        event_ids: set[str] = set()
        for index, entry in enumerate(entries):
            body = entry.get("body")
            event = body.get("event") if isinstance(body, Mapping) else None
            if entry.get("entry_kind") != "EVENT" or not isinstance(event, Mapping):
                continue
            event_id = event.get("event_id")
            job_key = event.get("idempotency_key")
            if (
                not isinstance(event_id, str)
                or not isinstance(job_key, str)
                or event_id in event_ids
            ):
                raise batch.AdapterError("controller_cooldown_event_identity_drift")
            event_ids.add(event_id)
            latest_event_by_job[job_key] = event
            if (
                event.get("state") == "retryable"
                and event.get("reason_code") == "provider_rate_limit"
                and event.get("binding", {}).get("profile_id") == "bulk_c30"
            ):
                rate_limit_rows.append(
                    {
                        "event_id": event_id,
                        "idempotency_key": job_key,
                        "event_sha256": batch._hash_object(event),
                        "entry_sha256": _sha256_file(paths[index]),
                    }
                )
        rate_limit_job_keys = [row["idempotency_key"] for row in rate_limit_rows]
        if len(rate_limit_job_keys) != len(set(rate_limit_job_keys)):
            raise batch.AdapterError("controller_cooldown_job_identity_drift")
        current_rows = [
            row
            for row in rate_limit_rows
            if latest_event_by_job.get(row["idempotency_key"], {}).get("event_id")
            == row["event_id"]
        ]
        return sorted(
            current_rows,
            key=lambda row: (row["event_sha256"], row["entry_sha256"]),
        )

    def record_fallback_cooldown(
        self,
        *,
        fallback_profile: batch.AdapterConfig,
        retry_after_seconds: float | None,
        now_seconds: float,
    ) -> dict[str, Any]:
        self.authenticate_authority()
        if (
            self.config.profile_id != "bulk_c30"
            or fallback_profile.profile_id != "bulk_c16"
            or not math.isfinite(now_seconds)
        ):
            raise batch.AdapterError("controller_cooldown_binding_invalid")
        delay = max(
            float(self.config.limits.cooldown_seconds),
            float(retry_after_seconds or 0.0),
        )
        if not math.isfinite(delay) or delay < 0:
            raise batch.AdapterError("controller_cooldown_binding_invalid")
        entries, _, _, _ = self._read_entries()
        paths = self._entry_paths()
        replay = super().replay()
        inventory = self._rate_limit_event_inventory(entries)
        if not inventory or replay.uncertain:
            raise batch.AdapterError("bulk_c16_authenticated_rate_limit_missing")
        inventory_sha256 = batch._hash_object(
            {
                "schema_version": (
                    "anchor.gemma3-chat-unbalanced-v2.rate-limit-event-inventory.v1"
                ),
                "events": inventory,
            }
        )
        source_chain_tip_sha256 = (
            _sha256_file(paths[-1]) if paths else _sha256_file(self.genesis_path)
        )
        not_before = int(math.ceil(now_seconds + delay))
        body = {
            "lease_id": batch._hash_object(
                {
                    "controller_run_id": self.controller_binding["controller_run_id"],
                    "retryable_event_inventory_sha256": inventory_sha256,
                    "retryable_event_count": len(inventory),
                    "source_wal_chain_tip_sha256": source_chain_tip_sha256,
                    "not_before_unix_seconds": not_before,
                    "fallback_profile_sha256": fallback_profile.physical_sha256,
                }
            ),
            "from_profile": "bulk_c30",
            "from_profile_sha256": self.config.physical_sha256,
            "to_profile": "bulk_c16",
            "to_profile_sha256": fallback_profile.physical_sha256,
            "reason_code": "provider_rate_limit",
            "retryable_event_inventory_sha256": inventory_sha256,
            "retryable_event_count": len(inventory),
            "source_wal_chain_tip_sha256": source_chain_tip_sha256,
            "not_before_unix_seconds": not_before,
            "uncertain_dispatches": 0,
            "status_json_authoritative": False,
        }
        lease_entry_sha256 = self._append_wal_entry(
            entry_kind="FALLBACK_COOLDOWN",
            job_keys=[],
            body=body,
        )
        self.authenticate_authority()
        return {**body, "lease_entry_sha256": lease_entry_sha256}

    def complete_fallback_cooldown(
        self,
        lease: Mapping[str, Any],
        *,
        now_seconds: float,
    ) -> dict[str, Any]:
        self.authenticate_authority()
        if not math.isfinite(now_seconds) or now_seconds < int(
            lease["not_before_unix_seconds"]
        ):
            raise batch.AdapterError("controller_cooldown_not_elapsed")
        replay = super().replay()
        if replay.uncertain:
            raise batch.AdapterError("bulk_c16_fallback_uncertain")
        entries, _, _, _ = self._read_entries()
        leases = [
            entry
            for entry in entries
            if entry.get("entry_kind") == "FALLBACK_COOLDOWN"
            and entry.get("body", {}).get("lease_id") == lease.get("lease_id")
        ]
        completed = [
            entry for entry in entries if entry.get("entry_kind") == "COOLDOWN_COMPLETE"
        ]
        if len(leases) != 1 or completed:
            raise batch.AdapterError("controller_cooldown_lease_drift")
        body = {
            "lease_id": lease["lease_id"],
            "lease_entry_sha256": lease["lease_entry_sha256"],
            "reason_code": "provider_rate_limit",
            "retryable_event_inventory_sha256": lease[
                "retryable_event_inventory_sha256"
            ],
            "retryable_event_count": lease["retryable_event_count"],
            "completed_at_unix_seconds": int(math.floor(now_seconds)),
            "uncertain_dispatches": 0,
            "status_json_authoritative": False,
        }
        entry_sha256 = self._append_wal_entry(
            entry_kind="COOLDOWN_COMPLETE",
            job_keys=[],
            body=body,
        )
        self.authenticate_authority()
        return {**body, "completion_entry_sha256": entry_sha256}

    def verify_completed_fallback_cooldown(
        self, *, fallback_profile: batch.AdapterConfig
    ) -> str:
        self.authenticate_authority()
        entries, _, _, _ = self._read_entries()
        paths = self._entry_paths()
        leases = [
            (entry, _sha256_file(paths[index]))
            for index, entry in enumerate(entries)
            if entry.get("entry_kind") == "FALLBACK_COOLDOWN"
        ]
        completions = [
            entry for entry in entries if entry.get("entry_kind") == "COOLDOWN_COMPLETE"
        ]
        if len(leases) != 1 or len(completions) != 1:
            raise batch.AdapterError("controller_cooldown_lease_drift")
        lease_entry, lease_sha256 = leases[0]
        lease = lease_entry["body"]
        complete = completions[0]["body"]
        replay = super().replay()
        lease_index = entries.index(lease_entry)
        event_entries = entries[:lease_index]
        event_inventory = self._rate_limit_event_inventory(event_entries)
        inventory_sha256 = batch._hash_object(
            {
                "schema_version": (
                    "anchor.gemma3-chat-unbalanced-v2.rate-limit-event-inventory.v1"
                ),
                "events": event_inventory,
            }
        )
        if (
            lease.get("to_profile") != "bulk_c16"
            or lease.get("to_profile_sha256") != fallback_profile.physical_sha256
            or lease.get("reason_code") != "provider_rate_limit"
            or not event_inventory
            or lease.get("retryable_event_count") != len(event_inventory)
            or lease.get("retryable_event_inventory_sha256") != inventory_sha256
            or lease_entry.get("previous_entry_sha256")
            != lease.get("source_wal_chain_tip_sha256")
            or lease.get("uncertain_dispatches") != 0
            or complete.get("lease_id") != lease.get("lease_id")
            or complete.get("lease_entry_sha256") != lease_sha256
            or complete.get("reason_code") != "provider_rate_limit"
            or complete.get("retryable_event_count") != len(event_inventory)
            or complete.get("retryable_event_inventory_sha256") != inventory_sha256
            or complete.get("completed_at_unix_seconds", -1)
            < lease.get("not_before_unix_seconds", 0)
            or complete.get("uncertain_dispatches") != 0
            or replay.uncertain
        ):
            raise batch.AdapterError("controller_cooldown_lease_drift")
        return "provider_rate_limit"


class AuthenticatedBatchRunner(batch.BatchRunner):
    """Frozen runner behavior with only its state implementation replaced."""

    def __init__(
        self,
        config: batch.AdapterConfig,
        inventory: batch.SourceInventory,
        *,
        teachers: Mapping[str, batch.BatchTeacher],
        runtime_slots: batch.RuntimeSecretSlots,
        controller_binding: Mapping[str, Any],
        environ: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            config,
            inventory,
            teachers=teachers,
            runtime_slots=runtime_slots,
            environ=environ,
        )
        self.state = AuthenticatedBatchState(
            config,
            hmac_key=self.hmac_key,
            controller_binding=controller_binding,
        )


async def _default_stage_executor(
    profile: batch.AdapterConfig,
    inventory: batch.SourceInventory,
    runtime_slots: batch.RuntimeSecretSlots,
    phase: str,
    environ: Mapping[str, str],
    controller_binding: Mapping[str, Any],
) -> batch.RunReport:
    runner = AuthenticatedBatchRunner(
        profile,
        inventory,
        teachers=batch.build_teachers(profile, runtime_slots),
        runtime_slots=runtime_slots,
        controller_binding=controller_binding,
        environ=environ,
    )
    return await runner.run(phase)


def _default_fallback_verifier(
    profile: batch.AdapterConfig,
    inventory: batch.SourceInventory,
    runtime_slots: batch.RuntimeSecretSlots,
    controller_binding: Mapping[str, Any],
) -> str:
    state = AuthenticatedBatchState(
        profile,
        hmac_key=runtime_slots.receipt_hmac_key(),
        controller_binding=controller_binding,
    )
    state.initialize(inventory)
    return state.verify_completed_fallback_cooldown(fallback_profile=profile)


def _kill_switch_armed(
    profile: batch.AdapterConfig, environ: Mapping[str, str]
) -> bool:
    return batch._kill_switch_armed(profile, environ)


def _guard_output_root(
    profile: batch.AdapterConfig, environ: Mapping[str, str]
) -> None:
    root = profile.output_root
    _reject_reparse_chain(root, stop=(batch.REPO_ROOT / "data").resolve())
    if _kill_switch_armed(profile, environ):
        raise batch.AdapterError("kill_switch_armed")
    if not root.exists():
        return
    if not root.is_dir():
        raise batch.AdapterError("controller_output_root_invalid")
    try:
        nonempty = next(os.scandir(root), None) is not None
    except OSError as error:
        raise batch.AdapterError("controller_output_root_scan_failed") from error
    if nonempty:
        raise batch.AdapterError("controller_existing_output_not_resumable")


def _expected_stage_total(profile_id: str) -> int:
    return {
        "smoke_exact1": 1,
        "bounded_small_c1": 15,
        "bulk_c30": batch.EXPECTED_TOTAL_JOBS,
        "bulk_c16": batch.EXPECTED_TOTAL_JOBS,
    }[profile_id]


def _validate_report(
    spec: ProfileSpec,
    report: batch.RunReport,
) -> None:
    expected = _expected_stage_total(spec.profile_id)
    if (
        report.phase != spec.phase
        or report.state != "complete"
        or report.total != expected
        or report.queued != 0
        or report.succeeded != expected
        or report.rejected != 0
        or report.blockers
        or (spec.profile_id == "smoke_exact1" and report.requests != 1)
        or (spec.profile_id == "bounded_small_c1" and report.requests != 15)
    ):
        raise batch.AdapterError("controller_stage_not_green")


class SingleProcessController:
    """Own a single secret/HMAC lifetime across all live stages."""

    def __init__(
        self,
        bound: BoundProfiles,
        runtime_slots: batch.RuntimeSecretSlots,
        *,
        profile_loader: ProfileLoader = batch.load_config,
        source_loader: SourceLoader = batch.load_source_inventory,
        heldout_validator: HeldoutValidator = batch.validate_heldout_receipt,
        stage_executor: StageExecutor = _default_stage_executor,
        fallback_verifier: FallbackVerifier = _default_fallback_verifier,
        consumer_validator: ConsumerValidator = validate_consumer_release_binding,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.bound = bound
        self.runtime_slots = runtime_slots
        self.profile_loader = profile_loader
        self.source_loader = source_loader
        self.heldout_validator = heldout_validator
        self.stage_executor = stage_executor
        self.fallback_verifier = fallback_verifier
        self.consumer_validator = consumer_validator
        self.environ = environ if environ is not None else os.environ
        self.clock = clock
        self.sleeper = sleeper
        self._controller_run_id = secrets.token_hex(16)
        self._slot_tag = (
            hmac.new(
                runtime_slots.receipt_hmac_key(),
                b"anchor.unbalanced-v2.single-process-controller.slot.v1",
                hashlib.sha256,
            ).digest()
            if runtime_slots.loaded
            else b""
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"config_sha256={self.bound.controller.physical_sha256!r}, "
            f"runtime_slots_loaded={self.runtime_slots.loaded})"
        )

    def _check_slot_continuity(self) -> None:
        if not self.runtime_slots.loaded:
            raise batch.AdapterError("controller_secret_slots_closed")
        observed = hmac.new(
            self.runtime_slots.receipt_hmac_key(),
            b"anchor.unbalanced-v2.single-process-controller.slot.v1",
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(observed, self._slot_tag):
            raise batch.AdapterError("controller_runtime_hmac_rotated")

    def _fresh_stage(
        self,
        spec: ProfileSpec,
        baseline_inventory: tuple[str, ...] | None,
    ) -> tuple[batch.AdapterConfig, batch.SourceInventory, tuple[str, ...]]:
        _recheck_bound_artifacts(self.bound)
        profile = self.profile_loader(spec.path)
        _validate_profile(self.bound.controller, spec, profile)
        self.heldout_validator(profile)
        inventory = self.source_loader(profile)
        identity = _inventory_identity(inventory)
        if baseline_inventory is not None and identity != baseline_inventory:
            raise batch.AdapterError("controller_source_identity_drift")
        self._check_slot_continuity()
        if _kill_switch_armed(profile, self.environ):
            raise batch.AdapterError("kill_switch_armed")
        return profile, inventory, identity

    def _controller_binding(self, inventory: batch.SourceInventory) -> dict[str, Any]:
        return {
            "controller_config_sha256": self.bound.controller.physical_sha256,
            "controller_implementation_sha256": (
                self.bound.controller_implementation_sha256
            ),
            "teacher_implementation_sha256": (
                self.bound.controller.teacher_implementation_sha256
            ),
            "source_identity_sha256": inventory.source_identity_sha256,
            "manifest_sha256": inventory.manifest_sha256,
            "protected_test_identity_sha256": (inventory.readonly_test_identity_sha256),
            "controller_run_id": self._controller_run_id,
            "accepted_profiles": {
                item.profile_id: item.sha256 for item in self.bound.controller.profiles
            },
        }

    def _authenticate_stage_output(
        self,
        profile: batch.AdapterConfig,
        inventory: batch.SourceInventory,
        binding: Mapping[str, Any],
    ) -> None:
        if self.stage_executor is not _default_stage_executor:
            return
        state = AuthenticatedBatchState(
            profile,
            hmac_key=self.runtime_slots.receipt_hmac_key(),
            controller_binding=binding,
        )
        state.initialize(inventory)
        state.authenticate_authority()

    async def _complete_authenticated_cooldown(
        self,
        *,
        primary_spec: ProfileSpec,
        primary_profile: batch.AdapterConfig,
        primary_inventory: batch.SourceInventory,
        fallback_profile: batch.AdapterConfig,
        binding: Mapping[str, Any],
        baseline: tuple[str, ...],
        rate_limit: batch.RateLimitError,
    ) -> None:
        state = AuthenticatedBatchState(
            primary_profile,
            hmac_key=self.runtime_slots.receipt_hmac_key(),
            controller_binding=binding,
        )
        state.initialize(primary_inventory)
        lease = state.record_fallback_cooldown(
            fallback_profile=fallback_profile,
            retry_after_seconds=rate_limit.retry_after_seconds,
            now_seconds=self.clock(),
        )
        while True:
            now_seconds = self.clock()
            remaining = int(lease["not_before_unix_seconds"]) - now_seconds
            if remaining <= 0:
                break
            self.consumer_validator(self.bound.controller)
            self._fresh_stage(primary_spec, baseline)
            state.authenticate_authority()
            await self.sleeper(min(60.0, remaining))
        self.consumer_validator(self.bound.controller)
        self._fresh_stage(primary_spec, baseline)
        state.authenticate_authority()
        state.complete_fallback_cooldown(
            lease,
            now_seconds=self.clock(),
        )

    async def run(self) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        baseline: tuple[str, ...] | None = None
        final_report: batch.RunReport | None = None
        selected_bulk = "bulk_c30"
        fallback_reason: str | None = None
        try:
            _reject_environment_credentials(self.environ)
            self.consumer_validator(self.bound.controller)
            self._check_slot_continuity()
            _guard_output_root(
                self.bound.values["smoke_exact1"],
                self.environ,
            )
            for profile_id in ("smoke_exact1", "bounded_small_c1"):
                spec = self.bound.controller.profile_map[profile_id]
                profile, inventory, identity = self._fresh_stage(spec, baseline)
                baseline = identity
                binding = self._controller_binding(inventory)
                report = await self.stage_executor(
                    profile,
                    inventory,
                    self.runtime_slots,
                    spec.phase,
                    self.environ,
                    binding,
                )
                self._authenticate_stage_output(profile, inventory, binding)
                _validate_report(spec, report)
                self._fresh_stage(spec, baseline)
                reports.append(
                    {
                        "profile_id": profile_id,
                        "phase": spec.phase,
                        "state": "complete",
                        "succeeded": report.succeeded,
                    }
                )
                final_report = report
            primary_spec = self.bound.controller.profile_map["bulk_c30"]
            profile, inventory, identity = self._fresh_stage(primary_spec, baseline)
            baseline = identity
            binding = self._controller_binding(inventory)
            try:
                report = await self.stage_executor(
                    profile,
                    inventory,
                    self.runtime_slots,
                    primary_spec.phase,
                    self.environ,
                    binding,
                )
                self._authenticate_stage_output(profile, inventory, binding)
                _validate_report(primary_spec, report)
                self._fresh_stage(primary_spec, baseline)
                final_report = report
                reports.append(
                    {
                        "profile_id": "bulk_c30",
                        "phase": "bulk",
                        "state": "complete",
                        "succeeded": report.succeeded,
                    }
                )
            except BaseException as error:
                eligible = isinstance(error, batch.RateLimitError) and not isinstance(
                    error, batch.ProviderQuotaExhausted
                )
                if not eligible:
                    raise
                self._fresh_stage(primary_spec, baseline)
                fallback_spec = self.bound.controller.profile_map["bulk_c16"]
                fallback_profile, fallback_inventory, _ = self._fresh_stage(
                    fallback_spec, baseline
                )
                fallback_binding = self._controller_binding(fallback_inventory)
                assert isinstance(error, batch.RateLimitError)
                await self._complete_authenticated_cooldown(
                    primary_spec=primary_spec,
                    primary_profile=profile,
                    primary_inventory=inventory,
                    fallback_profile=fallback_profile,
                    binding=binding,
                    baseline=baseline,
                    rate_limit=error,
                )
                fallback_reason = self.fallback_verifier(
                    fallback_profile,
                    fallback_inventory,
                    self.runtime_slots,
                    fallback_binding,
                )
                if fallback_reason != "provider_rate_limit":
                    raise batch.AdapterError(
                        "bulk_c16_authenticated_rate_limit_missing"
                    )
                selected_bulk = "bulk_c16"
                fallback_report = await self.stage_executor(
                    fallback_profile,
                    fallback_inventory,
                    self.runtime_slots,
                    fallback_spec.phase,
                    self.environ,
                    fallback_binding,
                )
                self._authenticate_stage_output(
                    fallback_profile,
                    fallback_inventory,
                    fallback_binding,
                )
                _validate_report(fallback_spec, fallback_report)
                self._fresh_stage(fallback_spec, baseline)
                final_report = fallback_report
                reports.extend(
                    [
                        {
                            "profile_id": "bulk_c30",
                            "phase": "bulk",
                            "state": "authenticated_rate_limit",
                            "succeeded": None,
                        },
                        {
                            "profile_id": "bulk_c16",
                            "phase": "bulk",
                            "state": "complete",
                            "succeeded": fallback_report.succeeded,
                        },
                    ]
                )
            assert baseline is not None and final_report is not None
            return {
                "schema_version": STATUS_SCHEMA_VERSION,
                "state": "complete",
                "mode": "execute",
                "stages": reports,
                "selected_bulk_profile": selected_bulk,
                "fallback_reason_code": fallback_reason,
                "wire_requests_from_final_replay": final_report.requests,
                "hashes": {
                    "controller_config": self.bound.controller.physical_sha256,
                    "controller_implementation": (
                        self.bound.controller_implementation_sha256
                    ),
                    "batch_implementation": (
                        self.bound.controller.batch_implementation_sha256
                    ),
                    "teacher_implementation": (
                        self.bound.controller.teacher_implementation_sha256
                    ),
                    "profiles": {
                        item.profile_id: item.sha256
                        for item in self.bound.controller.profiles
                    },
                    "source_identity": baseline[0],
                    "campaign": self.bound.controller.common_identity[
                        "campaign_sha256"
                    ],
                    "model": self.bound.controller.common_identity[
                        "model_binding_sha256"
                    ],
                },
                "runtime_secret_slots_shared": True,
                "cross_process_resume": False,
                "credential_persisted": False,
                "runtime_hmac_public": False,
                "content_retained": False,
                "formal_training_authorized": False,
            }
        finally:
            self.runtime_slots.close()


def _preflight(
    bound: BoundProfiles,
    *,
    validate_source: bool,
    profile_loader: ProfileLoader = batch.load_config,
    source_loader: SourceLoader = batch.load_source_inventory,
    heldout_validator: HeldoutValidator = batch.validate_heldout_receipt,
    consumer_validator: ConsumerValidator = validate_consumer_release_binding,
) -> dict[str, Any]:
    blockers: list[str] = []
    source_identity: tuple[str, ...] | None = None
    try:
        consumer_validator(bound.controller)
    except batch.AdapterError as error:
        blockers.append(error.reason_code)
    if validate_source:
        for spec in bound.controller.profiles:
            profile = profile_loader(spec.path)
            _validate_profile(bound.controller, spec, profile)
            heldout_validator(profile)
            inventory = source_loader(profile)
            identity = _inventory_identity(inventory)
            if source_identity is None:
                source_identity = identity
            elif identity != source_identity:
                raise batch.AdapterError("controller_source_identity_drift")
    _recheck_bound_artifacts(bound)
    blockers.extend(
        ["controller_credential_slot_unloaded", "runtime_hmac_slot_unloaded"]
    )
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "state": "blocked" if blockers else "ready",
        "mode": "validate_only" if validate_source else "dry_run",
        "profiles": {
            item.profile_id: item.sha256 for item in bound.controller.profiles
        },
        "profile_order": ["smoke_exact1", "bounded_small_c1", "bulk_c30"],
        "fallback_profile": "bulk_c16",
        "fallback_only_after": ["provider_rate_limit"],
        "source_identity_sha256": (
            source_identity[0] if source_identity is not None else None
        ),
        "consumer_release_state": bound.controller.consumer_release_binding["state"],
        "ready_for_execute": False,
        "blockers": list(dict.fromkeys(blockers)),
        "network_requests": 0,
        "model_requests": 0,
        "gpu_requests": 0,
        "secret_values_read": False,
        "cross_process_resume": False,
        "content_retained": False,
    }


async def execute_from_process_channel(
    bound: BoundProfiles,
    receiver: Callable[[], bytes],
    *,
    profile_loader: ProfileLoader = batch.load_config,
    source_loader: SourceLoader = batch.load_source_inventory,
    heldout_validator: HeldoutValidator = batch.validate_heldout_receipt,
    stage_executor: StageExecutor = _default_stage_executor,
    fallback_verifier: FallbackVerifier = _default_fallback_verifier,
    consumer_validator: ConsumerValidator = validate_consumer_release_binding,
    environ: Mapping[str, str] | None = None,
    hmac_factory: Callable[[int], bytes] = secrets.token_bytes,
) -> dict[str, Any]:
    """Execute after one in-process credential receive and no pre-read secrets."""

    active_environ = environ if environ is not None else os.environ
    _reject_environment_credentials(active_environ)
    consumer_validator(bound.controller)
    _guard_output_root(bound.values["smoke_exact1"], active_environ)
    slots = batch.RuntimeSecretSlots.from_process_channel(
        receiver, hmac_factory=hmac_factory
    )
    controller = SingleProcessController(
        bound,
        slots,
        profile_loader=profile_loader,
        source_loader=source_loader,
        heldout_validator=heldout_validator,
        stage_executor=stage_executor,
        fallback_verifier=fallback_verifier,
        consumer_validator=consumer_validator,
        environ=active_environ,
    )
    return await controller.run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-process GLM exact1 -> 15 -> bulk controller"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--credential-stdin", action="store_true")
    return parser


def _blocked(reason_code: str) -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "state": "blocked",
        "reason_code": reason_code,
        "network_requests": None,
        "secret_values_retained": False,
        "content_retained": False,
    }


def _body_free_reason(error: BaseException) -> str:
    if isinstance(error, batch.AdapterError):
        return error.reason_code
    if isinstance(error, batch.ProviderQuotaExhausted):
        return "provider_quota_exhausted"
    if isinstance(error, batch.RateLimitError):
        return "provider_rate_limit"
    if isinstance(error, batch.ClientDeadlineExceeded):
        return "provider_client_deadline_exceeded"
    if isinstance(error, batch.BudgetExceeded):
        return "provider_budget_exceeded"
    if isinstance(error, batch.TeacherError):
        return "provider_teacher_failure"
    return "controller_internal_failure"


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        _reject_argv_credentials(raw_argv)
    except batch.AdapterError as error:
        print(json.dumps(_blocked(error.reason_code), sort_keys=True))
        return 2
    parser = _parser()
    args = parser.parse_args(raw_argv)
    if args.execute != args.credential_stdin:
        parser.error("--execute requires --credential-stdin and vice versa")
    try:
        controller = load_controller_config(args.config)
        bound = bind_profiles(controller)
        if args.dry_run:
            result = _preflight(bound, validate_source=False)
        elif args.validate_only:
            result = _preflight(bound, validate_source=True)
        else:
            _reject_environment_credentials(os.environ)
            preflight = _preflight(bound, validate_source=True)
            non_secret_blockers = [
                item
                for item in preflight["blockers"]
                if item
                not in {
                    "controller_credential_slot_unloaded",
                    "runtime_hmac_slot_unloaded",
                }
            ]
            if non_secret_blockers:
                raise batch.AdapterError(non_secret_blockers[0])
            _guard_output_root(bound.values["smoke_exact1"], os.environ)
            _validate_anonymous_os_channel(sys.stdin.buffer)
            slots = batch.RuntimeSecretSlots.from_anonymous_stdin(sys.stdin.buffer)
            result = asyncio.run(SingleProcessController(bound, slots).run())
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("state") == "complete" else 2
    except BaseException as error:
        print(
            json.dumps(
                _blocked(_body_free_reason(error)),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
