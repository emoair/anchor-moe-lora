"""Additive GLM controller with same-process Teacher FINAL materialization.

The historical v1 controller bytes remain immutable and reproducible.  This
v2 controller authenticates and reuses v1's WAL/runtime implementation, then
adds one mandatory finalization phase after a green exact1 -> bounded15 ->
bulk lifecycle and before the single ``RuntimeSecretSlots`` instance closes.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import time
import types
from typing import Any, Awaitable, Callable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from . import gemma3_chat_unbalanced_v2_batch as batch
from . import gemma3_chat_unbalanced_v2_live_controller as base


CONFIG_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-integrated-controller.config.v2"
)
STATUS_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-integrated-controller.status.v2"
)
DEFAULT_CONFIG_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_v2.json"
)
INTEGRATED_IMPLEMENTATION_PATH = Path(__file__).absolute()


@dataclass(frozen=True)
class IntegratedControllerConfig:
    path: Path
    physical_sha256: str
    schema_path: Path
    schema_sha256: str
    implementation_path: Path
    implementation_sha256: str
    base_controller: base.ControllerConfig
    base_anchors: dict[str, Any]
    teacher_finalizer: dict[str, Any]
    teacher_finalizer_paths: dict[str, Path]
    snapshots: dict[str, FileSnapshot]
    lifecycle: dict[str, Any]
    claims: dict[str, Any]


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int]


TeacherFinalizer = Callable[
    [
        batch.AdapterConfig,
        batch.SourceInventory,
        batch.RuntimeSecretSlots,
        Mapping[str, Any],
        Mapping[str, Any],
        batch.RunReport,
        IntegratedControllerConfig,
    ],
    Mapping[str, Any],
]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _reject_raw_reparse_chain(path: Path) -> None:
    current = path.absolute()
    stop = Path(current.anchor)
    while True:
        try:
            value = current.lstat()
        except OSError as error:
            raise batch.AdapterError(
                "integrated_controller_physical_path_invalid"
            ) from error
        attributes = int(getattr(value, "st_file_attributes", 0))
        if stat.S_ISLNK(value.st_mode) or bool(attributes & 0x400):
            raise batch.AdapterError("integrated_controller_reparse_path_forbidden")
        if current == stop:
            return
        parent = current.parent
        if parent == current:
            raise batch.AdapterError("integrated_controller_physical_path_invalid")
        current = parent


def _snapshot_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    reason: str,
) -> FileSnapshot:
    absolute = path.absolute()
    try:
        _reject_raw_reparse_chain(absolute)
        path_before = absolute.lstat()
        attributes = int(getattr(path_before, "st_file_attributes", 0))
        if (
            not stat.S_ISREG(path_before.st_mode)
            or stat.S_ISLNK(path_before.st_mode)
            or bool(attributes & 0x400)
        ):
            raise batch.AdapterError(reason)
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
        flags |= int(getattr(os, "O_NOINHERIT", 0))
        descriptor = os.open(absolute, flags)
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = absolute.lstat()
        _reject_raw_reparse_chain(absolute)
    except OSError as error:
        raise batch.AdapterError(reason) from error
    identities = (
        _file_identity(path_before),
        _file_identity(before),
        _file_identity(after),
        _file_identity(path_after),
    )
    if len(set(identities)) != 1 or len(raw) != before.st_size:
        raise batch.AdapterError(reason)
    digest = _sha256(raw)
    if expected_sha256 is not None and not hmac.compare_digest(
        digest, str(expected_sha256)
    ):
        raise batch.AdapterError(reason)
    return FileSnapshot(absolute, raw, digest, identities[0])


def _recheck_snapshot(snapshot: FileSnapshot, *, reason: str) -> None:
    observed = _snapshot_file(
        snapshot.path,
        expected_sha256=snapshot.sha256,
        reason=reason,
    )
    if observed.identity != snapshot.identity or not hmac.compare_digest(
        observed.raw, snapshot.raw
    ):
        raise batch.AdapterError(reason)


def _lexical_repo_file(raw: Any, *, reason: str) -> Path:
    if not isinstance(raw, (str, Path)):
        raise batch.AdapterError(reason)
    candidate = Path(raw)
    root = batch.REPO_ROOT.absolute()
    value = candidate.absolute() if candidate.is_absolute() else (root / candidate)
    value = value.absolute()
    try:
        value.relative_to(root)
    except ValueError:
        raise batch.AdapterError(reason) from None
    _reject_raw_reparse_chain(value)
    try:
        physical = value.lstat()
    except OSError as error:
        raise batch.AdapterError(reason) from error
    if not stat.S_ISREG(physical.st_mode):
        raise batch.AdapterError(reason)
    return value


def _load_json(raw: bytes, *, reason: str) -> dict[str, Any]:
    value = base._strict_json(raw, reason=reason)
    if not isinstance(value, dict):
        raise batch.AdapterError(reason)
    return value


def _repo_file(raw: Any, *, reason: str) -> Path:
    return _lexical_repo_file(raw, reason=reason)


def _load_base_controller(
    base_anchors: Mapping[str, Any],
    snapshots: dict[str, FileSnapshot],
) -> base.ControllerConfig:
    base_path = _repo_file(
        base_anchors["config_path"],
        reason="integrated_base_controller_path_invalid",
    )
    base_config_snapshot = _snapshot_file(
        base_path,
        expected_sha256=str(base_anchors["config_sha256"]),
        reason="integrated_base_controller_identity_drift",
    )
    snapshots["base_config"] = base_config_snapshot
    value = _load_json(
        base_config_snapshot.raw,
        reason="integrated_base_controller_json_invalid",
    )

    schema_path = _repo_file(
        base_anchors["schema_path"],
        reason="integrated_base_schema_path_invalid",
    )
    schema_snapshot = _snapshot_file(
        schema_path,
        expected_sha256=str(base_anchors["schema_sha256"]),
        reason="integrated_base_controller_identity_drift",
    )
    snapshots["base_schema"] = schema_snapshot
    schema = _load_json(
        schema_snapshot.raw,
        reason="integrated_base_schema_json_invalid",
    )
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except (SchemaError, ValidationError) as error:
        raise batch.AdapterError("integrated_base_controller_schema_invalid") from error

    implementation_path = _repo_file(
        base_anchors["implementation_path"],
        reason="integrated_base_implementation_path_invalid",
    )
    implementation_snapshot = _snapshot_file(
        implementation_path,
        expected_sha256=str(base_anchors["implementation_sha256"]),
        reason="integrated_base_controller_identity_drift",
    )
    snapshots["base_implementation"] = implementation_snapshot
    batch_path = _repo_file(
        value["batch_implementation_path"],
        reason="integrated_base_batch_path_invalid",
    )
    batch_snapshot = _snapshot_file(
        batch_path,
        expected_sha256=str(value["batch_implementation_sha256"]),
        reason="integrated_base_batch_identity_drift",
    )
    snapshots["base_batch_implementation"] = batch_snapshot
    teacher_path = _repo_file(
        value["teacher_implementation_path"],
        reason="integrated_base_teacher_path_invalid",
    )
    teacher_snapshot = _snapshot_file(
        teacher_path,
        expected_sha256=str(value["teacher_implementation_sha256"]),
        reason="integrated_base_teacher_identity_drift",
    )
    snapshots["base_teacher_implementation"] = teacher_snapshot

    if (
        value.get("schema_version") != base.SCHEMA_VERSION
        or value.get("controller_schema_path") != base_anchors["schema_path"]
        or value.get("controller_schema_sha256") != base_anchors["schema_sha256"]
        or value.get("controller_implementation_path")
        != base_anchors["implementation_path"]
        or value.get("controller_implementation_sha256")
        != base_anchors["implementation_sha256"]
        or implementation_path != base.CONTROLLER_IMPLEMENTATION_PATH.absolute()
    ):
        raise batch.AdapterError("integrated_base_controller_binding_drift")

    profiles: list[base.ProfileSpec] = []
    for item in value["profiles"]:
        profile_id = str(item["profile_id"])
        profile_path = _repo_file(
            item["path"],
            reason="integrated_base_profile_path_invalid",
        )
        profile_snapshot = _snapshot_file(
            profile_path,
            expected_sha256=str(item["sha256"]),
            reason="integrated_base_profile_identity_drift",
        )
        snapshots[f"profile_{profile_id}"] = profile_snapshot
        profiles.append(
            base.ProfileSpec(
                stage=str(item["stage"]),
                phase=str(item["phase"]),
                profile_id=profile_id,
                path=profile_path,
                sha256=str(item["sha256"]),
            )
        )

    controller = base.ControllerConfig(
        path=base_path,
        physical_sha256=base_config_snapshot.sha256,
        schema_path=schema_path,
        schema_sha256=schema_snapshot.sha256,
        controller_implementation_path=implementation_path,
        controller_implementation_sha256=implementation_snapshot.sha256,
        batch_implementation_path=batch_path,
        batch_implementation_sha256=batch_snapshot.sha256,
        teacher_implementation_path=teacher_path,
        teacher_implementation_sha256=teacher_snapshot.sha256,
        consumer_release_binding=dict(value["consumer_release_binding"]),
        controller_runtime=dict(value["controller_runtime"]),
        wal=dict(value["wal"]),
        common_identity=dict(value["common_identity"]),
        profiles=tuple(profiles),
        lifecycle=dict(value["lifecycle"]),
        fault_policy=dict(value["fault_policy"]),
        security=dict(value["security"]),
        claims=dict(value["claims"]),
    )
    for snapshot in snapshots.values():
        _recheck_snapshot(
            snapshot,
            reason="integrated_base_controller_terminal_identity_drift",
        )
    return controller


def load_integrated_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> IntegratedControllerConfig:
    config_path = _lexical_repo_file(
        path,
        reason="integrated_controller_config_path_invalid",
    )
    config_snapshot = _snapshot_file(
        config_path,
        reason="integrated_controller_config_snapshot_drift",
    )
    raw = config_snapshot.raw
    value = _load_json(raw, reason="integrated_controller_config_json_invalid")
    schema_path = _repo_file(
        value.get("controller_schema_path"),
        reason="integrated_controller_schema_path_invalid",
    )
    schema_snapshot = _snapshot_file(
        schema_path,
        reason="integrated_controller_schema_snapshot_drift",
    )
    schema_raw = schema_snapshot.raw
    schema_sha256 = _sha256(schema_raw)
    if value.get("controller_schema_sha256") != schema_sha256:
        raise batch.AdapterError("integrated_controller_schema_hash_drift")
    schema = _load_json(
        schema_raw,
        reason="integrated_controller_schema_json_invalid",
    )
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except (SchemaError, ValidationError) as error:
        raise batch.AdapterError(
            "integrated_controller_config_schema_invalid"
        ) from error
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise batch.AdapterError("integrated_controller_version_drift")
    implementation_path = _repo_file(
        value["controller_implementation_path"],
        reason="integrated_controller_implementation_path_invalid",
    )
    if implementation_path != INTEGRATED_IMPLEMENTATION_PATH:
        raise batch.AdapterError("integrated_controller_implementation_path_drift")
    implementation_snapshot = _snapshot_file(
        implementation_path,
        reason="integrated_controller_implementation_snapshot_drift",
    )
    implementation_raw = implementation_snapshot.raw
    implementation_sha256 = _sha256(implementation_raw)
    if value["controller_implementation_sha256"] != implementation_sha256:
        raise batch.AdapterError("integrated_controller_implementation_hash_drift")

    snapshots: dict[str, FileSnapshot] = {
        "integrated_config": config_snapshot,
        "integrated_schema": schema_snapshot,
        "integrated_implementation": implementation_snapshot,
    }
    base_anchors = dict(value["base_controller"])
    base_controller = _load_base_controller(base_anchors, snapshots)
    if (
        base_controller.physical_sha256 != base_anchors["config_sha256"]
        or base_controller.schema_sha256 != base_anchors["schema_sha256"]
        or base_controller.controller_implementation_sha256
        != base_anchors["implementation_sha256"]
        or base_anchors["ancestor_commit"] != "adfa03e225441f010e80f3b2010aec5484a5a90b"
    ):
        raise batch.AdapterError("integrated_base_controller_binding_drift")

    finalizer = dict(value["teacher_finalizer"])
    path_fields = {
        "config": "config_path",
        "config_schema": "config_schema_path",
        "implementation": "implementation_path",
        "record_schema": "record_schema_path",
        "manifest_schema": "manifest_schema_path",
        "build_receipt_schema": "build_receipt_schema_path",
    }
    teacher_paths: dict[str, Path] = {}
    for label, field in path_fields.items():
        resolved = _repo_file(
            finalizer[field],
            reason=f"integrated_teacher_finalizer_{label}_path_invalid",
        )
        expected = str(finalizer[f"{label}_sha256"])
        snapshot = _snapshot_file(
            resolved,
            expected_sha256=expected,
            reason=f"integrated_teacher_finalizer_{label}_identity_drift",
        )
        snapshots[f"teacher_{label}"] = snapshot
        teacher_paths[label] = resolved
    if teacher_paths["implementation"].name != (
        "gemma3_chat_unbalanced_v2_teacher_finalizer_v1.py"
    ):
        raise batch.AdapterError(
            "integrated_teacher_finalizer_implementation_path_drift"
        )
    for snapshot in snapshots.values():
        _recheck_snapshot(
            snapshot,
            reason="integrated_controller_terminal_identity_drift",
        )
    return IntegratedControllerConfig(
        path=config_path,
        physical_sha256=_sha256(raw),
        schema_path=schema_path,
        schema_sha256=schema_sha256,
        implementation_path=implementation_path,
        implementation_sha256=implementation_sha256,
        base_controller=base_controller,
        base_anchors=base_anchors,
        teacher_finalizer=finalizer,
        teacher_finalizer_paths=teacher_paths,
        snapshots=snapshots,
        lifecycle=dict(value["lifecycle"]),
        claims=dict(value["claims"]),
    )


def _recheck_integrated(config: IntegratedControllerConfig) -> None:
    for snapshot in config.snapshots.values():
        _recheck_snapshot(
            snapshot,
            reason="integrated_controller_artifact_identity_drift",
        )


def bind_profiles(
    config: IntegratedControllerConfig,
    *,
    profile_loader: base.ProfileLoader = batch.load_config,
) -> base.BoundProfiles:
    _recheck_integrated(config)
    values: dict[str, batch.AdapterConfig] = {}
    source_binding: dict[str, Any] | None = None
    contract_hashes: dict[str, str] | None = None
    for spec in config.base_controller.profiles:
        snapshot = config.snapshots[f"profile_{spec.profile_id}"]
        _recheck_snapshot(
            snapshot,
            reason="integrated_base_profile_identity_drift",
        )
        profile = profile_loader(spec.path)
        _recheck_snapshot(
            snapshot,
            reason="integrated_base_profile_identity_drift",
        )
        base._validate_profile(config.base_controller, spec, profile)
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
    _recheck_integrated(config)
    return base.BoundProfiles(
        controller=config.base_controller,
        values=values,
        controller_implementation_sha256=(
            config.base_controller.controller_implementation_sha256
        ),
    )


def _load_finalizer_callable(
    config: IntegratedControllerConfig,
) -> Callable[..., Mapping[str, Any]]:
    snapshot = config.snapshots["teacher_implementation"]
    _recheck_snapshot(
        snapshot,
        reason="integrated_teacher_finalizer_implementation_drift",
    )
    path = snapshot.path
    raw = snapshot.raw
    digest = _sha256(raw)
    if digest != config.teacher_finalizer["implementation_sha256"]:
        raise batch.AdapterError("integrated_teacher_finalizer_implementation_drift")
    module_name = f"_anchor_teacher_finalizer_{digest}"
    if module_name in sys.modules:
        raise batch.AdapterError("integrated_teacher_finalizer_module_cache_forbidden")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = "anchor_mvp.data"
    sys.modules[module_name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
        if sys.modules.get(module_name) is not module:
            raise batch.AdapterError("integrated_teacher_finalizer_module_cache_drift")
    except BaseException as error:
        raise batch.AdapterError("integrated_teacher_finalizer_load_failed") from error
    finally:
        sys.modules.pop(module_name, None)
    if module.__dict__.get("__file__") != str(path) or not hmac.compare_digest(
        _sha256(raw), digest
    ):
        raise batch.AdapterError("integrated_teacher_finalizer_module_identity_drift")
    _recheck_snapshot(
        snapshot,
        reason="integrated_teacher_finalizer_implementation_drift",
    )
    function = module.__dict__.get("materialize_teacher_final")
    if not callable(function):
        raise batch.AdapterError("integrated_teacher_finalizer_entrypoint_missing")
    return function


def _default_teacher_finalizer(
    profile: batch.AdapterConfig,
    inventory: batch.SourceInventory,
    runtime_slots: batch.RuntimeSecretSlots,
    controller_binding: Mapping[str, Any],
    controller_identity: Mapping[str, Any],
    final_report: batch.RunReport,
    integrated: IntegratedControllerConfig,
) -> Mapping[str, Any]:
    function = _load_finalizer_callable(integrated)
    finalizer = integrated.teacher_finalizer
    return function(
        profile=profile,
        inventory=inventory,
        runtime_slots=runtime_slots,
        controller_binding=controller_binding,
        controller_identity=controller_identity,
        final_report=final_report,
        config_path=integrated.teacher_finalizer_paths["config"],
        config_sha256=finalizer["config_sha256"],
        config_schema_path=integrated.teacher_finalizer_paths["config_schema"],
        config_schema_sha256=finalizer["config_schema_sha256"],
    )


class IntegratedSingleProcessController(base.SingleProcessController):
    """v1 live stages plus mandatory same-process Teacher FINALization."""

    def __init__(
        self,
        integrated: IntegratedControllerConfig,
        bound: base.BoundProfiles,
        runtime_slots: batch.RuntimeSecretSlots,
        *,
        teacher_finalizer: TeacherFinalizer = _default_teacher_finalizer,
        profile_loader: base.ProfileLoader = batch.load_config,
        source_loader: base.SourceLoader = batch.load_source_inventory,
        heldout_validator: base.HeldoutValidator = batch.validate_heldout_receipt,
        stage_executor: base.StageExecutor = base._default_stage_executor,
        fallback_verifier: base.FallbackVerifier = base._default_fallback_verifier,
        consumer_validator: base.ConsumerValidator = (
            base.validate_consumer_release_binding
        ),
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(
            bound,
            runtime_slots,
            profile_loader=profile_loader,
            source_loader=source_loader,
            heldout_validator=heldout_validator,
            stage_executor=stage_executor,
            fallback_verifier=fallback_verifier,
            consumer_validator=consumer_validator,
            environ=environ,
            clock=clock,
            sleeper=sleeper,
        )
        self.integrated = integrated
        self.teacher_finalizer = teacher_finalizer

    def _fresh_stage(
        self,
        spec: base.ProfileSpec,
        baseline_inventory: tuple[str, ...] | None,
    ) -> tuple[batch.AdapterConfig, batch.SourceInventory, tuple[str, ...]]:
        _recheck_integrated(self.integrated)
        snapshot = self.integrated.snapshots[f"profile_{spec.profile_id}"]
        _recheck_snapshot(
            snapshot,
            reason="integrated_base_profile_identity_drift",
        )
        profile = self.profile_loader(spec.path)
        _recheck_snapshot(
            snapshot,
            reason="integrated_base_profile_identity_drift",
        )
        base._validate_profile(self.bound.controller, spec, profile)
        self.heldout_validator(profile)
        inventory = self.source_loader(profile)
        identity = base._inventory_identity(inventory)
        if baseline_inventory is not None and identity != baseline_inventory:
            raise batch.AdapterError("controller_source_identity_drift")
        self._check_slot_continuity()
        if base._kill_switch_armed(profile, self.environ):
            raise batch.AdapterError("kill_switch_armed")
        _recheck_integrated(self.integrated)
        return profile, inventory, identity

    def _integrated_identity(self) -> dict[str, Any]:
        return {
            "controller_ancestor_commit": self.integrated.base_anchors[
                "ancestor_commit"
            ],
            "integrated_controller_config_sha256": (self.integrated.physical_sha256),
            "integrated_controller_implementation_sha256": (
                self.integrated.implementation_sha256
            ),
            "base_controller_config_sha256": (
                self.integrated.base_controller.physical_sha256
            ),
            "base_controller_implementation_sha256": (
                self.integrated.base_controller.controller_implementation_sha256
            ),
            "batch_implementation_sha256": (
                self.integrated.base_controller.batch_implementation_sha256
            ),
        }

    async def run(self) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        baseline: tuple[str, ...] | None = None
        final_report: batch.RunReport | None = None
        final_profile: batch.AdapterConfig | None = None
        final_inventory: batch.SourceInventory | None = None
        final_binding: Mapping[str, Any] | None = None
        selected_bulk = "bulk_c30"
        fallback_reason: str | None = None
        try:
            base._reject_environment_credentials(self.environ)
            self.consumer_validator(self.bound.controller)
            _recheck_integrated(self.integrated)
            self._check_slot_continuity()
            base._guard_output_root(
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
                base._validate_report(spec, report)
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
                base._validate_report(primary_spec, report)
                self._fresh_stage(primary_spec, baseline)
                final_report = report
                final_profile = profile
                final_inventory = inventory
                final_binding = binding
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
                base._validate_report(fallback_spec, fallback_report)
                self._fresh_stage(fallback_spec, baseline)
                final_report = fallback_report
                final_profile = fallback_profile
                final_inventory = fallback_inventory
                final_binding = fallback_binding
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
            assert (
                baseline is not None
                and final_report is not None
                and final_profile is not None
                and final_inventory is not None
                and final_binding is not None
            )
            self.consumer_validator(self.bound.controller)
            _recheck_integrated(self.integrated)
            self._check_slot_continuity()
            teacher_final = dict(
                self.teacher_finalizer(
                    final_profile,
                    final_inventory,
                    self.runtime_slots,
                    final_binding,
                    self._integrated_identity(),
                    final_report,
                    self.integrated,
                )
            )
            if (
                teacher_final.get("state")
                != "candidate_pending_independent_release_review"
                or teacher_final.get("records") != 3520
                or teacher_final.get("main_records") != 3440
                or teacher_final.get("router_records") != 80
                or teacher_final.get("train_identity_bundles") != 20
                or teacher_final.get("train_identity_records") != 100
                or teacher_final.get("runtime_hmac_consumed_before_close") is not True
                or teacher_final.get("training_authorized") is not False
                or teacher_final.get("formal_training_authorized") is not False
                or teacher_final.get("live_authorized") is not False
            ):
                raise batch.AdapterError("integrated_teacher_finalizer_status_invalid")
            self._check_slot_continuity()
            self.consumer_validator(self.bound.controller)
            _recheck_integrated(self.integrated)
            reports.append(
                {
                    "profile_id": "teacher_finalizer_v1",
                    "phase": "teacher_final_materialization",
                    "state": teacher_final["state"],
                    "succeeded": teacher_final["records"],
                }
            )
            return {
                "schema_version": STATUS_SCHEMA_VERSION,
                "state": "complete",
                "mode": "execute",
                "stages": reports,
                "selected_bulk_profile": selected_bulk,
                "fallback_reason_code": fallback_reason,
                "wire_requests_from_final_replay": final_report.requests,
                "teacher_final": teacher_final,
                "hashes": {
                    "integrated_controller_config": (self.integrated.physical_sha256),
                    "integrated_controller_implementation": (
                        self.integrated.implementation_sha256
                    ),
                    "base_controller_config": (
                        self.integrated.base_controller.physical_sha256
                    ),
                    "base_controller_implementation": (
                        self.integrated.base_controller.controller_implementation_sha256
                    ),
                    "batch_implementation": (
                        self.integrated.base_controller.batch_implementation_sha256
                    ),
                    "teacher_implementation": (
                        self.integrated.base_controller.teacher_implementation_sha256
                    ),
                    "teacher_finalizer_config": (
                        self.integrated.teacher_finalizer["config_sha256"]
                    ),
                    "teacher_finalizer_implementation": (
                        self.integrated.teacher_finalizer["implementation_sha256"]
                    ),
                    "source_identity": baseline[0],
                    "campaign": self.bound.controller.common_identity[
                        "campaign_sha256"
                    ],
                    "model": self.bound.controller.common_identity[
                        "model_binding_sha256"
                    ],
                },
                "runtime_secret_slots_shared_through_finalization": True,
                "cross_process_resume": False,
                "credential_persisted": False,
                "runtime_hmac_public": False,
                "content_retained_in_status": False,
                "external_release_review_required": True,
                "training_authorized": False,
                "formal_training_authorized": False,
                "live_authorized": False,
            }
        finally:
            self.runtime_slots.close()


def _preflight(
    integrated: IntegratedControllerConfig,
    bound: base.BoundProfiles,
    *,
    validate_source: bool,
) -> dict[str, Any]:
    result = dict(base._preflight(bound, validate_source=validate_source))
    _recheck_integrated(integrated)
    result.update(
        {
            "schema_version": STATUS_SCHEMA_VERSION,
            "integrated_controller_config_sha256": (integrated.physical_sha256),
            "integrated_controller_implementation_sha256": (
                integrated.implementation_sha256
            ),
            "teacher_finalizer_config_sha256": (
                integrated.teacher_finalizer["config_sha256"]
            ),
            "teacher_finalizer_implementation_sha256": (
                integrated.teacher_finalizer["implementation_sha256"]
            ),
            "teacher_finalization_position": "before_runtime_slots_close",
            "external_release_review_required": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
        }
    )
    return result


async def execute_from_process_channel(
    integrated: IntegratedControllerConfig,
    bound: base.BoundProfiles,
    receiver: Callable[[], bytes],
    *,
    teacher_finalizer: TeacherFinalizer = _default_teacher_finalizer,
    profile_loader: base.ProfileLoader = batch.load_config,
    source_loader: base.SourceLoader = batch.load_source_inventory,
    heldout_validator: base.HeldoutValidator = batch.validate_heldout_receipt,
    stage_executor: base.StageExecutor = base._default_stage_executor,
    fallback_verifier: base.FallbackVerifier = base._default_fallback_verifier,
    consumer_validator: base.ConsumerValidator = (
        base.validate_consumer_release_binding
    ),
    environ: Mapping[str, str] | None = None,
    hmac_factory: Callable[[int], bytes] = secrets.token_bytes,
) -> dict[str, Any]:
    active_environ = environ if environ is not None else os.environ
    base._reject_environment_credentials(active_environ)
    consumer_validator(bound.controller)
    _recheck_integrated(integrated)
    base._guard_output_root(bound.values["smoke_exact1"], active_environ)
    slots = batch.RuntimeSecretSlots.from_process_channel(
        receiver,
        hmac_factory=hmac_factory,
    )
    controller = IntegratedSingleProcessController(
        integrated,
        bound,
        slots,
        teacher_finalizer=teacher_finalizer,
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
        description=("Integrated GLM exact1 -> 15 -> bulk -> Teacher FINAL controller")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--credential-stdin", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        base._reject_argv_credentials(raw_argv)
    except batch.AdapterError as error:
        print(json.dumps(base._blocked(error.reason_code), sort_keys=True))
        return 2
    parser = _parser()
    args = parser.parse_args(raw_argv)
    if args.execute != args.credential_stdin:
        parser.error("--execute requires --credential-stdin and vice versa")
    try:
        integrated = load_integrated_config(args.config)
        bound = bind_profiles(integrated)
        if args.dry_run:
            result = _preflight(integrated, bound, validate_source=False)
        elif args.validate_only:
            result = _preflight(integrated, bound, validate_source=True)
        else:
            base._reject_environment_credentials(os.environ)
            preflight = _preflight(integrated, bound, validate_source=True)
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
            base._guard_output_root(bound.values["smoke_exact1"], os.environ)
            base._validate_anonymous_os_channel(sys.stdin.buffer)
            slots = batch.RuntimeSecretSlots.from_anonymous_stdin(sys.stdin.buffer)
            result = asyncio.run(
                IntegratedSingleProcessController(
                    integrated,
                    bound,
                    slots,
                ).run()
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("state") == "complete" else 2
    except BaseException as error:
        print(
            json.dumps(
                base._blocked(base._body_free_reason(error)),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
