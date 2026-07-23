"""Authenticate the frozen-prefix Q-reader post-Gold orchestration profile.

The profile is deliberately metadata-only.  It authenticates the unchanged
SWE-bench five-stage execution core, the canonical-Gold boundary, and a
separate TaskBoard/scaffold/release dependency DAG.  It never reads record
bodies, starts a provider or model, touches a GPU, or authorizes execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, Mapping, Sequence
from uuid import uuid4


PROFILE_SCHEMA_VERSION = "anchor.frozen-prefix-qreader-orchestration-profile.v2"
PROFILE_ID = "frozen-prefix-qreader-v2"
PROFILE_SCHEMA_PATH = Path(
    "configs/orchestration/frozen_prefix_qreader_profile.schema.json"
)
FREEZE_MANIFEST_SCHEMA_PATH = Path(
    "configs/orchestration/frozen_prefix_qreader_profile_freeze_manifest.schema.json"
)
PROFILE_PATH = Path("configs/orchestration/profiles/frozen_prefix_qreader_v2.json")
V1_PROFILE_PATH = Path("configs/orchestration/profiles/task_level_moe_lora_v1.json")
PREFLIGHT_SCHEMA_VERSION = "anchor.frozen-prefix-qreader-profile-preflight.v2"
FREEZE_SCHEMA_VERSION = "anchor.frozen-prefix-qreader-profile-freeze-manifest.v2"
EXECUTION_CORE_ID = "anchor.swebench-five-stage-execution-core.v1"
VIEW_ID = "anchor.frozen-prefix-qreader-post-gold-view.v2"
SHARED_CORE_ROLES = (
    "anchor_launcher",
    "full_bank_builder",
    "full_bank_implementation",
    "full_bank_config",
    "coordinator_implementation",
    "coordinator_config",
    "execution_contract_implementation",
    "execution_runtime_implementation",
    "formal_gold_export_script",
    "formal_gold_implementation",
)
EXPECTED_DEPENDENCY_PATHS = {
    "anchor_launcher": "anchor.ps1",
    "full_bank_builder": "scripts/data/build_swebench_full_bank.py",
    "full_bank_implementation": "src/anchor_mvp/swebench/full_bank.py",
    "full_bank_config": "configs/data/swebench_full_bank.formal.yaml",
    "coordinator_implementation": "scripts/tooling/run_swebench_ccswitch.py",
    "coordinator_config": "configs/data/swebench_five_stage.ccswitch.yaml",
    "execution_contract_implementation": (
        "src/anchor_mvp/tooling/swebench_execution_v3.py"
    ),
    "execution_runtime_implementation": (
        "src/anchor_mvp/tooling/swebench_runtime_v3.py"
    ),
    "formal_gold_export_script": "scripts/data/export_swebench_formal_gold.py",
    "formal_gold_implementation": "src/anchor_mvp/swebench/formal_gold.py",
    "taskboard_projector_config": (
        "configs/research/swebench_taskboard_projector_v2.yaml"
    ),
    "taskboard_projector_implementation": (
        "src/anchor_mvp/swebench/taskboard_projector.py"
    ),
    "taskboard_projector_cli": "scripts/data/project_swebench_taskboard.py",
    "taskboard_projector_manifest_schema": (
        "configs/research/taskboard_projector_manifest.schema.json"
    ),
    "taskboard_projector_sidecar_schema": (
        "configs/research/taskboard_projector_sidecar.schema.json"
    ),
    "taskboard_segment_plan_schema": (
        "configs/research/hierarchical_task_kv_segment_plan.schema.json"
    ),
    "training_view_materializer_config": (
        "configs/research/swebench_natural_language_scaffold_v2.yaml"
    ),
    "training_view_materializer_implementation": (
        "src/anchor_mvp/swebench/natural_language_scaffold_v2.py"
    ),
    "training_view_materializer_config_schema": (
        "configs/research/swebench_natural_language_scaffold_v2_config.schema.json"
    ),
    "training_view_record_schema": (
        "configs/research/swebench_natural_language_scaffold_v2_record.schema.json"
    ),
    "training_view_manifest_schema": (
        "configs/research/swebench_natural_language_scaffold_v2_manifest.schema.json"
    ),
    "bundle_profile_record_schema": (
        "configs/research/"
        "swebench_natural_language_scaffold_v2_bundle_profile.schema.json"
    ),
    "bundle_profile_manifest_schema": (
        "configs/research/"
        "swebench_natural_language_scaffold_v2_bundle_profile_manifest.schema.json"
    ),
    "bundle_profile_descriptor_schema": (
        "configs/research/"
        "swebench_natural_language_scaffold_v2_bundle_profile_descriptor.schema.json"
    ),
    "training_view_materializer_builder": (
        "scripts/data/build_swebench_natural_language_scaffold_v2.py"
    ),
    "training_view_materializer_auditor": (
        "scripts/data/audit_swebench_natural_language_scaffold_v2.py"
    ),
    "generic_execution_contract_schema": (
        "configs/research/generic_train_execution_contract.schema.json"
    ),
    "source_disjoint_manifest_schema": (
        "configs/research/swebench_source_disjoint_manifest.schema.json"
    ),
    "generic_release_lock_schema": (
        "configs/research/generic_train_release_lock.schema.json"
    ),
    "training_release_implementation": ("src/anchor_mvp/swebench/training_release.py"),
    "training_release_cli": "scripts/data/freeze_swebench_training_release.py",
    "release_overlay_schema": (
        "configs/research/frozen_prefix_qreader_release_overlay_v1.schema.json"
    ),
    "release_overlay_implementation": (
        "src/anchor_mvp/swebench/frozen_prefix_qreader_release.py"
    ),
    "release_overlay_cli": ("scripts/data/freeze_frozen_prefix_qreader_release.py"),
    "profile_implementation": (
        "src/anchor_mvp/swebench/frozen_prefix_qreader_profile.py"
    ),
    "profile_runner": "scripts/data/run_frozen_prefix_qreader_profile.py",
}
EXPECTED_DEPENDENCY_ROLES = tuple(EXPECTED_DEPENDENCY_PATHS)

# A dependency may only point from a later orchestration layer to an earlier
# one.  There is intentionally no release-overlay config in this graph: the
# runtime overlay may authenticate the profile freeze manifest, while this
# profile only authenticates the overlay implementation/schema/CLI.
DEPENDENCY_LAYERS = {
    **{role: 0 for role in SHARED_CORE_ROLES},
    "taskboard_projector_config": 1,
    "taskboard_projector_implementation": 1,
    "taskboard_projector_cli": 1,
    "taskboard_projector_manifest_schema": 1,
    "taskboard_projector_sidecar_schema": 1,
    "taskboard_segment_plan_schema": 1,
    "training_view_materializer_config": 2,
    "training_view_materializer_implementation": 2,
    "training_view_materializer_config_schema": 2,
    "training_view_record_schema": 2,
    "training_view_manifest_schema": 2,
    "bundle_profile_record_schema": 2,
    "bundle_profile_manifest_schema": 2,
    "bundle_profile_descriptor_schema": 2,
    "training_view_materializer_builder": 2,
    "training_view_materializer_auditor": 2,
    "generic_execution_contract_schema": 3,
    "source_disjoint_manifest_schema": 3,
    "generic_release_lock_schema": 3,
    "training_release_implementation": 3,
    "training_release_cli": 3,
    "release_overlay_schema": 3,
    "release_overlay_implementation": 3,
    "release_overlay_cli": 3,
    "profile_implementation": 4,
    "profile_runner": 4,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROJECT_PATH = re.compile(
    r"^(?!/)(?![A-Za-z]:)(?!.*\\)(?!.*(?:^|/)\.\.?/)(?!.*//)"
    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$"
)
_REPARSE_POINT = 0x400
_PLACEHOLDER = "PLACEHOLDER_PENDING_PHYSICAL_FREEZE"


class FrozenPrefixQReaderProfileError(RuntimeError):
    """Content-free error code safe for the metadata-only CLI."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class BytesSnapshot:
    data: bytes
    sha256: str
    size: int
    stat_identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class FileBinding:
    role: str
    relative_path: str
    path: Path
    expected_sha256: str
    expected_size: int
    snapshot: BytesSnapshot


@dataclass(frozen=True)
class AuthenticatedFrozenPrefixQReaderProfile:
    project_root: Path
    profile_path: Path
    profile_snapshot: BytesSnapshot
    schema_binding: FileBinding
    freeze_schema_binding: FileBinding
    v1_reference_binding: FileBinding
    dependencies: tuple[FileBinding, ...]
    value: Mapping[str, Any]

    @property
    def profile_sha256(self) -> str:
        return self.profile_snapshot.sha256

    @property
    def dependency_by_role(self) -> Mapping[str, FileBinding]:
        return {item.role: item for item in self.dependencies}

    def recheck(self) -> None:
        current = _read_bytes_snapshot(
            self.profile_path, "profile_changed_during_operation"
        )
        if current != self.profile_snapshot:
            _fail("profile_changed_during_operation")
        for binding in (
            self.schema_binding,
            self.freeze_schema_binding,
            self.v1_reference_binding,
            *self.dependencies,
        ):
            _recheck_binding(binding)


def _fail(code: str) -> None:
    raise FrozenPrefixQReaderProfileError(code)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_reparse(value: os.stat_result) -> bool:
    return bool(getattr(value, "st_file_attributes", 0) & _REPARSE_POINT)


def _read_bytes_snapshot(path: Path, code: str) -> BytesSnapshot:
    try:
        lexical = os.lstat(path)
        if (
            not stat.S_ISREG(lexical.st_mode)
            or stat.S_ISLNK(lexical.st_mode)
            or _is_reparse(lexical)
        ):
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        terminal = os.lstat(path)
    except (OSError, FrozenPrefixQReaderProfileError) as exc:
        if isinstance(exc, FrozenPrefixQReaderProfileError):
            raise
        raise FrozenPrefixQReaderProfileError(code) from exc
    identity = _stat_identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_reparse(before)
        or identity != _stat_identity(after)
        or identity != _stat_identity(terminal)
        or len(data) != before.st_size
    ):
        _fail(code)
    return BytesSnapshot(
        data=data,
        sha256=sha256(data).hexdigest(),
        size=len(data),
        stat_identity=identity,
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _json(snapshot: BytesSnapshot, code: str) -> Any:
    try:
        return json.loads(
            snapshot.data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenPrefixQReaderProfileError(code) from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _safe_project_file(project_root: Path, relative: object, code: str) -> Path:
    parts = PurePosixPath(relative).parts if isinstance(relative, str) else ()
    if (
        not isinstance(relative, str)
        or not _PROJECT_PATH.fullmatch(relative)
        or PurePosixPath(relative).is_absolute()
        or any(part in {".", ".."} for part in parts)
    ):
        _fail(code)
    root = project_root.resolve()
    candidate = root.joinpath(*parts)
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise FrozenPrefixQReaderProfileError(code) from exc
    current = root
    for part in parts:
        current = current / part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise FrozenPrefixQReaderProfileError(code) from exc
        if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
            _fail(code)
    return candidate


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _binding(
    project_root: Path,
    *,
    role: str,
    raw: object,
    code: str,
) -> FileBinding:
    value = _mapping(raw, code)
    if set(value) != {"path", "sha256", "bytes"}:
        _fail(code)
    expected_sha = value.get("sha256")
    expected_size = value.get("bytes")
    if (
        not isinstance(expected_sha, str)
        or not _SHA256.fullmatch(expected_sha)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
    ):
        _fail(code)
    relative = value.get("path")
    path = _safe_project_file(project_root, relative, code)
    snapshot = _read_bytes_snapshot(path, code)
    if snapshot.sha256 != expected_sha or snapshot.size != expected_size:
        _fail(code)
    assert isinstance(relative, str)
    return FileBinding(
        role=role,
        relative_path=relative,
        path=path,
        expected_sha256=expected_sha,
        expected_size=expected_size,
        snapshot=snapshot,
    )


def _check_schema(schema: object, code: str) -> None:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError as exc:
        raise FrozenPrefixQReaderProfileError("jsonschema_runtime_unavailable") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise FrozenPrefixQReaderProfileError(code) from exc


def _validate_schema(instance: object, schema: object, code: str) -> None:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError
    except ImportError as exc:
        raise FrozenPrefixQReaderProfileError("jsonschema_runtime_unavailable") from exc
    _check_schema(schema, code)
    try:
        Draft202012Validator(schema).validate(instance)
    except ValidationError as exc:
        raise FrozenPrefixQReaderProfileError(code) from exc


def _recheck_binding(binding: FileBinding) -> None:
    current = _read_bytes_snapshot(
        binding.path, f"{binding.role}_changed_during_operation"
    )
    if current != binding.snapshot:
        _fail(f"{binding.role}_changed_during_operation")


def _v1_dependency_map(snapshot: BytesSnapshot) -> Mapping[str, Mapping[str, Any]]:
    value = _mapping(_json(snapshot, "v1_reference_invalid"), "v1_reference_invalid")
    if value.get("profile_id") != "task-level-moe-lora-v1":
        _fail("v1_reference_invalid")
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, list):
        _fail("v1_reference_invalid")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in dependencies:
        item = _mapping(raw, "v1_reference_invalid")
        role = item.get("role")
        if not isinstance(role, str) or role in result:
            _fail("v1_reference_invalid")
        result[role] = item
    return result


def _validate_dependency_dag() -> None:
    if set(DEPENDENCY_LAYERS) != set(EXPECTED_DEPENDENCY_ROLES):
        _fail("profile_dependency_dag_invalid")
    paths = tuple(EXPECTED_DEPENDENCY_PATHS.values())
    if len(paths) != len(set(paths)):
        _fail("profile_dependency_dag_invalid")
    if "release_overlay_config" in EXPECTED_DEPENDENCY_ROLES:
        _fail("profile_dependency_cycle_risk")
    if any(
        path == PROFILE_PATH.as_posix()
        or path == FREEZE_MANIFEST_SCHEMA_PATH.as_posix()
        for path in paths
    ):
        _fail("profile_dependency_cycle_risk")


def load_profile(
    project_root: str | Path,
    profile_path: str | Path = PROFILE_PATH,
) -> AuthenticatedFrozenPrefixQReaderProfile:
    """Authenticate the profile, unchanged v1 core, and companion DAG."""

    _validate_dependency_dag()
    root = Path(project_root).expanduser().resolve()
    requested = Path(profile_path).expanduser()
    if requested.is_absolute():
        try:
            lexical = Path(os.path.abspath(requested))
            relative = lexical.relative_to(root).as_posix()
        except ValueError as exc:
            raise FrozenPrefixQReaderProfileError("profile_path_invalid") from exc
    else:
        relative = requested.as_posix()
    resolved_profile = _safe_project_file(root, relative, "profile_path_invalid")
    canonical_profile = _safe_project_file(
        root, PROFILE_PATH.as_posix(), "profile_path_invalid"
    )
    if resolved_profile != canonical_profile:
        _fail("unsupported_profile_path")
    profile_snapshot = _read_bytes_snapshot(resolved_profile, "profile_invalid")
    value = _mapping(_json(profile_snapshot, "profile_invalid"), "profile_invalid")

    schema_binding = _binding(
        root,
        role="profile_schema",
        raw=value.get("profile_schema"),
        code="profile_schema_binding_invalid",
    )
    if schema_binding.relative_path != PROFILE_SCHEMA_PATH.as_posix():
        _fail("profile_schema_path_invalid")
    schema = _json(schema_binding.snapshot, "profile_schema_invalid")
    _validate_schema(value, schema, "profile_schema_validation_failed")

    freeze_schema_binding = _binding(
        root,
        role="freeze_manifest_schema",
        raw=value.get("freeze_manifest_schema"),
        code="freeze_manifest_schema_binding_invalid",
    )
    if freeze_schema_binding.relative_path != FREEZE_MANIFEST_SCHEMA_PATH.as_posix():
        _fail("freeze_manifest_schema_path_invalid")
    freeze_schema = _json(
        freeze_schema_binding.snapshot, "freeze_manifest_schema_invalid"
    )
    _check_schema(freeze_schema, "freeze_manifest_schema_invalid")

    v1_reference_binding = _binding(
        root,
        role="v1_reference",
        raw=value.get("v1_reference"),
        code="v1_reference_binding_invalid",
    )
    if v1_reference_binding.relative_path != V1_PROFILE_PATH.as_posix():
        _fail("v1_reference_path_invalid")

    if (
        value.get("schema_version") != PROFILE_SCHEMA_VERSION
        or value.get("profile_id") != PROFILE_ID
    ):
        _fail("unsupported_profile")

    raw_dependencies = value.get("dependencies")
    if not isinstance(raw_dependencies, list):
        _fail("profile_dependencies_invalid")
    if any(
        isinstance(item, Mapping)
        and (
            item.get("state") != "bound"
            or item.get("sha256") == _PLACEHOLDER
            or item.get("bytes") == 0
        )
        for item in raw_dependencies
    ):
        _fail("profile_dependencies_pending_physical_freeze")

    dependencies: list[FileBinding] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for raw in raw_dependencies:
        item = _mapping(raw, "profile_dependency_invalid")
        if set(item) != {"role", "path", "state", "sha256", "bytes"}:
            _fail("profile_dependency_invalid")
        role = item.get("role")
        if (
            not isinstance(role, str)
            or role not in EXPECTED_DEPENDENCY_ROLES
            or role in seen_roles
            or item.get("state") != "bound"
        ):
            _fail("profile_dependency_role_invalid")
        binding = _binding(
            root,
            role=role,
            raw={
                "path": item.get("path"),
                "sha256": item.get("sha256"),
                "bytes": item.get("bytes"),
            },
            code=f"{role}_binding_invalid",
        )
        if binding.relative_path != EXPECTED_DEPENDENCY_PATHS[role]:
            _fail("profile_dependency_path_invalid")
        if binding.relative_path in seen_paths:
            _fail("profile_dependency_path_reused")
        seen_roles.add(role)
        seen_paths.add(binding.relative_path)
        dependencies.append(binding)
    if tuple(item.role for item in dependencies) != EXPECTED_DEPENDENCY_ROLES:
        _fail("profile_dependency_order_invalid")

    v1_dependencies = _v1_dependency_map(v1_reference_binding.snapshot)
    for binding in dependencies[: len(SHARED_CORE_ROLES)]:
        v1_binding = v1_dependencies.get(binding.role)
        if (
            v1_binding is None
            or v1_binding.get("path") != binding.relative_path
            or v1_binding.get("sha256") != binding.snapshot.sha256
            or v1_binding.get("bytes") != binding.snapshot.size
        ):
            _fail("shared_core_identity_mismatch")

    by_role = {item.role: item for item in dependencies}
    runtime_implementation = _read_bytes_snapshot(
        Path(__file__).resolve(), "profile_runtime_implementation_invalid"
    )
    if (
        by_role["profile_implementation"].snapshot.data != runtime_implementation.data
        or by_role["execution_contract_implementation"].relative_path
        != "src/anchor_mvp/tooling/swebench_execution_v3.py"
        or by_role["execution_runtime_implementation"].relative_path
        != "src/anchor_mvp/tooling/swebench_runtime_v3.py"
    ):
        _fail("profile_dependency_semantics_invalid")

    loaded = AuthenticatedFrozenPrefixQReaderProfile(
        project_root=root,
        profile_path=resolved_profile,
        profile_snapshot=profile_snapshot,
        schema_binding=schema_binding,
        freeze_schema_binding=freeze_schema_binding,
        v1_reference_binding=v1_reference_binding,
        dependencies=tuple(dependencies),
        value=value,
    )
    loaded.recheck()
    return loaded


def _preflight_payload(
    profile: AuthenticatedFrozenPrefixQReaderProfile,
) -> dict[str, Any]:
    dependencies = profile.dependency_by_role
    reference = _mapping(
        profile.value["diagnostic_compatibility_reference"],
        "diagnostic_reference_invalid",
    )
    gemma = _mapping(reference["gemma3_1b_it"], "diagnostic_reference_invalid")
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "profile_ready": True,
        "profile_id": PROFILE_ID,
        "profile_sha256": profile.profile_sha256,
        "profile_schema_sha256": profile.schema_binding.snapshot.sha256,
        "freeze_manifest_schema_sha256": (
            profile.freeze_schema_binding.snapshot.sha256
        ),
        "v1_reference_sha256": profile.v1_reference_binding.snapshot.sha256,
        "execution_core_id": EXECUTION_CORE_ID,
        "post_gold_view_id": VIEW_ID,
        "full_bank_config_path": dependencies["full_bank_config"].relative_path,
        "coordinator_config_path": dependencies["coordinator_config"].relative_path,
        "authenticated_dependency_count": len(profile.dependencies) + 3,
        "shared_core_blob_count": len(SHARED_CORE_ROLES),
        "companion_dependency_count": (
            len(profile.dependencies) - len(SHARED_CORE_ROLES)
        ),
        "dependency_dag_acyclic": True,
        "canonical_gold_mutated": False,
        "diagnostic_fixture_manifest_sha256": reference["fixture_manifest_sha256"],
        "diagnostic_fixture_record_count": reference["record_count"],
        "gemma_sequence_length": gemma["sequence_length"],
        "gemma_strict_no_truncation": gemma["strict_no_truncation"],
        "provider_requests": 0,
        "credentials_read": False,
        "gold_bodies_read": False,
        "heldout_bodies_read": False,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
        "live_authorized": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "release_authorized": False,
    }


def preflight_profile(
    project_root: str | Path,
    profile_path: str | Path = PROFILE_PATH,
) -> dict[str, Any]:
    """Return authenticated metadata without reading any dataset record."""

    profile = load_profile(project_root, profile_path)
    payload = _preflight_payload(profile)
    profile.recheck()
    return payload


def core_command_metadata(
    project_root: str | Path,
    profile_path: str | Path = PROFILE_PATH,
) -> dict[str, Any]:
    """Describe the unchanged offline core command; never execute it."""

    profile = load_profile(project_root, profile_path)
    payload = {
        "schema_version": "anchor.frozen-prefix-qreader-core-command.v1",
        "profile_id": PROFILE_ID,
        "command": [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "anchor.ps1",
            "-Action",
            "distill-swebench",
        ],
        "confirm_live_included": False,
        "executes_command": False,
        "provider_requests": 0,
        "live_authorized": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "release_authorized": False,
    }
    profile.recheck()
    return payload


def _manifest_payload(
    profile: AuthenticatedFrozenPrefixQReaderProfile,
) -> dict[str, Any]:
    dependencies = [
        {
            "role": item.role,
            "path": item.relative_path,
            "sha256": item.snapshot.sha256,
            "bytes": item.snapshot.size,
        }
        for item in profile.dependencies
    ]
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "profile_sha256": profile.profile_sha256,
        "profile_path": profile.profile_path.relative_to(
            profile.project_root
        ).as_posix(),
        "profile_schema": {
            "path": profile.schema_binding.relative_path,
            "sha256": profile.schema_binding.snapshot.sha256,
            "bytes": profile.schema_binding.snapshot.size,
        },
        "freeze_manifest_schema": {
            "path": profile.freeze_schema_binding.relative_path,
            "sha256": profile.freeze_schema_binding.snapshot.sha256,
            "bytes": profile.freeze_schema_binding.snapshot.size,
        },
        "v1_reference": {
            "path": profile.v1_reference_binding.relative_path,
            "sha256": profile.v1_reference_binding.snapshot.sha256,
            "bytes": profile.v1_reference_binding.snapshot.size,
        },
        "execution_core_id": EXECUTION_CORE_ID,
        "post_gold_view_id": VIEW_ID,
        "profile_boundary": "after_authenticated_canonical_gold",
        "dependencies": dependencies,
        "authenticated_dependency_count": len(dependencies) + 3,
        "shared_core_blob_count": len(SHARED_CORE_ROLES),
        "all_dependencies_bound": True,
        "canonical_gold_written": False,
        "canonical_gold_mutated": False,
        "provider_requests": 0,
        "credentials_read": False,
        "gold_bodies_read": False,
        "heldout_bodies_read": False,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
        "live_authorized": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "release_authorized": False,
    }


def _safe_output_dir(project_root: Path, output_dir: str | Path) -> Path:
    requested = Path(output_dir).expanduser()
    lexical = requested if requested.is_absolute() else project_root / requested
    output = Path(os.path.abspath(lexical))
    try:
        relative = output.relative_to(project_root)
    except ValueError as exc:
        raise FrozenPrefixQReaderProfileError("freeze_output_path_invalid") from exc
    if (
        not relative.parts
        or relative.parts[0] != "artifacts"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _fail("freeze_output_path_invalid")
    current = project_root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise FrozenPrefixQReaderProfileError("freeze_output_path_invalid") from exc
        if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
            _fail("freeze_output_path_invalid")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(observed.st_mode):
            _fail("freeze_output_path_invalid")
    return output


def freeze_profile(
    project_root: str | Path,
    profile_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Atomically freeze a fully bound, non-authorizing profile manifest."""

    profile = load_profile(project_root, profile_path)
    output = _safe_output_dir(profile.project_root, output_dir)
    if output.exists() or output.is_symlink():
        _fail("freeze_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    if temporary.exists():
        _fail("freeze_temporary_exists")
    manifest = _manifest_payload(profile)
    freeze_schema = _json(
        profile.freeze_schema_binding.snapshot,
        "freeze_manifest_schema_invalid",
    )
    _validate_schema(
        manifest, freeze_schema, "freeze_manifest_schema_validation_failed"
    )
    encoded = _canonical_json_bytes(manifest)
    digest = sha256(encoded).hexdigest()
    sidecar = f"{digest}  manifest.json\n".encode("ascii")
    published = False
    try:
        temporary.mkdir()
        manifest_path = temporary / "manifest.json"
        sidecar_path = temporary / "manifest.json.sha256"
        manifest_path.write_bytes(encoded)
        sidecar_path.write_bytes(sidecar)
        manifest_snapshot = _read_bytes_snapshot(
            manifest_path, "freeze_manifest_invalid"
        )
        sidecar_snapshot = _read_bytes_snapshot(sidecar_path, "freeze_sidecar_invalid")
        if (
            manifest_snapshot.data != encoded
            or manifest_snapshot.sha256 != digest
            or sidecar_snapshot.data != sidecar
            or _json(manifest_snapshot, "freeze_manifest_invalid") != manifest
        ):
            _fail("freeze_output_invalid")
        profile.recheck()
        os.replace(temporary, output)
        published = True
        final_manifest = _read_bytes_snapshot(
            output / "manifest.json", "freeze_manifest_invalid"
        )
        final_sidecar = _read_bytes_snapshot(
            output / "manifest.json.sha256", "freeze_sidecar_invalid"
        )
        if final_manifest != manifest_snapshot or final_sidecar != sidecar_snapshot:
            _fail("freeze_output_changed")
        profile.recheck()
    except (OSError, FrozenPrefixQReaderProfileError):
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if published and output.exists():
            shutil.rmtree(output, ignore_errors=True)
        raise
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen_non_authorizing",
        "profile_id": PROFILE_ID,
        "profile_sha256": profile.profile_sha256,
        "manifest_sha256": digest,
        "output_dir": output.relative_to(profile.project_root).as_posix(),
        "provider_requests": 0,
        "gold_bodies_read": False,
        "heldout_bodies_read": False,
        "live_authorized": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "release_authorized": False,
    }


__all__ = [
    "AuthenticatedFrozenPrefixQReaderProfile",
    "DEPENDENCY_LAYERS",
    "EXPECTED_DEPENDENCY_PATHS",
    "EXPECTED_DEPENDENCY_ROLES",
    "FREEZE_MANIFEST_SCHEMA_PATH",
    "FREEZE_SCHEMA_VERSION",
    "FrozenPrefixQReaderProfileError",
    "PREFLIGHT_SCHEMA_VERSION",
    "PROFILE_ID",
    "PROFILE_PATH",
    "PROFILE_SCHEMA_PATH",
    "SHARED_CORE_ROLES",
    "core_command_metadata",
    "freeze_profile",
    "load_profile",
    "preflight_profile",
]
