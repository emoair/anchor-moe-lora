"""Authenticated, non-authorizing distillation pipeline profiles.

Profiles are a post-canonical-Gold routing boundary.  They authenticate the
already existing SWE-bench execution core and select a downstream view; they
never alter prompts, start providers, read Gold/heldout bodies, or authorize
training.
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


PROFILE_SCHEMA_VERSION = "anchor.distillation-pipeline-profile.v1"
PROFILE_ID = "task-level-moe-lora-v1"
PROFILE_SCHEMA_PATH = Path(
    "configs/orchestration/distillation_pipeline_profile.schema.json"
)
FREEZE_MANIFEST_SCHEMA_PATH = Path(
    "configs/orchestration/distillation_profile_freeze_manifest.schema.json"
)
PROFILE_PATH = Path("configs/orchestration/profiles/task_level_moe_lora_v1.json")
PREFLIGHT_SCHEMA_VERSION = "anchor.distillation-profile-preflight.v1"
FREEZE_SCHEMA_VERSION = "anchor.distillation-profile-freeze-manifest.v1"
EXECUTION_CORE_ID = "anchor.swebench-five-stage-execution-core.v1"
VIEW_ID = "anchor.task-level-moe-lora-view.v1"
EXPECTED_DEPENDENCY_ROLES = (
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
    "profile_implementation",
    "profile_runner",
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
    "profile_implementation": "src/anchor_mvp/swebench/distillation_profile.py",
    "profile_runner": "scripts/data/run_distillation_profile.py",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROJECT_PATH = re.compile(
    r"^(?!/)(?![A-Za-z]:)(?!.*\\)(?!.*(?:^|/)\.\.?/)(?!.*//)"
    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$"
)
_REPARSE_POINT = 0x400


class DistillationProfileError(RuntimeError):
    """Content-free failure suitable for CLI output."""

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
class AuthenticatedDistillationProfile:
    project_root: Path
    profile_path: Path
    profile_snapshot: BytesSnapshot
    schema_binding: FileBinding
    freeze_schema_binding: FileBinding
    dependencies: tuple[FileBinding, ...]
    value: Mapping[str, Any]

    @property
    def profile_sha256(self) -> str:
        return self.profile_snapshot.sha256

    @property
    def dependency_by_role(self) -> Mapping[str, FileBinding]:
        return {item.role: item for item in self.dependencies}

    def recheck(self) -> None:
        """Re-authenticate all physical bytes at the terminal boundary."""

        current_profile = _read_bytes_snapshot(
            self.profile_path, "profile_changed_during_operation"
        )
        if current_profile != self.profile_snapshot:
            _fail("profile_changed_during_operation")
        _recheck_binding(self.schema_binding)
        _recheck_binding(self.freeze_schema_binding)
        for binding in self.dependencies:
            _recheck_binding(binding)


def _fail(code: str) -> None:
    raise DistillationProfileError(code)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_reparse(value: os.stat_result) -> bool:
    return bool(getattr(value, "st_file_attributes", 0) & _REPARSE_POINT)


def _read_bytes_snapshot(path: Path, code: str) -> BytesSnapshot:
    """Read a regular file once while binding the opened physical object."""

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
    except (OSError, DistillationProfileError) as exc:
        if isinstance(exc, DistillationProfileError):
            raise
        raise DistillationProfileError(code) from exc
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
        raise DistillationProfileError(code) from exc


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
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise DistillationProfileError(code) from exc
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise DistillationProfileError(code) from exc
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


def _check_schema(schema: object) -> None:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError as exc:
        raise DistillationProfileError("jsonschema_runtime_unavailable") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise DistillationProfileError("profile_schema_validation_failed") from exc


def _validate_schema(instance: object, schema: object) -> None:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError
    except ImportError as exc:
        raise DistillationProfileError("jsonschema_runtime_unavailable") from exc
    _check_schema(schema)
    try:
        Draft202012Validator(schema).validate(instance)
    except ValidationError as exc:
        raise DistillationProfileError("profile_schema_validation_failed") from exc


def _recheck_binding(binding: FileBinding) -> None:
    current = _read_bytes_snapshot(
        binding.path, f"{binding.role}_changed_during_operation"
    )
    if current != binding.snapshot:
        _fail(f"{binding.role}_changed_during_operation")


def load_profile(
    project_root: str | Path,
    profile_path: str | Path = PROFILE_PATH,
) -> AuthenticatedDistillationProfile:
    """Authenticate the checked-in profile and every declared dependency."""

    root = Path(project_root).expanduser().resolve()
    requested = Path(profile_path).expanduser()
    if requested.is_absolute():
        try:
            lexical = Path(os.path.abspath(requested))
            relative = lexical.relative_to(root).as_posix()
        except ValueError as exc:
            raise DistillationProfileError("profile_path_invalid") from exc
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
    _validate_schema(value, schema)
    freeze_schema_binding = _binding(
        root,
        role="freeze_manifest_schema",
        raw=value.get("freeze_manifest_schema"),
        code="freeze_manifest_schema_binding_invalid",
    )
    if freeze_schema_binding.relative_path != FREEZE_MANIFEST_SCHEMA_PATH.as_posix():
        _fail("freeze_manifest_schema_path_invalid")
    freeze_schema = _json(
        freeze_schema_binding.snapshot,
        "freeze_manifest_schema_invalid",
    )
    _check_schema(freeze_schema)

    if (
        value.get("schema_version") != PROFILE_SCHEMA_VERSION
        or value.get("profile_id") != PROFILE_ID
    ):
        _fail("unsupported_profile")
    execution_core = _mapping(
        value.get("execution_core"), "profile_execution_core_invalid"
    )
    if execution_core.get("core_id") != EXECUTION_CORE_ID:
        _fail("profile_execution_core_invalid")

    raw_dependencies = value.get("dependencies")
    if not isinstance(raw_dependencies, list):
        _fail("profile_dependencies_invalid")
    dependencies: list[FileBinding] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for item in raw_dependencies:
        dependency = _mapping(item, "profile_dependency_invalid")
        if set(dependency) != {"role", "path", "sha256", "bytes"}:
            _fail("profile_dependency_invalid")
        role = dependency.get("role")
        if (
            not isinstance(role, str)
            or role not in EXPECTED_DEPENDENCY_ROLES
            or role in seen_roles
        ):
            _fail("profile_dependency_role_invalid")
        binding = _binding(
            root,
            role=role,
            raw={
                "path": dependency.get("path"),
                "sha256": dependency.get("sha256"),
                "bytes": dependency.get("bytes"),
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

    by_role = {item.role: item for item in dependencies}
    runtime_implementation = _read_bytes_snapshot(
        Path(__file__).resolve(), "profile_runtime_implementation_invalid"
    )
    if (
        execution_core.get("full_bank_builder_dependency") != "full_bank_builder"
        or execution_core.get("full_bank_implementation_dependency")
        != "full_bank_implementation"
        or execution_core.get("full_bank_config_dependency") != "full_bank_config"
        or execution_core.get("coordinator_implementation_dependency")
        != "coordinator_implementation"
        or execution_core.get("coordinator_config_dependency") != "coordinator_config"
        or execution_core.get("execution_contract_dependency")
        != "execution_contract_implementation"
        or execution_core.get("execution_runtime_dependency")
        != "execution_runtime_implementation"
        or by_role["profile_implementation"].relative_path
        != "src/anchor_mvp/swebench/distillation_profile.py"
        or by_role["profile_implementation"].snapshot.data
        != runtime_implementation.data
    ):
        _fail("profile_dependency_semantics_invalid")

    loaded = AuthenticatedDistillationProfile(
        project_root=root,
        profile_path=resolved_profile,
        profile_snapshot=profile_snapshot,
        schema_binding=schema_binding,
        freeze_schema_binding=freeze_schema_binding,
        dependencies=tuple(dependencies),
        value=value,
    )
    loaded.recheck()
    return loaded


def _preflight_payload(
    profile: AuthenticatedDistillationProfile,
) -> dict[str, Any]:
    dependencies = profile.dependency_by_role
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "profile_ready": True,
        "profile_id": PROFILE_ID,
        "profile_sha256": profile.profile_sha256,
        "profile_schema_sha256": profile.schema_binding.snapshot.sha256,
        "freeze_manifest_schema_sha256": (
            profile.freeze_schema_binding.snapshot.sha256
        ),
        "execution_core_id": EXECUTION_CORE_ID,
        "post_gold_view_id": VIEW_ID,
        "full_bank_config_path": dependencies["full_bank_config"].relative_path,
        "coordinator_config_path": dependencies["coordinator_config"].relative_path,
        "authenticated_dependency_count": len(profile.dependencies) + 2,
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


def preflight_profile(
    project_root: str | Path,
    profile_path: str | Path = PROFILE_PATH,
) -> dict[str, Any]:
    """Return authenticated content-free state without reading data bodies."""

    profile = load_profile(project_root, profile_path)
    payload = _preflight_payload(profile)
    profile.recheck()
    return payload


def _manifest_payload(
    profile: AuthenticatedDistillationProfile,
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
        "execution_core_id": EXECUTION_CORE_ID,
        "post_gold_view_id": VIEW_ID,
        "profile_scope": "post_canonical_gold_view_only",
        "dependencies": dependencies,
        "authenticated_dependency_count": len(dependencies) + 2,
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
        raise DistillationProfileError("freeze_output_path_invalid") from exc
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
            raise DistillationProfileError("freeze_output_path_invalid") from exc
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
    """Atomically freeze one authenticated metadata-only profile manifest."""

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
    _validate_schema(manifest, freeze_schema)
    encoded = _canonical_json_bytes(manifest)
    digest = sha256(encoded).hexdigest()
    sidecar = f"{digest}  manifest.json\n".encode("ascii")
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
        _validate_schema(
            _json(manifest_snapshot, "freeze_manifest_invalid"),
            freeze_schema,
        )
        profile.recheck()
        os.replace(temporary, output)
        final_manifest = _read_bytes_snapshot(
            output / "manifest.json", "freeze_manifest_invalid"
        )
        final_sidecar = _read_bytes_snapshot(
            output / "manifest.json.sha256", "freeze_sidecar_invalid"
        )
        if final_manifest != manifest_snapshot or final_sidecar != sidecar_snapshot:
            _fail("freeze_output_changed")
        profile.recheck()
    except (OSError, DistillationProfileError):
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
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
    "AuthenticatedDistillationProfile",
    "DistillationProfileError",
    "FREEZE_MANIFEST_SCHEMA_PATH",
    "FREEZE_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "PROFILE_ID",
    "PROFILE_PATH",
    "freeze_profile",
    "load_profile",
    "preflight_profile",
]
