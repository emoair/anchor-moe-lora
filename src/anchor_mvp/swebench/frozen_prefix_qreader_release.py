"""Fail-closed, non-authorizing release overlay for the Q-reader v2 profile.

The overlay authenticates manifests and their mandatory sidecars only.  It
does not inspect materialized JSONL partitions, canonical Gold, held-out
examples, prompts, answers, token ids, model weights, or provider credentials.
Even a self-ready generic v2 release lock remains only one conjunct: this
overlay is deliberately blocked until a future, separately versioned
authorization decision and execution lease exist.
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


CONFIG_SCHEMA_VERSION = "anchor.frozen-prefix-qreader-release-overlay-config.v1"
OVERLAY_SCHEMA_VERSION = "anchor.frozen-prefix-qreader-release-overlay.v1"
OVERLAY_STATUS = "profile_materialized_execution_blocked"
CONFIG_PATH = Path("configs/research/frozen_prefix_qreader_release_overlay_v1.json")
OVERLAY_SCHEMA_PATH = Path(
    "configs/research/frozen_prefix_qreader_release_overlay_v1.schema.json"
)
IMPLEMENTATION_PATH = Path("src/anchor_mvp/swebench/frozen_prefix_qreader_release.py")
FREEZE_SCRIPT_PATH = Path("scripts/data/freeze_frozen_prefix_qreader_release.py")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROJECT_PATH_RE = re.compile(
    r"^(?!/)(?![A-Za-z]:)(?!.*\\)(?!.*(?:^|/)\.\.?/)(?!.*//)"
    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$"
)
_REPARSE_POINT = 0x400
_MAX_METADATA_BYTES = 50_000_000
_INPUT_ROLES = (
    "generic_release_lock",
    "profile_freeze",
    "training_view",
    "bundle_profile",
    "consumer_reference",
)
_EXPECTED_INPUT_CONTRACTS = {
    "generic_release_lock": (
        "anchor.generic-train-release-lock.v2",
        "configs/research/generic_train_release_lock.schema.json",
        "artifacts/formal_v3/training_release/release_lock",
    ),
    "profile_freeze": (
        "anchor.frozen-prefix-qreader-profile-freeze-manifest.v2",
        "configs/orchestration/frozen_prefix_qreader_profile_freeze_manifest.schema.json",
        "artifacts/distillation-profiles/frozen-prefix-qreader-v2",
    ),
    "training_view": (
        "anchor.frozen-prefix-qreader-training-view-manifest.v2",
        "configs/research/swebench_natural_language_scaffold_v2_manifest.schema.json",
        "artifacts/swebench/frozen-prefix-qreader-view-v2",
    ),
    "bundle_profile": (
        "anchor.frozen-prefix-qreader-bundle-profile-manifest.v2",
        "configs/research/swebench_natural_language_scaffold_v2_bundle_profile_manifest.schema.json",
        "artifacts/swebench/frozen-prefix-qreader-bundle-profile-v2",
    ),
    "consumer_reference": (
        "anchor.synthetic-five-role-qonly-diagnostic-manifest.v1",
        "configs/research/frozen_prefix_qreader_release_overlay_v1.schema.json",
        "consumer-repository/fixtures/research/synthetic_five_role_qonly_diagnostic_v1",
    ),
}
_CROSS_BOUND_PROFILE_DEPENDENCY_ROLES = (
    "taskboard_projector_config",
    "taskboard_projector_implementation",
    "taskboard_projector_cli",
    "taskboard_projector_manifest_schema",
    "taskboard_projector_sidecar_schema",
    "taskboard_segment_plan_schema",
    "training_view_materializer_config",
    "training_view_materializer_implementation",
    "training_view_materializer_config_schema",
    "training_view_record_schema",
    "training_view_manifest_schema",
    "bundle_profile_record_schema",
    "bundle_profile_manifest_schema",
    "bundle_profile_descriptor_schema",
    "training_view_materializer_builder",
    "training_view_materializer_auditor",
    "generic_release_lock_schema",
)


class FrozenPrefixQReaderReleaseError(RuntimeError):
    """A stable, content-free refusal code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise FrozenPrefixQReaderReleaseError(code)


@dataclass(frozen=True)
class BytesSnapshot:
    data: bytes
    sha256: str
    size: int
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class InputArtifact:
    role: str
    root: Path
    manifest_path: Path
    manifest: BytesSnapshot
    sidecar_path: Path
    sidecar: BytesSnapshot
    value: Mapping[str, Any]
    schema_path: Path
    schema: BytesSnapshot


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_reparse(value: os.stat_result) -> bool:
    return bool(getattr(value, "st_file_attributes", 0) & _REPARSE_POINT)


def _contains_dot_segment(value: str | Path) -> bool:
    return any(part in {".", ".."} for part in re.split(r"[/\\]", str(value)))


def _assert_plain_path(path: Path, code: str, *, regular_file: bool) -> None:
    """Reject lexical symlinks/reparse points instead of trusting resolve()."""

    absolute = Path(os.path.abspath(path))
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.parts[1:]:
        current = current / part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise FrozenPrefixQReaderReleaseError(code) from exc
        if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
            _fail(code)
    terminal = os.lstat(absolute)
    if regular_file:
        if not stat.S_ISREG(terminal.st_mode):
            _fail(code)
    elif not stat.S_ISDIR(terminal.st_mode):
        _fail(code)


def _read_snapshot(path: Path, code: str) -> BytesSnapshot:
    try:
        _assert_plain_path(path, code, regular_file=True)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size < 1 or before.st_size >= _MAX_METADATA_BYTES:
                _fail(code)
            data = handle.read()
            after = os.fstat(handle.fileno())
        terminal = os.lstat(path)
    except FrozenPrefixQReaderReleaseError:
        raise
    except OSError as exc:
        raise FrozenPrefixQReaderReleaseError(code) from exc
    identity = _identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_reparse(before)
        or identity != _identity(after)
        or identity != _identity(terminal)
        or len(data) != before.st_size
    ):
        _fail(code)
    return BytesSnapshot(
        data=data,
        sha256=sha256(data).hexdigest(),
        size=len(data),
        identity=identity,
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _parse_json(snapshot: BytesSnapshot, code: str) -> Any:
    try:
        return json.loads(
            snapshot.data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        RecursionError,
    ) as exc:
        raise FrozenPrefixQReaderReleaseError(code) from exc


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _schema_validate(instance: object, schema: object, code: str) -> None:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError, ValidationError
    except ImportError as exc:
        raise FrozenPrefixQReaderReleaseError("jsonschema_runtime_unavailable") from exc
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance)
    except (SchemaError, ValidationError) as exc:
        raise FrozenPrefixQReaderReleaseError(code) from exc


def _safe_project_file(root: Path, relative: object, code: str) -> Path:
    if (
        not isinstance(relative, str)
        or not _PROJECT_PATH_RE.fullmatch(relative)
        or PurePosixPath(relative).is_absolute()
        or _contains_dot_segment(relative)
        or any(part in {".", ".."} for part in PurePosixPath(relative).parts)
    ):
        _fail(code)
    root = Path(os.path.abspath(root))
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise FrozenPrefixQReaderReleaseError(code) from exc
    _assert_plain_path(candidate, code, regular_file=True)
    return candidate


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _strict_sidecar(
    sidecar_path: Path,
    manifest: BytesSnapshot,
    code: str,
) -> BytesSnapshot:
    sidecar = _read_snapshot(sidecar_path, code)
    if sidecar.data != f"{manifest.sha256}  manifest.json\n".encode("ascii"):
        _fail(code)
    return sidecar


def _load_config(
    project_root: Path,
    config_path: Path,
    expected_config_sha256: str,
) -> tuple[
    Mapping[str, Any],
    dict[str, tuple[Path, BytesSnapshot]],
    Mapping[str, Mapping[str, Any]],
]:
    if not _is_sha256(expected_config_sha256):
        _fail("release_overlay_config_sha256_invalid")
    root = Path(os.path.abspath(project_root))
    requested = config_path
    if _contains_dot_segment(requested):
        _fail("release_overlay_config_path_invalid")
    if requested.is_absolute():
        try:
            relative = Path(os.path.abspath(requested)).relative_to(root).as_posix()
        except ValueError as exc:
            raise FrozenPrefixQReaderReleaseError(
                "release_overlay_config_path_invalid"
            ) from exc
    else:
        relative = requested.as_posix()
    if relative != CONFIG_PATH.as_posix():
        _fail("release_overlay_config_path_invalid")
    physical = _safe_project_file(root, relative, "release_overlay_config_path_invalid")
    snapshot = _read_snapshot(physical, "release_overlay_config_invalid")
    if snapshot.sha256 != expected_config_sha256:
        _fail("release_overlay_config_hash_mismatch")
    value = _mapping(
        _parse_json(snapshot, "release_overlay_config_invalid"),
        "release_overlay_config_invalid",
    )
    reparsed_config = _mapping(
        _parse_json(snapshot, "release_overlay_config_invalid"),
        "release_overlay_config_invalid",
    )
    if reparsed_config != value:
        _fail("release_overlay_config_invalid")
    if (
        set(value)
        != {
            "schema_version",
            "producer_bindings",
            "input_contracts",
            "dependency_dag",
            "architecture_boundary",
            "consumer_diagnostic_reference",
            "gemma_runner_compatibility",
            "unresolved_gates",
        }
        or value.get("schema_version") != CONFIG_SCHEMA_VERSION
    ):
        _fail("release_overlay_config_invalid")
    _validate_dependency_dag(
        value.get("dependency_dag"), "release_overlay_config_invalid"
    )

    inventory: dict[str, tuple[Path, BytesSnapshot]] = {"config": (physical, snapshot)}
    bindings = _mapping(
        value.get("producer_bindings"), "release_overlay_config_invalid"
    )
    expected_paths = {
        "overlay_schema": OVERLAY_SCHEMA_PATH,
        "implementation": IMPLEMENTATION_PATH,
        "freeze_script": FREEZE_SCRIPT_PATH,
    }
    for role, expected_path in expected_paths.items():
        binding = _mapping(bindings.get(role), "release_overlay_config_invalid")
        if set(binding) != {"path", "sha256", "bytes"}:
            _fail("release_overlay_config_invalid")
        if (
            binding.get("path") != expected_path.as_posix()
            or not _is_sha256(binding.get("sha256"))
            or isinstance(binding.get("bytes"), bool)
            or not isinstance(binding.get("bytes"), int)
            or int(binding["bytes"]) < 1
        ):
            _fail("release_overlay_config_invalid")
        path = _safe_project_file(
            root, binding["path"], "release_overlay_producer_binding_invalid"
        )
        observed = _read_snapshot(path, "release_overlay_producer_binding_invalid")
        if observed.sha256 != binding["sha256"] or observed.size != binding["bytes"]:
            _fail("release_overlay_producer_binding_invalid")
        inventory[role] = (path, observed)

    bound_implementation_path, bound_implementation = inventory["implementation"]
    executing_path = Path(os.path.abspath(__file__))
    if executing_path == bound_implementation_path:
        executing_implementation = bound_implementation
    else:
        executing_implementation = _read_snapshot(
            executing_path, "executing_implementation_binding_invalid"
        )
        inventory["executing_implementation"] = (
            executing_path,
            executing_implementation,
        )
    if (
        executing_implementation.data != bound_implementation.data
        or executing_implementation.sha256 != bound_implementation.sha256
        or executing_implementation.size != bound_implementation.size
    ):
        _fail("executing_implementation_binding_invalid")

    contracts_value = _mapping(
        value.get("input_contracts"), "release_overlay_config_invalid"
    )
    if set(contracts_value) != set(_INPUT_ROLES):
        _fail("release_overlay_config_invalid")
    contracts: dict[str, Mapping[str, Any]] = {}
    for role in _INPUT_ROLES:
        contract = _mapping(contracts_value.get(role), "release_overlay_config_invalid")
        expected_version, expected_schema_path, expected_runtime_dir = (
            _EXPECTED_INPUT_CONTRACTS[role]
        )
        if set(contract) != {
            "schema_version",
            "schema_path",
            "schema_sha256",
            "canonical_runtime_dir",
        }:
            _fail("release_overlay_config_invalid")
        if (
            not isinstance(contract.get("schema_version"), str)
            or not contract["schema_version"]
            or not isinstance(contract.get("canonical_runtime_dir"), str)
            or not contract["canonical_runtime_dir"]
            or not _is_sha256(contract.get("schema_sha256"))
            or contract.get("schema_version") != expected_version
            or contract.get("schema_path") != expected_schema_path
            or contract.get("canonical_runtime_dir") != expected_runtime_dir
        ):
            _fail("release_overlay_config_invalid")
        schema_path = _safe_project_file(
            root,
            contract.get("schema_path"),
            f"{role}_schema_binding_invalid",
        )
        schema_snapshot = _read_snapshot(schema_path, f"{role}_schema_binding_invalid")
        if schema_snapshot.sha256 != contract["schema_sha256"]:
            _fail(f"{role}_schema_binding_invalid")
        schema = _parse_json(schema_snapshot, f"{role}_schema_invalid")
        schema_reparsed = _parse_json(schema_snapshot, f"{role}_schema_invalid")
        if schema_reparsed != schema:
            _fail(f"{role}_schema_invalid")
        if role == "consumer_reference":
            root_schema = _mapping(schema, f"{role}_schema_invalid")
            definitions = _mapping(root_schema.get("$defs"), f"{role}_schema_invalid")
            schema = _mapping(
                definitions.get("consumer_source_manifest"),
                f"{role}_schema_invalid",
            )
        inventory[f"{role}_schema"] = (schema_path, schema_snapshot)
        contracts[role] = {
            **contract,
            "_schema": schema,
            "_path": schema_path,
            "_snapshot": schema_snapshot,
        }

    overlay_schema = _parse_json(
        inventory["overlay_schema"][1], "release_overlay_schema_invalid"
    )
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError

        Draft202012Validator.check_schema(overlay_schema)
    except (ImportError, SchemaError) as exc:
        raise FrozenPrefixQReaderReleaseError("release_overlay_schema_invalid") from exc
    return value, inventory, contracts


def _validate_dependency_dag(value: object, code: str) -> None:
    dag = _mapping(value, code)
    if set(dag) != {
        "nodes",
        "edges",
        "config_excludes_runtime_manifest_hashes",
        "v2_profile_may_bind_overlay_code_but_not_overlay_config",
        "acyclic",
    }:
        _fail(code)
    nodes = dag.get("nodes")
    edges = dag.get("edges")
    if (
        not isinstance(nodes, list)
        or not nodes
        or any(not isinstance(item, str) or not item for item in nodes)
        or len(set(nodes)) != len(nodes)
        or not isinstance(edges, list)
        or dag.get("config_excludes_runtime_manifest_hashes") is not True
        or dag.get("v2_profile_may_bind_overlay_code_but_not_overlay_config")
        is not True
        or dag.get("acyclic") is not True
    ):
        _fail(code)
    graph = {str(node): set() for node in nodes}
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or edge[0] not in graph
            or edge[1] not in graph
            or edge[0] == edge[1]
        ):
            _fail(code)
        graph[str(edge[0])].add(str(edge[1]))
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            _fail(code)
        if node in visited:
            return
        visiting.add(node)
        for successor in graph[node]:
            visit(successor)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
    if "overlay_config" in graph.get("v2_profile_freeze", set()):
        _fail(code)


def _load_artifact(
    role: str,
    directory: str | Path,
    expected_manifest_sha256: str,
    contract: Mapping[str, Any],
) -> InputArtifact:
    code = f"{role}_artifact_invalid"
    if not _is_sha256(expected_manifest_sha256):
        _fail(code)
    expanded = Path(directory).expanduser()
    if _contains_dot_segment(directory):
        _fail(code)
    root = Path(os.path.abspath(expanded))
    _assert_plain_path(root, code, regular_file=False)
    manifest_path = root / "manifest.json"
    manifest = _read_snapshot(manifest_path, code)
    if manifest.sha256 != expected_manifest_sha256:
        _fail(code)
    sidecar_path = root / "manifest.json.sha256"
    sidecar = _strict_sidecar(sidecar_path, manifest, code)
    value = _mapping(_parse_json(manifest, code), code)
    reparsed = _mapping(_parse_json(manifest, code), code)
    if reparsed != value:
        _fail(code)
    if value.get("schema_version") != contract["schema_version"]:
        _fail(code)
    _schema_validate(value, contract["_schema"], code)
    return InputArtifact(
        role=role,
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        sidecar_path=sidecar_path,
        sidecar=sidecar,
        value=value,
        schema_path=contract["_path"],
        schema=contract["_snapshot"],
    )


def _semantic_checks(
    artifacts: Mapping[str, InputArtifact],
    config: Mapping[str, Any],
    project_root: Path,
    inventory: dict[str, tuple[Path, BytesSnapshot]],
) -> None:
    for role, artifact in artifacts.items():
        _reject_authority_claims(artifact.value, f"{role}_authority_claim_invalid")
    generic = artifacts["generic_release_lock"].value
    if (
        generic.get("status") != "ready"
        or generic.get("claim_scope") != "research_proxy_only"
        or generic.get("provider_requests") != 0
        or generic.get("canonical_gold_written") is not False
        or generic.get("heldout_content_read") is not False
        or generic.get("heldout_content_emitted") is not False
    ):
        _fail("generic_release_lock_semantics_invalid")

    profile = artifacts["profile_freeze"].value
    profile_dependencies = profile.get("dependencies")
    if not isinstance(profile_dependencies, list):
        _fail("profile_freeze_semantics_invalid")
    dependency_by_role: dict[str, Mapping[str, Any]] = {}
    for item in profile_dependencies:
        dependency = _mapping(item, "profile_freeze_semantics_invalid")
        role = dependency.get("role")
        if not isinstance(role, str) or role in dependency_by_role:
            _fail("profile_freeze_semantics_invalid")
        dependency_by_role[role] = dependency
    for role in _CROSS_BOUND_PROFILE_DEPENDENCY_ROLES:
        dependency = dependency_by_role.get(role)
        if dependency is None:
            _fail("profile_dependency_binding_invalid")
        path = _safe_project_file(
            project_root,
            dependency.get("path"),
            "profile_dependency_binding_invalid",
        )
        snapshot = _read_snapshot(path, "profile_dependency_binding_invalid")
        if snapshot.sha256 != dependency.get(
            "sha256"
        ) or snapshot.size != dependency.get("bytes"):
            _fail("profile_dependency_binding_invalid")
        inventory[f"profile_dependency_{role}"] = (path, snapshot)

    def dependency_sha(role: str) -> str:
        dependency = dependency_by_role.get(role)
        if dependency is None or not _is_sha256(dependency.get("sha256")):
            _fail("profile_dependency_binding_invalid")
        return str(dependency["sha256"])

    producer_bindings = _mapping(
        config.get("producer_bindings"), "release_overlay_config_invalid"
    )
    expected_overlay_dependencies = {
        "release_overlay_schema": producer_bindings["overlay_schema"],
        "release_overlay_implementation": producer_bindings["implementation"],
        "release_overlay_cli": producer_bindings["freeze_script"],
    }
    for role, expected in expected_overlay_dependencies.items():
        observed = dependency_by_role.get(role)
        if (
            observed is None
            or observed.get("path") != expected["path"]
            or observed.get("sha256") != expected["sha256"]
            or observed.get("bytes") != expected["bytes"]
        ):
            _fail("profile_overlay_dependency_binding_invalid")
    if "release_overlay_config" in dependency_by_role:
        _fail("profile_overlay_dependency_cycle")
    if (
        profile.get("profile_id") != "frozen-prefix-qreader-v2"
        or profile.get("profile_boundary") != "after_authenticated_canonical_gold"
        or profile.get("canonical_gold_written") is not False
        or profile.get("canonical_gold_mutated") is not False
        or profile.get("provider_requests") != 0
        or profile.get("gold_bodies_read") is not False
        or profile.get("heldout_bodies_read") is not False
        or profile.get("training_authorized") is not False
        or profile.get("formal_training_authorized") is not False
        or profile.get("release_authorized") is not False
    ):
        _fail("profile_freeze_semantics_invalid")

    view_artifact = artifacts["training_view"]
    bundle_artifact = artifacts["bundle_profile"]
    view = view_artifact.value
    view_safety = _mapping(view.get("safety"), "training_view_semantics_invalid")
    view_input = _mapping(view.get("input"), "training_view_semantics_invalid")
    view_producer = _mapping(view.get("producer"), "training_view_semantics_invalid")
    generic_bindings = _mapping(
        generic.get("bindings"), "generic_release_lock_semantics_invalid"
    )
    generic_hierarchical = _mapping(
        generic.get("hierarchical_task_kv"),
        "generic_release_lock_semantics_invalid",
    )
    generic_projector_manifest_sha256 = generic_bindings.get(
        "projector_manifest_sha256"
    )
    if not _is_sha256(generic_projector_manifest_sha256):
        _fail("generic_release_lock_semantics_invalid")
    expected_projector_sidecar_sha256 = sha256(
        f"{generic_projector_manifest_sha256}  manifest.json\n".encode("ascii")
    ).hexdigest()
    if (
        artifacts["generic_release_lock"].schema.sha256
        != dependency_sha("generic_release_lock_schema")
        or view_artifact.schema.sha256
        != dependency_sha("training_view_manifest_schema")
        or view_producer.get("config_sha256")
        != dependency_sha("training_view_materializer_config")
        or view_producer.get("implementation_sha256")
        != dependency_sha("training_view_materializer_implementation")
        or view_producer.get("record_schema_sha256")
        != dependency_sha("training_view_record_schema")
        or view_producer.get("manifest_schema_sha256")
        != dependency_sha("training_view_manifest_schema")
        or view.get("status") != "materialized_research_proxy_only"
        or view.get("claim_scope") != "research_proxy_materialization_only"
        or view_safety.get("provider_requests") != 0
        or view_safety.get("model_loads") != 0
        or view_safety.get("gpu_requests") != 0
        or view_safety.get("network_requests") != 0
        or view_safety.get("canonical_gold_written") is not False
        or view_safety.get("heldout_read") is not False
        or view_safety.get("heldout_written") is not False
        or view_safety.get("training_authorized") is not False
        or view_safety.get("formal_training_authorized") is not False
        or view_safety.get("eval_proxy_is_heldout") is not False
        or view_input.get("bundle_profile_manifest_sha256")
        != bundle_artifact.manifest.sha256
        or view_input.get("bundle_profile_manifest_sidecar_sha256")
        != bundle_artifact.sidecar.sha256
        or view_input.get("bundle_profile_manifest_schema_sha256")
        != bundle_artifact.schema.sha256
        or view_input.get("bundle_profile_manifest_schema_sha256")
        != dependency_sha("bundle_profile_manifest_schema")
        or view_input.get("bundle_profile_record_schema_sha256")
        != dependency_sha("bundle_profile_record_schema")
        or view_input.get("projector_manifest_sha256")
        != generic_projector_manifest_sha256
        or view_input.get("projector_manifest_sidecar_sha256")
        != expected_projector_sidecar_sha256
        or view_input.get("projector_manifest_schema_sha256")
        != generic_bindings.get("projector_manifest_schema_sha256")
        or view_input.get("projector_manifest_schema_sha256")
        != dependency_sha("taskboard_projector_manifest_schema")
        or view_input.get("projector_record_schema_sha256")
        != generic_bindings.get("projector_sidecar_schema_sha256")
        or view_input.get("projector_record_schema_sha256")
        != dependency_sha("taskboard_projector_sidecar_schema")
        or view_input.get("segment_plan_schema_sha256")
        != generic_bindings.get("projector_segment_plan_schema_sha256")
        or view_input.get("segment_plan_schema_sha256")
        != generic_hierarchical.get("segment_plan_schema_sha256")
        or view_input.get("segment_plan_schema_sha256")
        != dependency_sha("taskboard_segment_plan_schema")
    ):
        _fail("training_view_semantics_invalid")

    bundle = bundle_artifact.value
    bundle_source = _mapping(bundle.get("source"), "bundle_profile_semantics_invalid")
    bundle_producer = _mapping(
        bundle.get("producer"), "bundle_profile_semantics_invalid"
    )
    if (
        bundle_artifact.schema.sha256
        != dependency_sha("bundle_profile_manifest_schema")
        or bundle_producer.get("config_sha256")
        != dependency_sha("training_view_materializer_config")
        or bundle_producer.get("implementation_sha256")
        != dependency_sha("training_view_materializer_implementation")
        or bundle_producer.get("record_schema_sha256")
        != dependency_sha("bundle_profile_record_schema")
        or bundle_producer.get("manifest_schema_sha256")
        != dependency_sha("bundle_profile_manifest_schema")
        or bundle_producer.get("descriptor_schema_sha256")
        != dependency_sha("bundle_profile_descriptor_schema")
        or bundle.get("status") != "metadata_only_ready"
        or bundle.get("claim_scope") != "research_proxy_metadata_only"
        or bundle.get("body_free") is not True
        or bundle.get("eval_proxy_is_heldout") is not False
        or bundle.get("training_authorized") is not False
        or bundle.get("formal_training_authorized") is not False
        or bundle_source.get("projector_manifest_sha256")
        != generic_projector_manifest_sha256
        or bundle_producer.get("manifest_schema_sha256")
        != bundle_artifact.schema.sha256
    ):
        _fail("bundle_profile_semantics_invalid")

    consumer = artifacts["consumer_reference"].value
    counts = _mapping(consumer.get("counts"), "consumer_reference_semantics_invalid")
    semantic = _mapping(
        consumer.get("semantic_identity_contract"),
        "consumer_reference_semantics_invalid",
    )
    claims = _mapping(consumer.get("claims"), "consumer_reference_semantics_invalid")
    audit = _mapping(consumer.get("audit"), "consumer_reference_semantics_invalid")
    ablation = _mapping(
        consumer.get("ablation_contract"),
        "consumer_reference_semantics_invalid",
    )
    source_disjoint = _mapping(
        consumer.get("source_disjoint_boundary"),
        "consumer_reference_semantics_invalid",
    )
    compatibility = _mapping(
        consumer.get("compatibility_boundary"),
        "consumer_reference_semantics_invalid",
    )
    if (
        consumer.get("status") != "dataset_proxy_ready_training_not_authorized"
        or consumer.get("claim_scope")
        != "synthetic_diagnostic_only_no_formal_or_training_authority"
        or counts.get("records") != 1000
        or counts.get("role_views") != 1000
        or counts.get("pair_count") != 0
        or counts.get("task_bundles") != 200
        or counts.get("roles") != 5
        or counts.get("languages") != 2
        or counts.get("variants") != 1
        or counts.get("variants_per_role") != 1
        or counts.get("train_records") != 800
        or counts.get("eval_proxy_records") != 200
        or semantic.get("en_unique_task_semantics") != 100
        or semantic.get("zh_cn_unique_task_semantics") != 100
        or semantic.get("translation_pair_count") != 0
        or ablation.get("primary_label") != "q_only"
        or ablation.get("q_only_is_only_primary") is not True
        or claims.get("diagnostic_only") is not True
        or claims.get("training_authorized") is not False
        or claims.get("formal") is not False
        or claims.get("quality_validated") is not False
        or claims.get("eval_proxy_is_heldout") is not False
        or claims.get("physical_kv_reuse_claimed") is not False
        or claims.get("numeric_equivalence_claimed") is not False
        or source_disjoint.get("zero_intersection_claimed") is not False
        or source_disjoint.get("formal_source_disjoint_proven") is not False
        or compatibility.get("pair_count") != 0
        or compatibility.get("variants_per_role") != 1
        or audit.get("provider_requests") != 0
        or audit.get("network_requests") != 0
        or audit.get("model_loads") != 0
        or audit.get("gpu_requests") != 0
        or audit.get("protected_body_reads") != 0
        or audit.get("real_tool_executions") != 0
        or audit.get("forbidden_current_future_content_excluded") is not True
    ):
        _fail("consumer_reference_semantics_invalid")
    reference = _mapping(
        config.get("consumer_diagnostic_reference"),
        "release_overlay_config_invalid",
    )
    if artifacts["consumer_reference"].manifest.sha256 != reference.get(
        "manifest_sha256"
    ) or artifacts["consumer_reference"].sidecar.sha256 != reference.get(
        "sidecar_physical_sha256"
    ):
        _fail("consumer_reference_binding_invalid")


def _reject_authority_claims(value: object, code: str) -> None:
    """Reject positive authority anywhere in an authenticated input."""

    authority_keys = {
        "authority_inherited",
        "formal",
        "formal_release_eligible",
        "formal_training_data",
        "training_eligible",
        "training_ready",
    }
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                normalized = key.lower()
                scoped_ready = normalized.endswith("_ready") and any(
                    scope in normalized.split("_")
                    for scope in ("live", "training", "formal", "release")
                )
                authorization_like = (
                    "authoriz" in normalized
                    and normalized != "controls_are_non_authorizing"
                )
                if (
                    isinstance(item, bool)
                    and item
                    and (
                        key in authority_keys
                        or normalized == "authorized"
                        or "authority" in normalized
                        or authorization_like
                        or normalized.endswith("_eligible")
                        or scoped_ready
                        or key.endswith("_authority_inherited")
                    )
                ):
                    _fail(code)
                stack.append(item)
        elif isinstance(current, list):
            stack.extend(current)


def _recheck(
    inventory: Mapping[str, tuple[Path, BytesSnapshot]],
    artifacts: Mapping[str, InputArtifact],
) -> None:
    for role, (path, expected) in inventory.items():
        observed = _read_snapshot(path, f"{role}_changed_during_operation")
        if observed != expected:
            _fail(f"{role}_changed_during_operation")
    for role, artifact in artifacts.items():
        manifest = _read_snapshot(
            artifact.manifest_path, f"{role}_changed_during_operation"
        )
        sidecar = _strict_sidecar(
            artifact.sidecar_path,
            manifest,
            f"{role}_changed_during_operation",
        )
        if manifest != artifact.manifest or sidecar != artifact.sidecar:
            _fail(f"{role}_changed_during_operation")


def _canonical_bytes(value: Any) -> bytes:
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


def _artifact_binding(
    artifact: InputArtifact,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_dir = str(contract["canonical_runtime_dir"]).rstrip("/")
    return {
        "schema_version": artifact.value["schema_version"],
        "source_location_kind": (
            "external_consumer_reference"
            if artifact.role == "consumer_reference"
            else "project_canonical_runtime_dir"
        ),
        "logical_manifest_path": f"{runtime_dir}/manifest.json",
        "manifest_sha256": artifact.manifest.sha256,
        "manifest_bytes": artifact.manifest.size,
        "logical_sidecar_path": f"{runtime_dir}/manifest.json.sha256",
        "sidecar_sha256": artifact.sidecar.sha256,
        "sidecar_bytes": artifact.sidecar.size,
    }


def _enforce_canonical_input_directory(
    project_root: Path,
    role: str,
    directory: str | Path,
    contract: Mapping[str, Any],
) -> None:
    if role == "consumer_reference":
        return
    if _contains_dot_segment(directory):
        _fail(f"{role}_artifact_path_invalid")
    actual = Path(os.path.abspath(Path(directory).expanduser()))
    canonical = project_root.joinpath(
        *PurePosixPath(str(contract["canonical_runtime_dir"])).parts
    )
    if os.path.normcase(str(actual)) != os.path.normcase(str(canonical)):
        _fail(f"{role}_artifact_path_invalid")


def _safe_output(project_root: Path, output_dir: str | Path) -> Path:
    root = Path(os.path.abspath(project_root))
    _assert_plain_path(root, "release_overlay_output_path_invalid", regular_file=False)
    requested = Path(output_dir).expanduser()
    if _contains_dot_segment(output_dir):
        _fail("release_overlay_output_path_invalid")
    output = Path(
        os.path.abspath(requested if requested.is_absolute() else root / requested)
    )
    try:
        relative = output.relative_to(root)
    except ValueError as exc:
        raise FrozenPrefixQReaderReleaseError(
            "release_overlay_output_path_invalid"
        ) from exc
    if (
        not relative.parts
        or relative.parts[0] != "artifacts"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _fail("release_overlay_output_path_invalid")
    if output.exists() or output.is_symlink():
        _fail("release_overlay_output_exists")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists():
            _assert_plain_path(
                current, "release_overlay_output_path_invalid", regular_file=False
            )
        else:
            break
    return output


def freeze_frozen_prefix_qreader_release(
    *,
    project_root: str | Path,
    config_path: str | Path,
    expected_config_sha256: str,
    generic_release_dir: str | Path,
    expected_generic_release_sha256: str,
    profile_freeze_dir: str | Path,
    expected_profile_freeze_sha256: str,
    training_view_dir: str | Path,
    expected_training_view_sha256: str,
    bundle_profile_dir: str | Path,
    expected_bundle_profile_sha256: str,
    consumer_reference_dir: str | Path,
    expected_consumer_reference_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Authenticate five manifest conjuncts and publish a blocked overlay."""

    root = Path(os.path.abspath(Path(project_root).expanduser()))
    config, inventory, contracts = _load_config(
        root, Path(config_path), expected_config_sha256
    )
    args = {
        "generic_release_lock": (
            generic_release_dir,
            expected_generic_release_sha256,
        ),
        "profile_freeze": (profile_freeze_dir, expected_profile_freeze_sha256),
        "training_view": (training_view_dir, expected_training_view_sha256),
        "bundle_profile": (bundle_profile_dir, expected_bundle_profile_sha256),
        "consumer_reference": (
            consumer_reference_dir,
            expected_consumer_reference_sha256,
        ),
    }
    for role, (directory, _expected) in args.items():
        _enforce_canonical_input_directory(root, role, directory, contracts[role])
    artifacts = {
        role: _load_artifact(role, directory, expected, contracts[role])
        for role, (directory, expected) in args.items()
    }
    _semantic_checks(artifacts, config, root, inventory)
    output = _safe_output(root, output_dir)
    input_roots = {artifact.root for artifact in artifacts.values()}
    if any(
        output == item or item in output.parents or output in item.parents
        for item in input_roots
    ):
        _fail("release_overlay_output_overlaps_input")

    source_conjunction = {
        role: artifacts[role].manifest.sha256 for role in _INPUT_ROLES
    }
    conjunction_sha256 = sha256(_canonical_bytes(source_conjunction)).hexdigest()
    producer = _mapping(config["producer_bindings"], "release_overlay_config_invalid")
    payload = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "status": OVERLAY_STATUS,
        "claim_scope": "additive_non_authorizing_release_overlay_only",
        "producer": {
            "config_sha256": expected_config_sha256,
            "schema_sha256": producer["overlay_schema"]["sha256"],
            "implementation_sha256": producer["implementation"]["sha256"],
            "freeze_script_sha256": producer["freeze_script"]["sha256"],
            "canonical_json_policy": "utf8_sort_keys_compact_lf_v1",
            "atomic_create_once": True,
            "final_toctou_recheck": True,
        },
        "bindings": {
            role: _artifact_binding(artifacts[role], contracts[role])
            for role in _INPUT_ROLES
        },
        "source_conjunction_sha256": conjunction_sha256,
        "base_release": {
            "self_reported_status": "ready",
            "self_reported_claim_scope": "research_proxy_only",
            "authority_inherited": False,
        },
        "profile": {
            "profile_id": "frozen-prefix-qreader-v2",
            "profile_boundary": "after_authenticated_canonical_gold",
            "execution_core_reused": "anchor.swebench-five-stage-execution-core.v1",
        },
        "dependency_dag": config["dependency_dag"],
        "architecture_boundary": config["architecture_boundary"],
        "consumer_diagnostic_reference": config["consumer_diagnostic_reference"],
        "gemma_runner_compatibility": config["gemma_runner_compatibility"],
        "unresolved_gates": config["unresolved_gates"],
        "audit": {
            "manifest_files_read": 5,
            "mandatory_sidecar_files_read": 5,
            "profile_dependency_files_authenticated": len(
                _CROSS_BOUND_PROFILE_DEPENDENCY_ROLES
            ),
            "partition_files_read": 0,
            "canonical_gold_bodies_read": 0,
            "heldout_bodies_read": 0,
            "protected_bodies_read": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
        },
        "canonical_gold_written": False,
        "canonical_gold_mutated": False,
        "heldout_written": False,
        "live_authorized": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "release_authorized": False,
    }
    schema = _parse_json(
        inventory["overlay_schema"][1], "release_overlay_schema_invalid"
    )
    _schema_validate(payload, schema, "release_overlay_payload_invalid")
    encoded = _canonical_bytes(payload)
    digest = sha256(encoded).hexdigest()
    sidecar_bytes = f"{digest}  manifest.json\n".encode("ascii")
    temporary = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    published = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        _assert_plain_path(
            output.parent,
            "release_overlay_output_path_invalid",
            regular_file=False,
        )
        if temporary.exists() or temporary.is_symlink():
            _fail("release_overlay_temporary_exists")
        temporary.mkdir()
        manifest_path = temporary / "manifest.json"
        sidecar_path = temporary / "manifest.json.sha256"
        manifest_path.write_bytes(encoded)
        sidecar_path.write_bytes(sidecar_bytes)
        manifest_snapshot = _read_snapshot(
            manifest_path, "release_overlay_output_invalid"
        )
        sidecar_snapshot = _strict_sidecar(
            sidecar_path, manifest_snapshot, "release_overlay_output_invalid"
        )
        if (
            manifest_snapshot.data != encoded
            or manifest_snapshot.sha256 != digest
            or sidecar_snapshot.data != sidecar_bytes
        ):
            _fail("release_overlay_output_invalid")
        _schema_validate(
            _parse_json(manifest_snapshot, "release_overlay_output_invalid"),
            schema,
            "release_overlay_output_invalid",
        )
        _recheck(inventory, artifacts)
        _assert_plain_path(
            output.parent,
            "release_overlay_output_path_invalid",
            regular_file=False,
        )
        if output.exists() or output.is_symlink():
            _fail("release_overlay_output_exists")
        os.replace(temporary, output)
        published = True
        _assert_plain_path(output, "release_overlay_output_invalid", regular_file=False)
        final_manifest = _read_snapshot(
            output / "manifest.json", "release_overlay_output_invalid"
        )
        final_sidecar = _strict_sidecar(
            output / "manifest.json.sha256",
            final_manifest,
            "release_overlay_output_invalid",
        )
        if final_manifest != manifest_snapshot or final_sidecar != sidecar_snapshot:
            _fail("release_overlay_output_invalid")
        _recheck(inventory, artifacts)
    except FrozenPrefixQReaderReleaseError:
        shutil.rmtree(temporary, ignore_errors=True)
        if published:
            shutil.rmtree(output, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        if published:
            shutil.rmtree(output, ignore_errors=True)
        raise FrozenPrefixQReaderReleaseError("release_overlay_internal_error") from exc
    return {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "status": OVERLAY_STATUS,
        "manifest_sha256": digest,
        "source_conjunction_sha256": conjunction_sha256,
        "output_dir": output.relative_to(root).as_posix(),
        "manifest_files_read": 5,
        "profile_dependency_files_authenticated": len(
            _CROSS_BOUND_PROFILE_DEPENDENCY_ROLES
        ),
        "partition_files_read": 0,
        "provider_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "training_authorized": False,
        "formal_training_authorized": False,
        "release_authorized": False,
    }


__all__ = [
    "CONFIG_PATH",
    "CONFIG_SCHEMA_VERSION",
    "FrozenPrefixQReaderReleaseError",
    "OVERLAY_SCHEMA_VERSION",
    "OVERLAY_STATUS",
    "freeze_frozen_prefix_qreader_release",
]
